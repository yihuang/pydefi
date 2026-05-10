"""V3 tick-walking swap math.

Walks Uniswap V3 swaps tick-by-tick: in each constant-liquidity range,
move ``sqrtPriceX96`` toward the next initialised tick; cross the
boundary by adding/subtracting that tick's ``liquidity_net``. Caller
supplies the tick state up front (active sqrtP, L, and the initialised
ticks list) — this module does no fetching.
"""

from __future__ import annotations

_Q96 = 2**96

# Uniswap V3 protocol-level tick / sqrtPrice bounds (TickMath.MIN_TICK /
# MAX_TICK and the matching MIN_SQRT_RATIO / MAX_SQRT_RATIO). A real V3 pool
# never quotes outside these — the swap loop uses them as the implicit
# terminal boundary when no initialised tick lies in the swap direction, so
# a within-range swap still executes against constant liquidity instead of
# returning zero.
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342


def _next_sqrt_price_zero_for_one(sqrt_p: int, liquidity: int, amount_in: int) -> int:
    """``sqrtP`` after token0-input swap (price moves down).

    Derivation: ``1/sqrtP_new = 1/sqrtP + amount_in / (L * Q96)`` →
    ``sqrtP_new = sqrtP * L * Q96 / (L * Q96 + amount_in * sqrtP)``.
    """
    if liquidity <= 0 or amount_in <= 0:
        return sqrt_p
    denom = liquidity * _Q96 + amount_in * sqrt_p
    if denom <= 0:
        return 0
    return sqrt_p * liquidity * _Q96 // denom


def _next_sqrt_price_one_for_zero(sqrt_p: int, liquidity: int, amount_in: int) -> int:
    """``sqrtP`` after token1-input swap (price moves up).

    Derivation: ``sqrtP_new = sqrtP + amount_in * Q96 / L``.
    """
    if liquidity <= 0:
        return sqrt_p
    return sqrt_p + amount_in * _Q96 // liquidity


def _amount_in_to_reach(sqrt_p_current: int, sqrt_p_target: int, liquidity: int, zero_for_one: bool) -> int:
    """Token-in amount required to push ``sqrtP`` from current → target.

    Rounds *up* (Uniswap V3 convention): the floor of the true real-valued
    requirement can be 1 unit short, which would let
    :func:`compute_swap_step` snap to the boundary on insufficient input.
    Ceiling guarantees we only declare "enough" when we really have it.
    """
    if liquidity <= 0:
        return 0
    if zero_for_one:
        if sqrt_p_target >= sqrt_p_current or sqrt_p_target <= 0:
            return 0
        num = liquidity * _Q96 * (sqrt_p_current - sqrt_p_target)
        den = sqrt_p_current * sqrt_p_target
        return -(-num // den)
    if sqrt_p_target <= sqrt_p_current:
        return 0
    return -(-(liquidity * (sqrt_p_target - sqrt_p_current)) // _Q96)


def _amount_out_from_delta(sqrt_p_current: int, sqrt_p_next: int, liquidity: int, zero_for_one: bool) -> int:
    """Output token amount for a sqrtP transition under constant liquidity."""
    if liquidity <= 0:
        return 0
    if zero_for_one:
        # token1 out: L * (sqrtP_current - sqrtP_next) / Q96
        if sqrt_p_next >= sqrt_p_current:
            return 0
        return liquidity * (sqrt_p_current - sqrt_p_next) // _Q96
    # token0 out: L * Q96 * (sqrtP_next - sqrtP_current) / (sqrtP_current * sqrtP_next)
    if sqrt_p_next <= sqrt_p_current or sqrt_p_current <= 0 or sqrt_p_next <= 0:
        return 0
    return liquidity * _Q96 * (sqrt_p_next - sqrt_p_current) // (sqrt_p_current * sqrt_p_next)


def compute_swap_step(
    sqrt_p_current: int,
    sqrt_p_target: int,
    liquidity: int,
    amount_remaining_net: int,
    zero_for_one: bool,
) -> tuple[int, int, int]:
    """One tick-range swap step under constant liquidity.

    Returns ``(sqrt_p_next, amount_in_used, amount_out)``. If
    ``amount_remaining_net`` is enough to reach ``sqrt_p_target`` we stop at
    the boundary and return how much was consumed; otherwise we stop within
    the range and ``sqrt_p_next`` is wherever the input runs out.

    Sub-``MIN_SQRT_RATIO`` synthetic note: when ``sqrt_p_*`` are absurdly
    small (well below the protocol's ``MIN_SQRT_RATIO ≈ 4.3e9`` floor),
    integer-floor in ``_next_sqrt_price_*`` can land on
    ``sqrt_p_target`` even at ``amount_remaining_net = max_in - 1``.
    Accounting stays sound (``amount_in_used`` is the actual input, not
    ``max_in``) and this matches on-chain Uniswap V3 ``SwapMath`` exactly,
    so we don't guard. Real V3 pools never reach that regime.
    """
    if liquidity <= 0 or amount_remaining_net <= 0:
        return sqrt_p_current, 0, 0

    max_in = _amount_in_to_reach(sqrt_p_current, sqrt_p_target, liquidity, zero_for_one)
    if amount_remaining_net >= max_in and max_in > 0:
        # Hit the boundary exactly.
        sqrt_p_next = sqrt_p_target
        amount_in_used = max_in
    else:
        # Stay within range.
        if zero_for_one:
            sqrt_p_next = _next_sqrt_price_zero_for_one(sqrt_p_current, liquidity, amount_remaining_net)
        else:
            sqrt_p_next = _next_sqrt_price_one_for_zero(sqrt_p_current, liquidity, amount_remaining_net)
        amount_in_used = amount_remaining_net

    amount_out = _amount_out_from_delta(sqrt_p_current, sqrt_p_next, liquidity, zero_for_one)
    return sqrt_p_next, amount_in_used, amount_out


def walk_v3_swap(
    sqrt_price_x96: int,
    liquidity: int,
    ticks: list[tuple[int, int, int]],
    amount_in: int,
    fee_bps: int,
    zero_for_one: bool,
) -> int:
    """Walk an exact-input V3 swap across initialised tick boundaries.

    Args:
        sqrt_price_x96: current pool ``sqrtPriceX96`` (Q64.96).
        liquidity: currently active ``L`` for the pool's active range.
        ticks: initialised ticks as ``(tick_index, sqrt_price_at_tick,
            liquidity_net)``. ``liquidity_net`` is added to ``L`` when the
            price *enters* the range above ``tick_index`` from below; we
            therefore subtract it when crossing downward (``zero_for_one``)
            and add it when crossing upward.
        amount_in: input amount in raw token units.
        fee_bps: pool fee in basis points (e.g. ``5`` for 0.05%).
        zero_for_one: ``True`` when input is token0 (price moves down).

    Returns total ``amount_out`` accumulated across crossings. Stops when
    input is exhausted or no more ticks lie in the swap direction (the
    latter implies the swap exceeds the pool's modelled liquidity).
    """
    if amount_in <= 0 or liquidity < 0:
        return 0

    # Single fee deduction is equivalent to per-step fee for a fixed-tier pool.
    amount_remaining = amount_in * (10_000 - fee_bps) // 10_000
    if amount_remaining <= 0:
        return 0

    sqrt_p = sqrt_price_x96
    active_l = liquidity
    total_out = 0

    # Always include a protocol-level terminal sentinel in the swap direction
    # so a missing-direction-tick input still executes a within-range step
    # rather than returning zero. liquidity_net=0 means crossing the sentinel
    # is a no-op (we'll never actually cross — the swap exhausts first).
    if zero_for_one:
        relevant = sorted((t for t in ticks if t[1] < sqrt_p), key=lambda t: -t[1])
        relevant.append((MIN_TICK, MIN_SQRT_RATIO, 0))
    else:
        relevant = sorted((t for t in ticks if t[1] > sqrt_p), key=lambda t: t[1])
        relevant.append((MAX_TICK, MAX_SQRT_RATIO, 0))

    for _tick_idx, sqrt_p_at_tick, liquidity_net in relevant:
        if amount_remaining <= 0 or active_l <= 0:
            break
        sqrt_p_next, used, out = compute_swap_step(sqrt_p, sqrt_p_at_tick, active_l, amount_remaining, zero_for_one)
        amount_remaining -= used
        total_out += out
        sqrt_p = sqrt_p_next
        if sqrt_p == sqrt_p_at_tick:
            # Boundary reached — apply liquidity_net for the cross.
            active_l = active_l - liquidity_net if zero_for_one else active_l + liquidity_net
            if active_l < 0:
                active_l = 0
        else:
            # Input exhausted within this range.
            break

    return total_out

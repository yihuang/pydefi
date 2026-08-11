"""V3 tick-walking swap math.

Within each constant-liquidity range, move ``sqrtPriceX96`` toward the next
initialised tick, then cross it by applying that tick's ``liquidity_net``.
Tick state comes from the caller; this module does no fetching.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import chain, islice
from typing import Iterator

_Q96 = 2**96

# Uniswap V3 protocol-level bounds (TickMath.MIN_TICK / MAX_TICK and the
# matching sqrt ratios). A real pool never quotes outside these, so the walk
# uses them as its terminal boundary — see :meth:`TickLadder.crossings`.
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

_MIN_SENTINEL = (MIN_TICK, MIN_SQRT_RATIO, 0)
_MAX_SENTINEL = (MAX_TICK, MAX_SQRT_RATIO, 0)


class TickLadder:
    """A pool's initialised ticks, sorted once by ``sqrt_price_at_tick``.

    Reuse it across quotes: the order is a property of the pool, and the
    starting price only decides where to bisect in.
    """

    __slots__ = ("prices", "ticks")

    def __init__(self, ticks: list[tuple[int, int, int]]) -> None:
        self.ticks = sorted(ticks, key=lambda t: t[1])
        self.prices = [t[1] for t in self.ticks]

    def __len__(self) -> int:
        return len(self.ticks)

    @property
    def price_range(self) -> tuple[int, int] | None:
        """``(lowest, highest)`` ``sqrtPriceX96`` covered, ``None`` if empty.

        A swap ending outside it ran past the last known tick against constant
        liquidity, so its output is an overestimate.
        """
        return (self.prices[0], self.prices[-1]) if self.prices else None

    def crossings(self, sqrt_p: int, zero_for_one: bool) -> Iterator[tuple[int, int, int]]:
        """Ticks a swap from *sqrt_p* would cross, nearest first.

        Ends on a protocol-boundary sentinel, so a direction with no ticks
        still yields one constant-liquidity step instead of nothing.
        """
        if zero_for_one:
            cut = bisect_left(self.prices, sqrt_p)  # ticks strictly below sqrt_p
            return chain((self.ticks[i] for i in reversed(range(cut))), (_MIN_SENTINEL,))
        cut = bisect_right(self.prices, sqrt_p)  # ticks strictly above sqrt_p
        return chain(islice(self.ticks, cut, None), (_MAX_SENTINEL,))


#: ``getSqrtRatioAtTick``'s magic constants, indexed by the bit they guard.
#: Each is ``2**128 / 1.0001**(2**k / 2)`` rounded, so multiplying the ones
#: whose bit is set composes ``1.0001**(-tick/2)`` in Q128.
_TICK_RATIOS = (
    0xFFFCB933BD6FAD37AA2D162D1A594001,
    0xFFF97272373D413259A46990580E213A,
    0xFFF2E50F5F656932EF12357CF3C7FDCC,
    0xFFE5CACA7E10E4E61C3624EAA0941CD0,
    0xFFCB9843D60F6159C9DB58835C926644,
    0xFF973B41FA98C081472E6896DFB254C0,
    0xFF2EA16466C96A3843EC78B326B52861,
    0xFE5DEE046A99A2A811C461F1969C3053,
    0xFCBE86C7900A88AEDCFFC83B479AA3A4,
    0xF987A7253AC413176F2B074CF7815E54,
    0xF3392B0822B70005940C7A398E4B70F3,
    0xE7159475A2C29B7443B29C7FA6E889D9,
    0xD097F3BDFD2022B8845AD8F792AA5825,
    0xA9F746462D870FDF8A65DC1F90E061E5,
    0x70D869A156D2A1B890BB3DF62BAF32F7,
    0x31BE135F97D08FD981231505542FCFA6,
    0x9AA508B5B7A84E1C677DE54F3E99BC9,
    0x5D6AF8DEDB81196699C329225EE604,
    0x2216E584F5FA1EA926041BEDFE98,
    0x48A170391F7DC42444E8FA2,
)


def get_sqrt_ratio_at_tick(tick: int) -> int:
    """Return ``sqrtPriceX96`` at *tick* — bit-exact Uniswap V3 ``TickMath``.

    The contract's integer cascade, not ``1.0001 ** (tick / 2)``: the float
    form drifts a few ULP, enough to put a boundary on the wrong side of the
    current price.

    Raises:
        ValueError: If *tick* is outside ``[MIN_TICK, MAX_TICK]``.
    """
    if not MIN_TICK <= tick <= MAX_TICK:
        raise ValueError(f"tick {tick} outside [{MIN_TICK}, {MAX_TICK}]")
    abs_tick = -tick if tick < 0 else tick

    ratio = 1 << 128
    for k, magic in enumerate(_TICK_RATIOS):
        if abs_tick & (1 << k):
            ratio = (ratio * magic) >> 128
    if tick > 0:
        ratio = ((1 << 256) - 1) // ratio
    # Round up when truncating Q128 to Q96, as the contract does.
    return (ratio >> 32) + (1 if ratio % (1 << 32) else 0)


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

    Rounds up, as V3 does: a floor can be 1 unit short, letting
    :func:`compute_swap_step` snap to the boundary on insufficient input.
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

    Returns ``(sqrt_p_next, amount_in_used, amount_out)``, stopping at
    ``sqrt_p_target`` if the input reaches it and wherever the input runs out
    if not.
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
    ticks: TickLadder | list[tuple[int, int, int]],
    amount_in: int,
    fee_bps: int,
    zero_for_one: bool,
) -> int:
    """Walk an exact-input V3 swap across initialised tick boundaries.

    Args:
        sqrt_price_x96: current pool ``sqrtPriceX96`` (Q64.96).
        liquidity: active ``L`` for the pool's current range.
        ticks: a :class:`TickLadder`, or raw ``(tick_index,
            sqrt_price_at_tick, liquidity_net)`` to sort into one per call.
            ``liquidity_net`` is signed for entering from below, so crossing
            downward subtracts it.
        amount_in: input amount in raw token units.
        fee_bps: pool fee in basis points (e.g. ``5`` for 0.05%).
        zero_for_one: ``True`` when input is token0 (price moves down).

    Returns total ``amount_out``. Running out of ladder means the swap outgrew
    the modelled liquidity, so the result is then an overestimate.
    """
    if amount_in <= 0 or liquidity < 0:
        return 0

    # One deduction up front is equivalent to charging per step at a fixed tier.
    amount_remaining = amount_in * (10_000 - fee_bps) // 10_000
    if amount_remaining <= 0:
        return 0

    sqrt_p = sqrt_price_x96
    active_l = liquidity
    total_out = 0
    ladder = ticks if isinstance(ticks, TickLadder) else TickLadder(ticks)

    for _tick_idx, sqrt_p_at_tick, liquidity_net in ladder.crossings(sqrt_p, zero_for_one):
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

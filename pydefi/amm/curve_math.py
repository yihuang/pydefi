"""
Pure-Python Curve pricing math — ports Curve's on-chain ``get_dy`` so prices
can be computed locally from a pool-state snapshot, no RPC per quote.

* **Stableswap** (:func:`stable_get_dy`) — plain V1, factory, and Stable-NG
  pools; the ``a_precision`` / ``ng_d_form`` / ``offpeg_fee_multiplier`` /
  ``legacy_fee_order`` flags reproduce all three bit-for-bit.
* **Cryptoswap / Curve V2** (:func:`crypto_get_dy`) — the ``newton_D`` /
  ``newton_y`` invariant with gamma (Twocrypto / Tricrypto).

All functions use integer floor division to mirror the EVM exactly.  Each
function's docstring carries a ``Vyper:`` line naming the contract source it
ports, in these canonical Curve repositories:

* ``curve-contract`` — legacy plain pools (``contracts/pools/3pool/
  StableSwap3Pool.vy``) and the factory/meta templates
  (``contracts/pool-templates/{base,meta}/SwapTemplate{Base,Meta}.vy``).
* ``stableswap-ng`` — ``contracts/main/CurveStableSwapNGViews.vy`` (NG quoting
  lives in the Views contract, not the pool).
* ``curve-crypto-contract`` — ``contracts/tricrypto/CurveCryptoMath3.vy`` /
  ``CurveCryptoSwap.vy`` / ``CurveCryptoViews3.vy``.
"""

from __future__ import annotations

from pydefi.exceptions import PydefiError

# ---------------------------------------------------------------------------
# Constants (match Curve contracts)
# ---------------------------------------------------------------------------

#: Fee denominator used by every Curve pool (fees are expressed as ``n / 1e10``).
FEE_DENOMINATOR: int = 10**10
#: Rate/precision base (1e18).
PRECISION: int = 10**18
#: Amplification precision for newer stableswap pools (NG / factory).  Legacy
#: plain pools such as 3pool predate this and use ``1``.
A_PRECISION: int = 100
#: Amplification multiplier used by cryptoswap (Curve V2) pools.
A_MULTIPLIER: int = 10000


class CurveConvergenceError(PydefiError):
    """Raised when a Curve Newton iteration fails to converge."""


# ===========================================================================
# Stableswap
# ===========================================================================


def _xp_mem(rates: list[int], balances: list[int]) -> list[int]:
    """Normalise *balances* to a common precision using per-coin *rates*.

    ``xp[i] = rates[i] * balances[i] / PRECISION``.  For a plain pool the rate
    is ``10**(36 - decimals)`` (so every coin is scaled to 18 decimals); for a
    rate-stabilised pool it additionally folds in the oracle exchange rate.

    Vyper: ``StableSwap3Pool.vy::_xp_mem`` (same shape in every stableswap).
    """
    return [r * b // PRECISION for r, b in zip(rates, balances)]


def stable_get_D(xp: list[int], amp: int, a_precision: int = A_PRECISION, *, ng_d_form: bool = False) -> int:
    """Compute the stableswap invariant ``D`` for normalised balances *xp*.

    Args:
        xp: Balances already normalised to a common precision (see
            :func:`_xp_mem`).
        amp: Amplification coefficient.  For legacy plain pools this is the raw
            ``A()`` value with ``a_precision=1``; for NG/factory pools it is the
            precise ``A`` (``A() * 100``) with ``a_precision=100``.
        a_precision: ``1`` for legacy plain pools, ``100`` otherwise.
        ng_d_form: When ``True`` use the Stable-NG ``D_P`` accumulation
            (divide by each balance then by ``n**n``); otherwise use the classic
            per-coin ``D_P * D / (x * n)`` form.

    Returns:
        The invariant ``D``.

    Vyper: ``StableSwap3Pool.vy::get_D`` (legacy, ``a_precision=1``);
    ``SwapTemplateBase.vy::get_D`` (factory, ``A_PRECISION=100``);
    ``CurveStableSwapNGViews.vy::get_D`` (``ng_d_form=True``).
    """
    n = len(xp)
    s = sum(xp)
    if s == 0:
        return 0

    d = s
    ann = amp * n
    for _ in range(255):
        if ng_d_form:
            d_p = d
            for x in xp:
                d_p = d_p * d // x
            d_p //= n**n
        else:
            d_p = d
            for x in xp:
                d_p = d_p * d // (x * n)
        d_prev = d
        d = ((ann * s // a_precision + d_p * n) * d) // (((ann - a_precision) * d) // a_precision + (n + 1) * d_p)
        if abs(d - d_prev) <= 1:
            return d
    raise CurveConvergenceError("stable_get_D did not converge")


def stable_get_y(
    i: int,
    j: int,
    x: int,
    xp: list[int],
    amp: int,
    d: int,
    a_precision: int = A_PRECISION,
) -> int:
    """Solve the stableswap invariant for the new balance of coin *j*.

    Given that coin *i* will hold ``x`` (in normalised units), return the
    resulting normalised balance of coin *j* that keeps the invariant ``d``.

    Vyper: ``StableSwap3Pool.vy::get_y`` / ``SwapTemplateBase.vy::get_y`` /
    ``CurveStableSwapNGViews.vy::get_y`` (identical iteration in all three).
    """
    n = len(xp)
    if i == j:
        raise ValueError("i and j must differ")
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("coin index out of range")

    ann = amp * n
    c = d
    s_ = 0
    for k in range(n):
        if k == i:
            _x = x
        elif k != j:
            _x = xp[k]
        else:
            continue
        s_ += _x
        c = c * d // (_x * n)
    c = c * d * a_precision // (ann * n)
    b = s_ + d * a_precision // ann
    y = d
    for _ in range(255):
        y_prev = y
        y = (y * y + c) // (2 * y + b - d)
        if abs(y - y_prev) <= 1:
            return y
    raise CurveConvergenceError("stable_get_y did not converge")


def stable_dynamic_fee(xpi: int, xpj: int, fee: int, fee_multiplier: int) -> int:
    """Stable-NG off-peg dynamic fee.

    When ``fee_multiplier <= FEE_DENOMINATOR`` the flat *fee* applies; otherwise
    the fee scales up as the two balances diverge from parity.

    Vyper: ``CurveStableSwapNGViews.vy::_dynamic_fee``.
    """
    if fee_multiplier <= FEE_DENOMINATOR:
        return fee
    xps2 = (xpi + xpj) ** 2
    return (fee_multiplier * fee) // ((fee_multiplier - FEE_DENOMINATOR) * 4 * xpi * xpj // xps2 + FEE_DENOMINATOR)


def stable_get_dy(
    i: int,
    j: int,
    dx: int,
    balances: list[int],
    rates: list[int],
    amp: int,
    fee: int,
    *,
    a_precision: int = A_PRECISION,
    ng_d_form: bool = False,
    offpeg_fee_multiplier: int = 0,
    legacy_fee_order: bool = False,
) -> int:
    """Compute the output amount of a stableswap ``exchange`` locally.

    Args:
        i: Index of the input coin.
        j: Index of the output coin.
        dx: Raw input amount (in coin *i*'s own units).
        balances: Raw pool balances (coin units).
        rates: Per-coin rate multipliers (see :func:`_xp_mem`).  For a plain
            pool, ``10**(36 - decimals)``; for an NG/rate pool, ``stored_rates``.
        amp: Amplification (precise for NG/factory, raw for legacy).
        fee: Base swap fee (units of ``1/1e10``).
        a_precision: ``1`` for legacy plain pools, ``100`` otherwise.
        ng_d_form: Use the Stable-NG ``D_P`` form (see :func:`stable_get_D`).
        offpeg_fee_multiplier: Stable-NG off-peg multiplier; ``0`` disables the
            dynamic fee (plain/factory pools use a flat fee).
        legacy_fee_order: When ``True`` (legacy plain pools like 3pool) the
            output is converted to coin *j*'s units *before* the fee is taken;
            otherwise the fee is taken in normalised units and converted after.

    Returns:
        Raw output amount in coin *j*'s units (``0`` if the swap is infeasible).

    Vyper: ``StableSwap3Pool.vy::get_dy`` (``legacy_fee_order=True``);
    ``SwapTemplateBase.vy::get_dy`` (factory); ``CurveStableSwapNGViews.vy::
    get_dy`` (NG, with the off-peg dynamic fee).
    """
    xp = _xp_mem(rates, balances)
    d = stable_get_D(xp, amp, a_precision, ng_d_form=ng_d_form)
    x = xp[i] + dx * rates[i] // PRECISION
    y = stable_get_y(i, j, x, xp, amp, d, a_precision)
    dy = xp[j] - y - 1
    if dy < 0:
        return 0

    if legacy_fee_order:
        # Legacy plain pools (e.g. 3pool) convert to coin units, then fee.
        dy = dy * PRECISION // rates[j]
        return dy - fee * dy // FEE_DENOMINATOR

    if offpeg_fee_multiplier:
        fee = stable_dynamic_fee((xp[i] + x) // 2, (xp[j] + y) // 2, fee, offpeg_fee_multiplier)
    dy -= fee * dy // FEE_DENOMINATOR
    return dy * PRECISION // rates[j]


def stable_get_dx(
    i: int,
    j: int,
    dy: int,
    balances: list[int],
    rates: list[int],
    amp: int,
    fee: int,
    *,
    a_precision: int = A_PRECISION,
    ng_d_form: bool = False,
    offpeg_fee_multiplier: int = 0,
    legacy_fee_order: bool = False,
) -> int:
    """Exact-output inverse of :func:`stable_get_dy` (Curve ``get_dx``).

    Returns the raw input amount of coin *i* required to receive *dy* of coin
    *j*.  Accepts the same pool-config keys as :func:`stable_get_dy` so a pool
    state dict unpacks into either; ``legacy_fee_order`` is unused here (the
    fee gross-up is identical for both fee orderings).

    Returns:
        Raw input amount in coin *i*'s units (``0`` if the swap is infeasible).

    Vyper: ``CurveStableSwapNGViews.vy::get_dx`` (older pools have no on-chain
    ``get_dx``; the same inversion applies to every stableswap flavour).
    """
    xp = _xp_mem(rates, balances)
    d = stable_get_D(xp, amp, a_precision, ng_d_form=ng_d_form)
    if offpeg_fee_multiplier:
        fee = stable_dynamic_fee(xp[i], xp[j], fee, offpeg_fee_multiplier)
    dy_with_fee = dy * rates[j] // PRECISION + 1
    y = xp[j] - dy_with_fee * FEE_DENOMINATOR // (FEE_DENOMINATOR - fee)
    if y < 0:
        return 0
    x = stable_get_y(j, i, y, xp, amp, d, a_precision)
    if x < xp[i]:
        return 0
    return (x - xp[i]) * PRECISION // rates[i]


# ---------------------------------------------------------------------------
# Stableswap meta-pool helpers (a coin is the LP token of a base pool)
# ---------------------------------------------------------------------------


def stable_get_y_D(amp: int, i: int, xp: list[int], d: int, a_precision: int = A_PRECISION) -> int:
    """Solve the stableswap invariant for coin *i* at a target invariant *d*.

    Unlike :func:`stable_get_y` (which fixes another coin's new balance), this
    fixes ``D`` directly — used by single-coin withdrawals.

    Vyper: ``StableSwap3Pool.vy::get_y_D`` / ``SwapTemplateMeta.vy::get_y_D``.
    """
    n = len(xp)
    ann = amp * n
    c = d
    s_ = 0
    for k in range(n):
        if k == i:
            continue
        _x = xp[k]
        s_ += _x
        c = c * d // (_x * n)
    c = c * d * a_precision // (ann * n)
    b = s_ + d * a_precision // ann
    y = d
    for _ in range(255):
        y_prev = y
        y = (y * y + c) // (2 * y + b - d)
        if abs(y - y_prev) <= 1:
            return y
    raise CurveConvergenceError("stable_get_y_D did not converge")


def stable_calc_token_amount(
    amounts: list[int],
    balances: list[int],
    rates: list[int],
    amp: int,
    total_supply: int,
    *,
    is_deposit: bool,
    a_precision: int = A_PRECISION,
    ng_d_form: bool = False,
) -> int:
    """LP tokens minted (deposit) or burned (withdraw) for *amounts* — fee-free.

    Matches Curve's ``calc_token_amount`` (the ideal, fee-excluding estimate);
    meta-pool pricing applies the approximate deposit/withdraw fee separately.

    Vyper: ``StableSwap3Pool.vy::calc_token_amount`` /
    ``SwapTemplateMeta.vy::calc_token_amount``.
    """
    d0 = stable_get_D(_xp_mem(rates, balances), amp, a_precision, ng_d_form=ng_d_form)
    new_balances = [b + a if is_deposit else b - a for b, a in zip(balances, amounts)]
    d1 = stable_get_D(_xp_mem(rates, new_balances), amp, a_precision, ng_d_form=ng_d_form)
    diff = d1 - d0 if is_deposit else d0 - d1
    return diff * total_supply // d0


def stable_calc_withdraw_one_coin(
    token_amount: int,
    i: int,
    balances: list[int],
    rates: list[int],
    amp: int,
    fee: int,
    total_supply: int,
    *,
    a_precision: int = A_PRECISION,
    ng_d_form: bool = False,
) -> int:
    """Coin *i* received for burning *token_amount* LP tokens (Curve withdraw).

    Vyper: ``StableSwap3Pool.vy::_calc_withdraw_one_coin`` /
    ``SwapTemplateMeta.vy::_calc_withdraw_one_coin``.
    """
    n = len(balances)
    xp = _xp_mem(rates, balances)
    precisions = [r // PRECISION for r in rates]  # 10**(18 - decimals)
    base_fee = fee * n // (4 * (n - 1))

    d0 = stable_get_D(xp, amp, a_precision, ng_d_form=ng_d_form)
    d1 = d0 - token_amount * d0 // total_supply
    new_y = stable_get_y_D(amp, i, xp, d1, a_precision)

    xp_reduced = list(xp)
    for k in range(n):
        dx_expected = xp[k] * d1 // d0 - new_y if k == i else xp[k] - xp[k] * d1 // d0
        xp_reduced[k] -= base_fee * dx_expected // FEE_DENOMINATOR
    dy = xp_reduced[i] - stable_get_y_D(amp, i, xp_reduced, d1, a_precision)
    return (dy - 1) // precisions[i]  # withdraw a touch less to absorb rounding


def meta_get_dy_underlying(i: int, j: int, dx: int, meta: dict, base: dict) -> int:
    """Price an underlying swap through a stableswap meta-pool.

    Underlying coin ``0`` is the meta-pool's primary coin; coins ``1..`` are the
    base pool's coins.  ``meta`` is a stableswap state dict whose ``rates[1]`` is
    the base pool's virtual price; ``base`` is the base pool's state dict plus a
    ``total_supply`` key.  Mirrors Curve's ``get_dy_underlying`` for USD/BTC
    factory meta-pools.

    Stable-NG metas differ in two ways, both keyed off the ``meta`` dict:
    ``offpeg_fee_multiplier`` selects the off-peg dynamic fee, and
    ``ng_d_form=True`` skips the factory metas' approximate ½-fee deduction on
    the base-deposit leg (NG quotes the ideal mint).

    Vyper: ``SwapTemplateMeta.vy::get_dy_underlying`` (factory MetaUSD
    implementations, e.g. MIM/3CRV);
    ``CurveStableSwapNGViews.vy::get_dy_underlying`` (Stable-NG metas).
    """
    rates = meta["rates"]
    amp, fee = meta["amp"], meta["fee"]
    a_prec, ng = meta["a_precision"], meta["ng_d_form"]
    base_n = len(base["balances"])
    xp = _xp_mem(rates, meta["balances"])

    base_i, base_j = i - 1, j - 1
    meta_i, meta_j = (0 if i == 0 else 1), (0 if j == 0 else 1)

    if i != 0 and j != 0:  # both legs inside the base pool
        return stable_get_dy(base_i, base_j, dx, **base_swap_kwargs(base))

    if i == 0:
        x = xp[0] + dx * rates[0] // PRECISION
    else:
        # Base coin in → mint base LP, value it in meta units.
        base_inputs = [dx if k == base_i else 0 for k in range(base_n)]
        minted = stable_calc_token_amount(
            base_inputs,
            base["balances"],
            base["rates"],
            base["amp"],
            base["total_supply"],
            is_deposit=True,
            a_precision=base["a_precision"],
            ng_d_form=base["ng_d_form"],
        )
        x = minted * rates[1] // PRECISION
        if not ng:
            x -= x * base["fee"] // (2 * FEE_DENOMINATOR)  # approximate deposit fee
        x += xp[1]

    d = stable_get_D(xp, amp, a_prec, ng_d_form=ng)
    y = stable_get_y(meta_i, meta_j, x, xp, amp, d, a_prec)
    dy = xp[meta_j] - y - 1
    offpeg = meta.get("offpeg_fee_multiplier", 0)
    if offpeg:
        fee = stable_dynamic_fee((xp[meta_i] + x) // 2, (xp[meta_j] + y) // 2, fee, offpeg)
    dy -= fee * dy // FEE_DENOMINATOR

    if j == 0:
        return dy // (rates[0] // PRECISION)
    # Base coin out → burn the equivalent base LP for coin base_j.
    return stable_calc_withdraw_one_coin(
        dy * PRECISION // rates[1],
        base_j,
        base["balances"],
        base["rates"],
        base["amp"],
        base["fee"],
        base["total_supply"],
        a_precision=base["a_precision"],
        ng_d_form=base["ng_d_form"],
    )


def base_swap_kwargs(base: dict) -> dict:
    """Project a base-pool state dict onto :func:`stable_get_dy`'s kwargs."""
    return {
        k: base[k]
        for k in (
            "balances",
            "rates",
            "amp",
            "fee",
            "a_precision",
            "ng_d_form",
            "offpeg_fee_multiplier",
            "legacy_fee_order",
        )
    }


# ===========================================================================
# Cryptoswap (Curve V2)
# ===========================================================================


def _geometric_mean(x: list[int]) -> int:
    """Integer geometric mean of *x* (Curve V2 helper; caller sorts desc).

    Vyper: ``CurveCryptoMath3.vy::geometric_mean``.
    """
    n = len(x)
    d = x[0]
    for _ in range(255):
        d_prev = d
        tmp = PRECISION
        for _x in x:
            tmp = tmp * _x // d
        d = d * ((n - 1) * PRECISION + tmp) // (n * PRECISION)
        diff = abs(d - d_prev)
        if diff <= 1 or diff * PRECISION < d:
            return d
    raise CurveConvergenceError("geometric_mean did not converge")


def _g1k0(gamma: int, k0: int) -> int:
    """``|gamma + 1e18 - K0| + 1`` — shared Newton term for Curve V2.

    Vyper: inlined in ``CurveCryptoMath3.vy::newton_D`` / ``newton_y`` (the
    ``g1k0`` block); extracted here because both iterations repeat it verbatim.
    """
    g1k0 = gamma + PRECISION
    return g1k0 - k0 + 1 if g1k0 > k0 else k0 - g1k0 + 1


def _mul1(d: int, gamma: int, g1k0: int, ann: int) -> int:
    """The ``mul1`` Newton term shared by :func:`newton_D` / :func:`newton_y`.

    Vyper: inlined in ``CurveCryptoMath3.vy::newton_D`` / ``newton_y``.
    """
    return PRECISION * d // gamma * g1k0 // gamma * g1k0 * A_MULTIPLIER // ann


def newton_D(ann: int, gamma: int, x_unsorted: list[int]) -> int:
    """Compute the cryptoswap invariant ``D`` (Curve V2 ``newton_D``).

    Args:
        ann: ``A`` as returned by the pool (already scaled by ``A_MULTIPLIER``
            and ``n**n``); used directly as ``ANN`` in the iteration.
        gamma: Pool ``gamma`` parameter (1e18 base).
        x_unsorted: Balances transformed into ``D`` space (precision- and
            price-scaled, see :func:`crypto_get_dy`).

    Vyper: ``CurveCryptoMath3.vy::newton_D``.
    """
    n = len(x_unsorted)
    x = sorted(x_unsorted, reverse=True)
    d = n * _geometric_mean(x)
    s = sum(x)

    for _ in range(255):
        d_prev = d
        k0 = PRECISION
        for _x in x:
            k0 = k0 * _x * n // d
        mul1 = _mul1(d, gamma, _g1k0(gamma, k0), ann)
        mul2 = (2 * PRECISION) * n // k0
        neg_fprime = (s + s * mul2 // PRECISION) + mul1 * n // k0 - mul2 * d // PRECISION
        d_plus = d * (neg_fprime + s) // neg_fprime
        d_minus = d * d // neg_fprime
        if PRECISION > k0:
            d_minus += d_plus * (mul1 // neg_fprime) // PRECISION * (PRECISION - k0) // k0
        else:
            d_minus -= d_plus * (mul1 // neg_fprime) // PRECISION * (k0 - PRECISION) // k0
        if d_plus > d_minus:
            d = d_plus - d_minus
        else:
            d = (d_minus - d_plus) // 2
        diff = abs(d - d_prev)
        if diff * 10**14 < max(10**16, d):
            return d
    raise CurveConvergenceError("newton_D did not converge")


def newton_y(ann: int, gamma: int, x: list[int], d: int, i: int) -> int:
    """Solve the cryptoswap invariant for coin *i*'s balance (Curve V2).

    Vyper: ``CurveCryptoMath3.vy::newton_y`` (the classic tricrypto2 solver;
    tricrypto-NG replaced it with an analytic ``get_y``, which agrees to
    ~1e-12 — see ``tests/live/test_curve_boa.py``).
    """
    n = len(x)
    y = d // n
    k0_i = PRECISION
    s_i = 0

    x_sorted = sorted((x[k] for k in range(n) if k != i), reverse=True)
    convergence_limit = max(x_sorted[0] // 10**14, d // 10**14, 100)
    for k in range(n - 1):
        y = y * d // (x_sorted[k] * n)
        s_i += x_sorted[k]
    for k in range(n - 1):
        k0_i = k0_i * x_sorted[k] * n // d

    for _ in range(255):
        y_prev = y
        k0 = k0_i * y * n // d
        s = s_i + y
        g1k0 = _g1k0(gamma, k0)
        mul1 = _mul1(d, gamma, g1k0, ann)
        mul2 = PRECISION + (2 * PRECISION) * k0 // g1k0
        yfprime = PRECISION * y + s * mul2 + mul1
        dyfprime = d * mul2
        if yfprime < dyfprime:
            y = y_prev // 2
            continue
        yfprime -= dyfprime
        fprime = yfprime // y
        y_minus = mul1 // fprime
        y_plus = (yfprime + PRECISION * d) // fprime + y_minus * PRECISION // k0
        y_minus += PRECISION * s // fprime
        if y_plus < y_minus:
            y = y_prev // 2
        else:
            y = y_plus - y_minus
        diff = abs(y - y_prev)
        if diff < max(convergence_limit, y // 10**14):
            return y
    raise CurveConvergenceError("newton_y did not converge")


def crypto_fee(xp: list[int], mid_fee: int, out_fee: int, fee_gamma: int) -> int:
    """Dynamic cryptoswap fee for transformed balances *xp* (Curve V2 ``_fee``).

    The fee interpolates between *mid_fee* (balanced) and *out_fee* (imbalanced)
    as the pool moves away from parity, controlled by *fee_gamma*.  All fee
    values are in units of ``1/1e10``.

    Vyper: ``CurveCryptoSwap.vy::_fee`` (the ``K`` term is
    ``CurveCryptoMath3.vy::reduction_coefficient``, folded inline here).
    """
    n = len(xp)
    sum_xp = sum(xp)
    k = PRECISION * n**n
    for x in xp:
        k = k * x // sum_xp
    f = fee_gamma * PRECISION // (fee_gamma + PRECISION - k)
    return (mid_fee * f + out_fee * (PRECISION - f)) // PRECISION


def crypto_get_dy(
    i: int,
    j: int,
    dx: int,
    balances: list[int],
    precisions: list[int],
    price_scale: list[int],
    amp: int,
    gamma: int,
    mid_fee: int,
    out_fee: int,
    fee_gamma: int,
    d: int | None = None,
) -> int:
    """Compute the output amount of a Curve V2 (cryptoswap) ``exchange`` locally.

    Args:
        i: Index of the input coin.
        j: Index of the output coin.
        dx: Raw input amount (coin *i* units).
        balances: Raw pool balances (coin units).
        precisions: Per-coin ``10**(18 - decimals)`` multipliers.
        price_scale: Per-coin price scale (1e18 base); ``price_scale[0]`` is the
            unit coin and must be ``PRECISION``.
        amp: Pool ``A`` (as returned on-chain — already scaled, used as ``ANN``).
        gamma: Pool ``gamma``.
        mid_fee: Fee at parity (units of ``1/1e10``).
        out_fee: Fee when fully imbalanced (units of ``1/1e10``).
        fee_gamma: Fee curvature parameter (1e18 base).
        d: Pool invariant ``D``. If ``None``, it is recomputed from *balances*
            via :func:`newton_D` (matches a non-ramping pool).

    Returns:
        Raw output amount in coin *j*'s units.

    Vyper: ``CurveCryptoViews3.vy::get_dy`` (the quoting wrapper around
    ``newton_y`` + ``_fee``; same flow in the pool's own ``get_dy``).
    """
    n = len(balances)

    def _transform(bals: list[int]) -> list[int]:
        return [bals[k] * precisions[k] * price_scale[k] // PRECISION for k in range(n)]

    if d is None:
        d = newton_D(amp, gamma, _transform(balances))

    bals = list(balances)
    bals[i] += dx
    xp = _transform(bals)

    y = newton_y(amp, gamma, xp, d, j)
    dy = xp[j] - y - 1
    xp[j] = y
    # Convert back out of D space into coin j units.
    dy = dy * PRECISION // price_scale[j]
    dy //= precisions[j]
    fee = crypto_fee(xp, mid_fee, out_fee, fee_gamma) * dy // FEE_DENOMINATOR
    return dy - fee

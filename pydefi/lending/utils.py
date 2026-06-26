from __future__ import annotations

from decimal import Decimal, getcontext

# Re-exported as part of this module's documented API (lending.utils): the
# "full balance" sentinel for supply / withdraw / repay, and what Aave returns
# as the health factor of an undebted account. Single definition in _utils.
from pydefi._utils import UINT256_MAX

#: Seconds in a calendar year (365 days) — the constant Aave and Compound
#: both use to annualise on-chain interest rates.
SECONDS_PER_YEAR: int = 31_536_000

#: ``1e18`` — the WAD fixed-point scale shared by Compound III and Morpho for
#: per-second rates and most ratios.
WAD: int = 10**18


def parse_health_factor(raw: int) -> Decimal:
    """Convert an on-chain WAD-scaled ``uint256`` health factor to a Decimal.

    Aave reports the health factor scaled by 1e18 (so 1e18 is exactly 1.00).
    A user with no debt has no meaningful ratio; Aave returns
    ``type(uint256).max`` there, surfaced here as ``Decimal("Infinity")``.
    """
    if raw == UINT256_MAX:
        return Decimal("Infinity")
    return Decimal(raw) / Decimal(WAD)


def per_second_rate_to_apy(rate_per_second_wad: int) -> Decimal:
    """Compound a 1e18-scaled per-second interest rate into an annual APY.

    Both Compound III (``Comet.getSupplyRate`` / ``getBorrowRate``) and
    Morpho (``IRM.borrowRateView``) report rates as per-second values scaled
    by 1e18. The annualised APY is ``(1 + r) ** SECONDS_PER_YEAR - 1`` where
    ``r = rate_per_second_wad / 1e18``.

    Args:
        rate_per_second_wad: Per-second rate, scaled by 1e18.

    Returns:
        APY as a :class:`~decimal.Decimal` fraction (``0.045`` = 4.5%).
    """
    if rate_per_second_wad == 0:
        return Decimal(0)
    # Bump precision for the exponentiation; the default 28 is too tight when
    # SECONDS_PER_YEAR is ~3.15e7.
    ctx = getcontext()
    prev = ctx.prec
    ctx.prec = max(prev, 50)
    try:
        per_second = Decimal(rate_per_second_wad) / Decimal(WAD)
        apy = (Decimal(1) + per_second) ** SECONDS_PER_YEAR - Decimal(1)
    finally:
        ctx.prec = prev
    return apy

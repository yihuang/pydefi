from __future__ import annotations

from decimal import Decimal, getcontext
from typing import Any, Literal

from pydefi.types import Address, Token, TokenAmount

#: ``type(uint256).max`` — the "full balance" sentinel for supply / withdraw /
#: repay, and what Aave returns as the health factor of an undebted account.
UINT256_MAX: int = (1 << 256) - 1

#: Seconds in a calendar year (365 days) — the constant Aave and Compound
#: both use to annualise on-chain interest rates.
SECONDS_PER_YEAR: int = 31_536_000

#: ``1e18`` — the WAD fixed-point scale shared by Compound III and Morpho for
#: per-second rates and most ratios.
WAD: int = 10**18


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


def resolve_amount(amount: TokenAmount | tuple[Token, Literal["max"]]) -> tuple[Token, int]:
    """Accept either an exact :class:`TokenAmount` or ``(token, "max")``."""
    if isinstance(amount, TokenAmount):
        return amount.token, amount.amount
    if isinstance(amount, tuple) and len(amount) == 2 and amount[1] == "max":
        return amount[0], UINT256_MAX
    raise TypeError("amount must be a TokenAmount or (Token, 'max') tuple")


def to_tx(to: Address, call_data: bytes) -> dict[str, Any]:
    """Format a calldata payload as the project-wide tx dict shape."""
    return {
        "to": to.to_0x_hex(),
        "data": "0x" + call_data.hex(),
        "value": "0",
    }

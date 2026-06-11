"""
Shared pure math helpers — slippage, fee basis points, price impact.

These are stateless, importable without any client instantiation.
"""

from __future__ import annotations

from decimal import Decimal

MAX_BPS = 10_000  #: 100% expressed in basis points.


def apply_slippage(amount: int, slippage_bps: int) -> int:
    """Return the minimum acceptable amount after applying slippage.

    Args:
        amount: Raw integer amount in the token's smallest unit.
        slippage_bps: Slippage tolerance in basis points (0–10000).

    Returns:
        ``amount * (10_000 - slippage_bps) // 10_000``
    """
    if not 0 <= slippage_bps <= MAX_BPS:
        raise ValueError(f"slippage_bps must be in [0, {MAX_BPS}], got {slippage_bps}")
    return int(amount * (MAX_BPS - slippage_bps) // MAX_BPS)


def slippage_to_fraction(slippage_bps: int) -> float:
    """Convert basis points to a decimal fraction (e.g. 50 → 0.005)."""
    return slippage_bps / MAX_BPS


def slippage_to_percent(slippage_bps: int) -> float:
    """Convert basis points to a percentage (e.g. 50 → 0.5)."""
    return slippage_bps / 100


def price_impact(amount_in_usd: Decimal, amount_out_usd: Decimal) -> Decimal:
    """Estimate price impact as a fraction in [0, 1]."""
    if amount_in_usd == 0:
        return Decimal(0)
    return abs(amount_in_usd - amount_out_usd) / amount_in_usd

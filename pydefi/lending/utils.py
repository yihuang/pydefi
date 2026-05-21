from __future__ import annotations

from typing import Any, Literal

from pydefi.types import Address, Token, TokenAmount

#: ``type(uint256).max`` — the "full balance" sentinel for supply / withdraw /
#: repay, and what Aave returns as the health factor of an undebted account.
UINT256_MAX: int = (1 << 256) - 1

#: Seconds in a calendar year (365 days) — the constant Aave and Compound
#: both use to annualise on-chain interest rates.
SECONDS_PER_YEAR: int = 31_536_000


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

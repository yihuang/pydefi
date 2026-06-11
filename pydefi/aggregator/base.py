"""Data types for DEX aggregator API integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydefi.types import Token, TokenAmount


@dataclass
class AggregatorQuote:
    """A swap quote returned by a DEX aggregator.

    Attributes:
        token_in: Input token.
        token_out: Output token.
        amount_in: Exact input amount.
        amount_out: Expected output amount.
        min_amount_out: Minimum acceptable output (after slippage).
        gas_estimate: Estimated gas units for the transaction.
        price_impact: Estimated price impact as a fraction (e.g. 0.005 = 0.5%).
        tx_data: Ready-to-broadcast transaction data (``to``, ``data``, ``value``).
        protocol: Aggregator name.
        route_summary: Human-readable description of the route.
    """

    token_in: Token
    token_out: Token
    amount_in: TokenAmount
    amount_out: TokenAmount
    min_amount_out: TokenAmount
    gas_estimate: int
    price_impact: Decimal = Decimal(0)
    tx_data: dict[str, Any] = field(default_factory=dict)
    protocol: str = ""
    route_summary: str = ""

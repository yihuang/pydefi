"""Data types for DEX aggregator API integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydefi._math import apply_slippage
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount


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

    def to_swap_route(self, *, pool_address: Address | None = None) -> SwapRoute:
        """Build a single-hop :class:`~pydefi.types.SwapRoute` from this quote.

        Args:
            pool_address: Address of the pool or router contract.  ``None`` for
                aggregator quotes where the individual pool is unknown.
        """
        step = SwapStep(
            token_in=self.token_in,
            token_out=self.token_out,
            pool_address=pool_address if pool_address is not None else None,
            protocol=self.protocol,
            fee=0,
        )
        return SwapRoute(
            steps=[step],
            amount_in=self.amount_in,
            amount_out=self.amount_out,
            price_impact=self.price_impact,
        )

    @classmethod
    def from_quote(
        cls,
        *,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        amount_out_raw: int,
        slippage_bps: int,
        gas_estimate: int = 0,
        price_impact: Decimal = Decimal(0),
        protocol: str = "",
        route_summary: str = "",
        tx_data: dict[str, Any] | None = None,
    ) -> AggregatorQuote:
        """Construct an :class:`AggregatorQuote` from raw quote fields.

        ``min_amount_out`` is computed from *amount_out_raw* via
        :func:`~pydefi._math.apply_slippage`.
        """
        min_out = apply_slippage(amount_out_raw, slippage_bps)
        return cls(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            min_amount_out=TokenAmount(token=token_out, amount=min_out),
            gas_estimate=gas_estimate,
            price_impact=price_impact,
            tx_data=tx_data or {},
            protocol=protocol,
            route_summary=route_summary,
        )

"""
Base class and data types for DEX aggregator API integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydefi.types import Token, TokenAmount, SwapRoute


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


class BaseAggregator(ABC):
    """Abstract base class for DEX aggregator API clients.

    Args:
        chain_id: EVM chain ID to query.
        api_key: Optional API key for authenticated endpoints.
    """

    def __init__(self, chain_id: int, api_key: str | None = None) -> None:
        self.chain_id = chain_id
        self.api_key = api_key

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the aggregator API."""

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Human-readable aggregator name."""

    @abstractmethod
    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a swap quote from the aggregator API.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Additional aggregator-specific parameters.

        Returns:
            An :class:`AggregatorQuote`.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """

    @abstractmethod
    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from an aggregator quote.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.

        Returns:
            A :class:`~pydefi.types.SwapRoute`.
        """

    def _slippage_to_fraction(self, slippage_bps: int) -> float:
        """Convert basis points to a fraction (e.g. 50 → 0.005)."""
        return slippage_bps / 10_000

    def _slippage_to_percent(self, slippage_bps: int) -> float:
        """Convert basis points to a percentage (e.g. 50 → 0.5)."""
        return slippage_bps / 100

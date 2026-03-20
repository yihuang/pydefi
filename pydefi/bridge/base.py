"""
Base class for cross-chain bridge integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydefi.types import BridgeQuote, Token, TokenAmount


class BaseBridge(ABC):
    """Abstract base class for cross-chain bridge integrations.

    Args:
        src_chain_id: Source (origin) chain ID.
        dst_chain_id: Destination chain ID.
    """

    def __init__(self, src_chain_id: int, dst_chain_id: int) -> None:
        self.src_chain_id = src_chain_id
        self.dst_chain_id = dst_chain_id

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Human-readable bridge protocol name."""

    @abstractmethod
    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Fetch a bridge quote.

        Args:
            token_in: Token on the source chain.
            token_out: Token on the destination chain.
            amount_in: Amount to bridge.
            **kwargs: Bridge-specific parameters.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API / contract errors.
        """

    @abstractmethod
    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the bridge transaction data.

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            **kwargs: Bridge-specific parameters.

        Returns:
            A dictionary containing the transaction fields
            (``to``, ``data``, ``value``, ``gas``).
        """

    def _apply_slippage(self, amount: int, slippage_bps: int) -> int:
        """Return minimum amount after applying slippage."""
        return int(amount * (10_000 - slippage_bps) // 10_000)

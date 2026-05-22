"""
Base class for cross-chain bridge integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydefi.types import Address, BridgeQuote, Token, TokenAmount


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
        recipient: Address,
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

    async def build_bridge_compose_tx(
        self,
        amount_in: TokenAmount,
        composer_address: Address,
        program: bytes,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the source-chain tx that bridges *amount_in* and runs *program*.

        One-signature cross-chain compose: the bridge moves *amount_in* to
        *composer_address* on the destination chain and, the instant the funds
        land, runs the embedded DeFiVM *program* — there is no follow-up leg.
        This is the bridge-agnostic entrypoint
        :func:`~pydefi.yields.build_compose_supply_route` builds on; bridges
        that support compose (CCTP, CCIP) override it.

        Args:
            amount_in: Token amount to bridge from the source chain.
            composer_address: The composer contract on the destination chain
                that receives the funds and executes *program*.
            program: Raw DeFiVM bytecode to run once the funds arrive.
            **kwargs: Bridge-specific parameters.

        Returns:
            A transaction dict (``to``, ``data``, ``value``, ``gas``).

        Raises:
            NotImplementedError: For bridges with no compose path.
        """
        raise NotImplementedError(f"{self.protocol_name} bridge has no compose path")

    @property
    def spender(self) -> Address | None:
        """The contract the source ERC-20 must be approved to before
        :meth:`build_bridge_tx` can pull it.

        ``None`` — the default — for bridges that move only the native asset
        (nothing to approve) or expose no single approval target. A bridge
        with no spender cannot carry an ERC-20 route through
        :func:`~pydefi.yields.build_yield_route`; token-pulling bridges
        override this with their entrypoint address.
        """
        return None

    def _apply_slippage(self, amount: int, slippage_bps: int) -> int:
        """Return minimum amount after applying slippage."""
        return int(amount * (10_000 - slippage_bps) // 10_000)

"""
Base classes for AMM DEX integrations (EVM and Solana).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from eth_contract import Contract
from web3 import AsyncWeb3

from pydefi.types import Address, SwapRoute, Token, TokenAmount


class BaseAMM(ABC):
    """Abstract base class for AMM DEX integrations.

    Sub-classes wrap specific on-chain AMM protocols (Uniswap V2/V3, Curve, …)
    and expose a uniform interface for querying prices and building swap routes.

    Args:
        w3: An :class:`~web3.AsyncWeb3` instance connected to the target chain.
        router_address: Address of the on-chain router/swap contract.
    """

    def __init__(self, w3: AsyncWeb3, router_address: Address) -> None:
        self.w3 = w3
        self.router_address = router_address

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Human-readable name of this AMM protocol."""

    @abstractmethod
    async def get_amounts_out(self, amount_in: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Simulate a swap and return the output amounts at each hop.

        Args:
            amount_in: Input token and amount.
            path: Ordered list of tokens representing the swap path.
                  Must have at least two elements (token_in and token_out).

        Returns:
            A list of :class:`~pydefi.types.TokenAmount` objects — one per
            token in *path* — where the last element is the final output
            amount.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If a pool
                along the path has insufficient liquidity.
        """

    @abstractmethod
    async def get_amounts_in(self, amount_out: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Calculate the required input amounts to obtain *amount_out*.

        Args:
            amount_out: Desired output token and amount.
            path: Ordered list of tokens representing the swap path.

        Returns:
            A list of :class:`~pydefi.types.TokenAmount` objects where the
            first element is the required input amount.
        """

    @abstractmethod
    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` for the given swap.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points
                (1 bp = 0.01 %).  Defaults to 50 bp (0.5 %).

        Returns:
            A fully described :class:`~pydefi.types.SwapRoute`.
        """

    @staticmethod
    def _apply_slippage(amount: int, slippage_bps: int) -> int:
        """Return the minimum acceptable amount after applying slippage."""
        if not 0 <= slippage_bps <= 10_000:
            raise ValueError("slippage_bps must be between 0 and 10_000 (inclusive)")
        return int(amount * (10_000 - slippage_bps) // 10_000)

    @staticmethod
    def _calculate_price_impact(amount_in_usd: Decimal, amount_out_usd: Decimal) -> Decimal:
        """Estimate price impact as a fraction in [0, 1]."""
        if amount_in_usd == 0:
            return Decimal(0)
        return abs(amount_in_usd - amount_out_usd) / amount_in_usd

    def _make_contract(self, abi_or_signatures: list, address: str) -> Contract:
        """Convenience: create a bound :class:`~eth_contract.Contract`."""
        return Contract.from_abi(abi_or_signatures, to=address)


class BaseSolanaAMM(ABC):
    """Abstract base class for Solana AMM integrations.

    Sub-classes wrap specific Solana AMM protocols (Raydium, Orca, …) and
    expose a uniform interface for querying prices and building swap routes
    using HTTP APIs rather than EVM contract calls.

    Args:
        api_url: Base URL for the AMM's HTTP API.  Sub-classes define a
            sensible default when *api_url* is ``None``.
    """

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Human-readable name of this AMM protocol."""

    @abstractmethod
    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> TokenAmount:
        """Get a swap quote for the given input and output tokens.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra protocol-specific parameters.

        Returns:
            Expected output :class:`~pydefi.types.TokenAmount`.
        """

    @abstractmethod
    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` for the given swap.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra protocol-specific parameters.

        Returns:
            A fully described :class:`~pydefi.types.SwapRoute`.
        """

    @staticmethod
    def _apply_slippage(amount: int, slippage_bps: int) -> int:
        """Return the minimum acceptable amount after applying slippage."""
        if not 0 <= slippage_bps <= 10_000:
            raise ValueError("slippage_bps must be between 0 and 10_000 (inclusive)")
        if not 0 <= slippage_bps <= 10_000:
            raise ValueError("slippage_bps must be between 0 and 10_000 (inclusive)")
        return int(amount * (10_000 - slippage_bps) // 10_000)

"""
Uniswap V2 (and forks: SushiSwap, PancakeSwap …) AMM integration.

The Uniswap V2 protocol uses the constant-product formula ``x * y = k``.
This module wraps the ``IUniswapV2Router02`` and ``IUniswapV2Factory``
interfaces via :class:`~eth_contract.Contract`.
"""

from __future__ import annotations

from decimal import Decimal

from eth_contract import Contract
from web3 import AsyncWeb3

from pydefi.abi.amm import UNISWAP_V2_FACTORY, UNISWAP_V2_PAIR, UNISWAP_V2_ROUTER
from pydefi.amm.base import BaseAMM
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount


class UniswapV2(BaseAMM):
    """Uniswap V2-compatible AMM integration.

    Works with any Uniswap V2 fork (SushiSwap, PancakeSwap, QuickSwap, …)
    by pointing ``router_address`` at the appropriate router contract.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        router_address: Address of the ``UniswapV2Router02`` contract.
        protocol_name: Override the default ``"UniswapV2"`` name.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        router_address: Address,
        protocol_name: str = "UniswapV2",
    ) -> None:
        super().__init__(w3, router_address)
        self._protocol_name = protocol_name

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    # ------------------------------------------------------------------
    # Factory / pair helpers
    # ------------------------------------------------------------------

    def get_factory_contract(self, factory_address: str) -> Contract:
        """Return a :class:`~eth_contract.Contract` bound to a V2 factory."""
        return UNISWAP_V2_FACTORY(to=factory_address)

    def get_pair_contract(self, pair_address: Address) -> Contract:
        """Return a :class:`~eth_contract.Contract` bound to a V2 pair."""
        return UNISWAP_V2_PAIR(to=pair_address)

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    async def get_amounts_out(self, amount_in: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Query the router for output amounts along *path*.

        Args:
            amount_in: Exact input amount.
            path: Swap path (token_in, [intermediate…], token_out).

        Returns:
            A list of :class:`~pydefi.types.TokenAmount` objects, one per
            token in *path*.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the
                router reverts due to insufficient liquidity.
            ValueError: If *path* has fewer than two elements.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")

        addresses = [t.address for t in path]
        try:
            raw_amounts: list[int] = await UNISWAP_V2_ROUTER.fns.getAmountsOut(amount_in.amount, addresses).call(
                self.w3, to=self.router_address
            )
        except Exception as exc:
            raise InsufficientLiquidityError(f"getAmountsOut failed: {exc}") from exc

        return [TokenAmount(token=path[i], amount=raw_amounts[i]) for i in range(len(path))]

    async def get_amounts_in(self, amount_out: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Query the router for required input amounts to obtain *amount_out*.

        Args:
            amount_out: Desired output amount.
            path: Swap path.

        Returns:
            A list of :class:`~pydefi.types.TokenAmount` objects.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the
                router reverts.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")

        addresses = [t.address for t in path]
        try:
            raw_amounts: list[int] = await UNISWAP_V2_ROUTER.fns.getAmountsIn(amount_out.amount, addresses).call(
                self.w3, to=self.router_address
            )
        except Exception as exc:
            raise InsufficientLiquidityError(f"getAmountsIn failed: {exc}") from exc

        return [TokenAmount(token=path[i], amount=raw_amounts[i]) for i in range(len(path))]

    # ------------------------------------------------------------------
    # Route builder
    # ------------------------------------------------------------------

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
    ) -> SwapRoute:
        """Build a direct (single-pair) swap route.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Slippage tolerance in basis points.

        Returns:
            A :class:`~pydefi.types.SwapRoute` for the requested swap.
        """
        path = [amount_in.token, token_out]
        amounts = await self.get_amounts_out(amount_in, path)
        amount_out = amounts[-1]

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=self.router_address,
            protocol=self.protocol_name,
            fee=3000,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=amount_out,
        )

    # ------------------------------------------------------------------
    # Constant-product math helpers (pure, no network calls)
    # ------------------------------------------------------------------

    @staticmethod
    def get_amount_out(amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int = 30) -> int:
        """Calculate output amount using the constant-product formula.

        Args:
            amount_in: Input amount (raw integer units).
            reserve_in: Reserve of the input token in the pool.
            reserve_out: Reserve of the output token in the pool.
            fee_bps: Pool swap fee in basis points (default 30 = 0.3%).

        Returns:
            Output amount (raw integer units).

        Raises:
            InsufficientLiquidityError: If reserves are zero.
        """
        if reserve_in == 0 or reserve_out == 0:
            raise InsufficientLiquidityError("Pool has no liquidity")
        fee_factor = 10_000 - fee_bps
        amount_in_with_fee = amount_in * fee_factor
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 10_000 + amount_in_with_fee
        return numerator // denominator

    @staticmethod
    def get_amount_in(amount_out: int, reserve_in: int, reserve_out: int, fee_bps: int = 30) -> int:
        """Calculate required input amount to receive *amount_out*.

        Args:
            amount_out: Desired output amount.
            reserve_in: Reserve of the input token.
            reserve_out: Reserve of the output token.
            fee_bps: Pool swap fee in basis points.

        Returns:
            Required input amount.

        Raises:
            InsufficientLiquidityError: If reserves are zero or insufficient.
        """
        if reserve_in == 0 or reserve_out == 0:
            raise InsufficientLiquidityError("Pool has no liquidity")
        if amount_out >= reserve_out:
            raise InsufficientLiquidityError("Insufficient output reserve")
        fee_factor = 10_000 - fee_bps
        numerator = reserve_in * amount_out * 10_000
        denominator = (reserve_out - amount_out) * fee_factor
        return numerator // denominator + 1

    @staticmethod
    def spot_price(reserve_in: int, reserve_out: int, decimals_in: int = 18, decimals_out: int = 18) -> Decimal:
        """Return the spot price of token_out in terms of token_in.

        Args:
            reserve_in: Reserve of the input token.
            reserve_out: Reserve of the output token.
            decimals_in: Decimal places of the input token.
            decimals_out: Decimal places of the output token.

        Returns:
            Price as a :class:`~decimal.Decimal`.
        """
        if reserve_in == 0:
            return Decimal(0)
        adj_in = Decimal(reserve_in) / Decimal(10**decimals_in)
        adj_out = Decimal(reserve_out) / Decimal(10**decimals_out)
        return adj_out / adj_in

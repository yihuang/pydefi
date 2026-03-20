"""
Uniswap V3 AMM integration.

Uniswap V3 uses concentrated liquidity with discrete fee tiers:
500 (0.05%), 3000 (0.3%), 10000 (1%).  This module wraps the
``ISwapRouter`` (V3) and ``IQuoterV2`` interfaces.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from eth_contract import Contract
from web3 import AsyncWeb3

from pydifi.amm.base import BaseAMM
from pydifi.exceptions import InsufficientLiquidityError
from pydifi.types import SwapRoute, SwapStep, Token, TokenAmount

# ---------------------------------------------------------------------------
# ABI struct NamedTuples
#
# NamedTuple is a subclass of tuple, so eth_abi's TupleEncoder accepts these
# directly while keeping fields readable by name.
# ---------------------------------------------------------------------------


class QuoteExactInputSingleParams(NamedTuple):
    """Params struct for ``QuoterV2.quoteExactInputSingle``."""

    tokenIn: str
    tokenOut: str
    amountIn: int
    fee: int
    sqrtPriceLimitX96: int


class QuoteExactOutputSingleParams(NamedTuple):
    """Params struct for ``QuoterV2.quoteExactOutputSingle``."""

    tokenIn: str
    tokenOut: str
    amount: int
    fee: int
    sqrtPriceLimitX96: int


class ExactInputSingleParams(NamedTuple):
    """Params struct for ``SwapRouter.exactInputSingle``."""

    tokenIn: str
    tokenOut: str
    fee: int
    recipient: str
    deadline: int
    amountIn: int
    amountOutMinimum: int
    sqrtPriceLimitX96: int


class ExactInputParams(NamedTuple):
    """Params struct for ``SwapRouter.exactInput``."""

    path: bytes
    recipient: str
    deadline: int
    amountIn: int
    amountOutMinimum: int


class ExactOutputSingleParams(NamedTuple):
    """Params struct for ``SwapRouter.exactOutputSingle``."""

    tokenIn: str
    tokenOut: str
    fee: int
    recipient: str
    deadline: int
    amountOut: int
    amountInMaximum: int
    sqrtPriceLimitX96: int


# ---------------------------------------------------------------------------
# ABI fragments
# ---------------------------------------------------------------------------

_QUOTER_V2_ABI = [
    "function quoteExactInputSingle((address tokenIn, address tokenOut, uint256 amountIn, uint24 fee, uint160 sqrtPriceLimitX96) params) external returns (uint256 amountOut, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate)",
    "function quoteExactOutputSingle((address tokenIn, address tokenOut, uint256 amount, uint24 fee, uint160 sqrtPriceLimitX96) params) external returns (uint256 amountIn, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate)",
    "function quoteExactInput(bytes path, uint256 amountIn) external returns (uint256 amountOut, uint160[] sqrtPriceX96AfterList, uint32[] initializedTicksCrossedList, uint256 gasEstimate)",
]

_ROUTER_V3_ABI = [
    "function exactInputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96) params) external payable returns (uint256 amountOut)",
    "function exactInput((bytes path, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum) params) external payable returns (uint256 amountOut)",
    "function exactOutputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountOut, uint256 amountInMaximum, uint160 sqrtPriceLimitX96) params) external payable returns (uint256 amountIn)",
]

_FACTORY_V3_ABI = [
    "function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)",
]

_POOL_V3_ABI = [
    "function slot0() external view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)",
    "function liquidity() external view returns (uint128)",
    "function fee() external view returns (uint24)",
    "function token0() external view returns (address)",
    "function token1() external view returns (address)",
]

# Canonical fee tiers (in hundredths of a basis point)
FEE_TIERS: tuple[int, ...] = (100, 500, 3000, 10000)


class UniswapV3(BaseAMM):
    """Uniswap V3 AMM integration.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        router_address: Address of the ``SwapRouter`` (V3) contract.
        quoter_address: Address of the ``QuoterV2`` contract.
        protocol_name: Override the default ``"UniswapV3"`` name.
        default_fee: Default fee tier to use when one is not specified
            (default ``3000`` = 0.3%).
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        router_address: str,
        quoter_address: str,
        protocol_name: str = "UniswapV3",
        default_fee: int = 3000,
    ) -> None:
        super().__init__(w3, router_address)
        self._protocol_name = protocol_name
        self.default_fee = default_fee
        self._router = Contract.from_abi(_ROUTER_V3_ABI, to=router_address)
        self._quoter = Contract.from_abi(_QUOTER_V2_ABI, to=quoter_address)

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    def get_factory_contract(self, factory_address: str) -> Contract:
        """Return a contract bound to a V3 factory."""
        return Contract.from_abi(_FACTORY_V3_ABI, to=factory_address)

    def get_pool_contract(self, pool_address: str) -> Contract:
        """Return a contract bound to a V3 pool."""
        return Contract.from_abi(_POOL_V3_ABI, to=pool_address)

    # ------------------------------------------------------------------
    # Price queries (via QuoterV2)
    # ------------------------------------------------------------------

    async def quote_exact_input_single(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        fee: int | None = None,
    ) -> TokenAmount:
        """Get a quote for a single-hop exact-input swap.

        Args:
            amount_in: Exact input amount.
            token_out: Output token.
            fee: Fee tier (defaults to ``self.default_fee``).

        Returns:
            Expected output :class:`~pydifi.types.TokenAmount`.

        Raises:
            :class:`~pydifi.exceptions.InsufficientLiquidityError`: If the
                quoter reverts.
        """
        fee = fee if fee is not None else self.default_fee
        params = QuoteExactInputSingleParams(
            tokenIn=amount_in.token.address,
            tokenOut=token_out.address,
            amountIn=amount_in.amount,
            fee=fee,
            sqrtPriceLimitX96=0,
        )
        try:
            result = await self._quoter.fns.quoteExactInputSingle(params).call(self.w3)
            amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(
                f"quoteExactInputSingle failed: {exc}"
            ) from exc

        return TokenAmount(token=token_out, amount=amount_out)

    async def get_amounts_out(
        self,
        amount_in: TokenAmount,
        path: list[Token],
        fees: list[int] | None = None,
    ) -> list[TokenAmount]:
        """Simulate a single-hop or multi-hop swap.

        For multi-hop swaps, ``fees`` specifies the fee tier for each hop.
        When omitted, ``self.default_fee`` is used for every hop.
        Use :meth:`quote_exact_input_single` for fine-grained single-hop control.

        Args:
            amount_in: Exact input amount.
            path: Swap path (at least two tokens).
            fees: Per-hop fee tiers (length must equal ``len(path) - 1``).
                Defaults to ``[self.default_fee] * (len(path) - 1)``.

        Returns:
            List of :class:`~pydifi.types.TokenAmount` objects.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")

        hop_fees = fees if fees is not None else [self.default_fee] * (len(path) - 1)
        if len(hop_fees) != len(path) - 1:
            raise ValueError(
                f"fees length ({len(hop_fees)}) must equal len(path) - 1 ({len(path) - 1})"
            )

        if len(path) == 2:
            out = await self.quote_exact_input_single(amount_in, path[1], fee=hop_fees[0])
            return [amount_in, out]

        # Multi-hop: encode path as bytes (tokenA + fee + tokenB + fee + tokenC …)
        encoded_path = self._encode_path(path, hop_fees)
        try:
            result = await self._quoter.fns.quoteExactInput(
                encoded_path, amount_in.amount
            ).call(self.w3)
            final_amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(
                f"quoteExactInput failed: {exc}"
            ) from exc

        # We only have the final output; intermediate amounts are unavailable
        # from quoteExactInput — return just start and end.
        return [amount_in, TokenAmount(token=path[-1], amount=final_amount_out)]

    async def get_amounts_in(
        self, amount_out: TokenAmount, path: list[Token]
    ) -> list[TokenAmount]:
        """Simulate an exact-output quote for a single-hop swap.

        Args:
            amount_out: Desired output.
            path: Swap path (must currently be exactly 2 tokens for exact-output).

        Returns:
            List of :class:`~pydifi.types.TokenAmount` objects.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")
        if len(path) != 2:
            raise ValueError(
                "get_amounts_in currently only supports single-hop (exactly 2 tokens) "
                "for exact-output quoting"
            )

        params = QuoteExactOutputSingleParams(
            tokenIn=path[0].address,
            tokenOut=amount_out.token.address,
            amount=amount_out.amount,
            fee=self.default_fee,
            sqrtPriceLimitX96=0,
        )
        try:
            result = await self._quoter.fns.quoteExactOutputSingle(params).call(self.w3)
            amount_in_raw = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(
                f"quoteExactOutputSingle failed: {exc}"
            ) from exc

        return [TokenAmount(token=path[0], amount=amount_in_raw), amount_out]

    # ------------------------------------------------------------------
    # Route builder
    # ------------------------------------------------------------------

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
    ) -> SwapRoute:
        """Build a direct single-hop V3 swap route.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Slippage tolerance in basis points.

        Returns:
            A :class:`~pydifi.types.SwapRoute`.
        """
        amount_out = await self.quote_exact_input_single(amount_in, token_out)

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=self.router_address,
            protocol=self.protocol_name,
            fee=self.default_fee,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=Decimal(0),
        )

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sqrt_price_to_price(
        sqrt_price_x96: int,
        token0_decimals: int = 18,
        token1_decimals: int = 18,
    ) -> Decimal:
        """Convert a V3 ``sqrtPriceX96`` value to a human-readable price.

        Args:
            sqrt_price_x96: The raw ``sqrtPriceX96`` value from ``slot0()``.
            token0_decimals: Decimals of token0.
            token1_decimals: Decimals of token1.

        Returns:
            Price of token0 denominated in token1.
        """
        sqrt_price = Decimal(sqrt_price_x96) / Decimal(2 ** 96)
        price_raw = sqrt_price ** 2
        adj = Decimal(10 ** token0_decimals) / Decimal(10 ** token1_decimals)
        return price_raw * adj

    @staticmethod
    def _encode_path(tokens: list[Token], fees: list[int]) -> bytes:
        """Encode a token path as ABI-packed bytes for V3 multi-hop calls.

        Args:
            tokens: Ordered list of tokens.
            fees: Fee tier between each consecutive pair of tokens.

        Returns:
            ABI-packed bytes path.
        """
        if len(fees) != len(tokens) - 1:
            raise ValueError("len(fees) must equal len(tokens) - 1")
        result = bytes.fromhex(tokens[0].address[2:].lower().zfill(40))
        for fee, token in zip(fees, tokens[1:]):
            result += fee.to_bytes(3, "big")
            result += bytes.fromhex(token.address[2:].lower().zfill(40))
        return result

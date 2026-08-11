"""
Uniswap V3 AMM integration.

Uniswap V3 uses concentrated liquidity with discrete fee tiers:
500 (0.05%), 3000 (0.3%), 10000 (1%).  This module wraps the
``ISwapRouter`` (V3) and ``IQuoterV2`` interfaces.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Sequence

from eth_contract import Contract
from web3 import AsyncWeb3

from pydefi.abi.amm import (
    UNISWAP_V3_FACTORY,
    UNISWAP_V3_POOL,
    UNISWAP_V3_QUOTER_V2,
    QuoteExactInputSingleParams,
    QuoteExactOutputSingleParams,
)
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.multicall import batch_call
from pydefi.pathfinder.v3_tick_math import TickLadder, get_sqrt_ratio_at_tick
from pydefi.pool_data.base import PoolData
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount

logger = logging.getLogger(__name__)

# Ladder fetches in flight at once. Each is several calls, two of them batched,
# and public RPCs answer a wide fan-out with a 429.
_LADDER_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Module-level pure functions
# ---------------------------------------------------------------------------


def v3_sqrt_price_to_price(
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
    sqrt_price = Decimal(sqrt_price_x96) / Decimal(2**96)
    price_raw = sqrt_price**2
    adj = Decimal(10**token0_decimals) / Decimal(10**token1_decimals)
    return price_raw * adj


def v3_encode_path(tokens: list[Token], fees: list[int]) -> bytes:
    """Encode a token path as ABI-packed bytes for V3 multi-hop calls.

    Args:
        tokens: Ordered list of tokens.
        fees: Fee tier between each consecutive pair of tokens.

    Returns:
        ABI-packed bytes path.
    """
    if len(fees) != len(tokens) - 1:
        raise ValueError("len(fees) must equal len(tokens) - 1")
    result = tokens[0].address
    for fee, token in zip(fees, tokens[1:]):
        result += fee.to_bytes(3, "big")
        result += token.address
    return result


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------


class UniswapV3:
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
        router_address: Address,
        quoter_address: Address,
        protocol_name: str = "UniswapV3",
        default_fee: int = 3000,
    ) -> None:
        self.w3 = w3
        self.router_address = router_address
        self._protocol_name = protocol_name
        self.default_fee = default_fee
        self.quoter_address = quoter_address

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    def get_factory_contract(self, factory_address: str) -> Contract:
        """Return a contract bound to a V3 factory."""
        return UNISWAP_V3_FACTORY(to=factory_address)

    def get_pool_contract(self, pool_address: str) -> Contract:
        """Return a contract bound to a V3 pool."""
        return UNISWAP_V3_POOL(to=pool_address)

    # ------------------------------------------------------------------
    # Price queries (via QuoterV2)
    # ------------------------------------------------------------------

    async def fetch_tick_ladder(
        self,
        pool_address: Address,
        *,
        tick_spacing: int | None = None,
        words: int = 4,
        max_ticks: int = 4096,
    ) -> TickLadder:
        """Read a pool's initialised ticks near the current price into a ladder.

        One call per bitmap word plus one per tick found, batched through
        Multicall3, so fetch once per pool and reuse the ladder.

        Accuracy holds only inside the scanned window and degrades silently
        past it: on WETH/USDC 0.05% a 20000 WETH trade is 8.85% out at
        ``words=1`` and 0.76% at ``words=4``. Widening is nearly free once
        batched, hence the wide default.

        Args:
            pool_address: The V3 pool.
            tick_spacing: Pool tick spacing; read from the pool when omitted.
            words: 256-tick bitmap words scanned either side of the current
                one. This, not *max_ticks*, is what bounds exact pricing.
            max_ticks: Boundaries kept nearest the price, bounding the batch.
        """
        pool = UNISWAP_V3_POOL.fns
        slot0 = await pool.slot0().call(self.w3, to=pool_address)
        current_tick = int(slot0[1])
        if tick_spacing is None:
            tick_spacing = int(await pool.tickSpacing().call(self.w3, to=pool_address))

        # Ticks live in the bitmap at `tick // spacing`, packed 256 per word.
        centre_word = (current_tick // tick_spacing) >> 8  # floors, matching the pool
        scanned = range(centre_word - words, centre_word + words + 1)
        bitmaps = await batch_call(self.w3, [(pool_address, pool.tickBitmap(word)) for word in scanned])

        initialised: list[int] = []
        for word, bitmap in zip(scanned, bitmaps):
            bitmap = int(bitmap)
            while bitmap:
                bit = (bitmap & -bitmap).bit_length() - 1  # lowest set bit
                initialised.append(((word << 8) + bit) * tick_spacing)
                bitmap &= bitmap - 1
        if not initialised:
            return TickLadder([])
        # Nearest-first, so a tight budget still covers the boundaries a swap
        # can actually reach.
        initialised.sort(key=lambda t: abs(t - current_tick))
        del initialised[max_ticks:]

        infos = await batch_call(self.w3, [(pool_address, pool.ticks(t)) for t in initialised])
        return TickLadder(
            [
                (tick, get_sqrt_ratio_at_tick(tick), int(info[1]))  # info[1] = liquidityNet
                for tick, info in zip(initialised, infos)
            ]
        )

    async def attach_tick_ladders(
        self,
        pools: Sequence[PoolData],
        *,
        concurrency: int = _LADDER_CONCURRENCY,
        **kwargs,
    ) -> list[PoolData]:
        """Fill ``tick_ladder`` on the V3 *pools*, in place, and return them.

        Each pool costs its own round trips, so run this over the candidates a
        search has narrowed to, not a whole pool set. Pools with no V3 price
        state are skipped; one whose fetch fails keeps the estimate.

        Args:
            pools: Pools to enrich, mutated in place.
            concurrency: Ladder fetches in flight at once.
            **kwargs: Passed through to :meth:`fetch_tick_ladder`.
        """
        targets = [p for p in pools if p.sqrt_price_x96 > 0 and p.liquidity > 0]
        sem = asyncio.Semaphore(concurrency)

        async def _one(pool: PoolData) -> None:
            async with sem:
                try:
                    pool.tick_ladder = await self.fetch_tick_ladder(Address(pool.pool_address), **kwargs)
                except Exception as exc:
                    logger.warning("tick ladder unavailable for %s (%s); keeping estimate", pool.pool_address, exc)

        await asyncio.gather(*(_one(p) for p in targets))
        return list(pools)

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
            Expected output :class:`~pydefi.types.TokenAmount`.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the
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
            result = await UNISWAP_V3_QUOTER_V2.fns.quoteExactInputSingle(params).call(self.w3, to=self.quoter_address)
            amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(f"quoteExactInputSingle failed: {exc}") from exc

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
            List of :class:`~pydefi.types.TokenAmount` objects.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")

        hop_fees = fees if fees is not None else [self.default_fee] * (len(path) - 1)
        if len(hop_fees) != len(path) - 1:
            raise ValueError(f"fees length ({len(hop_fees)}) must equal len(path) - 1 ({len(path) - 1})")

        if len(path) == 2:
            out = await self.quote_exact_input_single(amount_in, path[1], fee=hop_fees[0])
            return [amount_in, out]

        # Multi-hop: encode path as bytes (tokenA + fee + tokenB + fee + tokenC …)
        encoded_path = self._encode_path(path, hop_fees)
        try:
            result = await UNISWAP_V3_QUOTER_V2.fns.quoteExactInput(encoded_path, amount_in.amount).call(
                self.w3, to=self.quoter_address
            )
            final_amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(f"quoteExactInput failed: {exc}") from exc

        # We only have the final output; intermediate amounts are unavailable
        # from quoteExactInput — return just start and end.
        return [amount_in, TokenAmount(token=path[-1], amount=final_amount_out)]

    async def get_amounts_in(self, amount_out: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Simulate an exact-output quote for a single-hop swap.

        Args:
            amount_out: Desired output.
            path: Swap path (must currently be exactly 2 tokens for exact-output).

        Returns:
            List of :class:`~pydefi.types.TokenAmount` objects.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")
        if len(path) != 2:
            raise ValueError(
                "get_amounts_in currently only supports single-hop (exactly 2 tokens) for exact-output quoting"
            )

        params = QuoteExactOutputSingleParams(
            tokenIn=path[0].address,
            tokenOut=amount_out.token.address,
            amount=amount_out.amount,
            fee=self.default_fee,
            sqrtPriceLimitX96=0,
        )
        try:
            result = await UNISWAP_V3_QUOTER_V2.fns.quoteExactOutputSingle(params).call(self.w3, to=self.quoter_address)
            amount_in_raw = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(f"quoteExactOutputSingle failed: {exc}") from exc

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
            A :class:`~pydefi.types.SwapRoute`.
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

    sqrt_price_to_price = staticmethod(v3_sqrt_price_to_price)
    _encode_path = staticmethod(v3_encode_path)

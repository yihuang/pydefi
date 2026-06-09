"""Uniswap V4 AMM integration.

The on-chain quoter / state-reader / route-builder client for V4, mirroring
:class:`~pydefi.amm.uniswap_v3.UniswapV3`. Pools live in a singleton
``PoolManager`` keyed by ``PoolKey = (currency0, currency1, fee, tickSpacing,
hooks)``. State is read via ``StateView`` into a
:class:`~pydefi.pathfinder.graph.V4PoolEdge` (priced locally with the inherited
V3 math); quotes go through the V4 ``Quoter``. Swap execution lives elsewhere
(the Universal Router ``V4Hop`` encoding).
"""

from __future__ import annotations

from decimal import Decimal

from eth_abi import encode as abi_encode
from eth_utils import keccak
from web3 import AsyncWeb3

from pydefi.abi.amm import UNISWAP_V4_QUOTER, UNISWAP_V4_STATE_VIEW
from pydefi.amm.base import BaseAMM
from pydefi.deployments import get_address
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.pathfinder.graph import V4PoolEdge
from pydefi.types import ZERO_ADDRESS, Address, ChainId, SwapRoute, SwapStep, Token, TokenAmount

#: Canonical Uniswap V4 deployments on Ethereum mainnet, sourced from the
#: deployment registry.  Pass other addresses to the constructor for other chains.
MAINNET_POOL_MANAGER: Address = get_address("UNISWAP_V4_POOL_MANAGER", ChainId.ETHEREUM)
MAINNET_STATE_VIEW: Address = get_address("UNISWAP_V4_STATE_VIEW", ChainId.ETHEREUM)
MAINNET_QUOTER: Address = get_address("UNISWAP_V4_QUOTER", ChainId.ETHEREUM)


class UniswapV4(BaseAMM):
    """Uniswap V4 AMM integration (singleton PoolManager).

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        pool_manager_address: Address of the singleton ``PoolManager``.
        state_view_address: Address of the ``StateView`` periphery contract.
        quoter_address: Address of the V4 ``Quoter`` contract.
        protocol_name: Override the default ``"UniswapV4"`` name.
        default_fee: Default fee tier in **pips** (1e-6), e.g. ``500`` = 0.05%.
        default_tick_spacing: Default tick spacing (e.g. ``10`` for the 0.05% tier).
        default_hooks: Default hooks address (``ZERO_ADDRESS`` = no hooks).
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        pool_manager_address: Address,
        state_view_address: Address,
        quoter_address: Address,
        protocol_name: str = "UniswapV4",
        default_fee: int = 500,
        default_tick_spacing: int = 10,
        default_hooks: Address = ZERO_ADDRESS,
    ) -> None:
        super().__init__(w3, pool_manager_address)  # router_address := PoolManager singleton
        self._protocol_name = protocol_name
        self.state_view_address = state_view_address
        self.quoter_address = quoter_address
        self.default_fee = default_fee
        self.default_tick_spacing = default_tick_spacing
        self.default_hooks = default_hooks

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    # ------------------------------------------------------------------
    # PoolKey helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sort_currencies(a: Address, b: Address) -> tuple[Address, Address]:
        """Return ``(currency0, currency1)`` ordered by address, as V4 requires."""
        return (a, b) if int.from_bytes(bytes(a), "big") < int.from_bytes(bytes(b), "big") else (b, a)

    @classmethod
    def pool_id(cls, token_a: Address, token_b: Address, fee: int, tick_spacing: int, hooks: Address) -> bytes:
        """Return the 32-byte ``poolId`` (keccak256 of the ABI-encoded PoolKey).

        Currencies are sorted internally, so argument order does not matter.
        """
        c0, c1 = cls.sort_currencies(token_a, token_b)
        return keccak(
            abi_encode(
                ["address", "address", "uint24", "int24", "address"],
                [c0.to_0x_hex(), c1.to_0x_hex(), fee, tick_spacing, hooks.to_0x_hex()],
            )
        )

    def _pool_key_tuple(self, token_in: Token, token_out: Token, fee: int, tick_spacing: int, hooks: Address) -> tuple:
        """Build the ``(c0, c1, fee, tickSpacing, hooks)`` tuple plus ``zeroForOne``."""
        c0, c1 = self.sort_currencies(token_in.address, token_out.address)
        zero_for_one = c0 == token_in.address  # selling currency0?
        return (c0.to_0x_hex(), c1.to_0x_hex(), fee, tick_spacing, hooks.to_0x_hex()), zero_for_one, c0

    # ------------------------------------------------------------------
    # Pool state → local pricing edge
    # ------------------------------------------------------------------

    async def get_pool_edge(
        self,
        token_in: Token,
        token_out: Token,
        *,
        fee: int | None = None,
        tick_spacing: int | None = None,
        hooks: Address | None = None,
    ) -> V4PoolEdge:
        """Read live pool state from ``StateView`` and return a ``V4PoolEdge``.

        The returned edge prices ``token_in → token_out`` locally with the
        inherited V3 concentrated-liquidity math.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the pool
                has no liquidity (uninitialised key).
        """
        fee = fee if fee is not None else self.default_fee
        tick_spacing = tick_spacing if tick_spacing is not None else self.default_tick_spacing
        hooks = hooks if hooks is not None else self.default_hooks

        c0, c1 = self.sort_currencies(token_in.address, token_out.address)
        pool_id = self.pool_id(token_in.address, token_out.address, fee, tick_spacing, hooks)

        slot0 = await UNISWAP_V4_STATE_VIEW.fns.getSlot0(pool_id).call(self.w3, to=self.state_view_address)
        liquidity = await UNISWAP_V4_STATE_VIEW.fns.getLiquidity(pool_id).call(self.w3, to=self.state_view_address)
        sqrt_price_x96 = slot0[0] if isinstance(slot0, (list, tuple)) else slot0

        if not liquidity or not sqrt_price_x96:
            raise InsufficientLiquidityError(
                f"V4 pool {pool_id.hex()} ({token_in.symbol}/{token_out.symbol}) is uninitialised"
            )

        return V4PoolEdge(
            token_in=token_in,
            token_out=token_out,
            pool_address=self.router_address,  # singleton PoolManager
            protocol=self.protocol_name,
            fee_bps=fee // 100,  # pips → basis points (500 pips = 5 bps)
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
            is_token0_in=(c0 == token_in.address),
            tick_spacing=tick_spacing,
            hooks=hooks,
            pool_id=pool_id.hex(),
        )

    # ------------------------------------------------------------------
    # Price queries (via the V4 Quoter)
    # ------------------------------------------------------------------

    async def quote_exact_input_single(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        fee: int | None = None,
        tick_spacing: int | None = None,
        hooks: Address | None = None,
    ) -> TokenAmount:
        """On-chain quote for a single-hop exact-input swap via the V4 Quoter."""
        fee = fee if fee is not None else self.default_fee
        tick_spacing = tick_spacing if tick_spacing is not None else self.default_tick_spacing
        hooks = hooks if hooks is not None else self.default_hooks

        pool_key, zero_for_one, _c0 = self._pool_key_tuple(amount_in.token, token_out, fee, tick_spacing, hooks)
        params = (pool_key, zero_for_one, amount_in.amount, b"")
        try:
            result = await UNISWAP_V4_QUOTER.fns.quoteExactInputSingle(params).call(self.w3, to=self.quoter_address)
            amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(f"V4 quoteExactInputSingle failed: {exc}") from exc

        return TokenAmount(token=token_out, amount=amount_out)

    async def get_amounts_out(
        self,
        amount_in: TokenAmount,
        path: list[Token],
        fees: list[int] | None = None,
    ) -> list[TokenAmount]:
        """Simulate a single- or multi-hop exact-input swap (chained single quotes).

        ``fees`` gives the fee tier (pips) per hop; intermediate hops use the
        default tick spacing and hooks. Returns the amount at every node.
        """
        if len(path) < 2:
            raise ValueError("path must contain at least two tokens")
        hop_fees = fees if fees is not None else [self.default_fee] * (len(path) - 1)
        if len(hop_fees) != len(path) - 1:
            raise ValueError(f"fees length ({len(hop_fees)}) must equal len(path) - 1 ({len(path) - 1})")

        amounts = [amount_in]
        current = amount_in
        for token_out, hop_fee in zip(path[1:], hop_fees):
            current = await self.quote_exact_input_single(current, token_out, fee=hop_fee)
            amounts.append(current)
        return amounts

    async def get_amounts_in(self, amount_out: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """On-chain exact-output quote for a single-hop swap via the V4 Quoter."""
        if len(path) != 2:
            raise ValueError("get_amounts_in currently only supports single-hop (exactly 2 tokens)")

        token_in = path[0]
        pool_key, zero_for_one, _c0 = self._pool_key_tuple(
            token_in, amount_out.token, self.default_fee, self.default_tick_spacing, self.default_hooks
        )
        params = (pool_key, zero_for_one, amount_out.amount, b"")
        try:
            result = await UNISWAP_V4_QUOTER.fns.quoteExactOutputSingle(params).call(self.w3, to=self.quoter_address)
            amount_in_raw = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise InsufficientLiquidityError(f"V4 quoteExactOutputSingle failed: {exc}") from exc

        return [TokenAmount(token=token_in, amount=amount_in_raw), amount_out]

    # ------------------------------------------------------------------
    # Route builder
    # ------------------------------------------------------------------

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        *,
        fee: int | None = None,
        tick_spacing: int | None = None,
        hooks: Address | None = None,
    ) -> SwapRoute:
        """Build a direct single-hop V4 swap route.

        The ``SwapStep`` carries ``tick_spacing`` / ``hooks`` and the singleton
        PoolManager address; execution is left to the Universal Router.
        """
        fee = fee if fee is not None else self.default_fee
        tick_spacing = tick_spacing if tick_spacing is not None else self.default_tick_spacing
        hooks = hooks if hooks is not None else self.default_hooks

        amount_out = await self.quote_exact_input_single(
            amount_in, token_out, fee=fee, tick_spacing=tick_spacing, hooks=hooks
        )

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=self.router_address,  # singleton PoolManager
            protocol=self.protocol_name,
            fee=fee,
            tick_spacing=tick_spacing,
            hooks=hooks,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=Decimal(0),
        )

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

from dataclasses import dataclass, replace
from decimal import Decimal

from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError

from pydefi.abi.amm import UNISWAP_V4_QUOTER, UNISWAP_V4_STATE_VIEW
from pydefi.amm.v4_hooks import affects_swap_pricing
from pydefi.amm.v4_pool_key import pool_id, sort_currencies
from pydefi.deployments import get_address
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.pathfinder.graph import V4PoolEdge
from pydefi.types import ZERO_ADDRESS, Address, ChainId, SwapRoute, SwapStep, Token, TokenAmount

#: Canonical Uniswap V4 deployments on Ethereum mainnet, sourced from the
#: deployment registry.  Pass other addresses to the constructor for other chains.
MAINNET_POOL_MANAGER: Address = get_address("UNISWAP_V4_POOL_MANAGER", ChainId.ETHEREUM)
MAINNET_STATE_VIEW: Address = get_address("UNISWAP_V4_STATE_VIEW", ChainId.ETHEREUM)
MAINNET_QUOTER: Address = get_address("UNISWAP_V4_QUOTER", ChainId.ETHEREUM)

#: PoolKey ``fee`` value marking a dynamic-fee pool; the actual fee charged is
#: the hook-controlled ``lpFee`` returned by ``getSlot0``.
DYNAMIC_FEE_FLAG = 0x800000


@dataclass
class HookCalibration:
    """Result of probing a hooked pool's effective fee via the on-chain quoter.

    Attributes:
        implied_fee_pips: Hook-inclusive effective fee (pips; negative = subsidy).
        linear: Both probes implied the same fee — the hook take is proportional.
        deviation_pips: ``abs(fee_small - fee_large)``; large = custom curve
            (or a crossed tick boundary).
        probe_small: Small probe input amount (raw ``token_in`` units).
        probe_large: Large probe input amount (4x the small probe).
    """

    implied_fee_pips: int
    linear: bool
    deviation_pips: int
    probe_small: int
    probe_large: int


class UniswapV4:
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
        self.w3 = w3
        self.router_address = pool_manager_address  # router_address := PoolManager singleton
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
    # PoolKey helpers (shared with the Universal Router encoder)
    # ------------------------------------------------------------------

    #: ``(currency0, currency1)`` sorted by address — see :mod:`pydefi.amm.v4_pool_key`.
    sort_currencies = staticmethod(sort_currencies)
    #: 32-byte poolId from a PoolKey — see :mod:`pydefi.amm.v4_pool_key`.
    pool_id = staticmethod(pool_id)

    def _pool_key_tuple(self, token_in: Token, token_out: Token, fee: int, tick_spacing: int, hooks: Address) -> tuple:
        """Build the ``(c0, c1, fee, tickSpacing, hooks)`` tuple plus ``zeroForOne``."""
        c0, c1 = sort_currencies(token_in.address, token_out.address)
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
        calibrate_hooks: bool = False,
    ) -> V4PoolEdge:
        """Read live pool state from ``StateView`` and return a ``V4PoolEdge``.

        The returned edge prices ``token_in → token_out`` locally with the
        inherited V3 concentrated-liquidity math. The fee used for pricing is
        the live ``lpFee`` from ``slot0`` (in pips), so dynamic-fee pools are
        priced at their *current* fee. The protocol fee (``slot0.protocolFee``,
        zero on mainnet today) is not modelled.

        ``hook_affects_pricing`` is derived from the hook address's permission
        bits (:mod:`pydefi.amm.v4_hooks`); when set, local pricing is a
        hook-blind estimate — rank with it, confirm on-chain. Pass
        ``calibrate_hooks=True`` to run :meth:`calibrate_hook_fee` on such
        pools.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the pool
                has no liquidity (uninitialised key).
        """
        fee = fee if fee is not None else self.default_fee
        tick_spacing = tick_spacing if tick_spacing is not None else self.default_tick_spacing
        hooks = hooks if hooks is not None else self.default_hooks

        c0, _c1 = self.sort_currencies(token_in.address, token_out.address)
        pool_id = self.pool_id(token_in.address, token_out.address, fee, tick_spacing, hooks)

        slot0 = await UNISWAP_V4_STATE_VIEW.fns.getSlot0(pool_id).call(self.w3, to=self.state_view_address)
        liquidity = await UNISWAP_V4_STATE_VIEW.fns.getLiquidity(pool_id).call(self.w3, to=self.state_view_address)
        if isinstance(slot0, (list, tuple)):
            sqrt_price_x96, lp_fee = slot0[0], slot0[3]
        else:  # defensive: provider flattened the return to a single value
            sqrt_price_x96 = slot0
            if fee & DYNAMIC_FEE_FLAG:
                raise ValueError(f"cannot price dynamic-fee V4 pool {pool_id.hex()} without lpFee from slot0")
            lp_fee = fee

        if not liquidity or not sqrt_price_x96:
            raise InsufficientLiquidityError(
                f"V4 pool {pool_id.hex()} ({token_in.symbol}/{token_out.symbol}) is uninitialised"
            )

        is_dynamic_fee = bool(fee & DYNAMIC_FEE_FLAG)
        edge = V4PoolEdge(
            token_in=token_in,
            token_out=token_out,
            pool_address=self.router_address,  # singleton PoolManager
            protocol=self.protocol_name,
            fee_bps=lp_fee // 100,  # bps, display/identity only — pricing uses lp_fee_pips
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
            is_token0_in=(c0 == token_in.address),
            tick_spacing=tick_spacing,
            hooks=hooks,
            pool_id=pool_id.hex(),
            lp_fee_pips=lp_fee,
            key_fee_pips=fee,
            is_dynamic_fee=is_dynamic_fee,
            hook_affects_pricing=affects_swap_pricing(hooks, is_dynamic_fee=is_dynamic_fee),
        )
        if calibrate_hooks and edge.hook_affects_pricing:
            await self.calibrate_hook_fee(edge)
        return edge

    # ------------------------------------------------------------------
    # Hook-fee calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_amount(edge: V4PoolEdge) -> int:
        """Input amount that moves the pool price by ~1 bp (single-tick approx).

        Small enough to stay within the current tick on any realistic pool,
        large enough that integer rounding is negligible against pip-level
        fee resolution.
        """
        if edge.is_token0_in:
            amount = edge.liquidity * (1 << 96) // edge.sqrt_price_x96 // 10_000
        else:
            amount = edge.liquidity * edge.sqrt_price_x96 // (1 << 96) // 10_000
        return max(amount, 1)

    async def calibrate_hook_fee(
        self,
        edge: V4PoolEdge,
        *,
        hook_data: bytes = b"",
        tolerance_pips: int = 20,
    ) -> HookCalibration:
        """Back out a hooked pool's effective fee from two on-chain quotes.

        Quotes at 1x and 4x of a ~1 bp price move and compares against the
        local zero-fee curve; the shortfall is the hook-inclusive effective
        fee. If both probes agree within *tolerance_pips* (proportional hook
        take), the implied fee is folded into ``edge.lp_fee_pips`` and
        ``edge.hook_fee_calibrated`` is set; otherwise (custom curve, crossed
        tick) only the flag is cleared and the edge stays an estimate.

        Repeat-safe: the PoolKey comes from ``edge.key_fee_pips``, never the
        mutated ``lp_fee_pips``. Sender-dependent hooks may still quote
        differently than they execute.
        """
        key_fee = edge.key_fee_pips
        if not key_fee and not edge.is_dynamic_fee and not edge.hook_fee_calibrated:
            # Hand-built static edges may omit key_fee_pips; before any
            # calibration lp_fee_pips still equals the key fee. Backfill so
            # later calls key the same pool (fee-0 keys are legitimate).
            key_fee = edge.key_fee_pips = edge.lp_fee_pips

        probe_small = self._probe_amount(edge)
        probe_large = probe_small * 4
        # Zero-fee twin of the edge: raw concentrated-liquidity curve output.
        raw_curve = replace(edge, lp_fee_pips=0, fee_bps=0)

        implied: list[int] = []
        for amount in (probe_small, probe_large):
            quoted = await self.quote_exact_input_single(
                TokenAmount(token=edge.token_in, amount=amount),
                edge.token_out,
                fee=key_fee,
                tick_spacing=edge.tick_spacing,
                hooks=edge.hooks,
                hook_data=hook_data,
            )
            raw_out = raw_curve.amount_out(amount)
            if raw_out <= 0:
                raise InsufficientLiquidityError(f"V4 pool {edge.pool_id}: zero raw-curve output at probe {amount}")
            implied.append(round((1 - quoted.amount / raw_out) * 1_000_000))

        deviation = abs(implied[0] - implied[1])
        linear = deviation <= tolerance_pips
        result = HookCalibration(
            implied_fee_pips=implied[0],
            linear=linear,
            deviation_pips=deviation,
            probe_small=probe_small,
            probe_large=probe_large,
        )
        if linear:
            # fee_bps stays untouched: it is part of the router's edge identity.
            edge.lp_fee_pips = implied[0]
        # nonlinear revokes any stale trust from an earlier calibration
        edge.hook_fee_calibrated = linear
        return result

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
        hook_data: bytes = b"",
    ) -> TokenAmount:
        """On-chain quote for a single-hop exact-input swap via the V4 Quoter.

        The Quoter executes the real swap path inside the PoolManager lock
        (revert-and-catch), so hooks run for real: hook fees, dynamic-fee
        overrides and custom curves are all reflected in the returned amount.
        Pass *hook_data* for hooks that require it.
        """
        fee = fee if fee is not None else self.default_fee
        tick_spacing = tick_spacing if tick_spacing is not None else self.default_tick_spacing
        hooks = hooks if hooks is not None else self.default_hooks

        pool_key, zero_for_one, _c0 = self._pool_key_tuple(amount_in.token, token_out, fee, tick_spacing, hooks)
        params = (pool_key, zero_for_one, amount_in.amount, hook_data)
        try:
            result = await UNISWAP_V4_QUOTER.fns.quoteExactInputSingle(params).call(self.w3, to=self.quoter_address)
            amount_out = result[0] if isinstance(result, (list, tuple)) else result
        except ContractLogicError as exc:
            # Quoter revert = pool can't fill the swap; transport/RPC errors propagate.
            raise InsufficientLiquidityError(f"V4 quoteExactInputSingle reverted: {exc}") from exc

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
        except ContractLogicError as exc:
            # Quoter revert = pool can't fill the swap; transport/RPC errors propagate.
            raise InsufficientLiquidityError(f"V4 quoteExactOutputSingle reverted: {exc}") from exc

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

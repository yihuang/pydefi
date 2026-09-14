"""Live integration tests for the UniswapV4 client and V4PoolEdge pricing.

Against the mainnet WETH/USDC V4 pool (fee=500 pips, tickSpacing=10, no
hooks), these read live state via ``StateView`` and assert that local
``V4PoolEdge`` pricing matches the on-chain ``V4Quoter``, that
``build_swap_route`` emits a V4 step, and that the edge routes through the
pathfinder.
"""

import pytest

from pydefi.amm.uniswap_v4 import UniswapV4
from pydefi.pathfinder.graph import PoolGraph
from pydefi.pathfinder.router import Router
from pydefi.types import ZERO_ADDRESS, TokenAmount
from tests.addrs import (
    UNISWAP_V4_POOL_MANAGER,
    UNISWAP_V4_QUOTER,
    UNISWAP_V4_STATE_VIEW,
    USDC,
    WETH,
)

_AMOUNT_IN = 10**16  # 0.01 WETH — small enough to stay within the current tick


@pytest.fixture
def v4(eth_w3) -> UniswapV4:
    """A UniswapV4 client wired to the mainnet PoolManager / StateView / Quoter."""
    return UniswapV4(
        w3=eth_w3,
        pool_manager_address=UNISWAP_V4_POOL_MANAGER,
        state_view_address=UNISWAP_V4_STATE_VIEW,
        quoter_address=UNISWAP_V4_QUOTER,
    )


@pytest.fixture
def amount_in() -> TokenAmount:
    """0.01 WETH input used across the swap tests."""
    return TokenAmount(token=WETH, amount=_AMOUNT_IN)


@pytest.mark.live
class TestUniswapV4Live:
    """Live on-chain tests for the UniswapV4 client and V4PoolEdge pricing."""

    async def test_local_pricing_matches_quoter(self, v4: UniswapV4, amount_in: TokenAmount):
        """Local V4PoolEdge.amount_out should equal the on-chain V4Quoter output.

        For a small input that stays within the current tick, the inherited
        single-tick V3 math reproduces the quoter exactly.
        """
        edge = await v4.get_pool_edge(WETH, USDC)  # fee=500/tick=10/no-hooks defaults
        assert edge.liquidity > 0
        assert edge.pool_address == UNISWAP_V4_POOL_MANAGER
        assert edge.tick_spacing == 10

        local_out = edge.amount_out(_AMOUNT_IN)
        quoted = await v4.quote_exact_input_single(amount_in, USDC)

        assert quoted.token == USDC
        # 0.01 WETH is ~$5–$200 across any realistic ETH price.
        assert 5 * 10**6 < quoted.amount < 200 * 10**6, f"quote out of range: {quoted.amount / 10**6:.2f} USDC"

        rel_err = abs(local_out - quoted.amount) / quoted.amount
        assert rel_err < 1e-4, f"local={local_out} quoter={quoted.amount} ({rel_err:.6%})"

    async def test_build_swap_route(self, v4: UniswapV4, amount_in: TokenAmount):
        """build_swap_route emits a single V4 step carrying the pool key."""
        route = await v4.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0
        assert len(route.steps) == 1
        step = route.steps[0]
        assert step.protocol == v4.protocol_name
        assert step.pool_address == UNISWAP_V4_POOL_MANAGER
        assert step.tick_spacing == 10
        assert step.hooks == ZERO_ADDRESS

    async def test_pathfinder_routes_v4_edge(self, v4: UniswapV4, amount_in: TokenAmount):
        """A V4PoolEdge should be routable: find_best_route yields a single V4 hop."""
        graph = PoolGraph()
        graph.add_pool(await v4.get_pool_edge(WETH, USDC))

        route = Router(graph).find_best_route(amount_in, USDC)

        assert len(route.steps) == 1
        assert route.steps[0].protocol == v4.protocol_name
        assert route.amount_out.amount > 0

    async def test_calibration_recovers_static_fee(self, v4: UniswapV4):
        """calibrate_hook_fee against the real quoter recovers the pool's effective fee.

        The WETH/USDC pool is hookless, so the effective fee is just the 500-pip
        lpFee composed with the live protocol fee. That fee is a governance
        parameter, so the expectation is read from slot0 rather than hardcoded.
        """
        edge = await v4.get_pool_edge(WETH, USDC)
        assert not edge.hook_affects_pricing  # hookless pool
        assert edge.lp_fee_pips == 500  # static-fee pool: lpFee is the key fee

        expected = edge.effective_fee_pips()
        result = await v4.calibrate_hook_fee(edge)

        assert result.linear, f"deviation {result.deviation_pips} pips between probes"
        assert abs(result.implied_fee_pips - expected) <= 5, (
            f"implied {result.implied_fee_pips} pips, expected ~{expected} "
            f"(lpFee 500 + protocolFee {edge.protocol_fee_pips})"
        )
        assert edge.hook_fee_calibrated
        # Calibration folds the total take into lp_fee_pips.
        assert abs(edge.lp_fee_pips - expected) <= 5

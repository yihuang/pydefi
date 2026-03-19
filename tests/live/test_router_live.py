"""Live integration tests for the pathfinder Router using on-chain Uniswap V2 reserves.

These tests fetch live pool reserves from well-known Uniswap V2 pairs on
Ethereum mainnet, populate a :class:`~pydifi.pathfinder.graph.PoolGraph`, and
then verify that :class:`~pydifi.pathfinder.router.Router` returns plausible
routes and amounts.
"""

import pytest

from eth_contract import Contract
from pydifi.exceptions import NoRouteFoundError
from pydifi.pathfinder.graph import PoolEdge, PoolGraph
from pydifi.pathfinder.router import Router
from pydifi.types import TokenAmount

from .conftest import DAI, USDC, USDT, WETH

# ---------------------------------------------------------------------------
# Well-known Uniswap V2 pair addresses (Ethereum mainnet)
# ---------------------------------------------------------------------------

PAIR_WETH_USDC = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"  # WETH/USDC
PAIR_WETH_DAI = "0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11"   # WETH/DAI
PAIR_USDC_DAI = "0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5"   # USDC/DAI
PAIR_USDC_USDT = "0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f"  # USDC/USDT

_PAIR_ABI = [
    "function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)",
    "function token0() external view returns (address)",
]


async def _get_reserves(w3, pair_address: str, token_a, token_b):
    """Return (reserve_a, reserve_b) sorted to match (token_a, token_b) order."""
    pair = Contract.from_abi(_PAIR_ABI, to=pair_address)
    token0_addr = await pair.fns.token0().call(w3)
    reserves = await pair.fns.getReserves().call(w3)
    reserve0, reserve1 = reserves[0], reserves[1]
    if token0_addr.lower() == token_a.address.lower():
        return reserve0, reserve1
    return reserve1, reserve0


@pytest.mark.live
class TestRouterLive:
    """Live pathfinder router tests using on-chain Uniswap V2 pool reserves."""

    async def _build_graph(self, w3) -> PoolGraph:
        """Fetch live reserves and build a PoolGraph with 4 pairs."""
        g = PoolGraph()

        # WETH ↔ USDC
        r_weth, r_usdc = await _get_reserves(w3, PAIR_WETH_USDC, WETH, USDC)
        g.add_bidirectional_pool(WETH, USDC, PAIR_WETH_USDC, "UniswapV2",
                                 reserve_a=r_weth, reserve_b=r_usdc, fee_bps=30)

        # WETH ↔ DAI
        r_weth2, r_dai = await _get_reserves(w3, PAIR_WETH_DAI, WETH, DAI)
        g.add_bidirectional_pool(WETH, DAI, PAIR_WETH_DAI, "UniswapV2",
                                 reserve_a=r_weth2, reserve_b=r_dai, fee_bps=30)

        # USDC ↔ DAI
        r_usdc2, r_dai2 = await _get_reserves(w3, PAIR_USDC_DAI, USDC, DAI)
        g.add_bidirectional_pool(USDC, DAI, PAIR_USDC_DAI, "UniswapV2",
                                 reserve_a=r_usdc2, reserve_b=r_dai2, fee_bps=30)

        # USDC ↔ USDT
        r_usdc3, r_usdt = await _get_reserves(w3, PAIR_USDC_USDT, USDC, USDT)
        g.add_bidirectional_pool(USDC, USDT, PAIR_USDC_USDT, "UniswapV2",
                                 reserve_a=r_usdc3, reserve_b=r_usdt, fee_bps=30)

        return g

    async def test_direct_route_weth_to_usdc(self, eth_w3):
        """1 WETH → USDC: direct route should yield a plausible price."""
        g = await self._build_graph(eth_w3)
        router = Router(g)
        amount_in = TokenAmount.from_human(WETH, "1")
        route = router.find_best_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0
        # Sanity: 1 WETH should be worth between $500 and $10 000
        assert 500 * 10**6 < route.amount_out.amount < 10_000 * 10**6, (
            f"WETH→USDC out of expected range: {route.amount_out.human_amount} USDC"
        )

    async def test_two_hop_route_weth_to_usdt(self, eth_w3):
        """1 WETH → USDT: requires 2 hops (WETH→USDC→USDT)."""
        g = await self._build_graph(eth_w3)
        # Remove direct WETH/USDT path (not in graph) — only 2-hop available
        router = Router(g, max_hops=3)
        amount_in = TokenAmount.from_human(WETH, "1")
        route = router.find_best_route(amount_in, USDT)

        assert route.token_in == WETH
        assert route.token_out == USDT
        assert route.amount_out.amount > 0
        assert len(route.steps) >= 2

    async def test_find_all_routes_weth_to_dai(self, eth_w3):
        """find_all_routes: WETH → DAI should return both direct and 2-hop routes."""
        g = await self._build_graph(eth_w3)
        router = Router(g, max_hops=3)
        amount_in = TokenAmount.from_human(WETH, "1")
        routes = router.find_all_routes(amount_in, DAI, top_k=5)

        assert len(routes) >= 1
        # Routes are sorted by descending output amount
        for i in range(len(routes) - 1):
            assert routes[i].amount_out.amount >= routes[i + 1].amount_out.amount
        # Best route should be for ~1 WETH ≈ $500–$10 000 of DAI
        best = routes[0]
        assert 500 * 10**18 < best.amount_out.amount < 10_000 * 10**18, (
            f"WETH→DAI out of expected range: {best.amount_out.human_amount} DAI"
        )

    async def test_same_token_raises(self, eth_w3):
        """Router should raise ValueError when token_in == token_out."""
        g = await self._build_graph(eth_w3)
        router = Router(g)
        amount_in = TokenAmount.from_human(WETH, "1")
        with pytest.raises(ValueError, match="different"):
            router.find_best_route(amount_in, WETH)

    async def test_find_all_routes_same_token_raises(self, eth_w3):
        """find_all_routes should raise ValueError when token_in == token_out."""
        g = await self._build_graph(eth_w3)
        router = Router(g)
        amount_in = TokenAmount.from_human(WETH, "1")
        with pytest.raises(ValueError, match="different"):
            router.find_all_routes(amount_in, WETH)

    async def test_no_route_raises(self, eth_w3):
        """Router should raise NoRouteFoundError when no path exists."""
        g = PoolGraph()  # empty graph
        router = Router(g)
        amount_in = TokenAmount.from_human(WETH, "1")
        with pytest.raises(NoRouteFoundError):
            router.find_best_route(amount_in, USDC)

    async def test_route_fee_in_correct_units(self, eth_w3):
        """SwapStep.fee from router should be in hundredths of a basis-point."""
        g = await self._build_graph(eth_w3)
        router = Router(g)
        amount_in = TokenAmount.from_human(WETH, "1")
        route = router.find_best_route(amount_in, USDC)

        # Uniswap V2 is 30 bps; in hundredths of a bp that is 3000
        for step in route.steps:
            assert step.fee == 3000, (
                f"Expected fee=3000 (30 bps in hundredths), got fee={step.fee}"
            )

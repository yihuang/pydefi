"""Tests for pydefi.pathfinder — graph and router."""

import math
from decimal import Decimal

import pytest

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
from pydefi.pathfinder.router import Router
from pydefi.types import Address, ChainId, RouteDAG, RouteSwap, Token, TokenAmount
from tests.addrs import DAI, USDC, WETH

# ---------------------------------------------------------------------------
# Test tokens
# ---------------------------------------------------------------------------

WBTC = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"),
    symbol="WBTC",
    decimals=8,
)

POOL_A: Address = Address("0x" + "11" * 20)
POOL_B: Address = Address("0x" + "22" * 20)
POOL_C: Address = Address("0x" + "33" * 20)
POOL_D: Address = Address("0x" + "44" * 20)


# ---------------------------------------------------------------------------
# PoolEdge tests
# ---------------------------------------------------------------------------


class TestPoolEdge:
    def test_spot_price(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            fee_bps=30,
        )
        price = edge.spot_price
        assert price == Decimal("2000")

    def test_spot_price_zero_reserve(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=0,
            reserve_out=2_000_000 * 10**6,
        )
        assert edge.spot_price == Decimal(0)

    def test_amount_out_basic(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            fee_bps=30,
        )
        out = edge.amount_out(10**18)  # 1 WETH
        assert out > 1_990 * 10**6  # > 1990 USDC
        assert out < 2_000 * 10**6  # < 2000 USDC

    def test_amount_out_no_reserves(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
        )
        assert edge.amount_out(10**18) == 0

    def test_log_weight_finite(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            fee_bps=30,
        )
        weight = edge.log_weight(10**18)
        assert weight < float("inf")
        assert weight != 0  # fee causes some loss

    def test_log_weight_no_liquidity(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
        )
        assert edge.log_weight(10**18) == float("inf")

    def test_estimate_price_impact_nonzero(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            fee_bps=30,
        )
        impact = edge.estimate_price_impact(10**18)
        assert Decimal(0) < impact < Decimal(1)

    def test_estimate_price_impact_zero_reserve(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=0,
            reserve_out=0,
        )
        assert edge.estimate_price_impact(10**18).is_nan()

    def test_effective_log_weight_penalizes_high_gas(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=DAI,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=3_400_000 * 10**18,
            fee_bps=30,
            extra={"estimated_gas": 180_000},
        )
        low_gas = edge.effective_log_weight(
            10**18,
            gas_price_gwei=5,
            native_token_price_usd=3000,
            token_out_price_usd=1,
        )
        high_gas = edge.effective_log_weight(
            10**18,
            gas_price_gwei=100,
            native_token_price_usd=3000,
            token_out_price_usd=1,
        )
        assert high_gas > low_gas


# ---------------------------------------------------------------------------
# V3PoolEdge tests
# ---------------------------------------------------------------------------


def _make_v3_edge(
    token_in,
    token_out,
    pool_address,
    is_token0_in: bool,
    price_usdc_per_weth: float = 2000.0,
    fee_bps: int = 30,
):
    """Build a V3PoolEdge with a synthetic sqrtPriceX96 at the given USDC/WETH price."""
    # P_raw = price_raw_token1_per_token0 (in smallest units)
    if is_token0_in:
        # token_in = WETH (token0), token_out = USDC (token1)
        P_raw = price_usdc_per_weth * (10**token_out.decimals) / (10**token_in.decimals)
    else:
        # token_in = USDC (token1), token_out = WETH (token0)
        # P_raw is still defined as token1/token0 for the underlying pool
        P_raw = price_usdc_per_weth * (10**token_in.decimals) / (10**token_out.decimals)
    sqrtP_real = math.sqrt(P_raw)
    Q96 = 2**96
    sqrt_price_x96 = int(sqrtP_real * Q96)
    liquidity = 5 * 10**22  # large pool
    return V3PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=pool_address,
        protocol="UniswapV3",
        fee_bps=fee_bps,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=liquidity,
        is_token0_in=is_token0_in,
    )


class TestV3PoolEdge:
    def test_spot_price_token0_in(self):
        """Spot price should be ~2000 USDC/WETH when WETH is token0."""
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        assert abs(edge.spot_price - Decimal("2000")) < Decimal("1")

    def test_spot_price_token1_in(self):
        """Spot price should be ~1/2000 WETH/USDC when USDC is token1 in."""
        edge = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        assert abs(edge.spot_price - Decimal("1") / Decimal("2000")) < Decimal("0.001")

    def test_spot_price_zero_sqrt_price(self):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=0,
            liquidity=10**22,
        )
        assert edge.spot_price == Decimal(0)

    def test_amount_out_token0_in(self):
        """Swapping 1 WETH should yield ~1994 USDC (after 0.3% fee at $2000)."""
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        out = edge.amount_out(10**18)
        assert 1_990 * 10**6 < out < 1_998 * 10**6, f"Got {out / 10**6:.2f} USDC"

    def test_amount_out_token1_in(self):
        """Swapping 2000 USDC should yield ~0.997 WETH (after 0.3% fee)."""
        edge = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        out = edge.amount_out(2000 * 10**6)
        assert int(0.994 * 10**18) < out < int(0.998 * 10**18), f"Got {out / 10**18:.6f} WETH"

    def test_amount_out_zero_liquidity(self):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=10**20,
            liquidity=0,
        )
        assert edge.amount_out(10**18) == 0

    def test_amount_out_zero_sqrt_price(self):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=0,
            liquidity=10**22,
        )
        assert edge.amount_out(10**18) == 0

    def test_log_weight_finite(self):
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        weight = edge.log_weight(10**18)
        assert weight < float("inf")
        assert weight > 0  # fee causes loss

    def test_estimate_price_impact_token0_in(self):
        """V3 price impact should be a sensible positive fraction."""
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        impact = edge.estimate_price_impact(10**18)  # 1 WETH
        assert not impact.is_nan()
        assert Decimal(0) < impact < Decimal(1)

    def test_estimate_price_impact_token1_in(self):
        """V3 price impact should be sensible for token1→token0 direction."""
        edge = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        impact = edge.estimate_price_impact(2000 * 10**6)  # 2000 USDC
        assert not impact.is_nan()
        assert Decimal(0) < impact < Decimal(1)

    def test_estimate_price_impact_zero_liquidity(self):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=10**20,
            liquidity=0,
        )
        assert edge.estimate_price_impact(10**18).is_nan()

    def test_estimate_price_impact_zero_sqrt_price(self):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=0,
            liquidity=10**22,
        )
        assert edge.estimate_price_impact(10**18).is_nan()

    def test_v3_edge_in_router(self):
        """Router should find a route through a V3 pool edge."""
        g = PoolGraph()
        edge_in = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        edge_out = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        g.add_pool(edge_in)
        g.add_pool(edge_out)
        router = Router(g)
        route = router.find_best_route(TokenAmount(WETH, 10**18), USDC)
        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 1_990 * 10**6


# ---------------------------------------------------------------------------
# PoolGraph tests
# ---------------------------------------------------------------------------


class TestPoolGraph:
    def _make_graph(self) -> PoolGraph:
        """Build a small test graph: WETH ↔ USDC ↔ DAI, WETH ↔ WBTC."""
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=2_000_000 * 10**6,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "Curve",
            reserve_a=1_000_000 * 10**6,
            reserve_b=1_000_000 * 10**18,
            fee_bps=4,
        )
        g.add_bidirectional_pool(
            WETH,
            WBTC,
            POOL_C,
            "UniswapV3",
            reserve_a=500 * 10**18,
            reserve_b=20 * 10**8,
            fee_bps=5,
        )
        return g

    def test_add_bidirectional_pool(self):
        g = self._make_graph()
        assert len(g.edges_from(WETH)) == 2  # WETH → USDC, WETH → WBTC
        assert len(g.edges_from(USDC)) == 2  # USDC → WETH, USDC → DAI

    def test_edges_from(self):
        g = self._make_graph()
        edges = g.edges_from(WETH)
        out_tokens = {e.token_out.symbol for e in edges}
        assert "USDC" in out_tokens
        assert "WBTC" in out_tokens

    def test_edges_to(self):
        g = self._make_graph()
        edges = g.edges_to(USDC)
        in_tokens = {e.token_in.symbol for e in edges}
        assert "WETH" in in_tokens
        assert "DAI" in in_tokens

    def test_tokens_property(self):
        g = self._make_graph()
        symbols = {t.symbol for t in g.tokens}
        assert "WETH" in symbols
        assert "USDC" in symbols
        assert "DAI" in symbols
        assert "WBTC" in symbols

    def test_len(self):
        g = self._make_graph()
        # 3 bidirectional pools → 6 directed edges
        assert len(g) == 6

    def test_add_single_edge(self):
        g = PoolGraph()
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=10**21,
            reserve_out=2 * 10**9,
        )
        g.add_pool(edge)
        assert len(g.edges_from(WETH)) == 1
        assert len(g.edges_from(USDC)) == 0

    def test_iterate_graph(self):
        g = self._make_graph()
        edges = list(g)
        assert len(edges) == 6


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestRouter:
    def _make_graph(self) -> PoolGraph:
        g = PoolGraph()
        # WETH ↔ USDC  (direct)
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=2_000_000 * 10**6,
            fee_bps=30,
        )
        # USDC ↔ DAI  (stableswap)
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "Curve",
            reserve_a=5_000_000 * 10**6,
            reserve_b=5_000_000 * 10**18,
            fee_bps=4,
        )
        # WETH ↔ DAI  (indirect route available via WETH→USDC→DAI and direct)
        g.add_bidirectional_pool(
            WETH,
            DAI,
            POOL_C,
            "UniswapV3",
            reserve_a=500 * 10**18,
            reserve_b=1_000_000 * 10**18,
            fee_bps=3,
        )
        return g

    def test_find_best_route_direct(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        route = router.find_best_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0
        assert route.dag is not None
        assert isinstance(route.dag, RouteDAG)
        dag_payload = route.dag.to_dict()
        assert dag_payload["token_in"] == WETH
        assert isinstance(dag_payload["actions"][0], RouteSwap)
        assert len(dag_payload["actions"]) == len(route.steps)
        assert [a.pool.pool_address for a in dag_payload["actions"]] == [s.pool_address for s in route.steps]

    def test_find_best_route_two_hop(self):
        g = PoolGraph()
        # Only path: WETH → USDC → DAI (no direct WETH→DAI)
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=2_000_000 * 10**6,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "Curve",
            reserve_a=5_000_000 * 10**6,
            reserve_b=5_000_000 * 10**18,
            fee_bps=4,
        )
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        route = router.find_best_route(amount_in, DAI)

        assert route.token_in == WETH
        assert route.token_out == DAI
        assert len(route.steps) == 2
        assert route.dag is not None
        dag_payload = route.dag.to_dict()
        assert [a.pool.pool_address for a in dag_payload["actions"]] == [POOL_A, POOL_B]

    def test_find_best_route_no_path_raises(self):
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=2 * 10**9,
        )
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        with pytest.raises(NoRouteFoundError):
            router.find_best_route(amount_in, DAI)  # DAI not in graph

    def test_find_best_route_same_token_raises(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        with pytest.raises(ValueError):
            router.find_best_route(amount_in, WETH)

    def test_find_best_route_max_hops_respected(self):
        """Router should not find routes longer than max_hops."""
        g = PoolGraph()
        # Chain: WETH → USDC → DAI → WBTC (3 hops)
        g.add_pool(PoolEdge(WETH, USDC, POOL_A, "UniswapV2", 10**21, 2 * 10**9, 30))
        g.add_pool(PoolEdge(USDC, DAI, POOL_B, "Curve", 10**9, 10**21, 4))
        g.add_pool(PoolEdge(DAI, WBTC, POOL_C, "UniswapV3", 10**21, 10**8, 5))

        router_1hop = Router(g, max_hops=1)
        with pytest.raises(NoRouteFoundError):
            router_1hop.find_best_route(TokenAmount(WETH, 10**18), WBTC)

        router_3hop = Router(g, max_hops=3)
        route = router_3hop.find_best_route(TokenAmount(WETH, 10**18), WBTC)
        assert route.token_out == WBTC

    def test_find_all_routes_returns_sorted(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        routes = router.find_all_routes(amount_in, DAI)

        assert len(routes) >= 1
        # Routes should be sorted by descending output amount
        for i in range(len(routes) - 1):
            assert routes[i].amount_out.amount >= routes[i + 1].amount_out.amount
        assert all(route.dag is not None for route in routes)

    def test_find_all_routes_top_k(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10**18)
        routes = router.find_all_routes(amount_in, DAI, top_k=1)
        assert len(routes) == 1

    def test_find_all_routes_no_path_raises(self):
        g = PoolGraph()
        router = Router(g)
        with pytest.raises(NoRouteFoundError):
            router.find_all_routes(TokenAmount(WETH, 10**18), USDC)

    def test_price_impact_zero_reserves(self):
        """Routes through zero-reserve pools (e.g. V3) should return NaN (unestimated)."""
        edges = [PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=0, reserve_out=0)]
        impact = Router._estimate_price_impact(edges, 10**18)
        assert impact.is_nan()

    def test_price_impact_nonzero(self):
        edges = [
            PoolEdge(
                WETH,
                USDC,
                POOL_A,
                "UniswapV2",
                reserve_in=1_000 * 10**18,
                reserve_out=2_000_000 * 10**6,
                fee_bps=30,
            )
        ]
        impact = Router._estimate_price_impact(edges, 10**18)
        assert Decimal(0) < impact < Decimal(1)

    def test_find_best_route_with_gas_prefers_lower_hop_when_gas_is_high(self):
        g = PoolGraph()
        # Two-hop route has slightly better gross output:
        # WETH -> USDC -> DAI
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=3_400_000 * 10**6,
            fee_bps=0,
        )
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "Curve",
            reserve_a=3_400_000 * 10**6,
            reserve_b=3_395_000 * 10**18,
            fee_bps=0,
        )
        # Direct one-hop route has slightly worse gross output.
        g.add_bidirectional_pool(
            WETH,
            DAI,
            POOL_C,
            "UniswapV3",
            reserve_a=1_000 * 10**18,
            reserve_b=3_380_000 * 10**18,
            fee_bps=0,
        )

        router = Router(g, max_hops=3)
        amount_in = TokenAmount(token=WETH, amount=10**18)

        gross_best = router.find_best_route(amount_in, DAI)
        assert len(gross_best.steps) == 2

        net_best = router.find_best_route(
            amount_in,
            DAI,
            gas_price_gwei=100,
            native_token_price_usd=3000,
            token_out_price_usd=1,
            max_hops=3,
        )
        assert len(net_best.steps) == 1

    def test_graph_level_gas_aware_differs_from_gross_on_same_candidates(self):
        g = PoolGraph()
        # Candidate A (2 hops): slightly better gross output.
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=3_420_000 * 10**6,
            fee_bps=0,
        )
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "Curve",
            reserve_a=3_420_000 * 10**6,
            reserve_b=3_410_000 * 10**18,
            fee_bps=0,
        )
        # Candidate B (1 hop): slightly worse gross output, lower gas.
        g.add_bidirectional_pool(
            WETH,
            DAI,
            POOL_C,
            "UniswapV3",
            reserve_a=1_000 * 10**18,
            reserve_b=3_390_000 * 10**18,
            fee_bps=0,
        )

        router = Router(g, max_hops=3)
        amount_in = TokenAmount(token=WETH, amount=10**18)

        candidates = router.find_all_routes(amount_in, DAI, top_k=5)
        assert len(candidates) >= 2

        gross_best = router.find_best_route(amount_in, DAI)
        gas_best = router.find_best_route(
            amount_in,
            DAI,
            gas_price_gwei=120,
            native_token_price_usd=3000,
            token_out_price_usd=1,
            max_hops=3,
        )

        gross_path = tuple(step.pool_address for step in gross_best.steps)
        gas_path = tuple(step.pool_address for step in gas_best.steps)
        candidate_paths = {tuple(step.pool_address for step in r.steps) for r in candidates}

        # Same candidate universe, different objective.
        assert gross_path in candidate_paths
        assert gas_path in candidate_paths
        assert gross_path != gas_path
        assert len(gross_best.steps) == 2
        assert len(gas_best.steps) == 1

    def test_find_best_route_dag(self):
        g = self._make_graph()
        router = Router(g)
        dag = router.find_best_route_dag(TokenAmount(token=WETH, amount=10**18), USDC)
        payload = dag.to_dict()
        assert payload["token_in"] == WETH
        assert len(payload["actions"]) >= 1
        assert payload["actions"][-1].token_out == USDC

    def test_find_all_routes_dag(self):
        g = self._make_graph()
        router = Router(g)
        dags = router.find_all_routes_dag(TokenAmount(token=WETH, amount=10**18), DAI, top_k=2)
        assert len(dags) >= 1
        assert all(isinstance(dag, RouteDAG) for dag in dags)


# ---------------------------------------------------------------------------
# find_best_split tests
# ---------------------------------------------------------------------------


class TestFindBestSplit:
    """Tests for Router.find_best_split — N-way split routing."""

    POOL_A2 = "0x" + "55" * 20  # second WETH→USDC pool

    def _make_split_graph(self) -> PoolGraph:
        """Two independent WETH→USDC pools so a split is possible."""
        g = PoolGraph()
        g.add_pool(
            PoolEdge(
                WETH, USDC, POOL_A, "UniswapV2", reserve_in=1_000 * 10**18, reserve_out=2_000_000 * 10**6, fee_bps=30
            )
        )
        g.add_pool(
            PoolEdge(
                WETH,
                USDC,
                self.POOL_A2,
                "UniswapV3",
                reserve_in=1_000 * 10**18,
                reserve_out=2_000_000 * 10**6,
                fee_bps=5,
            )
        )
        return g

    def test_single_pool_returns_linear_dag(self):
        """With only one route available the result is a linear DAG (no split node)."""
        g = PoolGraph()
        g.add_pool(PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30))
        router = Router(g)
        dag = router.find_best_split(TokenAmount(WETH, 10**18), USDC)
        payload = dag.to_dict()
        assert payload["token_in"] == WETH
        from pydefi.types import RouteSplit

        assert not any(isinstance(a, RouteSplit) for a in payload["actions"])
        assert payload["actions"][-1].token_out == USDC

    def test_two_pools_may_produce_split(self):
        """With two pools of equal depth a split is at least as good as a single route."""
        g = self._make_split_graph()
        router = Router(g)
        amount_in = TokenAmount(WETH, 10**18)
        dag = router.find_best_split(amount_in, USDC)
        assert isinstance(dag, RouteDAG)
        payload = dag.to_dict()
        assert payload["token_in"] == WETH
        assert payload["actions"][-1].token_out == USDC

    def test_split_dag_structure(self):
        """When a split is chosen the DAG root action is a RouteSplit.

        Two shallow equal pools (same fee, 10 ETH reserve each) with a 1 ETH
        input: price impact per pool is ~9% for 100% allocation but only ~5%
        per pool for 50/50, so splitting strictly wins.
        """
        from pydefi.types import RouteSplit

        g = PoolGraph()
        g.add_pool(
            PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=10 * 10**18, reserve_out=20_000 * 10**6, fee_bps=30)
        )
        g.add_pool(
            PoolEdge(
                WETH, USDC, self.POOL_A2, "UniswapV2", reserve_in=10 * 10**18, reserve_out=20_000 * 10**6, fee_bps=30
            )
        )
        router = Router(g)
        dag = router.find_best_split(TokenAmount(WETH, 10**18), USDC, step_bps=5000)
        payload = dag.to_dict()
        split = payload["actions"][0]
        assert isinstance(split, RouteSplit)
        assert sum(leg.fraction_bps for leg in split.legs) == 10_000
        assert split.token_out == USDC

    def test_max_splits_one_returns_linear(self):
        """max_splits=1 forces a single-route result even when two pools exist."""
        g = self._make_split_graph()
        router = Router(g)
        dag = router.find_best_split(TokenAmount(WETH, 10**18), USDC, max_splits=1)
        payload = dag.to_dict()
        from pydefi.types import RouteSplit

        assert not any(isinstance(a, RouteSplit) for a in payload["actions"])

    def test_invalid_max_splits_raises(self):
        g = self._make_split_graph()
        router = Router(g)
        with pytest.raises(ValueError, match="max_splits"):
            router.find_best_split(TokenAmount(WETH, 10**18), USDC, max_splits=0)

    def test_no_route_raises(self):
        g = PoolGraph()
        g.add_pool(PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30))
        router = Router(g)
        with pytest.raises(NoRouteFoundError):
            router.find_best_split(TokenAmount(WETH, 10**18), DAI)

    def test_find_top_routes_multi_state_same_destination(self):
        """Two intermediate states at hop-1 both expand to WBTC at hop-2.

        Exercises the path where multiple ``current_states`` entries contribute
        to the same ``next_key`` in ``_find_top_routes``.  Both routes must be
        returned with correct output amounts sorted descending.
        """
        g = PoolGraph()
        # hop-1: WETH → USDC and WETH → DAI (two independent intermediate states)
        g.add_pool(PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30))
        g.add_pool(PoolEdge(WETH, DAI, POOL_B, "UniswapV2", reserve_in=10**21, reserve_out=2 * 10**21, fee_bps=30))
        # hop-2: both intermediate tokens expand to WBTC (same next_key (WBTC, 2))
        g.add_pool(PoolEdge(USDC, WBTC, POOL_C, "UniswapV2", reserve_in=2 * 10**9, reserve_out=5 * 10**7, fee_bps=30))
        g.add_pool(PoolEdge(DAI, WBTC, POOL_D, "UniswapV2", reserve_in=2 * 10**21, reserve_out=5 * 10**7, fee_bps=30))

        router = Router(g, max_hops=2)
        routes = router._find_top_routes(TokenAmount(WETH, 10**18), WBTC, top_n=2)

        assert len(routes) == 2
        assert all(len(r.steps) == 2 for r in routes)
        assert all(r.token_out == WBTC for r in routes)
        # diverse: each route starts through a different first-hop pool
        assert {r.steps[0].pool_address for r in routes} == {POOL_A, POOL_B}
        # sorted descending by output
        assert routes[0].amount_out.amount >= routes[1].amount_out.amount


# ---------------------------------------------------------------------------
# Router.simulate
# ---------------------------------------------------------------------------


class TestRouterSimulate:
    """Verify off-chain DAG simulation using constant-product pool math."""

    def _make_edge(
        self, token_in: Token, token_out: Token, pool: Address, reserve_in: int, reserve_out: int, fee_bps: int = 30
    ) -> PoolEdge:
        return PoolEdge(
            token_in=token_in,
            token_out=token_out,
            pool_address=pool,
            protocol="UniswapV2",
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            fee_bps=fee_bps,
        )

    def test_linear_single_hop_matches_pool_math(self):
        """simulate() result equals edge.amount_out() for a single-hop DAG."""
        edge = self._make_edge(WETH, USDC, POOL_A, 1_000 * 10**18, 2_000_000 * 10**6)
        g = PoolGraph()
        g.add_pool(edge)
        router = Router(g)

        amount_in = 10**18
        dag = router.find_best_route(TokenAmount(WETH, amount_in), USDC).dag
        result = router.simulate(dag, amount_in)

        assert result == edge.amount_out(amount_in)
        assert result > 0

    def test_two_hop_chains_amount_out(self):
        """simulate() chains amount_out through both hops."""
        edge_ab = self._make_edge(WETH, USDC, POOL_A, 1_000 * 10**18, 2_000_000 * 10**6)
        edge_bc = self._make_edge(USDC, DAI, POOL_B, 5_000_000 * 10**6, 5_000_000 * 10**18, fee_bps=4)
        g = PoolGraph()
        g.add_pool(edge_ab)
        g.add_pool(edge_bc)
        router = Router(g)

        amount_in = 10**18
        dag = router.find_best_route(TokenAmount(WETH, amount_in), DAI).dag
        result = router.simulate(dag, amount_in)

        mid = edge_ab.amount_out(amount_in)
        expected = edge_bc.amount_out(mid)
        assert result == expected

    def test_split_dag_sums_legs(self):
        """simulate() sums each split leg proportionally."""
        edge_a1 = self._make_edge(WETH, USDC, POOL_A, 10 * 10**18, 20_000 * 10**6)
        edge_a2 = self._make_edge(WETH, USDC, POOL_B, 10 * 10**18, 20_000 * 10**6)
        g = PoolGraph()
        g.add_pool(edge_a1)
        g.add_pool(edge_a2)
        router = Router(g)

        amount_in = 10**18
        dag = router.find_best_split(TokenAmount(WETH, amount_in), USDC, step_bps=5000)
        result = router.simulate(dag, amount_in)

        # Manual: each leg gets amount_in * bps / 10000
        from pydefi.types import RouteSplit

        payload = dag.to_dict()
        split = payload["actions"][0]
        assert isinstance(split, RouteSplit)
        expected = sum(leg.actions[0].pool.amount_out(amount_in * leg.fraction_bps // 10_000) for leg in split.legs)
        assert result == expected

    def test_zero_reserves_returns_zero(self):
        """simulate() propagates zero when a pool has no reserves."""
        edge = self._make_edge(WETH, USDC, POOL_A, 0, 0)
        g = PoolGraph()
        g.add_pool(edge)
        router = Router(g)

        dag = RouteDAG().from_token(WETH)
        dag.swap(USDC, edge)
        assert router.simulate(dag, 10**18) == 0

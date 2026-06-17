"""Tests for pydefi.pathfinder — graph and router."""

from decimal import Decimal

import pytest

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.asgm import _MARGINAL_EPS, _bundle_marginal, asgm_optimize
from pydefi.pathfinder.dag import RouteDAG, RouteSplit, RouteSwap
from pydefi.pathfinder.graph import PoolEdge, PoolGraph, ReserveOverlay, V3PoolEdge
from pydefi.pathfinder.multipath import MultiEdgePath, PathWeights, _distribute_int, merge_and_expand
from pydefi.pathfinder.router import Router
from pydefi.types import Address, ChainId, Token, TokenAmount
from tests._fixtures import make_v3_edge
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

    @pytest.mark.parametrize("amount_in", [0, -1, -(10**15)])
    def test_amount_out_non_positive_input_returns_zero(self, amount_in):
        """Quote contract is non-negative; negative inputs must not produce a negative quote."""
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10**18,
            reserve_out=2_000_000 * 10**6,
            fee_bps=30,
        )
        assert edge.amount_out(amount_in) == 0

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


class TestV3PoolEdge:
    @pytest.mark.parametrize(
        "tin,tout,is_token0_in,expected,tol",
        [
            (WETH, USDC, True, Decimal("2000"), Decimal("1")),
            (USDC, WETH, False, Decimal("1") / Decimal("2000"), Decimal("0.001")),
        ],
        ids=["token0_in", "token1_in"],
    )
    def test_spot_price(self, tin, tout, is_token0_in, expected, tol):
        edge = make_v3_edge(tin, tout, POOL_A, is_token0_in=is_token0_in)
        assert abs(edge.spot_price - expected) < tol

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

    @pytest.mark.parametrize(
        "tin,tout,is_token0_in,amount_in,lo,hi",
        [
            (WETH, USDC, True, 10**18, 1_990 * 10**6, 1_998 * 10**6),
            (USDC, WETH, False, 2000 * 10**6, int(0.994 * 10**18), int(0.998 * 10**18)),
        ],
        ids=["token0_in_1WETH", "token1_in_2000USDC"],
    )
    def test_amount_out(self, tin, tout, is_token0_in, amount_in, lo, hi):
        edge = make_v3_edge(tin, tout, POOL_A, is_token0_in=is_token0_in)
        assert lo < edge.amount_out(amount_in) < hi

    @pytest.mark.parametrize(
        "sqrt_price,liquidity",
        [(10**20, 0), (0, 10**22)],
        ids=["zero_liquidity", "zero_sqrt_price"],
    )
    def test_amount_out_degenerate_state_returns_zero(self, sqrt_price, liquidity):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=sqrt_price,
            liquidity=liquidity,
        )
        assert edge.amount_out(10**18) == 0

    def test_log_weight_finite(self):
        edge = make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        weight = edge.log_weight(10**18)
        assert weight < float("inf")
        assert weight > 0  # fee causes loss

    @pytest.mark.parametrize(
        "tin,tout,is_token0_in,amount_in",
        [
            (WETH, USDC, True, 10**18),
            (USDC, WETH, False, 2000 * 10**6),
        ],
        ids=["token0_in", "token1_in"],
    )
    def test_estimate_price_impact(self, tin, tout, is_token0_in, amount_in):
        edge = make_v3_edge(tin, tout, POOL_A, is_token0_in=is_token0_in)
        impact = edge.estimate_price_impact(amount_in)
        assert not impact.is_nan()
        assert Decimal(0) < impact < Decimal(1)

    @pytest.mark.parametrize(
        "sqrt_price,liquidity",
        [(10**20, 0), (0, 10**22)],
        ids=["zero_liquidity", "zero_sqrt_price"],
    )
    def test_estimate_price_impact_degenerate_state_is_nan(self, sqrt_price, liquidity):
        edge = V3PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV3",
            sqrt_price_x96=sqrt_price,
            liquidity=liquidity,
        )
        assert edge.estimate_price_impact(10**18).is_nan()

    def test_v3_edge_in_router(self):
        """Router should find a route through a V3 pool edge."""
        g = PoolGraph()
        edge_in = make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        edge_out = make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
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
        with pytest.raises(ValueError, match="must be different"):
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
# _find_top_routes (used by find_optimal_split)
# ---------------------------------------------------------------------------


class TestFindTopRoutes:
    def test_multi_state_same_destination(self):
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
        dag = router.find_optimal_split(TokenAmount(WETH, amount_in), USDC)
        result = router.simulate(dag, amount_in)

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


# ---------------------------------------------------------------------------
# Helpers + tests for ReserveOverlay and sequenced K-split routing
# ---------------------------------------------------------------------------


def _v2_pool_edge(
    token_in,
    token_out,
    pool_addr,
    *,
    reserve_in: int = 10**21,
    reserve_out: int = 2 * 10**9,
    fee_bps: int = 30,
    is_token0_in: bool = True,
) -> PoolEdge:
    return PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=pool_addr,
        protocol="UniswapV2",
        reserve_in=reserve_in,
        reserve_out=reserve_out,
        fee_bps=fee_bps,
        extra={"is_token0_in": is_token0_in},
    )


def _v2_weth_usdc_edge(*, is_token0_in: bool = True) -> PoolEdge:
    """V2 WETH/USDC edge at $2000 with deep reserves."""
    return _v2_pool_edge(
        WETH,
        USDC,
        POOL_A,
        reserve_in=1_000 * 10**18,
        reserve_out=2_000_000 * 10**6,
        is_token0_in=is_token0_in,
    )


class TestReserveOverlay:
    def test_empty(self):
        ov = ReserveOverlay()
        assert not ov.v2_deltas and not ov.v3_sqrt_price
        assert ov.v2_deltas.get(POOL_A, (0, 0)) == (0, 0)
        assert ov.v3_sqrt_price.get(POOL_A, 12345) == 12345

    def test_v2_delta_accumulates_via_record_swap(self):
        edge = _v2_weth_usdc_edge()
        ov = ReserveOverlay()
        edge.record_swap(ov, 10**18, 2 * 10**6)
        edge.record_swap(ov, 5 * 10**17, 10**6)
        assert ov.v2_deltas[POOL_A] == (15 * 10**17, -3 * 10**6)
        assert ov.v2_deltas.get(POOL_B, (0, 0)) == (0, 0)

    def test_v3_sqrt_price_replaces_not_accumulates(self):
        ov = ReserveOverlay()
        ov.v3_sqrt_price[POOL_A] = 79228162514264337593543950336  # 1.0 in Q96
        ov.v3_sqrt_price[POOL_A] = 1
        assert ov.v3_sqrt_price[POOL_A] == 1

    def test_pool_isolation(self):
        ov = ReserveOverlay()
        ov.v2_deltas[POOL_A] = (1, -1)
        ov.v3_sqrt_price[POOL_B] = 999
        assert POOL_B not in ov.v2_deltas
        assert POOL_A not in ov.v3_sqrt_price


class TestPoolEdgeOverlay:
    def test_record_swap_then_quote_matches_advanced_reserves(self):
        """amount_out under overlay must match a fresh edge with advanced reserves."""
        edge = _v2_weth_usdc_edge()
        ov = ReserveOverlay()
        x1 = 5 * 10**18
        y1 = edge.amount_out(x1)
        edge.record_swap(ov, x1, y1)
        advanced = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=edge.reserve_in + x1,
            reserve_out=edge.reserve_out - y1,
            fee_bps=30,
        )
        assert edge.amount_out(3 * 10**18, ov) == advanced.amount_out(3 * 10**18)

    def test_overlay_couples_bidirectional_edges(self):
        """A swap on A→B updates the effective reserves of B→A."""
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
        forward = next(e for e in g.edges_from(WETH) if e.token_out.address == USDC.address)
        reverse = next(e for e in g.edges_from(USDC) if e.token_out.address == WETH.address)
        ov = ReserveOverlay()
        x = 10 * 10**18
        y = forward.amount_out(x)
        forward.record_swap(ov, x, y)
        eff_in_rev, eff_out_rev = reverse._effective_reserves(ov)
        assert eff_in_rev == reverse.reserve_in - y
        assert eff_out_rev == reverse.reserve_out + x

    def test_record_swap_token1_direction(self):
        """is_token0_in=False must store deltas in the opposite slots."""
        edge = _v2_weth_usdc_edge(is_token0_in=False)
        ov = ReserveOverlay()
        x = 4 * 10**18
        y = edge.amount_out(x)
        edge.record_swap(ov, x, y)
        d0, d1 = ov.v2_deltas[POOL_A]
        assert d1 == x and d0 == -y

    def test_zero_swap_is_noop(self):
        edge = _v2_weth_usdc_edge()
        ov = ReserveOverlay()
        for x, y in [(0, 0), (100, 0), (0, 100)]:
            edge.record_swap(ov, x, y)
            assert not ov.v2_deltas and not ov.v3_sqrt_price

    def test_two_sequential_swaps_compose(self):
        edge = _v2_weth_usdc_edge()
        ov = ReserveOverlay()
        x1 = 5 * 10**18
        y1 = edge.amount_out(x1)
        edge.record_swap(ov, x1, y1)
        advanced = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=edge.reserve_in + x1,
            reserve_out=edge.reserve_out - y1,
            fee_bps=30,
        )
        assert edge.amount_out(5 * 10**18, ov) == advanced.amount_out(5 * 10**18)


class TestV3PoolEdgeOverlay:
    def test_record_swap_advances_sqrt_price(self):
        edge = make_v3_edge(WETH, USDC, POOL_B, is_token0_in=True, fee_bps=5)
        ov = ReserveOverlay()
        x = 10**18
        edge.record_swap(ov, x, edge.amount_out(x))
        # token0_in: price decreases ⇒ sqrt_price drops.
        assert 0 < ov.v3_sqrt_price[POOL_B] < edge.sqrt_price_x96

    def test_overlay_decreases_subsequent_output(self):
        edge = make_v3_edge(WETH, USDC, POOL_B, is_token0_in=True, fee_bps=5)
        ov = ReserveOverlay()
        x = 10 * 10**18
        y1 = edge.amount_out(x)
        edge.record_swap(ov, x, y1)
        y2 = edge.amount_out(x, ov)
        assert 0 < y2 < y1

    def test_overlay_couples_bidirectional_v3(self):
        forward = make_v3_edge(WETH, USDC, POOL_B, is_token0_in=True, fee_bps=5)
        reverse = make_v3_edge(USDC, WETH, POOL_B, is_token0_in=False, fee_bps=5)
        ov = ReserveOverlay()
        x = 10**18
        forward.record_swap(ov, x, forward.amount_out(x))
        # Reverse edge sees the same overlayed sqrt_price (lower than baseline).
        assert reverse._effective_sqrt_price(ov) == ov.v3_sqrt_price[POOL_B] < reverse.sqrt_price_x96


# ---------------------------------------------------------------------------
# _distribute_int (largest-remainder apportionment)
# ---------------------------------------------------------------------------


class TestDistributeInt:
    @pytest.mark.parametrize(
        "weights,total",
        [
            ([1 / 3] * 3, 10**18),  # leftover would exceed n under naive float math
            ([1 / 7] * 7, 10**18),
            ([1 / 100] * 100, 10**18),
            ([0.5, 0.5, 0.0001], 10**18),  # weights sum > 1
            ([0.9999, 0.0001], 10**18),
            ([1.0, 0.0, 0.0], 10**18),
            ([0.5, 0.5], 10**30),  # past 2^53 float precision
            ([0.5, 0.5], 1),
            ([1 / 3] * 3, 10),
        ],
    )
    def test_conserves_total_exactly(self, weights, total):
        """Output must sum to ``total`` exactly across float-pathological inputs."""
        result = _distribute_int(total, weights)
        assert sum(result) == total
        assert all(x >= 0 for x in result)

    def test_zero_or_negative_total_returns_zeros(self):
        assert _distribute_int(0, [0.5, 0.5]) == [0, 0]
        assert _distribute_int(-1, [0.5, 0.5]) == [0, 0]

    def test_empty_weights_returns_empty(self):
        assert _distribute_int(100, []) == []

    def test_all_zero_weights_returns_zeros(self):
        assert _distribute_int(100, [0.0, 0.0]) == [0, 0]

    def test_proportional_allocation(self):
        """Allocations should respect the relative weight ordering."""
        result = _distribute_int(10**18, [0.7, 0.2, 0.1])
        assert result[0] > result[1] > result[2]
        assert sum(result) == 10**18


# ---------------------------------------------------------------------------
# MultiEdgePath / merge_and_expand (PRIME §V)
# ---------------------------------------------------------------------------


class TestMultiEdgePath:
    def test_empty_hops_rejected(self):
        with pytest.raises(ValueError, match="at least one hop"):
            MultiEdgePath(hops=())

    def test_mismatched_tokens_in_bundle_rejected(self):
        e1 = _v2_pool_edge(WETH, USDC, POOL_A)
        e2 = _v2_pool_edge(WETH, DAI, POOL_B)  # different token_out
        with pytest.raises(ValueError, match="share"):
            MultiEdgePath(hops=((e1, e2),))

    def test_chain_break_rejected(self):
        e1 = _v2_pool_edge(WETH, USDC, POOL_A)
        e2 = _v2_pool_edge(DAI, USDC, POOL_B)  # hop1 in (DAI) != hop0 out (USDC)
        with pytest.raises(ValueError, match="must match"):
            MultiEdgePath(hops=((e1,), (e2,)))

    def test_amount_out_single_edge_matches_pool_edge(self):
        edge = _v2_pool_edge(WETH, USDC, POOL_A)
        path = MultiEdgePath(hops=((edge,),))
        weights = PathWeights(path_weight=1.0, intra_hop_weights=[[1.0]])
        assert path.amount_out(10**18, weights) == edge.amount_out(10**18)

    def test_amount_out_two_edge_bundle_50_50_matches_split(self):
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**21, reserve_out=2 * 10**9)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**21, reserve_out=2 * 10**9)
        path = MultiEdgePath(hops=((e1, e2),))
        ws = PathWeights(path_weight=1.0, intra_hop_weights=[[0.5, 0.5]])
        x = 10**18
        expected = e1.amount_out(x // 2) + e2.amount_out(x - x // 2)
        assert path.amount_out(x, ws) == expected

    def test_amount_out_zero_input_returns_zero(self):
        edge = _v2_pool_edge(WETH, USDC, POOL_A)
        path = MultiEdgePath(hops=((edge,),))
        ws = PathWeights(path_weight=1.0, intra_hop_weights=[[1.0]])
        assert path.amount_out(0, ws) == 0

    def test_amount_out_tiny_input_through_50_50_bundle_conserves_input(self):
        """1 wei split 50/50 across a 2-edge bundle must not be zeroed out.

        Tiny reserves (~10^3) so single-edge amount_out > 0 — deep pools
        return 0 and mask the floor-allocation bug.
        """
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=1_000, reserve_out=2_000, fee_bps=0)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=1_000, reserve_out=2_000, fee_bps=0)
        single_path = MultiEdgePath(hops=((e1,),))
        single_ws = PathWeights(path_weight=1.0, intra_hop_weights=[[1.0]])
        bundle_path = MultiEdgePath(hops=((e1, e2),))
        bundle_ws = PathWeights(path_weight=1.0, intra_hop_weights=[[0.5, 0.5]])
        single_out = single_path.amount_out(1, single_ws)
        bundle_out = bundle_path.amount_out(1, bundle_ws)
        assert single_out > 0
        # Pre-fix this returned 0 — the 1 wei was silently dropped by the
        # ``int(1 * 0.5) = 0`` floor + guarded-leftover logic.
        assert bundle_out > 0


class TestMergeAndExpand:
    def _graph_two_pools_weth_usdc(self) -> PoolGraph:
        """WETH↔USDC via two V2 pools (different fee tiers / depths)."""
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
            WETH,
            USDC,
            POOL_B,
            "UniswapV2",
            reserve_a=500 * 10**18,
            reserve_b=1_000_000 * 10**6,
            fee_bps=5,
        )
        return g

    def test_collapses_same_token_sequence(self):
        """Two routes with the same token sequence merge into one MultiEdgePath."""
        g = self._graph_two_pools_weth_usdc()
        edge_a = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_A and e.token_out == USDC)
        edge_b = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_B and e.token_out == USDC)
        paths = merge_and_expand([[edge_a], [edge_b]], g)
        assert len(paths) == 1
        # Single hop with both pools in the bundle.
        assert len(paths[0].hops) == 1
        bundle_pools = {e.pool_address for e in paths[0].hops[0]}
        assert bundle_pools == {POOL_A, POOL_B}

    def test_widens_with_graph_parallel_edges(self):
        """A single-route input still picks up every parallel edge from the graph."""
        g = self._graph_two_pools_weth_usdc()
        edge_a = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_A and e.token_out == USDC)
        paths = merge_and_expand([[edge_a]], g)
        assert len(paths) == 1
        bundle_pools = {e.pool_address for e in paths[0].hops[0]}
        # POOL_A from the input + POOL_B added during expand.
        assert bundle_pools == {POOL_A, POOL_B}

    def test_caps_bundle_size(self):
        g = PoolGraph()
        for i, pool in enumerate([POOL_A, POOL_B, POOL_C, POOL_D]):
            g.add_bidirectional_pool(
                WETH,
                USDC,
                pool,
                "UniswapV2",
                reserve_a=10**21,
                reserve_b=2 * 10**9,
                fee_bps=30,
            )
        edge_a = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_A and e.token_out == USDC)
        paths = merge_and_expand([[edge_a]], g, max_parallel_per_hop=2)
        assert len(paths[0].hops[0]) == 2  # capped

    def test_preserves_distinct_token_sequences(self):
        """Routes with different token sequences stay separate."""
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=2 * 10**9,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            WETH,
            DAI,
            POOL_B,
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=10**21,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            DAI,
            USDC,
            POOL_C,
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=2 * 10**9,
            fee_bps=30,
        )
        e_direct = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_A and e.token_out == USDC)
        e_dai = next(e for e in g.edges_from(WETH) if e.token_out == DAI)
        e_dai_usdc = next(e for e in g.edges_from(DAI) if e.token_out == USDC)
        paths = merge_and_expand([[e_direct], [e_dai, e_dai_usdc]], g)
        assert len(paths) == 2
        # Distinguishable by hop count.
        hop_counts = sorted(len(p.hops) for p in paths)
        assert hop_counts == [1, 2]

    def test_enforces_pool_disjoint_across_paths(self):
        """A pool claimed by the earlier-ranked path is excluded from later paths (PRIME §III)."""
        g = PoolGraph()
        # Two routes both pass through WETH→USDC via POOL_A, then diverge.
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=10**22,
            reserve_b=2 * 10**10,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC,
            DAI,
            POOL_B,
            "UniswapV2",
            reserve_a=2 * 10**10,
            reserve_b=2 * 10**22,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC,
            WBTC,
            POOL_C,
            "UniswapV2",
            reserve_a=2 * 10**10,
            reserve_b=10**9,
            fee_bps=30,
        )
        e_weth_usdc = next(e for e in g.edges_from(WETH) if e.pool_address == POOL_A and e.token_out == USDC)
        e_usdc_dai = next(e for e in g.edges_from(USDC) if e.pool_address == POOL_B and e.token_out == DAI)
        e_usdc_wbtc = next(e for e in g.edges_from(USDC) if e.pool_address == POOL_C and e.token_out == WBTC)
        paths = merge_and_expand([[e_weth_usdc, e_usdc_dai], [e_weth_usdc, e_usdc_wbtc]], g)
        # Each path is a distinct token sequence, so both survive — but POOL_A
        # may appear in only one of them.
        appearances = [pool for p in paths for bundle in p.hops for e in bundle if (pool := e.pool_address) == POOL_A]
        assert len(appearances) == 1, "POOL_A must not appear in more than one MultiEdgePath"

    def test_v4_pools_sharing_pool_manager_address_not_collapsed(self):
        """Distinct V4 pools share the singleton PoolManager address but differ
        by fee/tickSpacing/hooks; they must NOT collapse into one bundle.
        """
        singleton = Address("0x" + "44" * 20)  # shared PoolManager address
        g = PoolGraph()
        e_lo = PoolEdge(WETH, USDC, singleton, "UniswapV4", reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=5)
        e_hi = PoolEdge(WETH, USDC, singleton, "UniswapV4", reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)
        g.add_pool(e_lo)
        g.add_pool(e_hi)
        paths = merge_and_expand([[e_lo], [e_hi]], g)
        assert len(paths) == 1
        # Both fee tiers must survive in the bundle despite the shared address.
        assert len(paths[0].hops[0]) == 2
        assert {e.fee_bps for e in paths[0].hops[0]} == {5, 30}


# ---------------------------------------------------------------------------
# ASGM solver (PRIME §V-C)
# ---------------------------------------------------------------------------


class TestASGM:
    def test_single_path_single_edge_returns_full_weight(self):
        edge = _v2_pool_edge(WETH, USDC, POOL_A)
        path = MultiEdgePath(hops=((edge,),))
        out = asgm_optimize([path], 10**18)
        assert len(out) == 1
        assert out[0].path_weight == 1.0
        assert out[0].intra_hop_weights == [[1.0]]

    def test_two_symmetric_paths_converge_to_50_50(self):
        """Two identical pools across two paths should split evenly at the optimum."""
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)
        p1 = MultiEdgePath(hops=((e1,),))
        p2 = MultiEdgePath(hops=((e2,),))
        ws = asgm_optimize([p1, p2], 10**18)
        assert len(ws) == 2
        # Symmetric pools ⇒ expect ~50/50 within a few percent (granularity).
        assert abs(ws[0].path_weight - 0.5) < 0.05
        assert abs(ws[1].path_weight - 0.5) < 0.05
        assert abs(ws[0].path_weight + ws[1].path_weight - 1.0) < 1e-6

    def test_two_asymmetric_paths_favours_deeper_pool(self):
        """A deeper pool should attract more allocation than a shallow one."""
        deep = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**22, reserve_out=2 * 10**10, fee_bps=30)
        shallow = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=5 * 10**20, reserve_out=10**9, fee_bps=30)
        p1 = MultiEdgePath(hops=((deep,),))
        p2 = MultiEdgePath(hops=((shallow,),))
        ws = asgm_optimize([p1, p2], 10 * 10**18)
        assert ws[0].path_weight > ws[1].path_weight, "deeper pool should get larger share"
        assert abs(ws[0].path_weight + ws[1].path_weight - 1.0) < 1e-6

    def test_two_paths_beat_single_path(self):
        """Splitting across two parallel paths must yield ≥ either path alone."""
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)
        p1 = MultiEdgePath(hops=((e1,),))
        p2 = MultiEdgePath(hops=((e2,),))
        x = 10**19
        ws = asgm_optimize([p1, p2], x)
        split_out = sum(p.amount_out(int(w.path_weight * x), w) for p, w in zip([p1, p2], ws))
        single_out = max(e1.amount_out(x), e2.amount_out(x))
        assert split_out >= single_out

    def test_optimized_never_worse_than_uniform_split(self):
        """ASGM's result must be >= the uniform allocation it starts from.

        Guards the Armijo line search: a rejected trial must fully restore the
        in-place-tuned intra_hop_weights, otherwise the optimizer can return an
        allocation worse than its own starting point.
        """
        # Asymmetric depths/fees so the line search actually backtracks.
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**22, reserve_out=2 * 10**10, fee_bps=30)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=5)
        e3 = _v2_pool_edge(WETH, USDC, POOL_C, reserve_in=3 * 10**20, reserve_out=6 * 10**8, fee_bps=100)
        paths = [MultiEdgePath(hops=((e,),)) for e in (e1, e2, e3)]
        x = 5 * 10**18
        n = len(paths)
        uniform = [PathWeights(path_weight=1.0 / n, intra_hop_weights=[[1.0]]) for _ in paths]
        uniform_inputs = _distribute_int(x, [1.0 / n] * n)
        uniform_out = sum(p.amount_out(a, w) for p, w, a in zip(paths, uniform, uniform_inputs) if a > 0)
        ws = asgm_optimize(paths, x)
        opt_inputs = _distribute_int(x, [w.path_weight for w in ws])
        opt_out = sum(p.amount_out(a, w) for p, w, a in zip(paths, ws, opt_inputs) if a > 0)
        assert opt_out >= uniform_out

    def test_amount_in_zero_raises(self):
        edge = _v2_pool_edge(WETH, USDC, POOL_A)
        path = MultiEdgePath(hops=((edge,),))
        with pytest.raises(ValueError, match="positive"):
            asgm_optimize([path], 0)

    def test_empty_paths_returns_empty(self):
        assert asgm_optimize([], 10**18) == []

    def test_tiny_input_across_paths_conserves_amount_in(self):
        """amount_in=2 with 3 paths must not silently allocate [0,0,0] under floor distribution."""
        e1 = _v2_pool_edge(WETH, USDC, POOL_A)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B)
        e3 = _v2_pool_edge(WETH, USDC, POOL_C)
        p1 = MultiEdgePath(hops=((e1,),))
        p2 = MultiEdgePath(hops=((e2,),))
        p3 = MultiEdgePath(hops=((e3,),))
        ws = asgm_optimize([p1, p2, p3], 2)
        # All path_weights still on the simplex.
        total_w = sum(w.path_weight for w in ws)
        assert abs(total_w - 1.0) < 1e-6
        # The 2 wei must be distributed, not dropped to [0, 0, 0] by rounding.
        # (Output stays 0 only because the pools are too deep to yield on 1 wei.)
        inputs = _distribute_int(2, [w.path_weight for w in ws])
        assert sum(inputs) == 2
        assert any(a > 0 for a in inputs)

    def test_intra_hop_weights_optimised_for_bundle(self):
        """A path with a multi-edge bundle should allocate non-trivially across edges."""
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**22, reserve_out=2 * 10**10, fee_bps=30)  # deep
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**21, reserve_out=2 * 10**9, fee_bps=30)  # shallow
        path = MultiEdgePath(hops=((e1, e2),))
        ws = asgm_optimize([path], 10 * 10**18)
        assert ws[0].path_weight == 1.0
        # Both pools should get some share; deep one should get more.
        w_e1, w_e2 = ws[0].intra_hop_weights[0]
        assert w_e1 > 0 and w_e2 > 0
        assert w_e1 > w_e2, "deep pool should get larger intra-hop share"
        assert abs(w_e1 + w_e2 - 1.0) < 1e-6

    def test_bundle_marginal_returns_per_edge_marginal_price(self):
        """Returns f_e'(w_e · c) — positive for every edge, even an over-allocated one."""
        e1 = _v2_pool_edge(WETH, USDC, POOL_A, reserve_in=10**22, reserve_out=2 * 10**10, fee_bps=30)
        e2 = _v2_pool_edge(WETH, USDC, POOL_B, reserve_in=10**22, reserve_out=2 * 10**10, fee_bps=30)
        bundle = (e1, e2)
        hop_input = 10**18
        g_saturated = _bundle_marginal(bundle, hop_input, [0.99, 0.01], idx=0)
        g_underused = _bundle_marginal(bundle, hop_input, [0.99, 0.01], idx=1)
        assert g_saturated > 0, "per-edge marginal of a monotone swap function is positive"
        assert g_underused > g_saturated, "less-allocated identical edge has higher marginal"
        # Cross-check: marginal at the equal-split point matches a direct FD on the edge.
        equal = _bundle_marginal(bundle, hop_input, [0.5, 0.5], idx=0)
        x0, x1 = int(0.5 * hop_input), int(0.5 * hop_input) + int(_MARGINAL_EPS * hop_input)
        expected = (e1.amount_out(x1) - e1.amount_out(x0)) / (x1 - x0)
        assert abs(equal - expected) / max(expected, 1e-18) < 1e-6


# ---------------------------------------------------------------------------
# Router.find_optimal_split (Slice B integration)
# ---------------------------------------------------------------------------


class TestFindOptimalSplit:
    def _two_pool_graph(self, *, deep: bool = True) -> PoolGraph:
        """WETH↔USDC via two parallel V2 pools (deep=True ⇒ similar depth)."""
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
            WETH,
            USDC,
            POOL_B,
            "UniswapV2",
            reserve_a=1_000 * 10**18 if deep else 50 * 10**18,
            reserve_b=2_000_000 * 10**6 if deep else 100_000 * 10**6,
            fee_bps=30,
        )
        return g

    def test_single_path_returns_linear_dag(self):
        """When only one route exists, find_optimal_split returns a linear DAG."""
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH,
            USDC,
            POOL_A,
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=2 * 10**9,
            fee_bps=30,
        )
        router = Router(g)
        dag = router.find_optimal_split(TokenAmount(WETH, 10**18), USDC)
        payload = dag.to_dict()
        # One swap action, no split.
        assert len(payload["actions"]) == 1
        assert isinstance(payload["actions"][0], RouteSwap)

    def test_two_parallel_pools_emit_inner_split(self):
        """Two parallel WETH/USDC pools should emit a split DAG with both pools."""
        router = Router(self._two_pool_graph(deep=True))
        dag = router.find_optimal_split(TokenAmount(WETH, 10**18), USDC)
        payload = dag.to_dict()
        # Single hop with multi-edge bundle ⇒ outer action is RouteSplit.
        assert isinstance(payload["actions"][0], RouteSplit)
        legs = payload["actions"][0].legs
        # Both pools represented across the split legs.
        leg_pools = {leg.actions[0].pool.pool_address for leg in legs}
        assert {POOL_A, POOL_B}.issubset(leg_pools)
        # Fractions must sum exactly to MAX_BPS.
        assert sum(leg.fraction_bps for leg in legs) == 10_000

    def test_invalid_candidates_raises(self):
        router = Router(self._two_pool_graph())
        with pytest.raises(ValueError, match="candidates must be >= 1"):
            router.find_optimal_split(TokenAmount(WETH, 10**18), USDC, candidates=0)

    def test_no_route_raises(self):
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
        with pytest.raises(NoRouteFoundError):
            router.find_optimal_split(TokenAmount(WETH, 10**18), DAI)

    def test_multi_hop_path(self):
        """Multi-hop graph: find_optimal_split routes WETH→USDC→DAI as a single path."""
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
            reserve_a=5_000_000 * 10**6,
            reserve_b=5_000_000 * 10**18,
            fee_bps=4,
        )
        router = Router(g)
        dag = router.find_optimal_split(TokenAmount(WETH, 10**18), DAI)
        out = router.simulate(dag, 10**18)
        # ~$2000 worth of DAI in raw units (6→18 dec): expect 1k–10k * 1e18.
        assert 1_000 * 10**18 < out < 10_000 * 10**18

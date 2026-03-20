"""Tests for pydefi.pathfinder — graph and router."""

from decimal import Decimal
import math

import pytest

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
from pydefi.pathfinder.router import Router
from pydefi.types import ChainId, Token, TokenAmount


# ---------------------------------------------------------------------------
# Test tokens
# ---------------------------------------------------------------------------

WETH = Token(chain_id=ChainId.ETHEREUM, address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", symbol="WETH", decimals=18)
USDC = Token(chain_id=ChainId.ETHEREUM, address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", symbol="USDC", decimals=6)
DAI = Token(chain_id=ChainId.ETHEREUM, address="0x6B175474E89094C44Da98b954EedeAC495271d0F", symbol="DAI", decimals=18)
WBTC = Token(chain_id=ChainId.ETHEREUM, address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", symbol="WBTC", decimals=8)

POOL_A = "0x" + "11" * 20
POOL_B = "0x" + "22" * 20
POOL_C = "0x" + "33" * 20
POOL_D = "0x" + "44" * 20


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
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
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
            reserve_out=2_000_000 * 10 ** 6,
        )
        assert edge.spot_price == Decimal(0)

    def test_amount_out_basic(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        out = edge.amount_out(10 ** 18)  # 1 WETH
        assert out > 1_990 * 10 ** 6  # > 1990 USDC
        assert out < 2_000 * 10 ** 6  # < 2000 USDC

    def test_amount_out_no_reserves(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
        )
        assert edge.amount_out(10 ** 18) == 0

    def test_log_weight_finite(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        weight = edge.log_weight(10 ** 18)
        assert weight < float("inf")
        assert weight != 0  # fee causes some loss

    def test_log_weight_no_liquidity(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
        )
        assert edge.log_weight(10 ** 18) == float("inf")

    def test_estimate_price_impact_nonzero(self):
        edge = PoolEdge(
            token_in=WETH,
            token_out=USDC,
            pool_address=POOL_A,
            protocol="UniswapV2",
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        impact = edge.estimate_price_impact(10 ** 18)
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
        assert edge.estimate_price_impact(10 ** 18).is_nan()


# ---------------------------------------------------------------------------
# V3PoolEdge tests
# ---------------------------------------------------------------------------

def _make_v3_edge(token_in, token_out, pool_address, is_token0_in: bool, price_usdc_per_weth: float = 2000.0, fee_bps: int = 30):
    """Build a V3PoolEdge with a synthetic sqrtPriceX96 at the given USDC/WETH price."""
    # P_raw = price_raw_token1_per_token0 (in smallest units)
    if is_token0_in:
        # token_in = WETH (token0), token_out = USDC (token1)
        P_raw = price_usdc_per_weth * (10 ** token_out.decimals) / (10 ** token_in.decimals)
    else:
        # token_in = USDC (token1), token_out = WETH (token0)
        # P_raw is still defined as token1/token0 for the underlying pool
        P_raw = price_usdc_per_weth * (10 ** token_in.decimals) / (10 ** token_out.decimals)
    sqrtP_real = math.sqrt(P_raw)
    Q96 = 2 ** 96
    sqrt_price_x96 = int(sqrtP_real * Q96)
    liquidity = 5 * 10 ** 22  # large pool
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
            token_in=WETH, token_out=USDC, pool_address=POOL_A,
            protocol="UniswapV3", sqrt_price_x96=0, liquidity=10 ** 22,
        )
        assert edge.spot_price == Decimal(0)

    def test_amount_out_token0_in(self):
        """Swapping 1 WETH should yield ~1994 USDC (after 0.3% fee at $2000)."""
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        out = edge.amount_out(10 ** 18)
        assert 1_990 * 10 ** 6 < out < 1_998 * 10 ** 6, f"Got {out / 10**6:.2f} USDC"

    def test_amount_out_token1_in(self):
        """Swapping 2000 USDC should yield ~0.997 WETH (after 0.3% fee)."""
        edge = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        out = edge.amount_out(2000 * 10 ** 6)
        assert int(0.994 * 10 ** 18) < out < int(0.998 * 10 ** 18), (
            f"Got {out / 10**18:.6f} WETH"
        )

    def test_amount_out_zero_liquidity(self):
        edge = V3PoolEdge(
            token_in=WETH, token_out=USDC, pool_address=POOL_A,
            protocol="UniswapV3", sqrt_price_x96=10 ** 20, liquidity=0,
        )
        assert edge.amount_out(10 ** 18) == 0

    def test_amount_out_zero_sqrt_price(self):
        edge = V3PoolEdge(
            token_in=WETH, token_out=USDC, pool_address=POOL_A,
            protocol="UniswapV3", sqrt_price_x96=0, liquidity=10 ** 22,
        )
        assert edge.amount_out(10 ** 18) == 0

    def test_log_weight_finite(self):
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        weight = edge.log_weight(10 ** 18)
        assert weight < float("inf")
        assert weight > 0  # fee causes loss

    def test_estimate_price_impact_token0_in(self):
        """V3 price impact should be a sensible positive fraction."""
        edge = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        impact = edge.estimate_price_impact(10 ** 18)  # 1 WETH
        assert not impact.is_nan()
        assert Decimal(0) < impact < Decimal(1)

    def test_estimate_price_impact_token1_in(self):
        """V3 price impact should be sensible for token1→token0 direction."""
        edge = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        impact = edge.estimate_price_impact(2000 * 10 ** 6)  # 2000 USDC
        assert not impact.is_nan()
        assert Decimal(0) < impact < Decimal(1)

    def test_estimate_price_impact_zero_liquidity(self):
        edge = V3PoolEdge(
            token_in=WETH, token_out=USDC, pool_address=POOL_A,
            protocol="UniswapV3", sqrt_price_x96=10 ** 20, liquidity=0,
        )
        assert edge.estimate_price_impact(10 ** 18).is_nan()

    def test_estimate_price_impact_zero_sqrt_price(self):
        edge = V3PoolEdge(
            token_in=WETH, token_out=USDC, pool_address=POOL_A,
            protocol="UniswapV3", sqrt_price_x96=0, liquidity=10 ** 22,
        )
        assert edge.estimate_price_impact(10 ** 18).is_nan()

    def test_v3_edge_in_router(self):
        """Router should find a route through a V3 pool edge."""
        g = PoolGraph()
        edge_in = _make_v3_edge(WETH, USDC, POOL_A, is_token0_in=True)
        edge_out = _make_v3_edge(USDC, WETH, POOL_A, is_token0_in=False)
        g.add_pool(edge_in)
        g.add_pool(edge_out)
        router = Router(g)
        route = router.find_best_route(TokenAmount(WETH, 10 ** 18), USDC)
        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 1_990 * 10 ** 6


# ---------------------------------------------------------------------------
# PoolGraph tests
# ---------------------------------------------------------------------------

class TestPoolGraph:
    def _make_graph(self) -> PoolGraph:
        """Build a small test graph: WETH ↔ USDC ↔ DAI, WETH ↔ WBTC."""
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH, USDC, POOL_A, "UniswapV2",
            reserve_a=1_000 * 10 ** 18,
            reserve_b=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC, DAI, POOL_B, "Curve",
            reserve_a=1_000_000 * 10 ** 6,
            reserve_b=1_000_000 * 10 ** 18,
            fee_bps=4,
        )
        g.add_bidirectional_pool(
            WETH, WBTC, POOL_C, "UniswapV3",
            reserve_a=500 * 10 ** 18,
            reserve_b=20 * 10 ** 8,
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
            reserve_in=10 ** 21,
            reserve_out=2 * 10 ** 9,
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
            WETH, USDC, POOL_A, "UniswapV2",
            reserve_a=1_000 * 10 ** 18,
            reserve_b=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        # USDC ↔ DAI  (stableswap)
        g.add_bidirectional_pool(
            USDC, DAI, POOL_B, "Curve",
            reserve_a=5_000_000 * 10 ** 6,
            reserve_b=5_000_000 * 10 ** 18,
            fee_bps=4,
        )
        # WETH ↔ DAI  (indirect route available via WETH→USDC→DAI and direct)
        g.add_bidirectional_pool(
            WETH, DAI, POOL_C, "UniswapV3",
            reserve_a=500 * 10 ** 18,
            reserve_b=1_000_000 * 10 ** 18,
            fee_bps=3,
        )
        return g

    def test_find_best_route_direct(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        route = router.find_best_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0

    def test_find_best_route_two_hop(self):
        g = PoolGraph()
        # Only path: WETH → USDC → DAI (no direct WETH→DAI)
        g.add_bidirectional_pool(
            WETH, USDC, POOL_A, "UniswapV2",
            reserve_a=1_000 * 10 ** 18,
            reserve_b=2_000_000 * 10 ** 6,
            fee_bps=30,
        )
        g.add_bidirectional_pool(
            USDC, DAI, POOL_B, "Curve",
            reserve_a=5_000_000 * 10 ** 6,
            reserve_b=5_000_000 * 10 ** 18,
            fee_bps=4,
        )
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        route = router.find_best_route(amount_in, DAI)

        assert route.token_in == WETH
        assert route.token_out == DAI
        assert len(route.steps) == 2

    def test_find_best_route_no_path_raises(self):
        g = PoolGraph()
        g.add_bidirectional_pool(
            WETH, USDC, POOL_A, "UniswapV2",
            reserve_a=10 ** 21,
            reserve_b=2 * 10 ** 9,
        )
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        with pytest.raises(NoRouteFoundError):
            router.find_best_route(amount_in, DAI)  # DAI not in graph

    def test_find_best_route_same_token_raises(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        with pytest.raises(ValueError):
            router.find_best_route(amount_in, WETH)

    def test_find_best_route_max_hops_respected(self):
        """Router should not find routes longer than max_hops."""
        g = PoolGraph()
        # Chain: WETH → USDC → DAI → WBTC (3 hops)
        g.add_pool(PoolEdge(WETH, USDC, POOL_A, "UniswapV2", 10 ** 21, 2 * 10 ** 9, 30))
        g.add_pool(PoolEdge(USDC, DAI, POOL_B, "Curve", 10 ** 9, 10 ** 21, 4))
        g.add_pool(PoolEdge(DAI, WBTC, POOL_C, "UniswapV3", 10 ** 21, 10 ** 8, 5))

        router_1hop = Router(g, max_hops=1)
        with pytest.raises(NoRouteFoundError):
            router_1hop.find_best_route(TokenAmount(WETH, 10 ** 18), WBTC)

        router_3hop = Router(g, max_hops=3)
        route = router_3hop.find_best_route(TokenAmount(WETH, 10 ** 18), WBTC)
        assert route.token_out == WBTC

    def test_find_all_routes_returns_sorted(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        routes = router.find_all_routes(amount_in, DAI)

        assert len(routes) >= 1
        # Routes should be sorted by descending output amount
        for i in range(len(routes) - 1):
            assert routes[i].amount_out.amount >= routes[i + 1].amount_out.amount

    def test_find_all_routes_top_k(self):
        g = self._make_graph()
        router = Router(g)
        amount_in = TokenAmount(token=WETH, amount=10 ** 18)
        routes = router.find_all_routes(amount_in, DAI, top_k=1)
        assert len(routes) == 1

    def test_find_all_routes_no_path_raises(self):
        g = PoolGraph()
        router = Router(g)
        with pytest.raises(NoRouteFoundError):
            router.find_all_routes(TokenAmount(WETH, 10 ** 18), USDC)

    def test_price_impact_zero_reserves(self):
        """Routes through zero-reserve pools (e.g. V3) should return NaN (unestimated)."""
        edges = [
            PoolEdge(WETH, USDC, POOL_A, "UniswapV2", reserve_in=0, reserve_out=0)
        ]
        impact = Router._estimate_price_impact(edges, 10 ** 18)
        assert impact.is_nan()

    def test_price_impact_nonzero(self):
        edges = [
            PoolEdge(
                WETH, USDC, POOL_A, "UniswapV2",
                reserve_in=1_000 * 10 ** 18,
                reserve_out=2_000_000 * 10 ** 6,
                fee_bps=30,
            )
        ]
        impact = Router._estimate_price_impact(edges, 10 ** 18)
        assert Decimal(0) < impact < Decimal(1)

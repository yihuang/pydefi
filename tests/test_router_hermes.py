"""Phase 4 tests: Router with candidate_solver='hermes' must (a) match the
hop_dp solver's output within ASGM tolerance on the canonical fixtures and
(b) find routes that the hop-cap=3 DP misses on the multi-cluster fixture."""

from __future__ import annotations

import random

import pytest

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.graph import PoolGraph
from pydefi.pathfinder.router import Router
from pydefi.types import Address, TokenAmount
from tests._fixtures import core_periphery_fixture, make_token, multi_cluster_fixture


def _random_pairs(tokens: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        a, b = rng.sample(tokens, 2)
        pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Equivalence on core-periphery (single-DEX shape)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_periphery,n_pairs,seed",
    [
        (20, 30, 0),
        (50, 30, 1),
        (100, 20, 2),
    ],
)
def test_hermes_output_within_tolerance_of_hop_dp(n_periphery, n_pairs, seed):
    """hermes solver must stay within 15% of hop_dp on the canonical fixture.

    Guards against catastrophic regression, not equivalence — spot-rate
    ranking diverges from amount_out ranking at cross-decimal pairs above ~1× depth.
    """
    fix = core_periphery_fixture(n_periphery=n_periphery, seed=seed)
    pool_g = fix.pool_graph
    tokens = list(fix.tokens.values())
    rng = random.Random(seed)

    r_dp = Router(pool_g, max_hops=3, candidate_solver="hop_dp")
    r_he = Router(pool_g, candidate_solver="hermes")

    for src, dst in _random_pairs(tokens, n_pairs, seed):
        amt_in = TokenAmount(src, 10**src.decimals * rng.choice([1, 10]))
        try:
            dag_dp = r_dp.find_optimal_split(amt_in, dst)
            out_dp = r_dp.simulate(dag_dp, amt_in.amount)
        except (NoRouteFoundError, ValueError):
            out_dp = None
        try:
            dag_he = r_he.find_optimal_split(amt_in, dst)
            out_he = r_he.simulate(dag_he, amt_in.amount)
        except (NoRouteFoundError, ValueError):
            out_he = None

        if out_dp is None:
            continue  # hermes-only success is a strict gain
        if out_he is None:
            pytest.fail(f"hermes lost a route hop_dp had: {src.symbol}→{dst.symbol}")
        tolerance = max(10, (out_dp * 15) // 100)
        assert out_he + tolerance >= out_dp, (
            f"hermes regressed: {src.symbol}→{dst.symbol} amt={amt_in.amount}: "
            f"hop_dp={out_dp}, hermes={out_he}, gap={out_dp - out_he}"
        )


# ---------------------------------------------------------------------------
# Multi-hop reachability — Hermes finds routes hop_cap=3 misses
# ---------------------------------------------------------------------------


def test_hermes_routes_through_6_hop_bridge_when_hop_dp_fails():
    """Multi-cluster: alt_a → alt_b is a 6-hop route; hop_cap=3 DP fails.

    Both solvers honor ``max_hops`` (see _find_top_routes_hermes); the test
    raises hermes' cap to 6 so the long-tail bridge is reachable.
    """
    fix = multi_cluster_fixture(
        n_clusters=2,
        core_size_per_cluster=2,
        n_periphery_per_cluster=5,
        seed=0,
    )
    pool_g = fix.pool_graph
    src = fix.tokens["C0_ALT000"]
    dst = fix.tokens["C1_ALT000"]
    amt_in = TokenAmount(src, 10**18)

    r_dp = Router(pool_g, max_hops=3, candidate_solver="hop_dp")
    with pytest.raises(NoRouteFoundError):
        r_dp.find_optimal_split(amt_in, dst)

    r_he = Router(pool_g, max_hops=6, candidate_solver="hermes")
    dag = r_he.find_optimal_split(amt_in, dst)
    out = r_he.simulate(dag, amt_in.amount)
    assert out > 0


def test_hermes_default_remains_hop_dp():
    """Backwards compat: Router(graph) without kwarg uses hop_dp."""
    fix = core_periphery_fixture(n_periphery=10, seed=0)
    r = Router(fix.pool_graph)
    assert r.candidate_solver == "hop_dp"


# ---------------------------------------------------------------------------
# First-hop diversity contract
# ---------------------------------------------------------------------------


def test_hermes_candidates_deduplicated_by_first_hop_pool():
    """Multiple Yen paths sharing the same first edge must collapse to one in the candidate list."""
    weth, t1, t2, t3, dst = (make_token(s, 18, i) for i, s in enumerate(["WETH", "T1", "T2", "T3", "DST"], 1))
    g = PoolGraph()
    # Single WETH→T1 edge; three distinct downstream paths to dst all start through it.
    edges = [(weth, t1), (t1, dst), (t1, t2), (t2, dst), (t1, t3), (t3, dst)]
    for i, (a, b) in enumerate(edges, 1):
        g.add_bidirectional_pool(a, b, Address("0x" + format(i, "040x")), "UniswapV2", 10**22, 10**22)

    router = Router(g, max_hops=4, candidate_solver="hermes")
    # Precondition: several simple candidates all start through the one WETH->T1
    # pool, so the dedup has work to do (else the assertion below is vacuous).
    hermes = router._ensure_hermes()
    raw = [p for p in hermes.top_k_paths(weth.address, dst.address, 10) if len(set(p)) == len(p) and len(p) - 1 <= 4]
    assert sum(1 for p in raw if len(p) > 1 and p[1] == t1.address) >= 2

    routes = router._find_top_routes_hermes(TokenAmount(weth, 10**18), dst, top_n=5)
    first_pools = [r.steps[0].pool_address for r in routes]
    assert len(first_pools) == len(set(first_pools)), f"duplicate first-hop pools: {first_pools}"
    # The several same-first-hop candidates collapsed to exactly one route.
    assert len(routes) == 1


# ---------------------------------------------------------------------------
# weight_mode plumbing — Hermes can rank by amount_out instead of spot
# ---------------------------------------------------------------------------


def test_amount_out_mode_requires_positive_probe():
    fix = core_periphery_fixture(n_periphery=5, seed=0)
    with pytest.raises(ValueError, match="probe_amount"):
        Router(fix.pool_graph, candidate_solver="hermes", weight_mode="amount_out")


@pytest.mark.parametrize(
    "router_kwargs,expected",
    [
        ({}, {"weight": "spot", "probe_amount": 0}),
        (
            {"weight_mode": "amount_out", "probe_amount": 10**18},
            {"weight": "amount_out", "probe_amount": 10**18},
        ),
    ],
    ids=["spot_default", "amount_out_with_probe"],
)
def test_router_propagates_weight_mode_kwargs_to_from_pool_graph(monkeypatch, router_kwargs, expected):
    """Spy-asserts kwargs reach from_pool_graph; end-output tests can hide a plumbing break."""
    from pydefi.pathfinder import router as router_mod

    fix = core_periphery_fixture(n_periphery=5, seed=0)
    captured: dict = {}
    real = router_mod.from_pool_graph

    # Capture all kwargs so a future signature addition surfaces here instead
    # of being silently dropped.
    def _spy(graph, **kwargs):
        captured.update(kwargs)
        return real(graph, **kwargs)

    monkeypatch.setattr(router_mod, "from_pool_graph", _spy)
    Router(fix.pool_graph, candidate_solver="hermes", **router_kwargs)._ensure_hermes()
    assert captured == expected

"""Tests for pydefi.pathfinder.hermes — treewidth-based SSSP."""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from pydefi.pathfinder.graph import PoolEdge, PoolGraph
from pydefi.pathfinder.hermes import HermesRouter, from_pool_graph
from pydefi.types import Address
from tests._fixtures import make_token, make_v3_edge


def _figure1_graph() -> nx.DiGraph:
    """Hermes paper Figure 1: 7 vertices with symmetric weights."""
    g = nx.DiGraph()
    edges = [
        (1, 2, 5),
        (2, 3, 1),
        (2, 6, 12),
        (3, 4, 2),
        (4, 5, 7),
        (4, 7, 3),
        (6, 7, 4),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
        g.add_edge(v, u, weight=w)
    return g


class TestPaperFigure1:
    """Reproduce the paper's worked example exactly."""

    def test_treewidth_is_2(self):
        router = HermesRouter.build(_figure1_graph())
        assert router.treewidth == 2

    def test_query_from_v4_matches_paper(self):
        """Paper Phase 4 example: source v_4 → distances [3, 7, 0, 3, 2, 8, 7]
        in PEO order. We check the actual distances per vertex regardless of
        whichever PEO our heuristic picks.
        """
        router = HermesRouter.build(_figure1_graph())
        d = router.query(4)
        # By Bellman-Ford ground truth on the same edges:
        #   d(4,1) = 4→3→2→1 = 2+1+5 = 8
        #   d(4,2) = 4→3→2 = 3
        #   d(4,3) = 2
        #   d(4,4) = 0
        #   d(4,5) = 7
        #   d(4,6) = 4→7→6 = 3+4 = 7
        #   d(4,7) = 3
        expected = {1: 8, 2: 3, 3: 2, 4: 0, 5: 7, 6: 7, 7: 3}
        assert d == expected


# ---------------------------------------------------------------------------
# Property: Hermes ≡ Bellman-Ford on graphs without negative cycles
# ---------------------------------------------------------------------------


def _random_dag_like(n: int, edge_prob: float, seed: int) -> nx.DiGraph:
    """Random directed graph with positive weights — no negative cycles."""
    rng = random.Random(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < edge_prob:
                g.add_edge(u, v, weight=rng.uniform(0.1, 5.0))
    # Force connectivity by adding a Hamiltonian cycle backbone.
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        a, b = perm[i], perm[(i + 1) % n]
        if not g.has_edge(a, b):
            g.add_edge(a, b, weight=rng.uniform(0.1, 5.0))
    return g


def _bellman_ford(g: nx.DiGraph, src) -> dict:
    return dict(nx.single_source_bellman_ford_path_length(g, src, weight="weight"))


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_hermes_matches_bellman_ford_random(seed):
    """On positive-weight graphs Hermes' distances must equal Bellman-Ford."""
    n = 25
    g = _random_dag_like(n, edge_prob=0.15, seed=seed)
    router = HermesRouter.build(g)
    src = 0
    expected = _bellman_ford(g, src)
    got = router.query(src)
    # Ground-truth keys may be a subset (unreachable vertices); Hermes likewise.
    common = set(expected) & set(got)
    for v in common:
        assert math.isclose(got[v], expected[v], rel_tol=1e-9, abs_tol=1e-9), (
            f"src=0 → {v}: expected {expected[v]}, got {got[v]}"
        )
    # Reachable sets must agree.
    assert set(got) == set(expected), f"reachable sets diverged: hermes={set(got)}, bf={set(expected)}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph():
    router = HermesRouter.build(nx.DiGraph())
    assert router.treewidth == 0
    assert router.peo == []


def test_single_vertex():
    g = nx.DiGraph()
    g.add_node("a")
    router = HermesRouter.build(g)
    assert router.query("a") == {"a": 0.0}


def test_disconnected_components_query():
    """Query from one component — vertices in the other are unreachable."""
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "a", weight=1.0)
    g.add_edge("c", "d", weight=2.0)
    g.add_edge("d", "c", weight=2.0)
    router = HermesRouter.build(g)
    d = router.query("a")
    assert d == {"a": 0.0, "b": 1.0}


def test_query_unknown_source_raises():
    g = nx.DiGraph()
    g.add_edge(1, 2, weight=1.0)
    router = HermesRouter.build(g)
    with pytest.raises(ValueError, match="not in graph"):
        router.query(99)


# ---------------------------------------------------------------------------
# Incremental weight updates (Phase 3 re-run, phases 1-2 reused)
# ---------------------------------------------------------------------------


def _diamond_graph() -> nx.DiGraph:
    """1 → {2, 3} → 4 with edge weights chosen so 1→2→4 wins by 1.0."""
    g = nx.DiGraph()
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(1, 3, weight=10.0)
    g.add_edge(2, 4, weight=1.0)
    g.add_edge(3, 4, weight=1.0)
    return g


def test_update_weight_recomputes_distance():
    """After bumping the cheap edge, query must reflect the new distances."""
    router = HermesRouter.build(_diamond_graph())
    assert router.query(1)[4] == 2.0  # 1→2→4 = 1+1
    # Make the cheap edge expensive; the long path 1→3→4 becomes optimal.
    router.update_weight(1, 2, 100.0)
    assert router.query(1)[4] == 11.0  # 1→3→4 = 10+1


def test_update_weight_unknown_edge_raises():
    router = HermesRouter.build(_diamond_graph())
    with pytest.raises(ValueError, match="not in graph"):
        router.update_weight(99, 100, 1.0)


def test_update_weights_batch_matches_full_rebuild():
    """Batched updates must produce the same dist as building from scratch."""
    g = _diamond_graph()
    router = HermesRouter.build(g)

    new_weights = {(1, 2): 5.0, (3, 4): 0.5, (2, 4): 7.0}
    router.update_weights(new_weights)

    # Ground truth: rebuild from a fresh graph with the same updates applied.
    fresh = _diamond_graph()
    for (u, v), w in new_weights.items():
        fresh[u][v]["weight"] = w
    expected = HermesRouter.build(fresh).query(1)
    got = router.query(1)
    assert got == expected


def test_update_weights_silently_ignores_unknown():
    """Indexer events for pools we haven't registered must not crash."""
    router = HermesRouter.build(_diamond_graph())
    before = router.query(1)
    router.update_weights({(99, 100): 0.0, (1, 2): 1.0})  # mix of unknown + known
    # Known weight unchanged, dist still correct.
    assert router.query(1) == before


# ---------------------------------------------------------------------------
# from_pool_graph adapter (Phase 1 of integration plan)
# ---------------------------------------------------------------------------


def _two_pool_v2_graph():
    """WETH↔USDC via two parallel V2 pools (different fees → different rates).

    Pool A is 30 bps; Pool B is 5 bps with same depth, so B wins on post-fee rate.
    """
    weth = make_token("WETH", 18, 0x11)
    usdc = make_token("USDC", 6, 0x22)
    pa = Address("0x" + "aa" * 20)
    pb = Address("0x" + "bb" * 20)
    g = PoolGraph()
    for addr, fee in [(pa, 30), (pb, 5)]:
        g.add_bidirectional_pool(
            weth,
            usdc,
            addr,
            "UniswapV2",
            reserve_a=1_000 * 10**18,
            reserve_b=2_000_000 * 10**6,
            fee_bps=fee,
        )
    return g, weth, usdc, pa, pb


class TestFromPoolGraph:
    def test_collapses_multi_edges_to_best_rate(self):
        pool_g, weth, usdc, _pa, pb = _two_pool_v2_graph()
        nx_g = from_pool_graph(pool_g, weight="spot")
        # One edge per direction (not two — multi-edges collapsed).
        assert nx_g.number_of_edges() == 2
        # The chosen edge for WETH→USDC should be the 5-bps pool (lower fee).
        edge_data = nx_g.get_edge_data(weth.address, usdc.address)
        assert edge_data["edge"].pool_address == pb
        assert edge_data["edge"].fee_bps == 5

    def test_amount_out_mode_quotes_via_probe(self):
        pool_g, weth, usdc, _pa, _pb = _two_pool_v2_graph()
        nx_g = from_pool_graph(pool_g, weight="amount_out", probe_amount=10**18)
        # 1 WETH probe — edge weight = -log(amount_out/probe) ≈ -log(1994).
        edge_data = nx_g.get_edge_data(weth.address, usdc.address)
        # USDC is 6-dec, WETH is 18-dec; rate is per-wei so the weight
        # encodes the cross-decimal ratio directly.
        assert math.isfinite(edge_data["weight"])

    def test_amount_out_mode_requires_positive_probe(self):
        pool_g, *_ = _two_pool_v2_graph()
        with pytest.raises(ValueError, match="probe_amount"):
            from_pool_graph(pool_g, weight="amount_out", probe_amount=0)

    def test_directed_round_trip(self):
        """Both directions present and weighted (asymmetric across decimals)."""
        pool_g, weth, usdc, _pa, _pb = _two_pool_v2_graph()
        nx_g = from_pool_graph(pool_g, weight="spot")
        we_uc = nx_g.get_edge_data(weth.address, usdc.address)
        uc_we = nx_g.get_edge_data(usdc.address, weth.address)
        assert we_uc is not None and uc_we is not None
        # Asymmetric — they're inverses (modulo fee), so weights have opposite signs.
        assert we_uc["weight"] != uc_we["weight"]

    def test_v3_edges_preserved_in_spot_mode(self):
        """V3 spot weight: closed-form (sqrtP/Q96)², not an amount_out probe."""
        weth = make_token("WETH", 18, 0x11)
        usdc = make_token("USDC", 6, 0x22)
        pool_addr = Address("0x" + "ff" * 20)
        pool_g = PoolGraph()
        for ti, to, is0 in [(weth, usdc, True), (usdc, weth, False)]:
            pool_g.add_pool(make_v3_edge(ti, to, pool_addr, is_token0_in=is0, fee_bps=5, liquidity=10**22))

        nx_g = from_pool_graph(pool_g, weight="spot")
        assert nx_g.number_of_nodes() == 2
        # Both directions must be present.
        assert nx_g.number_of_edges() == 2, (
            f"V3 edge dropped: nodes={nx_g.number_of_nodes()}, edges={nx_g.number_of_edges()}"
        )
        # Per-token rate ≈ 1999 USDC/WETH after 5bps fee, so weight is
        # -log(1999) ≈ -7.6. Decimal-normalised so cycles around stable
        # rates close cleanly (vs per-wei magnitudes ±20+ that would force
        # spurious DPC shortcuts the original graph can't reproduce).
        we_uc_w = nx_g.get_edge_data(weth.address, usdc.address)["weight"]
        assert -8 < we_uc_w < -7, f"unexpected weight {we_uc_w}"

    def test_one_bad_edge_does_not_disable_graph(self):
        """A pool whose ``amount_out`` raises must be skipped, not propagate.

        Regression for the Sepolia case: some V3 pools' sqrtPrice math
        overflows on standard probe sizes; before the exception-safe wrap,
        a single such pool aborted the whole ``from_pool_graph`` call,
        leaving ``Router(candidate_solver='hermes')`` permanently broken on
        any graph containing that pool.
        """

        weth = make_token("WETH", 18, 0x11)
        usdc = make_token("USDC", 6, 0x22)
        dai = make_token("DAI", 18, 0x33)

        class _BadEdge(PoolEdge):
            def amount_out(self, amount_in, overlay=None):  # type: ignore[override]
                raise OverflowError("simulated math overflow")

        pool_g = PoolGraph()
        pool_g.add_pool(
            PoolEdge(
                token_in=weth,
                token_out=usdc,
                pool_address=Address("0x" + "aa" * 20),
                protocol="UniswapV2",
                reserve_in=10**21,
                reserve_out=2 * 10**9,
                fee_bps=30,
                extra={"is_token0_in": True},
            )
        )
        # Bad edge with no reserves — falls through to probe path that raises.
        pool_g.add_pool(
            _BadEdge(
                token_in=usdc,
                token_out=dai,
                pool_address=Address("0x" + "bb" * 20),
                protocol="bad",
                fee_bps=30,
                extra={"is_token0_in": True},
            )
        )

        nx_g = from_pool_graph(pool_g, weight="spot")
        # Healthy edge survives; bad edge skipped (returned +inf weight).
        assert nx_g.has_edge(weth.address, usdc.address)
        assert not nx_g.has_edge(usdc.address, dai.address)


# ---------------------------------------------------------------------------
# shortest_path + top_k_paths (Phase 2 of integration plan)
# ---------------------------------------------------------------------------


class TestShortestPathReconstruction:
    def test_recovers_optimal_node_sequence(self):
        # Diamond: 1 → {2, 3} → 4. Edge weights make 1 → 2 → 4 strictly cheaper.
        g = nx.DiGraph()
        g.add_edge(1, 2, weight=1.0)
        g.add_edge(1, 3, weight=10.0)
        g.add_edge(2, 4, weight=1.0)
        g.add_edge(3, 4, weight=1.0)
        router = HermesRouter.build(g)
        assert router.shortest_path(1, 4) == [1, 2, 4]

    def test_unreachable_returns_none(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, weight=1.0)
        g.add_node(99)
        router = HermesRouter.build(g)
        assert router.shortest_path(1, 99) is None

    def test_self_path(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, weight=1.0)
        router = HermesRouter.build(g)
        assert router.shortest_path(1, 1) == [1]

    def test_uses_parent_pointers_not_bf_fallback(self, monkeypatch):
        """On a clean graph, shortest_path must NOT delegate to nx.bellman_ford_path."""
        g = nx.DiGraph()
        g.add_edge(1, 2, weight=1.0)
        g.add_edge(2, 3, weight=1.0)
        g.add_edge(1, 3, weight=10.0)
        router = HermesRouter.build(g)

        called = {"bf": 0}

        def _spy(*_a, **_kw):
            called["bf"] += 1
            raise AssertionError("BF fallback was triggered on a clean graph")

        monkeypatch.setattr(nx, "bellman_ford_path", _spy)
        assert router.shortest_path(1, 3) == [1, 2, 3]
        assert called["bf"] == 0

    def test_chordal_shortcut_expansion(self):
        """Reconstructed path must use only original edges, not fill-ins."""
        g = nx.DiGraph()
        # 4-cycle structure forces tree-decomposition to introduce a fill-in.
        for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "d"), ("d", "b")]:
            g.add_edge(u, v, weight=1.0)
        path = HermesRouter.build(g).shortest_path("a", "c")
        assert path is not None
        for u, v in zip(path[:-1], path[1:]):
            assert g.has_edge(u, v), f"non-original edge in path: ({u}, {v})"

    def test_cyclic_witness_chain_returns_false_not_recursion_error(self):
        """Cyclic witness chain must return False, not blow the stack."""
        from pydefi.pathfinder.hermes import _expand_chordal_edge

        g = nx.DiGraph()
        g.add_edge("a", "b", weight=1.0)
        g.add_edge("b", "c", weight=1.0)
        g.add_edge("a", "c", weight=10.0)
        # Every shortcut descends into another shortcut; nothing bottoms out.
        witness = {"a": {"c": "b", "b": "c"}, "b": {"c": "a"}}
        assert _expand_chordal_edge("a", "c", g, witness, []) is False


def _three_token_two_path_graph():
    """WETH → USDC direct, plus WETH → DAI → USDC: lets top_k find diverse first-hops."""
    weth = make_token("WETH", 18, 0x11)
    usdc = make_token("USDC", 6, 0x22)
    dai = make_token("DAI", 18, 0x33)
    pa = Address("0x" + "aa" * 20)
    pb = Address("0x" + "bb" * 20)
    pc = Address("0x" + "cc" * 20)
    g = PoolGraph()
    g.add_bidirectional_pool(
        weth, usdc, pa, "UniswapV2", reserve_a=1_000 * 10**18, reserve_b=2_000_000 * 10**6, fee_bps=30
    )
    g.add_bidirectional_pool(
        weth, dai, pb, "UniswapV2", reserve_a=1_000 * 10**18, reserve_b=2_000_000 * 10**18, fee_bps=30
    )
    g.add_bidirectional_pool(
        dai, usdc, pc, "UniswapV2", reserve_a=5_000_000 * 10**18, reserve_b=5_000_000 * 10**6, fee_bps=30
    )
    return g, weth, usdc, dai, pa, pb, pc


class TestTopKPaths:
    def test_returns_diverse_token_sequences(self):
        pool_g, weth, usdc, dai, _pa, _pb, _pc = _three_token_two_path_graph()
        nx_g = from_pool_graph(pool_g, weight="spot")
        router = HermesRouter.build(nx_g)
        paths = router.top_k_paths(weth.address, usdc.address, k=2)
        # Top-2 should include: direct (length 2) and via DAI (length 3).
        assert len(paths) == 2
        path_lengths = sorted(len(p) for p in paths)
        assert path_lengths == [2, 3], f"expected one 2-hop and one 3-hop, got {path_lengths}"

    def test_k_bounded_by_available(self):
        """When only one pool exists, top_k returns just the one path."""
        weth = make_token("WETH", 18, 0x11)
        usdc = make_token("USDC", 6, 0x22)
        g = PoolGraph()
        g.add_bidirectional_pool(
            weth,
            usdc,
            Address("0x" + "ff" * 20),
            "UniswapV2",
            reserve_a=10**21,
            reserve_b=2 * 10**9,
            fee_bps=30,
        )
        nx_g = from_pool_graph(g)
        router = HermesRouter.build(nx_g)
        paths = router.top_k_paths(weth.address, usdc.address, k=4)
        assert len(paths) == 1

    def test_k_zero_returns_empty(self):
        pool_g, weth, usdc, *_ = _two_pool_v2_graph()
        nx_g = from_pool_graph(pool_g)
        router = HermesRouter.build(nx_g)
        assert router.top_k_paths(weth.address, usdc.address, k=0) == []

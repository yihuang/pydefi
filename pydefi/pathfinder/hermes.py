"""Hermes structure-aware SSSP via tree decomposition.

Implements https://hal.science/hal-05320618v1

Four-phase pipeline:

  1. Tree decomposition (NetworkX heuristic).
  2. Chordal completion + PEO via leaf-to-root bag walk.
  3. Enforce Directed Path Consistency (DPC) on the chordal completion.
  4. Per-source SSSP via bitonic two-pass over the PEO.

:func:`from_pool_graph` adapts a :class:`PoolGraph` into the
``-log(per-token rate)`` NetworkX view that :class:`HermesRouter` consumes;
the router exposes :meth:`HermesRouter.top_k_paths` for candidate discovery
without a hop cap.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in

from pydefi.pathfinder.graph import PoolEdge, PoolGraph

V = Hashable
WeightMode = Literal["spot", "amount_out"]


def from_pool_graph(
    graph: PoolGraph,
    *,
    weight: WeightMode = "spot",
    probe_amount: int = 0,
) -> nx.DiGraph:
    """Build the NetworkX view of *graph* with ``-log(rate)`` edge weights.

    Multi-edge pairs (two pools for the same directed token pair) collapse to
    the cheapest weight = highest post-fee rate.

    Args:
        graph: source :class:`PoolGraph`.
        weight: ``"spot"`` uses the post-fee marginal rate; ``"amount_out"``
            quotes ``edge.amount_out(probe_amount)`` and bakes finite-input
            slippage into the seed weight (better discovery for known-typical
            trade sizes; cost: one ``amount_out`` call per directed edge).
        probe_amount: input size for ``"amount_out"`` mode. Ignored for
            ``"spot"``. Must be > 0 if ``weight == "amount_out"``.

    Returns:
        Directed NetworkX graph keyed by ``Token.address`` (HexBytes-hashable).
        Edge data carries ``weight`` (float, may be +inf for unroutable edges)
        and ``edge`` (the chosen :class:`PoolEdge` for path reconstruction).
    """
    if weight == "amount_out" and probe_amount <= 0:
        raise ValueError("probe_amount must be > 0 when weight='amount_out'")
    g = nx.DiGraph()
    for edge in graph:
        u = edge.token_in.address
        v = edge.token_out.address
        g.add_node(u)
        g.add_node(v)
        w = _edge_weight(edge, weight, probe_amount)
        if not math.isfinite(w):
            continue
        existing = g.get_edge_data(u, v)
        if existing is None or w < existing["weight"]:
            g.add_edge(u, v, weight=w, edge=edge)
    return g


_Q96 = 2**96


def _edge_weight(edge: PoolEdge, mode: WeightMode, probe_amount: int) -> float:
    """Translate one :class:`PoolEdge` to its ``-log(rate)`` weight.

    Returns ``+inf`` whenever a finite positive rate can't be derived — never
    raises. A single pathological pool (e.g. V3 sqrtPrice math overflow on
    extreme reserves) must not disable the whole Hermes graph build.
    """
    try:
        if mode == "spot":
            return _spot_weight(edge)
        # mode == "amount_out"
        return _amount_out_log_weight(edge, probe_amount)
    except (OverflowError, ZeroDivisionError, ValueError):
        return math.inf


def _spot_weight(edge: PoolEdge) -> float:
    """``-log(per-token rate)`` derived analytically — no ``amount_out`` calls.

    Per-token (not per-wei): multiply by ``10**(d_in - d_out)`` so cycles
    around stable rates sum to ≈0 and DPC enforces no spurious shortcuts
    from decimal-mismatch -log magnitudes.
    """
    fee_factor = (10_000 - edge.fee_bps) / 10_000
    decimal_factor = 10 ** (edge.token_in.decimals - edge.token_out.decimals)

    rin = getattr(edge, "reserve_in", 0) or 0
    rout = getattr(edge, "reserve_out", 0) or 0
    if rin > 0 and rout > 0:
        rate = (rout / rin) * fee_factor * decimal_factor
        return -math.log(rate) if rate > 0 else math.inf

    # V3-style edge: per-wei rate from sqrtPriceX96, normalised to per-token.
    sqrtP = getattr(edge, "sqrt_price_x96", 0) or 0
    if sqrtP > 0:
        ratio = sqrtP / _Q96
        price_t1_per_t0 = ratio * ratio  # per-wei
        is_t0_in = bool(getattr(edge, "is_token0_in", True))
        per_wei_rate = price_t1_per_t0 if is_t0_in else (1.0 / price_t1_per_t0 if price_t1_per_t0 > 0 else 0.0)
        rate = per_wei_rate * fee_factor * decimal_factor
        return -math.log(rate) if rate > 0 else math.inf

    # Last resort: probe with one whole input token.
    return _amount_out_log_weight(edge, 10**edge.token_in.decimals)


def _amount_out_log_weight(edge: PoolEdge, probe: int) -> float:
    """``-log(per-token rate)`` derived from a finite-input ``amount_out`` probe."""
    out = edge.amount_out(probe)
    if out <= 0 or probe <= 0:
        return math.inf
    # (out / probe) is per-wei; multiply by 10**(d_in - d_out) for per-token.
    rate = (out / probe) * 10 ** (edge.token_in.decimals - edge.token_out.decimals)
    return -math.log(rate) if rate > 0 else math.inf


@dataclass
class HermesRouter:
    """Pre-computed Hermes routing structure for a fixed graph topology.

    Construction runs phases 1–3 (offline preprocessing). Each :meth:`query`
    call runs phase 4 (the bitonic two-pass) for one source vertex.
    """

    graph: nx.DiGraph
    treewidth: int
    bags: list[frozenset[V]]
    peo: list[V]  # vertex order, π[0] = v_1 (eliminated first)
    peo_index: dict[V, int]  # v -> position in π
    chordal_neighbors: dict[V, set[V]]  # undirected adjacency in Ĝ
    dpc_weights: dict[V, dict[V, float]]  # d*[u][v]

    @classmethod
    def build(cls, graph: nx.DiGraph) -> HermesRouter:
        """Run phases 1–3 against *graph*.

        *graph* must be a directed graph whose edge data carries a numeric
        ``weight`` attribute (default 1 if missing). Multi-edges are not
        supported — collapse them upstream by picking the min weight per
        directed pair.
        """
        if graph.number_of_nodes() == 0:
            return cls(graph, 0, [], [], {}, {}, {})
        undirected = graph.to_undirected(as_view=False)
        # NetworkX's tree-decomp solver doesn't tolerate self-loops.
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        tw, tree = treewidth_min_fill_in(undirected)
        bags = [frozenset(b) for b in tree.nodes()]
        chordal_neighbors, peo = _chordal_completion_and_peo(tree)
        peo_index = {v: i for i, v in enumerate(peo)}
        dpc = _enforce_dpc(graph, chordal_neighbors, peo, peo_index)
        return cls(
            graph=graph,
            treewidth=tw,
            bags=bags,
            peo=peo,
            peo_index=peo_index,
            chordal_neighbors=chordal_neighbors,
            dpc_weights=dpc,
        )

    def update_weight(self, u: V, v: V, new_weight: float) -> None:
        """Update one directed edge's weight and re-enforce DPC.

        Phases 1–2 stay valid (topology unchanged); only d* needs
        re-derivation. Cost: \\(O(N \\cdot \\text{tw}^2)\\).
        """
        if not self.graph.has_edge(u, v):
            raise ValueError(f"edge ({u!r}, {v!r}) not in graph")
        self.graph[u][v]["weight"] = new_weight
        self.dpc_weights = _enforce_dpc(self.graph, self.chordal_neighbors, self.peo, self.peo_index)

    def update_weights(self, updates: dict[tuple[V, V], float]) -> None:
        """Apply multiple edge-weight updates and re-enforce DPC once.

        Unknown edges are silently ignored. One DPC pass for the whole batch
        instead of one per change.
        """
        if not updates:
            return
        for (u, v), w in updates.items():
            if self.graph.has_edge(u, v):
                self.graph[u][v]["weight"] = w
        self.dpc_weights = _enforce_dpc(self.graph, self.chordal_neighbors, self.peo, self.peo_index)

    def query(self, source: V) -> dict[V, float]:
        """Single-source shortest paths from *source* (Algorithm 4).

        Returns a mapping ``{v: dist}``. Unreachable vertices are absent (the
        underlying ``inf`` is filtered).
        """
        if source not in self.peo_index:
            raise ValueError(f"source {source!r} not in graph")
        return _query_sssp(self.peo, self.peo_index, self.chordal_neighbors, self.dpc_weights, source)

    def shortest_path(self, source: V, target: V) -> list[V] | None:
        """Reconstruct the optimal node-sequence ``[source, ..., target]``.

        Delegates to NetworkX Bellman-Ford on the original graph (DPC's
        chordal-completion shortcuts can't be walked back to a single
        original edge); falls back to unweighted BFS on negative-cycle
        subgraphs.
        """
        if source not in self.peo_index or target not in self.peo_index:
            return None
        try:
            return nx.bellman_ford_path(self.graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        except nx.NetworkXUnbounded:
            # Real DEX data often has negative cycles (arbitrages or testnet
            # price quirks). Bellman-Ford refuses; fall back to unweighted BFS
            # which always finds *some* path. Optimality of the candidate set
            # is recovered downstream by ASGM re-quoting amount_out at the
            # actual trade size.
            try:
                return nx.shortest_path(self.graph, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

    def top_k_paths(self, source: V, target: V, k: int) -> list[list[V]]:
        """Return up to *k* loopless shortest paths from *source* to *target*.

        Yen's K-shortest paths with Bellman-Ford spurs (handles negative
        weights; an unweighted-BFS fallback fires when a spur subgraph has
        a negative cycle). Cost: \\(O(k^2 \\cdot |E| \\cdot |V|)\\) worst
        case.
        """
        if k <= 0 or source not in self.peo_index or target not in self.peo_index:
            return []

        first = self.shortest_path(source, target)
        if first is None or len(first) < 2:
            return []
        confirmed: list[list[V]] = [first]
        candidates: list[tuple[float, list[V]]] = []  # (cost, path), kept sorted by cost

        for _ in range(k - 1):
            prev = confirmed[-1]
            for i in range(len(prev) - 1):
                spur_node = prev[i]
                root_path = prev[: i + 1]

                # Edges to forbid: any (prev[i], prev[i+1])-style step that a
                # confirmed path took from this same root_path prefix. Without
                # this Yen would re-emit confirmed paths.
                forbidden_edges: set[tuple[V, V]] = set()
                for p in confirmed:
                    if len(p) > i and list(p[: i + 1]) == list(root_path):
                        forbidden_edges.add((p[i], p[i + 1]))

                # Build a subgraph view that hides:
                #   • forbidden_edges (so spur cannot retrace a confirmed step)
                #   • root nodes other than spur_node (loopless: a simple path
                #     can't revisit a node already on its prefix).
                forbidden_nodes = set(root_path[:-1])
                sub = self.graph.edge_subgraph(
                    (u, v)
                    for u, v in self.graph.edges()
                    if (u, v) not in forbidden_edges and u not in forbidden_nodes and v not in forbidden_nodes
                ).copy()

                if spur_node not in sub or target not in sub:
                    continue
                try:
                    spur = nx.bellman_ford_path(sub, spur_node, target, weight="weight")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                except nx.NetworkXUnbounded:
                    # Real DEX graphs often have negative-cost cycles (testnet
                    # quirks, decimal-shifted rates). Fall back to unweighted
                    # BFS so we still produce *some* alternate path; ASGM
                    # downstream re-quotes amount_out anyway and will reject
                    # the bad ones.
                    try:
                        spur = nx.shortest_path(sub, spur_node, target)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                total_path = list(root_path[:-1]) + spur
                # Skip duplicates (Yen can re-derive the same path from
                # different spurs in pathological graphs).
                if any(c[1] == total_path for c in candidates):
                    continue
                if any(p == total_path for p in confirmed):
                    continue
                total_cost = sum(
                    self.graph[total_path[j]][total_path[j + 1]].get("weight", 0.0) for j in range(len(total_path) - 1)
                )
                candidates.append((total_cost, total_path))

            if not candidates:
                break
            candidates.sort(key=lambda c: c[0])
            _, next_path = candidates.pop(0)
            confirmed.append(next_path)

        return confirmed


# ---------------------------------------------------------------------------
# Algorithm 2 — ComputeCompletionAndPEO
# ---------------------------------------------------------------------------


def _chordal_completion_and_peo(tree: nx.Graph) -> tuple[dict[V, set[V]], list[V]]:
    """Walk the tree of bags leaf-to-root; build chordal-completion adjacency + PEO.

    Returns ``(neighbors, peo)`` where *neighbors* is the undirected adjacency
    of the chordal completion (each clique-bag contributes a clique to it), and
    *peo* is a perfect elimination ordering π = (v_1, …, v_n) — vertices
    eliminated earliest are listed first.
    """
    # Work on a mutable copy of the bag-tree.
    work = nx.Graph()
    for bag in tree.nodes():
        work.add_node(frozenset(bag))
    for a, b in tree.edges():
        work.add_edge(frozenset(a), frozenset(b))

    chordal_neighbors: dict[V, set[V]] = {}
    peo: list[V] = []
    seen: set[V] = set()

    # Repeatedly pop a leaf bag, add its clique edges, prepend its new vertices.
    while work.number_of_nodes() > 0:
        if work.number_of_nodes() == 1:
            (last,) = list(work.nodes())
            parent_bag: frozenset[V] = frozenset()
            current_bag = last
        else:
            # any leaf (degree 1 in the current bag-tree)
            current_bag = next(b for b in work.nodes() if work.degree(b) == 1)
            parent_bag = next(iter(work.neighbors(current_bag)))
        # Make the bag a clique in the chordal completion.
        bag_list = list(current_bag)
        for v in bag_list:
            chordal_neighbors.setdefault(v, set())
        for i, u in enumerate(bag_list):
            for w in bag_list[i + 1 :]:
                chordal_neighbors[u].add(w)
                chordal_neighbors[w].add(u)
        # Prepend bag's "new" vertices (in this bag, not yet in π, not in parent).
        new_vertices = [v for v in bag_list if v not in seen and v not in parent_bag]
        peo = new_vertices + peo
        seen.update(new_vertices)
        work.remove_node(current_bag)

    return chordal_neighbors, peo


# ---------------------------------------------------------------------------
# Algorithm 3 — EnforceDPC
# ---------------------------------------------------------------------------


def _enforce_dpc(
    graph: nx.DiGraph,
    chordal_neighbors: dict[V, set[V]],
    peo: list[V],
    peo_index: dict[V, int],
) -> dict[V, dict[V, float]]:
    """Backward PEO walk that fills in d*(u, v) for every chordal-completion edge.

    For each ``v_k`` (from late to early in π), every pair ``(v_i, v_j)`` of
    neighbors of ``v_k`` with ``i < k`` and ``j < k`` is checked: if going
    through ``v_k`` is cheaper, ``d*[v_i][v_j]`` is relaxed. Original directed
    edges seed the table; pairs without a chordal edge between them (i.e. fill-
    ins outside any bag) are not initialised but get filled in by relaxation.
    """
    # Initialise d*[u][v] from original edge weights; default ∞.
    dpc: dict[V, dict[V, float]] = {v: {} for v in peo}
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        # Collapse parallel/self edges by min weight.
        if u == v:
            continue
        cur = dpc[u].get(v, math.inf)
        if w < cur:
            dpc[u][v] = w

    n = len(peo)
    # Walk PEO backward.
    for k in range(n - 1, -1, -1):
        vk = peo[k]
        # Neighbors of vk in Ĝ that come earlier in PEO.
        earlier = [u for u in chordal_neighbors.get(vk, ()) if peo_index[u] < k]
        # For every ordered pair (vi, vj) with i, j < k, try relaxing through vk.
        # The paper's algorithm covers BOTH directions because the graph is directed.
        for vi in earlier:
            d_vi_vk = dpc[vi].get(vk, math.inf)
            if d_vi_vk == math.inf:
                continue
            for vj in earlier:
                if vi == vj:
                    continue
                d_vk_vj = dpc[vk].get(vj, math.inf)
                if d_vk_vj == math.inf:
                    continue
                cand = d_vi_vk + d_vk_vj
                if cand < dpc[vi].get(vj, math.inf):
                    dpc[vi][vj] = cand
    return dpc


# ---------------------------------------------------------------------------
# Algorithm 4 — QuerySSSP (bitonic two-pass)
# ---------------------------------------------------------------------------


def _query_sssp(
    peo: list[V],
    peo_index: dict[V, int],
    chordal_neighbors: dict[V, set[V]],
    dpc: dict[V, dict[V, float]],
    source: V,
) -> dict[V, float]:
    """Two-pass scan over the PEO; relaxes only earlier-PEO then later-PEO neighbors."""
    n = len(peo)
    dist: dict[V, float] = {v: math.inf for v in peo}
    dist[source] = 0.0
    src_idx = peo_index[source]

    # Backward pass: from source's position to the start of the PEO,
    # relaxing edges to *earlier-PEO* neighbors.
    for k in range(src_idx, -1, -1):
        vk = peo[k]
        if dist[vk] == math.inf:
            continue
        for vj in chordal_neighbors.get(vk, ()):
            j = peo_index[vj]
            if j < k:
                w = dpc[vk].get(vj, math.inf)
                if w == math.inf:
                    continue
                cand = dist[vk] + w
                if cand < dist[vj]:
                    dist[vj] = cand

    # Forward pass: from start to end, relaxing edges to *later-PEO* neighbors.
    for k in range(n):
        vk = peo[k]
        if dist[vk] == math.inf:
            continue
        for vj in chordal_neighbors.get(vk, ()):
            j = peo_index[vj]
            if j > k:
                w = dpc[vk].get(vj, math.inf)
                if w == math.inf:
                    continue
                cand = dist[vk] + w
                if cand < dist[vj]:
                    dist[vj] = cand

    return {v: d for v, d in dist.items() if d != math.inf}

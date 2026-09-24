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

from pydefi.pathfinder.graph import Q96, PoolEdge, PoolGraph

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
        ratio = sqrtP / Q96
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
    # witness[u][v] = vk if d*[u][v] was relaxed through vk; absent if the
    # entry came directly from an original graph edge. Drives shortcut
    # expansion in :meth:`shortest_path`.
    dpc_witness: dict[V, dict[V, V]]
    # Cached dict-of-dict adjacency: ``adj_weights[u][v] = weight``. Built
    # once at :meth:`build` from the NetworkX graph; mutated in place by
    # :meth:`update_weight`/:meth:`update_weights` so :func:`_enforce_dpc`
    # can re-seed without re-iterating the NetworkX edges view (~3× faster
    # on the seed loop, dominates DPC cost on large graphs).
    adj_weights: dict[V, dict[V, float]]

    @classmethod
    def build(cls, graph: nx.DiGraph) -> HermesRouter:
        """Run phases 1–3 against *graph*.

        *graph* must be a directed graph whose edge data carries a numeric
        ``weight`` attribute (default 1 if missing). Multi-edges are not
        supported — collapse them upstream by picking the min weight per
        directed pair.
        """
        if graph.number_of_nodes() == 0:
            return cls(graph, 0, [], [], {}, {}, {}, {}, {})
        undirected = graph.to_undirected(as_view=False)
        # NetworkX's tree-decomp solver doesn't tolerate self-loops.
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        tw, tree = treewidth_min_fill_in(undirected)
        bags = [frozenset(b) for b in tree.nodes()]
        chordal_neighbors, peo = _chordal_completion_and_peo(tree)
        peo_index = {v: i for i, v in enumerate(peo)}
        adj_weights = _build_adj_weights(graph)
        dpc, witness = _enforce_dpc(adj_weights, chordal_neighbors, peo, peo_index)
        return cls(
            graph=graph,
            treewidth=tw,
            bags=bags,
            peo=peo,
            peo_index=peo_index,
            chordal_neighbors=chordal_neighbors,
            dpc_weights=dpc,
            dpc_witness=witness,
            adj_weights=adj_weights,
        )

    def update_weight(self, u: V, v: V, new_weight: float) -> None:
        """Update one directed edge's weight and re-enforce DPC.

        Phases 1–2 stay valid (topology unchanged); only d* needs
        re-derivation. Cost: \\(O(N \\cdot \\text{tw}^2)\\).

        Self-loops are silently dropped from DPC (matches
        :func:`_build_adj_weights`); the graph weight is updated for
        cosmetic consistency. Edges added directly to ``self.graph`` by
        bypassing this API and :meth:`add_edge` are not in the
        ``adj_weights`` cache and raise ``ValueError`` — use
        :meth:`add_edge` to register topology changes.
        """
        if not self.graph.has_edge(u, v):
            raise ValueError(f"edge ({u!r}, {v!r}) not in graph")
        self.graph[u][v]["weight"] = new_weight
        if u == v:
            return  # self-loop: unroutable, kept out of DPC by invariant
        if v not in self.adj_weights.get(u, {}):
            raise ValueError(
                f"edge ({u!r}, {v!r}) is in graph but not registered with the router; "
                f"use add_edge() to register topology changes"
            )
        self.adj_weights[u][v] = new_weight
        self.dpc_weights, self.dpc_witness = _enforce_dpc(
            self.adj_weights, self.chordal_neighbors, self.peo, self.peo_index
        )

    def update_weights(self, updates: dict[tuple[V, V], float]) -> None:
        """Apply multiple edge-weight updates and re-enforce DPC once.

        Self-loops and edges not registered with the router (added via
        external graph mutation) are silently skipped. One DPC pass for
        the whole batch instead of one per change.
        """
        if not updates:
            return
        for (u, v), w in updates.items():
            if not self.graph.has_edge(u, v):
                continue
            self.graph[u][v]["weight"] = w
            if u == v:
                continue
            if v not in self.adj_weights.get(u, {}):
                continue
            self.adj_weights[u][v] = w
        self.dpc_weights, self.dpc_witness = _enforce_dpc(
            self.adj_weights, self.chordal_neighbors, self.peo, self.peo_index
        )

    def add_edge(self, u: V, v: V, weight: float) -> bool:
        """Add/update edge ``(u, v)``; return True iff only Phase 3 reran.

        Paper §4.2 fast path: if the chordal completion already covers
        ``(u, v)`` the topology is unchanged and we only re-enforce DPC.
        Otherwise Phases 1-3 rebuild against the mutated graph.

        Self-loops are no-ops (return True) — they're unroutable and
        :func:`_build_adj_weights` filters them out anyway.
        """
        if u == v:
            return True
        self.graph.add_edge(u, v, weight=weight)
        if u in self.peo_index and v in self.peo_index and v in self.chordal_neighbors.get(u, set()):
            self.adj_weights.setdefault(u, {})[v] = weight
            self.dpc_weights, self.dpc_witness = _enforce_dpc(
                self.adj_weights, self.chordal_neighbors, self.peo, self.peo_index
            )
            return True
        # Topology change — rebuild Phases 1-3 in place from the mutated graph.
        rebuilt = HermesRouter.build(self.graph)
        self.treewidth = rebuilt.treewidth
        self.bags = rebuilt.bags
        self.peo = rebuilt.peo
        self.peo_index = rebuilt.peo_index
        self.chordal_neighbors = rebuilt.chordal_neighbors
        self.dpc_weights = rebuilt.dpc_weights
        self.dpc_witness = rebuilt.dpc_witness
        self.adj_weights = rebuilt.adj_weights
        return False

    def query(self, source: V) -> dict[V, float]:
        """Single-source shortest paths from *source* (Algorithm 4).

        Returns a mapping ``{v: dist}``. Unreachable vertices are absent (the
        underlying ``inf`` is filtered).
        """
        if source not in self.peo_index:
            raise ValueError(f"source {source!r} not in graph")
        return _query_sssp(self.peo, self.peo_index, self.chordal_neighbors, self.dpc_weights, source)

    def singular_vertices(self) -> set[V]:
        """Vertices whose Bellman-Ford distance is unbounded below.

        Paper §6 / arbitrage detection. On a ``-log(post-fee-rate)`` DEX
        graph a negative cycle is a profitable closed loop after fees;
        any vertex on the cycle, *or sharing an SCC with one*, has BF
        shortest-path ``-inf`` (the cycle can be traversed arbitrarily
        many times). This method returns that *closure*, which is the
        ref repo's `_pre_singular_vertices ∪ SCC closure` semantics —
        not strictly cycle members. A vertex ``w`` reachable from a
        cycle and able to reach back is included even when no simple
        cycle through ``w`` is itself negative.

        For *strict* cycle membership, run a simple-cycle enumeration
        on the seed-containing SCCs — Hermes' DPC table doesn't
        preserve that information.

        Algorithm:

        1. Seeds via DPC: for each chordal-completion pair ``(u, v)``,
           ``dpc[u][v] + dpc[v][u] < 0`` flags both as seeds. Treewidth
           boundedness guarantees every original-graph cycle has a
           chord, so every negative cycle surfaces through at least one
           such pair.
        2. Negative self-loops: explicit single-vertex check.
        3. SCC closure: every vertex sharing an SCC with a seed.
        """
        seeds: set[V] = set()
        for u in self.peo:
            u_dpc = self.dpc_weights.get(u, {})
            for v in self.chordal_neighbors.get(u, ()):
                if u == v:
                    continue
                forward = u_dpc.get(v, math.inf)
                if not math.isfinite(forward):
                    continue
                backward = self.dpc_weights.get(v, {}).get(u, math.inf)
                if not math.isfinite(backward):
                    continue
                if forward + backward < 0:
                    seeds.add(u)
                    seeds.add(v)
        # Negative self-loops: single-vertex cycles that the chordal-
        # completion pair check skips (u == v branch above).
        for v in self.graph.nodes():
            if self.graph.has_edge(v, v) and self.graph[v][v].get("weight", 0.0) < 0:
                seeds.add(v)
        if not seeds:
            return set()
        singular: set[V] = set()
        for scc in nx.strongly_connected_components(self.graph):
            scc_set = set(scc)
            if scc_set & seeds:
                singular |= scc_set
        return singular

    def shortest_path(self, source: V, target: V) -> list[V] | None:
        """Reconstruct the optimal ``[source, ..., target]`` node sequence.

        Walks DPC parent pointers and expands chordal shortcuts back to
        original edges (O(path-length)). BF/BFS fallback only fires if the
        parent walk fails — typically a corrupted witness chain on real
        DEX data with negative cycles.
        """
        if source not in self.peo_index or target not in self.peo_index:
            return None
        if source == target:
            return [source]

        dist, parent = _query_sssp_with_parents(
            self.peo, self.peo_index, self.chordal_neighbors, self.dpc_weights, source
        )
        if math.isfinite(dist.get(target, math.inf)):
            chain: list[V] = [target]
            cur = target
            seen: set[V] = {target}
            while cur != source:
                p = parent.get(cur)
                if p is None or p in seen:
                    chain = []
                    break
                chain.append(p)
                seen.add(p)
                cur = p
            if chain and chain[-1] == source:
                chain.reverse()
                expanded: list[V] = [source]
                ok = True
                for i in range(len(chain) - 1):
                    if not _expand_chordal_edge(chain[i], chain[i + 1], self.graph, self.dpc_witness, expanded):
                        ok = False
                        break
                if ok:
                    return expanded

        # Fallback: parent walk failed or chordal expansion bottomed out
        # against a missing original edge. This shouldn't happen on a clean
        # graph; real DEX data with abundant negative cycles can produce
        # witness chains that don't terminate cleanly. Defer to BF/BFS so
        # the caller still gets *some* path.
        try:
            return nx.bellman_ford_path(self.graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        except nx.NetworkXUnbounded:
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

                # Hide forbidden_edges (so the spur cannot retrace a confirmed
                # step) and the root nodes before spur_node (loopless: a simple
                # path cannot revisit its own prefix). ``restricted_view`` wraps
                # the graph in filters at O(1); materialising the subgraph here
                # instead cost a full copy per spur node, which dominated the
                # whole search.
                forbidden_nodes = set(root_path[:-1])
                sub = nx.restricted_view(self.graph, forbidden_nodes, forbidden_edges)

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


def _build_adj_weights(graph: nx.DiGraph) -> dict[V, dict[V, float]]:
    """Materialise a dict-of-dict ``adj[u][v] = weight`` view of *graph*.

    Iterating the result is ~3× faster than ``graph.edges(data=True)`` for
    the DPC seed loop on large graphs. Self-loops dropped (DPC can't use
    them and NetworkX's tree-decomp solver rejects them).
    """
    adj: dict[V, dict[V, float]] = {u: {} for u in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        adj[u][v] = data.get("weight", 1.0)
    return adj


def _enforce_dpc(
    adj_weights: dict[V, dict[V, float]],
    chordal_neighbors: dict[V, set[V]],
    peo: list[V],
    peo_index: dict[V, int],
) -> tuple[dict[V, dict[V, float]], dict[V, dict[V, V]]]:
    """Backward PEO walk that fills in d*(u, v) for every chordal-completion edge.

    For each ``v_k`` (from late to early in π), every pair ``(v_i, v_j)`` of
    neighbors of ``v_k`` with ``i < k`` and ``j < k`` is checked: if going
    through ``v_k`` is cheaper, ``d*[v_i][v_j]`` is relaxed. Original directed
    edges seed the table; pairs without a chordal edge between them (i.e. fill-
    ins outside any bag) are not initialised but get filled in by relaxation.

    Returns ``(dpc, witness)``. ``witness[u][v] = vk`` records which vertex
    realised the relaxation of ``d*[u][v]``; if ``(u, v)`` is absent from
    ``witness`` the entry corresponds to an original graph edge and needs no
    expansion.
    """
    # Initialise d*[u][v] from original edge weights; default ∞.
    dpc: dict[V, dict[V, float]] = {v: {} for v in peo}
    witness: dict[V, dict[V, V]] = {v: {} for v in peo}
    for u, neighbors in adj_weights.items():
        if u not in dpc:
            continue
        u_dpc = dpc[u]
        u_wit = witness[u]
        for v, w in neighbors.items():
            cur = u_dpc.get(v, math.inf)
            if w < cur:
                u_dpc[v] = w
                u_wit.pop(v, None)  # original edge; no shortcut to expand

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
                    witness[vi][vj] = vk
    return dpc, witness


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
    dist, _ = _query_sssp_with_parents(peo, peo_index, chordal_neighbors, dpc, source)
    return {v: d for v, d in dist.items() if d != math.inf}


def _query_sssp_with_parents(
    peo: list[V],
    peo_index: dict[V, int],
    chordal_neighbors: dict[V, set[V]],
    dpc: dict[V, dict[V, float]],
    source: V,
) -> tuple[dict[V, float], dict[V, V]]:
    """Bitonic two-pass scan returning ``(dist, parent)``.

    ``parent[v]`` is the *chordal-completion* predecessor that produced
    ``dist[v]``; the (parent, v) pair may be a fill-in shortcut and must be
    expanded via :func:`_expand_chordal_edge` to recover original edges.
    """
    n = len(peo)
    dist: dict[V, float] = {v: math.inf for v in peo}
    parent: dict[V, V] = {}
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
                    parent[vj] = vk

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
                    parent[vj] = vk

    return dist, parent


def _expand_chordal_edge(
    u: V,
    v: V,
    graph: nx.DiGraph,
    witness: dict[V, dict[V, V]],
    out: list[V],
    seen: set[tuple[V, V]] | None = None,
) -> bool:
    """Append original edges for ``u → … → v`` to *out* (excluding ``u``).

    Original edge: append ``v``. Shortcut via witness ``w``: recurse on
    ``(u, w)`` then ``(w, v)``. Returns ``False`` on missing original
    edge or witness cycle (the *seen* set prevents recursion blowup).
    """
    if seen is None:
        seen = set()
    pair = (u, v)
    if pair in seen:
        return False
    seen.add(pair)
    w = witness.get(u, {}).get(v)
    if w is None:
        if not graph.has_edge(u, v):
            return False
        out.append(v)
        return True
    # Shortcut: u → w → v, expand both halves.
    if not _expand_chordal_edge(u, w, graph, witness, out, seen):
        return False
    return _expand_chordal_edge(w, v, graph, witness, out, seen)

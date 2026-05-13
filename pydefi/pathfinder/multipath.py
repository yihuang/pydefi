"""Data structures for :mod:`pydefi.pathfinder.asgm`.

:class:`MultiEdgePath` is a chain of hops where each hop is a parallel-edge
bundle (e.g. several V3 fee tiers for the same pair). :func:`merge_and_expand`
is PRIME's ``MERGEANDEXPAND`` — collapses same-token-sequence candidate
routes into one path and widens each hop with parallel edges from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydefi.pathfinder.graph import PoolEdge, PoolGraph
from pydefi.types import Address

_DISTRIBUTE_SCALE = 1 << 53  # float mantissa width — exact round-trip for w ∈ [0, 1]


def _distribute_int(total: int, weights: list[float]) -> list[int]:
    """Allocate ``total`` across simplex *weights* with exact integer conservation.

    Largest-remainder (Hamilton) apportionment in integer arithmetic: scale
    the float weights into integers, then floor by ``total * w_int // sum``
    and hand the leftover units to the largest integer-remainder entries.
    Conserves ``total`` exactly — even when ``total > 2^53`` (float precision
    drops by integer ULPs there) or the float weights drift from sum=1 (e.g.
    ``[1/3]*3 ≈ 0.999…``).
    """
    if total <= 0 or not weights:
        return [0] * len(weights)
    int_weights = [max(0, round(w * _DISTRIBUTE_SCALE)) for w in weights]
    s = sum(int_weights)
    if s <= 0:
        return [0] * len(weights)
    numerators = [total * iw for iw in int_weights]
    floors = [n // s for n in numerators]
    remainders = [n % s for n in numerators]
    leftover = total - sum(floors)
    if leftover > 0:
        order = sorted(range(len(weights)), key=lambda i: remainders[i], reverse=True)
        for i in order[:leftover]:
            floors[i] += 1
    return floors


@dataclass(frozen=True)
class MultiEdgePath:
    """Multi-hop path where each hop is a parallel bundle of pool edges.

    ``hops[i]`` is a non-empty list of :class:`PoolEdge` instances, all sharing
    the same input/output tokens. Walking the path means: for each hop, split
    the running amount across the bundle's edges according to per-bundle
    weights, and sum the outputs to get the hop's output.
    """

    hops: tuple[tuple[PoolEdge, ...], ...]

    def __post_init__(self) -> None:
        if not self.hops:
            raise ValueError("MultiEdgePath must contain at least one hop")
        for i, bundle in enumerate(self.hops):
            if not bundle:
                raise ValueError(f"hop {i} is empty")
            t_in = bundle[0].token_in.address
            t_out = bundle[0].token_out.address
            for e in bundle[1:]:
                if e.token_in.address != t_in or e.token_out.address != t_out:
                    raise ValueError(f"hop {i} edges must share (token_in, token_out)")
            if i + 1 < len(self.hops):
                next_in = self.hops[i + 1][0].token_in.address
                if t_out != next_in:
                    raise ValueError(f"hop {i}'s output token must match hop {i + 1}'s input")

    def amount_out(self, amount_in: int, weights: PathWeights) -> int:
        """Walk the path under *weights*; return the integer output.

        Each hop's input is distributed across its bundle via
        :func:`_distribute_int` (largest-remainder, total-preserving). Per-hop
        outputs sum to the hop's total, feeding the next hop.
        """
        if amount_in <= 0:
            return 0
        if len(weights.intra_hop_weights) != len(self.hops):
            raise ValueError(f"weights cover {len(weights.intra_hop_weights)} hops but path has {len(self.hops)}")
        cur = amount_in
        for bundle, ws in zip(self.hops, weights.intra_hop_weights):
            if len(ws) != len(bundle):
                raise ValueError(f"hop weight count {len(ws)} does not match bundle size {len(bundle)}")
            allocs = _distribute_int(cur, ws)
            hop_out = sum(edge.amount_out(a) for edge, a in zip(bundle, allocs) if a > 0)
            cur = hop_out
            if cur <= 0:
                return 0
        return cur


@dataclass
class PathWeights:
    """Allocation weights for one :class:`MultiEdgePath`.

    ``path_weight`` is the path's share of the total input (the simplex sum
    across all paths considered together is 1). ``intra_hop_weights`` carries
    one inner list per hop whose entries sum to ~1 across that hop's bundle.

    Mutable on purpose — :func:`pydefi.pathfinder.asgm.asgm_optimize` updates
    these in place during gradient ascent.
    """

    path_weight: float
    intra_hop_weights: list[list[float]] = field(default_factory=list)


def merge_and_expand(
    routes: list[list[PoolEdge]],
    graph: PoolGraph,
    *,
    max_parallel_per_hop: int = 4,
) -> list[MultiEdgePath]:
    """Collapse same-token-sequence *routes* into multi-edge paths.

    Each hop bundle keeps the candidate routes' edges (in discovery order),
    then is widened with additional parallel edges from *graph* up to
    *max_parallel_per_hop* to bound bundle size on dense graphs.

    Enforces PRIME's pool-disjoint constraint across paths (§III): a pool
    claimed by an earlier-ranked path is excluded from later paths' bundles,
    both during merge and during graph-widening. A path whose every bundle is
    emptied by this filter is dropped. Earlier ranking comes from *routes*'
    input order (the caller sorts by output descending).
    """
    by_seq: dict[tuple[Address, ...], list[list[PoolEdge]]] = {}
    for path in routes:
        if not path:
            continue
        seq = (path[0].token_in.address,) + tuple(e.token_out.address for e in path)
        by_seq.setdefault(seq, []).append(path)

    result: list[MultiEdgePath] = []
    claimed: set[Address] = set()
    for paths in by_seq.values():
        n_hops = len(paths[0])
        bundles: list[list[PoolEdge]] = [[] for _ in range(n_hops)]
        seen: list[set[Address]] = [set() for _ in range(n_hops)]
        for path in paths:
            for i, edge in enumerate(path):
                if edge.pool_address in claimed or edge.pool_address in seen[i]:
                    continue
                bundles[i].append(edge)
                seen[i].add(edge.pool_address)
        for i in range(n_hops):
            if not bundles[i]:
                continue
            t_in = bundles[i][0].token_in
            t_out_addr = bundles[i][0].token_out.address
            for edge in graph.edges_from(t_in):
                if edge.token_out.address != t_out_addr:
                    continue
                if edge.pool_address in claimed or edge.pool_address in seen[i]:
                    continue
                bundles[i].append(edge)
                seen[i].add(edge.pool_address)
                if len(bundles[i]) >= max_parallel_per_hop:
                    break
        if any(not b for b in bundles):
            continue
        result.append(MultiEdgePath(hops=tuple(tuple(b) for b in bundles)))
        for bundle in bundles:
            for edge in bundle:
                claimed.add(edge.pool_address)
    return result

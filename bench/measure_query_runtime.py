"""Phase 3: per-query SSSP runtime — Hermes vs pydefi DP vs Bellman-Ford.

Tests H2 (per-query is O(n · tw²)) and H3 (crossover with pydefi DP).
The fixture is core-periphery; weights are -log(spot rate) so all three
algorithms see equivalent inputs.

Run: ``uv run python -m bench.measure_query_runtime``
"""

from __future__ import annotations

import json
import random
import sys
import time

import networkx as nx

from pydefi.pathfinder.hermes import HermesRouter
from pydefi.pathfinder.router import Router
from pydefi.types import TokenAmount
from tests._fixtures import core_periphery_fixture


def _benchmark_one(n_periphery: int, n_queries: int, seed: int) -> dict:
    fix = core_periphery_fixture(n_periphery=n_periphery, seed=seed)
    n_total = fix.nx_graph.number_of_nodes()
    rng = random.Random(seed)
    sources = [rng.choice(list(fix.tokens.keys())) for _ in range(n_queries)]

    # --- Hermes ---
    t0 = time.perf_counter()
    hermes = HermesRouter.build(fix.nx_graph)
    hermes_build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for s in sources:
        hermes.query(s)
    hermes_query_total = time.perf_counter() - t0
    hermes_query_avg = hermes_query_total / n_queries

    # --- Bellman-Ford (NetworkX, single-source) ---
    t0 = time.perf_counter()
    for s in sources:
        nx.single_source_bellman_ford_path_length(fix.nx_graph, s, weight="weight")
    bf_total = time.perf_counter() - t0
    bf_avg = bf_total / n_queries

    # --- pydefi Router (find_best_route, hop_cap=3) ---
    # Pick 1 unit of source; routing finds best path to each destination.
    # We measure full enumeration cost = N find_best_route calls per source.
    # This is a heavier workload (the equivalent of one Bellman-Ford SSSP in DP form).
    pydefi_router = Router(fix.pool_graph, max_hops=3)
    pydefi_total = 0.0
    pydefi_call_count = 0
    # Sample destinations to keep runtime manageable on big graphs.
    n_dst_sample = min(50, n_total - 1)
    for s in sources:
        src_token = fix.tokens[s]
        amt = TokenAmount(src_token, 10**src_token.decimals)
        dst_pool = [t for t in fix.tokens.values() if t.symbol != s]
        dst_sample = rng.sample(dst_pool, min(n_dst_sample, len(dst_pool)))
        t0 = time.perf_counter()
        for dst in dst_sample:
            try:
                pydefi_router.find_best_route(amt, dst)
            except Exception:
                pass
        pydefi_total += time.perf_counter() - t0
        pydefi_call_count += len(dst_sample)
    pydefi_per_call_avg = pydefi_total / pydefi_call_count if pydefi_call_count else 0.0
    # Equivalent SSSP cost: N find_best_route calls.
    pydefi_sssp_avg = pydefi_per_call_avg * (n_total - 1)

    return {
        "n_periphery": n_periphery,
        "n_total": n_total,
        "n_queries": n_queries,
        "hermes_build_seconds": hermes_build_s,
        "hermes_query_avg_seconds": hermes_query_avg,
        "bf_query_avg_seconds": bf_avg,
        "pydefi_per_route_avg_seconds": pydefi_per_call_avg,
        "pydefi_full_sssp_avg_seconds": pydefi_sssp_avg,
        "treewidth": hermes.treewidth,
        "seed": seed,
    }


def main() -> int:
    sizes = [50, 100, 200, 500, 1000, 2000]
    n_queries = 30
    print(
        f"{'N':>5} {'tw':>3} {'hermes_build':>12} {'hermes_q':>10} {'BF_q':>10} {'pydefi_route':>13} {'pydefi_sssp':>12}"
    )
    print("-" * 80)
    rows = []
    for n in sizes:
        r = _benchmark_one(n_periphery=n, n_queries=n_queries, seed=0)
        rows.append(r)
        print(
            f"{r['n_total']:>5d} {r['treewidth']:>3d} "
            f"{r['hermes_build_seconds']:>11.4f}s "
            f"{r['hermes_query_avg_seconds'] * 1000:>9.3f}ms "
            f"{r['bf_query_avg_seconds'] * 1000:>9.3f}ms "
            f"{r['pydefi_per_route_avg_seconds'] * 1000:>12.3f}ms "
            f"{r['pydefi_full_sssp_avg_seconds'] * 1000:>11.3f}ms"
        )
    out = "bench/query_runtime_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRaw → {out}")
    print("\nNotes:")
    print("  hermes_q  = single-source SSSP query time (after build).")
    print("  BF_q      = NetworkX Bellman-Ford single_source query time.")
    print("  pydefi_route = ONE find_best_route(src,dst) call (the hop-DP unit cost).")
    print("  pydefi_sssp  = pydefi's equivalent of an SSSP = N-1 find_best_route calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

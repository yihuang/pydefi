"""Phase 5 (now plan_hermes_followups): end-to-end ``find_optimal_split`` runtime
with candidate_solver='hop_dp' vs 'hermes' on core-periphery fixtures.

Reports:
  - Total wall time per query (build + candidate discovery + ASGM).
  - Hermes' build cost (one-time per Router).
  - Output amount delta (must stay within 10bp by Phase 4 tolerance).

Run: ``uv run python -m bench.measure_integrated_query``
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.router import Router
from pydefi.types import TokenAmount
from tests._fixtures import core_periphery_fixture


def _benchmark_one(n_periphery: int, n_queries: int, seed: int) -> dict:
    fix = core_periphery_fixture(n_periphery=n_periphery, seed=seed)
    rng = random.Random(seed)
    tokens = list(fix.tokens.values())
    pairs = []
    for _ in range(n_queries):
        s, d = rng.sample(tokens, 2)
        amt = TokenAmount(s, 10**s.decimals * rng.choice([1, 10, 100]))
        pairs.append((amt, d))

    # ---- hop_dp solver ----
    t0 = time.perf_counter()
    r_dp = Router(fix.pool_graph, max_hops=3, candidate_solver="hop_dp")
    dp_build_s = time.perf_counter() - t0
    dp_query_secs = []
    dp_outputs = []
    for amt, dst in pairs:
        t0 = time.perf_counter()
        try:
            dag = r_dp.find_optimal_split(amt, dst)
            out = r_dp.simulate(dag, amt.amount)
        except NoRouteFoundError:
            out = None
        dp_query_secs.append(time.perf_counter() - t0)
        dp_outputs.append(out)

    # ---- hermes solver ----
    r_he = Router(fix.pool_graph, candidate_solver="hermes")
    # Trigger the lazy Hermes build explicitly so we measure it separately
    # from per-query cost.
    t0 = time.perf_counter()
    r_he._ensure_hermes()
    he_build_s = time.perf_counter() - t0
    he_query_secs = []
    he_outputs = []
    for amt, dst in pairs:
        t0 = time.perf_counter()
        try:
            dag = r_he.find_optimal_split(amt, dst)
            out = r_he.simulate(dag, amt.amount)
        except NoRouteFoundError:
            out = None
        he_query_secs.append(time.perf_counter() - t0)
        he_outputs.append(out)

    # Output regression summary.
    deltas_bps = []
    for o_dp, o_he in zip(dp_outputs, he_outputs):
        if o_dp is None or o_he is None or o_dp == 0:
            continue
        deltas_bps.append((o_he - o_dp) / o_dp * 10_000)

    return {
        "n_periphery": n_periphery,
        "n_total": fix.nx_graph.number_of_nodes(),
        "n_queries": n_queries,
        "seed": seed,
        "dp_build_s": dp_build_s,
        "dp_query_avg_s": statistics.mean(dp_query_secs),
        "dp_query_p95_s": sorted(dp_query_secs)[int(0.95 * len(dp_query_secs))],
        "he_build_s": he_build_s,
        "he_query_avg_s": statistics.mean(he_query_secs),
        "he_query_p95_s": sorted(he_query_secs)[int(0.95 * len(he_query_secs))],
        "delta_bps_min": min(deltas_bps) if deltas_bps else 0.0,
        "delta_bps_max": max(deltas_bps) if deltas_bps else 0.0,
        "delta_bps_med": statistics.median(deltas_bps) if deltas_bps else 0.0,
    }


def main() -> int:
    sizes = [50, 100, 200, 500, 1000]
    n_queries = 30
    rows = []
    print(
        f"{'N':>5} {'q':>3} {'dp_avg_ms':>11} {'he_build_ms':>13} "
        f"{'he_avg_ms':>11} {'he_amort_break':>15} {'Δbps min/med/max':>22}"
    )
    print("-" * 95)
    for n in sizes:
        r = _benchmark_one(n_periphery=n, n_queries=n_queries, seed=0)
        rows.append(r)
        # break-even: he_build_s / (dp_avg - he_avg) — how many queries to amortise.
        per_query_savings = r["dp_query_avg_s"] - r["he_query_avg_s"]
        if per_query_savings > 0:
            break_even = r["he_build_s"] / per_query_savings
            be_str = f"{break_even:.1f}"
        else:
            be_str = "—"
        print(
            f"{r['n_total']:>5d} {r['n_queries']:>3d} "
            f"{r['dp_query_avg_s'] * 1000:>10.3f}ms "
            f"{r['he_build_s'] * 1000:>12.2f}ms "
            f"{r['he_query_avg_s'] * 1000:>10.3f}ms "
            f"{be_str:>14}q "
            f"{r['delta_bps_min']:+6.2f}/{r['delta_bps_med']:+5.2f}/{r['delta_bps_max']:+6.2f}"
        )

    out = "bench/integrated_query_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRaw → {out}")
    print("\nNotes:")
    print("  he_amort_break = queries-to-amortise Hermes' build cost over hop_dp.")
    print("  Δbps = (hermes_out - hop_dp_out) / hop_dp_out × 10000. Phase 4 tolerance: 10bp.")
    print("  Negative Δbps means Hermes returned slightly less; positive means strictly better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark: Rust engine route search vs the Python pathfinder.

Both sides run the same hop-bounded DP over the same synthetic V2 pool set, so this measures engine overhead, not algorithmic difference. V2 quoting is bit-identical between them (997/1000 and 997000/1000000 floor to the same integer), which the pytest entry point asserts before any timing is trusted.

Run with::

    python3 tests/bench_rust_engine.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from pydefi.pathfinder import PoolGraph, Router  # noqa: E402
from pydefi.types import Address, Token, TokenAmount  # noqa: E402

amm_aggregator = pytest.importorskip(
    "amm_aggregator",
    reason="amm-aggregator not built; run `maturin develop --release` in aggregator-rs/crates/py",
)

MAX_HOPS = 3
AMOUNT_IN = 10**15
FEE_BPS = 30  # == 3_000 pips on the Rust side


def _token(i: int) -> Token:
    return Token(chain_id=1, address=Address(i.to_bytes(20, "big")), symbol=f"T{i}")


def _engine_id(token: Token) -> str:
    return "0x" + token.address.hex()


def _reserve(i: int, side: int) -> int:
    """Deterministic, irregular reserves — irregularity avoids amount_out ties between distinct paths."""
    return 10**18 * (50 + (i * 7919 + side * 104729) % 100)


def _pool_pairs(n_tokens: int) -> list[tuple[int, int]]:
    """Ring plus hub-and-spoke: every token reachable from any other within 3 hops."""
    ring = [(i, (i + 1) % n_tokens) for i in range(n_tokens)]
    spokes = [(0, i) for i in range(2, n_tokens - 1)]
    return ring + spokes


def build_graphs(n_tokens: int) -> tuple[list[Token], Router, "amm_aggregator.PoolGraph"]:
    """Build the same pool set in both engines."""
    tokens = [_token(i) for i in range(n_tokens)]
    py_graph = PoolGraph()
    rs_graph = amm_aggregator.PoolGraph()
    for k, (a, b) in enumerate(_pool_pairs(n_tokens)):
        pool_addr = Address((10**9 + k).to_bytes(20, "big"))
        r_a, r_b = _reserve(k, 0), _reserve(k, 1)
        py_graph.add_bidirectional_pool(
            tokens[a], tokens[b], pool_addr, "UniswapV2", reserve_a=r_a, reserve_b=r_b, fee_bps=FEE_BPS
        )
        rs_graph.add_v2_pool(
            "0x" + pool_addr.hex(), _engine_id(tokens[a]), _engine_id(tokens[b]), r_a, r_b, fee_pips=FEE_BPS * 100
        )
    return tokens, Router(py_graph, max_hops=MAX_HOPS), rs_graph


def _queries(tokens: list[Token]) -> list[tuple[Token, Token]]:
    n = len(tokens)
    return [(tokens[1], tokens[n // 2]), (tokens[2], tokens[n - 2]), (tokens[n // 3], tokens[1])]


def run() -> None:
    print(f"{'pools':>6}  {'python/query':>14}  {'rust/query':>12}  {'speedup':>8}")
    print("-" * 50)
    for n_tokens in (32, 128, 512):
        tokens, router, rs_graph = build_graphs(n_tokens)
        queries = _queries(tokens)
        repeat = 5

        t0 = time.perf_counter()
        for _ in range(repeat):
            for src, dst in queries:
                router.find_best_route(TokenAmount(src, AMOUNT_IN), dst)
        py_s = (time.perf_counter() - t0) / (repeat * len(queries))

        t0 = time.perf_counter()
        for _ in range(repeat):
            for src, dst in queries:
                rs_graph.best_route(_engine_id(src), _engine_id(dst), AMOUNT_IN, MAX_HOPS)
        rs_s = (time.perf_counter() - t0) / (repeat * len(queries))

        n_pools = len(_pool_pairs(n_tokens))
        print(f"{n_pools:>6}  {py_s * 1e3:>12.3f}ms  {rs_s * 1e3:>10.3f}ms  {py_s / rs_s:>7.1f}x")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def test_engines_agree_on_v2_routes() -> None:
    """Same pool set, same queries: both engines must report the same best output."""
    tokens, router, rs_graph = build_graphs(32)
    for src, dst in _queries(tokens):
        py_route = router.find_best_route(TokenAmount(src, AMOUNT_IN), dst)
        rs_route = rs_graph.best_route(_engine_id(src), _engine_id(dst), AMOUNT_IN, MAX_HOPS)
        assert rs_route is not None
        assert py_route.amount_out.amount == rs_route.amount_out
        assert len(py_route.steps) == len(rs_route.hops)


if __name__ == "__main__":
    run()

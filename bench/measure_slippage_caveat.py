"""Phase 6: demonstrate Hermes' spot-rate-only blind spot.

Hermes' edge weights are -log(spot rate); for a non-trivial input it
returns the path that *would* be best at infinitesimal trade size, not the
path that maximises actual amount_out. This phase builds a fixture where
the two diverge: a shallow + slightly-better-spot-price pool vs a deep +
slightly-worse-spot pool. ASGM (find_optimal_split) sees the deep pool's
better post-slippage output; Hermes does not.

The conclusion reinforces Hermes' role in a multi-stage pipeline as
*candidate discovery*, with slippage-aware allocation done downstream.

Run: ``uv run python -m bench.measure_slippage_caveat``
"""

from __future__ import annotations

import math
import sys

import networkx as nx

from pydefi.pathfinder.graph import PoolEdge, PoolGraph
from pydefi.pathfinder.hermes import HermesRouter
from pydefi.pathfinder.router import Router
from pydefi.types import Address, ChainId, Token, TokenAmount


def _make_token(sym: str, dec: int, idx: int) -> Token:
    return Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + format(idx, "040x")), symbol=sym, decimals=dec)


def main() -> int:
    weth = _make_token("WETH", 18, 1)
    usdc = _make_token("USDC", 6, 2)

    # Shallow pool: best spot rate but tiny depth → big slippage.
    # Deep pool: worse spot rate but huge depth → tiny slippage.
    SHALLOW_FEE = 5
    DEEP_FEE = 30
    shallow_reserve_in = 10 * 10**18  # 10 WETH
    shallow_reserve_out = 21_000 * 10**6  # 21k USDC ⇒ 2100 USDC/WETH spot
    deep_reserve_in = 10_000 * 10**18  # 10k WETH
    deep_reserve_out = 20_000_000 * 10**6  # 20M USDC ⇒ 2000 USDC/WETH spot

    pool_g = PoolGraph()
    pool_a = Address("0x" + "11" * 20)
    pool_b = Address("0x" + "22" * 20)
    for ti, to, ri, ro, addr, fee, is0 in [
        (weth, usdc, shallow_reserve_in, shallow_reserve_out, pool_a, SHALLOW_FEE, True),
        (usdc, weth, shallow_reserve_out, shallow_reserve_in, pool_a, SHALLOW_FEE, False),
        (weth, usdc, deep_reserve_in, deep_reserve_out, pool_b, DEEP_FEE, True),
        (usdc, weth, deep_reserve_out, deep_reserve_in, pool_b, DEEP_FEE, False),
    ]:
        pool_g.add_pool(
            PoolEdge(
                token_in=ti,
                token_out=to,
                pool_address=addr,
                protocol="UniswapV2",
                reserve_in=ri,
                reserve_out=ro,
                fee_bps=fee,
                extra={"is_token0_in": is0},
            )
        )

    # Build the spot-rate -log(rate) NetworkX graph that Hermes uses (single-pool collapse).
    # For each (u, v), keep the BEST spot rate across pools.
    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(["WETH", "USDC"])
    spot_rates = {("WETH", "USDC"): [], ("USDC", "WETH"): []}
    for ri, ro, fee in [
        (shallow_reserve_in, shallow_reserve_out, SHALLOW_FEE),
        (deep_reserve_in, deep_reserve_out, DEEP_FEE),
    ]:
        ff = (10_000 - fee) / 10_000
        # WETH→USDC: rate = (R_usdc / R_weth) adjusted to per-WETH-unit price.
        rate_we_uc = (ro / 10**6) / (ri / 10**18) * ff
        rate_uc_we = (ri / 10**18) / (ro / 10**6) * ff
        spot_rates[("WETH", "USDC")].append(rate_we_uc)
        spot_rates[("USDC", "WETH")].append(rate_uc_we)
    # Highest rate wins (Hermes collapses multi-edges this way).
    nx_g.add_edge("WETH", "USDC", weight=-math.log(max(spot_rates[("WETH", "USDC")])))
    nx_g.add_edge("USDC", "WETH", weight=-math.log(max(spot_rates[("USDC", "WETH")])))

    print("Fixture: 2 WETH/USDC pools.")
    print(
        f"  Pool A (shallow): {shallow_reserve_in / 10**18:.0f} WETH / "
        f"{shallow_reserve_out / 10**6:.0f} USDC, fee={SHALLOW_FEE}bps  ⇒ spot 2100, post-fee {2100 * (10000 - SHALLOW_FEE) / 10000:.2f}"
    )
    print(
        f"  Pool B (deep):    {deep_reserve_in / 10**18:.0f} WETH / "
        f"{deep_reserve_out / 10**6:.0f} USDC, fee={DEEP_FEE}bps  ⇒ spot 2000, post-fee {2000 * (10000 - DEEP_FEE) / 10000:.2f}"
    )
    print()

    # --- Hermes top-route choice (spot-rate ranked) ---
    hermes = HermesRouter.build(nx_g)
    h_dist = hermes.query("WETH")
    print(f"Hermes -log(rate) distance WETH→USDC = {h_dist['USDC']:.6f}")
    print(
        f"  (chose the shallow pool because its spot rate is higher: "
        f"{max(spot_rates[('WETH', 'USDC')]):.2f} vs {min(spot_rates[('WETH', 'USDC')]):.2f})\n"
    )

    # --- ASGM (find_optimal_split): solves the actual slippage-aware problem ---
    router = Router(pool_g)
    print(
        f"{'amount_in_WETH':>16} {'shallow_alone':>16} {'deep_alone':>14} {'optimal_split':>15} {'optimal_alloc':>30}"
    )
    print("-" * 100)
    for amt_eth in (0.1, 1, 10, 100):
        amt_in = int(amt_eth * 10**18)
        amt = TokenAmount(weth, amt_in)
        # Direct quote on each pool
        edges = {e.pool_address: e for e in pool_g if e.token_in.symbol == "WETH" and e.token_out.symbol == "USDC"}
        out_shallow = edges[pool_a].amount_out(amt_in)
        out_deep = edges[pool_b].amount_out(amt_in)
        dag = router.find_optimal_split(amt, usdc)
        out_optimal = router.simulate(dag, amt_in)
        # Read leg fractions if it split
        actions = dag.to_dict()["actions"]
        from pydefi.types import RouteSplit

        if actions and isinstance(actions[0], RouteSplit):
            split = actions[0]
            allocs = []
            for leg in split.legs:
                # Tag by pool address fragment
                for act in leg.actions:
                    pa = str(act.pool.pool_address.to_0x_hex())
                    label = "deep" if "22" in pa else "shallow"
                    allocs.append(f"{label}={leg.fraction_bps / 100:.0f}%")
                    break
            alloc_str = " + ".join(allocs)
        else:
            # Single-pool route
            pa = str(actions[0].pool.pool_address.to_0x_hex())
            alloc_str = "deep=100%" if "22" in pa else "shallow=100%"
        print(
            f"{amt_eth:>16} {out_shallow / 10**6:>15.2f} "
            f"{out_deep / 10**6:>13.2f} {out_optimal / 10**6:>14.2f}  {alloc_str:>30}"
        )

    print("\nObservation: Hermes always picks the shallow pool because its")
    print("spot rate (2099) exceeds the deep pool's (1994). At any non-trivial trade")
    print("size the deep pool's slippage advantage flips the verdict — but Hermes")
    print("can't see it. ASGM does, hence the split allocations as input grows.")
    print("\nConclusion: Hermes is candidate-discovery only; allocation must be")
    print("solved downstream by a slippage-aware optimiser (e.g. PRIME ASGM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

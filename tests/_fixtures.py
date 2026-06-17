"""Synthetic graph generators for Hermes tests.

* core-periphery: blue-chip core (WETH/USDC/USDT/DAI/WBTC) + alt periphery.
* multi-cluster: disjoint cores linked by bridge tokens; cross-cluster routes
  need >3 hops.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx

from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
from pydefi.types import Address, ChainId, Token


@dataclass
class GraphFixture:
    """A graph emitted in two equivalent forms.

    *nx_graph* carries ``-log(spot rate)`` weights — fed straight into Hermes
    or Bellman-Ford. *pool_graph* keeps reserves so pydefi's DP / ASGM can
    quote real ``amount_out`` on the same topology.
    """

    nx_graph: nx.DiGraph
    pool_graph: PoolGraph
    tokens: dict[str, Token]
    label: str


def make_token(symbol: str, decimals: int, addr_idx: int) -> Token:
    """Synthetic Ethereum-mainnet Token whose address is ``0x{addr_idx:040x}``."""
    return Token(
        chain_id=ChainId.ETHEREUM,
        address=Address("0x" + format(addr_idx, "040x")),
        symbol=symbol,
        decimals=decimals,
    )


def make_v3_edge(
    token_in: Token,
    token_out: Token,
    pool_address: Address,
    *,
    is_token0_in: bool,
    price_token1_per_token0: float = 2000.0,
    fee_bps: int = 30,
    liquidity: int = 5 * 10**22,
) -> V3PoolEdge:
    """V3 pool with sqrtPriceX96 derived from a desired (raw-unit) ``token1/token0`` rate."""
    t0 = token_in if is_token0_in else token_out
    t1 = token_out if is_token0_in else token_in
    p_raw = price_token1_per_token0 * (10**t1.decimals) / (10**t0.decimals)
    sqrt_price_x96 = int(math.sqrt(p_raw) * (2**96))
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


def _add_edge_pair(
    nx_g: nx.DiGraph,
    pool_g: PoolGraph,
    pool_addr: Address,
    t0: Token,
    t1: Token,
    reserve0: int,
    reserve1: int,
    fee_bps: int = 30,
) -> None:
    """Add a bidirectional pool to both graphs with consistent rates."""
    pool_g.add_pool(
        PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=pool_addr,
            protocol="UniswapV2",
            reserve_in=reserve0,
            reserve_out=reserve1,
            fee_bps=fee_bps,
            extra={"is_token0_in": True},
        )
    )
    pool_g.add_pool(
        PoolEdge(
            token_in=t1,
            token_out=t0,
            pool_address=pool_addr,
            protocol="UniswapV2",
            reserve_in=reserve1,
            reserve_out=reserve0,
            fee_bps=fee_bps,
            extra={"is_token0_in": False},
        )
    )
    # Spot price (raw-unit) → -log for the SSSP graph.
    fee_factor = (10_000 - fee_bps) / 10_000
    rate_0_to_1 = (reserve1 / reserve0) * fee_factor
    rate_1_to_0 = (reserve0 / reserve1) * fee_factor
    # NetworkX won't accept duplicate (u, v) — keep the lowest -log (best rate).
    w_01 = -math.log(rate_0_to_1) if rate_0_to_1 > 0 else math.inf
    w_10 = -math.log(rate_1_to_0) if rate_1_to_0 > 0 else math.inf
    cur_01 = nx_g.get_edge_data(t0.symbol, t1.symbol, default={"weight": math.inf})["weight"]
    cur_10 = nx_g.get_edge_data(t1.symbol, t0.symbol, default={"weight": math.inf})["weight"]
    if w_01 < cur_01:
        nx_g.add_edge(t0.symbol, t1.symbol, weight=w_01)
    if w_10 < cur_10:
        nx_g.add_edge(t1.symbol, t0.symbol, weight=w_10)


# ---------------------------------------------------------------------------
# Core-periphery topology
# ---------------------------------------------------------------------------


def core_periphery_fixture(
    *,
    n_periphery: int,
    core_symbols: tuple[str, ...] = ("WETH", "USDC", "USDT", "DAI", "WBTC"),
    n_links_per_alt: tuple[int, ...] = (1, 1, 2, 2, 3),
    seed: int = 0,
) -> GraphFixture:
    """Build a single-DEX-style core-periphery fixture.

    *n_links_per_alt* is sampled (with replacement) for each alt — defaults
    bias toward 1–2 links, matching long-tail observations on Uniswap V2/V3.
    """
    rng = random.Random(seed)
    addr_counter = [1]

    def _next_addr() -> Address:
        addr_counter[0] += 1
        return Address("0x" + format(addr_counter[0], "040x"))

    # Build core tokens.
    core_decimals = {"WETH": 18, "USDC": 6, "USDT": 6, "DAI": 18, "WBTC": 8}
    core_prices_usd = {"WETH": 2000.0, "USDC": 1.0, "USDT": 1.0, "DAI": 1.0, "WBTC": 60000.0}
    tokens: dict[str, Token] = {}
    for i, sym in enumerate(core_symbols):
        tokens[sym] = make_token(sym, core_decimals.get(sym, 18), 0x10 + i)

    # Build alt tokens.
    for i in range(n_periphery):
        sym = f"ALT{i:05d}"
        dec = rng.choice([6, 8, 18])
        tokens[sym] = make_token(sym, dec, 0x1000 + i)

    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(tokens.keys())
    pool_g = PoolGraph()

    # Core clique (every core ↔ every other core).
    DEPTH_USD = 50_000_000
    for i, a in enumerate(core_symbols):
        for b in core_symbols[i + 1 :]:
            ta, tb = tokens[a], tokens[b]
            ra = int(DEPTH_USD / core_prices_usd[a] * 10**ta.decimals)
            rb = int(DEPTH_USD / core_prices_usd[b] * 10**tb.decimals)
            _add_edge_pair(nx_g, pool_g, _next_addr(), ta, tb, ra, rb)

    # Periphery: each ALT picks 1–3 random cores.
    for i in range(n_periphery):
        sym = f"ALT{i:05d}"
        alt = tokens[sym]
        n_links = rng.choice(n_links_per_alt)
        chosen = rng.sample(core_symbols, min(n_links, len(core_symbols)))
        alt_price_usd = rng.uniform(0.001, 100)
        alt_depth_usd = rng.uniform(10_000, 1_000_000)
        for cs in chosen:
            c = tokens[cs]
            ra = int(alt_depth_usd / alt_price_usd * 10**alt.decimals)
            rb = int(alt_depth_usd / core_prices_usd[cs] * 10**c.decimals)
            _add_edge_pair(nx_g, pool_g, _next_addr(), alt, c, ra, rb)

    return GraphFixture(
        nx_graph=nx_g,
        pool_graph=pool_g,
        tokens=tokens,
        label=f"core-periphery N={len(tokens)}",
    )


# ---------------------------------------------------------------------------
# Multi-cluster topology (the >3-hop case)
# ---------------------------------------------------------------------------


def multi_cluster_fixture(
    *,
    n_clusters: int = 2,
    core_size_per_cluster: int = 3,
    n_periphery_per_cluster: int = 20,
    seed: int = 0,
) -> GraphFixture:
    """Several disjoint cores connected by bridge tokens — long-route fixture.

    Cluster_a's core ↔ bridge ↔ cluster_b's core; alts attach to their own
    cluster's core. alt_a → alt_b queries minimum 4 hops.
    """
    rng = random.Random(seed)
    addr_counter = [1]

    def _next_addr() -> Address:
        addr_counter[0] += 1
        return Address("0x" + format(addr_counter[0], "040x"))

    tokens: dict[str, Token] = {}
    cluster_cores: list[list[str]] = []
    for c in range(n_clusters):
        core = []
        for i in range(core_size_per_cluster):
            sym = f"C{c}_{i}"
            tokens[sym] = make_token(sym, 18, 0x100 + c * 16 + i)
            core.append(sym)
        cluster_cores.append(core)

    bridges = []
    for c in range(n_clusters - 1):
        sym = f"BRIDGE_{c}_{c + 1}"
        tokens[sym] = make_token(sym, 18, 0xA00 + c)
        bridges.append((c, c + 1, sym))

    for c, core in enumerate(cluster_cores):
        for i in range(n_periphery_per_cluster):
            sym = f"C{c}_ALT{i:03d}"
            tokens[sym] = make_token(sym, 18, 0x10000 + c * 1000 + i)

    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(tokens.keys())
    pool_g = PoolGraph()

    # Each cluster's core forms a clique.
    DEPTH = 10**24
    for core in cluster_cores:
        for i, a in enumerate(core):
            for b in core[i + 1 :]:
                _add_edge_pair(nx_g, pool_g, _next_addr(), tokens[a], tokens[b], DEPTH, DEPTH)

    # Bridges: bridge ↔ first core of cluster_a and cluster_b.
    for c_a, c_b, bridge_sym in bridges:
        b = tokens[bridge_sym]
        _add_edge_pair(nx_g, pool_g, _next_addr(), tokens[cluster_cores[c_a][0]], b, DEPTH // 100, DEPTH // 100)
        _add_edge_pair(nx_g, pool_g, _next_addr(), b, tokens[cluster_cores[c_b][0]], DEPTH // 100, DEPTH // 100)

    # Alts attach to a random core in their cluster.
    for c, core in enumerate(cluster_cores):
        for i in range(n_periphery_per_cluster):
            sym = f"C{c}_ALT{i:03d}"
            attach_to = rng.choice(core)
            _add_edge_pair(nx_g, pool_g, _next_addr(), tokens[sym], tokens[attach_to], DEPTH // 1000, DEPTH // 1000)

    return GraphFixture(
        nx_graph=nx_g,
        pool_graph=pool_g,
        tokens=tokens,
        label=f"multi-cluster K={n_clusters} N={len(tokens)}",
    )

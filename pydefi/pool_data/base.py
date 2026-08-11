"""
Data types for pool data provider integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
from pydefi.pathfinder.v3_tick_math import TickLadder
from pydefi.types import Token


@dataclass
class PoolData:
    """Data for a single liquidity pool returned by an external data provider.

    Holds the normalised state of a pool so it can be directly converted into
    :class:`~pydefi.pathfinder.graph.PoolEdge` / :class:`~pydefi.pathfinder.graph.V3PoolEdge`
    objects and inserted into a :class:`~pydefi.pathfinder.graph.PoolGraph`.

    Attributes:
        pool_address: On-chain pool contract address.
        protocol: Human-readable protocol name (e.g. ``"UniswapV2"``).
        chain_id: EVM chain ID.
        token0: First token of the pair (canonical order from the source).
        token1: Second token of the pair.
        fee_bps: Swap fee in basis points (e.g. ``30`` = 0.3 %).
        reserve0: Raw on-chain reserve of *token0* (V2-style pools).
        reserve1: Raw on-chain reserve of *token1* (V2-style pools).
        sqrt_price_x96: Current ``sqrtPriceX96`` (Uniswap V3 pools only).
        liquidity: Active liquidity (Uniswap V3 pools only).
        tick_ladder: Initialised ticks near the price, making V3 pricing exact
            across boundaries. Providers here are off-chain, so fill it via
            :meth:`~pydefi.amm.uniswap_v3.UniswapV3.attach_tick_ladders`, and
            refetch once the pool leaves :attr:`TickLadder.price_range` — past
            it the walk extrapolates rather than raising.
        extra: Optional extra metadata (provider-specific).
    """

    pool_address: str
    protocol: str
    chain_id: int
    token0: Token
    token1: Token
    fee_bps: int = 30
    # V2-style reserves
    reserve0: int = 0
    reserve1: int = 0
    # V3-style concentrated liquidity state
    sqrt_price_x96: int = 0
    liquidity: int = 0
    tick_ladder: TickLadder | None = None
    extra: dict = field(default_factory=dict)

    def to_pool_edges(self) -> list[PoolEdge]:
        """Build a bidirectional pair of directed pool edges.

        Returns a list of two edges — one for each swap direction (token0 →
        token1 and token1 → token0).  If *sqrt_price_x96* and *liquidity*
        are both non-zero, :class:`~pydefi.pathfinder.graph.V3PoolEdge`
        objects are returned; otherwise standard constant-product
        :class:`~pydefi.pathfinder.graph.PoolEdge` objects are used.

        Returns:
            A list of exactly two :class:`~pydefi.pathfinder.graph.PoolEdge`
            instances ``[edge_0_to_1, edge_1_to_0]``.
        """
        if self.sqrt_price_x96 > 0 and self.liquidity > 0:
            extra_0_to_1 = dict(self.extra)
            extra_0_to_1.setdefault("is_token0_in", True)
            edge_0_to_1: PoolEdge = V3PoolEdge(
                token_in=self.token0,
                token_out=self.token1,
                pool_address=self.pool_address,
                protocol=self.protocol,
                fee_bps=self.fee_bps,
                sqrt_price_x96=self.sqrt_price_x96,
                liquidity=self.liquidity,
                is_token0_in=True,
                # Shared: a ladder is a set of points on the price axis, and
                # the swap direction is decided per call, not per ladder.
                tick_ladder=self.tick_ladder,
                extra=extra_0_to_1,
            )
            extra_1_to_0 = dict(self.extra)
            extra_1_to_0.setdefault("is_token0_in", False)
            edge_1_to_0: PoolEdge = V3PoolEdge(
                token_in=self.token1,
                token_out=self.token0,
                pool_address=self.pool_address,
                protocol=self.protocol,
                fee_bps=self.fee_bps,
                sqrt_price_x96=self.sqrt_price_x96,
                liquidity=self.liquidity,
                is_token0_in=False,
                tick_ladder=self.tick_ladder,
                extra=extra_1_to_0,
            )
        else:
            extra_0_to_1 = dict(self.extra)
            extra_0_to_1.setdefault("is_token0_in", True)
            edge_0_to_1 = PoolEdge(
                token_in=self.token0,
                token_out=self.token1,
                pool_address=self.pool_address,
                protocol=self.protocol,
                reserve_in=self.reserve0,
                reserve_out=self.reserve1,
                fee_bps=self.fee_bps,
                extra=extra_0_to_1,
            )
            extra_1_to_0 = dict(self.extra)
            extra_1_to_0.setdefault("is_token0_in", False)
            edge_1_to_0 = PoolEdge(
                token_in=self.token1,
                token_out=self.token0,
                pool_address=self.pool_address,
                protocol=self.protocol,
                reserve_in=self.reserve1,
                reserve_out=self.reserve0,
                fee_bps=self.fee_bps,
                extra=extra_1_to_0,
            )
        return [edge_0_to_1, edge_1_to_0]


def build_graph(pools: list[PoolData]) -> PoolGraph:
    """Convert a list of :class:`PoolData` objects into a :class:`~pydefi.pathfinder.graph.PoolGraph`.

    Each :class:`PoolData` entry contributes two directed edges (one per
    swap direction) to the graph via :meth:`PoolData.to_pool_edges`.
    """
    graph = PoolGraph()
    for pool in pools:
        for edge in pool.to_pool_edges():
            graph.add_pool(edge)
    return graph

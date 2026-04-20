"""
Base class and data types for pool data provider integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydefi.pathfinder.graph import PoolEdge, PoolGraph, V3PoolEdge
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


class BasePoolDataProvider(ABC):
    """Abstract base class for pool data providers.

    Sub-classes implement fetching pool state from external sources (e.g. the
    GeckoTerminal REST API or a protocol-specific subgraph) and expose a
    uniform interface that the rest of pydefi can consume.

    The :meth:`build_graph` helper converts a list of :class:`PoolData`
    objects into a :class:`~pydefi.pathfinder.graph.PoolGraph` ready for
    use with :class:`~pydefi.pathfinder.router.Router`.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this data provider."""

    @abstractmethod
    async def get_pool(self, pool_address: str) -> PoolData:
        """Fetch data for a single pool by its on-chain address.

        Args:
            pool_address: Checksummed (or lowercase) pool contract address.

        Returns:
            A populated :class:`PoolData` object.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: On API errors or when
                the pool is not found.
        """

    @abstractmethod
    async def get_top_pools(self, limit: int = 100) -> list[PoolData]:
        """Fetch the top pools ranked by liquidity / TVL.

        Args:
            limit: Maximum number of pools to return.

        Returns:
            A list of :class:`PoolData` objects.
        """

    @abstractmethod
    async def get_pools_for_token(self, token_address: str, limit: int = 100) -> list[PoolData]:
        """Fetch pools that contain a specific token.

        Args:
            token_address: ERC-20 token address to filter by.
            limit: Maximum number of pools to return.

        Returns:
            A list of :class:`PoolData` objects.
        """

    async def get_pools_for_tokens(self, token_addresses: list[str], limit: int = 100) -> list[PoolData]:
        """Fetch pools that contain any of the given tokens.

        Default implementation fans out to :meth:`get_pools_for_token` for
        each address and deduplicates results by pool address.  Providers
        that offer a native batch endpoint (e.g. GeckoTerminal) should
        override this method for better efficiency.

        Args:
            token_addresses: List of ERC-20 token addresses to query.
            limit: Maximum total number of pools to return.

        Returns:
            A deduplicated list of up to *limit* :class:`PoolData` objects.
        """
        pools: list[PoolData] = []
        seen: set[str] = set()
        for address in token_addresses:
            if len(pools) >= limit:
                break
            for pool in await self.get_pools_for_token(address, limit=limit):
                if pool.pool_address not in seen:
                    seen.add(pool.pool_address)
                    pools.append(pool)
                    if len(pools) >= limit:
                        break
        return pools

    def build_graph(self, pools: list[PoolData]) -> PoolGraph:
        """Convert a list of :class:`PoolData` objects into a :class:`~pydefi.pathfinder.graph.PoolGraph`.

        Each :class:`PoolData` entry contributes two directed edges (one per
        swap direction) to the graph via :meth:`PoolData.to_pool_edges`.

        Args:
            pools: Pool data objects to add to the graph.

        Returns:
            A :class:`~pydefi.pathfinder.graph.PoolGraph` populated with all
            provided pools.
        """
        graph = PoolGraph()
        for pool in pools:
            for edge in pool.to_pool_edges():
                graph.add_pool(edge)
        return graph

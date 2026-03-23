"""
Subgraph pool data providers for The Graph protocol.

Supports Uniswap V2 and Uniswap V3 subgraphs.  Any compatible GraphQL
endpoint can be used by supplying a custom ``url``.
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal
from typing import Any, Optional

import aiohttp

from pydefi.exceptions import PoolDataError
from pydefi.pool_data.base import BasePoolDataProvider, PoolData
from pydefi.types import ChainId, Token

# Well-known Uniswap V2 subgraph URLs keyed by chain ID
_UNISWAP_V2_SUBGRAPHS: dict[int, str] = {
    ChainId.ETHEREUM: ("https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"),
}

# Well-known Uniswap V3 subgraph URLs keyed by chain ID
_UNISWAP_V3_SUBGRAPHS: dict[int, str] = {
    ChainId.ETHEREUM: ("https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"),
    ChainId.ARBITRUM: ("https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal"),
    ChainId.OPTIMISM: ("https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis"),
    ChainId.POLYGON: ("https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon"),
    ChainId.BASE: ("https://api.studio.thegraph.com/query/48211/uniswap-v3-base/version/latest"),
}


class Subgraph(BasePoolDataProvider):
    """Abstract base class for GraphQL subgraph clients.

    Provides the shared :meth:`_query` helper for executing GraphQL
    requests.  Concrete sub-classes (:class:`UniswapV2Subgraph`,
    :class:`UniswapV3Subgraph`) implement the protocol-specific queries and
    response parsing.

    Args:
        chain_id: EVM chain ID.
        url: GraphQL endpoint URL of the subgraph.
    """

    _logger = logging.getLogger(__name__)

    def __init__(self, chain_id: int, url: str) -> None:
        self.chain_id = chain_id
        self.url = url

    @property
    def provider_name(self) -> str:
        return "Subgraph"

    async def _query(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query against the subgraph endpoint.

        Args:
            query: GraphQL query string.
            variables: Optional GraphQL variables dict.

        Returns:
            The ``data`` field of the GraphQL response.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: On HTTP errors or
                GraphQL ``errors`` in the response.
        """
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise PoolDataError(
                        f"Subgraph HTTP error {resp.status}: {data}",
                        status_code=resp.status,
                    )
                errors = data.get("errors")
                if errors:
                    raise PoolDataError(f"Subgraph GraphQL errors: {errors}")
                return data.get("data") or {}  # type: ignore[return-value]


class UniswapV2Subgraph(Subgraph):
    """Uniswap V2 subgraph client.

    Queries the Uniswap V2 subgraph for pair data including token reserves.
    Reserves are returned by the subgraph as human-readable decimal strings
    and are converted to raw integer amounts using :mod:`decimal` arithmetic
    to preserve precision.

    Args:
        chain_id: EVM chain ID.
        url: Custom subgraph URL.  If not provided, the canonical Uniswap V2
            subgraph URL for *chain_id* is used (see
            :data:`_UNISWAP_V2_SUBGRAPHS`).

    Raises:
        :class:`ValueError`: If no default URL is available for *chain_id*
            and *url* is not supplied.
    """

    def __init__(self, chain_id: int, url: Optional[str] = None) -> None:
        resolved = url or _UNISWAP_V2_SUBGRAPHS.get(chain_id)
        if not resolved:
            raise ValueError(
                f"No default Uniswap V2 subgraph URL for chain_id {chain_id}. Provide a custom url parameter."
            )
        super().__init__(chain_id, resolved)

    @property
    def provider_name(self) -> str:
        return "UniswapV2Subgraph"

    def _parse_pair(self, pair: dict[str, Any]) -> PoolData:
        """Parse a Uniswap V2 pair entity from the subgraph."""
        t0 = pair["token0"]
        t1 = pair["token1"]

        token0 = Token(
            chain_id=self.chain_id,
            address=t0["id"],
            symbol=t0["symbol"],
            decimals=int(t0["decimals"]),
            name=t0.get("name") or None,
        )
        token1 = Token(
            chain_id=self.chain_id,
            address=t1["id"],
            symbol=t1["symbol"],
            decimals=int(t1["decimals"]),
            name=t1.get("name") or None,
        )

        # The subgraph returns human-readable amounts as strings; use Decimal
        # to avoid floating-point precision loss when multiplying by 10^decimals.
        raw0 = Decimal(str(pair.get("reserve0") or "0"))
        raw1 = Decimal(str(pair.get("reserve1") or "0"))
        reserve0 = int((raw0 * Decimal(10**token0.decimals)).to_integral_value(rounding=ROUND_DOWN))
        reserve1 = int((raw1 * Decimal(10**token1.decimals)).to_integral_value(rounding=ROUND_DOWN))

        return PoolData(
            pool_address=pair["id"],
            protocol="UniswapV2",
            chain_id=self.chain_id,
            token0=token0,
            token1=token1,
            fee_bps=30,
            reserve0=reserve0,
            reserve1=reserve1,
        )

    async def get_pool(self, pool_address: str) -> PoolData:
        """Fetch a Uniswap V2 pair by its on-chain address.

        Args:
            pool_address: Pair contract address (checksummed or lowercase).

        Returns:
            A :class:`PoolData` object.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If the pair is not
                found or the subgraph returns an error.
        """
        query = """
        query GetPair($id: ID!) {
            pair(id: $id) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                reserve0
                reserve1
            }
        }
        """
        data = await self._query(query, {"id": pool_address.lower()})
        pair = data.get("pair")
        if not pair:
            raise PoolDataError(f"Pool {pool_address} not found in Uniswap V2 subgraph")
        return self._parse_pair(pair)

    async def get_top_pools(self, limit: int = 100) -> list[PoolData]:
        """Fetch the top Uniswap V2 pairs ranked by total reserve in USD.

        Args:
            limit: Maximum number of pairs to return (passed directly as the
                GraphQL ``first`` argument; the subgraph caps this at 1000).

        Returns:
            A list of :class:`PoolData` objects.
        """
        query = """
        query TopPairs($limit: Int!) {
            pairs(
                first: $limit
                orderBy: reserveUSD
                orderDirection: desc
            ) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                reserve0
                reserve1
            }
        }
        """
        data = await self._query(query, {"limit": limit})
        result: list[PoolData] = []
        for pair in data.get("pairs", []):
            try:
                result.append(self._parse_pair(pair))
            except Exception as exc:
                self._logger.warning("Failed to parse V2 pair %s: %s", pair.get("id", "?"), exc)
                continue
        return result

    async def get_pools_for_token(self, token_address: str, limit: int = 100) -> list[PoolData]:
        """Fetch Uniswap V2 pairs that contain a specific token.

        Args:
            token_address: ERC-20 token address.
            limit: Maximum number of pairs to return.

        Returns:
            A list of :class:`PoolData` objects.
        """
        query = """
        query PairsForToken($token: String!, $limit: Int!) {
            pairs(
                first: $limit
                orderBy: reserveUSD
                orderDirection: desc
                where: { or: [{ token0: $token }, { token1: $token }] }
            ) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                reserve0
                reserve1
            }
        }
        """
        data = await self._query(query, {"token": token_address.lower(), "limit": limit})
        result: list[PoolData] = []
        for pair in data.get("pairs", []):
            try:
                result.append(self._parse_pair(pair))
            except Exception as exc:
                self._logger.warning("Failed to parse V2 pair %s: %s", pair.get("id", "?"), exc)
                continue
        return result


class UniswapV3Subgraph(Subgraph):
    """Uniswap V3 subgraph client.

    Queries the Uniswap V3 subgraph for pool data including ``sqrtPrice``
    and ``liquidity``.  These values are used directly to build
    :class:`~pydefi.pathfinder.graph.V3PoolEdge` objects for concentrated
    liquidity price/amount estimation.

    Args:
        chain_id: EVM chain ID.
        url: Custom subgraph URL.  If not provided, the canonical Uniswap V3
            subgraph URL for *chain_id* is used (see
            :data:`_UNISWAP_V3_SUBGRAPHS`).

    Raises:
        :class:`ValueError`: If no default URL is available for *chain_id*
            and *url* is not supplied.
    """

    def __init__(self, chain_id: int, url: Optional[str] = None) -> None:
        resolved = url or _UNISWAP_V3_SUBGRAPHS.get(chain_id)
        if not resolved:
            raise ValueError(
                f"No default Uniswap V3 subgraph URL for chain_id {chain_id}. Provide a custom url parameter."
            )
        super().__init__(chain_id, resolved)

    @property
    def provider_name(self) -> str:
        return "UniswapV3Subgraph"

    def _parse_pool(self, pool: dict[str, Any]) -> PoolData:
        """Parse a Uniswap V3 pool entity from the subgraph."""
        t0 = pool["token0"]
        t1 = pool["token1"]

        token0 = Token(
            chain_id=self.chain_id,
            address=t0["id"],
            symbol=t0["symbol"],
            decimals=int(t0["decimals"]),
            name=t0.get("name") or None,
        )
        token1 = Token(
            chain_id=self.chain_id,
            address=t1["id"],
            symbol=t1["symbol"],
            decimals=int(t1["decimals"]),
            name=t1.get("name") or None,
        )

        # feeTier is in hundredths of a basis point (e.g. 3000 → 30 bps = 0.3%)
        fee_tier = int(pool.get("feeTier") or 3000)
        fee_bps = fee_tier // 100

        sqrt_price_x96 = int(pool.get("sqrtPrice") or 0)
        liquidity = int(pool.get("liquidity") or 0)

        return PoolData(
            pool_address=pool["id"],
            protocol="UniswapV3",
            chain_id=self.chain_id,
            token0=token0,
            token1=token1,
            fee_bps=fee_bps,
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
        )

    async def get_pool(self, pool_address: str) -> PoolData:
        """Fetch a Uniswap V3 pool by its on-chain address.

        Args:
            pool_address: Pool contract address (checksummed or lowercase).

        Returns:
            A :class:`PoolData` object.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If the pool is not
                found or the subgraph returns an error.
        """
        query = """
        query GetPool($id: ID!) {
            pool(id: $id) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                feeTier
                sqrtPrice
                liquidity
            }
        }
        """
        data = await self._query(query, {"id": pool_address.lower()})
        pool = data.get("pool")
        if not pool:
            raise PoolDataError(f"Pool {pool_address} not found in Uniswap V3 subgraph")
        return self._parse_pool(pool)

    async def get_top_pools(self, limit: int = 100) -> list[PoolData]:
        """Fetch the top Uniswap V3 pools ranked by total value locked (TVL).

        Args:
            limit: Maximum number of pools to return.

        Returns:
            A list of :class:`PoolData` objects.
        """
        query = """
        query TopPools($limit: Int!) {
            pools(
                first: $limit
                orderBy: totalValueLockedUSD
                orderDirection: desc
            ) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                feeTier
                sqrtPrice
                liquidity
            }
        }
        """
        data = await self._query(query, {"limit": limit})
        result: list[PoolData] = []
        for pool in data.get("pools", []):
            try:
                result.append(self._parse_pool(pool))
            except Exception as exc:
                self._logger.warning("Failed to parse V3 pool %s: %s", pool.get("id", "?"), exc)
                continue
        return result

    async def get_pools_for_token(self, token_address: str, limit: int = 100) -> list[PoolData]:
        """Fetch Uniswap V3 pools that contain a specific token.

        Args:
            token_address: ERC-20 token address.
            limit: Maximum number of pools to return.

        Returns:
            A list of :class:`PoolData` objects.
        """
        query = """
        query PoolsForToken($token: String!, $limit: Int!) {
            pools(
                first: $limit
                orderBy: totalValueLockedUSD
                orderDirection: desc
                where: { or: [{ token0: $token }, { token1: $token }] }
            ) {
                id
                token0 { id symbol decimals name }
                token1 { id symbol decimals name }
                feeTier
                sqrtPrice
                liquidity
            }
        }
        """
        data = await self._query(query, {"token": token_address.lower(), "limit": limit})
        result: list[PoolData] = []
        for pool in data.get("pools", []):
            try:
                result.append(self._parse_pool(pool))
            except Exception as exc:
                self._logger.warning("Failed to parse V3 pool %s: %s", pool.get("id", "?"), exc)
                continue
        return result

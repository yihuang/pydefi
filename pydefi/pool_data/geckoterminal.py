"""
GeckoTerminal pool data provider.

Fetches pool state from the GeckoTerminal public REST API v2.
Docs: https://www.geckoterminal.com/dex-api
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import aiohttp

from pydefi.exceptions import PoolDataError
from pydefi.pool_data.base import BasePoolDataProvider, PoolData
from pydefi.types import ChainId, Token

# GeckoTerminal network slugs keyed by EVM chain ID
_CHAIN_TO_NETWORK: dict[int, str] = {
    ChainId.ETHEREUM: "eth",
    ChainId.BSC: "bsc",
    ChainId.POLYGON: "polygon_pos",
    ChainId.ARBITRUM: "arbitrum",
    ChainId.OPTIMISM: "optimism",
    ChainId.AVALANCHE: "avax",
    ChainId.BASE: "base",
    ChainId.LINEA: "linea",
    ChainId.BLAST: "blast",
    ChainId.SCROLL: "scroll",
    ChainId.ZKSYNC: "zksync",
    ChainId.ZORA: "zora",
}

# Map GeckoTerminal dex_id values to human-readable protocol names
_DEX_TO_PROTOCOL: dict[str, str] = {
    "uniswap_v2": "UniswapV2",
    "uniswap_v3": "UniswapV3",
    "sushiswap": "SushiSwap",
    "curve": "Curve",
    "balancer": "Balancer",
    "pancakeswap_v2": "PancakeSwapV2",
    "pancakeswap_v3": "PancakeSwapV3",
    "aerodrome_v1": "Aerodrome",
    "velodrome_v2": "Velodrome",
}

# GeckoTerminal returns at most 20 pools per page
_PAGE_SIZE = 20
# Maximum number of token addresses per batch request
_MAX_ADDRESSES_PER_REQUEST = 10

logger = logging.getLogger(__name__)


class GeckoTerminal(BasePoolDataProvider):
    """GeckoTerminal pool data provider.

    Fetches pool data from the public GeckoTerminal REST API (v2).  No API
    key is required for basic usage.

    Because GeckoTerminal exposes USD-denominated reserve values rather than
    raw on-chain token amounts, reserves are *estimated* from
    ``reserve_in_usd`` and the per-token USD prices.  The estimates are
    accurate enough for pathfinding but should not be used as an authoritative
    source of liquidity depth.

    Args:
        chain_id: EVM chain ID to query.  Must be one of the supported chains
            listed in :data:`_CHAIN_TO_NETWORK`.
        base_url: Override the default API base URL (useful for testing).
    """

    _DEFAULT_BASE_URL = "https://api.geckoterminal.com/api/v2"

    def __init__(
        self,
        chain_id: int,
        base_url: Optional[str] = None,
    ) -> None:
        self.chain_id = chain_id
        self._base_url = base_url or self._DEFAULT_BASE_URL
        network = _CHAIN_TO_NETWORK.get(chain_id)
        if network is None:
            raise ValueError(
                f"Unsupported chain_id {chain_id} for GeckoTerminal. Supported chains: {list(_CHAIN_TO_NETWORK.keys())}"
            )
        self._network = network

    @property
    def provider_name(self) -> str:
        return "GeckoTerminal"

    def _headers(self) -> dict[str, str]:
        return {"Accept": "application/json;version=20230302"}

    async def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise PoolDataError(
                        f"GeckoTerminal API error {resp.status}: {data.get('errors', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_token(self, attrs: dict[str, Any], pool_id: str = "") -> Token:
        """Build a :class:`~pydefi.types.Token` from GeckoTerminal token attributes."""
        address = attrs.get("address")
        if not address:
            raise PoolDataError(
                f"Missing token address in pool {pool_id!r}: {attrs}",
                status_code=None,
            )
        decimals = int(attrs.get("decimals") or 18)
        return Token(
            chain_id=self.chain_id,
            address=address,
            symbol=attrs.get("symbol") or "???",
            decimals=decimals,
            name=attrs.get("name") or None,
        )

    def _extract_fee_bps(self, pool_name: str) -> int:
        """Try to parse the fee tier from a GeckoTerminal pool name string.

        Pool names typically look like ``"USDC / WETH 0.30%"``; this method
        extracts ``30`` from that string.  Returns ``30`` as a fallback when
        the fee cannot be determined.
        """
        if "%" in pool_name:
            try:
                pct_str = pool_name.rsplit("%", 1)[0].split()[-1]
                return round(float(pct_str) * 100)
            except (ValueError, IndexError):
                pass
        return 30

    def _estimate_reserve(
        self,
        reserve_in_usd: float,
        token_price_usd: float,
        token_decimals: int,
    ) -> int:
        """Estimate a raw on-chain token reserve from USD values.

        Assumes each token holds approximately half the pool's USD value.

        Args:
            reserve_in_usd: Total pool TVL in USD.
            token_price_usd: Price of the token in USD.
            token_decimals: ERC-20 decimals of the token.

        Returns:
            Raw integer reserve, or ``0`` if either price is zero.
        """
        if reserve_in_usd <= 0 or token_price_usd <= 0:
            return 0
        human_reserve = (reserve_in_usd / 2.0) / token_price_usd
        return int(human_reserve * 10**token_decimals)

    def _parse_pool(
        self,
        pool_item: dict[str, Any],
        included: list[dict[str, Any]],
    ) -> PoolData:
        """Parse a single pool item from a GeckoTerminal API response.

        Args:
            pool_item: The ``data`` object (or one element of ``data`` list).
            included: The ``included`` array from the same response.

        Returns:
            A :class:`PoolData` object.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: When required token
                metadata cannot be resolved from the ``included`` payload.
        """
        pool_id = pool_item.get("id", "?")
        attrs = pool_item.get("attributes", {})
        relationships = pool_item.get("relationships", {})

        # Build a lookup table from token id → token attributes
        token_map: dict[str, dict[str, Any]] = {}
        for item in included:
            if item.get("type") == "token":
                token_map[item["id"]] = item.get("attributes", {})

        base_token_id: str = relationships.get("base_token", {}).get("data", {}).get("id", "")
        quote_token_id: str = relationships.get("quote_token", {}).get("data", {}).get("id", "")

        if not base_token_id or base_token_id not in token_map:
            raise PoolDataError(
                f"Pool {pool_id!r}: base token {base_token_id!r} not found in included payload",
                status_code=None,
            )
        if not quote_token_id or quote_token_id not in token_map:
            raise PoolDataError(
                f"Pool {pool_id!r}: quote token {quote_token_id!r} not found in included payload",
                status_code=None,
            )

        base_attrs = token_map[base_token_id]
        quote_attrs = token_map[quote_token_id]

        token0 = self._parse_token(base_attrs, pool_id)
        token1 = self._parse_token(quote_attrs, pool_id)

        dex_id: str = relationships.get("dex", {}).get("data", {}).get("id", "")
        protocol = _DEX_TO_PROTOCOL.get(dex_id, dex_id)

        pool_name: str = attrs.get("name", "")
        fee_bps = self._extract_fee_bps(pool_name)

        reserve_in_usd = float(attrs.get("reserve_in_usd") or 0)
        base_price_usd = float(attrs.get("base_token_price_usd") or 0)
        quote_price_usd = float(attrs.get("quote_token_price_usd") or 0)

        reserve0 = self._estimate_reserve(reserve_in_usd, base_price_usd, token0.decimals)
        reserve1 = self._estimate_reserve(reserve_in_usd, quote_price_usd, token1.decimals)

        return PoolData(
            pool_address=attrs.get("address", ""),
            protocol=protocol,
            chain_id=self.chain_id,
            token0=token0,
            token1=token1,
            fee_bps=fee_bps,
            reserve0=reserve0,
            reserve1=reserve1,
            extra={
                "dex_id": dex_id,
                "reserve_in_usd": reserve_in_usd,
            },
        )

    # ------------------------------------------------------------------
    # BasePoolDataProvider interface
    # ------------------------------------------------------------------

    async def get_pool(self, pool_address: str) -> PoolData:
        """Fetch data for a single pool by its on-chain address.

        Args:
            pool_address: Pool contract address (checksummed or lowercase).

        Returns:
            A :class:`PoolData` object.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: On API errors.
        """
        data = await self._get(
            f"networks/{self._network}/pools/{pool_address.lower()}",
            params={"include": "base_token,quote_token,dex"},
        )
        return self._parse_pool(data["data"], data.get("included", []))

    async def get_top_pools(self, limit: int = 100) -> list[PoolData]:
        """Fetch top pools on this chain ranked by liquidity.

        Transparently paginates the GeckoTerminal API (20 pools per page) to
        collect up to *limit* results.

        Args:
            limit: Maximum number of pools to return.

        Returns:
            A list of up to *limit* :class:`PoolData` objects.
        """
        pools: list[PoolData] = []
        page = 1
        while len(pools) < limit:
            data = await self._get(
                f"networks/{self._network}/pools",
                params={
                    "include": "base_token,quote_token,dex",
                    "page": page,
                },
            )
            items = data.get("data", [])
            if not items:
                break
            included = data.get("included", [])
            for item in items:
                if len(pools) >= limit:
                    break
                try:
                    pools.append(self._parse_pool(item, included))
                except Exception as exc:
                    pool_addr = item.get("attributes", {}).get("address", "?")
                    logger.warning("Failed to parse pool %s: %s", pool_addr, exc)
                    continue
            if len(items) < _PAGE_SIZE:
                # Last page reached
                break
            page += 1
        return pools

    async def get_pools_for_tokens(self, token_addresses: list[str], limit: int = 100) -> list[PoolData]:
        """Fetch pools that contain any of the given tokens (batch query).

        Uses the GeckoTerminal ``/tokens/multi/{addresses}/pools`` endpoint to
        retrieve pools for multiple tokens in a single request per page.  This
        is more efficient than calling :meth:`get_pools_for_token` once per
        token and is particularly useful for building a
        :class:`~pydefi.pathfinder.graph.PoolGraph` that covers a set of
        *from/to* tokens plus well-known intermediate "hub" tokens (e.g. WETH,
        USDC) to enable multi-hop pathfinding.

        Input addresses are deduplicated (case-insensitive) and chunked into
        batches of up to :data:`_MAX_ADDRESSES_PER_REQUEST` (10) before being
        sent to the API.

        Args:
            token_addresses: List of ERC-20 token addresses to query.
            limit: Maximum total number of pools to return.

        Returns:
            A deduplicated list of up to *limit* :class:`PoolData` objects,
            sorted by descending liquidity as returned by the API.
        """
        if not token_addresses:
            return []

        # Deduplicate while preserving order
        seen_input: set[str] = set()
        unique_addresses: list[str] = []
        for addr in token_addresses:
            key = addr.lower()
            if key not in seen_input:
                seen_input.add(key)
                unique_addresses.append(addr.lower())

        # Chunk into batches of _MAX_ADDRESSES_PER_REQUEST
        chunks = [
            unique_addresses[i : i + _MAX_ADDRESSES_PER_REQUEST]
            for i in range(0, len(unique_addresses), _MAX_ADDRESSES_PER_REQUEST)
        ]

        pools: list[PoolData] = []
        seen_pool_addresses: set[str] = set()

        for chunk in chunks:
            if len(pools) >= limit:
                break
            addresses_param = ",".join(chunk)
            page = 1
            while len(pools) < limit:
                data = await self._get(
                    f"networks/{self._network}/tokens/multi/{addresses_param}/pools",
                    params={
                        "include": "base_token,quote_token,dex",
                        "page": page,
                    },
                )
                items = data.get("data", [])
                if not items:
                    break
                included = data.get("included", [])
                for item in items:
                    if len(pools) >= limit:
                        break
                    try:
                        pool = self._parse_pool(item, included)
                    except Exception as exc:
                        pool_addr = item.get("attributes", {}).get("address", "?")
                        logger.warning("Failed to parse pool %s: %s", pool_addr, exc)
                        continue
                    # Deduplicate by pool address (API may return duplicates when
                    # multiple queried tokens appear in the same pool)
                    if pool.pool_address not in seen_pool_addresses:
                        seen_pool_addresses.add(pool.pool_address)
                        pools.append(pool)
                if len(items) < _PAGE_SIZE:
                    break
                page += 1
        return pools

    async def get_pools_for_token(self, token_address: str, limit: int = 100) -> list[PoolData]:
        """Fetch pools that contain a specific token.

        Args:
            token_address: ERC-20 token address.
            limit: Maximum number of pools to return.

        Returns:
            A list of up to *limit* :class:`PoolData` objects.
        """
        pools: list[PoolData] = []
        page = 1
        while len(pools) < limit:
            data = await self._get(
                f"networks/{self._network}/tokens/{token_address.lower()}/pools",
                params={
                    "include": "base_token,quote_token,dex",
                    "page": page,
                },
            )
            items = data.get("data", [])
            if not items:
                break
            included = data.get("included", [])
            for item in items:
                if len(pools) >= limit:
                    break
                try:
                    pools.append(self._parse_pool(item, included))
                except Exception as exc:
                    pool_addr = item.get("attributes", {}).get("address", "?")
                    logger.warning("Failed to parse pool %s: %s", pool_addr, exc)
                    continue
            if len(items) < _PAGE_SIZE:
                break
            page += 1
        return pools

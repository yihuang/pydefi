"""Tests for pydefi.pool_data — PoolData, GeckoTerminal, and subgraph clients."""

from unittest.mock import AsyncMock, patch

import pytest

from pydefi.exceptions import PoolDataError
from pydefi.pathfinder.graph import PoolEdge, V3PoolEdge
from pydefi.pool_data.base import BasePoolDataProvider, PoolData
from pydefi.pool_data.geckoterminal import GeckoTerminal
from pydefi.pool_data.subgraph import UniswapV2Subgraph, UniswapV3Subgraph
from pydefi.types import Address, ChainId
from tests.addrs import DAI, USDC, WETH, ZERO_ADDR

# ---------------------------------------------------------------------------
# Shared test tokens
# ---------------------------------------------------------------------------


POOL_ADDR: Address = Address("0x" + "ab" * 20)


# ---------------------------------------------------------------------------
# PoolData tests
# ---------------------------------------------------------------------------


class TestPoolData:
    def test_to_pool_edges_v2(self):
        """V2-style PoolData produces plain PoolEdge objects."""
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            fee_bps=30,
            reserve0=1_000 * 10**18,
            reserve1=2_000_000 * 10**6,
        )
        edges = pool.to_pool_edges()
        assert len(edges) == 2
        # Both should be plain PoolEdge (not V3)
        for edge in edges:
            assert type(edge) is PoolEdge

        e0, e1 = edges
        assert e0.token_in == WETH
        assert e0.token_out == USDC
        assert e0.reserve_in == 1_000 * 10**18
        assert e0.reserve_out == 2_000_000 * 10**6

        assert e1.token_in == USDC
        assert e1.token_out == WETH
        assert e1.reserve_in == 2_000_000 * 10**6
        assert e1.reserve_out == 1_000 * 10**18

    def test_to_pool_edges_v3(self):
        """V3-style PoolData produces V3PoolEdge objects."""
        Q96 = 2**96
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV3",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            fee_bps=5,
            sqrt_price_x96=int(1e9 * Q96**0.5),  # arbitrary non-zero value
            liquidity=5 * 10**22,
        )
        edges = pool.to_pool_edges()
        assert len(edges) == 2
        for edge in edges:
            assert isinstance(edge, V3PoolEdge)

        e0, e1 = edges
        assert e0.token_in == WETH
        assert e0.is_token0_in is True
        assert e1.token_in == USDC
        assert e1.is_token0_in is False

    def test_to_pool_edges_zero_v3_fields_uses_v2(self):
        """If V3 fields are zero, plain PoolEdge is used even if reserves are set."""
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            sqrt_price_x96=0,
            liquidity=0,
            reserve0=10**21,
            reserve1=2 * 10**9,
        )
        edges = pool.to_pool_edges()
        assert all(type(e) is PoolEdge for e in edges)

    def test_to_pool_edges_fee_propagated(self):
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            fee_bps=5,
        )
        edges = pool.to_pool_edges()
        assert all(e.fee_bps == 5 for e in edges)

    def test_to_pool_edges_extra_copied(self):
        extra = {"foo": "bar"}
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            extra=extra,
        )
        edges = pool.to_pool_edges()
        assert edges[0].extra["foo"] == "bar"
        assert edges[0].extra["is_token0_in"] is True
        assert edges[1].extra["foo"] == "bar"
        assert edges[1].extra["is_token0_in"] is False
        for edge in edges:
            assert edge.extra is not extra  # must be a copy


# ---------------------------------------------------------------------------
# BasePoolDataProvider.build_graph tests
# ---------------------------------------------------------------------------


class ConcreteProvider(BasePoolDataProvider):
    """Minimal concrete implementation for testing build_graph."""

    @property
    def provider_name(self) -> str:
        return "test"

    async def get_pool(self, pool_address: str) -> PoolData:
        raise NotImplementedError

    async def get_top_pools(self, limit: int = 100) -> list[PoolData]:
        raise NotImplementedError

    async def get_pools_for_token(self, token_address: str, limit: int = 100) -> list[PoolData]:
        raise NotImplementedError


class TestBuildGraph:
    def test_build_graph_empty(self):
        provider = ConcreteProvider()
        graph = provider.build_graph([])
        assert len(graph) == 0

    def test_build_graph_single_v2_pool(self):
        provider = ConcreteProvider()
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            reserve0=10**21,
            reserve1=2 * 10**9,
        )
        graph = provider.build_graph([pool])
        assert len(graph) == 2  # bidirectional
        assert len(graph.edges_from(WETH)) == 1
        assert len(graph.edges_from(USDC)) == 1

    def test_build_graph_multiple_pools(self):
        provider = ConcreteProvider()
        POOL_ADDR2 = "0x" + "cd" * 20
        pools = [
            PoolData(
                pool_address=POOL_ADDR,
                protocol="UniswapV2",
                chain_id=ChainId.ETHEREUM,
                token0=WETH,
                token1=USDC,
                reserve0=10**21,
                reserve1=2 * 10**9,
            ),
            PoolData(
                pool_address=POOL_ADDR2,
                protocol="Curve",
                chain_id=ChainId.ETHEREUM,
                token0=USDC,
                token1=DAI,
                reserve0=10**9,
                reserve1=10**21,
                fee_bps=4,
            ),
        ]
        graph = provider.build_graph(pools)
        assert len(graph) == 4  # 2 pools × 2 directions
        assert len(graph.edges_from(USDC)) == 2  # USDC→WETH and USDC→DAI


# ---------------------------------------------------------------------------
# GeckoTerminal tests
# ---------------------------------------------------------------------------

_MOCK_POOL_RESPONSE = {
    "data": {
        "id": "eth_0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
        "type": "pool",
        "attributes": {
            "address": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
            "name": "USDT / WETH 0.30%",
            "reserve_in_usd": "200000000",
            "base_token_price_usd": "1.001",
            "quote_token_price_usd": "2000.0",
        },
        "relationships": {
            "base_token": {"data": {"id": "eth_0xdac17f958d2ee523a2206206994597c13d831ec7"}},
            "quote_token": {"data": {"id": "eth_0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"}},
            "dex": {"data": {"id": "uniswap_v2"}},
        },
    },
    "included": [
        {
            "id": "eth_0xdac17f958d2ee523a2206206994597c13d831ec7",
            "type": "token",
            "attributes": {
                "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "symbol": "USDT",
                "decimals": "6",
                "name": "Tether USD",
            },
        },
        {
            "id": "eth_0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "type": "token",
            "attributes": {
                "address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                "symbol": "WETH",
                "decimals": "18",
                "name": "Wrapped Ether",
            },
        },
    ],
}

_MOCK_LIST_RESPONSE = {
    "data": [_MOCK_POOL_RESPONSE["data"]],
    "included": _MOCK_POOL_RESPONSE["included"],
}


class TestGeckoTerminal:
    def test_provider_name(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        assert client.provider_name == "GeckoTerminal"

    def test_unsupported_chain_raises(self):
        with pytest.raises(ValueError, match="Unsupported chain_id"):
            GeckoTerminal(chain_id=99999)

    def test_network_mapping(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        assert client._network == "eth"

        client_base = GeckoTerminal(chain_id=ChainId.BASE)
        assert client_base._network == "base"

    def test_extract_fee_bps_normal(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        assert client._extract_fee_bps("USDC / WETH 0.30%") == 30
        assert client._extract_fee_bps("USDC / WETH 0.05%") == 5
        assert client._extract_fee_bps("USDC / WETH 1.00%") == 100

    def test_extract_fee_bps_fallback(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        assert client._extract_fee_bps("USDC / WETH") == 30

    def test_estimate_reserve_normal(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        # $200M TVL, token0 at $2000, 18 decimals
        # Expected: ($200M / 2) / $2000 * 1e18 = 50,000 WETH raw
        reserve = client._estimate_reserve(200_000_000.0, 2000.0, 18)
        expected = 50_000 * 10**18
        # Float arithmetic introduces tiny rounding; allow 1 wei tolerance
        assert abs(reserve - expected) <= expected * 1e-12

    def test_estimate_reserve_zero_price(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        assert client._estimate_reserve(1_000_000.0, 0.0, 18) == 0
        assert client._estimate_reserve(0.0, 2000.0, 18) == 0

    @pytest.mark.asyncio
    async def test_get_pool_success(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_get", new=AsyncMock(return_value=_MOCK_POOL_RESPONSE)):
            pool = await client.get_pool("0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852")

        assert pool.pool_address == "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852"
        assert pool.protocol == "UniswapV2"
        assert pool.token0.symbol == "USDT"
        assert pool.token1.symbol == "WETH"
        assert pool.fee_bps == 30
        assert pool.chain_id == ChainId.ETHEREUM
        assert pool.reserve0 > 0
        assert pool.reserve1 > 0

    @pytest.mark.asyncio
    async def test_get_pool_api_error(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(
            client,
            "_get",
            new=AsyncMock(side_effect=PoolDataError("Not found", status_code=404)),
        ):
            with pytest.raises(PoolDataError):
                await client.get_pool(ZERO_ADDR)

    @pytest.mark.asyncio
    async def test_get_top_pools_returns_list(self):
        # Single page, fewer than _PAGE_SIZE items → only one request
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_get", new=AsyncMock(return_value=_MOCK_LIST_RESPONSE)):
            pools = await client.get_top_pools(limit=5)

        assert isinstance(pools, list)
        assert len(pools) == 1
        assert pools[0].protocol == "UniswapV2"

    @pytest.mark.asyncio
    async def test_get_top_pools_respects_limit(self):
        """When one page has fewer than PAGE_SIZE items, pagination stops."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        mock_get = AsyncMock(return_value=_MOCK_LIST_RESPONSE)
        with patch.object(client, "_get", new=mock_get):
            await client.get_top_pools(limit=100)
        # Only one API call because the page has 1 item < _PAGE_SIZE
        assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_pools_for_token(self):
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_get", new=AsyncMock(return_value=_MOCK_LIST_RESPONSE)):
            pools = await client.get_pools_for_token(WETH.address, limit=5)

        assert len(pools) == 1

    @pytest.mark.asyncio
    async def test_get_pools_for_tokens_single_token(self):
        """Batch method with one token returns same results as single-token method."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_get", new=AsyncMock(return_value=_MOCK_LIST_RESPONSE)):
            pools = await client.get_pools_for_tokens([WETH.address], limit=5)

        assert len(pools) == 1
        assert pools[0].protocol == "UniswapV2"

    @pytest.mark.asyncio
    async def test_get_pools_for_tokens_multiple_tokens(self):
        """Batch method passes comma-joined addresses to the API."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        mock_get = AsyncMock(return_value=_MOCK_LIST_RESPONSE)
        with patch.object(client, "_get", new=mock_get):
            await client.get_pools_for_tokens([WETH.address, USDC.address], limit=5)

        # The endpoint path should contain both addresses comma-separated
        call_path = mock_get.call_args[0][0]
        assert ("0x" + WETH.address.hex()) in call_path
        assert ("0x" + USDC.address.hex()) in call_path
        assert "multi" in call_path

    @pytest.mark.asyncio
    async def test_get_pools_for_tokens_empty_list(self):
        """Empty address list returns immediately with no API calls."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        mock_get = AsyncMock(return_value=_MOCK_LIST_RESPONSE)
        with patch.object(client, "_get", new=mock_get):
            pools = await client.get_pools_for_tokens([], limit=5)

        assert pools == []
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_pools_for_tokens_deduplication(self):
        """Pools returned for overlapping tokens should be deduplicated."""
        # Same pool appearing twice in response data
        duplicate_response = {
            "data": [
                _MOCK_POOL_RESPONSE["data"],
                _MOCK_POOL_RESPONSE["data"],
            ],
            "included": _MOCK_POOL_RESPONSE["included"],
        }
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_get", new=AsyncMock(return_value=duplicate_response)):
            pools = await client.get_pools_for_tokens([WETH.address, USDC.address], limit=10)

        # Should deduplicate to 1 unique pool
        assert len(pools) == 1

    @pytest.mark.asyncio
    async def test_get_pools_for_tokens_deduplicates_input_addresses(self):
        """Duplicate input addresses produce a single API call, not two."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        mock_get = AsyncMock(return_value=_MOCK_LIST_RESPONSE)
        with patch.object(client, "_get", new=mock_get):
            # Pass the same address twice (mixed case)
            await client.get_pools_for_tokens([WETH.address, WETH.address], limit=5)

        # Only one unique address → one chunk → only one _get call per page
        assert mock_get.call_count == 1
        call_path = mock_get.call_args[0][0]
        # The comma-separated list should contain the address only once
        assert call_path.count("0x" + WETH.address.hex()) == 1

    def test_parse_pool_missing_token_raises_pool_data_error(self):
        """_parse_pool should raise PoolDataError when token is missing from included."""
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        pool_item = {
            "id": "eth_0xdeadbeef",
            "type": "pool",
            "attributes": {
                "address": "0xdeadbeef",
                "name": "FOO / BAR 0.30%",
                "reserve_in_usd": "1000",
                "base_token_price_usd": "1.0",
                "quote_token_price_usd": "1.0",
            },
            "relationships": {
                "base_token": {"data": {"id": "eth_0xaaaa"}},
                "quote_token": {"data": {"id": "eth_0xbbbb"}},
                "dex": {"data": {"id": "uniswap_v2"}},
            },
        }
        # No included tokens → should raise PoolDataError
        with pytest.raises(PoolDataError):
            client._parse_pool(pool_item, included=[])

    def test_build_graph_from_get_top_pools(self):
        """build_graph should produce a navigable PoolGraph."""
        pool = PoolData(
            pool_address=POOL_ADDR,
            protocol="UniswapV2",
            chain_id=ChainId.ETHEREUM,
            token0=WETH,
            token1=USDC,
            reserve0=1_000 * 10**18,
            reserve1=2_000_000 * 10**6,
        )
        client = GeckoTerminal(chain_id=ChainId.ETHEREUM)
        graph = client.build_graph([pool])
        assert len(graph.edges_from(WETH)) == 1
        assert len(graph.edges_from(USDC)) == 1


# ---------------------------------------------------------------------------
# UniswapV2Subgraph tests
# ---------------------------------------------------------------------------

_MOCK_V2_PAIR = {
    "id": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
    "token0": {
        "id": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "symbol": "WETH",
        "decimals": "18",
        "name": "Wrapped Ether",
    },
    "token1": {
        "id": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "symbol": "USDT",
        "decimals": "6",
        "name": "Tether USD",
    },
    "reserve0": "50000.000000000000000000",
    "reserve1": "100000000.000000",
}


class TestUniswapV2Subgraph:
    def test_provider_name(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        assert client.provider_name == "UniswapV2Subgraph"

    def test_no_default_url_raises(self):
        with pytest.raises(ValueError, match="No default Uniswap V2 subgraph URL"):
            UniswapV2Subgraph(chain_id=99999)

    def test_custom_url_accepted(self):
        client = UniswapV2Subgraph(chain_id=99999, url="https://custom.example.com/subgraphs/v2")
        assert client.url == "https://custom.example.com/subgraphs/v2"

    def test_parse_pair(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        pool = client._parse_pair(_MOCK_V2_PAIR)

        assert pool.protocol == "UniswapV2"
        assert pool.token0.symbol == "WETH"
        assert pool.token1.symbol == "USDT"
        assert pool.token0.decimals == 18
        assert pool.token1.decimals == 6
        assert pool.fee_bps == 30

        # 50 000 WETH in raw units
        assert pool.reserve0 == 50_000 * 10**18
        # 100 000 000 USDT in raw units
        assert pool.reserve1 == 100_000_000 * 10**6

    @pytest.mark.asyncio
    async def test_get_pool_success(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pair": _MOCK_V2_PAIR}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pool = await client.get_pool("0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852")

        assert pool.protocol == "UniswapV2"
        assert pool.token0.symbol == "WETH"

    @pytest.mark.asyncio
    async def test_get_pool_not_found_raises(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_query", new=AsyncMock(return_value={"pair": None})):
            with pytest.raises(PoolDataError, match="not found"):
                await client.get_pool(ZERO_ADDR)

    @pytest.mark.asyncio
    async def test_get_top_pools(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pairs": [_MOCK_V2_PAIR, _MOCK_V2_PAIR]}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pools = await client.get_top_pools(limit=10)

        assert len(pools) == 2
        assert all(p.protocol == "UniswapV2" for p in pools)

    @pytest.mark.asyncio
    async def test_get_pools_for_token(self):
        client = UniswapV2Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pairs": [_MOCK_V2_PAIR]}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pools = await client.get_pools_for_token(WETH.address, limit=10)

        assert len(pools) == 1
        assert pools[0].protocol == "UniswapV2"


# ---------------------------------------------------------------------------
# UniswapV3Subgraph tests
# ---------------------------------------------------------------------------

_Q96 = 2**96
_MOCK_V3_POOL = {
    "id": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    "token0": {
        "id": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "symbol": "USDC",
        "decimals": "6",
        "name": "USD Coin",
    },
    "token1": {
        "id": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "symbol": "WETH",
        "decimals": "18",
        "name": "Wrapped Ether",
    },
    "feeTier": "500",
    "sqrtPrice": str(int(2.236e13 * _Q96)),  # arbitrary non-zero
    "liquidity": str(5 * 10**22),
}


class TestUniswapV3Subgraph:
    def test_provider_name(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        assert client.provider_name == "UniswapV3Subgraph"

    def test_no_default_url_raises(self):
        with pytest.raises(ValueError, match="No default Uniswap V3 subgraph URL"):
            UniswapV3Subgraph(chain_id=99999)

    def test_custom_url_accepted(self):
        client = UniswapV3Subgraph(chain_id=99999, url="https://custom.example.com/subgraphs/v3")
        assert client.url == "https://custom.example.com/subgraphs/v3"

    def test_parse_pool_fee_tier(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        pool = client._parse_pool(_MOCK_V3_POOL)

        assert pool.protocol == "UniswapV3"
        assert pool.token0.symbol == "USDC"
        assert pool.token1.symbol == "WETH"
        # feeTier 500 → fee_bps = 500 // 100 = 5
        assert pool.fee_bps == 5
        assert pool.sqrt_price_x96 > 0
        assert pool.liquidity == 5 * 10**22

    def test_parse_pool_produces_v3_edges(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        pool = client._parse_pool(_MOCK_V3_POOL)
        edges = pool.to_pool_edges()
        assert all(isinstance(e, V3PoolEdge) for e in edges)

    @pytest.mark.asyncio
    async def test_get_pool_success(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pool": _MOCK_V3_POOL}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pool = await client.get_pool("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")

        assert pool.protocol == "UniswapV3"
        assert pool.fee_bps == 5

    @pytest.mark.asyncio
    async def test_get_pool_not_found_raises(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        with patch.object(client, "_query", new=AsyncMock(return_value={"pool": None})):
            with pytest.raises(PoolDataError, match="not found"):
                await client.get_pool(ZERO_ADDR)

    @pytest.mark.asyncio
    async def test_get_top_pools(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pools": [_MOCK_V3_POOL]}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pools = await client.get_top_pools(limit=5)

        assert len(pools) == 1
        assert pools[0].protocol == "UniswapV3"

    @pytest.mark.asyncio
    async def test_get_pools_for_token(self):
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        mock_data = {"pools": [_MOCK_V3_POOL]}
        with patch.object(client, "_query", new=AsyncMock(return_value=mock_data)):
            pools = await client.get_pools_for_token(WETH.address, limit=5)

        assert len(pools) == 1

    @pytest.mark.asyncio
    async def test_query_graphql_error_raises(self):
        """GraphQL errors in the response should raise PoolDataError."""
        client = UniswapV3Subgraph(chain_id=ChainId.ETHEREUM)
        with patch.object(
            client,
            "_query",
            new=AsyncMock(side_effect=PoolDataError("Subgraph GraphQL errors: [...]")),
        ):
            with pytest.raises(PoolDataError):
                await client.get_pool("0xdeadbeef")

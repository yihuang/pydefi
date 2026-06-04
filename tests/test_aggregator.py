"""Tests for pydefi.aggregator (no live HTTP calls)."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from hexbytes import HexBytes

from pydefi.abi.dex_aggregator import OKX_DEX_ROUTER
from pydefi.aggregator.base import AggregatorQuote
from pydefi.aggregator.okx import OKX
from pydefi.aggregator.okx_router_encoder import (
    RouterPathDescriptor,
    build_dag_swap_calldata,
    encode_edge_raw_data,
    route_dag_to_router_paths,
)
from pydefi.aggregator.oneinch import OneInch
from pydefi.aggregator.openocean import OpenOcean
from pydefi.aggregator.paraswap import ParaSwap
from pydefi.aggregator.uniswap import UniswapAPI
from pydefi.aggregator.zerox import ZeroX
from pydefi.exceptions import AggregatorError
from pydefi.pathfinder.graph import PoolEdge
from pydefi.types import Address, ChainId, RouteDAG, Token, TokenAmount
from tests.addrs import USDC, WETH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1inch tests
# ---------------------------------------------------------------------------


class TestOneInch:
    def test_protocol_name(self):
        client = OneInch(chain_id=1)
        assert client.protocol_name == "1inch"

    def test_chain_id_stored(self):
        client = OneInch(chain_id=137)
        assert client.chain_id == 137

    def test_base_url_default(self):
        client = OneInch(chain_id=1)
        assert "1inch" in client.base_url

    def test_base_url_custom(self):
        client = OneInch(chain_id=1, base_url="https://custom.api.example.com")
        assert client.base_url == "https://custom.api.example.com"

    def test_chain_url(self):
        client = OneInch(chain_id=1)
        url = client._chain_url("quote")
        assert "/1/" in url
        assert "quote" in url

    def test_headers_no_api_key(self):
        client = OneInch(chain_id=1)
        headers = client._headers()
        assert "Accept" in headers
        assert "Authorization" not in headers

    def test_headers_with_api_key(self):
        client = OneInch(chain_id=1, api_key="mykey")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer mykey"

    def test_slippage_to_percent(self):
        client = OneInch(chain_id=1)
        assert client._slippage_to_percent(50) == 0.5
        assert client._slippage_to_percent(100) == 1.0

    def test_slippage_to_fraction(self):
        client = OneInch(chain_id=1)
        assert client._slippage_to_fraction(50) == 0.005

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = OneInch(chain_id=1)
        mock_response_data = {
            "dstAmount": "2000000000",  # 2000 USDC
            "gas": 150000,
            "estimatedPriceImpact": "0.1",
            "protocols": [["UNISWAP_V2"]],
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_response_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_000_000_000
        assert quote.gas_estimate == 150_000
        assert quote.protocol == "1inch"
        # min_amount_out = 2000 USDC * (1 - 0.5%) = 1990 USDC
        assert quote.min_amount_out.amount == 2_000_000_000 * 9_950 // 10_000

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        client = OneInch(chain_id=1)
        with patch.object(client, "_get", new=AsyncMock(side_effect=AggregatorError("API error", 400))):
            amount_in = TokenAmount.from_human(WETH, "1")
            with pytest.raises(AggregatorError):
                await client.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        client = OneInch(chain_id=1)
        mock_data = {
            "dstAmount": "2000000000",
            "gas": 150000,
            "estimatedPriceImpact": "0.05",
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            route = await client.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "1inch"


# ---------------------------------------------------------------------------
# ParaSwap tests
# ---------------------------------------------------------------------------


class TestParaSwap:
    def test_protocol_name(self):
        client = ParaSwap(chain_id=1)
        assert client.protocol_name == "ParaSwap"

    def test_base_url(self):
        client = ParaSwap(chain_id=1)
        assert "paraswap" in client.base_url.lower()

    def test_headers_with_api_key(self):
        client = ParaSwap(chain_id=1, api_key="secret")
        assert client._headers()["x-api-key"] == "secret"

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = ParaSwap(chain_id=1)
        mock_price_data = {
            "priceRoute": {
                "destAmount": "1995000000",
                "gasCost": 200000,
                "percentChange": "0.25",
                "bestRoute": [{"swaps": []}],
            }
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_price_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=100)

        assert quote.amount_out.amount == 1_995_000_000
        assert quote.gas_estimate == 200_000
        assert quote.protocol == "ParaSwap"


# ---------------------------------------------------------------------------
# 0x tests
# ---------------------------------------------------------------------------


class TestZeroX:
    def test_protocol_name(self):
        client = ZeroX(chain_id=1)
        assert client.protocol_name == "0x"

    def test_chain_specific_url_ethereum(self):
        client = ZeroX(chain_id=1)
        assert client.base_url == "https://api.0x.org"

    def test_chain_specific_url_arbitrum(self):
        client = ZeroX(chain_id=42161)
        assert client.base_url == "https://arbitrum.api.0x.org"

    def test_chain_specific_url_custom(self):
        client = ZeroX(chain_id=1, base_url="https://my-0x.example.com")
        assert client.base_url == "https://my-0x.example.com"

    def test_headers_with_api_key(self):
        client = ZeroX(chain_id=1, api_key="abc123")
        assert client._headers()["0x-api-key"] == "abc123"

    def test_headers_without_api_key(self):
        client = ZeroX(chain_id=1)
        headers = client._headers()
        assert "0x-api-key" not in headers

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = ZeroX(chain_id=1)
        mock_data = {
            "buyAmount": "2001000000",
            "estimatedGas": 180000,
            "estimatedPriceImpact": "0.05",
            "to": "0x" + "EF" * 20,
            "data": "0xdeadbeef",
            "value": "0",
            "gasPrice": "10000000000",
            "sources": [{"name": "Uniswap_V3", "proportion": "1"}],
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_001_000_000
        assert quote.protocol == "0x"
        assert "to" in quote.tx_data
        assert "data" in quote.tx_data

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        client = ZeroX(chain_id=1)
        mock_data = {
            "buyAmount": "2001000000",
            "estimatedGas": 180000,
            "estimatedPriceImpact": "0.05",
            "to": "0x" + "EF" * 20,
            "data": "0xdeadbeef",
            "value": "0",
            "gasPrice": "10000000000",
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            route = await client.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC


# ---------------------------------------------------------------------------
# Uniswap Trading API tests
# ---------------------------------------------------------------------------


class TestUniswapAPI:
    def test_protocol_name(self):
        client = UniswapAPI(chain_id=1)
        assert client.protocol_name == "Uniswap"

    def test_chain_id_stored(self):
        client = UniswapAPI(chain_id=137)
        assert client.chain_id == 137

    def test_base_url_default(self):
        client = UniswapAPI(chain_id=1)
        assert client.base_url == "https://trade-api.gateway.uniswap.org"

    def test_base_url_custom(self):
        client = UniswapAPI(chain_id=1, base_url="https://custom.api.example.com")
        assert client.base_url == "https://custom.api.example.com"

    def test_headers_no_api_key(self):
        client = UniswapAPI(chain_id=1)
        headers = client._headers()
        assert "Content-Type" in headers
        assert "x-api-key" not in headers
        assert "Authorization" not in headers
        assert "Origin" not in headers

    def test_headers_with_api_key(self):
        client = UniswapAPI(chain_id=1, api_key="mykey")
        headers = client._headers()
        assert headers["x-api-key"] == "mykey"
        assert "Authorization" not in headers

    def test_headers_with_origin(self):
        client = UniswapAPI(chain_id=1, api_key="mykey", origin="https://app.uniswap.org")
        headers = client._headers()
        assert headers["Origin"] == "https://app.uniswap.org"

    def test_slippage_to_percent(self):
        client = UniswapAPI(chain_id=1)
        assert client._slippage_to_percent(50) == 0.5
        assert client._slippage_to_percent(100) == 1.0

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = UniswapAPI(chain_id=1)
        mock_response = {
            "routing": "CLASSIC",
            "quote": {
                "output": {"amount": "2000000000"},
                "gasFee": "180000",
                "priceImpact": 0.05,
                "routeString": "WETH -> USDC",
            },
        }
        with patch.object(client, "_post", new=AsyncMock(return_value=mock_response)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_000_000_000
        assert quote.gas_estimate == 180_000
        assert quote.protocol == "Uniswap"
        # min_amount_out = 2000 USDC * (1 - 0.5%) = 1990 USDC
        assert quote.min_amount_out.amount == 2_000_000_000 * 9_950 // 10_000
        assert "quoteData" in quote.tx_data

    @pytest.mark.asyncio
    async def test_get_quote_dutch_v2(self):
        """get_quote() correctly parses UniswapX DUTCH_V2 responses."""
        client = UniswapAPI(chain_id=1)
        mock_response = {
            "routing": "DUTCH_V2",
            "quote": {
                "encodedOrder": "0xdeadbeef",
                "aggregatedOutputs": [
                    {
                        "amount": "2052000000",
                        "minAmount": "2041000000",
                        "token": USDC.address,
                        "recipient": "0x" + "AA" * 20,
                        "bps": 10000,
                    }
                ],
                "orderInfo": {
                    "outputs": [
                        {
                            "token": USDC.address,
                            "startAmount": "2052000000",
                            "endAmount": "2041000000",
                            "recipient": "0x" + "AA" * 20,
                        }
                    ],
                    "input": {"token": WETH.address, "startAmount": str(10**18), "endAmount": str(10**18)},
                },
            },
        }
        with patch.object(client, "_post", new=AsyncMock(return_value=mock_response)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_052_000_000
        # min_amount_out comes from aggregatedOutputs[0].minAmount
        assert quote.min_amount_out.amount == 2_041_000_000
        assert quote.protocol == "Uniswap"

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        client = UniswapAPI(chain_id=1)
        with patch.object(client, "_post", new=AsyncMock(side_effect=AggregatorError("API error", 400))):
            amount_in = TokenAmount.from_human(WETH, "1")
            with pytest.raises(AggregatorError):
                await client.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_get_swap_success(self):
        client = UniswapAPI(chain_id=1)
        mock_quote_response = {
            "quote": {
                "output": {"amount": "1998000000"},
                "gasFee": "200000",
                "priceImpact": 0.1,
                "routeString": "WETH -> USDC",
            }
        }
        mock_swap_response = {
            "swap": {
                "to": "0x" + "AB" * 20,
                "data": "0xdeadbeef",
                "value": "0",
                "gasLimit": "200000",
            }
        }
        # get_swap() calls _post twice: once for /v1/quote, once for /v1/swap
        with patch.object(
            client,
            "_post",
            new=AsyncMock(side_effect=[mock_quote_response, mock_swap_response]),
        ):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_swap(amount_in, USDC, wallet_address=Address("0x" + "AA" * 20), slippage_bps=50)

        assert quote.amount_out.amount == 1_998_000_000
        assert quote.protocol == "Uniswap"
        assert quote.tx_data.get("to") == "0x" + "AB" * 20
        assert "data" in quote.tx_data

    @pytest.mark.asyncio
    async def test_get_swap_with_deadline(self):
        client = UniswapAPI(chain_id=1)
        mock_quote_response = {
            "quote": {
                "output": {"amount": "1998000000"},
                "gasFee": "0",
                "priceImpact": 0,
                "routeString": "",
            }
        }
        mock_swap_response = {
            "swap": {
                "to": "0x" + "AB" * 20,
                "data": "0xcafe",
                "value": "0",
            }
        }
        captured_bodies: list[dict] = []

        async def mock_post(endpoint: str, json_body: dict) -> dict:
            captured_bodies.append((endpoint, dict(json_body)))
            if "quote" in endpoint:
                return mock_quote_response
            return mock_swap_response

        with patch.object(client, "_post", new=AsyncMock(side_effect=mock_post)):
            amount_in = TokenAmount.from_human(WETH, "1")
            await client.get_swap(
                amount_in,
                USDC,
                wallet_address=Address("0x" + "AA" * 20),
                slippage_bps=50,
                deadline=9999999999,
            )

        # Second call is to /v1/swap and must carry the deadline
        swap_endpoint, swap_body = captured_bodies[1]
        assert "swap" in swap_endpoint
        assert swap_body["deadline"] == 9999999999

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        client = UniswapAPI(chain_id=1)
        mock_response = {
            "quote": {
                "output": {"amount": "2000000000"},
                "gasFee": "180000",
                "priceImpact": 0.05,
                "routeString": "WETH -> USDC",
            }
        }
        with patch.object(client, "_post", new=AsyncMock(return_value=mock_response)):
            amount_in = TokenAmount.from_human(WETH, "1")
            route = await client.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Uniswap"

    @pytest.mark.asyncio
    async def test_get_swap_raises_on_uniswapx_routing(self):
        """get_swap() must raise AggregatorError when the quote routing is UniswapX."""
        client = UniswapAPI(chain_id=1)
        # DUTCH_V2 routing cannot be submitted to /v1/swap (requires /v1/order)
        mock_dutch_quote_response = {
            "routing": "DUTCH_V2",
            "quote": {
                "encodedOrder": "0xdeadbeef",
                "orderId": "0x123",
                "orderInfo": {},
            },
            "permitData": None,
        }
        with patch.object(
            client,
            "_post",
            new=AsyncMock(return_value=mock_dutch_quote_response),
        ):
            amount_in = TokenAmount.from_human(WETH, "1")
            with pytest.raises(AggregatorError, match=r"routing type 'DUTCH_V2' cannot be submitted via /v1/swap"):
                await client.get_swap(amount_in, USDC, wallet_address=Address("0x" + "AA" * 20), slippage_bps=50)


# ---------------------------------------------------------------------------
# AggregatorQuote tests
# ---------------------------------------------------------------------------

# OKX tests


class TestOKX:
    def test_protocol_name(self):
        client = OKX(chain_id=1)
        assert client.protocol_name == "OKX"

    def test_chain_id_stored(self):
        client = OKX(chain_id=137)
        assert client.chain_id == 137

    def test_base_url_default(self):
        client = OKX(chain_id=1)
        assert "okx" in client.base_url.lower()

    def test_base_url_custom(self):
        client = OKX(chain_id=1, base_url="https://custom.okx.example.com")
        assert client.base_url == "https://custom.okx.example.com"

    def test_headers_no_api_key(self):
        client = OKX(chain_id=1)
        headers = client._headers()
        assert "OK-ACCESS-KEY" not in headers

    def test_headers_with_api_key(self):
        client = OKX(chain_id=1, api_key="mykey")
        headers = client._headers()
        assert headers["OK-ACCESS-KEY"] == "mykey"

    def test_slippage_to_percent(self):
        client = OKX(chain_id=1)
        assert client._slippage_to_percent(50) == 0.5
        assert client._slippage_to_percent(100) == 1.0

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = OKX(chain_id=1)
        mock_response_data = {
            "code": "0",
            "data": {
                "fromTokenAmount": "1000000000000000000",
                "toTokenAmount": "2000000000",
                "estimateGasFee": 160000,
                "priceImpactPercentage": "0.1",
                "routerResult": {},
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_response_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_000_000_000
        assert quote.gas_estimate == 160_000
        assert quote.protocol == "OKX"
        assert quote.min_amount_out.amount == 2_000_000_000 * 9_950 // 10_000

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        client = OKX(chain_id=1)
        with patch.object(client, "_get", new=AsyncMock(side_effect=AggregatorError("API error", 400))):
            amount_in = TokenAmount.from_human(WETH, "1")
            with pytest.raises(AggregatorError):
                await client.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_get_swap_success(self):
        client = OKX(chain_id=1)
        mock_response_data = {
            "code": "0",
            "data": {
                "routerResult": {"toTokenAmount": "1998000000"},
                "tx": {
                    "to": "0x" + "AB" * 20,
                    "data": "0xdeadbeef",
                    "value": "0",
                    "gas": 180000,
                    "gasPrice": "10000000000",
                },
                "priceImpactPercentage": "0.05",
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_response_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            quote = await client.get_swap(amount_in, USDC, from_address=Address("0x" + "AA" * 20))

        assert quote.amount_out.amount == 1_998_000_000
        assert quote.tx_data["to"] == "0x" + "AB" * 20
        assert "data" in quote.tx_data

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        client = OKX(chain_id=1)
        mock_data = {
            "code": "0",
            "data": {
                "fromTokenAmount": "1000000000000000000",
                "toTokenAmount": "2000000000",
                "estimateGasFee": 160000,
                "priceImpactPercentage": "0.05",
                "routerResult": {},
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            amount_in = TokenAmount.from_human(WETH, "1")
            route = await client.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "OKX"


# ---------------------------------------------------------------------------
# OpenOcean tests
# ---------------------------------------------------------------------------


class TestOpenOcean:
    def test_protocol_name(self):
        client = OpenOcean(chain_id=1)
        assert client.protocol_name == "OpenOcean"

    def test_chain_id_stored(self):
        client = OpenOcean(chain_id=137)
        assert client.chain_id == 137

    def test_base_url_default(self):
        client = OpenOcean(chain_id=1)
        assert "openocean" in client.base_url.lower()

    def test_base_url_custom(self):
        client = OpenOcean(chain_id=1, base_url="https://custom.openocean.example.com")
        assert client.base_url == "https://custom.openocean.example.com"

    def test_chain_slug_known(self):
        assert OpenOcean(chain_id=1).chain_slug == "eth"
        assert OpenOcean(chain_id=137).chain_slug == "polygon"
        assert OpenOcean(chain_id=42161).chain_slug == "arbitrum"
        assert OpenOcean(chain_id=8453).chain_slug == "base"

    def test_chain_slug_unknown(self):
        client = OpenOcean(chain_id=99999)
        assert client.chain_slug == "99999"

    def test_chain_url(self):
        client = OpenOcean(chain_id=1)
        url = client._chain_url("quote")
        assert "/eth/" in url
        assert "quote" in url

    def test_slippage_to_percent(self):
        client = OpenOcean(chain_id=1)
        assert client._slippage_to_percent(50) == 0.5
        assert client._slippage_to_percent(100) == 1.0

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        client = OpenOcean(chain_id=1)
        mock_response_data = {
            "code": "200",
            "data": {
                "inAmount": "1000000000000000000",
                "outAmount": "2003000000",
                "estimatedGas": 170000,
                "price_impact": "0.08",
                "path": [],
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_response_data)):
            with patch.object(client, "_get_gas_price", new=AsyncMock(return_value="20000000000")):
                amount_in = TokenAmount.from_human(WETH, "1")
                quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount == 2_003_000_000
        assert quote.gas_estimate == 170_000
        assert quote.protocol == "OpenOcean"
        assert quote.min_amount_out.amount == 2_003_000_000 * 9_950 // 10_000

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        client = OpenOcean(chain_id=1)
        with patch.object(client, "_get", new=AsyncMock(side_effect=AggregatorError("API error", 400))):
            with patch.object(client, "_get_gas_price", new=AsyncMock(return_value="20000000000")):
                amount_in = TokenAmount.from_human(WETH, "1")
                with pytest.raises(AggregatorError):
                    await client.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_get_swap_success(self):
        client = OpenOcean(chain_id=1)
        mock_response_data = {
            "code": "200",
            "data": {
                "inAmount": "1000000000000000000",
                "outAmount": "2003000000",
                "estimatedGas": 170000,
                "price_impact": "0.08",
                "to": "0x" + "CD" * 20,
                "data": "0xcafebabe",
                "value": "0",
                "gasPrice": "15000000000",
                "path": [],
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_response_data)):
            with patch.object(client, "_get_gas_price", new=AsyncMock(return_value="20000000000")):
                amount_in = TokenAmount.from_human(WETH, "1")
                quote = await client.get_swap(amount_in, USDC, from_address=Address("0x" + "AA" * 20))

        assert quote.amount_out.amount == 2_003_000_000
        assert quote.tx_data["to"] == "0x" + "CD" * 20
        assert "data" in quote.tx_data

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        client = OpenOcean(chain_id=1)
        mock_data = {
            "code": "200",
            "data": {
                "inAmount": "1000000000000000000",
                "outAmount": "2003000000",
                "estimatedGas": 170000,
                "price_impact": "0.08",
                "path": [],
            },
        }
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            with patch.object(client, "_get_gas_price", new=AsyncMock(return_value="20000000000")):
                amount_in = TokenAmount.from_human(WETH, "1")
                route = await client.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "OpenOcean"


# ---------------------------------------------------------------------------
# OKX DexRouter encoder tests
# ---------------------------------------------------------------------------


class TestOKXRouterEncoder:
    """Tests for the OKX DexRouter calldata encoder."""

    def test_encode_edge_raw_data_defaults(self):

        pool = HexBytes("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")
        raw = encode_edge_raw_data(pool)

        # Defaults: weight=10000, input_index=0, output_index=1, reverse=False

        # Verify extraction with contract masks
        _ADDRESS_MASK = 0x000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        _WEIGHT_MASK = 0x00000000000000000000FFFF0000000000000000000000000000000000000000
        _OUTPUT_INDEX_MASK = 0x000000000000000000FF00000000000000000000000000000000000000000000
        _INPUT_INDEX_MASK = 0x0000000000000000FF0000000000000000000000000000000000000000000000
        _REVERSE_MASK = 0x8000000000000000000000000000000000000000000000000000000000000000

        assert (raw & _ADDRESS_MASK) == int.from_bytes(pool, "big")
        assert (raw & _WEIGHT_MASK) >> 160 == 10000
        assert (raw & _INPUT_INDEX_MASK) >> 184 == 0
        assert (raw & _OUTPUT_INDEX_MASK) >> 176 == 1
        assert not bool(raw & _REVERSE_MASK)

    def test_encode_edge_raw_data_reverse(self):

        _REVERSE_MASK = 0x8000000000000000000000000000000000000000000000000000000000000000
        pool = HexBytes("0x" + "FF" * 20)
        raw = encode_edge_raw_data(pool, reverse=True, weight_bps=5000, input_index=2, output_index=3)

        assert bool(raw & _REVERSE_MASK)
        _WEIGHT_MASK = 0x00000000000000000000FFFF0000000000000000000000000000000000000000
        assert (raw & _WEIGHT_MASK) >> 160 == 5000
        _INPUT_INDEX_MASK = 0x0000000000000000FF0000000000000000000000000000000000000000000000
        _OUTPUT_INDEX_MASK = 0x000000000000000000FF00000000000000000000000000000000000000000000
        assert (raw & _INPUT_INDEX_MASK) >> 184 == 2
        assert (raw & _OUTPUT_INDEX_MASK) >> 176 == 3

    def test_encode_edge_raw_data_validation(self):

        pool = HexBytes("0x" + "00" * 20)

        with pytest.raises(ValueError):
            encode_edge_raw_data(pool, weight_bps=10001)
        with pytest.raises(ValueError):
            encode_edge_raw_data(pool, weight_bps=-1)
        with pytest.raises(ValueError):
            encode_edge_raw_data(pool, input_index=256)
        with pytest.raises(ValueError):
            encode_edge_raw_data(pool, output_index=256)
        with pytest.raises(ValueError):
            encode_edge_raw_data(HexBytes("0x" + "00" * 10))

    def test_build_dag_swap_calldata(self):

        pool = HexBytes("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640")
        from_token = HexBytes("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        to_token = HexBytes("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
        receiver = HexBytes("0x" + "AB" * 20)
        adapter = HexBytes("0x" + "99" * 20)

        raw_data = encode_edge_raw_data(pool, weight_bps=10000)
        calldata = build_dag_swap_calldata(
            order_id=42,
            receiver=receiver,
            from_token=from_token,
            to_token=to_token,
            amount=10**18,
            min_return=9 * 10**8,
            deadline=2_000_000_000,
            paths=[
                RouterPathDescriptor(
                    mix_adapters=[adapter],
                    asset_to=[pool],
                    raw_data=[raw_data],
                    from_token=from_token,
                ),
            ],
        )

        # Verify selector matches the ABI definition
        expected_selector = OKX_DEX_ROUTER.fns.dagSwapTo.selector
        assert calldata[:4] == expected_selector
        assert len(calldata) > 4 + 128  # at minimum: selector + 4 arg words

    def test_build_dag_swap_calldata_empty_paths_raises(self):

        with pytest.raises(ValueError, match="at least one"):
            build_dag_swap_calldata(
                order_id=0,
                receiver=HexBytes("0x" + "00" * 20),
                from_token=HexBytes("0x" + "00" * 20),
                to_token=HexBytes("0x" + "00" * 20),
                amount=1,
                min_return=0,
                deadline=9999999999,
                paths=[],
            )

    def test_route_dag_to_router_paths_linear(self):
        """A simple linear RouteDAG (two swaps) produces two RouterPath nodes."""

        t0 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "01" * 20), symbol="T0")
        t1 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "02" * 20), symbol="T1")
        t2 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "03" * 20), symbol="T2")

        pool_a = HexBytes("0x" + "0A" * 20)
        pool_b = HexBytes("0x" + "0B" * 20)

        edge_a = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=pool_a,
            protocol="UniswapV2",
            fee_bps=30,
            extra={"is_token0_in": True},
        )
        edge_b = PoolEdge(
            token_in=t1,
            token_out=t2,
            pool_address=pool_b,
            protocol="UniswapV2",
            fee_bps=30,
            extra={"is_token0_in": True},
        )

        dag = RouteDAG().from_token(t0).swap(t1, edge_a).swap(t2, edge_b)
        paths = route_dag_to_router_paths(dag)

        assert len(paths) == 2
        # First node
        assert paths[0].from_token == t0.address
        assert len(paths[0].mix_adapters) == 1
        assert paths[0].raw_data[0] & 0xFF == pool_a[19]  # part of address in raw
        # Second node
        assert paths[1].from_token == t1.address
        assert len(paths[1].mix_adapters) == 1

    def test_route_dag_to_router_paths_with_split(self):
        """A split RouteDAG (one split with two legs) produces one split node + one subsequent node."""

        t0 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "01" * 20), symbol="T0")
        t1 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "02" * 20), symbol="T1")

        pool_a = HexBytes("0x" + "0A" * 20)
        pool_b = HexBytes("0x" + "0B" * 20)

        edge_a = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=pool_a,
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )
        edge_b = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=pool_b,
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )

        dag = RouteDAG().from_token(t0).split().leg(5000).swap(t1, edge_a).leg(5000).swap(t1, edge_b).merge()
        paths = route_dag_to_router_paths(dag)

        assert len(paths) == 1
        # Split node
        node = paths[0]
        assert len(node.mix_adapters) == 2
        assert len(node.raw_data) == 2
        assert node.from_token == t0.address

    def test_route_dag_rejects_multi_hop_leg(self):
        """A leg with more than one swap inside a split must raise."""

        t0 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "01" * 20), symbol="T0")
        t1 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "02" * 20), symbol="T1")
        t2 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "03" * 20), symbol="T2")

        pool_a = HexBytes("0x" + "0A" * 20)
        pool_b = HexBytes("0x" + "0B" * 20)

        edge_a = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=pool_a,
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )
        edge_b = PoolEdge(
            token_in=t1,
            token_out=t2,
            pool_address=pool_b,
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )
        edge_t0_to_t2 = PoolEdge(
            token_in=t0,
            token_out=t2,
            pool_address=pool_b,
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )

        dag = (
            RouteDAG()
            .from_token(t0)
            .split()
            .leg(5000)
            .swap(t1, edge_a)
            .swap(t2, edge_b)  # second swap — multi-hop
            .leg(5000)
            .swap(t2, edge_t0_to_t2)  # single swap, same end token
            .merge()
        )

        with pytest.raises(ValueError, match="exactly one RouteSwap"):
            route_dag_to_router_paths(dag)

    def test_route_dag_rejects_unknown_protocol(self):
        """A RouteSwap with an unsupported protocol raises ValueError."""

        t0 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "01" * 20), symbol="T0")
        t1 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "02" * 20), symbol="T1")

        edge = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=HexBytes("0x" + "0C" * 20),
            protocol="Curve",  # no adapter registered
            extra={"is_token0_in": True},
        )

        dag = RouteDAG().from_token(t0).swap(t1, edge)

        with pytest.raises(ValueError, match="adapter"):
            route_dag_to_router_paths(dag)

    def test_adapter_overrides_work(self):
        """Custom adapter addresses via adapter_overrides."""
        custom_addr = HexBytes("0x" + "CA" * 20)

        t0 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "01" * 20), symbol="T0")
        t1 = Token(chain_id=ChainId.ETHEREUM, address=HexBytes("0x" + "02" * 20), symbol="T1")

        edge = PoolEdge(
            token_in=t0,
            token_out=t1,
            pool_address=HexBytes("0x" + "0A" * 20),
            protocol="UniswapV2",
            extra={"is_token0_in": True},
        )

        dag = RouteDAG().from_token(t0).swap(t1, edge)
        paths = route_dag_to_router_paths(dag, adapter_overrides={"uniswap_v2": custom_addr})

        assert paths[0].mix_adapters[0] == custom_addr


class TestAggregatorQuote:
    def test_creation(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        amount_out = TokenAmount.from_human(USDC, "2000")
        min_out = TokenAmount.from_human(USDC, "1990")

        quote = AggregatorQuote(
            token_in=WETH,
            token_out=USDC,
            amount_in=amount_in,
            amount_out=amount_out,
            min_amount_out=min_out,
            gas_estimate=150_000,
            price_impact=Decimal("0.001"),
            protocol="1inch",
        )

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.gas_estimate == 150_000
        assert quote.protocol == "1inch"

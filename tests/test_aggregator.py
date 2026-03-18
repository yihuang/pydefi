"""Tests for pydifi.aggregator (no live HTTP calls)."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydifi.aggregator.base import AggregatorQuote, BaseAggregator
from pydifi.aggregator.oneinch import OneInch
from pydifi.aggregator.paraswap import ParaSwap
from pydifi.aggregator.zerox import ZeroX
from pydifi.exceptions import AggregatorError
from pydifi.types import ChainId, Token, TokenAmount


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WETH = Token(chain_id=ChainId.ETHEREUM, address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", symbol="WETH", decimals=18)
USDC = Token(chain_id=ChainId.ETHEREUM, address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", symbol="USDC", decimals=6)


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
# AggregatorQuote tests
# ---------------------------------------------------------------------------

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

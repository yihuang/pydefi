"""Tests for Solana integrations: Jupiter aggregator and Raydium AMM."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from pydefi._utils import decode_address
from pydefi.aggregator.base import AggregatorQuote
from pydefi.aggregator.jupiter import _JUPITER_API_BASE, _JUPITER_SWAP_V2_BASE, Jupiter, JupiterSwapV2
from pydefi.amm.base import BaseSolanaAMM
from pydefi.amm.raydium import _RAYDIUM_API_BASE, Raydium
from pydefi.exceptions import AggregatorError, InsufficientLiquidityError
from pydefi.types import ChainId, SwapRoute, Token, TokenAmount

# ---------------------------------------------------------------------------
# Solana token fixtures
# ---------------------------------------------------------------------------

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

SOL = Token(chain_id=ChainId.SOLANA, address=decode_address(SOL_MINT, ChainId.SOLANA), symbol="SOL", decimals=9)
USDC = Token(chain_id=ChainId.SOLANA, address=decode_address(USDC_MINT, ChainId.SOLANA), symbol="USDC", decimals=6)
USDT = Token(chain_id=ChainId.SOLANA, address=decode_address(USDT_MINT, ChainId.SOLANA), symbol="USDT", decimals=6)


# ---------------------------------------------------------------------------
# ChainId tests
# ---------------------------------------------------------------------------


class TestChainIdSolana:
    def test_solana_chain_id_value(self):
        assert ChainId.SOLANA == 1399811149

    def test_solana_is_int_enum(self):
        assert isinstance(ChainId.SOLANA, int)
        assert int(ChainId.SOLANA) == 1399811149

    def test_solana_token_chain_id(self):
        assert SOL.chain_id == ChainId.SOLANA
        assert str(SOL) == f"SOL({ChainId.SOLANA})"


# ---------------------------------------------------------------------------
# BaseSolanaAMM abstract interface
# ---------------------------------------------------------------------------


class TestBaseSolanaAMM:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseSolanaAMM()  # type: ignore[abstract]

    def test_apply_slippage(self):
        # Test via a concrete subclass
        raydium = Raydium()
        assert raydium._apply_slippage(1_000_000, 50) == 995_000
        assert raydium._apply_slippage(1_000_000, 0) == 1_000_000
        assert raydium._apply_slippage(1_000_000, 10_000) == 0

    def test_apply_slippage_invalid(self):
        raydium = Raydium()
        with pytest.raises(ValueError):
            raydium._apply_slippage(1_000_000, -1)
        with pytest.raises(ValueError):
            raydium._apply_slippage(1_000_000, 10_001)


# ---------------------------------------------------------------------------
# Raydium tests
# ---------------------------------------------------------------------------


class TestRaydium:
    def test_protocol_name(self):
        raydium = Raydium()
        assert raydium.protocol_name == "Raydium"

    def test_default_api_url(self):
        raydium = Raydium()
        assert raydium.api_url == _RAYDIUM_API_BASE

    def test_custom_api_url(self):
        raydium = Raydium(api_url="https://my-raydium.example.com")
        assert raydium.api_url == "https://my-raydium.example.com"

    def test_inherits_base_solana_amm(self):
        raydium = Raydium()
        assert isinstance(raydium, BaseSolanaAMM)

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_response = {
            "success": True,
            "data": {
                "inputAmount": "1000000000",
                "outputAmount": "150000000",  # 150 USDC
                "minimumAmountOut": "149250000",
                "priceImpactPct": "0.05",
                "routePlan": [{"poolId": "pool_abc123"}],
            },
        }
        with patch.object(raydium, "_get", new=AsyncMock(return_value=mock_response)):
            result = await raydium.get_quote(amount_in, USDC)

        assert isinstance(result, TokenAmount)
        assert result.token == USDC
        assert result.amount == 150_000_000

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        with patch.object(
            raydium,
            "_get",
            new=AsyncMock(side_effect=InsufficientLiquidityError("Raydium API error (400): no route")),
        ):
            with pytest.raises(InsufficientLiquidityError):
                await raydium.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_build_swap_route_success(self):
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_response = {
            "success": True,
            "data": {
                "inputAmount": "1000000000",
                "outputAmount": "150000000",
                "minimumAmountOut": "149250000",
                "priceImpactPct": "0.05",
                "routePlan": [{"poolId": "58oQChx4yWmvKnami8n1LnxS7vQp5YCGLGjrQCZFdcxm"}],
            },
        }
        with patch.object(raydium, "_get", new=AsyncMock(return_value=mock_response)):
            route = await raydium.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Raydium"
        assert route.steps[0].pool_address == "58oQChx4yWmvKnami8n1LnxS7vQp5YCGLGjrQCZFdcxm"
        assert route.amount_out.amount == 150_000_000
        assert route.price_impact == Decimal("0.0005")  # 0.05% / 100

    @pytest.mark.asyncio
    async def test_build_swap_route_no_pool_id(self):
        """build_swap_route should tolerate missing poolId in routePlan."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "0.5")

        mock_response = {
            "success": True,
            "data": {
                "inputAmount": "500000000",
                "outputAmount": "75000000",
                "minimumAmountOut": "74625000",
                "priceImpactPct": "0.1",
                "routePlan": [],
            },
        }
        with patch.object(raydium, "_get", new=AsyncMock(return_value=mock_response)):
            route = await raydium.build_swap_route(amount_in, USDC)

        assert route.steps[0].pool_address == ""

    @pytest.mark.asyncio
    async def test_get_passes_slippage_bps(self):
        """Raydium _get should receive slippageBps as a query parameter."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_response = {
            "success": True,
            "data": {
                "inputAmount": "1000000000",
                "outputAmount": "150000000",
                "minimumAmountOut": "148500000",
                "priceImpactPct": "0",
                "routePlan": [],
            },
        }
        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured.update(params)
            return mock_response

        with patch.object(raydium, "_get", new=fake_get):
            await raydium.get_quote(amount_in, USDC, slippage_bps=100)

        assert captured["slippageBps"] == 100
        assert captured["inputMint"] == SOL_MINT
        assert captured["outputMint"] == USDC_MINT

    @pytest.mark.asyncio
    async def test_build_transaction(self):
        """build_transaction should call compute, then POST to /transaction endpoint."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")
        wallet = "4wPBNzaFLPcBitNjNmJP8FtEHRFYQW4eMeFE6HB5Xekr"

        mock_compute_response = {
            "success": True,
            "data": {
                "inputAmount": "1000000000",
                "outputAmount": "150000000",
                "minimumAmountOut": "149250000",
                "priceImpactPct": "0.02",
                "routePlan": [{"poolId": "pool123"}],
            },
        }
        mock_tx_response = {
            "success": True,
            "data": [{"transaction": "AQAAAA=="}, {"transaction": "BQAAAA=="}],
        }

        from unittest.mock import MagicMock

        mock_post_resp = MagicMock()
        mock_post_resp.status = 200
        mock_post_resp.json = AsyncMock(return_value=mock_tx_response)
        mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_post_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(raydium, "_get", new=AsyncMock(return_value=mock_compute_response)):
            with patch("pydefi.amm.raydium.aiohttp.ClientSession", return_value=mock_session):
                txs = await raydium.build_transaction(amount_in, USDC, wallet=wallet)

        assert txs == ["AQAAAA==", "BQAAAA=="]

    @pytest.mark.asyncio
    async def test_build_transaction_api_error(self):
        """build_transaction should raise InsufficientLiquidityError on failure."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")
        wallet = "4wPBNzaFLPcBitNjNmJP8FtEHRFYQW4eMeFE6HB5Xekr"

        mock_compute_response = {
            "success": True,
            "data": {
                "inputAmount": "1000000000",
                "outputAmount": "150000000",
                "minimumAmountOut": "149250000",
                "priceImpactPct": "0",
                "routePlan": [],
            },
        }

        from unittest.mock import MagicMock

        mock_post_resp = MagicMock()
        mock_post_resp.status = 400
        mock_post_resp.json = AsyncMock(return_value={"success": False, "msg": "invalid wallet"})
        mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
        mock_post_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(raydium, "_get", new=AsyncMock(return_value=mock_compute_response)):
            with patch("pydefi.amm.raydium.aiohttp.ClientSession", return_value=mock_session):
                with pytest.raises(InsufficientLiquidityError):
                    await raydium.build_transaction(amount_in, USDC, wallet=wallet)


# ---------------------------------------------------------------------------
# Jupiter tests
# ---------------------------------------------------------------------------


class TestJupiter:
    def test_protocol_name(self):
        jupiter = Jupiter()
        assert jupiter.protocol_name == "Jupiter"

    def test_chain_id_is_solana(self):
        jupiter = Jupiter()
        assert jupiter.chain_id == ChainId.SOLANA

    def test_default_base_url(self):
        jupiter = Jupiter()
        assert jupiter.base_url == _JUPITER_API_BASE

    def test_custom_base_url(self):
        jupiter = Jupiter(base_url="https://my-jupiter.example.com")
        assert jupiter.base_url == "https://my-jupiter.example.com"

    def test_headers_no_api_key(self):
        jupiter = Jupiter()
        headers = jupiter._headers()
        assert "Accept" in headers
        assert "Authorization" not in headers

    def test_headers_with_api_key(self):
        jupiter = Jupiter(api_key="test-key-123")
        headers = jupiter._headers()
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_inherits_base_aggregator(self):
        from pydefi.aggregator.base import BaseAggregator

        assert isinstance(Jupiter(), BaseAggregator)

    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_response = {
            "inputMint": SOL_MINT,
            "inAmount": "1000000000",
            "outputMint": USDC_MINT,
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "priceImpactPct": "0.03",
            "routePlan": [{"swapInfo": {"ammKey": "pool1"}, "percent": 100}],
        }
        with patch.object(jupiter, "_get", new=AsyncMock(return_value=mock_response)):
            quote = await jupiter.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(quote, AggregatorQuote)
        assert quote.protocol == "Jupiter"
        assert quote.amount_out.amount == 150_000_000
        assert quote.min_amount_out.amount == 149_250_000
        assert quote.gas_estimate == 0  # Solana uses compute units
        assert quote.price_impact == Decimal("0.0003")  # 0.03% / 100
        assert quote.token_in == SOL
        assert quote.token_out == USDC

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        with patch.object(
            jupiter,
            "_get",
            new=AsyncMock(side_effect=AggregatorError("Jupiter API error", 400)),
        ):
            with pytest.raises(AggregatorError):
                await jupiter.get_quote(amount_in, USDC)

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "2")

        mock_response = {
            "inputMint": SOL_MINT,
            "inAmount": "2000000000",
            "outputMint": USDC_MINT,
            "outAmount": "300000000",
            "otherAmountThreshold": "298500000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "priceImpactPct": "0.01",
            "routePlan": [],
        }
        with patch.object(jupiter, "_get", new=AsyncMock(return_value=mock_response)):
            route = await jupiter.build_swap_route(amount_in, USDC)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Jupiter"
        assert route.steps[0].pool_address == ""
        assert route.amount_out.amount == 300_000_000

    @pytest.mark.asyncio
    async def test_get_quote_passes_params(self):
        """get_quote should forward inputMint, outputMint, amount, slippageBps."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured.update(params)
            return {
                "outAmount": "100000000",
                "otherAmountThreshold": "99500000",
                "priceImpactPct": "0",
                "routePlan": [],
            }

        with patch.object(jupiter, "_get", new=fake_get):
            await jupiter.get_quote(amount_in, USDC, slippage_bps=100)

        assert captured["inputMint"] == SOL_MINT
        assert captured["outputMint"] == USDC_MINT
        assert captured["amount"] == "1000000000"
        assert captured["slippageBps"] == 100

    @pytest.mark.asyncio
    async def test_get_swap_transaction(self):
        """get_swap_transaction should call /quote then POST /swap."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")
        user_pubkey = "4wPBNzaFLPcBitNjNmJP8FtEHRFYQW4eMeFE6HB5Xekr"

        mock_quote = {
            "inputMint": SOL_MINT,
            "inAmount": "1000000000",
            "outputMint": USDC_MINT,
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "priceImpactPct": "0",
            "routePlan": [],
        }
        mock_swap_response = {
            "swapTransaction": "AQAAAA==",
            "lastValidBlockHeight": 123456789,
        }

        with patch.object(jupiter, "_get", new=AsyncMock(return_value=mock_quote)):
            from unittest.mock import MagicMock

            mock_post_resp = MagicMock()
            mock_post_resp.status = 200
            mock_post_resp.json = AsyncMock(return_value=mock_swap_response)
            mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
            mock_post_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            with patch("pydefi.aggregator.jupiter.aiohttp.ClientSession", return_value=mock_session):
                result = await jupiter.get_swap_transaction(amount_in, USDC, user_pubkey)

        assert result["swapTransaction"] == "AQAAAA=="
        assert result["lastValidBlockHeight"] == 123456789

    @pytest.mark.asyncio
    async def test_get_swap_transaction_api_error(self):
        """get_swap_transaction should raise AggregatorError on /swap failure."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")
        user_pubkey = "4wPBNzaFLPcBitNjNmJP8FtEHRFYQW4eMeFE6HB5Xekr"

        mock_quote = {
            "inputMint": SOL_MINT,
            "inAmount": "1000000000",
            "outputMint": USDC_MINT,
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "swapMode": "ExactIn",
            "slippageBps": 50,
            "priceImpactPct": "0",
            "routePlan": [],
        }

        with patch.object(jupiter, "_get", new=AsyncMock(return_value=mock_quote)):
            from unittest.mock import MagicMock

            mock_post_resp = MagicMock()
            mock_post_resp.status = 400
            mock_post_resp.json = AsyncMock(return_value={"error": "bad request"})
            mock_post_resp.__aenter__ = AsyncMock(return_value=mock_post_resp)
            mock_post_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            with patch("pydefi.aggregator.jupiter.aiohttp.ClientSession", return_value=mock_session):
                with pytest.raises(AggregatorError):
                    await jupiter.get_swap_transaction(amount_in, USDC, user_pubkey)


# ---------------------------------------------------------------------------
# Stargate Solana chain ID
# ---------------------------------------------------------------------------


class TestStargateSolana:
    def test_solana_lz_chain_id_in_mapping(self):
        from pydefi.bridge.stargate import _LZ_CHAIN_ID

        assert ChainId.SOLANA in _LZ_CHAIN_ID
        assert _LZ_CHAIN_ID[ChainId.SOLANA] == 30168

    def test_solana_lz_chain_id_via_bridge(self):
        from pydefi.bridge.stargate import Stargate

        sg = Stargate(
            w3=None,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.SOLANA,
            router_address="0x8731d54E9D02c286767d56ac03e8037C07e01e98",
        )
        assert sg._lz_chain_id(ChainId.SOLANA) == 30168


# ---------------------------------------------------------------------------
# JupiterSwapV2 tests
# ---------------------------------------------------------------------------


class TestJupiterSwapV2:
    def test_protocol_name(self):
        assert JupiterSwapV2().protocol_name == "Jupiter"

    def test_chain_id_is_solana(self):
        assert JupiterSwapV2().chain_id == ChainId.SOLANA

    def test_default_base_url(self):
        assert JupiterSwapV2().base_url == _JUPITER_SWAP_V2_BASE

    def test_custom_base_url(self):
        j = JupiterSwapV2(base_url="https://custom.example.com")
        assert j.base_url == "https://custom.example.com"

    def test_headers_no_api_key(self):
        headers = JupiterSwapV2()._headers()
        assert "Accept" in headers
        assert "x-api-key" not in headers

    def test_headers_with_api_key(self):
        headers = JupiterSwapV2(api_key="my-api-key")._headers()
        assert headers["x-api-key"] == "my-api-key"

    def test_inherits_base_aggregator(self):
        from pydefi.aggregator.base import BaseAggregator

        assert isinstance(JupiterSwapV2(), BaseAggregator)

    @pytest.mark.asyncio
    async def test_get_order_without_taker(self):
        """get_order without taker should pass no taker param and return raw dict."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_response = {
            "inputMint": SOL_MINT,
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "priceImpactPct": "0.02",
            "routePlan": [],
        }
        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured["endpoint"] = endpoint
            captured.update(params)
            return mock_response

        with patch.object(j, "_get", new=fake_get):
            result = await j.get_order(amount_in, USDC)

        assert captured["endpoint"] == "order"
        assert captured["inputMint"] == SOL_MINT
        assert captured["outputMint"] == USDC_MINT
        assert "taker" not in captured
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_get_order_with_taker(self):
        """get_order with taker should include taker and slippageBps params."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")
        taker = "4wPBNzaFLPcBitNjNmJP8FtEHRFYQW4eMeFE6HB5Xekr"

        mock_response = {
            "inputMint": SOL_MINT,
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "priceImpactPct": "0",
            "routePlan": [],
            "transaction": "AQAAAA==",
            "requestId": "req-123",
        }
        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured.update(params)
            return mock_response

        with patch.object(j, "_get", new=fake_get):
            result = await j.get_order(amount_in, USDC, taker=taker, slippage_bps=100)

        assert captured["taker"] == taker
        assert captured["slippageBps"] == 100
        assert result["transaction"] == "AQAAAA=="
        assert result["requestId"] == "req-123"

    @pytest.mark.asyncio
    async def test_get_quote_delegates_to_get_order(self):
        """get_quote should call get_order without taker and parse the response."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_order = {
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "priceImpactPct": "0.04",
            "routePlan": [{"swapInfo": {"ammKey": "pool1"}, "percent": 100}],
        }
        with patch.object(j, "get_order", new=AsyncMock(return_value=mock_order)):
            quote = await j.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(quote, AggregatorQuote)
        assert quote.amount_out.amount == 150_000_000
        assert quote.min_amount_out.amount == 149_250_000
        assert quote.price_impact == Decimal("0.0004")  # 0.04% / 100
        assert quote.protocol == "Jupiter"
        assert quote.gas_estimate == 0

    @pytest.mark.asyncio
    async def test_execute_order(self):
        """execute_order should POST to /execute with signedTransaction + requestId."""
        j = JupiterSwapV2()
        captured: dict = {}

        async def fake_post(endpoint: str, payload: dict) -> dict:
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"status": "Success", "signature": "abc123"}

        with patch.object(j, "_post", new=fake_post):
            result = await j.execute_order("SIGNED==", "req-123")

        assert captured["endpoint"] == "execute"
        assert captured["payload"]["signedTransaction"] == "SIGNED=="
        assert captured["payload"]["requestId"] == "req-123"
        assert result["status"] == "Success"
        assert result["signature"] == "abc123"

    @pytest.mark.asyncio
    async def test_execute_order_api_error(self):
        """execute_order should raise AggregatorError on failure."""
        j = JupiterSwapV2()
        with patch.object(j, "_post", new=AsyncMock(side_effect=AggregatorError("bad", 400))):
            with pytest.raises(AggregatorError):
                await j.execute_order("BAD==", "req-bad")

    @pytest.mark.asyncio
    async def test_get_build(self):
        """get_build should call GET /build and return raw instructions dict."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")
        taker = "DummyWaLLeTaDDreSS1111111111111111111111111"

        mock_response = {
            "inputMint": SOL_MINT,
            "outAmount": "150000000",
            "instructions": [{"programId": "prog1", "accounts": [], "data": "AAAA"}],
            "addressLookupTableAddresses": [],
        }
        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured["endpoint"] = endpoint
            captured.update(params)
            return mock_response

        with patch.object(j, "_get", new=fake_get):
            result = await j.get_build(amount_in, USDC, taker=taker, slippage_bps=50)

        assert captured["endpoint"] == "build"
        assert captured["inputMint"] == SOL_MINT
        assert captured["outputMint"] == USDC_MINT
        assert captured["taker"] == taker
        assert captured["slippageBps"] == 50
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_get_build_no_slippage(self):
        """get_build without slippage_bps should not include slippageBps param."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")
        taker = "DummyWaLLeTaDDreSS1111111111111111111111111"
        captured: dict = {}

        async def fake_get(endpoint: str, params: dict) -> dict:
            captured.update(params)
            return {"outAmount": "1"}

        with patch.object(j, "_get", new=fake_get):
            await j.get_build(amount_in, USDC, taker=taker)

        assert "slippageBps" not in captured
        assert captured["taker"] == taker

    @pytest.mark.asyncio
    async def test_build_swap_route(self):
        """build_swap_route should return a well-formed SwapRoute via get_quote."""
        j = JupiterSwapV2()
        amount_in = TokenAmount.from_human(SOL, "1")

        mock_order = {
            "outAmount": "150000000",
            "otherAmountThreshold": "149250000",
            "priceImpactPct": "0.01",
            "routePlan": [],
        }
        with patch.object(j, "get_order", new=AsyncMock(return_value=mock_order)):
            route = await j.build_swap_route(amount_in, USDC)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Jupiter"
        assert route.amount_out.amount == 150_000_000
        assert route.price_impact == Decimal("0.0001")  # 0.01% / 100

"""Tests for pydefi.bridge (no live calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.bridge.across import Across
from pydefi.bridge.base import BaseBridge
from pydefi.bridge.gaszip import _SUPPORTED_CHAINS, GasZip
from pydefi.bridge.mayan import _CHAIN_NAMES, Mayan
from pydefi.bridge.relay import Relay
from pydefi.bridge.stargate import _LZ_CHAIN_ID, _POOL_IDS, Stargate
from pydefi.exceptions import BridgeError
from pydefi.types import ChainId, Token, TokenAmount


def _make_aiohttp_mock(status: int, response_data) -> MagicMock:
    """Build a mock aiohttp.ClientSession that returns *response_data* for any GET/POST.

    ``session.get()`` and ``session.post()`` return an async context manager
    (not a coroutine), so we use MagicMock with ``__aenter__``/``__aexit__``.
    """
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data)

    # The response must be used as: async with session.get(...) as resp:
    resp_ctx = MagicMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=resp_ctx)
    mock_session.post = MagicMock(return_value=resp_ctx)

    # ClientSession itself is used as: async with aiohttp.ClientSession() as session:
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return session_ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

USDC_ETH = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    symbol="USDC",
    decimals=6,
)
USDC_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address="0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
    symbol="USDC",
    decimals=6,
)

STARGATE_ROUTER_ETH = "0x8731d54E9D02c286767d56ac03e8037C07e01e98"
SPOKE_POOL_ETH = "0x5c7BCd6E7De5423a257D81B442095A1a6ced35C5"


# ---------------------------------------------------------------------------
# Stargate tests
# ---------------------------------------------------------------------------


class TestStargate:
    def test_protocol_name(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        assert sg.protocol_name == "Stargate"

    def test_chain_ids_stored(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        assert sg.src_chain_id == 1
        assert sg.dst_chain_id == 42161

    def test_lz_chain_id_known(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        assert sg._lz_chain_id(1) == _LZ_CHAIN_ID[1]
        assert sg._lz_chain_id(42161) == _LZ_CHAIN_ID[42161]

    def test_lz_chain_id_unknown_raises(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=999999,
            router_address=STARGATE_ROUTER_ETH,
        )
        with pytest.raises(BridgeError):
            sg._lz_chain_id(999999)

    def test_pool_id_usdc(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        assert sg._pool_id(USDC_ETH) == _POOL_IDS["USDC"]

    def test_pool_id_unknown_raises(self):
        unknown_token = Token(chain_id=1, address="0x" + "AB" * 20, symbol="UNKNOWN")
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        with pytest.raises(BridgeError):
            sg._pool_id(unknown_token)

    def test_apply_slippage(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        assert sg._apply_slippage(1_000_000, 50) == 995_000
        assert sg._apply_slippage(1_000_000, 0) == 1_000_000

    @pytest.mark.asyncio
    async def test_get_quote(self):
        sg = Stargate(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            router_address=STARGATE_ROUTER_ETH,
        )
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        with patch.object(sg, "quote_lz_fee", new=AsyncMock(return_value=5 * 10**15)):
            quote = await sg.get_quote(USDC_ETH, USDC_ARB, amount_in)

        assert quote.protocol == "Stargate"
        assert quote.amount_out.amount < amount_in.amount  # fee deducted
        assert quote.bridge_fee.amount > 0
        assert quote.estimated_time_seconds == 180

    def test_lz_pool_id_constants(self):
        assert _POOL_IDS["USDC"] == 1
        assert _POOL_IDS["USDT"] == 2
        assert _POOL_IDS["ETH"] == 13

    def test_lz_chain_id_constants(self):
        assert _LZ_CHAIN_ID[1] == 101  # Ethereum
        assert _LZ_CHAIN_ID[42161] == 110  # Arbitrum


# ---------------------------------------------------------------------------
# Across tests
# ---------------------------------------------------------------------------


class TestAcross:
    def test_protocol_name(self):
        ac = Across(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            spoke_pool_address=SPOKE_POOL_ETH,
        )
        assert ac.protocol_name == "Across"

    def test_chain_ids_stored(self):
        ac = Across(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            spoke_pool_address=SPOKE_POOL_ETH,
        )
        assert ac.src_chain_id == 1
        assert ac.dst_chain_id == 42161

    def test_custom_api_base_url(self):
        ac = Across(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            spoke_pool_address=SPOKE_POOL_ETH,
            api_base_url="https://my-across.example.com/api",
        )
        assert ac._api_base == "https://my-across.example.com/api"

    @pytest.mark.asyncio
    async def test_get_quote(self):
        ac = Across(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            spoke_pool_address=SPOKE_POOL_ETH,
        )
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        mock_fee_data = {
            "totalRelayFee": {"pct": str(int(6e15))},  # 0.6% fee (6e15 / 1e18)
            "estimatedFillTimeSec": 120,
            "timestamp": 1700000000,
        }
        with patch.object(ac, "get_suggested_fees", new=AsyncMock(return_value=mock_fee_data)):
            quote = await ac.get_quote(USDC_ETH, USDC_ARB, amount_in)

        assert quote.protocol == "Across"
        assert quote.amount_out.amount < amount_in.amount
        assert quote.bridge_fee.amount > 0
        assert quote.estimated_time_seconds == 120

    def test_apply_slippage(self):
        ac = Across(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            spoke_pool_address=SPOKE_POOL_ETH,
        )
        result = ac._apply_slippage(1_000_000, 50)
        assert result == 995_000


# ---------------------------------------------------------------------------
# BaseBridge abstract interface
# ---------------------------------------------------------------------------


class TestBaseBridge:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseBridge(src_chain_id=1, dst_chain_id=42161)


# ---------------------------------------------------------------------------
# Mayan tests
# ---------------------------------------------------------------------------

ETH_NATIVE = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
    symbol="ETH",
    decimals=18,
)
ETH_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
    symbol="ETH",
    decimals=18,
)


class TestMayan:
    def test_protocol_name(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        assert m.protocol_name == "Mayan"

    def test_chain_ids_stored(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        assert m.src_chain_id == 1
        assert m.dst_chain_id == 42161

    def test_chain_name_known(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        assert m._chain_name(1) == _CHAIN_NAMES[1]
        assert m._chain_name(42161) == _CHAIN_NAMES[42161]

    def test_chain_name_unknown_raises(self):
        m = Mayan(src_chain_id=1, dst_chain_id=999999)
        with pytest.raises(BridgeError):
            m._chain_name(999999)

    def test_custom_api_base_url(self):
        m = Mayan(
            src_chain_id=1,
            dst_chain_id=42161,
            api_base_url="https://my-mayan.example.com/v3",
        )
        assert m._api_base == "https://my-mayan.example.com/v3"

    @pytest.mark.asyncio
    async def test_get_quote(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        mock_api_response = [
            {
                "type": "SWIFT",
                "expectedAmountOut": "998.5",
                "minAmountOut": "995.0",
                "effectiveAmountIn": "1000.0",
            }
        ]

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_mock(200, mock_api_response)):
            quote = await m.get_quote(USDC_ETH, USDC_ARB, amount_in)

        assert quote.protocol == "Mayan"
        assert quote.amount_out.amount > 0
        assert quote.estimated_time_seconds == 10  # SWIFT route

    @pytest.mark.asyncio
    async def test_get_quote_api_error(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_mock(400, {"error": "bad request"})):
            with pytest.raises(BridgeError):
                await m.get_quote(USDC_ETH, USDC_ARB, amount_in)

    @pytest.mark.asyncio
    async def test_get_quote_empty_routes(self):
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_mock(200, [])):
            with pytest.raises(BridgeError):
                await m.get_quote(USDC_ETH, USDC_ARB, amount_in)

    def test_chain_name_constants(self):
        assert _CHAIN_NAMES[1] == "ethereum"
        assert _CHAIN_NAMES[10] == "optimism"
        assert _CHAIN_NAMES[56] == "bsc"
        assert _CHAIN_NAMES[137] == "polygon"
        assert _CHAIN_NAMES[42161] == "arbitrum"
        assert _CHAIN_NAMES[8453] == "base"
        assert _CHAIN_NAMES[43114] == "avalanche"
        assert _CHAIN_NAMES[59144] == "linea"
        assert _CHAIN_NAMES[534352] == "scroll"
        assert _CHAIN_NAMES[81457] == "blast"
        assert _CHAIN_NAMES[324] == "zksync"
        assert _CHAIN_NAMES[7777777] == "zora"


# ---------------------------------------------------------------------------
# GasZip tests
# ---------------------------------------------------------------------------

GASZIP_CONTRACT = "0x" + "AA" * 20


class TestGasZip:
    def test_protocol_name(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=42161, contract_address=GASZIP_CONTRACT)
        assert gz.protocol_name == "GasZip"

    def test_chain_ids_stored(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=42161, contract_address=GASZIP_CONTRACT)
        assert gz.src_chain_id == 1
        assert gz.dst_chain_id == 42161

    def test_unsupported_chain_raises(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=999999, contract_address=GASZIP_CONTRACT)
        with pytest.raises(BridgeError):
            gz._check_chain(999999)

    def test_supported_chain_passes(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=42161, contract_address=GASZIP_CONTRACT)
        gz._check_chain(1)  # should not raise

    def test_custom_api_base_url(self):
        gz = GasZip(
            src_chain_id=1,
            dst_chain_id=42161,
            contract_address=GASZIP_CONTRACT,
            api_base_url="https://my-gaszip.example.com/v2",
        )
        assert gz._api_base == "https://my-gaszip.example.com/v2"

    @pytest.mark.asyncio
    async def test_get_quote(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=42161, contract_address=GASZIP_CONTRACT)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=10**17)  # 0.1 ETH

        mock_api_response = {
            "quotes": [
                {
                    "chain": 42161,
                    "expected": str(int(0.0998 * 10**18)),
                    "speed": 30,
                }
            ]
        }

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_mock(200, mock_api_response)):
            quote = await gz.get_quote(ETH_NATIVE, ETH_ARB, amount_in)

        assert quote.protocol == "GasZip"
        assert quote.amount_out.amount == int(0.0998 * 10**18)
        assert quote.bridge_fee.amount > 0
        assert quote.estimated_time_seconds == 30

    @pytest.mark.asyncio
    async def test_get_quote_unsupported_dst_chain(self):
        gz = GasZip(src_chain_id=1, dst_chain_id=999999, contract_address=GASZIP_CONTRACT)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=10**17)
        with pytest.raises(BridgeError):
            await gz.get_quote(ETH_NATIVE, ETH_ARB, amount_in)

    def test_supported_chains_set(self):
        assert 1 in _SUPPORTED_CHAINS  # Ethereum
        assert 10 in _SUPPORTED_CHAINS  # Optimism
        assert 56 in _SUPPORTED_CHAINS  # BSC
        assert 137 in _SUPPORTED_CHAINS  # Polygon
        assert 8453 in _SUPPORTED_CHAINS  # Base
        assert 42161 in _SUPPORTED_CHAINS  # Arbitrum
        assert 43114 in _SUPPORTED_CHAINS  # Avalanche
        assert 59144 in _SUPPORTED_CHAINS  # Linea
        assert 534352 in _SUPPORTED_CHAINS  # Scroll
        assert 81457 in _SUPPORTED_CHAINS  # Blast
        assert 324 in _SUPPORTED_CHAINS  # zkSync Era
        assert 7777777 in _SUPPORTED_CHAINS  # Zora
        assert 130 in _SUPPORTED_CHAINS  # Unichain
        assert 480 in _SUPPORTED_CHAINS  # World Chain


# ---------------------------------------------------------------------------
# Relay tests
# ---------------------------------------------------------------------------


class TestRelay:
    def test_protocol_name(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        assert r.protocol_name == "Relay"

    def test_chain_ids_stored(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        assert r.src_chain_id == 1
        assert r.dst_chain_id == 42161

    def test_custom_api_base_url(self):
        r = Relay(
            src_chain_id=1,
            dst_chain_id=42161,
            api_base_url="https://my-relay.example.com",
        )
        assert r._api_base == "https://my-relay.example.com"

    @pytest.mark.asyncio
    async def test_get_quote(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")

        mock_api_response = {
            "details": {
                "currencyOut": {"amount": str(int(997.5 * 10**6))},
                "timeEstimate": 15,
            },
            "steps": [
                {
                    "items": [
                        {
                            "data": {
                                "to": "0x" + "BB" * 20,
                                "data": "0xdeadbeef",
                                "value": "0",
                                "gas": 200000,
                            }
                        }
                    ]
                }
            ],
        }

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            quote = await r.get_quote(USDC_ETH, USDC_ARB, amount_in)

        assert quote.protocol == "Relay"
        assert quote.amount_out.amount == int(997.5 * 10**6)
        assert quote.estimated_time_seconds == 15

    @pytest.mark.asyncio
    async def test_build_bridge_tx(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")
        recipient = "0x" + "CC" * 20

        mock_api_response = {
            "details": {
                "currencyOut": {"amount": str(int(997.5 * 10**6))},
                "timeEstimate": 15,
            },
            "steps": [
                {
                    "items": [
                        {
                            "data": {
                                "to": "0x" + "BB" * 20,
                                "data": "0xdeadbeef",
                                "value": "0",
                                "gas": 200000,
                            }
                        }
                    ]
                }
            ],
        }

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            tx = await r.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, recipient)

        assert tx["to"] == "0x" + "BB" * 20
        assert tx["data"] == "0xdeadbeef"
        assert tx["value"] == "0"

    @pytest.mark.asyncio
    async def test_build_bridge_tx_no_steps_raises(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")
        recipient = "0x" + "CC" * 20

        mock_api_response = {"details": {}, "steps": []}

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            with pytest.raises(BridgeError):
                await r.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, recipient)

    @pytest.mark.asyncio
    async def test_build_bridge_tx_no_items_raises(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")
        recipient = "0x" + "CC" * 20

        mock_api_response = {"details": {}, "steps": [{"items": []}]}

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            with pytest.raises(BridgeError):
                await r.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, recipient)

"""Tests for pydefi.bridge (no live calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.bridge.across import Across
from pydefi.bridge.base import BaseBridge
from pydefi.bridge.cctp import (
    HYPERCORE_DEX_PERP,
    HYPERCORE_DEX_SPOT,
    encode_cctp_forward_hook_data,
)
from pydefi.bridge.gaszip import _SUPPORTED_CHAINS, GasZip
from pydefi.bridge.layerzero_oft import _LZ_EID, LayerZeroOFT
from pydefi.bridge.mayan import _CHAIN_NAMES, _MAYAN_FORWARDER, Mayan
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

    @pytest.mark.asyncio
    async def test_build_bridge_tx(self):
        """build_bridge_tx returns a tx dict with non-empty calldata.

        The outer calldata selector must match ``swapAndForwardEth``
        (0xfa74fd43) and the inner ``mayanData`` must embed
        ``createOrderWithToken`` (0xa3a30834).
        """
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=10**18)  # 1 ETH
        recipient = "0x" + "CC" * 20
        weth = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        swift_contract = "0x" + "AA" * 20
        swap_router = "0x" + "BB" * 20

        mock_quote_response = {
            "quotes": [
                {
                    "type": "SWIFT",
                    "swiftMayanContract": swift_contract,
                    "minAmountOut": "900",
                    "gasDrop": "0",
                    "cancelRelayerFee64": "1000",
                    "refundRelayerFee64": "1000",
                    "deadline64": "9999999999",
                    "swiftAuctionMode": 1,
                    "effectiveAmountIn64": str(10**18),
                    "swiftInputDecimals": 18,
                    "minMiddleAmount": "0.9",
                    "swiftInputContract": weth,
                }
            ]
        }
        mock_swap_response = {
            "swapRouterAddress": swap_router,
            "swapRouterCalldata": "0xdeadbeef",
        }

        quote_ctx = _make_aiohttp_mock(200, mock_quote_response)
        swap_ctx = _make_aiohttp_mock(200, mock_swap_response)

        with patch("aiohttp.ClientSession", MagicMock(side_effect=[quote_ctx, swap_ctx])):
            tx = await m.build_bridge_tx(ETH_NATIVE, USDC_ARB, amount_in, recipient)

        assert tx["to"] == _MAYAN_FORWARDER
        assert tx["value"] == str(10**18)
        # calldata is non-empty and carries the swapAndForwardEth selector
        assert tx["data"].startswith("0xfa74fd43")
        # the inner createOrderWithToken calldata must be embedded
        assert "a3a30834" in tx["data"]

    @pytest.mark.asyncio
    async def test_build_bridge_tx_erc20_raises(self):
        """build_bridge_tx raises BridgeError for ERC-20 token_in."""
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")
        with pytest.raises(BridgeError):
            await m.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, "0x" + "CC" * 20)

    @pytest.mark.asyncio
    async def test_build_bridge_tx_no_swift_route_raises(self):
        """build_bridge_tx raises BridgeError when no SWIFT route is returned."""
        m = Mayan(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=10**18)

        mock_quote_response = {
            "quotes": [
                {
                    "type": "MCTP",
                    "expectedAmountOut": "998",
                }
            ]
        }

        with patch("aiohttp.ClientSession", return_value=_make_aiohttp_mock(200, mock_quote_response)):
            with pytest.raises(BridgeError):
                await m.build_bridge_tx(ETH_NATIVE, USDC_ARB, amount_in, "0x" + "CC" * 20)


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


# ---------------------------------------------------------------------------
# LayerZeroOFT tests
# ---------------------------------------------------------------------------

OFT_ADDRESS = "0x" + "DD" * 20
OFT_DST_ADDRESS = "0x" + "EE" * 20

# Tokens for unified-address OFTs (same contract address on every chain,
# e.g. USDT0 at 0x1E4a5963aBFD975d8c9021ce480b42188849D41d).
OFT_TOKEN_ETH = Token(
    chain_id=ChainId.ETHEREUM,
    address=OFT_ADDRESS,
    symbol="OFT",
    decimals=18,
)
OFT_TOKEN_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address=OFT_ADDRESS,
    symbol="OFT",
    decimals=18,
)
# Token for a non-unified OFT (different address on the destination chain).
OFT_TOKEN_ARB_ALT = Token(
    chain_id=ChainId.ARBITRUM,
    address=OFT_DST_ADDRESS,
    symbol="OFT",
    decimals=18,
)


class TestLayerZeroOFT:
    def test_protocol_name(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        assert oft.protocol_name == "LayerZeroOFT"

    def test_chain_ids_stored(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        assert oft.src_chain_id == 1
        assert oft.dst_chain_id == 42161

    def test_dst_oft_address_defaults_to_oft_address(self):
        """For unified-address OFTs, dst_oft_address defaults to oft_address."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        assert oft.dst_oft_address == OFT_ADDRESS

    def test_dst_oft_address_explicit(self):
        """Non-unified OFTs can specify a distinct destination address."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
            dst_oft_address=OFT_DST_ADDRESS,
        )
        assert oft.dst_oft_address == OFT_DST_ADDRESS

    def test_lz_eid_known(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        assert oft._lz_eid(1) == _LZ_EID[1]
        assert oft._lz_eid(42161) == _LZ_EID[42161]

    def test_lz_eid_unknown_raises(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=999999,
            oft_address=OFT_ADDRESS,
        )
        with pytest.raises(BridgeError):
            oft._lz_eid(999999)

    def test_address_to_bytes32(self):
        addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        result = LayerZeroOFT._address_to_bytes32(addr)
        assert len(result) == 32
        # Address bytes should appear in the last 20 bytes
        assert result[:12] == b"\x00" * 12
        assert result[12:].hex() == addr[2:].lower()

    def test_apply_slippage(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        assert oft._apply_slippage(1_000_000, 50) == 995_000
        assert oft._apply_slippage(1_000_000, 0) == 1_000_000

    @pytest.mark.asyncio
    async def test_get_quote(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        amount_in = TokenAmount.from_human(OFT_TOKEN_ETH, "1000")
        quote = await oft.get_quote(OFT_TOKEN_ETH, OFT_TOKEN_ARB, amount_in)

        assert quote.protocol == "LayerZeroOFT"
        # OFT is 1:1 — no token protocol fee
        assert quote.amount_out.amount == amount_in.amount
        assert quote.bridge_fee.amount == 0
        assert quote.estimated_time_seconds == 30

    @pytest.mark.asyncio
    async def test_get_quote_validates_token_in(self):
        """get_quote raises BridgeError when token_in address does not match."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        wrong_token = Token(
            chain_id=ChainId.ETHEREUM,
            address="0x" + "FF" * 20,
            symbol="WRONG",
            decimals=18,
        )
        amount_in = TokenAmount.from_human(wrong_token, "10")
        with pytest.raises(BridgeError, match="token_in"):
            await oft.get_quote(wrong_token, OFT_TOKEN_ARB, amount_in)

    @pytest.mark.asyncio
    async def test_get_quote_validates_token_out(self):
        """get_quote raises BridgeError when token_out address does not match."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        wrong_token = Token(
            chain_id=ChainId.ARBITRUM,
            address="0x" + "FF" * 20,
            symbol="WRONG",
            decimals=18,
        )
        amount_in = TokenAmount.from_human(OFT_TOKEN_ETH, "10")
        with pytest.raises(BridgeError, match="token_out"):
            await oft.get_quote(OFT_TOKEN_ETH, wrong_token, amount_in)

    @pytest.mark.asyncio
    async def test_get_quote_non_unified_oft(self):
        """For a non-unified OFT, token_out must match dst_oft_address."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
            dst_oft_address=OFT_DST_ADDRESS,
        )
        amount_in = TokenAmount.from_human(OFT_TOKEN_ETH, "10")
        quote = await oft.get_quote(OFT_TOKEN_ETH, OFT_TOKEN_ARB_ALT, amount_in)
        assert quote.amount_out.amount == amount_in.amount

    @pytest.mark.asyncio
    async def test_quote_send_fee(self):
        """quote_send_fee calls quoteSend on the underlying OFT contract."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        # Mock the underlying OFT contract call so we exercise the real
        # quote_send_fee logic (EID mapping, send_param construction, contract wiring)
        # rather than patching quote_send_fee itself.
        mock_call = AsyncMock(return_value=(5 * 10**15, 0))
        mock_quote_send = MagicMock(return_value=MagicMock(call=mock_call))
        oft._oft = MagicMock()
        oft._oft.fns = MagicMock()
        oft._oft.fns.quoteSend = mock_quote_send

        fee = await oft.quote_send_fee(1_000_000, "0x" + "AA" * 20)

        assert fee == 5 * 10**15
        mock_quote_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_bridge_tx(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        amount_in = TokenAmount.from_human(OFT_TOKEN_ETH, "1000")
        recipient = "0x" + "AA" * 20

        with patch.object(oft, "quote_send_fee", new=AsyncMock(return_value=5 * 10**15)):
            tx = await oft.build_bridge_tx(OFT_TOKEN_ETH, OFT_TOKEN_ARB, amount_in, recipient)

        assert tx["to"] == OFT_ADDRESS
        assert tx["data"].startswith("0x")
        assert tx["value"] == str(5 * 10**15)
        assert int(tx["gas"]) > 0

    @pytest.mark.asyncio
    async def test_build_bridge_tx_uses_refund_address(self):
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        amount_in = TokenAmount.from_human(OFT_TOKEN_ETH, "100")
        recipient = "0x" + "AA" * 20
        refund = "0x" + "BB" * 20

        with patch.object(oft, "quote_send_fee", new=AsyncMock(return_value=10**15)):
            tx = await oft.build_bridge_tx(OFT_TOKEN_ETH, OFT_TOKEN_ARB, amount_in, recipient, refund_address=refund)

        assert tx["to"] == OFT_ADDRESS
        # The refund address must be encoded into the calldata
        assert refund[2:].lower() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_build_bridge_tx_validates_token_in(self):
        """build_bridge_tx raises BridgeError when token_in address does not match."""
        oft = LayerZeroOFT(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            oft_address=OFT_ADDRESS,
        )
        wrong_token = Token(
            chain_id=ChainId.ETHEREUM,
            address="0x" + "FF" * 20,
            symbol="WRONG",
            decimals=18,
        )
        amount_in = TokenAmount.from_human(wrong_token, "10")
        with pytest.raises(BridgeError, match="token_in"):
            with patch.object(oft, "quote_send_fee", new=AsyncMock(return_value=10**15)):
                await oft.build_bridge_tx(wrong_token, OFT_TOKEN_ARB, amount_in, "0x" + "AA" * 20)

    def test_lz_eid_constants(self):
        assert _LZ_EID[1] == 30101  # Ethereum
        assert _LZ_EID[42161] == 30110  # Arbitrum
        assert _LZ_EID[10] == 30111  # Optimism
        assert _LZ_EID[8453] == 30184  # Base
        assert _LZ_EID[56] == 30102  # BNB Chain
        assert _LZ_EID[137] == 30109  # Polygon
        assert _LZ_EID[43114] == 30106  # Avalanche
        assert _LZ_EID[59144] == 30183  # Linea
        assert _LZ_EID[534352] == 30214  # Scroll
        assert _LZ_EID[81457] == 30243  # Blast
        assert _LZ_EID[324] == 30165  # zkSync Era
        assert _LZ_EID[7777777] == 30195  # Zora
        assert _LZ_EID[130] == 30320  # Unichain
        assert _LZ_EID[480] == 30337  # World Chain


# ---------------------------------------------------------------------------
# CCTP encode_cctp_forward_hook_data tests
# ---------------------------------------------------------------------------

_MOCK_RECIPIENT = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


class TestEncodeCctpForwardHookData:
    """Unit tests for encode_cctp_forward_hook_data() byte layout."""

    def test_no_recipient_length(self):
        """Header-only hookData is 32 bytes (magic 24 + version 4 + dataLen 4)."""
        data = encode_cctp_forward_hook_data(recipient=None)
        assert len(data) == 32

    def test_no_recipient_magic(self):
        """First 12 bytes are ASCII 'cctp-forward'."""
        data = encode_cctp_forward_hook_data(recipient=None)
        assert data[:12] == b"cctp-forward"

    def test_no_recipient_padding(self):
        """Bytes 12-23 are null (12 bytes of zero padding)."""
        data = encode_cctp_forward_hook_data(recipient=None)
        assert data[12:24] == b"\x00" * 12

    def test_no_recipient_version(self):
        """Bytes 24-27 encode version = 0 (big-endian uint32)."""
        data = encode_cctp_forward_hook_data(recipient=None)
        assert data[24:28] == b"\x00\x00\x00\x00"

    def test_no_recipient_data_length_zero(self):
        """Bytes 28-31 encode dataLength = 0 when no recipient is given."""
        data = encode_cctp_forward_hook_data(recipient=None)
        assert data[28:32] == b"\x00\x00\x00\x00"

    def test_with_recipient_total_length(self):
        """With recipient, hookData is 56 bytes (32 header + 20 addr + 4 dex)."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT)
        assert len(data) == 56

    def test_with_recipient_magic(self):
        """Magic prefix is preserved when recipient is provided."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT)
        assert data[:12] == b"cctp-forward"
        assert data[12:24] == b"\x00" * 12

    def test_with_recipient_data_length_24(self):
        """dataLength field encodes 24 (20-byte addr + 4-byte dex)."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT)
        assert int.from_bytes(data[28:32], "big") == 24

    def test_with_recipient_address_embedded(self):
        """Address bytes occupy bytes 32-51 of the hookData."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT)
        expected_addr = bytes.fromhex(_MOCK_RECIPIENT[2:])
        assert data[32:52] == expected_addr

    def test_perp_dex_default(self):
        """HYPERCORE_DEX_PERP (0) encodes as four zero bytes in the dex field."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT)
        assert data[52:56] == b"\x00\x00\x00\x00"

    def test_perp_dex_explicit(self):
        """Passing HYPERCORE_DEX_PERP explicitly gives the same result."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT, destination_dex=HYPERCORE_DEX_PERP)
        assert data[52:56] == b"\x00\x00\x00\x00"

    def test_spot_dex(self):
        """HYPERCORE_DEX_SPOT (0xFFFFFFFF) encodes as four 0xFF bytes."""
        data = encode_cctp_forward_hook_data(recipient=_MOCK_RECIPIENT, destination_dex=HYPERCORE_DEX_SPOT)
        assert data[52:56] == b"\xff\xff\xff\xff"

    def test_invalid_recipient_raises(self):
        """A non-20-byte hex string raises ValueError."""
        with pytest.raises(ValueError):
            encode_cctp_forward_hook_data(recipient="0x1234abcd")

    def test_address_without_0x_prefix(self):
        """Address without '0x' prefix is accepted and correctly embedded."""
        addr_no_prefix = _MOCK_RECIPIENT[2:]  # strip '0x'
        data = encode_cctp_forward_hook_data(recipient=addr_no_prefix)
        expected_addr = bytes.fromhex(addr_no_prefix)
        assert data[32:52] == expected_addr

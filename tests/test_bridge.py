"""Tests for pydefi.bridge (no live calls)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hexbytes import HexBytes

from pydefi.bridge.across import Across
from pydefi.bridge.base import BaseBridge
from pydefi.bridge.ccip import (
    _CCIP_CHAIN_SELECTOR,
    _CCIP_ROUTER,
    CCIP,
    EVM_EXTRA_ARGS_V2_TAG,
)
from pydefi.bridge.cctp import (
    HYPERCORE_DEX_PERP,
    HYPERCORE_DEX_SPOT,
    encode_cctp_forward_hook_data,
)
from pydefi.bridge.gaszip import _SUPPORTED_CHAINS, GasZip
from pydefi.bridge.layerzero_oft import _LZ_EID, LayerZeroOFT
from pydefi.bridge.lucid import LucidBridge
from pydefi.bridge.mayan import _CHAIN_NAMES, _MAYAN_FORWARDER, Mayan
from pydefi.bridge.relay import Relay
from pydefi.bridge.router import BridgeRouter, rank_bridge_quotes
from pydefi.bridge.stargate import _LZ_CHAIN_ID, _POOL_IDS, Stargate
from pydefi.exceptions import BridgeError
from pydefi.types import Address, BridgeQuote, ChainId, Token, TokenAmount
from tests.addrs import (
    ETH_WHALE,
    LUCID_HL_ADAPTER,
    LUCID_LZ_ADAPTER,
    LUCID_USDC_CTRL_KITE,
    LUCID_USDC_CTRL_MANTRA,
    USDC,
    USDC_BASE,
    USDC_E_AVAX,
    USDC_E_KITE,
    USDC_MANTRA,
    WETH,
)


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
    address=USDC.address,
    symbol="USDC",
    decimals=6,
)
USDC_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address=Address("0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"),
    symbol="USDC",
    decimals=6,
)

STARGATE_ROUTER_ETH: Address = Address("0x8731d54E9D02c286767d56ac03e8037C07e01e98")
SPOKE_POOL_ETH: Address = Address("0x5c7BCd6E7De5423a257D81B442095A1a6ced35C5")


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
        unknown_token = Token(chain_id=1, address=Address("0x" + "AB" * 20), symbol="UNKNOWN")
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
    address=Address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),
    symbol="ETH",
    decimals=18,
)
ETH_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address=Address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),
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
        recipient: Address = Address("0x" + "CC" * 20)
        weth: Address = WETH.address
        swift_contract: Address = Address("0x" + "AA" * 20)
        swap_router: Address = Address("0x" + "BB" * 20)

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
            await m.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, Address("0x" + "CC" * 20))

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
                await m.build_bridge_tx(ETH_NATIVE, USDC_ARB, amount_in, Address("0x" + "CC" * 20))


# ---------------------------------------------------------------------------
# GasZip tests
# ---------------------------------------------------------------------------

GASZIP_CONTRACT: Address = Address("0x" + "AA" * 20)


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
        recipient: Address = Address("0x" + "CC" * 20)

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
        recipient: Address = Address("0x" + "CC" * 20)

        mock_api_response = {"details": {}, "steps": []}

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            with pytest.raises(BridgeError):
                await r.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, recipient)

    @pytest.mark.asyncio
    async def test_build_bridge_tx_no_items_raises(self):
        r = Relay(src_chain_id=1, dst_chain_id=42161)
        amount_in = TokenAmount.from_human(USDC_ETH, "1000")
        recipient: Address = Address("0x" + "CC" * 20)

        mock_api_response = {"details": {}, "steps": [{"items": []}]}

        with patch.object(r, "_request_quote", new=AsyncMock(return_value=mock_api_response)):
            with pytest.raises(BridgeError):
                await r.build_bridge_tx(USDC_ETH, USDC_ARB, amount_in, recipient)


# ---------------------------------------------------------------------------
# LayerZeroOFT tests
# ---------------------------------------------------------------------------

OFT_ADDRESS: Address = Address("0x" + "DD" * 20)
OFT_DST_ADDRESS: Address = Address("0x" + "EE" * 20)

# Tokens for unified-address OFTs (same contract address on every chain,
# e.g. USDT0 at 0x1E4a5963aBFD975d8c9021ce480b42188849D41d).
OFT_TOKEN_ETH = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address(OFT_ADDRESS),
    symbol="OFT",
    decimals=18,
)
OFT_TOKEN_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address=Address(OFT_ADDRESS),
    symbol="OFT",
    decimals=18,
)
# Token for a non-unified OFT (different address on the destination chain).
OFT_TOKEN_ARB_ALT = Token(
    chain_id=ChainId.ARBITRUM,
    address=Address(OFT_DST_ADDRESS),
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
        from pydefi._utils import address_to_bytes32

        addr = HexBytes(ETH_WHALE)
        result = address_to_bytes32(addr)
        assert len(result) == 32
        # Address bytes should appear in the last 20 bytes
        assert result[:12] == b"\x00" * 12
        assert result[12:] == bytes(addr)

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
            address=Address("0x" + "FF" * 20),
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
            address=Address("0x" + "FF" * 20),
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
        # Mock the module-level LAYERZERO_OFT constant so we exercise the real
        # quote_send_fee logic (EID mapping, send_param construction, contract wiring)
        # rather than patching quote_send_fee itself.
        mock_call = AsyncMock(return_value=(5 * 10**15, 0))
        mock_quote_send = MagicMock(return_value=MagicMock(call=mock_call))
        mock_contract = MagicMock()
        mock_contract.fns.quoteSend = mock_quote_send

        with patch("pydefi.bridge.layerzero_oft.LAYERZERO_OFT", mock_contract):
            fee = await oft.quote_send_fee(1_000_000, Address("0x" + "AA" * 20))

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
        recipient: Address = Address("0x" + "AA" * 20)

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
        recipient: Address = Address("0x" + "AA" * 20)
        refund: Address = Address("0x" + "BB" * 20)

        with patch.object(oft, "quote_send_fee", new=AsyncMock(return_value=10**15)):
            tx = await oft.build_bridge_tx(OFT_TOKEN_ETH, OFT_TOKEN_ARB, amount_in, recipient, refund_address=refund)

        assert tx["to"] == OFT_ADDRESS
        # The refund address must be encoded into the calldata
        assert refund.hex() in tx["data"].lower()

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
            address=Address("0x" + "FF" * 20),
            symbol="WRONG",
            decimals=18,
        )
        amount_in = TokenAmount.from_human(wrong_token, "10")
        with pytest.raises(BridgeError, match="token_in"):
            with patch.object(oft, "quote_send_fee", new=AsyncMock(return_value=10**15)):
                await oft.build_bridge_tx(wrong_token, OFT_TOKEN_ARB, amount_in, Address("0x" + "AA" * 20))

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

_MOCK_RECIPIENT = ETH_WHALE


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
        expected_addr = bytes(_MOCK_RECIPIENT)
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
        """A non-20-byte address raises ValueError."""
        with pytest.raises(ValueError):
            encode_cctp_forward_hook_data(recipient=HexBytes("0x1234abcd"))

    def test_address_without_0x_prefix(self):
        """Address passed as bare hex bytes is accepted and correctly embedded."""
        addr_no_prefix = bytes(_MOCK_RECIPIENT).hex()  # bare hex (no 0x) from Address
        data = encode_cctp_forward_hook_data(recipient=HexBytes(addr_no_prefix))
        expected_addr = bytes.fromhex(addr_no_prefix)
        assert data[32:52] == expected_addr


# ---------------------------------------------------------------------------
# CCIP tests
# ---------------------------------------------------------------------------

# Canonical CCIP Router function selectors.
_CCIP_SEND_SELECTOR = "96f4e9f9"  # ccipSend(uint64,(bytes,bytes,(address,uint256)[],address,bytes))
_CCIP_GET_FEE_SELECTOR = "20487ded"  # getFee(uint64,(bytes,bytes,(address,uint256)[],address,bytes))


# ---- Shared CCIP fixtures and constants -----------------------------------

CCIP_USDC_ETH = Token(chain_id=ChainId.ETHEREUM, address=USDC.address, symbol="USDC", decimals=6)
CCIP_USDC_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address=Address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831"),
    symbol="USDC",
    decimals=6,
)
_CCIP_RECIPIENT: Address = Address("0x" + "AA" * 20)
_CCIP_COMPOSER: Address = Address("0x" + "CC" * 20)


@pytest.fixture
def ccip() -> CCIP:
    """Vanilla CCIP for Ethereum→Arbitrum, native fee."""
    return CCIP(w3=None, src_chain_id=1, dst_chain_id=42161)


@pytest.fixture
def amount_in() -> TokenAmount:
    """100 USDC on Ethereum side — the default amount across CCIP tests."""
    return TokenAmount.from_human(CCIP_USDC_ETH, "100")


def _patched_quote_fee(ccip: CCIP, fee: int):
    """Stub ``ccip.quote_fee`` so build_*_tx skips its on-chain call."""
    return patch.object(ccip, "quote_fee", new=AsyncMock(return_value=fee))


def _mock_ccip_router(call: AsyncMock) -> tuple[MagicMock, MagicMock]:
    """Build a fake ``CCIP_ROUTER`` module constant and return ``(router, get_fee_factory)``.

    Patch via ``patch("pydefi.bridge.ccip.CCIP_ROUTER", router)`` and inspect
    invocations through ``get_fee_factory.call_args``.
    """
    mock_get_fee = MagicMock(return_value=MagicMock(call=call))
    mock_router = MagicMock()
    mock_router.fns.getFee = mock_get_fee
    return mock_router, mock_get_fee


class TestCCIP:
    # -- construction + registry --------------------------------------------

    def test_protocol_name(self, ccip):
        assert ccip.protocol_name == "CCIP"

    def test_router_address_defaults_to_registry(self, ccip):
        assert ccip.router_address == _CCIP_ROUTER[1]

    def test_router_address_override(self):
        custom = "0x" + "11" * 20
        assert CCIP(w3=None, src_chain_id=1, dst_chain_id=42161, router_address=custom).router_address == custom

    def test_unknown_src_chain_raises(self):
        with pytest.raises(BridgeError, match="Router"):
            CCIP(w3=None, src_chain_id=999999, dst_chain_id=42161)

    def test_unknown_dst_chain_raises(self):
        with pytest.raises(BridgeError, match="chain selector"):
            CCIP(w3=None, src_chain_id=1, dst_chain_id=999999)

    def test_dst_chain_selector_resolved(self, ccip):
        assert ccip.dst_chain_selector == _CCIP_CHAIN_SELECTOR[42161]

    def test_fee_token_defaults_to_zero(self, ccip):
        """fee_token=None means pay in native gas (zero address)."""
        assert bytes(ccip.fee_token) == b"\x00" * 20

    # -- fee_token_decimals validation --------------------------------------

    def test_fee_token_decimals_defaults_to_18(self, ccip):
        assert ccip.fee_token_decimals == 18

    def test_fee_token_decimals_override(self):
        """Caller can specify a non-18-decimal fee token (e.g. USDC=6)."""
        c = CCIP(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            fee_token=Address("0x" + "A0" * 20),
            fee_token_decimals=6,
            fee_token_symbol="USDC",
        )
        assert c.fee_token_decimals == 6
        assert c.fee_token_symbol == "USDC"

    @pytest.mark.parametrize(
        "value,match",
        [
            (-1, r"\[0, 36\]"),
            (37, r"\[0, 36\]"),
            # bool subclasses int — common gotcha; explicit type check rejects it.
            (True, "must be an int"),
            (18.0, "must be an int"),
        ],
        ids=["negative", "above_36", "bool", "float"],
    )
    def test_fee_token_decimals_rejected(self, value, match):
        with pytest.raises(BridgeError, match=match):
            CCIP(w3=None, src_chain_id=1, dst_chain_id=42161, fee_token_decimals=value)

    def test_fee_token_decimals_accepts_zero(self):
        """0 is a legal precision (sub-units only)."""
        assert CCIP(w3=None, src_chain_id=1, dst_chain_id=42161, fee_token_decimals=0).fee_token_decimals == 0

    @pytest.mark.asyncio
    async def test_get_quote_uses_fee_token_decimals(self, amount_in):
        """The synthetic fee Token in bridge_fee carries the decimals override."""
        usdc = Address("0x" + "A0" * 20)
        c = CCIP(
            w3=None,
            src_chain_id=1,
            dst_chain_id=42161,
            fee_token=usdc,
            fee_token_decimals=6,
            fee_token_symbol="USDC",
        )
        with _patched_quote_fee(c, 1_500_000):
            q = await c.get_quote(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in)
        assert q.bridge_fee.token.decimals == 6
        assert q.bridge_fee.token.symbol == "USDC"
        assert bytes(q.bridge_fee.token.address) == bytes(usdc)

    # -- quote_fee on-chain wrapper -----------------------------------------

    @pytest.mark.asyncio
    async def test_quote_fee_calls_getfee(self, ccip, amount_in):
        """quote_fee calls Router.getFee and returns its uint256."""
        router_mock, get_fee_mock = _mock_ccip_router(AsyncMock(return_value=5 * 10**15))
        with patch("pydefi.bridge.ccip.CCIP_ROUTER", router_mock):
            fee = await ccip.quote_fee(CCIP_USDC_ETH, amount_in, _CCIP_RECIPIENT)
        assert fee == 5 * 10**15
        get_fee_mock.assert_called_once()
        # First positional arg is the destination chain selector.
        assert get_fee_mock.call_args.args[0] == _CCIP_CHAIN_SELECTOR[42161]

    @pytest.mark.asyncio
    async def test_quote_fee_wraps_revert(self, ccip, amount_in):
        """A reverting getFee call surfaces as BridgeError."""
        router_mock, _ = _mock_ccip_router(AsyncMock(side_effect=RuntimeError("execution reverted")))
        with patch("pydefi.bridge.ccip.CCIP_ROUTER", router_mock):
            with pytest.raises(BridgeError, match="getFee"):
                await ccip.quote_fee(CCIP_USDC_ETH, amount_in, _CCIP_RECIPIENT)

    # -- get_quote / build_bridge_tx ----------------------------------------

    @pytest.mark.asyncio
    async def test_get_quote(self, ccip, amount_in):
        """get_quote returns a 1:1 token amount with the fee separated."""
        with _patched_quote_fee(ccip, 5 * 10**15):
            q = await ccip.get_quote(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in)
        assert q.protocol == "CCIP"
        assert q.amount_out.amount == amount_in.amount  # CCIP is 1:1
        assert q.bridge_fee.amount == 5 * 10**15
        assert bytes(q.bridge_fee.token.address) == b"\x00" * 20  # native default
        assert q.estimated_time_seconds > 0

    @pytest.mark.asyncio
    async def test_build_bridge_tx_native_fee(self, ccip, amount_in):
        """Native fee → tx.value carries the fee, tx.to is the Router."""
        with _patched_quote_fee(ccip, 5 * 10**15):
            tx = await ccip.build_bridge_tx(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in, _CCIP_RECIPIENT)
        assert tx["to"] == _CCIP_ROUTER[1]
        assert tx["data"].startswith("0x" + _CCIP_SEND_SELECTOR)
        assert tx["value"] == str(5 * 10**15)
        assert int(tx["gas"]) > 0

    @pytest.mark.asyncio
    async def test_build_bridge_tx_erc20_fee(self, amount_in):
        """ERC-20 fee → tx.value is 0; caller approves fee token separately."""
        link = Address("0x" + "CC" * 20)
        c = CCIP(w3=None, src_chain_id=1, dst_chain_id=42161, fee_token=link)
        with _patched_quote_fee(c, 42 * 10**18):
            tx = await c.build_bridge_tx(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in, _CCIP_RECIPIENT)
        assert tx["value"] == "0"
        assert link.hex().lower() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_build_bridge_tx_embeds_recipient(self, ccip, amount_in):
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_tx(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in, _CCIP_RECIPIENT)
        assert _CCIP_RECIPIENT.hex().lower() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_build_bridge_tx_uses_chain_selector(self, ccip, amount_in):
        """The 64-bit dst chain selector is the first 32-byte arg after the function selector."""
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_tx(CCIP_USDC_ETH, CCIP_USDC_ARB, amount_in, _CCIP_RECIPIENT)
        body = tx["data"][2 + 8 :]  # strip 0x + selector
        assert int(body[:64], 16) == _CCIP_CHAIN_SELECTOR[42161]

    @pytest.mark.asyncio
    async def test_build_bridge_tx_with_compose_payload(self, ccip, amount_in):
        """A non-empty `data` payload is encoded into the message's data field."""
        program = b"\xde\xad\xbe\xef" * 8
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_tx(
                CCIP_USDC_ETH,
                CCIP_USDC_ARB,
                amount_in,
                _CCIP_RECIPIENT,
                data=program,
                gas_limit=200_000,
            )
        assert program.hex() in tx["data"].lower()

    @pytest.mark.parametrize(
        "gas_limit,allow_ooo,last_byte",
        [
            (200_000, True, b"\x01"),
            (0, False, b"\x00"),
        ],
        ids=["gas=200k,ooo=True", "gas=0,ooo=False"],
    )
    @pytest.mark.asyncio
    async def test_build_bridge_tx_extra_args_v2_layout(self, ccip, amount_in, gas_limit, allow_ooo, last_byte):
        """``extraArgs`` is encoded as 0x181dcf10 ++ abi.encode(uint256 gas, bool ooo).

        Verifies the v2 tag, the 68-byte total length, and that the gas_limit
        and allow_out_of_order_execution flag round-trip through the public
        ``build_bridge_tx`` calldata — covering what the deleted
        encode_ccip_extra_args_v2 helper used to verify in isolation.
        """
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_tx(
                CCIP_USDC_ETH,
                CCIP_USDC_ARB,
                amount_in,
                _CCIP_RECIPIENT,
                gas_limit=gas_limit,
                allow_out_of_order_execution=allow_ooo,
            )

        # The complete 68-byte extraArgs blob must appear verbatim in the calldata.
        expected = EVM_EXTRA_ARGS_V2_TAG + gas_limit.to_bytes(32, "big") + (b"\x00" * 31 + last_byte)
        assert len(expected) == 4 + 32 + 32
        assert EVM_EXTRA_ARGS_V2_TAG.hex() == "181dcf10"
        assert expected.hex() in tx["data"].lower()

    # -- canonical constants -------------------------------------------------

    @pytest.mark.parametrize(
        "chain_id,expected",
        [
            (1, 5009297550715157269),  # Ethereum
            (42161, 4949039107694359620),  # Arbitrum
            (8453, 15971525489660198786),  # Base
            (10, 3734403246176062136),  # Optimism
            (137, 4051577828743386545),  # Polygon
            (43114, 6433500567565415381),  # Avalanche
            (56, 11344663589394136015),  # BNB Chain
            (59144, 4627098889531055414),  # Linea
        ],
    )
    def test_chain_selector_constants(self, chain_id, expected):
        """Spot-check published CCIP chain-selector values."""
        assert _CCIP_CHAIN_SELECTOR[chain_id] == expected

    def test_router_function_selectors(self):
        """Verify the bound Router contract's encoded selectors match the published ones."""
        from pydefi.abi.bridge import CCIP_ROUTER, CCIPEVM2AnyMessage, CCIPEVMTokenAmount

        msg = CCIPEVM2AnyMessage(
            receiver=b"\x00" * 32,
            data=b"",
            tokenAmounts=(CCIPEVMTokenAmount(token="0x" + "aa" * 20, amount=0),),
            feeToken="0x" + "00" * 20,
            extraArgs=b"",
        )
        assert CCIP_ROUTER.fns.ccipSend(0, msg).data.hex().startswith(_CCIP_SEND_SELECTOR)
        assert CCIP_ROUTER.fns.getFee(0, msg).data.hex().startswith(_CCIP_GET_FEE_SELECTOR)


class TestCCIPCompose:
    """Tests for the CCIP compose flow (``build_bridge_compose_tx``)."""

    @pytest.mark.asyncio
    async def test_rejects_empty_program(self, ccip, amount_in):
        with _patched_quote_fee(ccip, 10**15):
            with pytest.raises(BridgeError, match="program"):
                await ccip.build_bridge_compose_tx(CCIP_USDC_ETH, amount_in, _CCIP_COMPOSER, program=b"")

    @pytest.mark.asyncio
    async def test_embeds_program_in_data(self, ccip, amount_in):
        """The DeFiVM bytecode appears verbatim inside the ccipSend calldata."""
        program = b"\xde\xad\xbe\xef" * 16
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_compose_tx(CCIP_USDC_ETH, amount_in, _CCIP_COMPOSER, program)
        assert tx["data"].startswith("0x" + _CCIP_SEND_SELECTOR)
        assert program.hex() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_receiver_is_composer(self, ccip, amount_in):
        with _patched_quote_fee(ccip, 10**15):
            tx = await ccip.build_bridge_compose_tx(
                CCIP_USDC_ETH,
                amount_in,
                _CCIP_COMPOSER,
                program=b"\x01\x02\x03\x04",
            )
        assert _CCIP_COMPOSER.hex().lower() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_quote_fee_called_with_program_and_gas_limit(self, ccip, amount_in):
        """quote_fee receives the program payload and the requested gas_limit."""
        program = b"\xaa" * 64
        spy = AsyncMock(return_value=10**15)
        with patch.object(ccip, "quote_fee", new=spy):
            await ccip.build_bridge_compose_tx(
                CCIP_USDC_ETH,
                amount_in,
                _CCIP_COMPOSER,
                program,
                gas_limit=1_500_000,
            )
        spy.assert_called_once()
        assert spy.call_args.kwargs["data"] == program
        assert spy.call_args.kwargs["gas_limit"] == 1_500_000

    @pytest.mark.asyncio
    async def test_native_fee_attaches_value(self, ccip, amount_in):
        with _patched_quote_fee(ccip, 7 * 10**15):
            tx = await ccip.build_bridge_compose_tx(
                CCIP_USDC_ETH,
                amount_in,
                _CCIP_COMPOSER,
                program=b"\xff" * 32,
            )
        assert tx["value"] == str(7 * 10**15)

    @pytest.mark.asyncio
    async def test_erc20_fee_token_zero_value(self, amount_in):
        """ERC-20 fee_token → tx.value is 0; caller approves fee token separately."""
        link = Address("0x" + "DD" * 20)
        c = CCIP(w3=None, src_chain_id=1, dst_chain_id=42161, fee_token=link)
        with _patched_quote_fee(c, 20 * 10**18):
            tx = await c.build_bridge_compose_tx(
                CCIP_USDC_ETH,
                amount_in,
                _CCIP_COMPOSER,
                program=b"\xff" * 32,
            )
        assert tx["value"] == "0"
        assert link.hex().lower() in tx["data"].lower()


# ---------------------------------------------------------------------------
# BridgeRouter / rank_bridge_quotes tests
# ---------------------------------------------------------------------------

_NATIVE_TOKEN = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0x" + "00" * 20),
    symbol="NATIVE",
    decimals=18,
)
_ROUTER_AMOUNT = TokenAmount(USDC_ETH, 100_000_000)


def _legacy_quote(amount_out: int, fee: int, protocol: str = "CCTP") -> BridgeQuote:
    """A bridge whose fee is denominated in token_in — fee already baked into amount_out."""
    return BridgeQuote(
        token_in=USDC_ETH,
        token_out=USDC_ARB,
        amount_in=TokenAmount(USDC_ETH, 100_000_000),
        amount_out=TokenAmount(USDC_ARB, amount_out),
        bridge_fee=TokenAmount(USDC_ETH, fee),
        estimated_time_seconds=180,
        protocol=protocol,
    )


def _ccip_native_quote(amount_in: int, native_fee: int) -> BridgeQuote:
    """A CCIP-style quote: amount_out == amount_in, fee in a separate native token."""
    return BridgeQuote(
        token_in=USDC_ETH,
        token_out=USDC_ARB,
        amount_in=TokenAmount(USDC_ETH, amount_in),
        amount_out=TokenAmount(USDC_ARB, amount_in),
        bridge_fee=TokenAmount(_NATIVE_TOKEN, native_fee),
        estimated_time_seconds=600,
        protocol="CCIP",
    )


def _native_fee_to_usdc(q: BridgeQuote) -> int | None:
    """Test-only fee converter: NATIVE → USDC at 3000 USDC/ETH; everything else → None."""
    if q.bridge_fee.token.symbol != "NATIVE":
        return None
    return q.bridge_fee.amount * 3000 * 10**6 // 10**18


def _mock_bridge(*, quote: BridgeQuote | None = None, side_effect: BaseException | None = None) -> MagicMock:
    """Build a fake BaseBridge whose ``get_quote`` returns a quote or raises."""
    b = MagicMock(spec=BaseBridge)
    if side_effect is not None:
        b.get_quote = AsyncMock(side_effect=side_effect)
    else:
        b.get_quote = AsyncMock(return_value=quote)
    return b


class TestRankBridgeQuotes:
    def test_legacy_quotes_compared_by_amount_out(self):
        """Bridges with fee in token_in are ranked directly by amount_out."""
        better = _legacy_quote(amount_out=99_900_000, fee=100_000, protocol="Across")
        worse = _legacy_quote(amount_out=99_500_000, fee=500_000, protocol="Stargate")
        ranked = rank_bridge_quotes([worse, better])
        assert [r.protocol for r in ranked] == ["Across", "Stargate"]
        assert all(r.fee_in_token_out_units == 0 and not r.fee_unranked for r in ranked)

    def test_unconverted_ccip_is_marked_unranked_and_last(self):
        legacy = _legacy_quote(amount_out=99_800_000, fee=200_000)
        ccip = _ccip_native_quote(amount_in=100_000_000, native_fee=2 * 10**15)
        ranked = rank_bridge_quotes([ccip, legacy])
        assert [r.protocol for r in ranked] == ["CCTP", "CCIP"]
        assert ranked[1].fee_unranked is True
        assert ranked[1].effective_amount_out == 100_000_000  # not penalised

    def test_ccip_with_fee_in_token_in_is_not_treated_as_netted(self):
        """CCIP fee in token_in (e.g. USDC-as-fee bridging USDC) has
        amount_out == amount_in but a separate non-zero fee — the fast-path
        must not fire.
        """
        q = BridgeQuote(
            token_in=USDC_ETH,
            token_out=USDC_ARB,
            amount_in=TokenAmount(USDC_ETH, 100_000_000),
            amount_out=TokenAmount(USDC_ARB, 100_000_000),  # CCIP is 1:1
            bridge_fee=TokenAmount(USDC_ETH, 500_000),  # 0.5 USDC fee on top
            estimated_time_seconds=600,
            protocol="CCIP",
        )

        # Without a converter the quote is unrankable, not silently fee-free.
        unranked = rank_bridge_quotes([q])[0]
        assert unranked.fee_unranked is True
        assert unranked.fee_in_token_out_units == 0

        # With a converter the fee is subtracted from amount_out.
        ranked = rank_bridge_quotes([q], fee_converter=lambda r: r.bridge_fee.amount)[0]
        assert ranked.fee_unranked is False
        assert ranked.fee_in_token_out_units == 500_000
        assert ranked.effective_amount_out == 100_000_000 - 500_000

    def test_converter_lets_ccip_compete(self):
        """0.002 ETH ≈ 6 USDC fee outweighs CCTP's 0.2 USDC margin."""
        legacy = _legacy_quote(amount_out=99_800_000, fee=200_000)
        ccip = _ccip_native_quote(amount_in=100_000_000, native_fee=2 * 10**15)
        ranked = rank_bridge_quotes([ccip, legacy], fee_converter=_native_fee_to_usdc)

        ccip_r = next(r for r in ranked if r.protocol == "CCIP")
        cctp_r = next(r for r in ranked if r.protocol == "CCTP")
        assert ccip_r.fee_in_token_out_units == 6_000_000
        assert ccip_r.effective_amount_out == 100_000_000 - 6_000_000
        assert cctp_r.effective_amount_out > ccip_r.effective_amount_out
        assert ranked[0].protocol == "CCTP"

    def test_converter_returning_none_marks_unranked(self):
        ccip = _ccip_native_quote(amount_in=100_000_000, native_fee=2 * 10**15)
        ranked = rank_bridge_quotes([ccip], fee_converter=lambda _: None)
        assert ranked[0].fee_unranked is True

    def test_ccip_wins_when_fee_is_small(self):
        """If the converted CCIP fee is small, CCIP can rank above legacy bridges."""
        legacy = _legacy_quote(amount_out=99_500_000, fee=500_000)  # 0.5% legacy fee
        ccip = _ccip_native_quote(amount_in=100_000_000, native_fee=10**14)  # ~0.3 USDC
        ranked = rank_bridge_quotes([legacy, ccip], fee_converter=_native_fee_to_usdc)
        assert ranked[0].protocol == "CCIP"
        assert ranked[0].effective_amount_out > ranked[1].effective_amount_out

    def test_unrankable_quotes_keep_input_order(self):
        """Unrankable quotes preserve insertion order regardless of amount_out."""
        q1 = _ccip_native_quote(amount_in=50_000_000, native_fee=10**15)
        q1.protocol = "first"
        q2 = _ccip_native_quote(amount_in=200_000_000, native_fee=10**15)
        q2.protocol = "second"
        assert [r.protocol for r in rank_bridge_quotes([q1, q2])] == ["first", "second"]

    def test_empty_input_returns_empty_list(self):
        assert rank_bridge_quotes([]) == []


class TestBridgeRouter:
    def test_requires_at_least_one_bridge(self):
        with pytest.raises(ValueError):
            BridgeRouter([])

    @pytest.mark.asyncio
    async def test_quote_all_calls_every_bridge(self):
        b1 = _mock_bridge(quote=_legacy_quote(amount_out=99_800_000, fee=200_000))
        b2 = _mock_bridge(quote=_ccip_native_quote(amount_in=100_000_000, native_fee=10**15))
        quotes = await BridgeRouter([b1, b2]).quote_all(USDC_ETH, USDC_ARB, _ROUTER_AMOUNT)
        assert len(quotes) == 2
        b1.get_quote.assert_called_once_with(USDC_ETH, USDC_ARB, _ROUTER_AMOUNT)
        b2.get_quote.assert_called_once_with(USDC_ETH, USDC_ARB, _ROUTER_AMOUNT)

    @pytest.mark.asyncio
    async def test_quote_all_skips_bridge_errors(self):
        """BridgeError from a single bridge does not poison the whole batch."""
        good = _mock_bridge(quote=_legacy_quote(amount_out=99_800_000, fee=200_000))
        bad = _mock_bridge(side_effect=BridgeError("lane unsupported"))
        quotes = await BridgeRouter([bad, good]).quote_all(USDC_ETH, USDC_ARB, _ROUTER_AMOUNT)
        assert [q.protocol for q in quotes] == ["CCTP"]

    @pytest.mark.asyncio
    async def test_quote_all_propagates_other_exceptions(self):
        """Non-BridgeError exceptions bubble up to the caller."""
        bad = _mock_bridge(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            await BridgeRouter([bad]).quote_all(USDC_ETH, USDC_ARB, _ROUTER_AMOUNT)


# ---------------------------------------------------------------------------
# LucidBridge tests
# ---------------------------------------------------------------------------

_RECIPIENT: Address = Address("0x" + "AA" * 20)
_REFUND: Address = Address("0x" + "BB" * 20)
_DEST_CTRL_STUB: Address = Address("0x" + "DD" * 20)

_MANTRA_CASE = SimpleNamespace(
    src=ChainId.MANTRA,
    dst=ChainId.BASE,
    src_token=USDC_MANTRA,
    dst_token=USDC_BASE,
    ctrl=LUCID_USDC_CTRL_MANTRA,
    adapter=LUCID_HL_ADAPTER,
    kind="hyperlane",
)
_KITE_CASE = SimpleNamespace(
    src=ChainId.KITE,
    dst=ChainId.AVALANCHE,
    src_token=USDC_E_KITE,
    dst_token=USDC_E_AVAX,
    ctrl=LUCID_USDC_CTRL_KITE,
    adapter=LUCID_LZ_ADAPTER,
    kind="layerzero",
)
_LUCID_CASES = [
    pytest.param(_MANTRA_CASE, id="mantra"),
    pytest.param(_KITE_CASE, id="kite"),
]


def _make_lucid(case: SimpleNamespace, **kw) -> LucidBridge:
    bridge = LucidBridge(
        w3=None,
        src_chain_id=case.src,
        dst_chain_id=case.dst,
        controller_address=case.ctrl,
        adapter_address=case.adapter,
        **kw,
    )
    bridge._cached_source_token = case.src_token.address  # skip the controller.token() RPC
    return bridge


@pytest.fixture(params=_LUCID_CASES)
def lucid(request) -> SimpleNamespace:
    c = request.param
    c.bridge = _make_lucid(c)
    return c


async def _build_tx(bridge: LucidBridge, src, dst, *, amount="100", fee=10**15, **kw):
    with patch.object(bridge, "quote_native_fee", new=AsyncMock(return_value=fee)):
        return await bridge.build_bridge_tx(src, dst, TokenAmount.from_human(src, amount), _RECIPIENT, **kw)


class TestLucidBridge:
    """Tests that pass identically for every supported source chain."""

    def test_construction_state(self, lucid):
        b = lucid.bridge
        assert b.protocol_name == "Lucid"
        assert b.adapter_kind == lucid.kind
        assert b.src_chain_id == lucid.src
        assert b.dst_chain_id == lucid.dst
        assert b.controller_address == lucid.ctrl
        assert b.adapter_address == lucid.adapter

    def test_encode_bridge_options_shape(self, lucid):
        # 64 bytes total; gasLimit lives in the low bytes of the second word
        # for both adapters (uint128 and uint256 differ only at >2^128, see
        # TestLucidAdapter.test_layerzero_rejects_oversized_gas).
        opts = lucid.bridge._encode_bridge_options(_REFUND, 300_000)
        assert len(opts) == 64
        assert int.from_bytes(opts[32:], "big") == 300_000

    @pytest.mark.asyncio
    async def test_get_quote_uses_native_fee(self, lucid):
        amount_in = TokenAmount.from_human(lucid.src_token, "1000")
        with patch.object(lucid.bridge, "quote_native_fee", new=AsyncMock(return_value=5 * 10**16)):
            quote = await lucid.bridge.get_quote(lucid.src_token, lucid.dst_token, amount_in)
        assert quote.protocol == "Lucid"
        assert quote.amount_out.amount == amount_in.amount
        assert quote.bridge_fee.amount == 5 * 10**16
        assert quote.bridge_fee.token.address == Address(b"\x00" * 20)
        assert quote.bridge_fee.token.symbol == "NATIVE"
        assert quote.bridge_fee.token.decimals == 18

    @pytest.mark.asyncio
    async def test_get_quote_passes_unwrap_through(self, lucid):
        mock_quote = AsyncMock(return_value=10**15)
        amount_in = TokenAmount.from_human(lucid.src_token, "1")
        with patch.object(lucid.bridge, "quote_native_fee", new=mock_quote):
            await lucid.bridge.get_quote(lucid.src_token, lucid.dst_token, amount_in, unwrap=True)
        assert mock_quote.await_args.args[2] is True

    @pytest.mark.asyncio
    async def test_build_bridge_tx_calldata_shape(self, lucid):
        from web3 import Web3

        tx = await _build_tx(lucid.bridge, lucid.src_token, lucid.dst_token, amount="1000", fee=5 * 10**16)
        assert tx["to"].lower() == "0x" + bytes(lucid.ctrl).hex()
        assert tx["data"].startswith("0x")
        sig = Web3.keccak(text="transferTo(address,uint256,bool,uint256,address,bytes)")[:4]
        assert tx["data"][2:10] == sig.hex()
        assert tx["value"] == str(5 * 10**16)
        assert int(tx["gas"]) > 0

    @pytest.mark.asyncio
    async def test_build_bridge_tx_refund_defaults_to_recipient(self, lucid):
        tx = await _build_tx(lucid.bridge, lucid.src_token, lucid.dst_token)
        # Recipient appears twice: as the transfer recipient and as refundAddress in bridgeOptions.
        assert tx["data"].lower().count(_RECIPIENT.hex()) == 2

    @pytest.mark.asyncio
    async def test_build_bridge_tx_explicit_refund(self, lucid):
        tx = await _build_tx(lucid.bridge, lucid.src_token, lucid.dst_token, refund_address=_REFUND)
        assert _RECIPIENT.hex() in tx["data"].lower()
        assert _REFUND.hex() in tx["data"].lower()

    @pytest.mark.asyncio
    async def test_build_bridge_tx_fee_buffer_bumps_value(self, lucid):
        tx = await _build_tx(lucid.bridge, lucid.src_token, lucid.dst_token, fee=1_000_000, fee_buffer_bps=1000)
        assert tx["value"] == str(1_100_000)  # +10%

    @pytest.mark.asyncio
    async def test_build_bridge_tx_dest_gas_in_options(self, lucid):
        bridge = _make_lucid(lucid, dest_gas_limit=750_000)
        tx = await _build_tx(bridge, lucid.src_token, lucid.dst_token, amount="1")
        # 750_000 == 0xb71b0 right-aligned in a 32-byte word (identical bytes for uint128 and uint256).
        assert "00000000000000000000000000000000000000000000000000000000000b71b0" in tx["data"].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["get_quote", "build_bridge_tx"])
    async def test_rejects_wrong_token_in(self, lucid, method):
        wrong = Token(chain_id=lucid.src, address=Address("0x" + "DE" * 20), symbol="WRONG", decimals=6)
        amount_in = TokenAmount.from_human(wrong, "1")
        args = [wrong, lucid.dst_token, amount_in]
        if method == "build_bridge_tx":
            args.append(_RECIPIENT)
        with pytest.raises(BridgeError, match="does not match controller token"):
            await getattr(lucid.bridge, method)(*args)

    @pytest.mark.asyncio
    async def test_assert_token_in_rejects_wrong_chain_id(self, lucid):
        # Same wrapped token bytes but tagged with the destination chain.
        cross_chain = Token(
            chain_id=lucid.dst,
            address=lucid.src_token.address,
            symbol=lucid.src_token.symbol,
            decimals=lucid.src_token.decimals,
        )
        with pytest.raises(BridgeError, match="token_in.chain_id"):
            await lucid.bridge._assert_token_in(cross_chain)

    @pytest.mark.asyncio
    async def test_assert_token_in_short_circuits_when_cached(self, lucid):
        # w3=None would AttributeError on any eth.call — a clean return proves the cache path.
        await lucid.bridge._assert_token_in(lucid.src_token)


class TestLucidAdapter:
    """Tests for the adapter switch, source-chain validation, and the actual
    uint128/uint256 discriminator — none of which generalize per-chain."""

    def test_encode_transfer_message_length(self):
        # 7 fixed-size words → 224 bytes. Chain-agnostic.
        msg = LucidBridge._encode_transfer_message(
            nonce=0,
            dest_chain_id=ChainId.BASE,
            recipient=_RECIPIENT,
            amount=10**6,
            unwrap=False,
        )
        assert len(msg) == 7 * 32

    def test_kite_testnet_source_allowed(self):
        bridge = LucidBridge(
            w3=None,
            src_chain_id=ChainId.KITE_TESTNET,
            dst_chain_id=ChainId.AVALANCHE,
            controller_address=LUCID_USDC_CTRL_KITE,
            adapter_address=LUCID_LZ_ADAPTER,
        )
        assert bridge.src_chain_id == ChainId.KITE_TESTNET
        assert bridge.adapter_kind == "layerzero"

    def test_explicit_adapter_kind_override(self):
        bridge = _make_lucid(_MANTRA_CASE, adapter_kind="layerzero")
        assert bridge.adapter_kind == "layerzero"

    def test_rejects_unknown_adapter_kind(self):
        with pytest.raises(BridgeError, match="unknown adapter_kind"):
            _make_lucid(_MANTRA_CASE, adapter_kind="wormhole")  # type: ignore[arg-type]

    def test_rejects_unsupported_source_chain(self):
        with pytest.raises(BridgeError, match="source chain must be Kite"):
            LucidBridge(
                w3=None,
                src_chain_id=ChainId.ETHEREUM,
                dst_chain_id=ChainId.BASE,
                controller_address=LUCID_USDC_CTRL_MANTRA,
                adapter_address=LUCID_HL_ADAPTER,
            )

    def test_layerzero_rejects_oversized_gas(self):
        """uint128 (LZ) overflows at 2^128 while uint256 (Hyperlane) accepts it.
        This is the only assertion that actually discriminates the two encodings
        (small gas values produce identical bytes either way)."""
        oversized = 2**129
        with pytest.raises(Exception):
            _make_lucid(_KITE_CASE)._encode_bridge_options(_REFUND, oversized)
        _make_lucid(_MANTRA_CASE)._encode_bridge_options(_REFUND, oversized)  # must not raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", [_MANTRA_CASE, _KITE_CASE], ids=["mantra", "kite"])
    async def test_quote_native_fee_uses_correct_adapter_abi(self, case):
        """The hyperlane/layerzero switch in quote_native_fee must hit the
        matching ABI — verified by mocking both quoteMessage callables."""
        from pydefi.abi import bridge as abi_mod

        bridge = _make_lucid(case)
        with (
            patch.object(bridge, "_dest_controller", new=AsyncMock(return_value=_DEST_CTRL_STUB)),
            patch.object(abi_mod.LUCID_HYPERLANE_ADAPTER.fns, "quoteMessage") as hl_fn,
            patch.object(abi_mod.LUCID_LAYERZERO_ADAPTER.fns, "quoteMessage") as lz_fn,
        ):
            hl_fn.return_value.call = AsyncMock(return_value=7 * 10**15)
            lz_fn.return_value.call = AsyncMock(return_value=3 * 10**15)
            await bridge.quote_native_fee(amount=10**6, recipient=Address(b"\xaa" * 20))

        if case.kind == "hyperlane":
            assert hl_fn.called and not lz_fn.called
        else:
            assert lz_fn.called and not hl_fn.called

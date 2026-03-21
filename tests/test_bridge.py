"""Tests for pydefi.bridge (no live calls)."""

from unittest.mock import AsyncMock, patch

import pytest

from pydefi.bridge.across import Across
from pydefi.bridge.base import BaseBridge
from pydefi.bridge.stargate import _LZ_CHAIN_ID, _POOL_IDS, Stargate
from pydefi.exceptions import BridgeError
from pydefi.types import ChainId, Token, TokenAmount

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

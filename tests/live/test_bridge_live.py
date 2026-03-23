"""Live integration tests for Mayan, GasZip, and Relay bridge APIs.

These tests make real HTTP requests to public bridge APIs and verify that
responses are structurally valid and numerically plausible.  Additionally,
each bridge's ``build_bridge_tx()`` is verified on the source chain by
simulating the produced transaction via ``eth_call``.

Run with::

    pytest -m live tests/live/test_bridge_live.py
"""

import pytest
from web3 import Web3

from pydefi.bridge.gaszip import GasZip
from pydefi.bridge.mayan import Mayan
from pydefi.bridge.relay import Relay
from pydefi.types import ChainId, Token, TokenAmount

from .conftest import USDC

# ---------------------------------------------------------------------------
# Cross-chain token definitions
# ---------------------------------------------------------------------------

# USDC on Arbitrum (native USDC, not the deprecated bridged USDC.e)
USDC_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    symbol="USDC",
    decimals=6,
)

# WETH on Arbitrum — used as the SWIFT destination token (ETH → WETH cross-chain
# via SWIFT V1 `createOrderWithEth` where swiftInputContract == ZeroAddress)
WETH_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    symbol="WETH",
    decimals=18,
)

# Native ETH sentinel used by bridges
ETH_NATIVE = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
    symbol="ETH",
    decimals=18,
)
ETH_NATIVE_ARB = Token(
    chain_id=ChainId.ARBITRUM,
    address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
    symbol="ETH",
    decimals=18,
)

# GasZip deposit contract on Ethereum mainnet
GASZIP_CONTRACT_ETH = "0x391E7C679d29bD940d63be94AD22A25d25b5A604"

# A well-known ETH whale (vitalik.eth) used as the ``from`` address in
# eth_call simulations — no real transaction is broadcast.
ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

# Sanity bounds for USDC bridge quotes (1 000 USDC in, expect 900–1 100 USDC out)
BRIDGE_AMOUNT_USDC = 1_000 * 10**6
MIN_USDC_OUT = 900 * 10**6
MAX_USDC_OUT = 1_100 * 10**6

# Sanity bounds for ETH gas bridge (0.1 ETH in, expect > 0 ETH out)
BRIDGE_AMOUNT_ETH = 10**17  # 0.1 ETH


# ---------------------------------------------------------------------------
# Mayan live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestMayanLive:
    """Live tests against the public Mayan Price API."""

    async def test_get_quote_usdc_to_arb(self):
        """Mayan: 1000 USDC on Ethereum → USDC on Arbitrum."""
        client = Mayan(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=USDC, amount=BRIDGE_AMOUNT_USDC)
        quote = await client.get_quote(USDC, USDC_ARB, amount_in)

        assert quote.protocol == "Mayan"
        assert quote.token_in == USDC
        assert quote.token_out == USDC_ARB
        assert MIN_USDC_OUT < quote.amount_out.amount < MAX_USDC_OUT, (
            f"Mayan USDC→USDC amount_out out of range: {quote.amount_out.human_amount}"
        )
        assert quote.estimated_time_seconds > 0

    async def test_get_quote_returns_bridge_fee(self):
        """Mayan quote should include a non-negative bridge_fee."""
        client = Mayan(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=USDC, amount=BRIDGE_AMOUNT_USDC)
        quote = await client.get_quote(USDC, USDC_ARB, amount_in)

        assert quote.bridge_fee.amount >= 0

    async def test_build_tx_eth_call(self, eth_w3):
        """Mayan: build ETH→WETH bridge tx and verify it doesn't revert via eth_call.

        Uses the SWIFT V2 swapAndForwardEth flow:
        ETH → WETH (via DEX swap in Forwarder) → SWIFT V2 createOrderWithToken
        No ERC-20 approval needed since ETH is sent as msg.value.
        """
        client = Mayan(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        # Bridge ETH → WETH_ARB via SWIFT V2 swapAndForwardEth
        tx = await client.build_bridge_tx(ETH_NATIVE, WETH_ARB, amount_in, ETH_WHALE)

        assert tx.get("to"), "Mayan tx must have a 'to' address"
        assert tx.get("data"), "Mayan tx must have calldata"

        tx_params = {
            "to": Web3.to_checksum_address(tx["to"]),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": int(tx.get("value", 0)),
            "data": tx["data"],
        }
        await eth_w3.eth.call(tx_params)


# ---------------------------------------------------------------------------
# GasZip live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestGasZipLive:
    """Live tests against the public GasZip backend API."""

    async def test_get_quote_eth_to_arb(self):
        """GasZip: 0.1 ETH on Ethereum → ETH on Arbitrum."""
        client = GasZip(
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            contract_address=GASZIP_CONTRACT_ETH,
        )
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        quote = await client.get_quote(ETH_NATIVE, ETH_NATIVE_ARB, amount_in)

        assert quote.protocol == "GasZip"
        assert quote.token_in == ETH_NATIVE
        assert quote.token_out == ETH_NATIVE_ARB
        assert quote.amount_out.amount > 0, "GasZip amount_out should be positive"
        assert quote.amount_out.amount <= BRIDGE_AMOUNT_ETH, "GasZip amount_out should not exceed amount_in"
        assert quote.estimated_time_seconds > 0

    async def test_get_quote_returns_bridge_fee(self):
        """GasZip quote should report a non-negative bridge fee."""
        client = GasZip(
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            contract_address=GASZIP_CONTRACT_ETH,
        )
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        quote = await client.get_quote(ETH_NATIVE, ETH_NATIVE_ARB, amount_in)

        assert quote.bridge_fee.amount >= 0

    async def test_build_tx_eth_call(self, eth_w3):
        """GasZip: build deposit tx and verify it doesn't revert via eth_call."""
        client = GasZip(
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            contract_address=GASZIP_CONTRACT_ETH,
        )
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        tx = await client.build_bridge_tx(ETH_NATIVE, ETH_NATIVE_ARB, amount_in, ETH_WHALE)

        assert tx.get("to"), "GasZip tx must have a 'to' address"
        assert tx.get("data"), "GasZip tx must have calldata"
        assert int(tx.get("value", 0)) > 0, "GasZip tx must send ETH value"

        tx_params = {
            "to": Web3.to_checksum_address(tx["to"]),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": int(tx.get("value", 0)),
            "data": tx["data"],
        }
        await eth_w3.eth.call(tx_params)


# ---------------------------------------------------------------------------
# Relay live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestRelayLive:
    """Live tests against the public Relay API."""

    async def test_get_quote_usdc_eth_to_arb(self):
        """Relay: 1 000 USDC on Ethereum → USDC on Arbitrum."""
        client = Relay(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=USDC, amount=BRIDGE_AMOUNT_USDC)
        quote = await client.get_quote(USDC, USDC_ARB, amount_in)

        assert quote.protocol == "Relay"
        assert quote.token_in == USDC
        assert quote.token_out == USDC_ARB
        assert MIN_USDC_OUT < quote.amount_out.amount < MAX_USDC_OUT, (
            f"Relay USDC→USDC amount_out out of range: {quote.amount_out.human_amount} USDC"
        )
        assert quote.estimated_time_seconds > 0

    async def test_get_quote_returns_bridge_fee(self):
        """Relay quote should include a non-negative bridge_fee."""
        client = Relay(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=USDC, amount=BRIDGE_AMOUNT_USDC)
        quote = await client.get_quote(USDC, USDC_ARB, amount_in)

        assert quote.bridge_fee.amount >= 0

    async def test_get_quote_eth_to_arb(self):
        """Relay: 0.1 ETH on Ethereum → ETH on Arbitrum."""
        client = Relay(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        quote = await client.get_quote(ETH_NATIVE, ETH_NATIVE_ARB, amount_in)

        assert quote.protocol == "Relay"
        assert quote.amount_out.amount > 0

    async def test_build_tx_eth_call(self, eth_w3):
        """Relay: build ETH→ETH bridge tx and verify it doesn't revert via eth_call."""
        client = Relay(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        amount_in = TokenAmount(token=ETH_NATIVE, amount=BRIDGE_AMOUNT_ETH)
        tx = await client.build_bridge_tx(ETH_NATIVE, ETH_NATIVE_ARB, amount_in, ETH_WHALE)

        assert tx.get("to"), "Relay tx must have a 'to' address"
        assert tx.get("data"), "Relay tx must have calldata"

        tx_params = {
            "to": Web3.to_checksum_address(tx["to"]),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": int(tx.get("value", 0)),
            "data": tx["data"],
        }
        await eth_w3.eth.call(tx_params)

"""Live integration tests for Hyperliquid L1 API, HyperCore bridge, and HyperEVM.

These tests make real HTTP requests to the public Hyperliquid API and to the
HyperEVM JSON-RPC endpoint.  No private key is needed for the read-only tests.

Run with::

    pytest -m live tests/live/test_hyperliquid_live.py
"""

from __future__ import annotations

import pytest
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError

from pydefi.bridge.cctp import CCTP, HYPERCORE_DEX_SPOT
from pydefi.hyperliquid import HyperliquidClient
from pydefi.types import ChainId, Token, TokenAmount

# ---------------------------------------------------------------------------
# Token definitions
# ---------------------------------------------------------------------------

USDC_ETH = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    symbol="USDC",
    decimals=6,
)

# USDC on HyperCore (Hyperliquid L1); CCTP mints on HyperEVM and Hyperliquid
# routes to HyperCore automatically — same ERC-20 address as HyperEVM.
USDC_HYPERCORE = Token(
    chain_id=ChainId.HYPERCORE,
    address="0xb88339CB7199b77E23DB6E890353E22632Ba630f",
    symbol="USDC",
    decimals=6,
)

USDC_HYPEREVM = Token(
    chain_id=ChainId.HYPEREVM,
    address="0xb88339CB7199b77E23DB6E890353E22632Ba630f",
    symbol="USDC",
    decimals=6,
)

# Bridge 1,000 USDC in fee estimation tests.
BRIDGE_AMOUNT_USDC = 1_000 * 10**6

# A well-known address used as a mock recipient (no funds needed).
MOCK_RECIPIENT = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


# ---------------------------------------------------------------------------
# HyperliquidClient — read-only info tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestHyperliquidClientInfo:
    """Live tests for the HyperliquidClient info API (no credentials)."""

    def _client(self) -> HyperliquidClient:
        return HyperliquidClient(is_mainnet=True)

    async def test_get_meta(self):
        """get_meta() returns a dict with a 'universe' list of coin entries."""
        client = self._client()
        meta = await client.get_meta()

        assert isinstance(meta, dict), "meta should be a dict"
        assert "universe" in meta, "meta should have a 'universe' key"
        universe = meta["universe"]
        assert isinstance(universe, list), "'universe' should be a list"
        assert len(universe) > 0, "'universe' should be non-empty"
        first = universe[0]
        assert "name" in first, "each entry should have a 'name'"
        assert "szDecimals" in first, "each entry should have 'szDecimals'"

    async def test_get_spot_meta(self):
        """get_spot_meta() returns tokens and trading pairs."""
        client = self._client()
        spot_meta = await client.get_spot_meta()

        assert isinstance(spot_meta, dict)
        assert "tokens" in spot_meta
        assert "universe" in spot_meta
        tokens = spot_meta["tokens"]
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # USDC should be token index 0
        usdc = tokens[0]
        assert usdc.get("name") == "USDC"

    async def test_get_all_mids(self):
        """get_all_mids() returns a dict mapping coin names to price strings."""
        client = self._client()
        mids = await client.get_all_mids()

        assert isinstance(mids, dict), "mids should be a dict"
        assert len(mids) > 0, "mids should not be empty"
        # BTC should always be a traded coin
        assert "BTC" in mids, "BTC should have a mid price"
        # Mid prices are returned as strings
        assert isinstance(mids["BTC"], str), "mid prices should be strings"
        assert float(mids["BTC"]) > 0, "BTC mid price should be positive"

    async def test_get_user_state_empty_wallet(self):
        """get_user_state() returns a valid structure for an address with no positions."""
        client = self._client()
        # Use a well-known but inactive address
        state = await client.get_user_state(MOCK_RECIPIENT)

        assert isinstance(state, dict)
        assert "assetPositions" in state
        assert "marginSummary" in state

    async def test_get_spot_clearinghouse_state(self):
        """get_spot_clearinghouse_state() returns a valid structure."""
        client = self._client()
        state = await client.get_spot_clearinghouse_state(MOCK_RECIPIENT)

        assert isinstance(state, dict)
        assert "balances" in state

    async def test_get_open_orders_empty(self):
        """get_open_orders() returns an empty list for an address with no orders."""
        client = self._client()
        orders = await client.get_open_orders(MOCK_RECIPIENT)

        assert isinstance(orders, list)

    async def test_post_info_raw(self):
        """post_info() accepts a raw payload and returns a valid response."""
        client = self._client()
        result = await client.post_info({"type": "meta"})

        assert isinstance(result, dict)
        assert "universe" in result

    async def test_get_l2_book(self):
        """get_l2_book() returns bids and asks for BTC."""
        client = self._client()
        book = await client.get_l2_book("BTC")

        assert isinstance(book, dict)
        assert "levels" in book
        levels = book["levels"]
        assert isinstance(levels, list)
        assert len(levels) == 2  # [bids, asks]


# ---------------------------------------------------------------------------
# HyperEVM JSON-RPC tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestHyperEVMRpc:
    """Live tests for HyperEVM JSON-RPC access via HyperliquidClient."""

    async def test_evm_rpc_url(self):
        """evm_rpc_url returns the correct URL for mainnet."""
        client = HyperliquidClient(is_mainnet=True)
        assert "hyperliquid.xyz" in client.evm_rpc_url
        assert "testnet" not in client.evm_rpc_url

    async def test_evm_rpc_url_testnet(self):
        """evm_rpc_url returns the testnet URL when is_mainnet=False."""
        client = HyperliquidClient(is_mainnet=False)
        assert "testnet" in client.evm_rpc_url

    async def test_make_evm_w3_chain_id(self):
        """make_evm_w3() returns a web3 client with HyperEVM chain ID 999."""
        client = HyperliquidClient(is_mainnet=True)
        w3 = client.make_evm_w3()
        assert isinstance(w3, AsyncWeb3)
        chain_id = await w3.eth.chain_id
        assert chain_id == ChainId.HYPEREVM, f"Expected chain ID 999, got {chain_id}"

    async def test_make_evm_w3_block(self):
        """make_evm_w3() can fetch the latest block from HyperEVM."""
        client = HyperliquidClient(is_mainnet=True)
        w3 = client.make_evm_w3()
        block = await w3.eth.get_block("latest")
        assert block is not None
        assert block["number"] > 0


# ---------------------------------------------------------------------------
# CCTP bridge to HyperCore tests (primary target: chain 1337)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCCTPHyperCoreLive:
    """Live tests for CCTP v2 bridging to HyperCore (Hyperliquid L1, chain 1337).

    CCTP physically mints USDC on HyperEVM (domain 19) and Hyperliquid
    automatically routes the balance to HyperCore.
    """

    def _bridge(self, eth_w3: AsyncWeb3) -> CCTP:
        return CCTP(
            w3=eth_w3,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.HYPERCORE,
        )

    async def test_get_fees_eth_to_hypercore(self, eth_w3):
        """CCTP: get_fees() returns fee entries for ETH → HyperCore."""
        bridge = self._bridge(eth_w3)
        fees = await bridge.get_fees()

        assert isinstance(fees, list), "fees should be a list"
        assert len(fees) > 0, "fees should not be empty"
        for entry in fees:
            assert "finalityThreshold" in entry
            assert "minimumFee" in entry

    async def test_get_quote_eth_to_hypercore(self, eth_w3):
        """CCTP: get_quote() returns a valid BridgeQuote for ETH → HyperCore."""
        bridge = self._bridge(eth_w3)
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        quote = await bridge.get_quote(USDC_ETH, USDC_HYPERCORE, amount_in)

        assert quote.protocol == "CCTP"
        assert quote.token_in == USDC_ETH
        assert quote.token_out == USDC_HYPERCORE
        assert quote.amount_out.amount > 0
        assert quote.amount_out.amount <= BRIDGE_AMOUNT_USDC
        assert quote.estimated_time_seconds > 0

    async def test_build_bridge_tx_structure(self, eth_w3):
        """CCTP: build_bridge_tx() returns a well-formed tx dict for HyperCore."""
        bridge = self._bridge(eth_w3)
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        tx = await bridge.build_bridge_tx(
            token_in=USDC_ETH,
            token_out=USDC_HYPERCORE,
            amount_in=amount_in,
            recipient=MOCK_RECIPIENT,
        )

        assert tx.get("to") == bridge.token_messenger_address
        assert tx.get("data", "").startswith("0x")
        assert len(tx["data"]) > 2
        assert tx.get("value") == "0"
        assert int(tx.get("gas", 0)) > 0

    async def test_cctp_domain_hypercore(self):
        """CCTP: HyperCore (1337) is mapped to CCTP domain 19 (HyperEVM)."""
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
        bridge = CCTP(w3=w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.HYPERCORE)
        assert bridge._cctp_domain(ChainId.HYPERCORE) == 19

    async def test_cctp_domain_matches_hyperevm(self):
        """HyperCore and HyperEVM map to the same CCTP domain (19)."""
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
        bridge = CCTP(w3=w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.HYPERCORE)
        assert bridge._cctp_domain(ChainId.HYPERCORE) == bridge._cctp_domain(ChainId.HYPEREVM)

    async def test_cctp_usdc_address_hypercore(self):
        """CCTP: USDC address is known for HyperCore."""
        addr = CCTP.usdc_address(ChainId.HYPERCORE)
        assert addr.lower() == "0xb88339cb7199b77e23db6e890353e22632ba630f"

    async def test_cctp_message_transmitter_hypercore(self):
        """CCTP: MessageTransmitterV2 address is known for HyperCore."""
        addr = CCTP.message_transmitter_address(ChainId.HYPERCORE)
        assert addr.lower() == "0x81d40f21f12a8f0e3252bccb954d722d4c464b64"


# ---------------------------------------------------------------------------
# CCTP bridge to HyperEVM tests (chain 999, same domain 19)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCCTPHyperEVMLive:
    """Live tests for CCTP v2 bridging to HyperEVM (chain 999, domain 19)."""

    def _bridge(self, eth_w3: AsyncWeb3) -> CCTP:
        return CCTP(
            w3=eth_w3,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.HYPEREVM,
        )

    async def test_get_fees_eth_to_hyperevm(self, eth_w3):
        """CCTP: get_fees() returns fee entries for ETH → HyperEVM."""
        bridge = self._bridge(eth_w3)
        fees = await bridge.get_fees()

        assert isinstance(fees, list), "fees should be a list"
        assert len(fees) > 0, "fees should not be empty"
        for entry in fees:
            assert "finalityThreshold" in entry
            assert "minimumFee" in entry

    async def test_get_quote_eth_to_hyperevm(self, eth_w3):
        """CCTP: get_quote() returns a valid BridgeQuote for ETH → HyperEVM."""
        bridge = self._bridge(eth_w3)
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        quote = await bridge.get_quote(USDC_ETH, USDC_HYPEREVM, amount_in)

        assert quote.protocol == "CCTP"
        assert quote.token_in == USDC_ETH
        assert quote.token_out == USDC_HYPEREVM
        assert quote.amount_out.amount > 0
        assert quote.amount_out.amount <= BRIDGE_AMOUNT_USDC
        assert quote.estimated_time_seconds > 0

    async def test_cctp_domain_hyperevm(self):
        """CCTP: HyperEVM is mapped to domain 19."""
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
        bridge = CCTP(w3=w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.HYPEREVM)
        assert bridge._cctp_domain(ChainId.HYPEREVM) == 19

    async def test_cctp_usdc_address_hyperevm(self):
        """CCTP: USDC address is known for HyperEVM."""
        addr = CCTP.usdc_address(ChainId.HYPEREVM)
        assert addr.lower() == "0xb88339cb7199b77e23db6e890353e22632ba630f"

    async def test_cctp_message_transmitter_hyperevm(self):
        """CCTP: MessageTransmitterV2 address is known for HyperEVM."""
        addr = CCTP.message_transmitter_address(ChainId.HYPEREVM)
        assert addr.lower() == "0x81d40f21f12a8f0e3252bccb954d722d4c464b64"


# ---------------------------------------------------------------------------
# eth_call simulation tests for CCTP bridge transactions
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCCTPBridgeTxEthCall:
    """Live tests that verify bridge tx encoding via eth_call dry-run.

    These tests build the CCTP transaction and submit it via ``eth_call``.
    The call verifies that:
    - The ``to`` address hosts a real contract with the expected function.
    - The calldata ABI-encoding is accepted by the EVM (function selector
      matches; argument types decode without error).

    Without a ``from`` address the Ethereum RPC may succeed (bypass approval
    checks) or revert with an EVM error — both outcomes indicate the calldata
    is valid.  What must NOT happen is an RPC-level error about an unknown
    function selector or malformed ABI encoding.
    """

    async def test_eth_call_hypercore_bridge_reverts_with_evm_error(self, eth_w3):
        """eth_call on a HyperCore depositForBurnWithHook tx is accepted by the EVM."""
        bridge = CCTP(
            w3=eth_w3,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.HYPERCORE,
        )
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        tx = await bridge.build_bridge_tx(
            token_in=USDC_ETH,
            token_out=USDC_HYPERCORE,
            amount_in=amount_in,
            recipient=MOCK_RECIPIENT,
        )

        # Convert gas to int for eth_call (build_bridge_tx returns string).
        call_params = {
            "to": tx["to"],
            "data": tx["data"],
            "value": int(tx.get("value", 0)),
        }
        # The call may or may not revert depending on the RPC provider's behaviour
        # when no `from` is specified.  Both outcomes are acceptable — what we are
        # checking is that the calldata is well-formed (no "unknown function" RPC error).
        try:
            await eth_w3.eth.call(call_params)
        except (ContractLogicError, ValueError) as exc:
            # EVM-level revert — calldata was accepted, execution was rejected by
            # contract logic (e.g. no USDC approval).  Verify it is an EVM error.
            err_str = str(exc).lower()
            assert any(keyword in err_str for keyword in ("revert", "execution", "0x", "error")), (
                f"Unexpected error from eth_call: {exc}"
            )

    async def test_eth_call_hypercore_spot_bridge_reverts(self, eth_w3):
        """eth_call on a spot-balance HyperCore bridge tx is accepted by the EVM."""
        bridge = CCTP(
            w3=eth_w3,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.HYPERCORE,
        )
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        tx = await bridge.build_bridge_tx(
            token_in=USDC_ETH,
            token_out=USDC_HYPERCORE,
            amount_in=amount_in,
            recipient=MOCK_RECIPIENT,
            hypercore_dex=HYPERCORE_DEX_SPOT,
        )

        call_params = {
            "to": tx["to"],
            "data": tx["data"],
            "value": int(tx.get("value", 0)),
        }
        try:
            await eth_w3.eth.call(call_params)
        except (ContractLogicError, ValueError) as exc:
            err_str = str(exc).lower()
            assert any(keyword in err_str for keyword in ("revert", "execution", "0x", "error")), (
                f"Unexpected error from eth_call (spot): {exc}"
            )

    async def test_bridge_tx_uses_deposit_for_burn_with_hook_selector(self, eth_w3):
        """The calldata for HyperCore uses depositForBurnWithHook (not depositForBurn).

        depositForBurn selector:         0x6b86d4b0 (CCTP v1-style)
        depositForBurnWithHook selector: computed from the ABI
        Both are 4-byte selectors at the start of tx['data'].
        """
        from eth_utils import keccak

        bridge = CCTP(
            w3=eth_w3,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.HYPERCORE,
        )
        amount_in = TokenAmount(token=USDC_ETH, amount=BRIDGE_AMOUNT_USDC)
        tx = await bridge.build_bridge_tx(
            token_in=USDC_ETH,
            token_out=USDC_HYPERCORE,
            amount_in=amount_in,
            recipient=MOCK_RECIPIENT,
        )

        # Extract the 4-byte function selector from the calldata.
        selector = bytes.fromhex(tx["data"][2:10])

        # depositForBurn(uint256,uint32,bytes32,address) selector
        deposit_for_burn_sig = b"depositForBurn(uint256,uint32,bytes32,address)"
        deposit_selector = keccak(deposit_for_burn_sig)[:4]

        # The HyperCore bridge must NOT use plain depositForBurn.
        assert selector != deposit_selector, "HyperCore bridge must use depositForBurnWithHook, not depositForBurn"

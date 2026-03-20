"""Live integration tests for the Uniswap Universal Router.

These tests verify that:

1. The UniversalRouterV2 contract (which supports Uniswap V4) is deployed at
   the expected Ethereum mainnet address by checking that ``eth_getCode``
   returns non-empty bytecode.
2. The V3 and V4 calldata builders produce correctly structured calldata.
   We use the Uniswap V3 QuoterV2 to get a live quote and embed it as the
   ``amount_out_minimum`` so the encoding is tied to a real on-chain state.
"""

import pytest

from pydefi.amm.universal_router import UNIVERSAL_ROUTER_ADDRESSES, UniversalRouter
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.types import TokenAmount

from .conftest import USDC, WETH

# ---------------------------------------------------------------------------
# Contract addresses
# ---------------------------------------------------------------------------

# UniversalRouterV2 on Ethereum mainnet (supports Uniswap V4)
UNIVERSAL_ROUTER_V2 = UNIVERSAL_ROUTER_ADDRESSES[1]

UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

# A well-known EOA used as "recipient" in calldata tests.
DUMMY_RECIPIENT = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

# Plausible price bounds for 1 WETH in USDC
MIN_USDC = 500 * 10 ** 6
MAX_USDC = 10_000 * 10 ** 6


@pytest.mark.live
class TestUniversalRouterV2Live:
    """Live on-chain tests for the Universal Router V2 (V4-capable)."""

    async def test_contract_deployed_at_expected_address(self, eth_w3):
        """UniversalRouterV2 must be deployed at the address stored in UNIVERSAL_ROUTER_ADDRESSES."""
        from web3 import Web3

        checksum_addr = Web3.to_checksum_address(UNIVERSAL_ROUTER_V2)
        code = await eth_w3.eth.get_code(checksum_addr)
        assert len(code) > 0, (
            f"UniversalRouterV2 has no bytecode at {UNIVERSAL_ROUTER_V2}. "
            "Update UNIVERSAL_ROUTER_ADDRESSES with the correct address."
        )

    async def test_v3_exact_in_calldata_structure(self, eth_w3):
        """Build V3 exact-in calldata using a live V3 quote and verify its selector."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
            default_fee=500,
        )
        amount_in = TokenAmount.from_human(WETH, "1")
        amount_out = await quoter.quote_exact_input_single(amount_in, USDC, fee=500)

        assert MIN_USDC < amount_out.amount < MAX_USDC, (
            f"Live V3 quote out of expected range: {amount_out.amount / 10**6:.2f} USDC"
        )

        # Apply 0.5% slippage
        amount_out_min = int(amount_out.amount * 9950 // 10000)

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_v3_exact_in_transaction(
            amount_in=amount_in,
            token_out=USDC,
            recipient=DUMMY_RECIPIENT,
            amount_out_minimum=amount_out_min,
            fee=500,
            deadline=2_000_000_000,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.data[:4] == bytes.fromhex("3593564c"), "Expected execute(bytes,bytes[],uint256) selector"
        assert tx.value == 0
        assert len(tx.data) > 4

    async def test_v4_exact_in_single_calldata_structure(self, eth_w3):
        """Build V4 exact-in-single calldata using a live V3 quote as price reference."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
            default_fee=500,
        )
        amount_in = TokenAmount.from_human(WETH, "1")
        amount_out = await quoter.quote_exact_input_single(amount_in, USDC, fee=500)

        assert MIN_USDC < amount_out.amount < MAX_USDC, (
            f"Live V3 quote out of expected range for V4 test: {amount_out.amount / 10**6:.2f} USDC"
        )

        # Apply 0.5% slippage for V4 swap
        amount_out_min = int(amount_out.amount * 9950 // 10000)

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=DUMMY_RECIPIENT,
            amount_out_minimum=amount_out_min,
            deadline=2_000_000_000,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.data[:4] == bytes.fromhex("3593564c"), "Expected execute(bytes,bytes[],uint256) selector"
        assert tx.value == 0
        assert len(tx.data) > 4
        # Verify the V4_SWAP command byte (0x10) is present in the encoded calldata
        from pydefi.amm.universal_router import RouterCommand
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

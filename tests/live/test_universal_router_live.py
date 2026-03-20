"""Live integration tests for the Uniswap Universal Router.

These tests:

1. Verify that the UniversalRouterV2 contract (which supports Uniswap V4) is
   deployed at the expected Ethereum mainnet address using ``eth_getCode``.
2. Execute simulated swaps via ``eth_call`` to confirm that the calldata
   produced by :class:`~pydefi.amm.universal_router.UniversalRouter` is
   accepted by the live contract without reverting.

Both swap tests use a WRAP_ETH-first pattern so that no Permit2 approvals
are required:

* **V3**: ``WRAP_ETH`` + ``V3_SWAP_EXACT_IN`` (router-funded, payer_is_user=False)
* **V4**: ``WRAP_ETH`` + ``V4_SWAP`` (SETTLE with payerIsUser=False)

A V3 QuoterV2 quote is fetched first to set a realistic ``amount_out_minimum``
with 0.5 % slippage.
"""

import pytest
from web3 import Web3

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

# Plausible price bounds for 1 WETH in USDC
MIN_USDC = 500 * 10 ** 6
MAX_USDC = 10_000 * 10 ** 6

# A well-known ETH whale used as the transaction sender in eth_call simulations.
# Using eth_call, no actual ETH is spent.
ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth

# Swap amount: 0.01 ETH to keep the simulated swap within any single-block
# gas limits while still being realistic.
ETH_SWAP_AMOUNT = 10 ** 16  # 0.01 ETH in wei


async def _get_v3_quote(eth_w3, eth_amount: int) -> int:
    """Return a live V3 quote (WETH→USDC) with 0.5 % slippage already applied."""
    quoter = UniswapV3(
        w3=eth_w3,
        router_address=UNISWAP_V3_ROUTER,
        quoter_address=UNISWAP_V3_QUOTER,
        default_fee=500,
    )
    weth_amount = TokenAmount(WETH, eth_amount)
    amount_out = await quoter.quote_exact_input_single(weth_amount, USDC, fee=500)
    # Scale the per-ETH bounds down to the actual swap amount
    min_expected = MIN_USDC * ETH_SWAP_AMOUNT // 10 ** 18
    max_expected = MAX_USDC * ETH_SWAP_AMOUNT // 10 ** 18
    assert min_expected < amount_out.amount < max_expected, (
        f"V3 quote out of expected range: {amount_out.amount / 10**6:.4f} USDC "
        f"for {eth_amount / 10**18} WETH (expected {min_expected / 10**6:.2f}–{max_expected / 10**6:.2f} USDC)"
    )
    # Apply 0.5 % slippage tolerance
    return int(amount_out.amount * 9950 // 10000)


@pytest.mark.live
class TestUniversalRouterV2Live:
    """Live on-chain tests for the Universal Router V2 (V4-capable)."""

    async def test_contract_deployed_at_expected_address(self, eth_w3):
        """UniversalRouterV2 must be deployed at the address stored in UNIVERSAL_ROUTER_ADDRESSES."""
        checksum_addr = Web3.to_checksum_address(UNIVERSAL_ROUTER_V2)
        code = await eth_w3.eth.get_code(checksum_addr)
        assert len(code) > 0, (
            f"UniversalRouterV2 has no bytecode at {UNIVERSAL_ROUTER_V2}. "
            "Update UNIVERSAL_ROUTER_ADDRESSES with the correct address."
        )

    async def test_v3_wrap_and_swap_via_eth_call(self, eth_w3):
        """Simulate WRAP_ETH + V3 exact-in swap via eth_call.

        Builds a ``WRAP_ETH + V3_SWAP_EXACT_IN`` transaction (no Permit2 needed)
        and executes it as an ``eth_call`` against the live UniversalRouterV2
        contract.  A successful call (no revert) confirms that the calldata
        produced by the builder is structurally and semantically valid.
        """
        amount_out_min = await _get_v3_quote(eth_w3, ETH_SWAP_AMOUNT)

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v3_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            recipient=ETH_WHALE,
            amount_out_minimum=amount_out_min,
            fee=500,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.value == ETH_SWAP_AMOUNT

        # Execute via eth_call – raises ContractLogicError / ValueError on revert
        result = await eth_w3.eth.call(
            {
                "to": Web3.to_checksum_address(tx.to),
                "from": Web3.to_checksum_address(ETH_WHALE),
                "value": tx.value,
                "data": tx.data,
            }
        )
        # execute() returns no value; an empty-bytes result means success
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"

    async def test_v4_wrap_and_swap_via_eth_call(self, eth_w3):
        """Simulate WRAP_ETH + V4 exact-in swap via eth_call.

        Builds a ``WRAP_ETH + V4_SWAP`` transaction where the router settles
        WETH from its own balance (no Permit2 needed) and executes it as an
        ``eth_call`` against the live UniversalRouterV2 contract.

        Pool parameters: WETH/USDC, fee=500 (0.05 %), tickSpacing=10.
        These are the expected parameters for the most-liquid V4 WETH/USDC pool.
        """
        amount_out_min = await _get_v3_quote(eth_w3, ETH_SWAP_AMOUNT)

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v4_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=ETH_WHALE,
            amount_out_minimum=amount_out_min,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.value == ETH_SWAP_AMOUNT

        # Execute via eth_call – raises ContractLogicError / ValueError on revert
        result = await eth_w3.eth.call(
            {
                "to": Web3.to_checksum_address(tx.to),
                "from": Web3.to_checksum_address(ETH_WHALE),
                "value": tx.value,
                "data": tx.data,
            }
        )
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"


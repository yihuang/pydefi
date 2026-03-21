"""Live integration tests for the Uniswap Universal Router.

These tests:

1. Verify that the UniversalRouterV2 contract (which supports Uniswap V4) is
   deployed at the expected Ethereum mainnet address using ``eth_getCode``.
2. Execute simulated swaps via ``eth_call`` to confirm that the calldata
   produced by :class:`~pydefi.amm.universal_router.UniversalRouter` is
   accepted by the live contract without reverting.

All swap tests use a WRAP_ETH-first pattern so that no Permit2 approvals
are required:

* **V3**: ``WRAP_ETH`` + ``V3_SWAP_EXACT_IN`` (router-funded, payer_is_user=False)
* **V4 single-hop**: ``WRAP_ETH`` + ``V4_SWAP`` (SETTLE with payerIsUser=False)
* **V4 multi-hop**: ``WRAP_ETH`` + ``V4_SWAP`` (SWAP_EXACT_IN action, router-funded)
* **Cross-type V3→V4**: ``WRAP_ETH`` + ``V3_SWAP_EXACT_IN`` + ``V4_SWAP``

A V3 QuoterV2 quote is fetched first to set a realistic ``amount_out_minimum``
with 0.5 % slippage.
"""

import json

import pytest
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from web3 import Web3

from pydefi.amm.universal_router import UNIVERSAL_ROUTER_ADDRESSES, UniversalRouter, V3Hop, V4Hop
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

# Uniswap V4 PoolManager on Ethereum mainnet
V4_POOL_MANAGER = "0x000000000004444c5dc75cB358380D2e3dE08A90"

# Plausible price bounds for 1 WETH in USDC
MIN_USDC = 500 * 10 ** 6
MAX_USDC = 10_000 * 10 ** 6

# A well-known ETH whale used as the transaction sender in eth_call simulations.
# Using eth_call, no actual ETH is spent.
ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth

# Swap amount: 0.01 ETH to keep the simulated swap within any single-block
# gas limits while still being realistic.
ETH_SWAP_AMOUNT = 10 ** 16  # 0.01 ETH in wei

# Standard V4 hookless fee tiers tried in order of popularity when auto-discovering
# an initialized WETH/USDC pool.
_V4_FEE_TIERS = [(500, 10), (3000, 60), (100, 1), (10000, 200)]

# keccak256("getSlot0(bytes32)")[:4]
_GET_SLOT0_SELECTOR: bytes = Web3.keccak(text="getSlot0(bytes32)")[:4]


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


def _compute_v4_pool_id(
    currency0: str,
    currency1: str,
    fee: int,
    tick_spacing: int,
    hooks: str = "0x0000000000000000000000000000000000000000",
) -> bytes:
    """Return the 32-byte PoolId = keccak256(abi.encode(PoolKey))."""
    return Web3.keccak(
        abi_encode(
            ["address", "address", "uint24", "int24", "address"],
            [
                Web3.to_checksum_address(currency0),
                Web3.to_checksum_address(currency1),
                fee,
                tick_spacing,
                Web3.to_checksum_address(hooks),
            ],
        )
    )


async def _v4_pool_sqrt_price(
    eth_w3,
    currency0: str,
    currency1: str,
    fee: int,
    tick_spacing: int,
) -> int:
    """Return sqrtPriceX96 for a V4 pool, or 0 if the pool is uninitialized."""
    pool_id = _compute_v4_pool_id(currency0, currency1, fee, tick_spacing)
    call_data = _GET_SLOT0_SELECTOR + pool_id
    try:
        result = await eth_w3.eth.call(
            {
                "to": Web3.to_checksum_address(V4_POOL_MANAGER),
                "data": "0x" + call_data.hex(),
            }
        )
        if len(result) >= 32:
            # getSlot0 returns (uint160 sqrtPriceX96, int24 tick, uint24, uint24)
            # sqrtPriceX96 is the first uint160, stored right-aligned in 32 bytes
            (sqrt_price,) = abi_decode(["uint160"], result[:32])
            return sqrt_price
    except Exception:
        pass
    return 0


async def _find_v4_pool(eth_w3, currency0: str, currency1: str) -> tuple[int, int] | None:
    """Return (fee, tick_spacing) for the first initialized hookless pool, or None."""
    for fee, tick_spacing in _V4_FEE_TIERS:
        sqrt_price = await _v4_pool_sqrt_price(eth_w3, currency0, currency1, fee, tick_spacing)
        if sqrt_price > 0:
            return fee, tick_spacing
    return None


async def _debug_trace_call(eth_w3, tx_params: dict, label: str = "") -> None:
    """Call ``debug_traceCall`` and print the full trace to help diagnose reverts.

    This is a best-effort helper — if the RPC node does not support the
    ``debug`` namespace it logs a short error instead of raising.

    Args:
        eth_w3: Async Web3 instance.
        tx_params: Transaction dict (``to``, ``from``, ``value``, ``data``).
        label: Short description printed in the header line.
    """
    try:
        response = await eth_w3.provider.make_request(
            "debug_traceCall",
            [
                tx_params,
                "latest",
                {"tracer": "callTracer"},
            ],
        )
        print(
            f"\n=== debug_traceCall [{label}] ===\n"
            f"{json.dumps(response, indent=2, default=str)}\n"
            f"=== end trace ==="
        )
    except Exception as exc:
        print(f"\n=== debug_traceCall [{label}] unavailable: {exc} ===")


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

        tx_params = {
            "to": Web3.to_checksum_address(tx.to),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": tx.value,
            "data": tx.data,
        }
        try:
            result = await eth_w3.eth.call(tx_params)
        except Exception as exc:
            await _debug_trace_call(eth_w3, tx_params, "V3 WETH->USDC swap revert")
            raise
        # execute() returns no value; an empty-bytes result means success
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"

    async def test_v4_wrap_and_swap_via_eth_call(self, eth_w3):
        """Simulate WRAP_ETH + V4 exact-in swap via eth_call.

        Builds a ``WRAP_ETH + V4_SWAP`` transaction where the router settles
        WETH from its own balance (no Permit2 needed) and executes it as an
        ``eth_call`` against the live UniversalRouterV2 contract.

        The test auto-discovers the first initialized hookless WETH/USDC V4 pool
        by probing ``getSlot0`` for the standard fee tiers in order of
        popularity: 0.05 % (500/10), 0.3 % (3000/60), 0.01 % (100/1),
        1 % (10000/200).  If no pool is found the test is skipped.
        """
        amount_out_min = await _get_v3_quote(eth_w3, ETH_SWAP_AMOUNT)

        currency0, currency1 = UniversalRouter._sort_v4_currencies(
            WETH.address, USDC.address
        )

        pool_params = await _find_v4_pool(eth_w3, currency0, currency1)
        if pool_params is None:
            pytest.skip(
                f"No initialized V4 WETH/USDC hookless pool found on mainnet "
                f"(tried fee tiers: {_V4_FEE_TIERS})"
            )
        fee, tick_spacing = pool_params

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v4_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            fee=fee,
            tick_spacing=tick_spacing,
            recipient=ETH_WHALE,
            amount_out_minimum=amount_out_min,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.value == ETH_SWAP_AMOUNT

        tx_params = {
            "to": Web3.to_checksum_address(tx.to),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": tx.value,
            "data": tx.data,
        }
        try:
            result = await eth_w3.eth.call(tx_params)
        except Exception as exc:
            await _debug_trace_call(
                eth_w3,
                tx_params,
                f"V4 WETH->USDC swap revert (fee={fee}, tickSpacing={tick_spacing})",
            )
            raise
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"


    async def test_v4_multihop_single_hop_wrap_and_swap_via_eth_call(self, eth_w3):
        """Simulate WRAP_ETH + V4 multi-hop (SWAP_EXACT_IN, single-hop) via eth_call.

        Uses :meth:`~pydefi.amm.universal_router.UniversalRouter.build_wrap_and_v4_multihop_swap_transaction`
        which exercises the ``SWAP_EXACT_IN`` action and
        :meth:`~pydefi.amm.universal_router.UniversalRouter.encode_v4_exact_in_params`.

        The test auto-discovers an initialized hookless WETH/USDC V4 pool and
        skips if none is found.
        """
        amount_out_min = await _get_v3_quote(eth_w3, ETH_SWAP_AMOUNT)

        currency0, currency1 = UniversalRouter._sort_v4_currencies(
            WETH.address, USDC.address
        )
        pool_params = await _find_v4_pool(eth_w3, currency0, currency1)
        if pool_params is None:
            pytest.skip(
                f"No initialized V4 WETH/USDC hookless pool found on mainnet "
                f"(tried fee tiers: {_V4_FEE_TIERS})"
            )
        fee, tick_spacing = pool_params

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=fee, tick_spacing=tick_spacing)],
            recipient=ETH_WHALE,
            amount_out_minimum=amount_out_min,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.value == ETH_SWAP_AMOUNT

        tx_params = {
            "to": Web3.to_checksum_address(tx.to),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": tx.value,
            "data": tx.data,
        }
        try:
            result = await eth_w3.eth.call(tx_params)
        except Exception:
            await _debug_trace_call(
                eth_w3,
                tx_params,
                f"V4 multi-hop WETH->USDC swap revert (fee={fee}, tickSpacing={tick_spacing})",
            )
            raise
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"

    async def test_cross_type_v3_v4_wrap_and_swap_via_eth_call(self, eth_w3):
        """Simulate WRAP_ETH + V3(WETH→USDC) + V4(USDC→WETH) via eth_call.

        Uses :meth:`~pydefi.amm.universal_router.UniversalRouter.build_wrap_and_multihop_exact_in_transaction`
        with a two-hop cross-type path: a V3 hop for the first leg and a V4
        hop for the second leg.

        The V4 leg reuses the same WETH/USDC pool in the reverse direction
        (USDC→WETH) so no additional pool discovery is required.  The test
        skips if no V4 WETH/USDC pool is found.

        ``amount_out_minimum`` is set to 1 (wei) since the round-trip incurs
        fees from both pools and is not intended to be profitable.
        """
        currency0, currency1 = UniversalRouter._sort_v4_currencies(
            WETH.address, USDC.address
        )
        pool_params = await _find_v4_pool(eth_w3, currency0, currency1)
        if pool_params is None:
            pytest.skip(
                f"No initialized V4 WETH/USDC hookless pool found on mainnet "
                f"(tried fee tiers: {_V4_FEE_TIERS})"
            )
        v4_fee, v4_tick_spacing = pool_params

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            hops=[
                # Leg 1: WETH → USDC via V3 (fee=500)
                V3Hop(token_in=WETH, token_out=USDC, fee=500),
                # Leg 2: USDC → WETH via V4 (reverse direction on the same pool)
                V4Hop(
                    token_in=USDC,
                    token_out=WETH,
                    fee=v4_fee,
                    tick_spacing=v4_tick_spacing,
                ),
            ],
            recipient=ETH_WHALE,
            # Accept any amount back: this is a round-trip with fees.
            amount_out_minimum=1,
        )

        assert tx.to == UNIVERSAL_ROUTER_V2
        assert tx.value == ETH_SWAP_AMOUNT

        tx_params = {
            "to": Web3.to_checksum_address(tx.to),
            "from": Web3.to_checksum_address(ETH_WHALE),
            "value": tx.value,
            "data": tx.data,
        }
        try:
            result = await eth_w3.eth.call(tx_params)
        except Exception:
            await _debug_trace_call(
                eth_w3,
                tx_params,
                f"Cross-type V3→V4 WETH->USDC->WETH swap revert "
                f"(v4_fee={v4_fee}, v4_tickSpacing={v4_tick_spacing})",
            )
            raise
        assert result == b"", f"Unexpected non-empty return data: {result.hex()}"

"""Fork-based transaction execution tests for the Uniswap Universal Router.

How the tests work
------------------
1. An Ethereum whale account (``ETH_WHALE``) that already holds plenty of ETH
   on mainnet is *impersonated* via ``anvil_impersonateAccount``, so we can
   send transactions on its behalf without needing a private key.
2. A ``WRAP_ETH + V3_SWAP_EXACT_IN`` or ``WRAP_ETH + V4_SWAP`` transaction is
   submitted with ``eth_sendTransaction`` (not ``eth_call``).
3. The test waits for the transaction receipt and verifies:
   - The transaction was *not* reverted (``status == 1``).
   - The recipient's output token balance increased by at least
     ``amount_out_minimum``.

Run with::

    pytest -m fork -v
"""

import pytest
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3, Web3

from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.amm.universal_router import UNIVERSAL_ROUTER_ADDRESSES, UniversalRouter, V4Hop
from pydefi.types import TokenAmount
from tests.addrs import ETH_WHALE, UNISWAP_V3_QUOTER, UNISWAP_V3_ROUTER, UNISWAP_V4_POOL_MANAGER, USDC, WETH, ZERO_ADDR

# ---------------------------------------------------------------------------
# Contract addresses
# ---------------------------------------------------------------------------

UNIVERSAL_ROUTER_V2 = UNIVERSAL_ROUTER_ADDRESSES[1]
V4_POOL_MANAGER = UNISWAP_V4_POOL_MANAGER

# 0.01 ETH in wei
ETH_SWAP_AMOUNT = 10**16

# Minimum plausible USDC output for 0.01 ETH (at worst $500/ETH = $5)
MIN_USDC_OUT = 5 * 10**6

# Standard V4 hookless fee tiers tried in order of popularity
_V4_FEE_TIERS = [(500, 10), (3000, 60), (100, 1), (10_000, 200)]

# keccak256("getSlot0(bytes32)")[:4]
_GET_SLOT0_SELECTOR: bytes = Web3.keccak(text="getSlot0(bytes32)")[:4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _impersonate(w3: AsyncWeb3, address: str) -> None:
    """Ask Anvil to impersonate *address* so we can send transactions as it."""
    await w3.provider.make_request("anvil_impersonateAccount", [address])


async def _get_v3_quote(w3: AsyncWeb3, eth_amount: int) -> int:
    """Return a V3 WETH→USDC quote with 0.5 % slippage applied."""
    quoter = UniswapV3(
        w3=w3,
        router_address=UNISWAP_V3_ROUTER,
        quoter_address=UNISWAP_V3_QUOTER,
        default_fee=500,
    )
    weth_amount = TokenAmount(WETH, eth_amount)
    amount_out = await quoter.quote_exact_input_single(weth_amount, USDC, fee=500)
    return int(amount_out.amount * 9950 // 10000)


def _compute_v4_pool_id(
    currency0: str,
    currency1: str,
    fee: int,
    tick_spacing: int,
    hooks: str = ZERO_ADDR.to_0x_hex(),
) -> bytes:
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


async def _v4_pool_sqrt_price(w3: AsyncWeb3, currency0: str, currency1: str, fee: int, tick_spacing: int) -> int:
    pool_id = _compute_v4_pool_id(currency0, currency1, fee, tick_spacing)
    call_data = _GET_SLOT0_SELECTOR + pool_id
    try:
        result = await w3.eth.call(
            {
                "to": Web3.to_checksum_address(V4_POOL_MANAGER),
                "data": "0x" + call_data.hex(),
            }
        )
        if len(result) >= 32:
            (sqrt_price,) = abi_decode(["uint160"], result[:32])
            return sqrt_price
    except Exception:
        pass
    return 0


async def _find_v4_pool(w3: AsyncWeb3, currency0: str, currency1: str) -> tuple[int, int] | None:
    """Return (fee, tick_spacing) for the first initialized hookless pool, or None."""
    for fee, tick_spacing in _V4_FEE_TIERS:
        sqrt_price = await _v4_pool_sqrt_price(w3, currency0, currency1, fee, tick_spacing)
        if sqrt_price > 0:
            return fee, tick_spacing
    return None


async def _send_and_mine(w3: AsyncWeb3, tx: dict) -> dict:
    """Submit *tx* via ``eth_sendTransaction`` and return its receipt."""
    tx_hash = await w3.eth.send_transaction(tx)
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestTxExecutionFork:
    """Transaction execution tests against a local Anvil fork of Ethereum mainnet.

    Each test impersonates ``ETH_WHALE`` (no private key required), submits a
    real transaction, and verifies the on-chain state change.
    """

    async def test_v3_wrap_and_swap_tx_execution(self, fork_w3):
        """Execute WRAP_ETH + V3_SWAP_EXACT_IN and confirm the USDC balance increases.

        This test verifies that the calldata produced by
        :meth:`~pydefi.amm.universal_router.UniversalRouter.build_wrap_and_v3_swap_transaction`
        actually succeeds when sent as a real transaction to the Universal Router,
        and that the recipient's USDC balance increases by at least the quoted
        minimum.
        """
        amount_out_min = await _get_v3_quote(fork_w3, ETH_SWAP_AMOUNT)

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v3_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            token_out=USDC,
            recipient=ETH_WHALE,
            amount_out_minimum=amount_out_min,
            fee=500,
        )

        await _impersonate(fork_w3, ETH_WHALE)
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)

        receipt = await _send_and_mine(
            fork_w3,
            {
                "to": Web3.to_checksum_address(tx.to),
                "from": Web3.to_checksum_address(ETH_WHALE),
                "value": tx.value,
                "data": tx.data,
            },
        )

        assert receipt["status"] == 1, f"Transaction reverted: {receipt['transactionHash'].hex()}"

        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        usdc_received = usdc_after - usdc_before

        assert usdc_received >= amount_out_min, (
            f"USDC received ({usdc_received / 10**6:.4f}) is less than the quoted "
            f"minimum ({amount_out_min / 10**6:.4f})"
        )
        assert usdc_received >= MIN_USDC_OUT, (
            f"USDC received ({usdc_received / 10**6:.4f}) is implausibly low for {ETH_SWAP_AMOUNT / 10**18} ETH"
        )

    async def test_v4_wrap_and_swap_tx_execution(self, fork_w3):
        """Execute WRAP_ETH + V4_SWAP and confirm the USDC balance increases.

        The test auto-discovers the first initialized hookless WETH/USDC V4 pool
        and skips if none is found.  It then builds and executes a real
        ``WRAP_ETH + V4_SWAP`` transaction via
        :meth:`~pydefi.amm.universal_router.UniversalRouter.build_wrap_and_v4_swap_transaction`
        and verifies that:

        - The transaction was not reverted.
        - The recipient's USDC balance increased by at least ``MIN_USDC_OUT``.
        """
        currency0, currency1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        pool_params = await _find_v4_pool(fork_w3, currency0, currency1)
        if pool_params is None:
            pytest.skip(
                f"No initialized V4 WETH/USDC hookless pool found on the fork (tried fee tiers: {_V4_FEE_TIERS})"
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
            amount_out_minimum=MIN_USDC_OUT,
        )

        await _impersonate(fork_w3, ETH_WHALE)
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)

        receipt = await _send_and_mine(
            fork_w3,
            {
                "to": Web3.to_checksum_address(tx.to),
                "from": Web3.to_checksum_address(ETH_WHALE),
                "value": tx.value,
                "data": tx.data,
            },
        )

        assert receipt["status"] == 1, f"Transaction reverted: {receipt['transactionHash'].hex()}"

        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        usdc_received = usdc_after - usdc_before

        assert usdc_received >= MIN_USDC_OUT, (
            f"USDC received ({usdc_received / 10**6:.4f}) is implausibly low for {ETH_SWAP_AMOUNT / 10**18} ETH"
        )

    async def test_v4_multihop_wrap_and_swap_tx_execution(self, fork_w3):
        """Execute WRAP_ETH + V4 multi-hop (SWAP_EXACT_IN) and verify USDC balance increase.

        Uses
        :meth:`~pydefi.amm.universal_router.UniversalRouter.build_wrap_and_v4_multihop_swap_transaction`
        with a single V4 hop to exercise the ``SWAP_EXACT_IN`` action path.
        The transaction is submitted as a real ``eth_sendTransaction`` against
        the local Anvil fork and the test verifies on-chain state changes.
        """
        currency0, currency1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        pool_params = await _find_v4_pool(fork_w3, currency0, currency1)
        if pool_params is None:
            pytest.skip(
                f"No initialized V4 WETH/USDC hookless pool found on the fork (tried fee tiers: {_V4_FEE_TIERS})"
            )
        fee, tick_spacing = pool_params

        router = UniversalRouter(UNIVERSAL_ROUTER_V2)
        tx = router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=ETH_SWAP_AMOUNT,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=fee, tick_spacing=tick_spacing)],
            recipient=ETH_WHALE,
            amount_out_minimum=MIN_USDC_OUT,
        )

        await _impersonate(fork_w3, ETH_WHALE)
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)

        receipt = await _send_and_mine(
            fork_w3,
            {
                "to": Web3.to_checksum_address(tx.to),
                "from": Web3.to_checksum_address(ETH_WHALE),
                "value": tx.value,
                "data": tx.data,
            },
        )

        assert receipt["status"] == 1, f"Transaction reverted: {receipt['transactionHash'].hex()}"

        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        usdc_received = usdc_after - usdc_before

        assert usdc_received >= MIN_USDC_OUT, (
            f"USDC received ({usdc_received / 10**6:.4f}) is implausibly low for {ETH_SWAP_AMOUNT / 10**18} ETH"
        )

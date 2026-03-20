"""Live integration tests for Uniswap V3 using a public Ethereum RPC.

These tests call the Uniswap V3 QuoterV2 deployed on Ethereum mainnet to
verify that ``quote_exact_input_single`` returns plausible values.
"""

import pytest

from pydifi.amm.uniswap_v3 import UniswapV3
from pydifi.types import TokenAmount

from .conftest import DAI, USDC, WETH

# Uniswap V3 contracts on Ethereum mainnet
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"

MIN_USDC = 500 * 10**6
MAX_USDC = 10_000 * 10**6

MIN_DAI = 500 * 10**18
MAX_DAI = 10_000 * 10**18


@pytest.mark.live
class TestUniswapV3Live:
    """Live on-chain tests for UniswapV3."""

    async def test_quote_exact_input_single_weth_usdc_3000(self, eth_w3):
        """QuoterV2 exactInputSingle: 1 WETH → USDC (0.3% pool) should be plausible."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
            default_fee=3000,
        )
        amount_in = TokenAmount.from_human(WETH, "1")
        amount_out = await quoter.quote_exact_input_single(amount_in, USDC, fee=3000)

        assert amount_out.token == USDC
        assert MIN_USDC < amount_out.amount < MAX_USDC, (
            f"V3 WETH/USDC price out of expected range: {amount_out.amount / 10**6:.2f} USDC"
        )

    async def test_quote_exact_input_single_weth_usdc_500(self, eth_w3):
        """QuoterV2 exactInputSingle: 1 WETH → USDC (0.05% pool) should be plausible."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
        )
        amount_in = TokenAmount.from_human(WETH, "1")
        amount_out = await quoter.quote_exact_input_single(amount_in, USDC, fee=500)

        assert amount_out.token == USDC
        assert MIN_USDC < amount_out.amount < MAX_USDC, (
            f"V3 0.05% pool price out of expected range: {amount_out.amount / 10**6:.2f} USDC"
        )

    async def test_get_amounts_out_multihop_weth_dai(self, eth_w3):
        """Multi-hop quote: 1 WETH → USDC → DAI should yield close to $1 per DAI."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
            default_fee=3000,
        )
        amount_in = TokenAmount.from_human(WETH, "1")
        # Use liquid fee tiers: WETH/USDC 0.05% (500) and USDC/DAI 0.01% (100)
        amounts = await quoter.get_amounts_out(amount_in, [WETH, USDC, DAI], fees=[500, 100])

        assert len(amounts) == 2  # start and end only for multi-hop
        assert amounts[-1].token == DAI
        assert MIN_DAI < amounts[-1].amount < MAX_DAI, (
            f"V3 multi-hop price out of range: {amounts[-1].amount / 10**18:.2f} DAI"
        )

    async def test_build_swap_route(self, eth_w3):
        """build_swap_route should return a valid SwapRoute for WETH → USDC."""
        quoter = UniswapV3(
            w3=eth_w3,
            router_address=UNISWAP_V3_ROUTER,
            quoter_address=UNISWAP_V3_QUOTER,
        )
        amount_in = TokenAmount.from_human(WETH, "0.1")
        route = await quoter.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.amount_out.amount > 0

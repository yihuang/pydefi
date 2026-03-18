"""Live integration tests for Uniswap V2 using a public Ethereum RPC.

These tests call the Uniswap V2 Router02 deployed on Ethereum mainnet to
verify that ``get_amounts_out`` and ``get_amounts_in`` return plausible values.
"""

import pytest

from pydifi.amm.uniswap_v2 import UniswapV2
from pydifi.types import TokenAmount

from .conftest import DAI, USDC, WETH

# Uniswap V2 Router02 on Ethereum mainnet
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

# Sanity bounds: 1 WETH should fetch between $500 and $10 000 in USDC/DAI
MIN_USDC = 500 * 10**6   # 500 USDC
MAX_USDC = 10_000 * 10**6  # 10 000 USDC

MIN_DAI = 500 * 10**18
MAX_DAI = 10_000 * 10**18


@pytest.mark.live
class TestUniswapV2Live:
    """Live on-chain tests for UniswapV2."""

    async def test_get_amounts_out_weth_usdc(self, eth_w3):
        """1 WETH → USDC via getAmountsOut should return a plausible price."""
        router = UniswapV2(w3=eth_w3, router_address=UNISWAP_V2_ROUTER)
        amount_in = TokenAmount.from_human(WETH, "1")
        amounts = await router.get_amounts_out(amount_in, [WETH, USDC])

        assert len(amounts) == 2
        assert amounts[0].token == WETH
        assert amounts[1].token == USDC
        assert MIN_USDC < amounts[1].amount < MAX_USDC, (
            f"WETH/USDC price out of expected range: {amounts[1].amount / 10**6:.2f} USDC"
        )

    async def test_get_amounts_out_weth_dai(self, eth_w3):
        """1 WETH → DAI via getAmountsOut should return a plausible price."""
        router = UniswapV2(w3=eth_w3, router_address=UNISWAP_V2_ROUTER)
        amount_in = TokenAmount.from_human(WETH, "1")
        amounts = await router.get_amounts_out(amount_in, [WETH, DAI])

        assert len(amounts) == 2
        assert amounts[1].token == DAI
        assert MIN_DAI < amounts[1].amount < MAX_DAI, (
            f"WETH/DAI price out of expected range: {amounts[1].amount / 10**18:.2f} DAI"
        )

    async def test_get_amounts_in_usdc_to_weth(self, eth_w3):
        """getAmountsIn: buying 2000 USDC worth of WETH should cost < 2 WETH."""
        router = UniswapV2(w3=eth_w3, router_address=UNISWAP_V2_ROUTER)
        amount_out = TokenAmount.from_human(USDC, "2000")
        amounts = await router.get_amounts_in(amount_out, [WETH, USDC])

        assert len(amounts) == 2
        assert amounts[0].token == WETH
        # 2000 USDC should cost well under 2 ETH at any realistic price
        assert amounts[0].amount < 2 * 10**18, (
            f"Cost for 2000 USDC unexpectedly high: {amounts[0].amount / 10**18:.4f} WETH"
        )

    async def test_build_swap_route(self, eth_w3):
        """build_swap_route should return a valid SwapRoute for WETH → USDC."""
        router = UniswapV2(w3=eth_w3, router_address=UNISWAP_V2_ROUTER)
        amount_in = TokenAmount.from_human(WETH, "0.1")
        route = await router.build_swap_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.amount_out.amount > 0

"""Live integration tests for Uniswap V3 using a public Ethereum RPC.

These tests call the Uniswap V3 QuoterV2 deployed on Ethereum mainnet to
verify that ``quote_exact_input_single`` returns plausible values.
"""

import pytest

from pydefi.abi.amm import UNISWAP_V3_POOL
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.pathfinder.graph import V3PoolEdge
from pydefi.types import TokenAmount
from tests.addrs import (
    DAI,
    POOL_WETH_USDC_500,
    UNISWAP_V3_QUOTER,
    UNISWAP_V3_ROUTER,
    USDC,
    WETH,
)

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


@pytest.mark.live
async def test_fetch_tick_ladder_prices_better_than_the_approximation(eth_w3):
    """An exact ladder must track the on-chain quoter where the estimate drifts.

    The single-tick estimate assumes liquidity never changes, so it only errs on
    trades large enough to cross boundaries — which is exactly what the ladder
    exists to fix. Both are compared against the quoter, the ground truth.
    """
    v3 = UniswapV3(w3=eth_w3, router_address=UNISWAP_V3_ROUTER, quoter_address=UNISWAP_V3_QUOTER)
    ladder = await v3.fetch_tick_ladder(POOL_WETH_USDC_500)
    assert len(ladder) > 0, "pool should have initialised ticks near the price"
    assert ladder.prices == sorted(ladder.prices)

    slot0 = await UNISWAP_V3_POOL.fns.slot0().call(eth_w3, to=POOL_WETH_USDC_500)
    liquidity = int(await UNISWAP_V3_POOL.fns.liquidity().call(eth_w3, to=POOL_WETH_USDC_500))
    common = {
        "token_in": WETH,
        "token_out": USDC,
        "pool_address": POOL_WETH_USDC_500,
        "protocol": "UniswapV3",
        "fee_bps": 5,
        "sqrt_price_x96": int(slot0[0]),
        "liquidity": liquidity,
        "is_token0_in": bytes(WETH.address) < bytes(USDC.address),
    }
    approx = V3PoolEdge(**common)
    exact = V3PoolEdge(**common, tick_ladder=ladder)

    big = 2000 * 10**18
    quoted = (await v3.quote_exact_input_single(TokenAmount(WETH, big), USDC, fee=500)).amount
    exact_err = abs(exact.amount_out(big) - quoted) / quoted
    approx_err = abs(approx.amount_out(big) - quoted) / quoted

    assert exact_err < 1e-4, f"exact walk drifted from the quoter: {exact_err:.2%}"
    assert exact_err <= approx_err, "the ladder must not be worse than the estimate it replaces"

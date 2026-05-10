"""Live integration tests for the ParaSwap aggregator API.

ParaSwap's /prices endpoint does not require an API key and is freely
accessible.  These tests verify that a live quote for 1 WETH → USDC returns
a structurally valid and numerically plausible response.
"""

import pytest

from pydefi.aggregator.paraswap import ParaSwap
from pydefi.types import TokenAmount
from tests.addrs import USDC, WETH

MIN_USDC = 500 * 10**6
MAX_USDC = 10_000 * 10**6


@pytest.mark.live
class TestParaSwapLive:
    """Live tests against the public ParaSwap v5 API."""

    async def test_get_price_weth_usdc(self):
        """GET /prices should return a priceRoute with a non-zero destAmount."""
        client = ParaSwap(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "1")
        data = await client.get_price(amount_in, USDC)

        price_route = data.get("priceRoute", {})
        assert price_route, "priceRoute missing from ParaSwap /prices response"
        dest_amount = int(price_route.get("destAmount", 0))
        assert MIN_USDC < dest_amount < MAX_USDC, (
            f"ParaSwap WETH→USDC price out of range: {dest_amount / 10**6:.2f} USDC"
        )

    async def test_get_quote_weth_usdc(self):
        """get_quote should parse the priceRoute into a well-formed AggregatorQuote."""
        client = ParaSwap(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "1")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.protocol == "ParaSwap"
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"Quote amount_out out of range: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        # min_amount_out must be ≤ amount_out (slippage applied)
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_get_price_small_amount(self):
        """ParaSwap should also handle small amounts (0.01 WETH → USDC)."""
        client = ParaSwap(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "0.01")
        data = await client.get_price(amount_in, USDC)

        price_route = data.get("priceRoute", {})
        dest_amount = int(price_route.get("destAmount", 0))
        assert dest_amount > 0, "Expected non-zero destAmount for 0.01 WETH"

"""Live integration tests for the OKX and OpenOcean DEX aggregator clients.

These tests call the real OKX DEX Aggregator v6 API
(``https://www.okx.com/api/v6/dex/aggregator``) and the real OpenOcean v4 API
(``https://open-api.openocean.finance/v4``).

Neither API requires an API key for read-only (quote) requests, so these tests
run against the public endpoints without credentials.  If the ``OKX_API_KEY``
environment variable is set, it is forwarded to the OKX client.

Run the live tests with::

    pytest -m live tests/live/test_okx_openocean_live.py
"""

import os

import pytest

from pydefi.aggregator.okx import OKX
from pydefi.aggregator.openocean import OpenOcean
from pydefi.types import TokenAmount

from .conftest import USDC, WETH

# ---------------------------------------------------------------------------
# Plausible WETH → USDC price bounds
# ---------------------------------------------------------------------------

MIN_USDC = 500 * 10**6  # 500 USDC per ETH minimum
MAX_USDC = 10_000 * 10**6  # 10 000 USDC per ETH maximum

# Vitalik's address – used as a read-only wallet for swap-quote requests.
_WALLET = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

_OKX_API_KEY: str | None = os.environ.get("OKX_API_KEY")


# ---------------------------------------------------------------------------
# OKX live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestOKXLive:
    """Live tests against the public OKX DEX Aggregator v6 API."""

    @pytest.fixture(autouse=True)
    def _skip_without_api_key(self):
        """Skip all OKX tests when OKX_API_KEY is not configured."""
        if not _OKX_API_KEY:
            pytest.skip("OKX_API_KEY environment variable is not set")

    async def test_get_quote_weth_usdc(self):
        """GET /quote should return a non-zero output amount for 1 WETH → USDC."""
        client = OKX(chain_id=1, api_key=_OKX_API_KEY)
        amount_in = TokenAmount.from_human(WETH, "1")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.protocol == "OKX"
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"OKX WETH→USDC quote out of range: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_get_quote_small_amount(self):
        """GET /quote should return a non-zero output for 0.01 WETH → USDC."""
        client = OKX(chain_id=1, api_key=_OKX_API_KEY)
        amount_in = TokenAmount.from_human(WETH, "0.01")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount > 0, "Expected non-zero amount out for 0.01 WETH"
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_build_swap_route(self):
        """build_swap_route should return a well-formed SwapRoute."""
        client = OKX(chain_id=1, api_key=_OKX_API_KEY)
        amount_in = TokenAmount.from_human(WETH, "1")
        route = await client.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) >= 1
        assert route.steps[0].protocol == "OKX"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC


# ---------------------------------------------------------------------------
# OpenOcean live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestOpenOceanLive:
    """Live tests against the public OpenOcean v4 API."""

    async def test_get_quote_weth_usdc(self):
        """GET /quote should return a non-zero output amount for 1 WETH → USDC."""
        client = OpenOcean(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "1")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.protocol == "OpenOcean"
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"OpenOcean WETH→USDC quote out of range: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_get_quote_small_amount(self):
        """GET /quote should return a non-zero output for 0.01 WETH → USDC."""
        client = OpenOcean(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "0.01")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50)

        assert quote.amount_out.amount > 0, "Expected non-zero amount out for 0.01 WETH"
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_build_swap_route(self):
        """build_swap_route should return a well-formed SwapRoute."""
        client = OpenOcean(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "1")
        route = await client.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) >= 1
        assert route.steps[0].protocol == "OpenOcean"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC

    async def test_get_swap_returns_tx_data(self):
        """GET /swap should populate tx_data with calldata for the swap."""
        client = OpenOcean(chain_id=1)
        amount_in = TokenAmount.from_human(WETH, "0.01")
        quote = await client.get_swap(amount_in, USDC, from_address=_WALLET, slippage_bps=50)

        assert quote.amount_out.amount > 0
        assert quote.tx_data, "tx_data should be populated by get_swap()"
        assert quote.tx_data.get("to"), "tx_data should contain 'to' address"
        assert quote.tx_data.get("data"), "tx_data should contain calldata"

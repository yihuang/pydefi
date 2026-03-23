"""Live integration tests for the Uniswap Trading API aggregator.

These tests call the real Uniswap Trading API
(``https://trade-api.gateway.uniswap.org``) to verify that
:class:`~pydefi.aggregator.uniswap.UniswapAPI` constructs valid requests and
parses responses correctly.

The Uniswap Trading API requires an API key for production use.  If the
``UNISWAP_API_KEY`` environment variable is not set the tests are skipped
(so they do not fail in CI without credentials).

Run the live tests with::

    UNISWAP_API_KEY=<key> pytest -m live tests/live/test_uniswap_api_live.py
"""

import os

import pytest

from pydefi.aggregator.uniswap import UniswapAPI
from pydefi.types import TokenAmount

from .conftest import DAI, USDC, WETH

# ---------------------------------------------------------------------------
# Plausible WETH → USDC price bounds
# ---------------------------------------------------------------------------

MIN_USDC = 500 * 10**6  # 500 USDC per ETH minimum
MAX_USDC = 10_000 * 10**6  # 10 000 USDC per ETH maximum

# ---------------------------------------------------------------------------
# API key — tests are skipped when the key is absent
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("UNISWAP_API_KEY")
# Wallet used as swapper for live calls that require a wallet address.
_WALLET = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth


def _make_client(chain_id: int = 1) -> UniswapAPI:
    """Return a UniswapAPI client configured from environment variables."""
    return UniswapAPI(chain_id=chain_id, api_key=_API_KEY)


@pytest.mark.live
class TestUniswapAPILive:
    """Live tests against the public Uniswap Trading API."""

    @pytest.fixture(autouse=True)
    def _skip_without_api_key(self):
        """Skip all tests in this class if no UNISWAP_API_KEY is configured."""
        if not _API_KEY:
            pytest.skip("UNISWAP_API_KEY environment variable is not set")

    async def test_get_quote_weth_usdc(self):
        """POST /v1/quote should return a non-zero output amount for 1 WETH → USDC."""
        client = _make_client()
        amount_in = TokenAmount.from_human(WETH, "1")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50, swapper=_WALLET)

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.protocol == "Uniswap"
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"Quote amount_out out of range: {quote.amount_out.amount / 10**6:.2f} USDC for 1 WETH"
        )
        # min_amount_out must be ≤ amount_out (slippage applied)
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_get_quote_small_amount(self):
        """POST /v1/quote should return a non-zero output for 0.01 WETH → USDC."""
        client = _make_client()
        amount_in = TokenAmount.from_human(WETH, "0.01")
        quote = await client.get_quote(amount_in, USDC, slippage_bps=50, swapper=_WALLET)

        assert quote.amount_out.amount > 0, "Expected non-zero amount out for 0.01 WETH"
        assert quote.min_amount_out.amount <= quote.amount_out.amount

    async def test_get_quote_weth_dai(self):
        """POST /v1/quote should also work for WETH → DAI (18-decimal stablecoin)."""
        client = _make_client()
        amount_in = TokenAmount.from_human(WETH, "1")
        quote = await client.get_quote(amount_in, DAI, slippage_bps=50, swapper=_WALLET)

        assert quote.token_in == WETH
        assert quote.token_out == DAI
        # 1 DAI ≈ 1 USD; bounds in 18-decimal units
        min_dai = 500 * 10**18
        max_dai = 10_000 * 10**18
        assert min_dai < quote.amount_out.amount < max_dai, (
            f"WETH→DAI quote out of range: {quote.amount_out.amount / 10**18:.2f} DAI"
        )

    async def test_build_swap_route(self):
        """build_swap_route should return a well-formed SwapRoute."""
        client = _make_client()
        amount_in = TokenAmount.from_human(WETH, "1")
        route = await client.build_swap_route(amount_in, USDC, slippage_bps=50, swapper=_WALLET)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert len(route.steps) >= 1
        assert route.steps[0].protocol == "Uniswap"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC

    async def test_get_swap_end_to_end(self):
        """Two-step flow (POST /v1/quote → POST /v1/swap) should return tx calldata.

        This test exercises the full end-to-end swap-building flow described in
        the Uniswap Trading API guide.  It does *not* broadcast the transaction.
        """
        client = _make_client()
        amount_in = TokenAmount.from_human(WETH, "0.01")

        quote = await client.get_swap(
            amount_in,
            USDC,
            wallet_address=_WALLET,
            slippage_bps=50,
            # Force classic AMM routing so the quote is compatible with /v1/swap.
            # Without this the API may return a UniswapX (DUTCH_V2/V3) quote.
            protocols=["V2", "V3", "V4"],
        )

        assert quote.token_in == WETH
        assert quote.token_out == USDC
        assert quote.amount_out.amount > 0

        # tx_data must contain at minimum the target address and calldata
        tx = quote.tx_data
        assert tx, "tx_data should be populated by get_swap()"
        assert tx.get("to") or tx.get("data"), "tx_data should contain 'to' or 'data' from the /v1/swap response"

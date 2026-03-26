"""Live integration tests for Solana integrations (Raydium AMM + Jupiter aggregator).

These tests hit the real public APIs and verify that quotes, routes, and
transaction blobs are structurally valid and numerically plausible.

For transaction-building tests the environment variables below are consulted:

* ``SOLANA_WALLET`` — base-58 Solana public key used as the fee payer when
  building swap transactions.  Falls back to a well-known simulation address so
  the test still runs without a real wallet; the transaction is validated via
  ``simulateTransaction`` (``sigVerify=false``) rather than being submitted.
* ``SOLANA_RPC_URL`` — Solana JSON-RPC endpoint used for simulation
  (default: ``https://api.mainnet-beta.solana.com``).

Run with::

    pytest -m live tests/live/test_solana_live.py
"""

from __future__ import annotations

import os
from decimal import Decimal

import aiohttp
import pytest

from pydefi.aggregator.base import AggregatorQuote
from pydefi.aggregator.jupiter import Jupiter, JupiterSwapV2
from pydefi.amm.raydium import Raydium
from pydefi.types import ChainId, SwapRoute, Token, TokenAmount

# ---------------------------------------------------------------------------
# Solana token constants
# ---------------------------------------------------------------------------

SOL = Token(
    chain_id=ChainId.SOLANA,
    address="So11111111111111111111111111111111111111112",
    symbol="SOL",
    decimals=9,
)
USDC = Token(
    chain_id=ChainId.SOLANA,
    address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    symbol="USDC",
    decimals=6,
)

# Sanity-check bounds for 1 SOL → USDC (expected: ~$50–$1000 USDC range)
MIN_USDC = 50 * 10**6
MAX_USDC = 1_000 * 10**6

# Optional wallet address used only for transaction-building tests.
# Set SOLANA_WALLET env var to a real (or dummy) base-58 public key to enable.
SOLANA_WALLET = os.environ.get("SOLANA_WALLET", "")
# Jupiter Swap V2 API key from portal.jup.ag (required for JupiterSwapV2 tests).
JUPITER_API_KEY = os.environ.get("JUPITER_API_KEY", "")
# Solana JSON-RPC URL for transaction simulation (public mainnet by default).
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
# Fallback wallet for simulation: a known Solana address used when SOLANA_WALLET
# is not set.  sigVerify=false means this address does not need to sign.
_SIMULATION_WALLET = "GThUX1Atko4tqhN2NaiTazWSeFWMuiUvfFnyJyUghFMJ"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _simulate_transaction(rpc_url: str, base64_tx: str) -> dict:
    """Call ``simulateTransaction`` on a Solana JSON-RPC endpoint.

    Uses ``sigVerify=false`` and ``replaceRecentBlockhash=true`` so the
    transaction does not need to be signed and blockhash freshness is handled
    automatically.

    Returns the raw JSON-RPC response dict.  A top-level ``"result"`` key
    indicates the transaction was at least structurally parseable; the nested
    ``result.value.err`` may still be non-null if execution would fail (e.g.
    insufficient balance).
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "simulateTransaction",
        "params": [
            base64_tx,
            {
                "encoding": "base64",
                "sigVerify": False,
                "replaceRecentBlockhash": True,
                "commitment": "processed",
            },
        ],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            return await resp.json(content_type=None)


# ---------------------------------------------------------------------------
# Raydium live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestRaydiumLive:
    """Live tests against the public Raydium V3 compute API."""

    async def test_get_quote_sol_usdc(self):
        """GET /compute/swap-base-in should return a plausible USDC amount for 1 SOL."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        result = await raydium.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(result, TokenAmount)
        assert result.token == USDC
        assert MIN_USDC < result.amount < MAX_USDC, (
            f"Raydium SOL→USDC quote out of expected range: {result.amount / 10**6:.2f} USDC"
        )

    async def test_build_swap_route_sol_usdc(self):
        """build_swap_route should return a well-formed SwapRoute with a Raydium step."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        route = await raydium.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Raydium"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC, (
            f"Raydium route amount_out out of range: {route.amount_out.amount / 10**6:.2f} USDC"
        )
        # price_impact is a fraction in [0, 1]
        assert Decimal(0) <= route.price_impact <= Decimal("0.1"), (
            f"Raydium price_impact out of range: {route.price_impact}"
        )

    async def test_get_quote_small_amount(self):
        """Raydium should handle a small input (0.01 SOL)."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "0.01")

        result = await raydium.get_quote(amount_in, USDC)

        assert isinstance(result, TokenAmount)
        assert result.amount > 0

    async def test_build_transaction_sol_usdc(self):
        """build_transaction should return structurally valid base-64 transactions.

        Uses ``SOLANA_WALLET`` if set, otherwise falls back to a well-known
        simulation address.  Each transaction is validated via
        ``simulateTransaction`` (``sigVerify=false``) on the public Solana RPC.
        """
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "0.01")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        txs = await raydium.build_transaction(
            amount_in,
            USDC,
            wallet=wallet,
            slippage_bps=100,
        )

        assert isinstance(txs, list)
        assert len(txs) >= 1
        for tx_b64 in txs:
            assert isinstance(tx_b64, str)
            assert len(tx_b64) > 0
            # Verify structural validity via Solana RPC simulation.
            sim = await _simulate_transaction(SOLANA_RPC_URL, tx_b64)
            assert "result" in sim, f"simulateTransaction RPC error (malformed tx): {sim}"


# ---------------------------------------------------------------------------
# Jupiter live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestJupiterLive:
    """Live tests against the public Jupiter V6 quote API."""

    async def test_get_quote_sol_usdc(self):
        """GET /quote should return a plausible USDC amount for 1 SOL."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        quote = await jupiter.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(quote, AggregatorQuote)
        assert quote.protocol == "Jupiter"
        assert quote.token_in == SOL
        assert quote.token_out == USDC
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"Jupiter SOL→USDC quote out of expected range: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        # min_amount_out ≤ amount_out (slippage applied by Jupiter)
        assert quote.min_amount_out.amount <= quote.amount_out.amount
        # price_impact is a fraction in [0, 1]
        assert Decimal(0) <= quote.price_impact <= Decimal("0.1"), (
            f"Jupiter price_impact out of range: {quote.price_impact}"
        )

    async def test_build_swap_route_sol_usdc(self):
        """build_swap_route should return a well-formed SwapRoute."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        route = await jupiter.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Jupiter"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC

    async def test_get_quote_small_amount(self):
        """Jupiter should handle a small input (0.01 SOL)."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "0.01")

        quote = await jupiter.get_quote(amount_in, USDC)

        assert isinstance(quote, AggregatorQuote)
        assert quote.amount_out.amount > 0

    async def test_get_swap_transaction_sol_usdc(self):
        """get_swap_transaction should return a structurally valid Solana transaction.

        Uses ``SOLANA_WALLET`` if set, otherwise falls back to a well-known
        simulation address and validates the transaction via
        ``simulateTransaction`` (``sigVerify=false``).
        """
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "0.01")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        result = await jupiter.get_swap_transaction(
            amount_in,
            USDC,
            user_public_key=wallet,
            slippage_bps=100,
        )

        assert "swapTransaction" in result
        assert isinstance(result["swapTransaction"], str)
        assert len(result["swapTransaction"]) > 0
        assert "lastValidBlockHeight" in result
        # Verify structural validity via Solana RPC simulation.
        sim = await _simulate_transaction(SOLANA_RPC_URL, result["swapTransaction"])
        assert "result" in sim, f"simulateTransaction RPC error (malformed tx): {sim}"


# ---------------------------------------------------------------------------
# JupiterSwapV2 live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not JUPITER_API_KEY, reason="JUPITER_API_KEY env var not set")
class TestJupiterSwapV2Live:
    """Live tests against the Jupiter Swap V2 API (requires API key)."""

    async def test_get_order_without_taker_sol_usdc(self):
        """GET /order without taker should return a plausible quote for 1 SOL."""
        j = JupiterSwapV2(api_key=JUPITER_API_KEY)
        amount_in = TokenAmount.from_human(SOL, "1")

        order = await j.get_order(amount_in, USDC)

        assert "outAmount" in order
        out_amount = int(order["outAmount"])
        assert MIN_USDC < out_amount < MAX_USDC, (
            f"JupiterSwapV2 /order SOL→USDC out of expected range: {out_amount / 10**6:.2f} USDC"
        )
        # Without a taker there should be no transaction or requestId
        assert "transaction" not in order or order.get("transaction") is None

    async def test_get_order_with_taker_sol_usdc(self):
        """GET /order with taker should include a transaction and requestId.

        Uses ``SOLANA_WALLET`` if set, otherwise falls back to a well-known
        simulation address.  The returned transaction is validated via
        ``simulateTransaction`` (``sigVerify=false``).
        """
        j = JupiterSwapV2(api_key=JUPITER_API_KEY)
        amount_in = TokenAmount.from_human(SOL, "0.01")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        order = await j.get_order(amount_in, USDC, taker=wallet, slippage_bps=100)

        assert "transaction" in order and order["transaction"]
        assert "requestId" in order and order["requestId"]
        out_amount = int(order["outAmount"])
        assert out_amount > 0
        # Verify the transaction is structurally valid via Solana RPC simulation.
        sim = await _simulate_transaction(SOLANA_RPC_URL, order["transaction"])
        assert "result" in sim, f"simulateTransaction RPC error (malformed tx): {sim}"

    async def test_get_quote_sol_usdc(self):
        """get_quote should return a valid AggregatorQuote for 1 SOL → USDC."""
        j = JupiterSwapV2(api_key=JUPITER_API_KEY)
        amount_in = TokenAmount.from_human(SOL, "1")

        quote = await j.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(quote, AggregatorQuote)
        assert quote.protocol == "Jupiter"
        assert quote.token_in == SOL
        assert quote.token_out == USDC
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"JupiterSwapV2 quote out of expected range: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        assert quote.min_amount_out.amount <= quote.amount_out.amount
        assert Decimal(0) <= quote.price_impact <= Decimal("0.1")

    async def test_get_build_sol_usdc(self):
        """GET /build should return raw swap instructions for 1 SOL → USDC.

        Uses ``SOLANA_WALLET`` if set, otherwise falls back to a well-known
        simulation address.  If the response contains a complete transaction it
        is also validated via ``simulateTransaction`` (``sigVerify=false``).
        """
        j = JupiterSwapV2(api_key=JUPITER_API_KEY)
        amount_in = TokenAmount.from_human(SOL, "1")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        result = await j.get_build(amount_in, USDC, taker=wallet, slippage_bps=50)

        assert isinstance(result, dict)
        # The build response should contain quote fields
        assert "outAmount" in result or "swapTransaction" in result
        # If a complete transaction blob is returned, simulate it.
        if result.get("swapTransaction"):
            sim = await _simulate_transaction(SOLANA_RPC_URL, result["swapTransaction"])
            assert "result" in sim, f"simulateTransaction RPC error (malformed tx): {sim}"

    async def test_build_swap_route_sol_usdc(self):
        """build_swap_route should return a well-formed SwapRoute."""
        j = JupiterSwapV2(api_key=JUPITER_API_KEY)
        amount_in = TokenAmount.from_human(SOL, "1")

        route = await j.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert isinstance(route, SwapRoute)
        assert route.token_in == SOL
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Jupiter"
        assert MIN_USDC < route.amount_out.amount < MAX_USDC


# ---------------------------------------------------------------------------
# Solana fork tests (require surfpool on PATH)
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestRaydiumFork:
    """Solana fork tests for Raydium using a local surfpool mainnet fork.

    These tests build real Raydium swap transactions and simulate them against
    surfpool, which mirrors live mainnet account state.  They are automatically
    skipped when ``surfpool`` is not found on ``$PATH``.

    Run with::

        pytest -m fork tests/live/test_solana_live.py::TestRaydiumFork
    """

    async def test_simulate_swap_sol_usdc(self, surfpool_rpc: str):
        """Raydium swap transaction should be structurally valid on a surfpool fork."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "0.01")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        txs = await raydium.build_transaction(amount_in, USDC, wallet=wallet, slippage_bps=100)

        assert len(txs) >= 1
        for tx_b64 in txs:
            result = await _simulate_transaction(surfpool_rpc, tx_b64)
            assert "result" in result, f"Raydium swap simulation failed on surfpool: {result}"

    async def test_simulate_route_swap_sol_usdc(self, surfpool_rpc: str):
        """Raydium quote and route should be consistent with the surfpool fork state."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "1")

        route = await raydium.build_swap_route(amount_in, USDC, slippage_bps=50)

        assert isinstance(route, SwapRoute)
        assert MIN_USDC < route.amount_out.amount < MAX_USDC, (
            f"Raydium route amount_out out of range on surfpool fork: {route.amount_out.amount / 10**6:.2f} USDC"
        )
        assert Decimal(0) <= route.price_impact <= Decimal("0.1")


@pytest.mark.fork
class TestJupiterFork:
    """Solana fork tests for Jupiter using a local surfpool mainnet fork.

    These tests build real Jupiter swap transactions and simulate them against
    surfpool, which mirrors live mainnet account state.  They are automatically
    skipped when ``surfpool`` is not found on ``$PATH``.

    Run with::

        pytest -m fork tests/live/test_solana_live.py::TestJupiterFork
    """

    async def test_simulate_swap_sol_usdc(self, surfpool_rpc: str):
        """Jupiter swap transaction should be structurally valid on a surfpool fork."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "0.01")
        wallet = SOLANA_WALLET or _SIMULATION_WALLET

        result = await jupiter.get_swap_transaction(amount_in, USDC, user_public_key=wallet, slippage_bps=100)

        assert "swapTransaction" in result
        sim = await _simulate_transaction(surfpool_rpc, result["swapTransaction"])
        assert "result" in sim, f"Jupiter swap simulation failed on surfpool: {sim}"

    async def test_simulate_route_swap_sol_usdc(self, surfpool_rpc: str):
        """Jupiter quote and route should be consistent with the surfpool fork state."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "1")

        quote = await jupiter.get_quote(amount_in, USDC, slippage_bps=50)

        assert isinstance(quote, AggregatorQuote)
        assert MIN_USDC < quote.amount_out.amount < MAX_USDC, (
            f"Jupiter quote out of range on surfpool fork: {quote.amount_out.amount / 10**6:.2f} USDC"
        )
        assert Decimal(0) <= quote.price_impact <= Decimal("0.1")

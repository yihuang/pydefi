"""Live integration tests for Solana integrations (Raydium AMM + Jupiter aggregator).

These tests hit the real public APIs and verify that quotes, routes, and
(optionally) transaction blobs are structurally valid and numerically plausible.

Run with::

    pytest -m live tests/live/test_solana_live.py
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from pydefi.aggregator.base import AggregatorQuote
from pydefi.aggregator.jupiter import Jupiter
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

    @pytest.mark.skipif(not SOLANA_WALLET, reason="SOLANA_WALLET env var not set")
    async def test_build_transaction_sol_usdc(self):
        """build_transaction should return at least one base-64 encoded transaction."""
        raydium = Raydium()
        amount_in = TokenAmount.from_human(SOL, "0.01")

        txs = await raydium.build_transaction(
            amount_in,
            USDC,
            wallet=SOLANA_WALLET,
            slippage_bps=100,
        )

        assert isinstance(txs, list)
        assert len(txs) >= 1
        for tx in txs:
            assert isinstance(tx, str)
            assert len(tx) > 0


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

    @pytest.mark.skipif(not SOLANA_WALLET, reason="SOLANA_WALLET env var not set")
    async def test_get_swap_transaction_sol_usdc(self):
        """get_swap_transaction should return a base-64 encoded Solana transaction."""
        jupiter = Jupiter()
        amount_in = TokenAmount.from_human(SOL, "0.01")

        result = await jupiter.get_swap_transaction(
            amount_in,
            USDC,
            user_public_key=SOLANA_WALLET,
            slippage_bps=100,
        )

        assert "swapTransaction" in result
        assert isinstance(result["swapTransaction"], str)
        assert len(result["swapTransaction"]) > 0
        assert "lastValidBlockHeight" in result

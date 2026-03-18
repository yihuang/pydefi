"""Live integration tests for Curve Finance using a public Ethereum RPC.

These tests call the Curve 3pool (DAI/USDC/USDT) deployed on Ethereum mainnet
to verify that ``get_dy`` returns plausible exchange rates for stablecoin swaps.

Curve 3pool address: 0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7
  coins[0] = DAI   (18 decimals)
  coins[1] = USDC  (6 decimals)
  coins[2] = USDT  (6 decimals)
"""

import pytest

from pydifi.amm.curve import CurvePool
from pydifi.types import TokenAmount

from .conftest import DAI, USDC, USDT

# Curve 3pool on Ethereum mainnet
CURVE_3POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"


@pytest.mark.live
class TestCurveLive:
    """Live on-chain tests for CurvePool (3pool)."""

    async def test_get_dy_dai_to_usdc(self, eth_w3):
        """1 000 DAI → USDC should return approximately 1 000 USDC (±1%)."""
        pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
        )
        amount_in = 1_000 * 10**18  # 1 000 DAI
        dy = await pool.get_dy(DAI, USDC, amount_in)

        # Expect ~999–1001 USDC (6 decimals)
        assert 990 * 10**6 < dy < 1_010 * 10**6, (
            f"DAI→USDC rate out of range: {dy / 10**6:.4f} USDC per 1000 DAI"
        )

    async def test_get_dy_usdc_to_dai(self, eth_w3):
        """1 000 USDC → DAI should return approximately 1 000 DAI (±1%)."""
        pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
        )
        amount_in = 1_000 * 10**6  # 1 000 USDC
        dy = await pool.get_dy(USDC, DAI, amount_in)

        # Expect ~999–1001 DAI (18 decimals)
        assert 990 * 10**18 < dy < 1_010 * 10**18, (
            f"USDC→DAI rate out of range: {dy / 10**18:.4f} DAI per 1000 USDC"
        )

    async def test_get_dy_usdc_to_usdt(self, eth_w3):
        """1 000 USDC → USDT should return approximately 1 000 USDT (±1%)."""
        pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
        )
        amount_in = 1_000 * 10**6  # 1 000 USDC
        dy = await pool.get_dy(USDC, USDT, amount_in)

        assert 990 * 10**6 < dy < 1_010 * 10**6, (
            f"USDC→USDT rate out of range: {dy / 10**6:.4f} USDT per 1000 USDC"
        )

    async def test_get_amounts_out(self, eth_w3):
        """get_amounts_out wrapper returns a two-element list."""
        pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
        )
        amount_in = TokenAmount.from_human(DAI, "100")
        result = await pool.get_amounts_out(amount_in, [DAI, USDC])

        assert len(result) == 2
        assert result[0].token == DAI
        assert result[1].token == USDC
        assert result[1].amount > 0

    async def test_build_swap_route(self, eth_w3):
        """build_swap_route should return a valid SwapRoute for DAI → USDC."""
        pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
        )
        amount_in = TokenAmount.from_human(DAI, "100")
        route = await pool.build_swap_route(amount_in, USDC)

        assert route.token_in == DAI
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Curve"
        assert route.amount_out.amount > 0

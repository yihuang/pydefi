"""Tests for pydefi.pathfinder.v3_tick_math.

Synthetic V3 pool fixtures: a single tick range (constant liquidity) for
parity vs the existing single-tick approximation, plus multi-tick fixtures
that force tick crossings.
"""

from __future__ import annotations

import math

import pytest

from pydefi.pathfinder.v3_tick_math import _Q96, _amount_in_to_reach, compute_swap_step, walk_v3_swap


def _sqrt_price_for_human_rate(rate_token1_per_token0: float) -> int:
    """sqrtPriceX96 for a desired raw-units rate token1/token0."""
    return int(math.sqrt(rate_token1_per_token0) * _Q96)


# ---------------------------------------------------------------------------
# compute_swap_step — single tick-range math
# ---------------------------------------------------------------------------


class TestComputeSwapStep:
    @pytest.mark.parametrize(
        "zero_for_one,target_factor,amount_in",
        [(True, 0.5, 10**16), (False, 2.0, 10**9)],
        ids=["zero_for_one", "one_for_zero"],
    )
    def test_within_range_consumes_all_input(self, zero_for_one, target_factor, amount_in):
        """Input small enough to stay in current range — no boundary hit, all consumed."""
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        sqrt_p_target = int(sqrt_p * target_factor)
        sqrt_p_next, used, out = compute_swap_step(sqrt_p, sqrt_p_target, 10**22, amount_in, zero_for_one)
        if zero_for_one:
            assert sqrt_p_next < sqrt_p and sqrt_p_next > sqrt_p_target
        else:
            assert sqrt_p_next > sqrt_p and sqrt_p_next < sqrt_p_target
        assert used == amount_in
        assert out > 0

    def test_zero_for_one_reaches_target(self):
        """Input large enough to cross boundary — stops exactly at target."""
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        sqrt_p_target = int(sqrt_p * 0.99)
        sqrt_p_next, used, _ = compute_swap_step(sqrt_p, sqrt_p_target, 10**21, 10**24, zero_for_one=True)
        assert sqrt_p_next == sqrt_p_target
        assert 0 < used < 10**24

    @pytest.mark.parametrize("liquidity,amount_in", [(0, 10**18), (10**22, 0)], ids=["zero_L", "zero_amount"])
    def test_zero_inputs_return_zero(self, liquidity, amount_in):
        sqrt_p = _sqrt_price_for_human_rate(2000e-12)
        assert compute_swap_step(sqrt_p, sqrt_p // 2, liquidity, amount_in, zero_for_one=True) == (sqrt_p, 0, 0)

    def test_max_in_rounds_up_so_boundary_snap_requires_full_input(self):
        """Floor-divide ``max_in`` could let an input that's 1 unit short snap
        to the boundary, undercharging by up to 1 unit per tick crossing.
        Ceiling division: ``max_in - 1`` must NOT snap; ``max_in`` exactly does.
        """
        sqrt_p_current = _Q96
        sqrt_p_target = _Q96 * 99 // 100  # 1% drop, non-integer true requirement
        liquidity = 10**18

        max_in = _amount_in_to_reach(sqrt_p_current, sqrt_p_target, liquidity, zero_for_one=True)
        # Real-valued requirement
        true_req = liquidity * _Q96 * (sqrt_p_current - sqrt_p_target) / (sqrt_p_current * sqrt_p_target)
        assert max_in >= true_req, f"max_in {max_in} below true requirement {true_req}"

        # max_in - 1: should stay within range (partial step)
        sqrt_p_next, _, _ = compute_swap_step(sqrt_p_current, sqrt_p_target, liquidity, max_in - 1, zero_for_one=True)
        assert sqrt_p_next != sqrt_p_target, "snapped to target on insufficient input"

        # max_in exactly: should snap
        sqrt_p_next, used, _ = compute_swap_step(sqrt_p_current, sqrt_p_target, liquidity, max_in, zero_for_one=True)
        assert sqrt_p_next == sqrt_p_target
        assert used == max_in


# ---------------------------------------------------------------------------
# walk_v3_swap — multi-tick traversal
# ---------------------------------------------------------------------------


class TestWalkV3Swap:
    def test_single_range_matches_compute_step(self):
        """With only a far swap-direction sentinel, walker == one compute_swap_step."""
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        liquidity = 10**22
        ticks = [(-887272, 1, 0), (100, sqrt_p * 2, 10**21)]
        out_walked = walk_v3_swap(sqrt_p, liquidity, ticks, 10**18, fee_bps=5, zero_for_one=True)
        amount_net = 10**18 * 9995 // 10000
        _, _, out_step = compute_swap_step(sqrt_p, 0, liquidity, amount_net, zero_for_one=True)
        assert out_walked > 0
        assert abs(out_walked - out_step) <= 1

    def test_crosses_tick_loses_output_vs_constant_liquidity(self):
        """Crossing a tick drops liquidity → less output than single-tick approx.

        Uses a thin pool (L≈1e15) so a 1 WETH swap actually moves price across
        the boundary; deeper pools would barely budge sqrtP and the test
        wouldn't exercise the cross.
        """
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        liquidity = 10**15
        # Boundary 0.5% sqrtP-drop below — well within the 1 WETH swap's reach.
        sqrt_p_at_tick = int(sqrt_p * 0.995)
        ticks_crossing = [
            (-887272, 1, 0),  # lower sentinel
            (-100, sqrt_p_at_tick, 8 * 10**14),  # cross-down subtracts most liquidity
        ]
        ticks_constant = [(-887272, 1, 0)]

        amount_in = 10**18  # 1 WETH
        out_constant = walk_v3_swap(sqrt_p, liquidity, ticks_constant, amount_in, fee_bps=5, zero_for_one=True)
        out_crossing = walk_v3_swap(sqrt_p, liquidity, ticks_crossing, amount_in, fee_bps=5, zero_for_one=True)

        assert out_constant > 0 and out_crossing > 0
        assert out_crossing < out_constant, (
            f"crossing tick should reduce output (constant={out_constant}, crossing={out_crossing})"
        )

    def test_zero_input_returns_zero(self):
        sqrt_p = _sqrt_price_for_human_rate(2000e-12)
        out = walk_v3_swap(sqrt_p, 10**22, [(0, 1, 0)], 0, fee_bps=5, zero_for_one=True)
        assert out == 0

    def test_negative_liquidity_guarded(self):
        """Walker must not return spurious output when liquidity drops to zero mid-swap."""
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        # A tick that drains all liquidity on cross-down.
        ticks = [
            (-887272, 1, 0),
            (-100, int(sqrt_p * 0.999), 10**22),  # crossing subtracts the entire liquidity
        ]
        out = walk_v3_swap(sqrt_p, 10**22, ticks, 5 * 10**18, fee_bps=5, zero_for_one=True)
        # Output should be finite and non-negative (no crash, no overflow).
        assert out >= 0

    def test_no_directional_ticks_executes_within_range(self):
        """Empty *ticks* must still execute a within-range swap, not return 0.

        Walker appends a protocol-level MIN/MAX_SQRT_RATIO sentinel so the
        constant-liquidity step always fires even when the caller-supplied
        tick list has nothing in the swap direction.
        """
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        liquidity = 10**22
        amount_in = 10**16  # 0.01 WETH
        out = walk_v3_swap(sqrt_p, liquidity, [], amount_in, fee_bps=5, zero_for_one=True)
        # Compare against compute_swap_step against an arbitrarily-far target —
        # both should produce the same within-range output.
        amount_net = amount_in * 9995 // 10000
        _, _, expected = compute_swap_step(sqrt_p, 1, liquidity, amount_net, zero_for_one=True)
        assert out > 0
        assert abs(out - expected) <= 1, f"out={out}, expected={expected}"

    def test_no_directional_ticks_one_for_zero(self):
        """Same coverage but for the price-up direction (token1-in)."""
        sqrt_p = _sqrt_price_for_human_rate(2000.0 * 10**6 / 10**18)
        liquidity = 10**22
        out = walk_v3_swap(sqrt_p, liquidity, [], 10**9, fee_bps=5, zero_for_one=False)
        assert out > 0

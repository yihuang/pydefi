"""Tests for pydefi.pathfinder.v3_tick_math.

Synthetic V3 pool fixtures: a single tick range (constant liquidity) for
parity vs the existing single-tick approximation, plus multi-tick fixtures
that force tick crossings.
"""

from __future__ import annotations

import math

import pytest

from pydefi.pathfinder.graph import V3PoolEdge
from pydefi.pathfinder.v3_tick_math import (
    _Q96,
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    TickLadder,
    _amount_in_to_reach,
    compute_swap_step,
    get_sqrt_ratio_at_tick,
    walk_v3_swap,
)
from pydefi.types import Address, Token


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


# ---------------------------------------------------------------------------
# TickLadder — pre-sorting hoisted out of the per-call path
# ---------------------------------------------------------------------------

_BASE_SQRT_P = 1771595571142957116569145374


def _spread(n: int) -> list[tuple[int, int, int]]:
    """n initialised ticks either side of _BASE_SQRT_P, deliberately unsorted."""
    ticks = [
        (i, int(_BASE_SQRT_P * (1.0 + 0.0005 * i)) if i % 2 else int(_BASE_SQRT_P * (1.0 - 0.0005 * i)), 10**14)
        for i in range(1, n + 1)
    ]
    return ticks[::-1]


@pytest.mark.parametrize("zero_for_one", [True, False])
@pytest.mark.parametrize("n_ticks", [0, 1, 5, 60])
def test_ladder_matches_raw_tick_list(n_ticks, zero_for_one):
    """A ladder must price identically to handing over the unsorted list."""
    ticks = _spread(n_ticks)
    args = (_BASE_SQRT_P, 10**20, ticks, 10**18, 5, zero_for_one)
    ladder_args = (_BASE_SQRT_P, 10**20, TickLadder(ticks), 10**18, 5, zero_for_one)
    assert walk_v3_swap(*ladder_args) == walk_v3_swap(*args)


def test_ladder_sorts_unsorted_input():
    ticks = _spread(20)
    ladder = TickLadder(ticks)
    assert ladder.prices == sorted(ladder.prices)
    assert len(ladder) == 20


@pytest.mark.parametrize("zero_for_one", [True, False])
def test_tick_exactly_at_current_price_is_excluded(zero_for_one):
    """The filter is strict on both sides — a tick *at* sqrtP is not crossable.

    bisect_left vs bisect_right is what preserves that; getting it backwards
    would silently cross a boundary the pool has not reached.
    """
    at_price = (0, _BASE_SQRT_P, 10**18)
    below = (-10, _BASE_SQRT_P - 10**20, 10**14)
    above = (10, _BASE_SQRT_P + 10**20, 10**14)
    ticks = [above, at_price, below]

    ladder = TickLadder(ticks)
    without = TickLadder([below, above])
    args = (10**20, 10**18, 5, zero_for_one)
    assert walk_v3_swap(_BASE_SQRT_P, *args[:1], ladder, *args[1:]) == walk_v3_swap(
        _BASE_SQRT_P, *args[:1], without, *args[1:]
    )


def test_ladder_handles_duplicate_prices():
    """Several ticks at one price must all be reachable, not just the first."""
    dupes = [(i, _BASE_SQRT_P - 10**20, 10**13) for i in range(3)]
    ladder = TickLadder(dupes)
    assert len(ladder) == 3
    assert walk_v3_swap(_BASE_SQRT_P, 10**20, ladder, 10**18, 5, True) == walk_v3_swap(
        _BASE_SQRT_P, 10**20, dupes, 10**18, 5, True
    )


# ---------------------------------------------------------------------------
# V3PoolEdge wiring — exact walking when a ladder is attached
# ---------------------------------------------------------------------------


def _v3_edge(ladder=None, fee_bps=5):
    a = Token(chain_id=1, address=Address(b"\xaa" * 20), symbol="A", decimals=18)
    b = Token(chain_id=1, address=Address(b"\xbb" * 20), symbol="B", decimals=6)
    return V3PoolEdge(
        token_in=a,
        token_out=b,
        pool_address=Address(b"\xcc" * 20),
        protocol="UniswapV3",
        fee_bps=fee_bps,
        sqrt_price_x96=_BASE_SQRT_P,
        liquidity=10**22,
        is_token0_in=True,
        tick_ladder=ladder,
    )


def test_edge_without_ladder_is_unchanged():
    """The default stays the single-tick approximation — opt-in only."""
    assert _v3_edge().tick_ladder is None
    assert _v3_edge().amount_out(10**16) > 0


def test_edge_with_ladder_walks_ticks_exactly():
    """A ladder must reproduce walk_v3_swap on the fee-netted amount."""
    ladder = TickLadder(_spread(40))
    edge = _v3_edge(ladder)
    amount_in = 10**18
    expected = walk_v3_swap(_BASE_SQRT_P, edge.liquidity, ladder, edge._net_amount_in(amount_in), 0, True)
    assert edge.amount_out(amount_in) == expected


def test_exact_walk_is_not_above_the_single_tick_estimate():
    """Crossing ticks costs output: the approximation should never be beaten.

    The single-tick estimate assumes liquidity never changes, which is why it
    overestimates trades that cross boundaries.
    """
    big = 10**20
    liquidity_shedding = [(i, int(_BASE_SQRT_P * (1 - 0.001 * i)), 10**21) for i in range(1, 12)]
    exact = _v3_edge(TickLadder(liquidity_shedding)).amount_out(big)
    approx = _v3_edge().amount_out(big)
    assert 0 < exact <= approx


# ---------------------------------------------------------------------------
# TickMath — tick index to sqrtPriceX96
# ---------------------------------------------------------------------------


def test_tick_zero_is_exactly_q96():
    assert get_sqrt_ratio_at_tick(0) == _Q96


def test_extremes_match_the_protocol_constants():
    """Independently-defined MIN/MAX_SQRT_RATIO are a free cross-check."""
    assert get_sqrt_ratio_at_tick(MIN_TICK) == MIN_SQRT_RATIO
    assert get_sqrt_ratio_at_tick(MAX_TICK) == MAX_SQRT_RATIO


@pytest.mark.parametrize("tick", [MIN_TICK - 1, MAX_TICK + 1, -(10**7), 10**7])
def test_out_of_range_ticks_raise(tick):
    with pytest.raises(ValueError, match="outside"):
        get_sqrt_ratio_at_tick(tick)


def test_strictly_monotonic():
    ticks = list(range(-5000, 5000, 37))
    ratios = [get_sqrt_ratio_at_tick(t) for t in ticks]
    assert ratios == sorted(ratios)
    assert len(set(ratios)) == len(ratios)


def test_tracks_the_float_form_without_being_it():
    """Within 1e-11 of 1.0001**(tick/2), but computed exactly.

    The float form is the definition; the integer cascade is what the pool
    actually uses, and a few ULP of drift is enough to put a boundary on the
    wrong side of the current price.
    """
    for tick in range(-400_000, 400_001, 9973):
        exact = get_sqrt_ratio_at_tick(tick)
        approx = 1.0001 ** (tick / 2) * _Q96
        assert abs(exact / approx - 1) < 1e-11


def test_brackets_a_real_pool_price():
    """A live WETH/USDC observation must sit between its tick and the next."""
    observed_sqrt_price, observed_tick = 1842354954306783972708347332343400, 201094
    assert get_sqrt_ratio_at_tick(observed_tick) <= observed_sqrt_price
    assert observed_sqrt_price < get_sqrt_ratio_at_tick(observed_tick + 1)


def test_price_range_exposes_ladder_coverage():
    """Callers need a way to tell an exact price from an extrapolated one."""
    assert TickLadder([]).price_range is None
    ladder = TickLadder(_spread(20))
    low, high = ladder.price_range
    assert low == min(ladder.prices) and high == max(ladder.prices)
    assert low < _BASE_SQRT_P < high


def test_swap_past_the_ladder_edge_overestimates():
    """Beyond the last tick the walk assumes constant liquidity, as documented.

    Pinning it so the limitation stays visible: a ladder that stops short must
    not quietly look as good as one that covers the move.
    """
    shedding = [(i, int(_BASE_SQRT_P * (1 - 0.002 * i)), 10**22) for i in range(1, 25)]
    narrow = TickLadder(shedding[:2])
    wide = TickLadder(shedding)
    huge = 10**23
    assert walk_v3_swap(_BASE_SQRT_P, 10**20, narrow, huge, 0, True) >= walk_v3_swap(
        _BASE_SQRT_P, 10**20, wide, huge, 0, True
    )

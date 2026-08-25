"""Tests for pydefi.pathfinder.rust_engine — Rust engine route -> pydefi calldata.

Pool prices are synthetic: these tests cover the seam (unit conversion, hop descriptors, command encoding), not AMM pricing, which the Rust crate tests itself. Skips when the extension is not built (``maturin develop --release`` in aggregator-rs/crates/py).
"""

from __future__ import annotations

import pytest
from eth_abi import decode as abi_decode

from pydefi._math import apply_slippage
from pydefi._utils import encode_address
from pydefi.amm.universal_router import MSG_SENDER, RouterCommand, UniversalRouter, V2Hop, V3Hop
from pydefi.pathfinder.rust_engine import (
    build_exact_in_transaction,
    route_to_hops,
    route_to_swap_route,
)
from pydefi.types import Token
from tests.addrs import DAI, ETH_WHALE, UNIVERSAL_ROUTER, USDC, WETH

amm_aggregator = pytest.importorskip(
    "amm_aggregator",
    reason="amm-aggregator not built; run `maturin develop --release` in aggregator-rs/crates/py",
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

DEADLINE_SELECTOR = bytes.fromhex("3593564c")
NO_DEADLINE_SELECTOR = bytes.fromhex("24856bc3")

POOL_V2 = "0x" + "a1" * 20
POOL_V3 = "0x" + "b2" * 20
POOL_CURVE = "0x" + "c3" * 20

#: Full tick range for a 60-spacing pool, so no test swap can cross out of it.
MIN_TICK, MAX_TICK = -887_220, 887_220


def engine_id(token: Token) -> str:
    """The string the pool graph is keyed by — an address, as the indexer emits it."""
    return encode_address(token.address, token.chain_id)


TOKENS: dict[str, Token] = {engine_id(t): t for t in (WETH, USDC, DAI)}

USDC_3K = 3_000 * 10**6


def decode_execute(tx) -> tuple[bytes, list[bytes]]:
    """Decode UniversalRouter ``execute`` calldata into ``(commands, inputs)``."""
    selector, payload = tx.data[:4], tx.data[4:]
    if selector == DEADLINE_SELECTOR:
        commands, inputs, _deadline = abi_decode(["bytes", "bytes[]", "uint256"], payload)
    else:
        assert selector == NO_DEADLINE_SELECTOR
        commands, inputs = abi_decode(["bytes", "bytes[]"], payload)
    return commands, list(inputs)


def decode_v3_input(tx) -> tuple:
    """``(recipient, amountIn, amountOutMinimum, path, payerIsUser, minHopPriceX36)`` of the final V3 command."""
    _commands, inputs = decode_execute(tx)
    return abi_decode(["address", "uint256", "uint256", "bytes", "bool", "uint256[]"], inputs[-1])


@pytest.fixture
def graph():
    """USDC -> WETH over V2, then WETH -> DAI over V3."""
    g = amm_aggregator.PoolGraph()
    g.add_v2_pool(
        POOL_V2,
        engine_id(WETH),
        engine_id(USDC),
        reserve0=10_000 * 10**18,
        reserve1=30_000_000 * 10**6,
    )
    liquidity = 10**24
    g.add_v3_pool(
        POOL_V3,
        engine_id(WETH),
        engine_id(DAI),
        fee_pips=3_000,
        sqrt_price_x96=2**96,  # price 1.0; both tokens have 18 decimals
        liquidity=liquidity,
        tick_current=0,
        tick_spacing=60,
        ticks=[(MIN_TICK, liquidity, liquidity), (MAX_TICK, -liquidity, liquidity)],
    )
    return g


@pytest.fixture
def route(graph):
    route = graph.best_route(engine_id(USDC), engine_id(DAI), USDC_3K, max_hops=2)
    assert route is not None, "fixture pools must admit a USDC -> WETH -> DAI route"
    return route


@pytest.fixture
def router() -> UniversalRouter:
    return UniversalRouter(UNIVERSAL_ROUTER)


# ---------------------------------------------------------------------------
# Route -> hop descriptors
# ---------------------------------------------------------------------------


class TestRouteToHops:
    def test_maps_each_kind_to_its_descriptor(self, route):
        hops = route_to_hops(route, TOKENS)
        assert hops == [
            V2Hop(token_in=USDC, token_out=WETH),
            V3Hop(token_in=WETH, token_out=DAI, fee=3_000),
        ]

    def test_v3_fee_stays_in_pips_but_swap_step_fee_is_bps(self, route):
        """The two units differ by 100x and both are called "fee"."""
        v3_hop = route_to_hops(route, TOKENS)[1]
        v3_step = route_to_swap_route(route, TOKENS).steps[1]
        assert v3_hop.fee == 3_000  # hundredths of a bp, as the router wants
        assert v3_step.fee == 30  # basis points, as SwapStep documents

    def test_missing_token_identifier_raises(self, route):
        with pytest.raises(KeyError):
            route_to_hops(route, {engine_id(USDC): USDC})

    def test_curve_hop_is_rejected(self):
        g = amm_aggregator.PoolGraph()
        g.add_curve_pool(
            POOL_CURVE,
            engine_id(USDC),
            engine_id(DAI),
            balances=[10**9, 10**9],
            amplification=100,
            fee_pips=400,
        )
        curve_route = g.best_route(engine_id(USDC), engine_id(DAI), 10**6, max_hops=1)
        assert curve_route is not None
        with pytest.raises(ValueError, match="curve"):
            route_to_hops(curve_route, TOKENS)


# ---------------------------------------------------------------------------
# Route -> SwapRoute
# ---------------------------------------------------------------------------


class TestRouteToSwapRoute:
    def test_carries_tokens_amounts_and_pool_addresses(self, route):
        swap_route = route_to_swap_route(route, TOKENS)
        assert swap_route.token_in == USDC
        assert swap_route.token_out == DAI
        assert swap_route.amount_in.amount == USDC_3K
        assert swap_route.amount_out.amount == route.amount_out
        assert [step.protocol for step in swap_route.steps] == ["uniswap_v2", "uniswap_v3"]
        assert swap_route.steps[0].pool_address == bytes.fromhex(POOL_V2[2:])
        assert swap_route.steps[1].tick_spacing == 60

    def test_hop_amounts_chain(self, route):
        """Each hop's input is the previous hop's output — no re-quoting needed."""
        first, second = route.hops
        assert first.amount_in == USDC_3K
        assert second.amount_in == first.amount_out
        assert second.amount_out == route.amount_out


# ---------------------------------------------------------------------------
# Route -> Universal Router calldata
# ---------------------------------------------------------------------------


class TestBuildExactInTransaction:
    def test_emits_one_command_per_pool_type(self, route, router):
        tx = build_exact_in_transaction(route, TOKENS, router, ETH_WHALE)
        commands, inputs = decode_execute(tx)
        assert tx.to == UNIVERSAL_ROUTER
        assert commands == bytes([RouterCommand.V2_SWAP_EXACT_IN, RouterCommand.V3_SWAP_EXACT_IN])
        assert len(inputs) == 2

    def test_slippage_sets_amount_out_minimum(self, route, router):
        tx = build_exact_in_transaction(route, TOKENS, router, ETH_WHALE, slippage_bps=100)
        assert decode_v3_input(tx)[2] == apply_slippage(route.amount_out, 100)

    def test_explicit_minimum_overrides_slippage(self, route, router):
        tx = build_exact_in_transaction(route, TOKENS, router, ETH_WHALE, slippage_bps=100, amount_out_minimum=7)
        assert decode_v3_input(tx)[2] == 7

    def test_deadline_switches_selector(self, route, router):
        tx = build_exact_in_transaction(route, TOKENS, router, ETH_WHALE, deadline=1_700_000_000)
        assert tx.data[:4] == DEADLINE_SELECTOR

    def test_defaults_to_msg_sender(self, route, router):
        recipient = decode_v3_input(build_exact_in_transaction(route, TOKENS, router))[0]
        assert bytes.fromhex(recipient[2:]) == MSG_SENDER

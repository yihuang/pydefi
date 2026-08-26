"""Tests for pydefi.pathfinder.rust_engine — Rust engine route -> pydefi calldata.

The engine runs as a service and Python is its client, so these tests drive the adapter from example messages — what the engine emits for the pool sets described beside them. Pricing is the engine's own concern and is tested there.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
from eth_abi import decode as abi_decode

from pydefi._math import apply_slippage
from pydefi._utils import encode_address
from pydefi.amm.universal_router import MSG_SENDER, RouterCommand, UniversalRouter, V2Hop, V3Hop
from pydefi.pathfinder.rust_engine import (
    build_exact_in_transaction,
    route_from_message,
    route_to_hops,
    route_to_message,
    route_to_swap_route,
)
from pydefi.types import Token
from tests.addrs import DAI, ETH_WHALE, UNIVERSAL_ROUTER, USDC, WETH

DEADLINE_SELECTOR = bytes.fromhex("3593564c")
NO_DEADLINE_SELECTOR = bytes.fromhex("24856bc3")

POOL_V2 = "0x" + "a1" * 20
POOL_V3 = "0x" + "b2" * 20
POOL_CURVE = "0x" + "c3" * 20

USDC_3K = 3_000 * 10**6


def engine_id(token: Token) -> str:
    """The string the pool graph keys tokens by."""
    return encode_address(token.address, token.chain_id)


TOKENS: dict[str, Token] = {engine_id(t): t for t in (WETH, USDC, DAI)}

# ---------------------------------------------------------------------------
# Example messages, as the engine emits them
# ---------------------------------------------------------------------------

#: 3000 USDC -> WETH -> DAI, max_hops=2, over a V2 pool of 10_000 WETH / 30_000_000 USDC and a full-range V3 pool at price 1.0 with liquidity 10**24, fee 3000, spacing 60.
ROUTE_MESSAGE: dict = {
    "hops": [
        {
            "address": POOL_V2,
            "kind": "uniswap_v2",
            "token_in": engine_id(USDC),
            "token_out": engine_id(WETH),
            "amount_in": "3000000000",
            "amount_out": "996900609009281774",
            "fee_pips": 3000,
            "tick_spacing": 0,
        },
        {
            "address": POOL_V3,
            "kind": "uniswap_v3",
            "token_in": engine_id(WETH),
            "token_out": engine_id(DAI),
            "amount_in": "996900609009281774",
            "amount_out": "993908919326332172",
            "fee_pips": 3000,
            "tick_spacing": 60,
        },
    ],
    "amount_in": "3000000000",
    "amount_out": "993908919326332172",
    "gas_used": 167396,
}

#: 1 USDC -> DAI over a Curve pool (balances 10**9 each, A=100, fee 400) — a kind the Universal Router has no command for.
CURVE_MESSAGE: dict = {
    "hops": [
        {
            "address": POOL_CURVE,
            "kind": "curve",
            "token_in": engine_id(USDC),
            "token_out": engine_id(DAI),
            "amount_in": "1000000",
            "amount_out": "999591",
            "fee_pips": 400,
            "tick_spacing": 0,
        }
    ],
    "amount_in": "1000000",
    "amount_out": "999591",
    "gas_used": 70000,
}


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
def route():
    return route_from_message(ROUTE_MESSAGE)


@pytest.fixture
def router() -> UniversalRouter:
    return UniversalRouter(UNIVERSAL_ROUTER)


# ---------------------------------------------------------------------------
# Message -> Route
# ---------------------------------------------------------------------------


def _with_hop(index: int, **fields) -> dict:
    """ROUTE_MESSAGE with one hop's fields overridden; ``None`` deletes a field."""
    hops = [dict(hop) for hop in ROUTE_MESSAGE["hops"]]
    hops[index].update(fields)
    hops[index] = {k: v for k, v in hops[index].items() if v is not None}
    return {"hops": hops}


class TestRouteFromMessage:
    def test_decodes_hops_and_amounts(self, route):
        assert [hop.kind for hop in route.hops] == ["uniswap_v2", "uniswap_v3"]
        assert (route.amount_in, route.amount_out, route.gas_used) == (USDC_3K, 993908919326332172, 167396)

    def test_survives_a_json_round_trip(self):
        decoded = route_from_message(json.loads(json.dumps(ROUTE_MESSAGE)))
        assert route_to_message(decoded) == ROUTE_MESSAGE

    def test_accepts_int_amounts(self, route):
        """A codec like msgpack carries u256 as ints, not strings."""
        assert route_from_message({**ROUTE_MESSAGE, "hops": [asdict(hop) for hop in route.hops]}) == route

    def test_route_amounts_are_optional(self, route):
        assert route_from_message({"hops": ROUTE_MESSAGE["hops"]}) == replace(route, gas_used=0)

    @pytest.mark.parametrize(
        ("message", "match"),
        [
            pytest.param({"hops": []}, "no hops", id="empty"),
            pytest.param({**ROUTE_MESSAGE, "hops": ROUTE_MESSAGE["hops"][:1]}, "amount_out", id="dropped-hop"),
            pytest.param({**ROUTE_MESSAGE, "amount_out": "1"}, "amount_out", id="route-amount-disagrees"),
            pytest.param(_with_hop(1, amount_in="1"), "hops\\[1\\] takes", id="amount-gap"),
            pytest.param(_with_hop(1, token_in=engine_id(DAI)), "hops\\[1\\] takes", id="token-gap"),
            pytest.param(_with_hop(0, fee_pips=None), "hops\\[0\\]: missing field 'fee_pips'", id="missing-field"),
            pytest.param(_with_hop(0, amount_in="0x1f"), "not an integer", id="hex-amount"),
            pytest.param(_with_hop(0, amount_in=3e9), "is a float", id="float-amount"),
        ],
    )
    def test_malformed_message_raises(self, message, match):
        """Bad hop lists fail at the boundary, not as short or mis-sized calldata."""
        with pytest.raises(ValueError, match=match):
            route_from_message(message)


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
        with pytest.raises(ValueError, match="curve"):
            route_to_hops(route_from_message(CURVE_MESSAGE), TOKENS)


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

    def test_accepts_every_engine_kind(self):
        """A SwapRoute only describes the route, so unencodable kinds still convert."""
        swap_route = route_to_swap_route(route_from_message(CURVE_MESSAGE), TOKENS)
        assert [step.protocol for step in swap_route.steps] == ["curve"]
        assert swap_route.steps[0].fee == 4  # 400 pips == 4 bps


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

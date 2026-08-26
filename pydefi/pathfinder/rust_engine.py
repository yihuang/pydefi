"""Adapter for routes from the Rust engine (``amm-aggregator``).

The engine runs as a service (``serve`` in aggregator-rs); this module is its client. Hops describe themselves, so nothing here looks a pool up — it decodes, converts units, and packs calldata::

    route = EngineClient("http://127.0.0.1:8080").best_route(weth_addr, usdc_addr, 10**18)
    tx = build_exact_in_transaction(route, {weth_addr: WETH, usdc_addr: USDC}, router)

Message format::

    {"hops": [{"address": "0x…", "kind": "uniswap_v2", "token_in": "0x…", "token_out": "0x…",
               "amount_in": "3000000000", "amount_out": "996006981", "fee_pips": 3000, "tick_spacing": 0}],
     "amount_in": "3000000000", "amount_out": "996006981", "gas_used": 116000, "block_number": 21000000}

u256 amounts are decimal strings (ints work too); route-level fields are optional. ``block_number`` is the chain head the search reflected.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

import requests

from pydefi._math import apply_slippage
from pydefi.amm.universal_router import MSG_SENDER, PoolHop, UniversalRouter, V2Hop, V3Hop
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount
from pydefi.vm.swap import SwapTransaction

__all__ = [
    "EngineClient",
    "Route",
    "RouteHop",
    "build_exact_in_transaction",
    "route_from_message",
    "route_to_hops",
    "route_to_message",
    "route_to_swap_route",
]


@dataclass(frozen=True)
class RouteHop:
    """One hop of an engine route, as the engine's ``RouteHop`` serializes."""

    address: str
    kind: str
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    fee_pips: int
    tick_spacing: int = 0


@dataclass(frozen=True)
class Route:
    """An engine route, as the engine's ``Route`` serializes."""

    hops: Sequence[RouteHop]
    amount_in: int
    amount_out: int
    gas_used: int = 0
    #: The chain head the search reflected; 0 when unknown.
    block_number: int = 0


def _amount(value: Any, field: str) -> int:
    """A u256 sent as a decimal string or an int. Floats are refused: ``int()`` would truncate them."""
    if isinstance(value, float):
        raise ValueError(f"{field}: {value!r} is a float")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {value!r} is not an integer") from exc


def _hop_from_message(message: Mapping[str, Any], at: str) -> RouteHop:
    try:
        return RouteHop(
            address=message["address"],
            kind=message["kind"],
            token_in=message["token_in"],
            token_out=message["token_out"],
            amount_in=_amount(message["amount_in"], f"{at}.amount_in"),
            amount_out=_amount(message["amount_out"], f"{at}.amount_out"),
            fee_pips=int(message["fee_pips"]),
            tick_spacing=int(message.get("tick_spacing", 0)),
        )
    except KeyError as exc:
        raise ValueError(f"{at}: missing field {exc.args[0]!r}") from exc


def route_from_message(message: Mapping[str, Any]) -> Route:
    """Decode a route message. Hops must chain and route-level amounts must match them, so a bad hop list fails here, not in the calldata."""
    hops = tuple(_hop_from_message(hop, f"hops[{i}]") for i, hop in enumerate(message.get("hops", ())))
    if not hops:
        raise ValueError("route message has no hops")
    for i in range(1, len(hops)):
        prev, hop = hops[i - 1], hops[i]
        if (hop.token_in, hop.amount_in) != (prev.token_out, prev.amount_out):
            raise ValueError(
                f"hops[{i}] takes {hop.amount_in} {hop.token_in} but hops[{i - 1}] gives {prev.amount_out} {prev.token_out}"
            )
    route = Route(
        hops,
        hops[0].amount_in,
        hops[-1].amount_out,
        int(message.get("gas_used", 0)),
        int(message.get("block_number", 0)),
    )
    for field in ("amount_in", "amount_out"):
        if field in message and _amount(message[field], field) != getattr(route, field):
            raise ValueError(f"{field} is {message[field]} but the hops give {getattr(route, field)}")
    return route


def route_to_message(route: Route) -> dict[str, Any]:
    """The inverse of :func:`route_from_message`; amounts become decimal strings."""
    return {
        "hops": [
            {
                "address": hop.address,
                "kind": hop.kind,
                "token_in": hop.token_in,
                "token_out": hop.token_out,
                "amount_in": str(hop.amount_in),
                "amount_out": str(hop.amount_out),
                "fee_pips": hop.fee_pips,
                "tick_spacing": hop.tick_spacing,
            }
            for hop in route.hops
        ],
        "amount_in": str(route.amount_in),
        "amount_out": str(route.amount_out),
        "gas_used": route.gas_used,
        "block_number": route.block_number,
    }


#: The engine and the Universal Router use pips (1_000_000 = 100%), :class:`SwapStep` uses basis points — 100x apart, same name. Convert here and nowhere else.
_PIPS_PER_BPS = 100

#: Engine kinds that encode as a Universal Router V3 command. The forks share pool math and encoding but live at other factories, so a fork hop needs a router deployment that knows its factory.
_V3_KINDS = frozenset({"uniswap_v3", "pancakeswap_v3", "camelot_v3"})


def _hop_to_descriptor(hop: RouteHop, tokens: Mapping[str, Token]) -> PoolHop:
    token_in, token_out = tokens[hop.token_in], tokens[hop.token_out]
    if hop.kind == "uniswap_v2":
        return V2Hop(token_in=token_in, token_out=token_out)
    if hop.kind in _V3_KINDS:
        return V3Hop(token_in=token_in, token_out=token_out, fee=hop.fee_pips)  # V3Hop.fee is pips too
    raise ValueError(f"pool {hop.address}: the Universal Router has no command for {hop.kind!r} pools")


def route_to_hops(route: Route, tokens: Mapping[str, Token]) -> list[PoolHop]:
    """Convert an engine route into Universal Router hop descriptors.

    *tokens* maps the engine's token identifiers (the strings the pool graph was built with) to :class:`Token`; a missing one raises ``KeyError``. A hop the Universal Router cannot encode (Curve, Balancer) raises ``ValueError``.
    """
    if not route.hops:
        raise ValueError("route has no hops")
    return [_hop_to_descriptor(hop, tokens) for hop in route.hops]


def route_to_swap_route(route: Route, tokens: Mapping[str, Token]) -> SwapRoute:
    """Convert an engine route into pydefi's protocol-neutral :class:`SwapRoute`.

    Accepts every engine AMM kind, unlike :func:`route_to_hops` — a ``SwapRoute`` only describes the route. ``price_impact`` stays zero: the engine reports realised output, not a spot reference.
    """
    if not route.hops:
        raise ValueError("route has no hops")
    steps = [
        SwapStep(
            token_in=tokens[hop.token_in],
            token_out=tokens[hop.token_out],
            pool_address=Address(hop.address) if hop.address.startswith("0x") else None,
            protocol=hop.kind,
            fee=hop.fee_pips // _PIPS_PER_BPS,
            tick_spacing=hop.tick_spacing,
        )
        for hop in route.hops
    ]
    return SwapRoute(
        steps=steps,
        amount_in=TokenAmount(steps[0].token_in, route.amount_in),
        amount_out=TokenAmount(steps[-1].token_out, route.amount_out),
        price_impact=Decimal(0),
    )


def build_exact_in_transaction(
    route: Route,
    tokens: Mapping[str, Token],
    router: UniversalRouter,
    recipient: Address = MSG_SENDER,
    *,
    slippage_bps: int = 50,
    amount_out_minimum: int | None = None,
    deadline: int | None = None,
) -> SwapTransaction:
    """Pack an engine route into Universal Router ``execute`` calldata.

    *slippage_bps* is applied to the engine's quoted output unless *amount_out_minimum* is given. Raises as :func:`route_to_hops` does.
    """
    hops = route_to_hops(route, tokens)
    if amount_out_minimum is None:
        amount_out_minimum = apply_slippage(route.amount_out, slippage_bps)
    return router.build_multihop_exact_in_transaction(
        amount_in=TokenAmount(hops[0].token_in, route.amount_in),
        hops=hops,
        recipient=recipient,
        amount_out_minimum=amount_out_minimum,
        deadline=deadline,
    )


class EngineClient:
    """Client for the engine service."""

    def __init__(self, url: str = "http://127.0.0.1:8080", *, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        """``{"chain_id", "block_number", "pools"}``."""
        response = requests.get(f"{self.url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def best_route(self, token_in: str, token_out: str, amount_in: int, max_hops: int = 3) -> Route | None:
        """Best exact-in route, or ``None`` when no path exists within *max_hops*."""
        body = {"token_in": token_in, "token_out": token_out, "amount_in": str(amount_in), "max_hops": max_hops}
        response = requests.post(f"{self.url}/route", json=body, timeout=self.timeout)
        if response.status_code == 404 and response.json().get("error") == "no route":
            return None
        response.raise_for_status()
        return route_from_message(response.json())

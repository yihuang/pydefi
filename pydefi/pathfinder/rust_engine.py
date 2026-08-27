"""Adapter for routes from the Rust engine (``amm-aggregator``).

The engine runs as a service (``serve`` in aggregator-rs); this module is its client. Hops describe themselves, so nothing here looks a pool up — it decodes, converts units, and packs calldata::

    route = EngineClient("http://127.0.0.1:8080").best_route(weth_addr, usdc_addr, 10**18)
    tx = build_exact_in_transaction(route, {weth_addr: WETH, usdc_addr: USDC}, router)

Message format::

    {"mode": "single",
     "hops": [{"address": "0x…", "kind": "uniswap_v2", "token_in": "0x…", "token_out": "0x…",
               "amount_in": "3000000000", "amount_out": "996006981", "fee_pips": 3000, "tick_spacing": 0}],
     "amount_in": "3000000000", "amount_out": "996006981", "gas_used": 116000,
     "block_number": 21000000, "state_version": 12, "stale": false}

Ask for ``split`` and the engine may instead spread the trade across paths, which it does only when the allocation beats every single route net of gas::

    {"mode": "split",
     "legs": [{"hops": [...], "amount_in": "…", "amount_out": "…", "gas_used": 116000}, ...],
     "amount_in": "total", "amount_out": "total", "gas_used": 232000,
     "block_number": 21000000, "state_version": 12, "stale": false}

Legs sum to what was asked, so a split is several swaps. :func:`quote_from_message` decodes either shape on ``mode``: a ``split`` whose allocation lost comes back ``single``.

u256 amounts are decimal strings (ints work too); route-level fields are optional. ``block_number`` is the head the search reflected, ``state_version`` counts applied updates, and ``stale`` means a pool was scanned too far behind it — do not build a minimum output on a stale quote.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    "SplitQuote",
    "build_exact_in_transaction",
    "quote_from_message",
    "route_from_message",
    "route_to_hops",
    "route_to_message",
    "route_to_swap_route",
    "split_from_message",
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
    """An engine route. Always has hops, so the packers below never re-check."""

    hops: Sequence[RouteHop]
    amount_in: int
    amount_out: int
    gas_used: int = 0
    #: The chain head the search reflected; 0 when unknown.
    block_number: int = 0
    #: The engine's count of applied pool updates; identical quotes with the same version are identical.
    state_version: int = 0
    #: A pool on the route was scanned too far behind the head for the engine's guard; indicative only.
    stale: bool = False

    def __post_init__(self) -> None:
        if not self.hops:
            raise ValueError("route has no hops")


def _amount(value: Any, field: str) -> int:
    """An integer as a decimal string or an int. Floats are refused: a truncated fee tier is a wrong pool, not a rounding error."""
    if isinstance(value, float):
        raise ValueError(f"{field}: {value!r} is a float")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {value!r} is not an integer") from exc


def _optional(message: Mapping[str, Any], field: str, default: int = 0) -> int:
    """A route-level integer the engine may omit."""
    return _amount(message[field], field) if field in message else default


def _hop_from_message(message: Mapping[str, Any], at: str) -> RouteHop:
    try:
        return RouteHop(
            address=message["address"],
            kind=message["kind"],
            token_in=message["token_in"],
            token_out=message["token_out"],
            amount_in=_amount(message["amount_in"], f"{at}.amount_in"),
            amount_out=_amount(message["amount_out"], f"{at}.amount_out"),
            fee_pips=_amount(message["fee_pips"], f"{at}.fee_pips"),
            tick_spacing=_optional(message, "tick_spacing"),
        )
    except KeyError as exc:
        raise ValueError(f"{at}: missing field {exc.args[0]!r}") from exc
    except TypeError as exc:
        raise ValueError(f"{at}: {message!r} is not a hop") from exc


def route_from_message(message: Mapping[str, Any]) -> Route:
    """Decode a route message. Hops must chain and the amounts must match them, so a bad list fails here, not in the calldata."""
    raw = message.get("hops", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"hops: {raw!r} is not a list of hops")
    hops = tuple(_hop_from_message(hop, f"hops[{i}]") for i, hop in enumerate(raw))
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
        _optional(message, "gas_used"),
        _optional(message, "block_number"),
        _optional(message, "state_version"),
        bool(message.get("stale", False)),
    )
    for field in ("amount_in", "amount_out"):
        if field in message and _amount(message[field], field) != getattr(route, field):
            raise ValueError(f"{field} is {message[field]} but the hops give {getattr(route, field)}")
    return route


def route_to_message(route: Route) -> dict[str, Any]:
    """The inverse of :func:`route_from_message`; amounts become decimal strings."""
    return {
        "mode": "single",
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
        "state_version": route.state_version,
        "stale": route.stale,
    }


@dataclass(frozen=True)
class SplitQuote:
    """``amount_in`` spread across routes, because that beat every single one net of gas.

    Each leg is a full :class:`Route` and they sum to :attr:`amount_in`, so a split is several swaps: pack one leg at a time, there is no ``execute`` for the whole allocation yet.
    """

    legs: Sequence[Route]
    amount_in: int
    amount_out: int
    gas_used: int = 0
    #: The chain head the search reflected; 0 when unknown.
    block_number: int = 0
    #: The engine's count of applied pool updates; identical quotes with the same version are identical.
    state_version: int = 0
    #: Some leg was scanned too far behind the head for the engine's guard; indicative only.
    stale: bool = False


def split_from_message(message: Mapping[str, Any]) -> SplitQuote:
    """Decode a split message.

    Legs must be the same trade and must sum to the totals, so a mis-routed leg fails here, not as a swap into the wrong token. Each is stamped with the quote's freshness: a leg is what gets packed.
    """
    block_number = _optional(message, "block_number")
    state_version = _optional(message, "state_version")
    stale = bool(message.get("stale", False))
    legs = []
    for i, leg in enumerate(message.get("legs", ())):
        try:
            legs.append(
                replace(
                    route_from_message(leg),
                    block_number=block_number,
                    state_version=state_version,
                    stale=stale,
                )
            )
        except ValueError as exc:
            raise ValueError(f"legs[{i}]: {exc}") from exc
    if not legs:
        raise ValueError("split message has no legs")
    first = legs[0]
    for i, leg in enumerate(legs[1:], start=1):
        ends = (leg.hops[0].token_in, leg.hops[-1].token_out)
        if ends != (first.hops[0].token_in, first.hops[-1].token_out):
            raise ValueError(
                f"legs[{i}] swaps {ends[0]} to {ends[1]}, but legs[0] swaps {first.hops[0].token_in} to {first.hops[-1].token_out}"
            )
    split = SplitQuote(
        tuple(legs),
        sum(leg.amount_in for leg in legs),
        sum(leg.amount_out for leg in legs),
        sum(leg.gas_used for leg in legs),
        block_number,
        state_version,
        stale,
    )
    for field in ("amount_in", "amount_out", "gas_used"):
        if field in message and _amount(message[field], field) != getattr(split, field):
            raise ValueError(f"{field} is {message[field]} but the legs give {getattr(split, field)}")
    return split


def quote_from_message(message: Mapping[str, Any]) -> Route | SplitQuote:
    """Decode either shape, on ``mode``. Asking for a split can still answer ``single``, so handle both."""
    mode = message.get("mode")
    if mode == "split":
        return split_from_message(message)
    if mode in (None, "single"):
        return route_from_message(message)
    raise ValueError(f"unknown route mode {mode!r}")


#: Engine and router use pips (1_000_000 = 100%), :class:`SwapStep` uses bps — 100x apart, same name. Convert here and nowhere else.
_PIPS_PER_BPS = 100

#: Kinds that encode as a V3 command. The forks share the encoding but live at other factories, so they need a router deployment that knows theirs.
_V3_KINDS = frozenset({"uniswap_v3", "pancakeswap_v3", "camelot_v3"})


#: The V2 command carries no pool address: the router re-derives the pair from the Uniswap factory, where the fee is always 0.3%.
_V2_FEE_PIPS = 3000


def _hop_to_descriptor(hop: RouteHop, tokens: Mapping[str, Token]) -> PoolHop:
    token_in, token_out = tokens[hop.token_in], tokens[hop.token_out]
    if hop.kind == "uniswap_v2":
        if hop.fee_pips != _V2_FEE_PIPS:
            raise ValueError(
                f"pool {hop.address}: quoted at {hop.fee_pips} pips, but the V2 command would execute against the 3000-pip Uniswap pair"
            )
        return V2Hop(token_in=token_in, token_out=token_out)
    if hop.kind in _V3_KINDS:
        return V3Hop(token_in=token_in, token_out=token_out, fee=hop.fee_pips)  # V3Hop.fee is pips too
    raise ValueError(f"pool {hop.address}: the Universal Router has no command for {hop.kind!r} pools")


def route_to_hops(route: Route, tokens: Mapping[str, Token]) -> list[PoolHop]:
    """Convert an engine route into Universal Router hop descriptors.

    *tokens* maps the engine's token identifiers (the strings the pool graph was built with) to :class:`Token`; a missing one raises ``KeyError``. A hop the Universal Router cannot encode (Curve, Balancer) raises ``ValueError``.
    """
    return [_hop_to_descriptor(hop, tokens) for hop in route.hops]


def route_to_swap_route(route: Route, tokens: Mapping[str, Token]) -> SwapRoute:
    """Convert an engine route into pydefi's protocol-neutral :class:`SwapRoute`.

    Accepts every engine AMM kind, unlike :func:`route_to_hops` — a ``SwapRoute`` only describes the route. ``price_impact`` stays zero: the engine reports realised output, not a spot reference.
    """
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
    if isinstance(route, SplitQuote):
        raise ValueError("a split is several swaps; pack one leg at a time, or ask for a single route")
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


def _no_route(response: requests.Response) -> bool:
    """Whether a 404 is the engine's own "no route". A proxy 404s with a body we cannot read, and must raise as HTTP, not JSON."""
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, Mapping) and body.get("error") == "no route"


class EngineClient:
    """Client for the engine service."""

    def __init__(self, url: str = "http://127.0.0.1:8080", *, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        #: Kept alive across quotes: a fresh handshake costs more than the engine takes to answer.
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        """``{"chain_id", "block_number", "state_version", "pools"}``."""
        response = self._session.get(f"{self.url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        max_hops: int = 3,
        *,
        k: int = 1,
        split: bool = False,
        gas_price_out: int | None = None,
    ) -> Route | SplitQuote | None:
        """A quote, or ``None`` when no path exists within *max_hops*.

        *k* is how many candidate paths the engine materializes and ranks; the default 1 is its exact single-route search, which ignores *gas_price_out*. Above 1, and with *split*, candidates are ranked net of gas and one that costs more gas than it returns is refused as "no route" rather than quoted — so pass *gas_price_out* in output-token units to get that. *split* lets the engine spread the trade across paths, which it does only when that beats every single route net of gas; the answer is a :class:`SplitQuote` when it did and a :class:`Route` when it did not.
        """
        body: dict[str, Any] = {
            "token_in": token_in,
            "token_out": token_out,
            "amount_in": str(amount_in),
            "max_hops": max_hops,
        }
        if k != 1:
            body["k"] = k
        if split:
            body["split"] = True
        if gas_price_out is not None:
            body["gas_price_out"] = str(gas_price_out)
        response = self._session.post(f"{self.url}/route", json=body, timeout=self.timeout)
        if response.status_code == 404 and _no_route(response):
            return None
        response.raise_for_status()
        return quote_from_message(response.json())

    def best_route(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        max_hops: int = 3,
        *,
        k: int = 1,
        gas_price_out: int | None = None,
    ) -> Route | None:
        """Best exact-in route, or ``None``. Never splits. Check ``route.stale`` first."""
        quote = self.quote(token_in, token_out, amount_in, max_hops, k=k, gas_price_out=gas_price_out)
        if isinstance(quote, SplitQuote):
            raise ValueError("engine answered with a split for a single-route request")
        return quote

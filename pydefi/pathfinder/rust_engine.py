"""Adapter for routes from the Rust engine (``amm-aggregator``).

The engine simulates AMMs and searches routes purely in memory; pydefi turns the winning route into calldata. Engine hops describe themselves (pool address, kind, tokens, per-hop amounts, fee), so this module never looks a pool up again — it only converts units and builds hop descriptors::

    graph = amm_aggregator.PoolGraph()
    graph.add_v2_pool(pool_addr, weth_addr, usdc_addr, reserve0, reserve1)
    route = graph.best_route(weth_addr, usdc_addr, 10**18, max_hops=3)
    tx = build_exact_in_transaction(route, {weth_addr: WETH, usdc_addr: USDC}, router)

The engine is not on PyPI: ``cd aggregator-rs/crates/py && maturin develop --release``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Protocol, Sequence

from pydefi._math import apply_slippage
from pydefi.amm.universal_router import MSG_SENDER, PoolHop, UniversalRouter, V2Hop, V3Hop
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount
from pydefi.vm.swap import SwapTransaction

__all__ = [
    "RouteHopLike",
    "RouteLike",
    "build_exact_in_transaction",
    "route_to_hops",
    "route_to_swap_route",
]


class RouteHopLike(Protocol):
    """Structural type of one ``amm_aggregator.RouteHop``."""

    address: str
    kind: str
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    fee_pips: int
    tick_spacing: int


class RouteLike(Protocol):
    """Structural type of one ``amm_aggregator.Route``."""

    hops: Sequence[RouteHopLike]
    amount_in: int
    amount_out: int
    gas_used: int


#: The engine and the Universal Router use pips (1_000_000 = 100%), :class:`SwapStep` uses basis points — 100x apart, same name. Convert here and nowhere else.
_PIPS_PER_BPS = 100

#: Engine kinds that encode as a Universal Router V3 command. The forks share pool math and encoding but live at other factories, so a fork hop needs a router deployment that knows its factory.
_V3_KINDS = frozenset({"uniswap_v3", "pancakeswap_v3", "camelot_v3"})


def _hop_to_descriptor(hop: RouteHopLike, tokens: Mapping[str, Token]) -> PoolHop:
    token_in, token_out = tokens[hop.token_in], tokens[hop.token_out]
    if hop.kind == "uniswap_v2":
        return V2Hop(token_in=token_in, token_out=token_out)
    if hop.kind in _V3_KINDS:
        return V3Hop(token_in=token_in, token_out=token_out, fee=hop.fee_pips)  # V3Hop.fee is pips too
    raise ValueError(f"pool {hop.address}: the Universal Router has no command for {hop.kind!r} pools")


def route_to_hops(route: RouteLike, tokens: Mapping[str, Token]) -> list[PoolHop]:
    """Convert an engine route into Universal Router hop descriptors.

    *tokens* maps the engine's token identifiers (the strings the pool graph was built with) to :class:`Token`; a missing one raises ``KeyError``. A hop the Universal Router cannot encode (Curve, Balancer) raises ``ValueError``.
    """
    if not route.hops:
        raise ValueError("route has no hops")
    return [_hop_to_descriptor(hop, tokens) for hop in route.hops]


def route_to_swap_route(route: RouteLike, tokens: Mapping[str, Token]) -> SwapRoute:
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
    route: RouteLike,
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

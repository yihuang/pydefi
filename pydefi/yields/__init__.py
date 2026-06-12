"""Cross-protocol yield orchestration over :mod:`pydefi.lending` and
:mod:`pydefi.bridge`. No new on-chain primitive; everything composes the
existing readers / tx builders."""

from pydefi.yields.compose import build_compose_supply_program, build_compose_supply_route
from pydefi.yields.rebalance import expected_apy_gain, find_best_rebalance, rebalance_tick
from pydefi.yields.router import (
    PendingLeg,
    YieldMarket,
    YieldRoute,
    YieldStep,
    build_approve_tx,
    build_followup_route,
    build_yield_route,
    get_yield_markets,
    sign_route,
)
from pydefi.yields.tracker import Position, get_positions, wait_for_bridge_settlement

__all__ = [
    "PendingLeg",
    "Position",
    "YieldMarket",
    "YieldRoute",
    "YieldStep",
    "build_approve_tx",
    "build_compose_supply_route",
    "build_followup_route",
    "build_compose_supply_program",
    "build_yield_route",
    "expected_apy_gain",
    "find_best_rebalance",
    "get_positions",
    "get_yield_markets",
    "rebalance_tick",
    "sign_route",
    "wait_for_bridge_settlement",
]

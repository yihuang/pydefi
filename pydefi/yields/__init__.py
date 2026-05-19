"""Cross-protocol yield orchestration over :mod:`pydefi.lending` and
:mod:`pydefi.bridge`. No new on-chain primitive; everything composes the
existing readers / tx builders."""

from pydefi.yields.router import (
    YieldMarket,
    YieldRoute,
    YieldStep,
    build_approve_tx,
    build_yield_route,
    get_yield_markets,
)

__all__ = [
    "YieldMarket",
    "YieldRoute",
    "YieldStep",
    "build_approve_tx",
    "build_yield_route",
    "get_yield_markets",
]

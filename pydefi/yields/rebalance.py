"""Closes the loop between :func:`get_positions` (what the user holds) and
:func:`build_yield_route` (how to move it). Gas-vs-yield comparison is
left to the caller — gas is paid in native currency and converting to the
position token needs a price feed this module does not depend on."""

from __future__ import annotations

from decimal import Decimal

from web3 import AsyncWeb3

from pydefi.lending.utils import SECONDS_PER_YEAR
from pydefi.types import Address
from pydefi.yields.router import (
    Protocol,
    YieldMarket,
    YieldRoute,
    build_yield_route,
    get_yield_markets,
)
from pydefi.yields.tracker import Position, get_positions


def expected_apy_gain(
    current: Position,
    candidate: YieldMarket,
    horizon_seconds: int,
) -> int:
    """Token-sub-unit gain from moving *current* into *candidate*. Negative
    when the candidate is worse. Linear approximation — assumes neither
    rate moves over the horizon and ignores compounding."""
    delta_apy = candidate.supply_apy - current.market.supply_apy
    horizon = Decimal(horizon_seconds) / SECONDS_PER_YEAR
    gain = Decimal(current.balance.amount) * delta_apy * horizon
    return int(gain.to_integral_value())


async def find_best_rebalance(
    user: Address,
    positions: list[Position],
    markets: list[YieldMarket],
    w3s: dict[int, AsyncWeb3],
    *,
    horizon_seconds: int,
    min_apy_gain_bps: int = 50,
) -> YieldRoute | None:
    """Most profitable same-chain rebalance, or ``None`` when no candidate
    clears ``min_apy_gain_bps``. Cross-chain isn't auto-selected — that
    route is multi-tx with a bridge wait; build it explicitly."""
    threshold = Decimal(min_apy_gain_bps) / Decimal(10_000)
    best: tuple[int, Position, YieldMarket] | None = None
    for position in positions:
        for candidate in markets:
            if candidate.chain_id != position.market.chain_id:
                continue
            if candidate.market_id == position.market.market_id:
                continue
            if candidate.supply_apy - position.market.supply_apy < threshold:
                continue
            gain = expected_apy_gain(position, candidate, horizon_seconds)
            if gain <= 0:
                continue
            if best is None or gain > best[0]:
                best = (gain, position, candidate)

    if best is None:
        return None
    _gain, position, candidate = best
    return await build_yield_route(
        "withdraw_then_supply",
        user=user,
        amount_in=position.balance,
        w3s=w3s,
        source_market=position.market,
        target_market=candidate,
    )


async def rebalance_tick(
    user: Address,
    token_symbol: str,
    w3s: dict[int, AsyncWeb3],
    *,
    horizon_seconds: int = 30 * 86400,
    min_apy_gain_bps: int = 50,
    chains: list[int] | None = None,
    protocols: list[Protocol] | None = None,
) -> YieldRoute | None:
    """One scan + decision cycle for an automated rebalancer. Callers drive
    it from their own scheduler / cron / loop — pydefi does not ship a daemon."""
    positions = await get_positions(user, token_symbol, w3s, chains=chains, protocols=protocols)
    if not positions:
        return None
    markets = await get_yield_markets(token_symbol, w3s, chains=chains, protocols=protocols)
    return await find_best_rebalance(
        user,
        positions=positions,
        markets=markets,
        w3s=w3s,
        horizon_seconds=horizon_seconds,
        min_apy_gain_bps=min_apy_gain_bps,
    )

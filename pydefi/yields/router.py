"""Enumerate yield markets and build sequenced execution plans
(``withdraw_then_supply``, ``supply_then_bridge``, ``bridge_then_supply``).
Caller supplies a per-chain ``AsyncWeb3`` map and broadcasts the steps."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3

from pydefi.bridge.lucid import LucidBridge
from pydefi.deployments import chains_for, comet_contract_for, get_token
from pydefi.lending.aave_v3 import AaveV3
from pydefi.lending.compound_v3 import CompoundV3
from pydefi.types import Address, Token, TokenAmount

Protocol = Literal["aave_v3", "compound_v3"]
Strategy = Literal["withdraw_then_supply", "supply_then_bridge", "bridge_then_supply"]
StepKind = Literal["approve", "supply", "withdraw", "bridge"]

# Generous; fits quirky tokens whose approve consumes more than the EIP-20 minimum.
_APPROVE_GAS = 100_000


@dataclass(frozen=True)
class YieldMarket:
    """``market_id`` is stable across runs — safe for cache keys."""

    protocol: Protocol
    chain_id: int
    token: Token
    supply_apy: Decimal
    utilization: Decimal
    available_liquidity: TokenAmount
    market_id: str


@dataclass(frozen=True)
class YieldStep:
    """``tx`` is the standard pydefi tx-dict ``{to, data, value, gas}``."""

    kind: StepKind
    chain_id: int
    tx: dict[str, Any]


@dataclass(frozen=True)
class PendingLeg:
    """The deferred destination-chain supply of a cross-chain route.

    A cross-chain :class:`YieldRoute` carries only source-chain steps: the
    destination supply cannot be built until the bridge settles and the
    received amount is known. This records what that follow-up will do —
    pass it, with the settled amount, to :func:`build_followup_route`.

    Until the follow-up runs the user simply holds ``market.token`` on
    ``chain_id``; a partially completed cross-chain route never strands
    funds, it leaves them as the bridged token.

    Attributes:
        chain_id: The destination chain the supply will run on.
        market: The :class:`YieldMarket` the bridged funds will be supplied to.
    """

    chain_id: int
    market: YieldMarket


@dataclass(frozen=True)
class YieldRoute:
    """Sequenced plan of source-chain steps.

    For a cross-chain route the destination supply is not pre-built — the
    bridged amount is unknown until the bridge settles. :attr:`pending`
    describes that deferred leg; build it with :func:`build_followup_route`
    once the funds arrive. A same-chain route has ``pending is None``."""

    strategy: Strategy
    source_chain: int
    steps: tuple[YieldStep, ...]
    route_id: str
    target_market: YieldMarket | None = None
    target_chain: int | None = None
    pending: PendingLeg | None = None

    def build_transactions(self) -> list[dict[str, Any]]:
        return [s.tx for s in self.steps]


def _candidate_chains(token_symbol: str) -> list[int]:
    aave = set(chains_for("AAVE_V3_ADDRESSES_PROVIDER"))
    comet_name = comet_contract_for(token_symbol)
    comet = set(chains_for(comet_name)) if comet_name else set()
    return sorted(aave | comet)


def _same_token_identity(a: Token, b: Token) -> bool:
    # Identity = (chain, address). Symbol/decimals are not compared because
    # Compound markets carry a "BASE" placeholder symbol from baseToken().
    return a.chain_id == b.chain_id and a.address == b.address


def _require_token(token: Token, expected: Token, what: str) -> None:
    """Raise ``ValueError`` unless *token* matches *expected* by (chain, address)."""
    if not _same_token_identity(token, expected):
        raise ValueError(f"{what} (got {token}, expected {expected})")


# ---------------------------------------------------------------------------
# Transaction / step builders
# ---------------------------------------------------------------------------


def build_approve_tx(token: Token, spender: Address, amount: int, gas: int = _APPROVE_GAS) -> dict[str, Any]:
    return {
        "to": token.address,
        "data": "0x" + ERC20.fns.approve(bytes(spender), amount).data.hex(),
        "value": "0",
        "gas": str(gas),
    }


async def _withdraw_tx(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> dict[str, Any]:
    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        return aave.build_withdraw_tx(user, amount)
    return CompoundV3.from_chain(w3, market.chain_id, market.token.symbol).build_withdraw_tx(amount)


async def _supply_steps(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> list[YieldStep]:
    """``[approve, supply]`` — approve the market's spender, then supply.

    The ERC-20 ``approve`` spender is the contract that pulls the funds —
    the Aave V3 ``Pool`` or the Compound III ``Comet``. Both come off the
    constructed client, so the approval always targets the exact contract
    the supply call hits."""
    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        spender, supply_tx = aave.pool_address, aave.build_supply_tx(user, amount)
    else:
        comet = CompoundV3.from_chain(w3, market.chain_id, market.token.symbol)
        spender, supply_tx = comet.comet_address, comet.build_supply_tx(amount)
    return [
        YieldStep("approve", market.chain_id, build_approve_tx(amount.token, spender, amount.amount)),
        YieldStep("supply", market.chain_id, supply_tx),
    ]


async def _bridge_steps(
    bridge: LucidBridge, user: Address, amount: TokenAmount, target_token: Token
) -> list[YieldStep]:
    """``[approve, bridge]`` — approve the Lucid controller, then bridge."""
    bridge_tx = await bridge.build_bridge_tx(amount.token, target_token, amount, user)
    return [
        YieldStep(
            "approve", bridge.src_chain_id, build_approve_tx(amount.token, bridge.controller_address, amount.amount)
        ),
        YieldStep("bridge", bridge.src_chain_id, bridge_tx),
    ]


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------


async def _aave_market(w3: AsyncWeb3, chain_id: int, token: Token, token_symbol: str) -> YieldMarket | None:
    if chain_id not in chains_for("AAVE_V3_ADDRESSES_PROVIDER"):
        return None
    aave = await AaveV3.from_chain(w3, chain_id)
    reserve = await aave.get_reserve_data(token)
    # Inactive / frozen / paused reserves cannot accept supplies — the APY
    # would be informational only.
    if not reserve.is_active or reserve.is_frozen or reserve.is_paused:
        return None
    return YieldMarket(
        protocol="aave_v3",
        chain_id=chain_id,
        token=token,
        supply_apy=reserve.supply_apy,
        utilization=reserve.utilization,
        available_liquidity=reserve.available_liquidity,
        market_id=f"aave_v3:{chain_id}:{token_symbol}",
    )


async def _compound_market(w3: AsyncWeb3, chain_id: int, token_symbol: str) -> YieldMarket | None:
    name = comet_contract_for(token_symbol)
    if name is None or chain_id not in chains_for(name):
        return None
    market = await CompoundV3.from_chain(w3, chain_id, token_symbol).get_market_data()
    # Comet's baseToken() has no symbol(); restore the real ticker so downstream
    # symbol-based dispatch (`comet_contract_for`) works on this market.
    base_token = dataclasses.replace(market.base_token, symbol=token_symbol)
    available = TokenAmount(
        token=base_token,
        amount=max(0, market.total_supply.amount - market.total_borrow.amount),
    )
    return YieldMarket(
        protocol="compound_v3",
        chain_id=chain_id,
        token=base_token,
        supply_apy=market.supply_apy,
        utilization=market.utilization,
        available_liquidity=available,
        market_id=f"compound_v3:{chain_id}:{token_symbol}",
    )


async def get_yield_markets(
    token_symbol: str,
    w3s: dict[int, AsyncWeb3],
    chains: list[int] | None = None,
    protocols: list[Protocol] | None = None,
) -> list[YieldMarket]:
    """Enumerate Aave V3 + Compound V3 supply markets for *token_symbol*,
    sorted by APY descending. Chains absent from ``w3s`` are skipped
    silently — the caller decides which RPCs to spend on. Inactive /
    frozen / paused Aave reserves are omitted."""
    selected: tuple[Protocol, ...] = tuple(protocols or ("aave_v3", "compound_v3"))
    candidates = chains if chains is not None else _candidate_chains(token_symbol)

    out: list[YieldMarket] = []
    for chain_id in candidates:
        w3 = w3s.get(chain_id)
        if w3 is None:
            continue
        try:
            token = get_token(token_symbol, chain_id)
        except KeyError:
            continue

        if "aave_v3" in selected:
            market = await _aave_market(w3, chain_id, token, token_symbol)
            if market is not None:
                out.append(market)
        if "compound_v3" in selected:
            market = await _compound_market(w3, chain_id, token_symbol)
            if market is not None:
                out.append(market)

    out.sort(key=lambda m: m.supply_apy, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Route builders — one per strategy, dispatched by build_yield_route
# ---------------------------------------------------------------------------


async def _withdraw_then_supply(
    user: Address,
    amount_in: TokenAmount,
    w3s: dict[int, AsyncWeb3],
    target_market: YieldMarket,
    source_market: YieldMarket | None,
    bridge: LucidBridge | None,
) -> YieldRoute:
    if source_market is None:
        raise ValueError("withdraw_then_supply requires source_market")
    _require_token(
        amount_in.token, source_market.token, "withdraw_then_supply: amount_in.token must match source_market.token"
    )

    same_chain = source_market.chain_id == target_market.chain_id
    if same_chain:
        _require_token(
            amount_in.token,
            target_market.token,
            "same-chain withdraw_then_supply: target_market.token must match amount_in.token",
        )
    elif bridge is None:
        raise ValueError(
            "cross-chain withdraw_then_supply requires a configured LucidBridge "
            f"(source={source_market.chain_id}, target={target_market.chain_id})"
        )
    elif bridge.src_chain_id != source_market.chain_id:
        raise ValueError(
            "bridge.src_chain_id must match source_market.chain_id "
            f"(bridge src={bridge.src_chain_id}, source={source_market.chain_id})"
        )
    elif bridge.dst_chain_id != target_market.chain_id:
        raise ValueError(
            "bridge.dst_chain_id must match target_market.chain_id "
            f"(bridge dst={bridge.dst_chain_id}, target={target_market.chain_id})"
        )

    chain_id = source_market.chain_id
    w3 = w3s[chain_id]
    steps: list[YieldStep] = [
        YieldStep("withdraw", chain_id, await _withdraw_tx(source_market, user, w3, amount_in)),
    ]
    if same_chain:
        steps += await _supply_steps(target_market, user, w3, amount_in)
    else:
        assert bridge is not None  # narrowed above
        steps += await _bridge_steps(bridge, user, amount_in, target_market.token)

    return YieldRoute(
        strategy="withdraw_then_supply",
        source_chain=chain_id,
        steps=tuple(steps),
        route_id=f"withdraw_then_supply:{source_market.market_id}->{target_market.market_id}",
        target_market=target_market,
        target_chain=None if same_chain else target_market.chain_id,
        pending=None if same_chain else PendingLeg(target_market.chain_id, target_market),
    )


async def _supply_then_bridge(
    user: Address,
    amount_in: TokenAmount,
    w3s: dict[int, AsyncWeb3],
    target_market: YieldMarket,
    target_chain: int | None,
) -> YieldRoute:
    if target_chain is None:
        raise ValueError(
            "supply_then_bridge requires target_chain — the destination the user "
            "intends to bridge into after the entry supply"
        )
    chain_id = amount_in.token.chain_id
    if target_market.chain_id != chain_id:
        raise ValueError(
            "supply_then_bridge expects target_market on amount_in.token's chain "
            f"(got market={target_market.chain_id}, token={chain_id})"
        )
    _require_token(
        amount_in.token, target_market.token, "supply_then_bridge: amount_in.token must match target_market.token"
    )

    steps = await _supply_steps(target_market, user, w3s[chain_id], amount_in)
    return YieldRoute(
        strategy="supply_then_bridge",
        source_chain=chain_id,
        steps=tuple(steps),
        route_id=f"supply_then_bridge:{target_market.market_id}->chain:{target_chain}",
        target_market=target_market,
        target_chain=target_chain,
    )


async def _bridge_then_supply(
    user: Address,
    amount_in: TokenAmount,
    target_market: YieldMarket,
    bridge: LucidBridge | None,
) -> YieldRoute:
    if bridge is None:
        raise ValueError("bridge_then_supply requires a configured LucidBridge")
    if amount_in.token.chain_id != bridge.src_chain_id:
        raise ValueError(
            "bridge_then_supply expects amount_in.token on the bridge's source chain "
            f"(token chain={amount_in.token.chain_id}, bridge src={bridge.src_chain_id})"
        )
    if target_market.chain_id != bridge.dst_chain_id:
        raise ValueError(
            "bridge_then_supply expects target_market on the bridge's destination chain "
            f"(market chain={target_market.chain_id}, bridge dst={bridge.dst_chain_id})"
        )

    steps = await _bridge_steps(bridge, user, amount_in, target_market.token)
    return YieldRoute(
        strategy="bridge_then_supply",
        source_chain=bridge.src_chain_id,
        steps=tuple(steps),
        route_id=f"bridge_then_supply:chain:{bridge.src_chain_id}->{target_market.market_id}",
        target_market=target_market,
        target_chain=bridge.dst_chain_id,
        pending=PendingLeg(target_market.chain_id, target_market),
    )


async def build_yield_route(
    strategy: Strategy,
    user: Address,
    amount_in: TokenAmount,
    w3s: dict[int, AsyncWeb3],
    *,
    target_market: YieldMarket,
    source_market: YieldMarket | None = None,
    bridge: LucidBridge | None = None,
    target_chain: int | None = None,
) -> YieldRoute:
    """Source-chain steps only; cross-chain follow-ups are deferred to a
    second invocation after the bridge settles.

    * ``withdraw_then_supply`` — rebalance; needs ``source_market``.
      Same-chain produces ``[withdraw, approve, supply]``. Cross-chain
      (pass ``bridge``) produces ``[withdraw, approve, bridge]``; the
      destination supply runs on a follow-up call.
    * ``supply_then_bridge`` — entry leg ``[approve, supply]`` on the
      token's chain. The exit (withdraw + bridge to ``target_chain``)
      is recorded on the route but not built.
    * ``bridge_then_supply`` — source-chain ``[approve, bridge]``; the
      destination supply runs on a follow-up call. Requires ``bridge``.

    A cross-chain route carries a :class:`PendingLeg` in ``route.pending``;
    feed it to :func:`build_followup_route` once the bridge settles.
    """
    if strategy == "withdraw_then_supply":
        return await _withdraw_then_supply(user, amount_in, w3s, target_market, source_market, bridge)
    if strategy == "supply_then_bridge":
        return await _supply_then_bridge(user, amount_in, w3s, target_market, target_chain)
    if strategy == "bridge_then_supply":
        return await _bridge_then_supply(user, amount_in, target_market, bridge)
    raise ValueError(f"unknown strategy: {strategy!r}")


async def build_followup_route(
    route: YieldRoute,
    user: Address,
    received: TokenAmount,
    w3s: dict[int, AsyncWeb3],
) -> YieldRoute:
    """Build the deferred destination leg of a cross-chain *route*.

    Call this once the bridge settles, with *received* set to the amount of
    the destination token that actually arrived — read it from the user's
    on-chain balance, since bridge fees mean it is not known in advance.
    Returns a same-chain :class:`YieldRoute` of ``[approve, supply]`` on the
    destination chain.

    Raises :class:`ValueError` if *route* has no :attr:`~YieldRoute.pending`
    leg (it was same-chain, or already a follow-up), or if *received* is not
    the pending market's token.
    """
    pending = route.pending
    if pending is None:
        raise ValueError(f"route {route.route_id!r} has no pending destination leg")
    _require_token(
        received.token,
        pending.market.token,
        "build_followup_route: received.token must match the pending market token",
    )
    steps = await _supply_steps(pending.market, user, w3s[pending.chain_id], received)
    return YieldRoute(
        strategy=route.strategy,
        source_chain=pending.chain_id,
        steps=tuple(steps),
        route_id=f"followup:{route.route_id}",
        target_market=pending.market,
    )

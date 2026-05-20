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
from pydefi.deployments import (
    COMET_CONTRACT_BY_SYMBOL,
    address_for,
    chains_for,
    comet_contract_for,
    get_token,
)
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
    description: str


@dataclass(frozen=True)
class YieldRoute:
    """Sequenced plan. Cross-chain follow-ups (destination supply after a
    bridge settles) are not pre-built — caller invokes the router again
    once the prior leg confirms."""

    strategy: Strategy
    source_chain: int
    steps: tuple[YieldStep, ...]
    route_id: str
    target_market: YieldMarket | None = None
    target_chain: int | None = None

    def build_transactions(self) -> list[dict[str, Any]]:
        return [s.tx for s in self.steps]


def _candidate_chains(token_symbol: str) -> list[int]:
    aave = set(chains_for("AAVE_V3_POOL"))
    comet_name = COMET_CONTRACT_BY_SYMBOL.get(token_symbol)
    comet = set(chains_for(comet_name)) if comet_name else set()
    return sorted(aave | comet)


def _market_spender(market: YieldMarket) -> Address:
    name = "AAVE_V3_POOL" if market.protocol == "aave_v3" else comet_contract_for(market.token.symbol)
    return address_for(name, market.chain_id)


def _same_token_identity(a: Token, b: Token) -> bool:
    # Identity = (chain, address). Symbol/decimals are not compared because
    # Compound markets carry a "BASE" placeholder symbol from baseToken().
    return a.chain_id == b.chain_id and a.address == b.address


def build_approve_tx(token: Token, spender: Address, amount: int, gas: int = _APPROVE_GAS) -> dict[str, Any]:
    return {
        "to": token.address,
        "data": "0x" + ERC20.fns.approve(bytes(spender), amount).data.hex(),
        "value": "0",
        "gas": str(gas),
    }


def _supply_tx(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> dict[str, Any]:
    if market.protocol == "aave_v3":
        aave = AaveV3(
            w3=w3,
            chain_id=market.chain_id,
            pool_address=address_for("AAVE_V3_POOL", market.chain_id),
            data_provider_address=address_for("AAVE_V3_DATA_PROVIDER", market.chain_id),
        )
        return aave.build_supply_tx(user, amount)
    comet = CompoundV3(
        w3=w3,
        chain_id=market.chain_id,
        comet_address=address_for(comet_contract_for(market.token.symbol), market.chain_id),
    )
    return comet.build_supply_tx(amount)


def _withdraw_tx(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> dict[str, Any]:
    if market.protocol == "aave_v3":
        aave = AaveV3(
            w3=w3,
            chain_id=market.chain_id,
            pool_address=address_for("AAVE_V3_POOL", market.chain_id),
            data_provider_address=address_for("AAVE_V3_DATA_PROVIDER", market.chain_id),
        )
        return aave.build_withdraw_tx(user, amount)
    comet = CompoundV3(
        w3=w3,
        chain_id=market.chain_id,
        comet_address=address_for(comet_contract_for(market.token.symbol), market.chain_id),
    )
    return comet.build_withdraw_tx(amount)


async def _aave_market(w3: AsyncWeb3, chain_id: int, token: Token, token_symbol: str) -> YieldMarket | None:
    if chain_id not in chains_for("AAVE_V3_POOL"):
        return None
    aave = AaveV3(
        w3=w3,
        chain_id=chain_id,
        pool_address=address_for("AAVE_V3_POOL", chain_id),
        data_provider_address=address_for("AAVE_V3_DATA_PROVIDER", chain_id),
    )
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
    name = COMET_CONTRACT_BY_SYMBOL.get(token_symbol)
    if name is None or chain_id not in chains_for(name):
        return None
    comet = CompoundV3(w3=w3, chain_id=chain_id, comet_address=address_for(name, chain_id))
    market = await comet.get_market_data()
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
    """
    if strategy == "withdraw_then_supply":
        if source_market is None:
            raise ValueError("withdraw_then_supply requires source_market")
        if not _same_token_identity(amount_in.token, source_market.token):
            raise ValueError(
                "withdraw_then_supply: amount_in.token must match source_market.token "
                f"(amount_in={amount_in.token}, source={source_market.token})"
            )
        same_chain = source_market.chain_id == target_market.chain_id
        if same_chain and not _same_token_identity(amount_in.token, target_market.token):
            raise ValueError(
                "same-chain withdraw_then_supply: target_market.token must match amount_in.token "
                f"(amount_in={amount_in.token}, target={target_market.token})"
            )
        if not same_chain:
            if bridge is None:
                raise ValueError(
                    "cross-chain withdraw_then_supply requires a configured LucidBridge "
                    f"(source={source_market.chain_id}, target={target_market.chain_id})"
                )
            if bridge.src_chain_id != source_market.chain_id:
                raise ValueError(
                    "bridge.src_chain_id must match source_market.chain_id "
                    f"(bridge src={bridge.src_chain_id}, source={source_market.chain_id})"
                )
            if bridge.dst_chain_id != target_market.chain_id:
                raise ValueError(
                    "bridge.dst_chain_id must match target_market.chain_id "
                    f"(bridge dst={bridge.dst_chain_id}, target={target_market.chain_id})"
                )

        chain_id = source_market.chain_id
        w3 = w3s[chain_id]
        withdraw_tx = _withdraw_tx(source_market, user, w3, amount_in)
        steps_list: list[YieldStep] = [
            YieldStep(
                "withdraw",
                chain_id,
                withdraw_tx,
                f"withdraw {amount_in.human_amount} {amount_in.token.symbol} from {source_market.market_id}",
            ),
        ]
        if same_chain:
            approve_tx = build_approve_tx(amount_in.token, _market_spender(target_market), amount_in.amount)
            supply_tx = _supply_tx(target_market, user, w3, amount_in)
            steps_list.append(
                YieldStep(
                    "approve",
                    chain_id,
                    approve_tx,
                    f"approve {amount_in.human_amount} {amount_in.token.symbol} to {target_market.market_id}",
                )
            )
            steps_list.append(
                YieldStep(
                    "supply",
                    chain_id,
                    supply_tx,
                    f"supply {amount_in.human_amount} {amount_in.token.symbol} into {target_market.market_id}",
                )
            )
        else:
            assert bridge is not None  # narrowed above
            approve_tx = build_approve_tx(amount_in.token, bridge.controller_address, amount_in.amount)
            bridge_tx = await bridge.build_bridge_tx(
                amount_in.token,
                target_market.token,
                amount_in,
                user,
            )
            steps_list.append(
                YieldStep(
                    "approve",
                    chain_id,
                    approve_tx,
                    f"approve {amount_in.human_amount} {amount_in.token.symbol} to Lucid controller",
                )
            )
            steps_list.append(
                YieldStep(
                    "bridge",
                    chain_id,
                    bridge_tx,
                    f"bridge {amount_in.human_amount} {amount_in.token.symbol} to chain {target_market.chain_id}",
                )
            )
        return YieldRoute(
            strategy="withdraw_then_supply",
            source_chain=chain_id,
            steps=tuple(steps_list),
            route_id=f"withdraw_then_supply:{source_market.market_id}->{target_market.market_id}",
            target_market=target_market,
            target_chain=target_market.chain_id if not same_chain else None,
        )

    if strategy == "supply_then_bridge":
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
        if not _same_token_identity(amount_in.token, target_market.token):
            raise ValueError(
                "supply_then_bridge: amount_in.token must match target_market.token "
                f"(amount_in={amount_in.token}, target={target_market.token})"
            )
        w3 = w3s[chain_id]
        approve_tx = build_approve_tx(amount_in.token, _market_spender(target_market), amount_in.amount)
        supply_tx = _supply_tx(target_market, user, w3, amount_in)
        steps = (
            YieldStep(
                "approve",
                chain_id,
                approve_tx,
                f"approve {amount_in.human_amount} {amount_in.token.symbol} to {target_market.market_id}",
            ),
            YieldStep(
                "supply",
                chain_id,
                supply_tx,
                f"supply {amount_in.human_amount} {amount_in.token.symbol} into {target_market.market_id}",
            ),
        )
        return YieldRoute(
            strategy="supply_then_bridge",
            source_chain=chain_id,
            steps=steps,
            route_id=f"supply_then_bridge:{target_market.market_id}->chain:{target_chain}",
            target_market=target_market,
            target_chain=target_chain,
        )

    if strategy == "bridge_then_supply":
        if bridge is None:
            raise ValueError("bridge_then_supply requires a configured LucidBridge")
        source_chain = bridge.src_chain_id
        if amount_in.token.chain_id != source_chain:
            raise ValueError(
                "bridge_then_supply expects amount_in.token on the bridge's source chain "
                f"(token chain={amount_in.token.chain_id}, bridge src={source_chain})"
            )
        if target_market.chain_id != bridge.dst_chain_id:
            raise ValueError(
                "bridge_then_supply expects target_market on the bridge's destination chain "
                f"(market chain={target_market.chain_id}, bridge dst={bridge.dst_chain_id})"
            )
        approve_tx = build_approve_tx(amount_in.token, bridge.controller_address, amount_in.amount)
        bridge_tx = await bridge.build_bridge_tx(
            amount_in.token,
            target_market.token,
            amount_in,
            user,
        )
        steps = (
            YieldStep(
                "approve",
                source_chain,
                approve_tx,
                f"approve {amount_in.human_amount} {amount_in.token.symbol} to Lucid controller",
            ),
            YieldStep(
                "bridge",
                source_chain,
                bridge_tx,
                f"bridge {amount_in.human_amount} {amount_in.token.symbol} to chain {bridge.dst_chain_id}",
            ),
        )
        return YieldRoute(
            strategy="bridge_then_supply",
            source_chain=source_chain,
            steps=steps,
            route_id=f"bridge_then_supply:chain:{source_chain}->{target_market.market_id}",
            target_market=target_market,
            target_chain=bridge.dst_chain_id,
        )

    raise ValueError(f"unknown strategy: {strategy!r}")

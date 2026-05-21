"""Enumerate yield markets and build sequenced execution plans
(``withdraw_then_supply``, ``supply_then_bridge``, ``bridge_then_supply``).
Caller supplies a per-chain ``AsyncWeb3`` map and broadcasts the steps."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import aiohttp
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3

from pydefi._utils import decode_address
from pydefi.bridge.lucid import LucidBridge
from pydefi.deployments import chains_for, comet_contract_for, get_token
from pydefi.lending.aave_v3 import AaveV3
from pydefi.lending.aave_v4 import AaveV4
from pydefi.lending.compound_v3 import CompoundV3
from pydefi.lending.morpho import MarketParams, MorphoBlue
from pydefi.types import Address, ChainId, Token, TokenAmount

logger = logging.getLogger(__name__)

Protocol = Literal["aave_v3", "compound_v3", "morpho", "aave_v4"]
Strategy = Literal["withdraw_then_supply", "supply_then_bridge", "bridge_then_supply"]
StepKind = Literal["approve", "supply", "withdraw", "bridge"]

# Generous; fits quirky tokens whose approve consumes more than the EIP-20 minimum.
_APPROVE_GAS = 100_000

# Aave V4 is Ethereum-only; yield markets are surfaced from the MAIN_SPOKE —
# the broad-asset Spoke that carries WETH / USDC / WBTC / etc.
_AAVE_V4_YIELD_SPOKE = "MAIN_SPOKE"

# Morpho markets are isolated (one collateral + one loan token each) and not
# keyed by token symbol, so discovery goes through the Morpho indexer API
# rather than an on-chain registry read.
_MORPHO_API_URL = "https://blue-api.morpho.org/graphql"
_MORPHO_API_TIMEOUT = 15.0
# The indexer ranks markets by total supply across every asset, so one page
# would drop a less-liquid loan token's markets before they are seen —
# _fetch_morpho_markets pages through the whole whitelisted set instead.
# _MORPHO_MAX_PAGES is a safety bound far above the live count (~620).
_MORPHO_PAGE_SIZE = 200
_MORPHO_MAX_PAGES = 25
_MORPHO_MARKETS_QUERY = """
query Markets($chainIds: [Int!], $first: Int!, $skip: Int!) {
  markets(
    first: $first
    skip: $skip
    orderBy: SupplyAssetsUsd
    orderDirection: Desc
    where: { chainId_in: $chainIds, whitelisted: true }
  ) {
    items {
      uniqueKey
      morphoBlue { chain { id } }
      loanAsset { symbol address decimals }
      state { supplyApy utilization liquidityAssets }
    }
  }
}
"""


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
    aave = set(chains_for("AAVE_V3_ADDRESSES_PROVIDER"))
    try:
        comet = set(chains_for(comet_contract_for(token_symbol)))
    except (ValueError, KeyError):
        # Compound III has no market for this token — Aave / Morpho may still.
        comet = set()
    morpho = set(chains_for("MORPHO_BLUE"))
    return sorted(aave | comet | morpho)


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


def morpho_id(market: YieldMarket) -> bytes:
    """Extract the bytes32 Morpho market Id from a ``morpho:<chain>:0x..`` id."""
    parts = market.market_id.split(":")
    if len(parts) != 3 or parts[0] != "morpho":
        raise ValueError(f"not a Morpho market_id: {market.market_id!r}")
    return bytes.fromhex(parts[2].removeprefix("0x"))


async def morpho_params(morpho: MorphoBlue, market: YieldMarket) -> MarketParams:
    """Resolve a Morpho market's immutable ``MarketParams`` from its Id. The
    loan token is already known; the collateral is read on-chain."""
    return await morpho.get_market_params(morpho_id(market), tokens={market.token.address: market.token})


def aave_v4_ref(market: YieldMarket) -> tuple[str, int]:
    """Parse an ``aave_v4:<chain>:<spoke>:<reserveId>`` market_id into the
    Spoke name and reserve id."""
    parts = market.market_id.split(":")
    if len(parts) != 4 or parts[0] != "aave_v4":
        raise ValueError(f"not an Aave V4 market_id: {market.market_id!r}")
    return parts[2], int(parts[3])


async def _withdraw_tx(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> dict[str, Any]:
    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        return aave.build_withdraw_tx(user, amount)
    if market.protocol == "compound_v3":
        return CompoundV3.from_chain(w3, market.chain_id, market.token.symbol).build_withdraw_tx(amount)
    if market.protocol == "aave_v4":
        spoke_name, reserve_id = aave_v4_ref(market)
        return AaveV4.from_chain(w3, market.chain_id, spoke_name).build_withdraw_tx(reserve_id, amount, user)
    if market.protocol == "morpho":
        morpho = MorphoBlue.from_chain(w3, market.chain_id)
        params = await morpho_params(morpho, market)
        return morpho.build_withdraw_tx(params, assets=amount, on_behalf_of=user, receiver=user)
    raise ValueError(f"unknown protocol: {market.protocol!r}")


async def _supply_steps(market: YieldMarket, user: Address, w3: AsyncWeb3, amount: TokenAmount) -> list[YieldStep]:
    """``[approve, supply]`` — approve the market's spender, then supply.

    The ERC-20 ``approve`` spender is the contract that pulls the funds —
    the Aave V3 ``Pool``, the Compound III ``Comet``, the Morpho Blue
    singleton, or the Aave V4 ``Spoke``. Each comes off the constructed
    client, so the approval always targets the exact contract the supply
    call hits."""
    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        spender, supply_tx = aave.pool_address, aave.build_supply_tx(user, amount)
    elif market.protocol == "compound_v3":
        comet = CompoundV3.from_chain(w3, market.chain_id, market.token.symbol)
        spender, supply_tx = comet.comet_address, comet.build_supply_tx(amount)
    elif market.protocol == "aave_v4":
        spoke_name, reserve_id = aave_v4_ref(market)
        v4 = AaveV4.from_chain(w3, market.chain_id, spoke_name)
        spender, supply_tx = v4.spoke_address, v4.build_supply_tx(reserve_id, amount, user)
    elif market.protocol == "morpho":
        morpho = MorphoBlue.from_chain(w3, market.chain_id)
        params = await morpho_params(morpho, market)
        spender = morpho.morpho_address
        supply_tx = morpho.build_supply_tx(params, assets=amount, on_behalf_of=user)
    else:
        raise ValueError(f"unknown protocol: {market.protocol!r}")
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
    try:
        name = comet_contract_for(token_symbol)
    except ValueError:
        return None  # Compound III has no Comet market for this token.
    if chain_id not in chains_for(name):
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


async def _aave_v4_market(w3: AsyncWeb3, chain_id: int, token: Token) -> YieldMarket | None:
    """The Aave V4 ``MAIN_SPOKE`` reserve for *token*, if one is listed.

    V4 is Ethereum-only and reserves are keyed by index, not asset address,
    so the matching reserve is found by scanning the Spoke's reserve list.
    """
    if chain_id != ChainId.ETHEREUM:
        return None
    spoke = AaveV4.from_chain(w3, chain_id, _AAVE_V4_YIELD_SPOKE)
    count = await spoke.get_reserve_count()
    reserves = await asyncio.gather(*(spoke.get_reserve(rid) for rid in range(count)))
    reserve_id = next((r.reserve_id for r in reserves if r.underlying.address == token.address), None)
    if reserve_id is None:
        return None
    data = await spoke.get_reserve_data(reserve_id, token)
    # A paused / frozen reserve cannot accept supplies.
    if data.is_paused or data.is_frozen:
        return None
    available = max(0, data.supplied.amount - data.total_debt.amount)
    return YieldMarket(
        protocol="aave_v4",
        chain_id=chain_id,
        token=token,
        supply_apy=data.supply_apy,
        utilization=data.utilization,
        available_liquidity=TokenAmount(token=token, amount=available),
        market_id=f"aave_v4:{chain_id}:{_AAVE_V4_YIELD_SPOKE}:{reserve_id}",
    )


async def _fetch_morpho_page(session: aiohttp.ClientSession, chain_ids: list[int], skip: int) -> list[dict[str, Any]]:
    """Fetch one ``_MORPHO_PAGE_SIZE`` page of the whitelisted-markets query."""
    payload = {
        "query": _MORPHO_MARKETS_QUERY,
        "variables": {"chainIds": chain_ids, "first": _MORPHO_PAGE_SIZE, "skip": skip},
    }
    async with session.post(_MORPHO_API_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    items = (((data or {}).get("data") or {}).get("markets") or {}).get("items") or []
    return [m for m in items if isinstance(m, dict)]


async def _fetch_morpho_markets(chain_ids: list[int]) -> list[dict[str, Any]]:
    """Query the Morpho indexer for every whitelisted market on *chain_ids*,
    deepest-supply first.

    The indexer ranks markets by supply across all assets, so the result is
    paged through in full — a single page would silently drop a less-liquid
    loan token's markets. Returns ``[]`` on any failure so a Morpho API
    outage degrades the markets list gracefully instead of breaking the
    whole response.
    """
    if not chain_ids:
        return []
    out: list[dict[str, Any]] = []
    try:
        timeout = aiohttp.ClientTimeout(total=_MORPHO_API_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for page in range(_MORPHO_MAX_PAGES):
                items = await _fetch_morpho_page(session, chain_ids, page * _MORPHO_PAGE_SIZE)
                out.extend(items)
                if len(items) < _MORPHO_PAGE_SIZE:
                    break
    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
        logger.warning("Morpho market query failed: %s", exc)
        return []
    return out


async def _morpho_markets(token_symbol: str, chain_ids: list[int]) -> list[YieldMarket]:
    """Every Morpho market on *chain_ids* whose loan asset is *token_symbol*,
    deepest-supply first (bounded only by the indexer query's page size).

    Each market's ``market_id`` is ``morpho:<chain_id>:<bytes32 Id>`` — the
    Id is what :func:`morpho_params` later resolves back into ``MarketParams``.
    """
    wanted = token_symbol.upper()
    chain_set = set(chain_ids)
    raw = await _fetch_morpho_markets(sorted(chain_set))
    out: list[YieldMarket] = []
    for m in raw:  # the API already orders by supply, deepest first
        loan = m.get("loanAsset") or {}
        if (loan.get("symbol") or "").upper() != wanted:
            continue
        chain = (m.get("morphoBlue") or {}).get("chain") or {}
        try:
            chain_id = int(chain.get("id"))
        except (TypeError, ValueError):
            continue
        if chain_id not in chain_set:
            continue
        unique_key = m.get("uniqueKey")
        loan_address = loan.get("address")
        state = m.get("state") or {}
        if not unique_key or not loan_address or state.get("supplyApy") is None:
            continue
        token = Token(
            chain_id=chain_id,
            address=decode_address(str(loan_address), chain_id),
            symbol=str(loan.get("symbol") or token_symbol),
            decimals=int(loan.get("decimals") or 18),
        )
        out.append(
            YieldMarket(
                protocol="morpho",
                chain_id=chain_id,
                token=token,
                supply_apy=Decimal(str(state.get("supplyApy") or 0)),
                utilization=Decimal(str(state.get("utilization") or 0)),
                available_liquidity=TokenAmount(token, int(state.get("liquidityAssets") or 0)),
                market_id=f"morpho:{chain_id}:{unique_key}",
            )
        )
    return out


async def get_yield_markets(
    token_symbol: str,
    w3s: dict[int, AsyncWeb3],
    chains: list[int] | None = None,
    protocols: list[Protocol] | None = None,
) -> list[YieldMarket]:
    """Enumerate Aave V3 + Compound V3 + Morpho + Aave V4 supply markets for
    *token_symbol*, sorted by APY descending. Chains absent from ``w3s`` are
    skipped silently — the caller decides which RPCs to spend on. Inactive /
    frozen / paused Aave reserves are omitted; Morpho markets are discovered
    via the Morpho indexer API."""
    selected: tuple[Protocol, ...] = tuple(protocols or ("aave_v3", "compound_v3", "morpho", "aave_v4"))
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
        if "aave_v4" in selected:
            market = await _aave_v4_market(w3, chain_id, token)
            if market is not None:
                out.append(market)

    # Morpho discovery is a single indexer query covering every candidate
    # chain the caller has an RPC for, so it runs once outside the loop.
    if "morpho" in selected:
        morpho_chains = sorted(c for c in candidates if c in w3s)
        out.extend(await _morpho_markets(token_symbol, morpho_chains))

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
    """
    if strategy == "withdraw_then_supply":
        return await _withdraw_then_supply(user, amount_in, w3s, target_market, source_market, bridge)
    if strategy == "supply_then_bridge":
        return await _supply_then_bridge(user, amount_in, w3s, target_market, target_chain)
    if strategy == "bridge_then_supply":
        return await _bridge_then_supply(user, amount_in, target_market, bridge)
    raise ValueError(f"unknown strategy: {strategy!r}")

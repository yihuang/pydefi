"""Unit tests for pydefi.yields (no live node required)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from web3 import AsyncWeb3, Web3

from pydefi.bridge import CCIP, CCTP, Bridge, LucidBridge
from pydefi.deployments import get_token
from pydefi.exceptions import BridgeError
from pydefi.lending.aave_v3 import ReserveData
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.yields import (
    PendingLeg,
    Position,
    YieldMarket,
    YieldRoute,
    build_approve_tx,
    build_compose_supply_program,
    build_compose_supply_route,
    build_followup_route,
    build_yield_route,
    expected_apy_gain,
    find_best_rebalance,
    get_positions,
    get_yield_markets,
    rebalance_tick,
    wait_for_bridge_settlement,
)
from pydefi.yields.router import (
    Protocol,
    RouteStrategy,
    YieldStep,
    _morpho_markets,
    _supply_steps,
    aave_v4_ref,
    morpho_id,
    sign_route,
)

# ---------------------------------------------------------------------------
# Tokens / addresses / canned tx dicts (shared across every test)
# ---------------------------------------------------------------------------

USDC_BASE = Token(chain_id=ChainId.BASE, address=Address("0x" + "11" * 20), symbol="USDC", decimals=6)
USDC_OPT = Token(chain_id=ChainId.OPTIMISM, address=Address("0x" + "22" * 20), symbol="USDC", decimals=6)
USDC_MANTRA = Token(chain_id=ChainId.MANTRA, address=Address("0x" + "EE" * 20), symbol="USDC", decimals=6)

_USER = Address("0x" + "AA" * 20)
_STUB_WITHDRAW = {"to": Address("0x" + "11" * 20), "data": "0xdead", "value": "0", "gas": "100000"}
_STUB_APPROVE = {"to": Address("0x" + "33" * 20), "data": "0xface", "value": "0", "gas": "100000"}
_STUB_SUPPLY = {"to": Address("0x" + "22" * 20), "data": "0xbeef", "value": "0", "gas": "100000"}

# Canned [approve, supply] pair returned by a patched router._supply_steps.
_STUB_SUPPLY_STEPS = [
    YieldStep("approve", ChainId.BASE, _STUB_APPROVE),
    YieldStep("supply", ChainId.BASE, _STUB_SUPPLY),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(protocol: Protocol, chain_id: int, token: Token, apy: str = "0.04") -> YieldMarket:
    return YieldMarket(
        protocol=protocol,
        chain_id=chain_id,
        token=token,
        supply_apy=Decimal(apy),
        utilization=Decimal("0.5"),
        available_liquidity=TokenAmount(token=token, amount=10**12),
        market_id=f"{protocol}:{chain_id}:{token.symbol}",
    )


def _fake_bridge(src_chain: int, dst_chain: int, spender: Address = Address("0x" + "CC" * 20)) -> object:
    """Stand-in for a Bridge — exposes only the attributes build_yield_route reads."""
    bridge = AsyncMock()
    bridge.src_chain_id = src_chain
    bridge.dst_chain_id = dst_chain
    bridge.spender = spender
    bridge.protocol_name = "FakeBridge"
    bridge.build_bridge_tx = AsyncMock(return_value={"to": spender, "data": "0xfeed", "value": "1000", "gas": "500000"})
    return bridge


def _stub_get_token(symbol: str, chain_id: int) -> Token:
    if chain_id == ChainId.BASE:
        return USDC_BASE
    if chain_id == ChainId.OPTIMISM:
        return USDC_OPT
    raise KeyError(f"Token {symbol!r} has no deployment on chain {chain_id}")


def _paused_reserve(token: Token) -> ReserveData:
    """Minimal ReserveData with the paused flag set — every other field is filler."""
    return ReserveData(
        token=token,
        supply_apy=Decimal("0.05"),
        variable_borrow_apy=Decimal("0.06"),
        utilization=Decimal("0.5"),
        available_liquidity=TokenAmount(token=token, amount=10**12),
        total_debt=TokenAmount(token=token, amount=10**11),
        liquidity_index=10**27,
        variable_borrow_index=10**27,
        a_token_address=Address("0x" + "aa" * 20),
        variable_debt_token_address=Address("0x" + "bb" * 20),
        ltv=7500,
        liquidation_threshold=8000,
        liquidation_bonus=10500,
        reserve_factor=1000,
        is_active=True,
        is_frozen=False,
        is_paused=True,
        borrowing_enabled=True,
        usage_as_collateral_enabled=True,
    )


# ---------------------------------------------------------------------------
# get_yield_markets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_yield_markets_sorts_by_apy_descending():
    async def fake_aave(w3, chain_id, token, symbol):
        return _market("aave_v3", chain_id, token, apy="0.042") if chain_id == ChainId.BASE else None

    async def fake_comet(w3, chain_id, symbol):
        return _market("compound_v3", chain_id, USDC_OPT, apy="0.055") if chain_id == ChainId.OPTIMISM else None

    with (
        patch("pydefi.yields.router.get_token", side_effect=_stub_get_token),
        patch("pydefi.yields.router._aave_market", new=fake_aave),
        patch("pydefi.yields.router._compound_market", new=fake_comet),
        patch("pydefi.yields.router._morpho_markets", new=AsyncMock(return_value=[])),
    ):
        out = await get_yield_markets(
            "USDC", w3s={ChainId.BASE: object(), ChainId.OPTIMISM: object()}, chains=[ChainId.BASE, ChainId.OPTIMISM]
        )

    assert [m.protocol for m in out] == ["compound_v3", "aave_v3"]
    assert out[0].supply_apy > out[1].supply_apy


@pytest.mark.asyncio
async def test_get_yield_markets_skips_paused_aave_reserve():
    with (
        patch("pydefi.yields.router.get_token", side_effect=_stub_get_token),
        patch("pydefi.yields.router.AaveV3") as Aave,
        patch("pydefi.yields.router._compound_market", new=AsyncMock(return_value=None)),
    ):
        aave_client = AsyncMock()
        aave_client.get_reserve_data = AsyncMock(return_value=_paused_reserve(USDC_BASE))
        Aave.from_chain = AsyncMock(return_value=aave_client)
        out = await get_yield_markets(
            "USDC", w3s={ChainId.BASE: object()}, chains=[ChainId.BASE], protocols=["aave_v3"]
        )
    assert out == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "w3s,chains",
    [
        ({}, [ChainId.BASE, ChainId.OPTIMISM]),  # no w3 supplied → skip both
        ({ChainId.LINEA: object()}, [ChainId.LINEA]),  # token not in registry → skip
    ],
    ids=["no_w3", "no_token"],
)
async def test_get_yield_markets_skips_unscannable_chains(w3s, chains):
    aave = AsyncMock(return_value=None)
    comet = AsyncMock(return_value=None)
    with (
        patch("pydefi.yields.router._aave_market", new=aave),
        patch("pydefi.yields.router._compound_market", new=comet),
        patch("pydefi.yields.router._morpho_markets", new=AsyncMock(return_value=[])),
    ):
        out = await get_yield_markets("USDC", w3s=w3s, chains=chains)
    assert out == []
    assert aave.await_count == 0
    assert comet.await_count == 0


@pytest.mark.asyncio
async def test_get_yield_markets_protocols_filter():
    aave = AsyncMock(return_value=None)
    comet = AsyncMock(return_value=None)
    with (
        patch("pydefi.yields.router.get_token", side_effect=_stub_get_token),
        patch("pydefi.yields.router._aave_market", new=aave),
        patch("pydefi.yields.router._compound_market", new=comet),
    ):
        await get_yield_markets("USDC", w3s={ChainId.BASE: object()}, chains=[ChainId.BASE], protocols=["aave_v3"])
    assert aave.await_count == 1
    assert comet.await_count == 0


# ---------------------------------------------------------------------------
# Morpho discovery
# ---------------------------------------------------------------------------

_USDC_BASE_ADDR = get_token("USDC", ChainId.BASE).address.to_0x_hex()


def _morpho_api_item(unique_key: str, chain_id: int, symbol: str, address: str, decimals: int, apy: float) -> dict:
    """One raw item shaped like the Morpho indexer's ``markets.items[]``."""
    return {
        "uniqueKey": unique_key,
        "morphoBlue": {"chain": {"id": chain_id}},
        "loanAsset": {"symbol": symbol, "address": address, "decimals": decimals},
        "state": {"supplyApy": apy, "utilization": 0.8, "liquidityAssets": "1000000"},
    }


def testmorpho_id_round_trips_market_id():
    market = dataclasses.replace(_market("morpho", ChainId.BASE, USDC_BASE), market_id="morpho:8453:0x" + "ab" * 32)
    assert morpho_id(market) == bytes.fromhex("ab" * 32)


def testmorpho_id_rejects_non_morpho_market():
    with pytest.raises(ValueError):
        morpho_id(_market("aave_v3", ChainId.BASE, USDC_BASE))


@pytest.mark.asyncio
async def test_morpho_markets_filters_by_loan_symbol():
    # The indexer returns markets for every loan asset; only USDC-loan ones
    # are surfaced, and the indexer's deepest-supply-first order is preserved.
    canned = [
        _morpho_api_item("0x" + "a1" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.05),
        _morpho_api_item("0x" + "a2" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.04),
        _morpho_api_item("0x" + "b1" * 32, 8453, "WETH", "0x" + "ee" * 20, 18, 0.09),
        _morpho_api_item("0x" + "a3" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.03),
    ]
    with patch("pydefi.yields.router._fetch_morpho_markets", new=AsyncMock(return_value=canned)):
        out = await _morpho_markets("USDC", [ChainId.BASE])

    assert [m.market_id for m in out] == [
        "morpho:8453:0x" + "a1" * 32,
        "morpho:8453:0x" + "a2" * 32,
        "morpho:8453:0x" + "a3" * 32,
    ]
    assert all(m.protocol == "morpho" and m.token.symbol == "USDC" and m.token.decimals == 6 for m in out)
    assert out[0].supply_apy == Decimal("0.05")
    assert out[0].available_liquidity.amount == 1_000_000


@pytest.mark.asyncio
async def test_morpho_markets_skips_malformed_items():
    # A malformed indexer item — missing loanAsset.address — must be skipped,
    # not raised on, so one bad row never breaks the whole markets list.
    good = _morpho_api_item("0x" + "a1" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.05)
    no_address = _morpho_api_item("0x" + "a2" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.04)
    del no_address["loanAsset"]["address"]
    with patch("pydefi.yields.router._fetch_morpho_markets", new=AsyncMock(return_value=[no_address, good])):
        out = await _morpho_markets("USDC", [ChainId.BASE])

    assert [m.market_id for m in out] == ["morpho:8453:0x" + "a1" * 32]


@pytest.mark.asyncio
async def test_fetch_morpho_markets_paginates_until_short_page():
    # The indexer ranks across all assets, so discovery pages through every
    # market — full pages accumulate and the walk stops on the first short one.
    from pydefi.yields.router import _MORPHO_PAGE_SIZE, _fetch_morpho_markets

    full = [{"uniqueKey": "k"}] * _MORPHO_PAGE_SIZE
    pages = [full, full, [{"uniqueKey": "tail"}]]
    with patch("pydefi.yields.router._fetch_morpho_page", new=AsyncMock(side_effect=pages)) as page:
        out = await _fetch_morpho_markets([ChainId.BASE])

    assert len(out) == 2 * _MORPHO_PAGE_SIZE + 1
    assert page.await_count == 3  # stopped after the short final page


@pytest.mark.asyncio
async def test_get_yield_markets_includes_morpho():
    canned = [_morpho_api_item("0x" + "c1" * 32, 8453, "USDC", _USDC_BASE_ADDR, 6, 0.06)]
    with (
        patch("pydefi.yields.router._aave_market", new=AsyncMock(return_value=None)),
        patch("pydefi.yields.router._compound_market", new=AsyncMock(return_value=None)),
        patch("pydefi.yields.router._fetch_morpho_markets", new=AsyncMock(return_value=canned)),
    ):
        out = await get_yield_markets("USDC", w3s={ChainId.BASE: object()}, chains=[ChainId.BASE])

    assert [m.protocol for m in out] == ["morpho"]
    assert out[0].market_id == "morpho:8453:0x" + "c1" * 32


# ---------------------------------------------------------------------------
# Aave V4 discovery
# ---------------------------------------------------------------------------


def testaave_v4_ref_parses_market_id():
    market = dataclasses.replace(_market("aave_v4", ChainId.ETHEREUM, USDC_BASE), market_id="aave_v4:1:MAIN_SPOKE:7")
    assert aave_v4_ref(market) == ("MAIN_SPOKE", 7)


def testaave_v4_ref_rejects_non_v4_market():
    with pytest.raises(ValueError):
        aave_v4_ref(_market("aave_v3", ChainId.ETHEREUM, USDC_BASE))


@pytest.mark.asyncio
async def test_get_yield_markets_skips_aave_v4_off_ethereum():
    """Aave V4 is Ethereum-only — _aave_v4_market returns None elsewhere
    without touching the RPC, so non-Ethereum scans stay V4-free."""
    with (
        patch("pydefi.yields.router._aave_market", new=AsyncMock(return_value=None)),
        patch("pydefi.yields.router._compound_market", new=AsyncMock(return_value=None)),
        patch("pydefi.yields.router._morpho_markets", new=AsyncMock(return_value=[])),
    ):
        out = await get_yield_markets("USDC", w3s={ChainId.BASE: object()}, chains=[ChainId.BASE])
    assert out == []


def test_yield_market_is_frozen_dataclass():
    m = _market("aave_v3", ChainId.BASE, USDC_BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.supply_apy = Decimal("0.05")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_approve_tx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compound_unsupported_symbol_raises_value_error():
    """Building supply steps for an unsupported Compound token must surface a
    descriptive ValueError listing what *is* supported — not a bare KeyError."""
    from pydefi.yields.router import _supply_steps

    weird = Token(chain_id=ChainId.BASE, address=Address("0x" + "44" * 20), symbol="WEIRD", decimals=18)
    bogus_market = _market("compound_v3", ChainId.BASE, weird)
    with pytest.raises(ValueError, match="Compound V3 has no market for token 'WEIRD'"):
        await _supply_steps(bogus_market, _USER, cast(AsyncWeb3, object()), TokenAmount(weird, 1))


def test_build_approve_tx_shape():
    tx = build_approve_tx(USDC_BASE, Address("0x" + "BB" * 20), 10**12)
    assert tx["to"] == USDC_BASE.address
    selector = Web3.keccak(text="approve(address,uint256)")[:4].hex()
    assert tx["data"][:10] == "0x" + selector
    assert tx["value"] == "0"
    assert int(tx["gas"]) > 0


# ---------------------------------------------------------------------------
# build_yield_route — withdraw_then_supply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_then_supply_same_chain():
    src = _market("aave_v3", ChainId.BASE, USDC_BASE)
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.05")
    amount = TokenAmount.from_human(USDC_BASE, "1000")
    with (
        patch("pydefi.yields.router._withdraw_tx", new=AsyncMock(return_value=_STUB_WITHDRAW)),
        patch("pydefi.yields.router._supply_steps", new=AsyncMock(return_value=_STUB_SUPPLY_STEPS)),
    ):
        route = await build_yield_route(
            "withdraw_then_supply",
            _USER,
            amount,
            w3s={ChainId.BASE: object()},
            source_market=src,
            target_market=dst,
        )

    assert route.strategy == "withdraw_then_supply"
    assert route.source_chain == ChainId.BASE
    assert [s.kind for s in route.steps] == ["withdraw", "approve", "supply"]
    assert route.pending is None  # same-chain — nothing deferred
    txs = route.build_transactions()
    assert txs[0]["data"] == _STUB_WITHDRAW["data"]
    assert txs[2]["data"] == _STUB_SUPPLY["data"]


@pytest.mark.asyncio
async def test_withdraw_then_supply_cross_chain_emits_bridge():
    src = _market("aave_v3", ChainId.MANTRA, USDC_MANTRA)
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    amount = TokenAmount(USDC_MANTRA, 1_000_000)
    with patch("pydefi.yields.router._withdraw_tx", new=AsyncMock(return_value=_STUB_WITHDRAW)):
        route = await build_yield_route(
            "withdraw_then_supply",
            _USER,
            amount,
            w3s={ChainId.MANTRA: object()},
            source_market=src,
            target_market=dst,
            bridge=bridge,
        )
    assert [s.kind for s in route.steps] == ["withdraw", "approve", "bridge"]
    assert route.source_chain == ChainId.MANTRA
    assert route.target_chain == ChainId.BASE
    assert route.pending == PendingLeg(ChainId.BASE, dst)  # deferred destination supply


# ---------------------------------------------------------------------------
# build_yield_route — supply_then_bridge / bridge_then_supply happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supply_then_bridge_entry_leg():
    market = _market("aave_v3", ChainId.BASE, USDC_BASE)
    amount = TokenAmount.from_human(USDC_BASE, "1000")
    with patch("pydefi.yields.router._supply_steps", new=AsyncMock(return_value=_STUB_SUPPLY_STEPS)):
        route = await build_yield_route(
            "supply_then_bridge",
            _USER,
            amount,
            w3s={ChainId.BASE: object()},
            target_market=market,
            target_chain=ChainId.KITE,
        )
    assert [s.kind for s in route.steps] == ["approve", "supply"]
    assert route.target_chain == ChainId.KITE
    assert all(s.kind != "bridge" for s in route.steps)
    # The exit is a withdraw+bridge, not a deferred supply — no pending leg.
    assert route.pending is None


@pytest.mark.asyncio
async def test_bridge_then_supply_emits_approve_and_bridge():
    controller = Address("0x" + "CC" * 20)
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE, controller)
    target = _market("aave_v3", ChainId.BASE, USDC_BASE)
    route = await build_yield_route(
        "bridge_then_supply",
        _USER,
        TokenAmount(USDC_MANTRA, 1_000_000),
        w3s={ChainId.MANTRA: object()},
        target_market=target,
        bridge=bridge,
    )
    assert [s.kind for s in route.steps] == ["approve", "bridge"]
    assert route.source_chain == ChainId.MANTRA
    assert route.target_chain == ChainId.BASE
    assert route.pending == PendingLeg(ChainId.BASE, target)  # deferred destination supply
    txs = route.build_transactions()
    assert txs[0]["to"] == USDC_MANTRA.address  # approve targets the source token
    assert txs[1]["to"] == controller  # bridge targets the controller


# ---------------------------------------------------------------------------
# build_followup_route — the deferred destination leg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_followup_route_builds_destination_supply():
    """The follow-up is [approve, supply] on the destination, built against
    the amount that actually arrived."""
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    route = YieldRoute(
        strategy="withdraw_then_supply",
        source_chain=ChainId.MANTRA,
        steps=(),
        route_id="withdraw_then_supply:src->dst",
        target_market=dst,
        target_chain=ChainId.BASE,
        pending=PendingLeg(ChainId.BASE, dst),
    )
    received = TokenAmount(USDC_BASE, 990_000)  # post-bridge balance, < amount sent
    with patch("pydefi.yields.router._supply_steps", new=AsyncMock(return_value=_STUB_SUPPLY_STEPS)) as supply:
        followup = await build_followup_route(route, _USER, received, w3s={ChainId.BASE: object()})

    assert [s.kind for s in followup.steps] == ["approve", "supply"]
    assert followup.source_chain == ChainId.BASE
    assert followup.target_market == dst
    assert followup.pending is None  # nothing left to defer
    assert followup.route_id == "followup:withdraw_then_supply:src->dst"
    # _supply_steps(market, user, w3, amount) — built against the received amount.
    assert supply.await_args.args[0] == dst
    assert supply.await_args.args[3] == received


@pytest.mark.asyncio
@pytest.mark.parametrize("knob", ["defivm", "delegate"])
async def test_build_followup_route_forwards_gasless_knobs(knob):
    """defivm / delegate pass through to _supply_steps so the destination leg
    can collapse to one gasless step, exactly like build_yield_route."""
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    route = YieldRoute(
        strategy="withdraw_then_supply",
        source_chain=ChainId.MANTRA,
        steps=(),
        route_id="withdraw_then_supply:src->dst",
        target_market=dst,
        target_chain=ChainId.BASE,
        pending=PendingLeg(ChainId.BASE, dst),
    )
    target = Address("0x" + "D7" * 20)
    gasless_step = [YieldStep("supply_with_permit2", ChainId.BASE, None, sign_request={"x": 1})]
    with patch("pydefi.yields.router._supply_steps", new=AsyncMock(return_value=gasless_step)) as supply:
        followup = await build_followup_route(
            route, _USER, TokenAmount(USDC_BASE, 990_000), w3s={ChainId.BASE: object()}, **{knob: target}
        )

    assert [s.kind for s in followup.steps] == ["supply_with_permit2"]
    expected = {"defivm": None, "delegate": None} | {knob: target}
    assert supply.await_args.kwargs == expected


@pytest.mark.asyncio
async def test_build_followup_route_rejects_route_without_pending():
    route = YieldRoute(
        strategy="withdraw_then_supply",
        source_chain=ChainId.BASE,
        steps=(),
        route_id="same-chain",
    )
    with pytest.raises(ValueError, match="no pending destination leg"):
        await build_followup_route(route, _USER, TokenAmount(USDC_BASE, 1), w3s={ChainId.BASE: object()})


@pytest.mark.asyncio
async def test_build_followup_route_rejects_token_mismatch():
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE)
    route = YieldRoute(
        strategy="withdraw_then_supply",
        source_chain=ChainId.MANTRA,
        steps=(),
        route_id="r",
        pending=PendingLeg(ChainId.BASE, dst),
    )
    dai = Token(chain_id=ChainId.BASE, address=Address("0x" + "DD" * 20), symbol="DAI", decimals=18)
    with pytest.raises(ValueError, match="received.token must match"):
        await build_followup_route(route, _USER, TokenAmount(dai, 1), w3s={ChainId.BASE: object()})


# ---------------------------------------------------------------------------
# Recovery story — a partial cross-chain route never strands funds
# ---------------------------------------------------------------------------
#
# The guarantee rests on one invariant: the bridge delivers exactly the token
# route.pending names. Skip build_followup_route and the user simply holds
# pending.market.token on pending.chain_id.


async def _cross_chain_route(strategy: RouteStrategy, bridge: object) -> YieldRoute:
    """Build a cross-chain route through *bridge* for either cross-chain strategy."""
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    amount = TokenAmount(USDC_MANTRA, 1_000_000)
    w3s = {ChainId.MANTRA: object()}
    if strategy == "withdraw_then_supply":
        src = _market("aave_v3", ChainId.MANTRA, USDC_MANTRA)
        with patch("pydefi.yields.router._withdraw_tx", new=AsyncMock(return_value=_STUB_WITHDRAW)):
            return await build_yield_route(
                strategy, _USER, amount, w3s=w3s, source_market=src, target_market=dst, bridge=bridge
            )
    return await build_yield_route(strategy, _USER, amount, w3s=w3s, target_market=dst, bridge=bridge)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["withdraw_then_supply", "bridge_then_supply"])
async def test_cross_chain_route_bridges_into_the_pending_token(strategy: RouteStrategy):
    """The bridge delivers exactly the token route.pending names, on the chain
    it names — so a skipped follow-up strands nothing."""
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    route = await _cross_chain_route(strategy, bridge)

    assert route.pending is not None
    # _bridge_steps calls build_bridge_tx(src_token, target_token, ...) — the
    # delivered token is positional arg 1.
    delivered_token = bridge.build_bridge_tx.await_args.args[1]
    assert delivered_token == route.pending.market.token == USDC_BASE
    assert route.pending.chain_id == delivered_token.chain_id == ChainId.BASE


@pytest.mark.asyncio
async def test_followup_route_consumes_the_token_the_user_holds():
    """build_followup_route accepts exactly the token a skipped cross-chain
    route leaves the user holding — the recovered funds stay actionable."""
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    route = await _cross_chain_route("withdraw_then_supply", bridge)
    assert route.pending is not None

    held = TokenAmount(route.pending.market.token, 990_000)
    with patch("pydefi.yields.router._supply_steps", new=AsyncMock(return_value=_STUB_SUPPLY_STEPS)):
        followup = await build_followup_route(route, _USER, held, w3s={ChainId.BASE: object()})
    assert [s.kind for s in followup.steps] == ["approve", "supply"]


# ---------------------------------------------------------------------------
# build_yield_route — bridge-agnostic (any Bridge, not just Lucid)
# ---------------------------------------------------------------------------


def _ccip_bridge_case() -> tuple[Bridge, Token, Token]:
    """A real CCIP bridge with its source / destination USDC tokens."""
    src = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "E7" * 20), symbol="USDC", decimals=6)
    dst = Token(chain_id=42161, address=Address("0x" + "A7" * 20), symbol="USDC", decimals=6)
    return CCIP(w3=None, src_chain_id=ChainId.ETHEREUM, dst_chain_id=42161), src, dst  # type: ignore[arg-type]


def _lucid_bridge_case() -> tuple[Bridge, Token, Token]:
    """A real LucidBridge with its source / destination USDC tokens."""
    src = Token(chain_id=ChainId.MANTRA, address=Address("0x" + "E8" * 20), symbol="USDC", decimals=6)
    dst = Token(chain_id=ChainId.BASE, address=Address("0x" + "A8" * 20), symbol="USDC", decimals=6)
    bridge = LucidBridge(
        w3=None,  # type: ignore[arg-type]
        src_chain_id=ChainId.MANTRA,
        dst_chain_id=ChainId.BASE,
        controller_address=Address("0x" + "C0" * 20),
        adapter_address=Address("0x" + "AD" * 20),
    )
    return bridge, src, dst


def _cctp_bridge_case() -> tuple[Bridge, Token, Token]:
    """A real CCTP bridge with its source / destination USDC tokens."""
    src = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "E9" * 20), symbol="USDC", decimals=6)
    dst = Token(chain_id=42161, address=Address("0x" + "A9" * 20), symbol="USDC", decimals=6)
    return CCTP(w3=None, src_chain_id=ChainId.ETHEREUM, dst_chain_id=42161), src, dst  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_case", [_ccip_bridge_case, _lucid_bridge_case, _cctp_bridge_case], ids=["ccip", "lucid", "cctp"]
)
async def test_build_yield_route_accepts_any_concrete_bridge(make_case):
    """build_yield_route works with every Bridge that exposes a spender —
    CCIP, Lucid, and CCTP — encoding that bridge's own spender (Router /
    controller / TokenMessenger) into the approve step, with no per-bridge
    coupling."""
    bridge, src_token, dst_token = make_case()
    bridge.build_bridge_tx = AsyncMock(  # type: ignore[method-assign]
        return_value={"to": "0xbridge", "data": "0xc0de", "value": "0", "gas": "500000"}
    )
    route = await build_yield_route(
        "bridge_then_supply",
        _USER,
        TokenAmount(src_token, 1_000_000),
        w3s={src_token.chain_id: object()},
        target_market=_market("aave_v3", dst_token.chain_id, dst_token),
        bridge=bridge,
    )
    assert [s.kind for s in route.steps] == ["approve", "bridge"]
    assert bridge.spender in Address(route.steps[0].tx["data"])


@pytest.mark.asyncio
async def test_cross_chain_route_rejects_bridge_without_spender():
    """A bridge that exposes no ERC-20 spender (the Bridge default) cannot
    carry a yield route — rejected before any transaction is built."""
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    bridge.spender = None
    bridge.protocol_name = "NativeOnly"
    with pytest.raises(ValueError, match="NativeOnly bridge exposes no ERC-20 spender"):
        await build_yield_route(
            "bridge_then_supply",
            _USER,
            TokenAmount(USDC_MANTRA, 1_000_000),
            w3s={ChainId.MANTRA: object()},
            target_market=_market("aave_v3", ChainId.BASE, USDC_BASE),
            bridge=bridge,
        )


@pytest.mark.asyncio
async def test_cross_chain_route_rejects_bridge_lacking_spender_attribute():
    """A bridge class that never defines spender (most native-asset bridges,
    e.g. GasZip) is rejected with the same ValueError — not an AttributeError."""
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    del bridge.spender  # mock attribute deletion → AttributeError on access
    bridge.protocol_name = "NativeOnly"
    with pytest.raises(ValueError, match="NativeOnly bridge exposes no ERC-20 spender"):
        await build_yield_route(
            "bridge_then_supply",
            _USER,
            TokenAmount(USDC_MANTRA, 1_000_000),
            w3s={ChainId.MANTRA: object()},
            target_market=_market("aave_v3", ChainId.BASE, USDC_BASE),
            bridge=bridge,
        )


# ---------------------------------------------------------------------------
# build_yield_route — validation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_then_supply_cross_chain_requires_bridge():
    with pytest.raises(ValueError, match="cross-chain.*requires a configured bridge"):
        await build_yield_route(
            "withdraw_then_supply",
            _USER,
            TokenAmount(USDC_BASE, 1),
            w3s={ChainId.BASE: object(), ChainId.OPTIMISM: object()},
            source_market=_market("aave_v3", ChainId.BASE, USDC_BASE),
            target_market=_market("compound_v3", ChainId.OPTIMISM, USDC_OPT),
        )


@pytest.mark.asyncio
async def test_withdraw_then_supply_validates_bridge_endpoints():
    """bridge.src_chain_id must match source_market.chain_id."""
    wrong_bridge = _fake_bridge(ChainId.KITE, ChainId.BASE)
    with pytest.raises(ValueError, match="bridge.src_chain_id must match"):
        await build_yield_route(
            "withdraw_then_supply",
            _USER,
            TokenAmount(USDC_MANTRA, 1),
            w3s={ChainId.MANTRA: object()},
            source_market=_market("aave_v3", ChainId.MANTRA, USDC_MANTRA),
            target_market=_market("compound_v3", ChainId.BASE, USDC_BASE),
            bridge=wrong_bridge,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bridge_src,bridge_dst,target_chain,err",
    [
        # amount_in.token lives on MANTRA but bridge sources from KITE.
        (ChainId.KITE, ChainId.AVALANCHE, ChainId.AVALANCHE, "amount_in.token on the bridge's source chain"),
        # target_market on OPTIMISM but bridge delivers to BASE.
        (ChainId.MANTRA, ChainId.BASE, ChainId.OPTIMISM, "target_market on the bridge's destination chain"),
    ],
    ids=["src_chain_mismatch", "dst_chain_mismatch"],
)
async def test_bridge_then_supply_validation_errors(bridge_src, bridge_dst, target_chain, err):
    bridge = _fake_bridge(bridge_src, bridge_dst)
    target = _market("aave_v3", target_chain, USDC_OPT if target_chain == ChainId.OPTIMISM else USDC_BASE)
    with pytest.raises(ValueError, match=err):
        await build_yield_route(
            "bridge_then_supply",
            _USER,
            TokenAmount(USDC_MANTRA, 1),
            w3s={bridge_src: object()},
            target_market=target,
            bridge=bridge,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy,kw,err",
    [
        ("withdraw_then_supply", {}, "requires source_market"),
        ("bridge_then_supply", {}, "requires a configured bridge"),
        ("supply_then_bridge", {}, "requires target_chain"),
        ("supply_then_borrow", {}, "unknown strategy"),
    ],
    ids=["missing_source_market", "missing_bridge", "missing_target_chain", "unknown_strategy"],
)
async def test_build_yield_route_validation_errors(strategy: str, kw: dict, err: str):
    target = _market("aave_v3", ChainId.BASE, USDC_BASE)
    with pytest.raises(ValueError, match=err):
        await build_yield_route(
            cast(RouteStrategy, strategy),  # "supply_then_borrow" is intentionally not in RouteStrategy
            _USER,
            TokenAmount.from_human(USDC_BASE, "1"),
            w3s={ChainId.BASE: cast(AsyncWeb3, object())},
            target_market=target,
            **kw,
        )


@pytest.mark.asyncio
async def test_build_yield_route_rejects_token_mismatch():
    """amount_in.token must match the relevant market token by (chain_id, address)
    — else the route would act on the wrong asset while metadata pointed elsewhere."""
    dai = Token(chain_id=ChainId.BASE, address=Address("0x" + "33" * 20), symbol="DAI", decimals=18)
    usdc_m = _market("aave_v3", ChainId.BASE, USDC_BASE)
    dai_m = _market("compound_v3", ChainId.BASE, dai)
    w3s: dict[int, AsyncWeb3] = {ChainId.BASE: cast(AsyncWeb3, object())}

    cases: list[tuple[RouteStrategy, Token, YieldMarket, YieldMarket, str]] = [
        ("withdraw_then_supply", dai, usdc_m, usdc_m, "amount_in.token must match source_market.token"),
        ("withdraw_then_supply", USDC_BASE, usdc_m, dai_m, "target_market.token must match amount_in.token"),
        ("supply_then_bridge", dai, usdc_m, usdc_m, "amount_in.token must match target_market.token"),
    ]
    for strategy, token, src, tgt, err in cases:
        with pytest.raises(ValueError, match=err):
            await build_yield_route(
                strategy,
                _USER,
                TokenAmount(token, 1),
                w3s=w3s,
                source_market=src,
                target_market=tgt,
                target_chain=ChainId.OPTIMISM,
            )


def _position(market: YieldMarket, amount: int) -> Position:
    return Position(market=market, balance=TokenAmount(token=market.token, amount=amount))


# ---------------------------------------------------------------------------
# get_positions (PositionTracker)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_positions_returns_only_nonzero_balances_sorted():
    aave_mkt = _market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04")
    comet_mkt = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.05")
    with (
        patch("pydefi.yields.tracker.get_yield_markets", new=AsyncMock(return_value=[aave_mkt, comet_mkt])),
        patch(
            "pydefi.yields.tracker._aave_balance",
            new=AsyncMock(return_value=TokenAmount(token=USDC_BASE, amount=200_000)),
        ),
        patch(
            "pydefi.yields.tracker._compound_balance",
            new=AsyncMock(return_value=TokenAmount(token=USDC_BASE, amount=500_000)),
        ),
    ):
        positions = await get_positions(_USER, "USDC", w3s={ChainId.BASE: object()})
    assert [p.market.protocol for p in positions] == ["compound_v3", "aave_v3"]
    assert [p.balance.amount for p in positions] == [500_000, 200_000]


@pytest.mark.asyncio
async def test_get_positions_filters_zero_balance():
    with (
        patch(
            "pydefi.yields.tracker.get_yield_markets",
            new=AsyncMock(return_value=[_market("aave_v3", ChainId.BASE, USDC_BASE)]),
        ),
        patch(
            "pydefi.yields.tracker._aave_balance", new=AsyncMock(return_value=TokenAmount(token=USDC_BASE, amount=0))
        ),
    ):
        positions = await get_positions(_USER, "USDC", w3s={ChainId.BASE: object()})
    assert positions == []


def test_position_is_frozen_dataclass():
    p = _position(_market("aave_v3", ChainId.BASE, USDC_BASE), 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.balance = TokenAmount(token=USDC_BASE, amount=20)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# wait_for_bridge_settlement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_bridge_settlement_returns_when_delta_reached():
    # First two polls stay at the start balance; third one shows the bridge landed.
    call_mock = AsyncMock(side_effect=[100, 100, 1_100])
    with (
        patch("pydefi.yields.tracker.ERC20") as ERC20_mock,
        patch("pydefi.yields.tracker.asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        ERC20_mock.fns.balanceOf.return_value.call = call_mock
        new_balance = await wait_for_bridge_settlement(
            w3=cast(AsyncWeb3, object()),  # ERC20.balanceOf is patched, real w3 never used
            token_address=Address("0x" + "11" * 20),
            recipient=_USER,
            starting_balance=100,
            min_delta=1_000,
            timeout_seconds=60.0,
            poll_interval_seconds=0.01,
        )
    assert new_balance == 1_100


@pytest.mark.asyncio
async def test_wait_for_bridge_settlement_raises_on_timeout():
    with (
        patch("pydefi.yields.tracker.ERC20") as ERC20_mock,
        patch("pydefi.yields.tracker.asyncio.sleep", new=AsyncMock(return_value=None)),
    ):
        ERC20_mock.fns.balanceOf.return_value.call = AsyncMock(return_value=100)
        with pytest.raises(BridgeError, match="bridge settlement timeout"):
            await wait_for_bridge_settlement(
                w3=cast(AsyncWeb3, object()),  # ERC20.balanceOf is patched
                token_address=Address("0x" + "11" * 20),
                recipient=_USER,
                starting_balance=100,
                min_delta=1_000,
                timeout_seconds=0.0,
                poll_interval_seconds=0.01,
            )


# ---------------------------------------------------------------------------
# expected_apy_gain / find_best_rebalance
# ---------------------------------------------------------------------------

_YEAR = 365 * 24 * 60 * 60


def test_expected_apy_gain_positive_for_better_market():
    current = _position(_market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04"), 1_000 * 10**6)
    better = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    # 1000 USDC × (6% − 4%) × 1 year = 20 USDC == 20_000_000 sub-units.
    assert expected_apy_gain(current, better, horizon_seconds=_YEAR) == 20_000_000


def test_expected_apy_gain_negative_for_worse_market():
    current = _position(_market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06"), 1_000 * 10**6)
    worse = _market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04")
    assert expected_apy_gain(current, worse, horizon_seconds=_YEAR) == -20_000_000


def test_expected_apy_gain_scales_linearly_with_horizon():
    current = _position(_market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04"), 10_000 * 10**6)
    better = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.05")
    year = expected_apy_gain(current, better, horizon_seconds=_YEAR)
    month = expected_apy_gain(current, better, horizon_seconds=30 * 24 * 60 * 60)
    # 30/365 of the full-year gain (±1 unit for integer rounding).
    assert abs(month - year * 30 // 365) <= 1


@pytest.mark.asyncio
async def test_find_best_rebalance_picks_highest_gain_same_chain():
    current = _position(_market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04"), 1_000 * 10**6)
    candidates = [
        _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06"),
        _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.08"),  # winner
        _market("compound_v3", ChainId.OPTIMISM, USDC_OPT, apy="0.20"),  # ignored: wrong chain
    ]
    captured: dict = {}

    async def fake_build(strategy, **kw):
        captured.update(strategy=strategy, target=kw["target_market"], source=kw["source_market"])
        return "ROUTE_SENTINEL"

    with patch("pydefi.yields.rebalance.build_yield_route", new=fake_build):
        result = await find_best_rebalance(
            _USER,
            positions=[current],
            markets=candidates,
            w3s={ChainId.BASE: object()},
            horizon_seconds=_YEAR,
        )

    assert result == "ROUTE_SENTINEL"
    assert captured["strategy"] == "withdraw_then_supply"
    assert captured["target"].supply_apy == Decimal("0.08")
    assert captured["source"].market_id == current.market.market_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate_apy,reason",
    [
        ("0.043", "below_threshold"),  # 30 bps over current 4%, default threshold is 50
        ("0.04", "same_market_id"),  # same market — must be skipped even at equal APY
    ],
)
async def test_find_best_rebalance_returns_none(candidate_apy, reason):
    current = _position(_market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04"), 1_000 * 10**6)
    candidate = _market(
        "aave_v3" if reason == "same_market_id" else "compound_v3", ChainId.BASE, USDC_BASE, apy=candidate_apy
    )
    with patch("pydefi.yields.rebalance.build_yield_route", new=AsyncMock()) as build:
        result = await find_best_rebalance(
            _USER,
            positions=[current],
            markets=[candidate],
            w3s={ChainId.BASE: object()},
            horizon_seconds=_YEAR,
        )
    assert result is None
    assert build.await_count == 0


# ---------------------------------------------------------------------------
# rebalance_tick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebalance_tick_returns_none_when_user_has_no_positions():
    markets_mock = AsyncMock(return_value=[])
    find_mock = AsyncMock(return_value=None)
    with (
        patch("pydefi.yields.rebalance.get_positions", new=AsyncMock(return_value=[])),
        patch("pydefi.yields.rebalance.get_yield_markets", new=markets_mock),
        patch("pydefi.yields.rebalance.find_best_rebalance", new=find_mock),
    ):
        result = await rebalance_tick(_USER, "USDC", w3s={ChainId.BASE: object()})
    assert result is None
    # Short-circuited: no point scanning markets or picking when there are no positions.
    assert markets_mock.await_count == 0
    assert find_mock.await_count == 0


@pytest.mark.asyncio
async def test_rebalance_tick_forwards_thresholds():
    positions = [_position(_market("aave_v3", ChainId.BASE, USDC_BASE, apy="0.04"), 1_000 * 10**6)]
    candidate = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    find_mock = AsyncMock(return_value="ROUTE_SENTINEL")
    with (
        patch("pydefi.yields.rebalance.get_positions", new=AsyncMock(return_value=positions)),
        patch("pydefi.yields.rebalance.get_yield_markets", new=AsyncMock(return_value=[candidate])),
        patch("pydefi.yields.rebalance.find_best_rebalance", new=find_mock),
    ):
        result = await rebalance_tick(
            _USER,
            "USDC",
            w3s={ChainId.BASE: object()},
            horizon_seconds=86_400,
            min_apy_gain_bps=10,
        )
    assert result == "ROUTE_SENTINEL"
    kwargs = find_mock.await_args.kwargs
    assert kwargs["horizon_seconds"] == 86_400
    assert kwargs["min_apy_gain_bps"] == 10
    assert kwargs["positions"] == positions
    assert kwargs["markets"] == [candidate]


# ---------------------------------------------------------------------------
# Compose route — one-signature cross-chain yield deposit (CCTP / CCIP)
# ---------------------------------------------------------------------------

USDC_ETH = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "E7" * 20), symbol="USDC", decimals=6)
_COMPOSER = Address("0x" + "C0" * 20)
_COMPOSE_POOL = Address("0x" + "A0" * 20)
_COMPOSE_AMOUNT = TokenAmount(USDC_ETH, 1_000_000)
_DELEGATE = Address("0x" + "DE" * 20)


def _compose_bridge(kind: str) -> Bridge:
    """A compose-capable bridge moving USDC from Ethereum to Base."""
    cls = CCTP if kind == "cctp" else CCIP
    return cls(w3=None, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.BASE)


async def _compose_route(bridge, w3s, *, amount=_COMPOSE_AMOUNT, **knobs):
    """Build a compose route with the common args; the caller sets up patches."""
    return await build_compose_supply_route(
        _USER,
        amount,
        _market("aave_v3", ChainId.BASE, USDC_BASE),
        composer_address=_COMPOSER,
        bridge=bridge,
        w3s=w3s,
        **knobs,
    )


@pytest.mark.parametrize("protocol", ["aave_v3", "compound_v3"])
def test_build_compose_supply_program_compiles(protocol):
    """build_compose_supply_program emits DeFiVM bytecode for each supported protocol."""
    program = build_compose_supply_program(protocol, _COMPOSE_POOL, USDC_ETH, _USER)
    assert isinstance(program, bytes) and len(program) > 0


def test_build_compose_supply_program_rejects_unsupported_protocol():
    with pytest.raises(ValueError, match="unsupported protocol"):
        build_compose_supply_program("morpho", _COMPOSE_POOL, USDC_ETH, _USER)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_kind", ["cctp", "ccip"])
async def test_build_compose_supply_route_is_single_signature(bridge_kind):
    """A compose-supply route is [approve, bridge] on the source chain only —
    the destination supply rides in the bridge's compose hook, so there is no
    pending leg. The route shape is identical for every compose-capable bridge."""
    bridge = _compose_bridge(bridge_kind)
    with (
        patch("pydefi.yields.compose.supply_contract", new=AsyncMock(return_value=_COMPOSE_POOL)),
        patch.object(CCIP, "quote_fee", new=AsyncMock(return_value=10**15)),
    ):
        route = await _compose_route(bridge, {ChainId.BASE: object()})

    assert route.strategy == "compose_supply"
    assert route.pending is None  # the destination supply runs inside the hook
    assert [s.kind for s in route.steps] == ["approve", "bridge"]
    approve_tx, bridge_tx = route.build_transactions()
    # The approve targets USDC and grants the bridge entrypoint the compose tx hits.
    assert Address(approve_tx["to"]) == USDC_ETH.address
    assert Address(bridge_tx["to"]) in Address(approve_tx["data"])


@pytest.mark.asyncio
async def test_build_compose_supply_route_delegate_collapses_to_one_sponsored_step():
    """delegate= (EIP-7702): the source leg is a single bridge_with_7702 step
    batching [approve, bridge] in the EOA's own context — the user signs once
    via sign_route and a relayer broadcasts. The destination supply still rides
    in the compose hook, so the whole cross-chain deposit is one signature."""
    bridge = _compose_bridge("cctp")
    w3_src = MagicMock()
    w3_src.eth.get_transaction_count = AsyncMock(return_value=3)
    with (
        patch("pydefi.yields.compose.supply_contract", new=AsyncMock(return_value=_COMPOSE_POOL)),
        patch("pydefi.yields.router.delegation_status", AsyncMock(return_value=(0, True))),
    ):
        route = await _compose_route(bridge, {ChainId.BASE: object(), ChainId.ETHEREUM: w3_src}, delegate=_DELEGATE)

    assert route.pending is None
    assert [s.kind for s in route.steps] == ["bridge_with_7702"]
    step = route.steps[0]
    assert step.chain_id == ChainId.ETHEREUM
    assert step.tx is None  # unbroadcastable until sign_route fills it in
    req = step.sign_request
    # The batch is [approve(USDC -> messenger), depositForBurnWithHook].
    assert Address(req["calls"][0]["to"]) == USDC_ETH.address
    assert Address(req["calls"][1]["to"]) == bridge.spender
    assert req["needs_auth"] is True
    assert req["auth_nonce"] == 3
    # sign_route turns the batch into one broadcast-ready execute tx on the EOA.
    signed = sign_route(route, ANVIL_PK)
    assert signed.steps[0].sign_request is None
    assert Address(signed.steps[0].tx["to"]) == _USER  # the delegated EOA itself


@pytest.mark.asyncio
async def test_build_compose_supply_route_defivm_collapses_to_one_witness_step():
    """defivm= (Permit2): the source leg is a single bridge_with_permit2 step
    whose witness-bound program approves the bridge entrypoint and runs the burn
    inside the VM, so the user signs once and a relayer broadcasts."""
    from pydefi.vm.permit2_supply import build_supply_program

    bridge = _compose_bridge("cctp")
    w3_src = MagicMock()
    w3_src.eth.call = AsyncMock(return_value=(0).to_bytes(32, "big"))  # no standing VM allowance
    with (
        patch("pydefi.yields.compose.supply_contract", new=AsyncMock(return_value=_COMPOSE_POOL)),
        patch("pydefi.yields.router.pick_nonce", AsyncMock(return_value=42)),
    ):
        route = await _compose_route(bridge, {ChainId.BASE: object(), ChainId.ETHEREUM: w3_src}, defivm=_DEFIVM)

    assert route.pending is None
    assert [s.kind for s in route.steps] == ["bridge_with_permit2"]
    step = route.steps[0]
    assert step.chain_id == ChainId.ETHEREUM and step.tx is None
    req = step.sign_request
    assert (req["defivm"], req["token"], req["amount"], req["nonce"]) == (
        _DEFIVM,
        USDC_ETH.address,
        _COMPOSE_AMOUNT.amount,
        42,
    )
    # The witness-bound program is exactly [approve(messenger), burn calldata].
    hook = build_compose_supply_program("aave_v3", _COMPOSE_POOL, USDC_BASE, _USER)
    bridge_tx = await bridge.build_bridge_compose_tx(_COMPOSE_AMOUNT, _COMPOSER, hook)
    assert req["program"] == build_supply_program(
        USDC_ETH.address, bridge.spender, bytes.fromhex(bridge_tx["data"][2:]), approve=True, reset=False
    )
    # sign_route fills in the broadcast-ready executeWithPermit2 tx on the VM.
    signed = sign_route(route, ANVIL_PK)
    assert signed.steps[0].sign_request is None
    assert Address(signed.steps[0].tx["to"]) == _DEFIVM


@pytest.mark.asyncio
async def test_build_compose_supply_route_defivm_rejects_native_fee_bridge():
    """The VM holds no native funds — a bridge whose compose tx carries value
    (CCIP's native-fee lane) cannot ride the Permit2 path."""
    bridge = _compose_bridge("ccip")
    with (
        patch("pydefi.yields.compose.supply_contract", new=AsyncMock(return_value=_COMPOSE_POOL)),
        patch.object(CCIP, "quote_fee", new=AsyncMock(return_value=10**15)),
        pytest.raises(ValueError, match="native fee"),
    ):
        await _compose_route(bridge, {ChainId.BASE: object(), ChainId.ETHEREUM: MagicMock()}, defivm=_DEFIVM)


@pytest.mark.asyncio
async def test_build_compose_supply_route_delegate_requires_source_w3():
    """The gasless source leg reads delegation state on the source chain, so
    w3s must carry it — rejected with a descriptive error, not a KeyError."""
    bridge = _compose_bridge("cctp")
    with (
        patch("pydefi.yields.compose.supply_contract", new=AsyncMock(return_value=_COMPOSE_POOL)),
        pytest.raises(ValueError, match="no entry for source chain"),
    ):
        await _compose_route(bridge, {ChainId.BASE: object()}, delegate=_DELEGATE)  # destination only


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_kind", ["cctp", "ccip"])
async def test_build_compose_supply_route_validates_chains(bridge_kind):
    """The bridge's src / dst must match the amount and target chains."""
    bridge = _compose_bridge(bridge_kind)
    with pytest.raises(ValueError, match="bridge source"):
        # source token on Base, but the bridge sources Ethereum
        await _compose_route(bridge, {ChainId.BASE: object()}, amount=TokenAmount(USDC_BASE, 1))


# ---------------------------------------------------------------------------
# Gasless deposits via DeFiVM.executeWithPermit2 (supply_with_permit2)
# ---------------------------------------------------------------------------

_DEFIVM = Address("0x" + "DD" * 20)
_COMET = Address("0x" + "CE" * 20)
_USDC_ETH = Token(chain_id=ChainId.ETHEREUM, address=Address("0x" + "a0" * 20), symbol="USDC", decimals=6)
_MOCK_W3 = cast(AsyncWeb3, object())  # never called; protocol client is mocked
ANVIL_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Anvil account[0]


def _mock_comet(comet_addr: Address):
    """CompoundV3.from_chain → client with comet_address + build_supply_tx(amount, dst)."""
    client = AsyncMock()
    client.comet_address = comet_addr
    client.build_supply_tx = lambda amount, dst=None: {
        "to": "0x" + bytes(comet_addr).hex(),
        "data": "0xdeadbeef",  # stand-in supply / supplyTo calldata
        "value": "0",
        "gas": "100000",
    }
    return patch("pydefi.yields.router.CompoundV3"), client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowance,approve,reset",
    [(0, True, False), (10, True, True), (2**255, False, False)],
    ids=["first", "partial", "primed"],
)
async def test_supply_steps_gasless_emits_permit2_step(allowance, approve, reset):
    """With a defivm, the supply leg is a single supply_with_permit2 step whose
    compiled program the witness signature will bind; the approve leg is included
    only when the VM's standing allowance is insufficient, and the USDT-safe
    reset only when that allowance is nonzero."""
    from pydefi.vm.permit2_supply import build_supply_program

    market = _market("compound_v3", ChainId.ETHEREUM, _USDC_ETH)
    comet_patch, client = _mock_comet(_COMET)
    w3 = MagicMock()
    w3.eth.call = AsyncMock(return_value=allowance.to_bytes(32, "big"))  # ERC20.allowance(vm, protocol)
    with comet_patch as Comet, patch("pydefi.yields.router.pick_nonce", AsyncMock(return_value=42)):
        Comet.from_chain = lambda *a, **k: client  # from_chain is sync
        steps = await _supply_steps(market, _USER, w3, TokenAmount(_USDC_ETH, 1_000_000), defivm=_DEFIVM)

    assert [s.kind for s in steps] == ["supply_with_permit2"]
    req = steps[0].sign_request
    assert req["defivm"] == _DEFIVM
    assert req["nonce"] == 42
    # The program embeds the supply calldata and matches a direct compile.
    assert bytes.fromhex("deadbeef") in req["program"]
    assert req["program"] == build_supply_program(
        _USDC_ETH.address, _COMET, bytes.fromhex("deadbeef"), approve=approve, reset=reset
    )
    assert "deadline" not in req  # stamped at sign time, not build time
    assert steps[0].tx is None  # unbroadcastable until sign_route fills it in


@pytest.mark.asyncio
async def test_supply_steps_7702_auth_nonce_skips_preceding_user_txs():
    """The 7702 authorization is consumed at submission time: route steps the EOA
    broadcasts first (e.g. a withdraw) each consume an account nonce, so the auth
    must be signed past them or it is silently skipped on-chain."""
    market = _market("compound_v3", ChainId.ETHEREUM, _USDC_ETH)
    comet_patch, client = _mock_comet(_COMET)
    w3 = MagicMock()
    w3.eth.get_transaction_count = AsyncMock(return_value=7)
    with (
        comet_patch as Comet,
        patch("pydefi.yields.router.delegation_status", AsyncMock(return_value=(0, True))),
    ):
        Comet.from_chain = lambda *a, **k: client
        steps = await _supply_steps(
            market,
            _USER,
            w3,
            TokenAmount(_USDC_ETH, 1_000_000),
            delegate=_DEFIVM,
            user_txs_before=1,
        )
    assert steps[0].sign_request["auth_nonce"] == 8  # 7 current + 1 preceding withdraw


@pytest.mark.asyncio
async def test_supply_steps_gasless_rejects_aave_v4():
    """Aave V4's Spoke gates onBehalfOf supplies behind position-manager approval,
    so the Permit2 path can never succeed — refuse it up front."""
    market = _market("aave_v4", ChainId.ETHEREUM, _USDC_ETH)
    with pytest.raises(ValueError, match="position manager"):
        await _supply_steps(market, _USER, _MOCK_W3, TokenAmount(_USDC_ETH, 1_000_000), defivm=_DEFIVM)


@pytest.mark.asyncio
async def test_supply_steps_without_defivm_keeps_approve_supply():
    """No defivm → the classic [approve, supply] pair."""
    market = _market("compound_v3", ChainId.ETHEREUM, _USDC_ETH)
    comet_patch, client = _mock_comet(_COMET)
    with comet_patch as Comet:
        Comet.from_chain = lambda *a, **k: client
        steps = await _supply_steps(market, _USER, _MOCK_W3, TokenAmount(_USDC_ETH, 1_000_000))
    assert [s.kind for s in steps] == ["approve", "supply"]


def test_sign_route_builds_permit2_supply_tx():
    """sign_route stamps a fresh deadline, signs the program-bound witness, and
    fills in the DeFiVM.executeWithPermit2 calldata."""
    import time

    from pydefi.abi.vm import DeFiVM

    step = YieldStep(
        "supply_with_permit2",
        ChainId.ETHEREUM,
        None,
        sign_request={
            "defivm": _DEFIVM,
            "token": _USDC_ETH.address,
            "amount": 1_000_000,
            "nonce": 42,
            "owner": _USER,
            "program": bytes.fromhex("deadbeef"),
        },
    )
    route = YieldRoute("supply_then_bridge", ChainId.ETHEREUM, (step,), "test")
    signed = sign_route(route, ANVIL_PK)
    tx = signed.steps[0].tx
    assert signed.steps[0].sign_request is None
    assert tx["to"].lower() == "0x" + "dd" * 20
    assert tx["data"][2:10] == DeFiVM.fns.executeWithPermit2.selector.hex()
    # The permit deadline (3rd word of the permit tail, after the 4-word head:
    # permit offset, owner, signature offset, program offset) starts at sign
    # time — roughly now + 1h, never a stale build-time stamp.
    deadline = int.from_bytes(bytes.fromhex(tx["data"][2:])[4 + 32 * 6 : 4 + 32 * 7], "big")
    assert abs(deadline - (int(time.time()) + 3600)) < 60


def test_build_transactions_raises_on_unsigned_permit2_step():
    step = YieldStep("supply_with_permit2", ChainId.ETHEREUM, None, sign_request={"x": 1})
    route = YieldRoute("supply_then_bridge", ChainId.ETHEREUM, (step,), "test")
    with pytest.raises(RuntimeError, match="unsigned steps"):
        route.build_transactions()

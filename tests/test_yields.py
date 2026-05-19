"""Unit tests for pydefi.yields (no live node required)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from web3 import Web3

from pydefi.exceptions import BridgeError
from pydefi.lending.aave_v3 import ReserveData
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.yields import (
    Position,
    YieldMarket,
    build_approve_tx,
    build_yield_route,
    get_positions,
    get_yield_markets,
    wait_for_bridge_settlement,
)

# ---------------------------------------------------------------------------
# Tokens / addresses / canned tx dicts (shared across every test)
# ---------------------------------------------------------------------------

USDC_BASE = Token(chain_id=ChainId.BASE, address=Address("0x" + "11" * 20), symbol="USDC", decimals=6)
USDC_OPT = Token(chain_id=ChainId.OPTIMISM, address=Address("0x" + "22" * 20), symbol="USDC", decimals=6)
USDC_MANTRA = Token(chain_id=ChainId.MANTRA, address=Address("0x" + "EE" * 20), symbol="USDC", decimals=6)

_USER = Address("0x" + "AA" * 20)
_STUB_WITHDRAW = {"to": Address("0x" + "11" * 20), "data": "0xdead", "value": "0", "gas": "100000"}
_STUB_SUPPLY = {"to": Address("0x" + "22" * 20), "data": "0xbeef", "value": "0", "gas": "100000"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(protocol: str, chain_id: int, token: Token, apy: str = "0.04") -> YieldMarket:
    return YieldMarket(
        protocol=protocol,  # type: ignore[arg-type]
        chain_id=chain_id,
        token=token,
        supply_apy=Decimal(apy),
        utilization=Decimal("0.5"),
        available_liquidity=TokenAmount(token=token, amount=10**12),
        market_id=f"{protocol}:{chain_id}:{token.symbol}",
    )


def _fake_bridge(src_chain: int, dst_chain: int, controller: Address = Address("0x" + "CC" * 20)) -> object:
    """Stand-in for LucidBridge — exposes only the attributes build_yield_route reads."""
    bridge = AsyncMock()
    bridge.src_chain_id = src_chain
    bridge.dst_chain_id = dst_chain
    bridge.controller_address = controller
    bridge.build_bridge_tx = AsyncMock(
        return_value={"to": controller, "data": "0xfeed", "value": "1000", "gas": "500000"}
    )
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
        Aave.return_value.get_reserve_data = AsyncMock(return_value=_paused_reserve(USDC_BASE))
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


def test_yield_market_is_frozen_dataclass():
    m = _market("aave_v3", ChainId.BASE, USDC_BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.supply_apy = Decimal("0.05")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_approve_tx
# ---------------------------------------------------------------------------


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
        patch("pydefi.yields.router._withdraw_tx", return_value=_STUB_WITHDRAW),
        patch("pydefi.yields.router._supply_tx", return_value=_STUB_SUPPLY),
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
    txs = route.build_transactions()
    assert txs[0]["data"] == _STUB_WITHDRAW["data"]
    assert txs[2]["data"] == _STUB_SUPPLY["data"]


@pytest.mark.asyncio
async def test_withdraw_then_supply_cross_chain_emits_bridge():
    src = _market("aave_v3", ChainId.MANTRA, USDC_MANTRA)
    dst = _market("compound_v3", ChainId.BASE, USDC_BASE, apy="0.06")
    bridge = _fake_bridge(ChainId.MANTRA, ChainId.BASE)
    amount = TokenAmount(USDC_MANTRA, 1_000_000)
    with patch("pydefi.yields.router._withdraw_tx", return_value=_STUB_WITHDRAW):
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


# ---------------------------------------------------------------------------
# build_yield_route — supply_then_bridge / bridge_then_supply happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supply_then_bridge_entry_leg():
    market = _market("aave_v3", ChainId.BASE, USDC_BASE)
    amount = TokenAmount.from_human(USDC_BASE, "1000")
    with patch("pydefi.yields.router._supply_tx", return_value=_STUB_SUPPLY):
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
    txs = route.build_transactions()
    assert txs[0]["to"] == USDC_MANTRA.address  # approve targets the source token
    assert txs[1]["to"] == controller  # bridge targets the controller


# ---------------------------------------------------------------------------
# build_yield_route — validation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_withdraw_then_supply_cross_chain_requires_bridge():
    with pytest.raises(ValueError, match="cross-chain.*requires a configured LucidBridge"):
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
        ("bridge_then_supply", {}, "requires a configured LucidBridge"),
        ("supply_then_borrow", {}, "unknown strategy"),
    ],
    ids=["missing_source_market", "missing_bridge", "unknown_strategy"],
)
async def test_build_yield_route_validation_errors(strategy, kw, err):
    target = _market("aave_v3", ChainId.BASE, USDC_BASE)
    with pytest.raises(ValueError, match=err):
        await build_yield_route(
            strategy,
            _USER,
            TokenAmount.from_human(USDC_BASE, "1"),  # type: ignore[arg-type]
            w3s={ChainId.BASE: object()},
            target_market=target,
            **kw,
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
            w3=object(),  # type: ignore[arg-type]
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
                w3=object(),  # type: ignore[arg-type]
                token_address=Address("0x" + "11" * 20),
                recipient=_USER,
                starting_balance=100,
                min_delta=1_000,
                timeout_seconds=0.0,
                poll_interval_seconds=0.01,
            )

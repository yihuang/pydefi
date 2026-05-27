"""Fork tests for :mod:`pydefi.yields` against an Anvil mainnet fork.

Covers the two strategies whose source-chain leg is fully built today:

* ``supply_then_bridge`` — entry leg (approve + supply). The bridge tail
  is deferred and not exercised here.
* ``withdraw_then_supply`` — same-chain rebalance from Aave V3 to
  Compound V3 USDC.

Both reuse :data:`tests.addrs.ETH_WHALE` (vitalik) as the user, top its
ETH balance for gas, and seed USDC from :data:`tests.addrs.USDC_WHALE`.

Run with::

    pytest -m fork tests/live/test_yields_fork.py
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from eth_contract.erc20 import ERC20

from pydefi.deployments import get_address
from pydefi.lending import AaveV3, CompoundV3
from pydefi.types import ChainId, TokenAmount
from pydefi.yields import YieldMarket, YieldRoute, build_yield_route
from pydefi.yields.router import Protocol
from tests.addrs import ETH_WHALE, USDC
from tests.live.anvil_helpers import fund_usdc, impersonate, send_tx, set_balance

# ---------------------------------------------------------------------------
# Pinned addresses + per-test constants
# ---------------------------------------------------------------------------

COMET_USDC = get_address("COMPOUND_V3_USDC", ChainId.ETHEREUM)

USDC_TEST_AMOUNT = 1_000 * 10**6  # 1000 USDC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usdc_market(protocol: Protocol, apy: str = "0.04") -> YieldMarket:
    """Synthesized YieldMarket — APY/util/liquidity are placeholders since
    build_yield_route doesn't read them; live versions live in get_yield_markets."""
    return YieldMarket(
        protocol=protocol,
        chain_id=ChainId.ETHEREUM,
        token=USDC,
        supply_apy=Decimal(apy),
        utilization=Decimal("0.7"),
        available_liquidity=TokenAmount(USDC, 10**18),
        market_id=f"{protocol}:{ChainId.ETHEREUM}:USDC",
    )


async def _seed_whale_with_usdc(fork_w3, amount: int) -> None:
    await impersonate(fork_w3, ETH_WHALE)
    await set_balance(fork_w3, ETH_WHALE, 100 * 10**18)  # gas
    await fund_usdc(fork_w3, USDC.address, ETH_WHALE, amount)


async def _seed_aave_position(fork_w3, amount: int) -> None:
    """Set up a pre-existing aUSDC position on Aave for rebalance scenarios."""
    aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
    approve_tx = {
        "to": USDC.address,
        "data": "0x" + ERC20.fns.approve(bytes(aave.pool_address), amount).data.hex(),
        "value": "0",
        "gas": "100000",
    }
    r = await send_tx(fork_w3, ETH_WHALE, approve_tx)
    assert r["status"] == 1, "USDC approve to Aave reverted"
    r = await send_tx(fork_w3, ETH_WHALE, aave.build_supply_tx(ETH_WHALE, TokenAmount(USDC, amount)))
    assert r["status"] == 1, "Aave supply reverted"


async def _broadcast(fork_w3, route: YieldRoute) -> None:
    for step in route.steps:
        receipt = await send_tx(fork_w3, ETH_WHALE, step.tx)
        assert receipt["status"] == 1, f"{step.kind} step reverted"


async def _a_usdc_balance(fork_w3) -> int:
    """Whale's aUSDC balance (Aave V3 receipt token on USDC)."""
    aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
    a_usdc = (await aave.get_reserve_data(USDC)).a_token_address
    return int(await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=a_usdc))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestYieldRouterFork:
    """End-to-end: build_yield_route + broadcast all steps against a real fork."""

    async def test_supply_then_bridge_entry_leg_mints_atoken(self, fork_w3):
        """[approve, supply] from a supply_then_bridge route must leave the whale holding aUSDC."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)
        before = await _a_usdc_balance(fork_w3)

        route = await build_yield_route(
            "supply_then_bridge",
            user=ETH_WHALE,
            amount_in=TokenAmount(USDC, USDC_TEST_AMOUNT),
            w3s={ChainId.ETHEREUM: fork_w3},
            target_market=_usdc_market("aave_v3"),
            target_chain=ChainId.KITE,
        )
        assert [s.kind for s in route.steps] == ["approve", "supply"]
        await _broadcast(fork_w3, route)

        # aUSDC mints ~1:1 with supplied principal (Aave may round by ±1 unit).
        assert (await _a_usdc_balance(fork_w3)) - before >= USDC_TEST_AMOUNT - 1

    async def test_withdraw_then_supply_rebalances_aave_to_compound(self, fork_w3):
        """A withdraw_then_supply route must move the whale's USDC from Aave V3
        into Compound V3 in one [withdraw, approve, supply] sequence."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)
        await _seed_aave_position(fork_w3, USDC_TEST_AMOUNT)

        # Aave's rayDiv/rayMul rounding can shave one wei off the supplied
        # principal, so withdrawing USDC_TEST_AMOUNT exactly trips
        # NotEnoughAvailableUserBalance(). Pull a notch under that — also
        # what a real rebalancer would do after calling get_positions.
        rebalance_amount = USDC_TEST_AMOUNT - 1_000  # 0.001 USDC slack
        a_before = await _a_usdc_balance(fork_w3)
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        comet_before = (await comet.get_user_position(ETH_WHALE)).base_supply.amount

        route = await build_yield_route(
            "withdraw_then_supply",
            user=ETH_WHALE,
            amount_in=TokenAmount(USDC, rebalance_amount),
            w3s={ChainId.ETHEREUM: fork_w3},
            source_market=_usdc_market("aave_v3"),
            target_market=_usdc_market("compound_v3", apy="0.05"),
        )
        assert [s.kind for s in route.steps] == ["withdraw", "approve", "supply"]
        await _broadcast(fork_w3, route)

        a_after = await _a_usdc_balance(fork_w3)
        comet_after = (await comet.get_user_position(ETH_WHALE)).base_supply.amount

        # aUSDC drained by ~rebalance_amount; Compound base supply grew by it.
        # Aave's rayMul/rayDiv adds a few wei of slack each direction, so the
        # tolerance has to absorb both the supply-side and withdraw-side rounding.
        _ROUNDING_SLACK = 100  # 0.0001 USDC
        assert a_before - a_after >= rebalance_amount - _ROUNDING_SLACK
        assert comet_after - comet_before >= rebalance_amount - _ROUNDING_SLACK

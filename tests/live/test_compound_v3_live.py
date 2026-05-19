"""Live integration tests for the Compound V3 (Comet) module.

* ``@pytest.mark.live`` — read-only calls against a public Ethereum RPC
  (``eth_w3`` fixture). Verifies ``get_market_data``, collateral list,
  and user reads against the cUSDCv3 mainnet market.
* ``@pytest.mark.fork`` — supply / withdraw / borrow round-trip on an
  Anvil mainnet fork, sourcing USDC via :data:`tests.addrs.USDC_WHALE`.

Run live read tests with::

    pytest -m live tests/live/test_compound_v3_live.py

Run fork tests with::

    pytest -m fork tests/live/test_compound_v3_live.py
"""

from __future__ import annotations

import secrets
from decimal import Decimal

import pytest
from eth_contract.erc20 import ERC20

from pydefi.deployments import get_address
from pydefi.lending import CompoundV3
from pydefi.lending.compound_v3 import UINT256_MAX
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import ETH_WHALE, USDC, WETH
from tests.live.anvil_helpers import (
    erc20_approve,
    fund_usdc,
    impersonate,
    send_tx,
    set_balance,
    wrap_eth,
)

CUSDC_V3 = Address(get_address("COMPOUND_V3_USDC", ChainId.ETHEREUM))

WHALE_BALANCE = 100 * 10**18
WETH_SUPPLY_AMOUNT = 5 * 10**18
USDC_SUPPLY_AMOUNT = 10_000 * 10**6
USDC_BORROW_AMOUNT = 1_000 * 10**6


def _make_comet(w3) -> CompoundV3:
    return CompoundV3(w3=w3, chain_id=ChainId.ETHEREUM, comet_address=CUSDC_V3, base_token=USDC)


# ---------------------------------------------------------------------------
# Live read tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCompoundV3LiveReads:
    """Read-only smoke tests against cUSDCv3 on Ethereum mainnet."""

    async def test_market_data(self, eth_w3):
        m = await _make_comet(eth_w3).get_market_data()
        assert m.base_token == USDC
        assert Decimal(0) <= m.supply_apy < Decimal("0.5"), f"implausible supply APY: {m.supply_apy}"
        assert Decimal(0) <= m.borrow_apy < Decimal("0.5"), f"implausible borrow APY: {m.borrow_apy}"
        # The spread between supply and borrow APY funds reserves.
        assert m.supply_apy <= m.borrow_apy
        assert Decimal(0) <= m.utilization <= Decimal(1)
        assert m.total_supply.amount > 0
        assert m.total_borrow.amount >= 0

    async def test_collateral_assets_listed(self, eth_w3):
        assets = await _make_comet(eth_w3).get_collateral_assets()
        assert len(assets) > 0
        # WETH is reliably a collateral asset in cUSDCv3.
        weth_lc = bytes(WETH.address).hex().lower()
        assert any(bytes(a.asset.address).hex().lower() == weth_lc for a in assets)

    async def test_user_position_empty_account(self, eth_w3):
        # Ephemeral address — guaranteed never to have interacted with the market.
        fresh = Address(secrets.token_bytes(20))
        pos = await _make_comet(eth_w3).get_user_position(fresh)
        assert pos.base_supply.amount == 0
        assert pos.base_borrow.amount == 0
        assert pos.is_liquidatable is False
        assert all(b.amount == 0 for b in pos.collateral.values())


# ---------------------------------------------------------------------------
# Fork tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestCompoundV3Fork:
    """End-to-end supply / withdraw / borrow tests on Anvil mainnet fork."""

    @staticmethod
    async def _prepare_whale_with_usdc(fork_w3, usdc_amount: int) -> None:
        """Impersonate the whale, top up ETH for gas, and fund it with USDC."""
        await impersonate(fork_w3, ETH_WHALE)
        await set_balance(fork_w3, ETH_WHALE, WHALE_BALANCE)
        await fund_usdc(fork_w3, USDC.address, ETH_WHALE, usdc_amount)
        await erc20_approve(fork_w3, USDC.address, ETH_WHALE, CUSDC_V3, usdc_amount)

    @staticmethod
    async def _prepare_whale_with_weth(fork_w3, weth_amount: int) -> None:
        """Impersonate the whale, top up ETH for gas, and wrap+approve WETH."""
        await impersonate(fork_w3, ETH_WHALE)
        await set_balance(fork_w3, ETH_WHALE, WHALE_BALANCE)
        await wrap_eth(fork_w3, ETH_WHALE, WETH.address, weth_amount)
        await erc20_approve(fork_w3, WETH.address, ETH_WHALE, CUSDC_V3, weth_amount)

    async def test_supply_usdc_base_increases_balance(self, fork_w3):
        """Supplying the base asset must grow ``balanceOf(user)`` 1:1."""
        comet = _make_comet(fork_w3)
        await self._prepare_whale_with_usdc(fork_w3, USDC_SUPPLY_AMOUNT)

        assert (await comet.get_user_position(ETH_WHALE)).base_supply.amount == 0

        receipt = await send_tx(fork_w3, ETH_WHALE, comet.build_supply_tx(TokenAmount(USDC, USDC_SUPPLY_AMOUNT)))
        assert receipt["status"] == 1, "supply reverted"

        after = await comet.get_user_position(ETH_WHALE)
        # Comet may round by ±1 unit (accrual scaling).
        assert abs(after.base_supply.amount - USDC_SUPPLY_AMOUNT) <= 1
        assert after.base_borrow.amount == 0

    async def test_supply_collateral_and_borrow_base(self, fork_w3):
        """Supply WETH as collateral, borrow USDC via withdraw(), then repay."""
        comet = _make_comet(fork_w3)
        await self._prepare_whale_with_weth(fork_w3, WETH_SUPPLY_AMOUNT)

        # 1. Supply WETH collateral.
        receipt = await send_tx(fork_w3, ETH_WHALE, comet.build_supply_tx(TokenAmount(WETH, WETH_SUPPLY_AMOUNT)))
        assert receipt["status"] == 1, "WETH collateral supply reverted"
        weth_collateral = (await comet.get_user_position(ETH_WHALE)).collateral.get(WETH.address)
        assert weth_collateral is not None
        assert weth_collateral.amount == WETH_SUPPLY_AMOUNT

        # 2. Borrow USDC via withdraw() — Comet's unified entrypoint
        # opens a borrow when the base balance crosses zero.
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        receipt = await send_tx(fork_w3, ETH_WHALE, comet.build_withdraw_tx(TokenAmount(USDC, USDC_BORROW_AMOUNT)))
        assert receipt["status"] == 1, "USDC borrow (via withdraw) reverted"

        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        assert usdc_after - usdc_before == USDC_BORROW_AMOUNT
        pos = await comet.get_user_position(ETH_WHALE)
        assert pos.base_supply.amount == 0
        assert pos.base_borrow.amount >= USDC_BORROW_AMOUNT
        assert pos.is_liquidatable is False

        # 3. Over-repay by supplying with a small buffer for accrual. Comet
        # treats a supply with base_borrow > 0 as a repay first; any surplus
        # becomes a supply position.
        accrual_buffer = 10
        repay_amount = pos.base_borrow.amount + accrual_buffer
        await fund_usdc(fork_w3, USDC.address, ETH_WHALE, accrual_buffer)
        await erc20_approve(fork_w3, USDC.address, ETH_WHALE, CUSDC_V3, UINT256_MAX)
        receipt = await send_tx(fork_w3, ETH_WHALE, comet.build_supply_tx(TokenAmount(USDC, repay_amount)))
        assert receipt["status"] == 1, "supply-to-repay reverted"

        pos = await comet.get_user_position(ETH_WHALE)
        assert pos.base_borrow.amount == 0, "USDC borrow should be cleared after over-repay"
        assert pos.base_supply.amount > 0

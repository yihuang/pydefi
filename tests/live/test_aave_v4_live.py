"""Live integration tests for the Aave V4 lending module.

Aave V4 has no testnet deployment (``@aave-dao/aave-address-book`` ships only
``AaveV4Ethereum``), so — like :mod:`tests.live.test_aave_v3_live` — coverage
is a public-RPC read test plus an Anvil mainnet-fork write test.

* ``@pytest.mark.live`` — read-only calls against a public Ethereum RPC
  (``eth_w3``). Verifies ``get_reserve_data`` / ``get_user_account_data``
  on the real ``MAIN_SPOKE``.
* ``@pytest.mark.fork`` — supply / borrow / repay / withdraw round-trip
  against an Anvil mainnet fork (``fork_w3``). Impersonates ``ETH_WHALE``,
  wraps ETH to WETH, and exercises the full lifecycle on ``MAIN_SPOKE``.

Run with::

    pytest -m live tests/live/test_aave_v4_live.py
    pytest -m fork tests/live/test_aave_v4_live.py
"""

from __future__ import annotations

import secrets
from decimal import Decimal

import pytest
from eth_contract.erc20 import ERC20

from pydefi.lending import AaveV4
from pydefi.lending.utils import UINT256_MAX
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import ETH_WHALE, USDC, WETH
from tests.live.anvil_helpers import erc20_approve, fund_usdc, impersonate, send_ok, set_balance, wrap_eth

#: On the Aave V4 ``MAIN_SPOKE``, reserve 0 is WETH and reserve 7 is USDC.
#: Reserve ids are immutable indices, so these are stable.
WETH_RESERVE = 0
USDC_RESERVE = 7

WETH_SUPPLY_AMOUNT = 10 * 10**18
USDC_BORROW_AMOUNT = 5_000 * 10**6


# ---------------------------------------------------------------------------
# Live read tests (public RPC)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestAaveV4LiveReads:
    """Read-only smoke tests against the real MAIN_SPOKE on Ethereum mainnet."""

    async def test_get_reserve_data_weth(self, eth_w3):
        spoke = AaveV4.from_chain(eth_w3, ChainId.ETHEREUM, "MAIN_SPOKE")
        count = await spoke.get_reserve_count()
        assert count > 0

        data = await spoke.get_reserve_data(WETH_RESERVE)
        assert data.reserve.underlying.address == WETH.address
        assert data.reserve.underlying.decimals == 18
        assert not data.is_paused
        assert data.borrowable is True
        # Borrow APY comes from the Hub's drawn rate; supply APY scales it by
        # Hub utilization and the fee, so it is always the smaller of the two.
        assert Decimal(0) <= data.borrow_apy < Decimal("0.5"), f"implausible borrow APY: {data.borrow_apy}"
        assert Decimal(0) <= data.supply_apy <= data.borrow_apy
        assert data.utilization >= Decimal(0)
        assert data.supplied.amount > 0

    async def test_get_user_account_data_no_debt(self, eth_w3):
        """A fresh address has no debt and an infinite health factor."""
        spoke = AaveV4.from_chain(eth_w3, ChainId.ETHEREUM, "MAIN_SPOKE")
        fresh = Address(secrets.token_bytes(20))
        acc = await spoke.get_user_account_data(fresh)
        assert acc.borrow_count == 0
        assert acc.total_debt_value_ray == 0
        assert acc.health_factor.is_infinite()


# ---------------------------------------------------------------------------
# Fork tests (Anvil)
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestAaveV4Fork:
    """Supply / borrow / repay / withdraw round-trip on an Anvil mainnet fork."""

    async def test_supply_borrow_repay_withdraw(self, fork_w3):
        """Full lifecycle on MAIN_SPOKE: supply WETH → borrow USDC → repay → withdraw."""
        spoke = AaveV4.from_chain(fork_w3, ChainId.ETHEREUM, "MAIN_SPOKE")

        await impersonate(fork_w3, ETH_WHALE)
        await set_balance(fork_w3, ETH_WHALE, 100 * 10**18)
        await wrap_eth(fork_w3, ETH_WHALE, WETH.address, WETH_SUPPLY_AMOUNT)
        await erc20_approve(fork_w3, WETH.address, ETH_WHALE, spoke.spoke_address, WETH_SUPPLY_AMOUNT)

        # 1. Supply WETH.
        await send_ok(
            fork_w3,
            ETH_WHALE,
            spoke.build_supply_tx(WETH_RESERVE, TokenAmount(WETH, WETH_SUPPLY_AMOUNT), ETH_WHALE),
            "supply",
        )
        supplied = (await spoke.get_user_reserve(WETH_RESERVE, ETH_WHALE)).supplied.amount
        assert supplied >= WETH_SUPPLY_AMOUNT - 10

        # 2. Enable WETH as collateral.
        await send_ok(
            fork_w3, ETH_WHALE, spoke.build_set_collateral_tx(WETH_RESERVE, True, ETH_WHALE), "setUsingAsCollateral"
        )

        # 3. Borrow USDC against the WETH collateral.
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        await send_ok(
            fork_w3,
            ETH_WHALE,
            spoke.build_borrow_tx(USDC_RESERVE, TokenAmount(USDC, USDC_BORROW_AMOUNT), ETH_WHALE),
            "borrow",
        )
        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        assert usdc_after - usdc_before == USDC_BORROW_AMOUNT

        acc = await spoke.get_user_account_data(ETH_WHALE)
        assert acc.borrow_count >= 1
        assert acc.health_factor > Decimal(1), f"unexpectedly unhealthy: {acc.health_factor}"

        # 4. Repay the full debt. Fund a buffer for accrued interest and pass
        #    the live debt plus a small margin — the Spoke caps repayment at
        #    the actual debt.
        await fund_usdc(fork_w3, USDC.address, ETH_WHALE, USDC_BORROW_AMOUNT)
        await impersonate(fork_w3, ETH_WHALE)
        await erc20_approve(fork_w3, USDC.address, ETH_WHALE, spoke.spoke_address, UINT256_MAX)
        live_debt = (await spoke.get_user_reserve(USDC_RESERVE, ETH_WHALE)).debt.amount
        await send_ok(
            fork_w3,
            ETH_WHALE,
            spoke.build_repay_tx(USDC_RESERVE, TokenAmount(USDC, live_debt + 10 * 10**6), ETH_WHALE),
            "repay",
        )
        assert (await spoke.get_user_reserve(USDC_RESERVE, ETH_WHALE)).debt.amount == 0

        # 5. Withdraw the full WETH balance — principal plus the interest
        #    accrued over the lifecycle — now that the debt is cleared.
        weth_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=WETH.address)
        supplied_now = (await spoke.get_user_reserve(WETH_RESERVE, ETH_WHALE)).supplied.amount
        assert supplied_now >= WETH_SUPPLY_AMOUNT
        await send_ok(
            fork_w3,
            ETH_WHALE,
            spoke.build_withdraw_tx(WETH_RESERVE, TokenAmount(WETH, supplied_now), ETH_WHALE),
            "withdraw",
        )
        weth_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=WETH.address)
        assert weth_after - weth_before == supplied_now
        # Only interest accrued in the one block since the read can remain.
        residual = (await spoke.get_user_reserve(WETH_RESERVE, ETH_WHALE)).supplied.amount
        assert residual < 10**13, f"unexpected residual supply: {residual}"

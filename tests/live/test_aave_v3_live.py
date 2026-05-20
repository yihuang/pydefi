"""Live integration tests for the Aave V3 lending module.

* ``@pytest.mark.live`` — read-only calls against a public Ethereum RPC
  (``eth_w3`` fixture). Verifies that ``get_reserve_data``,
  ``get_user_account_data`` and ``get_asset_price`` return plausible
  values on real mainnet state.
* ``@pytest.mark.fork`` — supply / borrow / repay / withdraw round-trip
  against an Anvil mainnet fork (``fork_w3`` fixture). Impersonates
  ``ETH_WHALE`` (vitalik.eth), wraps ETH to WETH, and exercises the
  full lending lifecycle.

Run live read tests with::

    pytest -m live tests/live/test_aave_v3_live.py

Run fork write tests with::

    pytest -m fork tests/live/test_aave_v3_live.py
"""

from __future__ import annotations

import secrets
from decimal import Decimal

import pytest
from eth_contract.erc20 import ERC20
from web3.exceptions import Web3Exception

from pydefi.lending import AaveV3
from pydefi.lending.aave_v3 import UINT256_MAX
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import AAVE_ADDRESSES_PROVIDER, ETH_WHALE, USDC, USDC_WHALE, WETH
from tests.live.anvil_helpers import (
    erc20_approve,
    impersonate,
    send_tx,
    set_balance,
    wrap_eth,
)
from tests.live.sol_utils import compile_sol_source, deploy

# Test parameters — small enough to keep blocks fast and price-impact negligible.
WHALE_BALANCE = 100 * 10**18
WETH_SUPPLY_AMOUNT = 10 * 10**18
USDC_BORROW_AMOUNT = 5_000 * 10**6


async def _prepare_whale(fork_w3, pool: Address, weth_amount: int = 0) -> None:
    """Impersonate the whale, give it ETH, and optionally wrap some to WETH."""
    await impersonate(fork_w3, ETH_WHALE)
    await set_balance(fork_w3, ETH_WHALE, WHALE_BALANCE)
    if weth_amount > 0:
        await wrap_eth(fork_w3, ETH_WHALE, WETH.address, weth_amount)
        await erc20_approve(fork_w3, WETH.address, ETH_WHALE, pool, weth_amount)


# ---------------------------------------------------------------------------
# Live read tests (public RPC)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestAaveV3LiveReads:
    """Read-only smoke tests against Ethereum mainnet."""

    async def test_get_reserve_data_usdc(self, eth_w3):
        aave = await AaveV3.from_chain(eth_w3, ChainId.ETHEREUM)
        data = await aave.get_reserve_data(USDC)
        assert data.token == USDC
        assert data.is_active is True
        assert data.borrowing_enabled is True
        assert Decimal(0) <= data.supply_apy < Decimal("0.5"), f"implausible supply APY: {data.supply_apy}"
        assert Decimal(0) <= data.variable_borrow_apy < Decimal("0.5"), (
            f"implausible borrow APY: {data.variable_borrow_apy}"
        )
        assert data.supply_apy <= data.variable_borrow_apy
        assert 0 < data.ltv <= 10_000
        assert data.liquidation_threshold >= data.ltv
        assert len(data.a_token_address) == 20
        assert len(data.variable_debt_token_address) == 20

    async def test_get_reserve_data_weth(self, eth_w3):
        aave = await AaveV3.from_chain(eth_w3, ChainId.ETHEREUM)
        data = await aave.get_reserve_data(WETH)
        assert data.is_active is True
        assert data.usage_as_collateral_enabled is True
        assert data.ltv > 0

    async def test_get_asset_price_usdc(self, eth_w3):
        # Aave V3 mainnet oracle uses USD with 8 decimals.
        aave = await AaveV3.from_chain(eth_w3, ChainId.ETHEREUM)
        price = await aave.get_asset_price(USDC)
        assert 0.95 * 10**8 < price < 1.05 * 10**8, f"USDC oracle price out of range: {price / 1e8}"

    async def test_get_user_account_data_no_debt(self, eth_w3):
        """A user with no debt must have infinite health factor.

        Use an ephemeral address — ``address(0)`` is not safe because
        someone historically called ``supply(..., onBehalfOf=0)`` and
        dumped real collateral into it.
        """
        fresh = Address(secrets.token_bytes(20))
        aave = await AaveV3.from_chain(eth_w3, ChainId.ETHEREUM)
        acc = await aave.get_user_account_data(fresh)
        assert acc.total_debt_base == 0
        assert acc.total_collateral_base == 0
        assert acc.health_factor.is_infinite()


# ---------------------------------------------------------------------------
# Fork tests (Anvil)
# ---------------------------------------------------------------------------

MINIMAL_FLASH_RECEIVER_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IPoolAddressesProvider {
    function getPool() external view returns (address);
}

/// @notice Minimal IFlashLoanSimpleReceiver — approves the Pool to pull
/// (amount + premium) back and records the callback args for assertions.
contract MinimalFlashLoanReceiver {
    address public immutable POOL;
    IPoolAddressesProvider public immutable ADDRESSES_PROVIDER;
    bool public executed;
    uint256 public amountSeen;
    uint256 public premiumSeen;
    bytes public paramsSeen;

    constructor(address pool, address addressesProvider) {
        POOL = pool;
        ADDRESSES_PROVIDER = IPoolAddressesProvider(addressesProvider);
    }

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address /* initiator */,
        bytes calldata params
    ) external returns (bool) {
        require(msg.sender == POOL, "only pool");
        executed = true;
        amountSeen = amount;
        premiumSeen = premium;
        paramsSeen = params;
        IERC20(asset).approve(POOL, amount + premium);
        return true;
    }
}
"""


@pytest.mark.fork
class TestAaveV3Fork:
    """Supply / borrow / repay / withdraw round-trip on an Anvil mainnet fork."""

    async def test_supply_weth_increases_atoken_balance(self, fork_w3):
        """Wrapping ETH and supplying WETH must mint aWETH to the depositor."""
        aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
        await _prepare_whale(fork_w3, aave.pool_address, weth_amount=WETH_SUPPLY_AMOUNT)

        reserve = await aave.get_reserve_data(WETH)
        a_weth = reserve.a_token_address
        a_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=a_weth)

        receipt = await send_tx(
            fork_w3, ETH_WHALE, aave.build_supply_tx(ETH_WHALE, TokenAmount(WETH, WETH_SUPPLY_AMOUNT))
        )
        assert receipt["status"] == 1, "supply() reverted"

        a_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=a_weth)
        # aToken minted 1:1 with supplied principal.
        assert a_after - a_before >= WETH_SUPPLY_AMOUNT - 10
        acc = await aave.get_user_account_data(ETH_WHALE)
        assert acc.total_collateral_base > 0
        assert acc.available_borrows_base > 0

    async def test_supply_then_borrow_then_repay_then_withdraw(self, fork_w3):
        """Full lifecycle: supply WETH → borrow USDC → repay USDC → withdraw WETH."""
        aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
        await _prepare_whale(fork_w3, aave.pool_address, weth_amount=WETH_SUPPLY_AMOUNT)

        # 1. Supply WETH.
        receipt = await send_tx(
            fork_w3, ETH_WHALE, aave.build_supply_tx(ETH_WHALE, TokenAmount(WETH, WETH_SUPPLY_AMOUNT))
        )
        assert receipt["status"] == 1, "supply reverted"

        # 2. Borrow USDC against the WETH collateral.
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        receipt = await send_tx(
            fork_w3,
            ETH_WHALE,
            aave.build_borrow_tx(ETH_WHALE, TokenAmount(USDC, USDC_BORROW_AMOUNT)),
        )
        assert receipt["status"] == 1, "borrow reverted"

        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        assert usdc_after - usdc_before == USDC_BORROW_AMOUNT
        assert (await aave.get_user_reserve_data(ETH_WHALE, USDC)).variable_debt.amount >= USDC_BORROW_AMOUNT

        # 3. Repay the full USDC debt.
        await erc20_approve(fork_w3, USDC.address, ETH_WHALE, aave.pool_address, UINT256_MAX)
        receipt = await send_tx(
            fork_w3,
            ETH_WHALE,
            aave.build_repay_tx(ETH_WHALE, (USDC, "max")),
        )
        assert receipt["status"] == 1, "repay reverted"
        assert (await aave.get_user_reserve_data(ETH_WHALE, USDC)).variable_debt.amount == 0

        # 4. Withdraw all WETH.
        reserve = await aave.get_reserve_data(WETH)
        weth_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=WETH.address)
        receipt = await send_tx(fork_w3, ETH_WHALE, aave.build_withdraw_tx(ETH_WHALE, (WETH, "max")))
        assert receipt["status"] == 1, "withdraw reverted"

        assert await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=reserve.a_token_address) == 0
        weth_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=WETH.address)
        assert weth_after - weth_before >= WETH_SUPPLY_AMOUNT - 10

    async def test_flashloan_simple_round_trip(self, fork_w3):
        """flashLoanSimple must invoke the receiver and pull amount + premium."""
        aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)

        # Deploy a minimal receiver using one of Anvil's pre-funded accounts.
        accounts = await fork_w3.eth.accounts
        deployer = accounts[0]
        compiled = compile_sol_source(MINIMAL_FLASH_RECEIVER_SOL, "MinimalFlashLoanReceiver")
        receiver = await deploy(fork_w3, compiled, deployer, aave.pool_address, AAVE_ADDRESSES_PROVIDER)

        # Pre-fund the receiver with 10 USDC for the premium.
        await impersonate(fork_w3, USDC_WHALE)
        await set_balance(fork_w3, USDC_WHALE, 10**18)
        prefund = 10 * 10**6
        await send_tx(
            fork_w3,
            USDC_WHALE,
            {
                "to": "0x" + bytes(USDC.address).hex(),
                "data": "0x" + ERC20.fns.transfer(receiver, prefund).data.hex(),
                "value": "0",
            },
        )
        receiver_before = await ERC20.fns.balanceOf(receiver).call(fork_w3, to=USDC.address)
        assert receiver_before == prefund

        # Take a 1 000 USDC flash loan.
        loan_amount = 1_000 * 10**6
        params = b"\xca\xfe"
        receipt = await send_tx(
            fork_w3,
            Address(deployer),
            aave.build_flashloan_simple_tx(receiver, TokenAmount(USDC, loan_amount), params=params),
        )
        assert receipt["status"] == 1, "flashLoanSimple reverted"

        probe = fork_w3.eth.contract(address=receiver, abi=compiled["abi"])
        assert await probe.functions.executed().call() is True
        assert await probe.functions.amountSeen().call() == loan_amount
        premium = await probe.functions.premiumSeen().call()
        assert 0 < premium <= loan_amount // 100, f"implausible premium: {premium}"

        # Receiver lost exactly the premium.
        receiver_after = await ERC20.fns.balanceOf(receiver).call(fork_w3, to=USDC.address)
        assert receiver_before - receiver_after == premium

    async def test_emode_round_trip(self, fork_w3):
        """Enabling and disabling an E-Mode category must persist on-chain.

        E-Mode is unconditional with no outstanding debt, so we just toggle
        the category 0 → ETH-correlated → 0 and assert get_user_emode
        tracks each transition.
        """
        aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
        await _prepare_whale(fork_w3, aave.pool_address)
        assert await aave.get_user_emode(ETH_WHALE) == 0

        # Resolve the ETH-correlated category by label scan so the test
        # survives governance id-reshuffles. Aave V3 returns an empty
        # label for unconfigured ids, so we walk the full uint8 range
        # and skip them; getEModeCategoryLabel does not revert. We
        # narrow exception handling to Web3Exception (covers contract
        # reverts, ABI-decode errors and RPC failures from the call)
        # and surface the last one in the failure message — broader
        # `except Exception` would silently swallow regressions like
        # an ABI rename or an assertion bug in get_emode_category.
        eth_category_id = None
        last_error: Web3Exception | None = None
        for cid in range(1, 256):
            try:
                cat = await aave.get_emode_category(cid)
            except Web3Exception as exc:
                last_error = exc
                continue
            if not cat.label:
                continue
            if "ETH" in cat.label.upper():
                eth_category_id = cid
                break
        assert eth_category_id is not None, (
            f"no ETH-correlated E-Mode category found on mainnet (last call error: {last_error!r})"
        )

        receipt = await send_tx(fork_w3, ETH_WHALE, aave.build_set_emode_tx(eth_category_id))
        assert receipt["status"] == 1, "setUserEMode reverted"
        assert await aave.get_user_emode(ETH_WHALE) == eth_category_id

        receipt = await send_tx(fork_w3, ETH_WHALE, aave.build_set_emode_tx(0))
        assert receipt["status"] == 1
        assert await aave.get_user_emode(ETH_WHALE) == 0

    async def test_set_collateral_toggle(self, fork_w3):
        """Disabling WETH as collateral must zero the user's borrowing power."""
        aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
        await _prepare_whale(fork_w3, aave.pool_address, weth_amount=WETH_SUPPLY_AMOUNT)
        receipt = await send_tx(
            fork_w3, ETH_WHALE, aave.build_supply_tx(ETH_WHALE, TokenAmount(WETH, WETH_SUPPLY_AMOUNT))
        )
        assert receipt["status"] == 1

        # Disable.
        receipt = await send_tx(fork_w3, ETH_WHALE, aave.build_set_collateral_tx(WETH, False))
        assert receipt["status"] == 1
        assert (await aave.get_user_reserve_data(ETH_WHALE, WETH)).usage_as_collateral_enabled is False
        acc = await aave.get_user_account_data(ETH_WHALE)
        assert acc.total_collateral_base == 0
        assert acc.available_borrows_base == 0

        # Re-enable.
        receipt = await send_tx(fork_w3, ETH_WHALE, aave.build_set_collateral_tx(WETH, True))
        assert receipt["status"] == 1
        assert (await aave.get_user_reserve_data(ETH_WHALE, WETH)).usage_as_collateral_enabled is True

"""Live integration tests for the Morpho Blue lending module.

* ``@pytest.mark.live`` — read-only calls against a public Ethereum RPC
  (``eth_w3`` fixture). Resolves a real mainnet market by its ``Id`` and
  checks that ``get_market`` / ``get_position`` return plausible values.
* ``@pytest.mark.fork`` — a full supply / borrow / repay / withdraw
  lifecycle against an Anvil mainnet fork (``fork_w3`` fixture). The test
  deploys a constant-price oracle and ``createMarket``s its own WETH / USDC
  market, so it depends on no live market's liquidity or state.

Run live read tests with::

    pytest -m live tests/live/test_morpho_live.py

Run fork write tests with::

    pytest -m fork tests/live/test_morpho_live.py
"""

from __future__ import annotations

import secrets
from decimal import Decimal

import pytest
from eth_contract.erc20 import ERC20

from pydefi.lending import MorphoBlue
from pydefi.lending.morpho import MarketParams
from pydefi.lending.utils import UINT256_MAX
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import ETH_WHALE, MORPHO_IRM, USDC, USDC_WHALE, WETH
from tests.live.anvil_helpers import (
    erc20_approve,
    fund_usdc,
    impersonate,
    send_ok,
    set_balance,
    wrap_eth,
)
from tests.live.sol_utils import compile_sol_source, deploy

#: A real, immutable mainnet Morpho market — cbBTC collateral / USDC loan,
#: 86% LLTV. A market Id is the hash of immutable params, so it never moves.
CBBTC_USDC_MARKET_ID = bytes.fromhex("64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64")

#: An enabled standard LLTV (86%), WAD-scaled.
LLTV_86 = 860_000_000_000_000_000


# ---------------------------------------------------------------------------
# Live read tests (public RPC)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestMorphoLiveReads:
    """Read-only smoke tests against Ethereum mainnet."""

    @pytest.fixture
    def morpho(self, eth_w3) -> MorphoBlue:
        return MorphoBlue.from_chain(eth_w3, ChainId.ETHEREUM)

    @pytest.fixture
    async def params(self, morpho: MorphoBlue) -> MarketParams:
        return await morpho.get_market_params(CBBTC_USDC_MARKET_ID)

    async def test_get_market_params_round_trips_to_id(self, params: MarketParams):
        """Params resolved from an Id must hash back to that same Id."""
        assert params.id == CBBTC_USDC_MARKET_ID
        assert params.lltv == LLTV_86
        assert params.loan_token.address == USDC.address
        assert params.loan_token.decimals == 6

    async def test_get_market_rates_plausible(self, morpho: MorphoBlue, params: MarketParams):
        market = await morpho.get_market(params)
        assert Decimal(0) <= market.borrow_apy < Decimal("0.5"), f"implausible borrow APY: {market.borrow_apy}"
        assert Decimal(0) <= market.supply_apy <= market.borrow_apy
        assert Decimal(0) <= market.utilization <= Decimal(1)
        assert market.liquidity.amount >= 0
        assert market.state.total_borrow_assets <= market.state.total_supply_assets

    async def test_get_position_no_debt(self, morpho: MorphoBlue, params: MarketParams):
        """A fresh address has an empty, infinitely healthy position."""
        pos = await morpho.get_position(Address(secrets.token_bytes(20)), params)
        assert pos.supply_shares == 0
        assert pos.borrow_shares == 0
        assert pos.borrow_assets.amount == 0
        assert pos.health_factor.is_infinite()


# ---------------------------------------------------------------------------
# Fork tests (Anvil)
# ---------------------------------------------------------------------------

MOCK_ORACLE_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal Morpho IOracle returning a constant 1e36-scaled price.
contract MockOracle {
    uint256 public immutable PRICE;

    constructor(uint256 price_) {
        PRICE = price_;
    }

    function price() external view returns (uint256) {
        return PRICE;
    }
}
"""

# WETH (18 dec) priced in USDC (6 dec) at $3000, scaled by 1e36:
# 3000 * 10 ** (36 + 6 - 18).
WETH_USDC_PRICE = 3_000 * 10**24
SUPPLY_USDC = 200_000 * 10**6
COLLATERAL_WETH = 10 * 10**18
BORROW_USDC = 5_000 * 10**6


@pytest.mark.fork
class TestMorphoFork:
    """Supply / borrow / repay / withdraw round-trip on an Anvil mainnet fork."""

    async def test_full_lifecycle(self, fork_w3):
        morpho = MorphoBlue.from_chain(fork_w3, ChainId.ETHEREUM)

        # Deploy a constant-price oracle so the test owns a self-contained
        # market instead of depending on a live market's liquidity.
        deployer = Address((await fork_w3.eth.accounts)[0])
        compiled = compile_sol_source(MOCK_ORACLE_SOL, "MockOracle")
        oracle = await deploy(fork_w3, compiled, deployer, WETH_USDC_PRICE)

        market = MarketParams(
            loan_token=USDC,
            collateral_token=WETH,
            oracle=oracle,
            irm=MORPHO_IRM,
            lltv=LLTV_86,
        )

        # 1. Create the market (permissionless — IRM and LLTV are enabled).
        await send_ok(fork_w3, deployer, morpho.build_create_market_tx(market), "createMarket")

        # 2. A supplier provides USDC liquidity.
        await impersonate(fork_w3, USDC_WHALE)
        await set_balance(fork_w3, USDC_WHALE, 10**18)
        await erc20_approve(fork_w3, USDC.address, USDC_WHALE, morpho.morpho_address, SUPPLY_USDC)
        await send_ok(
            fork_w3,
            USDC_WHALE,
            morpho.build_supply_tx(market, assets=TokenAmount(USDC, SUPPLY_USDC), on_behalf_of=USDC_WHALE),
            "supply",
        )
        snapshot = await morpho.get_market(market)
        assert snapshot.liquidity.amount >= SUPPLY_USDC - 10
        assert snapshot.borrow_apy >= Decimal(0)

        # 3. The borrower posts WETH collateral.
        await impersonate(fork_w3, ETH_WHALE)
        await set_balance(fork_w3, ETH_WHALE, 100 * 10**18)
        await wrap_eth(fork_w3, ETH_WHALE, WETH.address, COLLATERAL_WETH)
        await erc20_approve(fork_w3, WETH.address, ETH_WHALE, morpho.morpho_address, COLLATERAL_WETH)
        await send_ok(
            fork_w3,
            ETH_WHALE,
            morpho.build_supply_collateral_tx(market, TokenAmount(WETH, COLLATERAL_WETH), ETH_WHALE),
            "supplyCollateral",
        )

        # 4. Borrow USDC against the collateral.
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        await send_ok(
            fork_w3,
            ETH_WHALE,
            morpho.build_borrow_tx(
                market, assets=TokenAmount(USDC, BORROW_USDC), on_behalf_of=ETH_WHALE, receiver=ETH_WHALE
            ),
            "borrow",
        )
        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        assert usdc_after - usdc_before == BORROW_USDC

        pos = await morpho.get_position(ETH_WHALE, market)
        assert pos.collateral.amount == COLLATERAL_WETH
        assert pos.borrow_assets.amount >= BORROW_USDC
        # 10 WETH @ $3000 against a $5k borrow at 86% LLTV — comfortably healthy.
        assert pos.health_factor > Decimal(1), f"unexpectedly unhealthy: {pos.health_factor}"

        # 5. Repay the full debt by shares. Fund a small USDC buffer first so
        #    accrued interest beyond the borrowed principal is covered.
        await fund_usdc(fork_w3, USDC.address, ETH_WHALE, 100 * 10**6)
        await impersonate(fork_w3, ETH_WHALE)
        await erc20_approve(fork_w3, USDC.address, ETH_WHALE, morpho.morpho_address, UINT256_MAX)
        await send_ok(
            fork_w3,
            ETH_WHALE,
            morpho.build_repay_tx(market, shares=pos.borrow_shares, on_behalf_of=ETH_WHALE),
            "repay",
        )
        repaid = await morpho.get_position(ETH_WHALE, market)
        assert repaid.borrow_shares == 0
        assert repaid.borrow_assets.amount == 0
        assert repaid.health_factor.is_infinite()

        # 6. Withdraw all collateral now that the debt is cleared.
        await send_ok(
            fork_w3,
            ETH_WHALE,
            morpho.build_withdraw_collateral_tx(market, TokenAmount(WETH, COLLATERAL_WETH), ETH_WHALE, ETH_WHALE),
            "withdrawCollateral",
        )
        final = await morpho.get_position(ETH_WHALE, market)
        assert final.collateral.amount == 0

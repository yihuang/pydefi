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
from tests.addrs import ETH_WHALE, LLTV_86, MORPHO_IRM, USDC, WETH
from tests.live.anvil_helpers import (
    erc20_approve,
    fund_usdc,
    impersonate,
    seed_morpho_market,
    send_ok,
)
from tests.live.sol_utils import deploy_mutable_oracle

#: A real, immutable mainnet Morpho market — cbBTC collateral / USDC loan,
#: 86% LLTV. A market Id is the hash of immutable params, so it never moves.
CBBTC_USDC_MARKET_ID = bytes.fromhex("64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64")


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
        # market instead of depending on a live market's liquidity, then
        # create + seed it: whale supply, borrower collateral, borrow.
        deployer = Address((await fork_w3.eth.accounts)[0])
        oracle = await deploy_mutable_oracle(fork_w3, deployer, WETH_USDC_PRICE)
        market = MarketParams(
            loan_token=USDC,
            collateral_token=WETH,
            oracle=oracle,
            irm=MORPHO_IRM,
            lltv=LLTV_86,
        )
        usdc_before = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        await seed_morpho_market(
            fork_w3, morpho, market, deployer, supply=SUPPLY_USDC, collateral=COLLATERAL_WETH, borrow=BORROW_USDC
        )

        # The borrow proceeds arrived, and the market accounts for both legs.
        usdc_after = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        assert usdc_after - usdc_before == BORROW_USDC
        snapshot = await morpho.get_market(market)
        assert snapshot.liquidity.amount >= SUPPLY_USDC - BORROW_USDC - 10
        assert snapshot.borrow_apy >= Decimal(0)

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

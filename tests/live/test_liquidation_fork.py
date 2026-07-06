"""Live fork test for the liquidation router against a real underwater position.

``@pytest.mark.fork`` — builds a self-contained Morpho Blue market on an Anvil
mainnet fork, drops a mutable-price oracle to push a borrower underwater, and
verifies :class:`~pydefi.lending.liquidation.LiquidationRouter` discovers and
liquidates the position end-to-end. Morpho is used because its caller-chosen
oracle moves deterministically on a fork — Aave / Compound Chainlink feeds
don't.

Run with::

    pytest -m fork tests/live/test_liquidation_fork.py
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from eth_contract import Contract
from eth_contract.erc20 import ERC20

from pydefi._utils import to_tx
from pydefi.lending import MorphoBlue
from pydefi.lending.liquidation import LiquidationRouter, MorphoCandidate
from pydefi.lending.morpho import MarketParams
from pydefi.lending.utils import UINT256_MAX
from pydefi.types import Address, ChainId
from tests.addrs import ETH_WHALE, LLTV_86, MORPHO_IRM, USDC, WETH
from tests.live.anvil_helpers import erc20_approve, fund_usdc, seed_morpho_market, send_ok
from tests.live.sol_utils import deploy_mutable_oracle

_ORACLE = Contract.from_abi(["function setPrice(uint256 newPrice) external"])

# WETH (18 dec) priced in USDC (6 dec), 1e36-scaled: usd * 10 ** (36 + 6 - 18).
_PRICE_SCALE = 10**24
# $3000/WETH: 10 WETH collateral against a $20k borrow at 86% LLTV is healthy.
HEALTHY_PRICE = 3_000 * _PRICE_SCALE
# $2200/WETH: collateral ($22k) still exceeds debt * liquidation-incentive
# (~$20.9k), so the position is liquidatable without tipping into bad debt —
# a full-debt liquidation seizes ~9.5 WETH and leaves the rest.
UNDERWATER_PRICE = 2_200 * _PRICE_SCALE

SUPPLY_USDC = 200_000 * 10**6
COLLATERAL_WETH = 10 * 10**18
BORROW_USDC = 20_000 * 10**6
LIQUIDATOR_USDC = 25_000 * 10**6  # covers the debt plus accrued interest


@pytest.mark.fork
class TestLiquidationRouterFork:
    """LiquidationRouter discovery + execution on an Anvil mainnet fork."""

    async def test_finds_and_liquidates_underwater_morpho_position(self, fork_w3):
        morpho = MorphoBlue.from_chain(fork_w3, ChainId.ETHEREUM)
        deployer = Address((await fork_w3.eth.accounts)[0])
        liquidator = Address((await fork_w3.eth.accounts)[1])

        # A self-contained WETH-collateral / USDC-loan market around a
        # mutable-price oracle, seeded with a borrower healthy at $3000.
        oracle = await deploy_mutable_oracle(fork_w3, deployer, HEALTHY_PRICE)
        market = MarketParams(
            loan_token=USDC,
            collateral_token=WETH,
            oracle=oracle,
            irm=MORPHO_IRM,
            lltv=LLTV_86,
        )
        await seed_morpho_market(
            fork_w3, morpho, market, deployer, supply=SUPPLY_USDC, collateral=COLLATERAL_WETH, borrow=BORROW_USDC
        )

        router = LiquidationRouter({ChainId.ETHEREUM: fork_w3}, liquidator)
        candidate = MorphoCandidate(ChainId.ETHEREUM, market, ETH_WHALE)

        # While the position is healthy the router must surface nothing.
        healthy = await morpho.get_position(ETH_WHALE, market)
        assert healthy.health_factor > Decimal(1), f"expected healthy, got HF {healthy.health_factor}"
        assert await router.find_liquidatable_positions([candidate]) == []

        # Drop the oracle price — the position falls below a health factor of 1.
        await send_ok(
            fork_w3,
            deployer,
            to_tx(oracle, _ORACLE.fns.setPrice(UNDERWATER_PRICE).data),
            "setPrice",
        )
        underwater = await morpho.get_position(ETH_WHALE, market)
        assert underwater.health_factor < Decimal(1), f"expected underwater, got HF {underwater.health_factor}"

        # The router now discovers exactly one liquidation opportunity.
        opps = await router.find_liquidatable_positions([candidate])
        assert len(opps) == 1
        opp = opps[0]
        assert opp.protocol == "morpho"
        assert opp.borrower == ETH_WHALE
        assert opp.health_factor is not None and opp.health_factor < Decimal(1)
        assert opp.collateral_to_seize == WETH
        assert opp.debt_to_repay.amount >= BORROW_USDC

        # Fund and approve the liquidator, then execute the built transaction.
        await fund_usdc(fork_w3, USDC.address, liquidator, LIQUIDATOR_USDC)
        await erc20_approve(fork_w3, USDC.address, liquidator, morpho.morpho_address, UINT256_MAX)

        weth_before = await ERC20.fns.balanceOf(liquidator).call(fork_w3, to=WETH.address)
        usdc_before = await ERC20.fns.balanceOf(liquidator).call(fork_w3, to=USDC.address)
        await send_ok(fork_w3, liquidator, opp.tx, "liquidate")
        weth_after = await ERC20.fns.balanceOf(liquidator).call(fork_w3, to=WETH.address)
        usdc_after = await ERC20.fns.balanceOf(liquidator).call(fork_w3, to=USDC.address)

        # The liquidator seized WETH collateral and spent USDC repaying the debt.
        assert weth_after > weth_before, "liquidator received no seized collateral"
        assert usdc_after < usdc_before, "liquidator spent no USDC on the repayment"

        # The borrower's debt is fully cleared (the opportunity repays every
        # borrow share); part of the collateral was seized, the rest remains,
        # and the now debt-free position reads as infinitely healthy.
        cleared = await morpho.get_position(ETH_WHALE, market)
        assert cleared.borrow_shares == 0
        assert cleared.borrow_assets.amount == 0
        assert 0 < cleared.collateral.amount < COLLATERAL_WETH
        assert cleared.health_factor.is_infinite()

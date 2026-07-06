"""Unit tests for pydefi.lending.liquidation (no live node required).

The protocol clients are mocked — these cover the router's dispatch,
health-factor filtering, opportunity construction and graceful per-candidate
error handling, not real RPC behaviour.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.lending import liquidation
from pydefi.lending.liquidation import (
    AaveV3Candidate,
    AaveV4Candidate,
    CompoundV3Candidate,
    LiquidationOpportunity,
    LiquidationRouter,
    MorphoCandidate,
)
from pydefi.lending.morpho import MarketParams
from pydefi.types import TokenAmount
from tests.addrs import ETH_WHALE, USDC, WETH

# A WETH-collateral / USDC-loan Morpho market (params, not on-chain state).
MARKET = MarketParams(
    loan_token=USDC,
    collateral_token=WETH,
    oracle=ETH_WHALE,  # any address — these tests never read the oracle
    irm=ETH_WHALE,
    lltv=860_000_000_000_000_000,
)

BORROWER = ETH_WHALE
LIQUIDATOR = USDC.address  # any address; reused as a stand-in liquidator
STUB_TX = {"to": "0xc0ffee", "data": "0xdead", "value": "0"}

#: w3s map whose values are never dereferenced — every from_chain is patched.
W3S = {1: MagicMock(name="w3")}


@contextmanager
def _router_for(client_cls: type, client: MagicMock):
    """Patch *client_cls*.from_chain → *client* and yield a router on chain 1.

    AaveV3.from_chain is awaited by the router (so an AsyncMock); the others
    are plain registry lookups.
    """
    factory = AsyncMock(return_value=client) if client_cls is liquidation.AaveV3 else MagicMock(return_value=client)
    with patch.object(client_cls, "from_chain", new=factory):
        yield LiquidationRouter(W3S, LIQUIDATOR)


def _opportunity(protocol: str = "morpho", chain_id: int = 1, tx: dict | None = None) -> LiquidationOpportunity:
    """A liquidation opportunity with placeholder fields, for plumbing tests."""
    return LiquidationOpportunity(
        protocol=protocol,  # type: ignore[arg-type]
        chain_id=chain_id,
        borrower=BORROWER,
        health_factor=Decimal("0.5"),
        debt_to_repay=TokenAmount(USDC, 1),
        collateral_to_seize=WETH,
        tx=tx if tx is not None else STUB_TX,
    )


def _v3_debt_reserve(amount: int = 7000) -> MagicMock:
    return MagicMock(variable_debt=TokenAmount(USDC, amount))


def _v3_collateral_reserve(amount: int = 5 * 10**18, *, enabled: bool = True) -> MagicMock:
    return MagicMock(a_token_balance=TokenAmount(WETH, amount), usage_as_collateral_enabled=enabled)


def _v4_debt_reserve(amount: int = 3000) -> MagicMock:
    return MagicMock(debt=TokenAmount(USDC, amount))


def _v4_collateral_reserve(amount: int = 5 * 10**18, *, enabled: bool = True) -> MagicMock:
    return MagicMock(supplied=TokenAmount(WETH, amount), using_as_collateral=enabled)


#: A unit oracle price (1e36-scaled): one collateral token is worth one loan token.
MORPHO_UNIT_PRICE = 10**36


def _morpho_position(
    *,
    health_factor: Decimal,
    borrow_shares: int = 999,
    borrow_assets: int = 5000,
    collateral: int = 10**24,
    collateral_price: int = MORPHO_UNIT_PRICE,
) -> MagicMock:
    """Stand-in MorphoPosition. Defaults are solvent-underwater (collateral far
    exceeds ``borrow_assets * incentive``, so the full debt repays by shares);
    shrink *collateral* to model bad debt and force a seize-all."""
    return MagicMock(
        market=MARKET,
        health_factor=health_factor,
        borrow_shares=borrow_shares,
        borrow_assets=TokenAmount(USDC, borrow_assets),
        collateral=TokenAmount(WETH, collateral),
        collateral_price=collateral_price,
    )


# ---------------------------------------------------------------------------
# LiquidationOpportunity
# ---------------------------------------------------------------------------


class TestLiquidationOpportunity:
    def test_is_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            _opportunity().tx = {"other": "tx"}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-protocol checks
# ---------------------------------------------------------------------------


class TestCheckMorpho:
    @pytest.mark.asyncio
    async def test_unhealthy_returns_opportunity(self):
        morpho = MagicMock(
            get_position=AsyncMock(return_value=_morpho_position(health_factor=Decimal("0.8"))),
            build_liquidate_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.MorphoBlue, morpho) as router:
            opp = await router.check_morpho(MorphoCandidate(1, MARKET, BORROWER))

        assert opp is not None
        assert opp.protocol == "morpho"
        assert opp.health_factor == Decimal("0.8")
        assert opp.debt_to_repay == TokenAmount(USDC, 5000)
        assert opp.collateral_to_seize == WETH
        assert opp.tx is STUB_TX
        # A solvent-underwater position repays the whole debt by shares.
        morpho.build_liquidate_tx.assert_called_once_with(MARKET, BORROWER, seized_assets=None, repaid_shares=999)

    @pytest.mark.asyncio
    async def test_bad_debt_position_seizes_all_collateral(self):
        """Full-debt repayment would over-seize → seize the whole collateral
        balance instead of reverting on-chain."""
        morpho = MagicMock(
            get_position=AsyncMock(return_value=_morpho_position(health_factor=Decimal("0.6"), collateral=1000)),
            build_liquidate_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.MorphoBlue, morpho) as router:
            opp = await router.check_morpho(MorphoCandidate(1, MARKET, BORROWER))

        assert opp is not None
        morpho.build_liquidate_tx.assert_called_once_with(
            MARKET, BORROWER, seized_assets=TokenAmount(WETH, 1000), repaid_shares=None
        )

    @pytest.mark.asyncio
    async def test_healthy_returns_none(self):
        morpho = MagicMock(get_position=AsyncMock(return_value=MagicMock(health_factor=Decimal("1.5"))))
        with _router_for(liquidation.MorphoBlue, morpho) as router:
            assert await router.check_morpho(MorphoCandidate(1, MARKET, BORROWER)) is None

    @pytest.mark.asyncio
    async def test_health_factor_exactly_one_is_not_liquidatable(self):
        """HF == 1 is healthy — liquidation needs HF < 1."""
        morpho = MagicMock(get_position=AsyncMock(return_value=MagicMock(health_factor=Decimal("1"))))
        with _router_for(liquidation.MorphoBlue, morpho) as router:
            assert await router.check_morpho(MorphoCandidate(1, MARKET, BORROWER)) is None


class TestCheckAaveV3:
    @pytest.mark.asyncio
    async def test_unhealthy_returns_opportunity(self):
        aave = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.95"))),
            # get_user_reserve_data is gathered debt-first, then collateral.
            get_user_reserve_data=AsyncMock(side_effect=[_v3_debt_reserve(7000), _v3_collateral_reserve()]),
            build_liquidation_call_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.AaveV3, aave) as router:
            opp = await router.check_aave_v3(AaveV3Candidate(1, BORROWER, WETH, USDC))

        assert opp is not None
        assert opp.protocol == "aave_v3"
        assert opp.health_factor == Decimal("0.95")
        assert opp.debt_to_repay == TokenAmount(USDC, 7000)
        assert opp.collateral_to_seize == WETH
        assert opp.tx is STUB_TX
        aave.build_liquidation_call_tx.assert_called_once_with(WETH, (USDC, "max"), BORROWER, receive_atoken=False)

    @pytest.mark.asyncio
    async def test_receive_atoken_flag_is_forwarded(self):
        aave = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.9"))),
            get_user_reserve_data=AsyncMock(side_effect=[_v3_debt_reserve(1), _v3_collateral_reserve()]),
            build_liquidation_call_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.AaveV3, aave) as router:
            await router.check_aave_v3(AaveV3Candidate(1, BORROWER, WETH, USDC, receive_atoken=True))
        aave.build_liquidation_call_tx.assert_called_once_with(WETH, (USDC, "max"), BORROWER, receive_atoken=True)

    @pytest.mark.asyncio
    async def test_healthy_returns_none(self):
        aave = MagicMock(get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("Infinity"))))
        with _router_for(liquidation.AaveV3, aave) as router:
            assert await router.check_aave_v3(AaveV3Candidate(1, BORROWER, WETH, USDC)) is None

    @pytest.mark.asyncio
    async def test_client_resolved_once_per_chain(self):
        """AaveV3.from_chain reads Pool/DataProvider/Oracle on-chain; the router
        memoizes the client so a batch of same-chain candidates resolves it once
        even when the checks run concurrently."""
        aave = MagicMock(get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("2"))))
        from_chain = AsyncMock(return_value=aave)
        with patch.object(liquidation.AaveV3, "from_chain", new=from_chain):
            router = LiquidationRouter(W3S, LIQUIDATOR)
            await router.find_liquidatable_positions(
                [AaveV3Candidate(1, BORROWER, WETH, USDC), AaveV3Candidate(1, BORROWER, WETH, USDC)]
            )
        from_chain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_resolve_is_evicted_and_retried(self):
        """A failed from_chain must not stay cached — the next call retries."""
        client = MagicMock(name="AaveV3")
        from_chain = AsyncMock(side_effect=[RuntimeError("rpc down"), client])
        with patch.object(liquidation.AaveV3, "from_chain", new=from_chain):
            router = LiquidationRouter(W3S, LIQUIDATOR)
            with pytest.raises(RuntimeError, match="rpc down"):
                await router._aave_v3_client(1)
            assert await router._aave_v3_client(1) is client
        assert from_chain.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_resolve_evicts_only_its_own_task(self):
        """A late awaiter of a failed resolve must not evict the fresh task a
        concurrent retry cached in the meantime — only its own."""
        router = LiquidationRouter(W3S, LIQUIDATOR)
        loop = asyncio.get_running_loop()
        failing, replacement = loop.create_future(), loop.create_future()
        router._aave_v3[1] = failing
        awaiter = asyncio.ensure_future(router._aave_v3_client(1))
        await asyncio.sleep(0)  # the awaiter is now suspended on `failing`
        router._aave_v3[1] = replacement  # a concurrent retry raced in
        failing.set_exception(RuntimeError("resolve failed"))
        with pytest.raises(RuntimeError, match="resolve failed"):
            await awaiter
        assert router._aave_v3[1] is replacement

    @pytest.mark.asyncio
    async def test_rejects_pair_with_no_debt(self):
        """No debt in candidate.debt → reject (liquidationCall would revert)."""
        aave = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.9"))),
            get_user_reserve_data=AsyncMock(side_effect=[_v3_debt_reserve(0), _v3_collateral_reserve()]),
        )
        with _router_for(liquidation.AaveV3, aave) as router, pytest.raises(ValueError, match="carries no .* debt"):
            await router.check_aave_v3(AaveV3Candidate(1, BORROWER, WETH, USDC))

    @pytest.mark.asyncio
    async def test_rejects_pair_with_disabled_collateral(self):
        """Collateral not enabled → reject (liquidationCall would seize nothing)."""
        aave = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.9"))),
            get_user_reserve_data=AsyncMock(
                side_effect=[_v3_debt_reserve(7000), _v3_collateral_reserve(enabled=False)]
            ),
        )
        with (
            _router_for(liquidation.AaveV3, aave) as router,
            pytest.raises(ValueError, match="no .* collateral enabled"),
        ):
            await router.check_aave_v3(AaveV3Candidate(1, BORROWER, WETH, USDC))


class TestCheckAaveV4:
    @pytest.mark.asyncio
    async def test_unhealthy_returns_opportunity(self):
        v4 = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.99"))),
            # get_user_reserve is gathered debt-first, then collateral.
            get_user_reserve=AsyncMock(side_effect=[_v4_debt_reserve(3000), _v4_collateral_reserve()]),
            build_liquidation_call_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.AaveV4, v4) as router:
            opp = await router.check_aave_v4(AaveV4Candidate(1, "MAIN_SPOKE", BORROWER, 0, 7))

        assert opp is not None
        assert opp.protocol == "aave_v4"
        assert opp.debt_to_repay == TokenAmount(USDC, 3000)
        assert opp.collateral_to_seize == WETH
        assert opp.tx is STUB_TX
        v4.build_liquidation_call_tx.assert_called_once_with(
            0, 7, BORROWER, TokenAmount(USDC, 3000), receive_shares=False
        )

    @pytest.mark.asyncio
    async def test_healthy_returns_none(self):
        v4 = MagicMock(get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("1.2"))))
        with _router_for(liquidation.AaveV4, v4) as router:
            assert await router.check_aave_v4(AaveV4Candidate(1, "MAIN_SPOKE", BORROWER, 0, 7)) is None

    @pytest.mark.asyncio
    async def test_rejects_reserve_with_no_debt(self):
        """No debt in the chosen reserve → reject (liquidationCall would revert)."""
        v4 = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.9"))),
            get_user_reserve=AsyncMock(side_effect=[_v4_debt_reserve(0), _v4_collateral_reserve()]),
        )
        with (
            _router_for(liquidation.AaveV4, v4) as router,
            pytest.raises(ValueError, match="carries no debt in reserve"),
        ):
            await router.check_aave_v4(AaveV4Candidate(1, "MAIN_SPOKE", BORROWER, 0, 7))

    @pytest.mark.asyncio
    async def test_rejects_reserve_with_disabled_collateral(self):
        """Chosen collateral reserve not enabled → reject (would seize nothing)."""
        v4 = MagicMock(
            get_user_account_data=AsyncMock(return_value=MagicMock(health_factor=Decimal("0.9"))),
            get_user_reserve=AsyncMock(side_effect=[_v4_debt_reserve(3000), _v4_collateral_reserve(enabled=False)]),
        )
        with (
            _router_for(liquidation.AaveV4, v4) as router,
            pytest.raises(ValueError, match="no collateral enabled in reserve"),
        ):
            await router.check_aave_v4(AaveV4Candidate(1, "MAIN_SPOKE", BORROWER, 0, 7))


class TestCheckCompoundV3:
    @pytest.mark.asyncio
    async def test_liquidatable_returns_opportunity(self):
        comet = MagicMock(
            get_user_position=AsyncMock(
                return_value=MagicMock(is_liquidatable=True, base_borrow=TokenAmount(USDC, 2500))
            ),
            build_absorb_tx=MagicMock(return_value=STUB_TX),
        )
        with _router_for(liquidation.CompoundV3, comet) as router:
            opp = await router.check_compound_v3(CompoundV3Candidate(1, "USDC", BORROWER))

        assert opp is not None
        assert opp.protocol == "compound_v3"
        assert opp.health_factor is None  # Comet exposes only a boolean
        assert opp.debt_to_repay == TokenAmount(USDC, 2500)
        assert opp.collateral_to_seize is None  # absorb seizes every collateral
        assert opp.tx is STUB_TX
        # The router's liquidator is the absorber credited with reward points.
        comet.build_absorb_tx.assert_called_once_with(LIQUIDATOR, [BORROWER])

    @pytest.mark.asyncio
    async def test_not_liquidatable_returns_none(self):
        comet = MagicMock(get_user_position=AsyncMock(return_value=MagicMock(is_liquidatable=False)))
        with _router_for(liquidation.CompoundV3, comet) as router:
            assert await router.check_compound_v3(CompoundV3Candidate(1, "USDC", BORROWER)) is None


# ---------------------------------------------------------------------------
# Router plumbing
# ---------------------------------------------------------------------------


class TestRouterPlumbing:
    def test_missing_w3_raises_in_single_check(self):
        router = LiquidationRouter({}, LIQUIDATOR)
        with pytest.raises(ValueError, match="no AsyncWeb3 configured for chain 1"):
            router._w3(1)

    @pytest.mark.asyncio
    async def test_find_filters_healthy_and_dispatches_by_type(self):
        router = LiquidationRouter(W3S, LIQUIDATOR)
        unhealthy = _opportunity()
        router.check_morpho = AsyncMock(return_value=unhealthy)
        router.check_aave_v3 = AsyncMock(return_value=None)  # healthy → dropped
        router.check_compound_v3 = AsyncMock(return_value=None)

        candidates = [
            MorphoCandidate(1, MARKET, BORROWER),
            AaveV3Candidate(1, BORROWER, WETH, USDC),
            CompoundV3Candidate(1, "USDC", BORROWER),
        ]
        opps = await router.find_liquidatable_positions(candidates)

        assert opps == [unhealthy]
        router.check_morpho.assert_awaited_once()
        router.check_aave_v3.assert_awaited_once()
        router.check_compound_v3.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_find_preserves_input_order(self):
        router = LiquidationRouter(W3S, LIQUIDATOR)
        first, second = _opportunity(tx={"n": 1}), _opportunity(tx={"n": 2})
        router.check_morpho = AsyncMock(side_effect=[first, second])
        opps = await router.find_liquidatable_positions(
            [MorphoCandidate(1, MARKET, BORROWER), MorphoCandidate(1, MARKET, BORROWER)]
        )
        assert opps == [first, second]

    @pytest.mark.asyncio
    async def test_find_skips_candidate_on_unconfigured_chain(self):
        router = LiquidationRouter(W3S, LIQUIDATOR)  # only chain 1
        router.check_morpho = AsyncMock(return_value=None)
        opps = await router.find_liquidatable_positions([MorphoCandidate(999, MARKET, BORROWER)])
        assert opps == []
        router.check_morpho.assert_not_awaited()  # never dispatched

    @pytest.mark.asyncio
    async def test_find_downgrades_per_candidate_failure_to_skip(self):
        router = LiquidationRouter(W3S, LIQUIDATOR)
        router.check_morpho = AsyncMock(side_effect=RuntimeError("rpc down"))
        # One bad candidate must not sink the batch — it is logged and skipped.
        opps = await router.find_liquidatable_positions([MorphoCandidate(1, MARKET, BORROWER)])
        assert opps == []

    @pytest.mark.asyncio
    async def test_check_rejects_unknown_candidate_type(self):
        router = LiquidationRouter(W3S, LIQUIDATOR)
        with pytest.raises(TypeError, match="unknown candidate type"):
            await router._check(object())  # type: ignore[arg-type]

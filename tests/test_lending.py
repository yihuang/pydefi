"""Unit tests for pydefi.lending (no live node required).

Covers every protocol under the lending umbrella:

* Aave V3 — pool-per-reserve model with aTokens, health factors, E-Mode.
* Compound V3 (Comet) — one-base-asset-per-market model where supply /
  withdraw also handle borrow / repay through the same dispatch.
* Morpho Blue — isolated markets keyed by ``MarketParams``.
* Aave V4 — Hub-and-Spoke; reserves keyed by ``reserveId`` within a Spoke.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hexbytes import HexBytes

from pydefi.abi.lending import (
    AAVE_ADAPTER,
    AAVE_V3_POOL,
    AAVE_V4_SPOKE,
    AAVE_V4_TOKENIZATION_SPOKE,
    COMPOUND_V3_COMET,
    MORPHO_BLUE,
)
from pydefi.deployments import chains_for, comet_contract_names, get_address, get_token
from pydefi.exceptions import PydefiError
from pydefi.lending import AaveV3, BabylonAaveAdapter, UserAccountData, aave_v4
from pydefi.lending.aave_v3 import (
    RAY,
    ray_rate_to_apy,
)
from pydefi.lending.compound_v3 import (
    COMET_SCALE,
    CompoundV3,
    per_second_rate_to_apy,
)
from pydefi.lending.morpho import (
    MAX_LIQUIDATION_INCENTIVE_FACTOR,
    ORACLE_PRICE_SCALE,
    MarketParams,
    MarketState,
    MorphoBlue,
    MorphoPosition,
    accrue_interest,
    compute_health_factor,
    liquidation_incentive_factor,
    max_liquidation,
    supply_apy_from_borrow,
)
from pydefi.lending.utils import (
    SECONDS_PER_YEAR,
    UINT256_MAX,
    WAD,
    parse_health_factor,
)
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import ETH_WHALE, LLTV_86, USDC, WETH

if TYPE_CHECKING:
    from eth_contract.contract import ContractFunction


def decode_call(tx: dict, fn: ContractFunction) -> tuple[Any, ...]:
    """Decode a pydefi tx dict's calldata against *fn*'s input types.

    Address-typed arguments come back as :class:`Address` so callers can
    compare them directly with :attr:`Token.address`.
    """
    raw = fn.decode_input(HexBytes(tx["data"]))
    # decode_input unwraps single-input functions; re-wrap so callers
    # can always tuple-unpack.
    if not isinstance(raw, tuple):
        raw = (raw,)
    return tuple(Address(v) if t == "address" else v for v, t in zip(raw, fn.input_types))


# ===========================================================================
# Aave V3
# ===========================================================================

AAVE_POOL_ADDR = Address("0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2")
AAVE_DATA_PROVIDER_ADDR = Address("0x0a16f2FCC0D44FaE41cc54e079281D84A363bECD")


def _apy_to_annual_ray(target_apy: Decimal) -> int:
    """Invert :func:`ray_rate_to_apy` for round-trip testing."""
    per_second = (Decimal(1) + target_apy) ** (Decimal(1) / Decimal(SECONDS_PER_YEAR)) - Decimal(1)
    return int(per_second * Decimal(SECONDS_PER_YEAR) * Decimal(RAY))


# ---------------------------------------------------------------------------
# Rate model
# ---------------------------------------------------------------------------


class TestAaveRateModel:
    def test_zero_rate(self):
        assert ray_rate_to_apy(0) == Decimal(0)

    @pytest.mark.parametrize("target,tolerance", [(Decimal("0.03"), "0.0001"), (Decimal("0.20"), "0.0005")])
    def test_round_trip(self, target: Decimal, tolerance: str):
        apy = ray_rate_to_apy(_apy_to_annual_ray(target))
        assert abs(apy - target) < Decimal(tolerance)

    def test_realistic_aave_rate(self):
        # Observed Aave V3 mainnet USDC supply rate ≈ 3.37 % APR.
        apy = ray_rate_to_apy(33_704_216_600_279_803_939_752_544)
        assert Decimal("0.030") < apy < Decimal("0.040")


# ---------------------------------------------------------------------------
# Health-factor parsing
# ---------------------------------------------------------------------------


class TestAaveHealthFactor:
    def test_finite(self):
        assert parse_health_factor(15 * 10**17) == Decimal("1.5")

    def test_infinite_when_no_debt(self):
        assert parse_health_factor(UINT256_MAX).is_infinite()

    def test_user_account_data_construction(self):
        data = UserAccountData(
            total_collateral_base=1_000 * 10**8,
            total_debt_base=500 * 10**8,
            available_borrows_base=250 * 10**8,
            current_liquidation_threshold=8500,
            ltv=7500,
            health_factor=Decimal("1.5"),
        )
        assert data.health_factor == Decimal("1.5")
        assert data.current_liquidation_threshold == 8500


# ---------------------------------------------------------------------------
# Tx-builder calldata encoding
# ---------------------------------------------------------------------------


@pytest.fixture
def aave() -> AaveV3:
    return AaveV3(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.ETHEREUM,
        pool_address=AAVE_POOL_ADDR,
        data_provider_address=AAVE_DATA_PROVIDER_ADDR,
    )


class TestAaveBuildSupplyTx:
    def test_to_and_value(self, aave: AaveV3):
        tx = aave.build_supply_tx(ETH_WHALE, TokenAmount.from_human(USDC, "1000"))
        assert Address(tx["to"]) == AAVE_POOL_ADDR
        assert tx["value"] == "0"

    def test_args_encoded(self, aave: AaveV3):
        amount = TokenAmount.from_human(USDC, "1000")
        tx = aave.build_supply_tx(ETH_WHALE, amount, referral_code=42)
        asset, raw_amount, on_behalf_of, referral = decode_call(tx, AAVE_V3_POOL.fns.supply)
        assert asset == USDC.address
        assert raw_amount == amount.amount
        assert on_behalf_of == ETH_WHALE
        assert referral == 42


class TestAaveBuildWithdrawTx:
    def test_explicit_amount(self, aave: AaveV3):
        amount = TokenAmount.from_human(USDC, "500")
        asset, raw_amount, _to = decode_call(aave.build_withdraw_tx(ETH_WHALE, amount), AAVE_V3_POOL.fns.withdraw)
        assert asset == USDC.address
        assert raw_amount == amount.amount

    def test_max_amount(self, aave: AaveV3):
        _asset, raw_amount, _to = decode_call(
            aave.build_withdraw_tx(ETH_WHALE, (USDC, "max")), AAVE_V3_POOL.fns.withdraw
        )
        assert raw_amount == UINT256_MAX

    def test_rejects_invalid_amount(self, aave: AaveV3):
        with pytest.raises(TypeError):
            aave.build_withdraw_tx(ETH_WHALE, "1000")  # type: ignore[arg-type]


class TestAaveBuildBorrowTx:
    def test_encodes_variable_rate_mode(self, aave: AaveV3):
        amount = TokenAmount.from_human(USDC, "100")
        _asset, _amount, mode, _referral, _on_behalf_of = decode_call(
            aave.build_borrow_tx(ETH_WHALE, amount), AAVE_V3_POOL.fns.borrow
        )
        assert mode == 2  # Aave V3 variable-rate mode

    def test_uses_on_behalf_of_override(self, aave: AaveV3):
        other = Address("0x" + "ab" * 20)
        _asset, _amount, _mode, _referral, on_behalf_of = decode_call(
            aave.build_borrow_tx(ETH_WHALE, TokenAmount.from_human(USDC, "100"), on_behalf_of=other),
            AAVE_V3_POOL.fns.borrow,
        )
        assert on_behalf_of == other


class TestAaveBuildRepayTx:
    def test_explicit_amount(self, aave: AaveV3):
        amount = TokenAmount.from_human(USDC, "100")
        asset, raw_amount, _mode, _on_behalf_of = decode_call(
            aave.build_repay_tx(ETH_WHALE, amount), AAVE_V3_POOL.fns.repay
        )
        assert asset == USDC.address
        assert raw_amount == amount.amount

    def test_max_amount(self, aave: AaveV3):
        _asset, raw_amount, _mode, _on_behalf_of = decode_call(
            aave.build_repay_tx(ETH_WHALE, (USDC, "max")), AAVE_V3_POOL.fns.repay
        )
        assert raw_amount == UINT256_MAX


class TestAaveBuildSetCollateralTx:
    @pytest.mark.parametrize("flag", [True, False])
    def test_flag(self, aave: AaveV3, flag: bool):
        asset, use_as_collateral = decode_call(
            aave.build_set_collateral_tx(WETH, flag),
            AAVE_V3_POOL.fns.setUserUseReserveAsCollateral,
        )
        assert asset == WETH.address
        assert use_as_collateral is flag


class TestAaveBuildSetEmodeTx:
    @pytest.mark.parametrize("category_id", [0, 1, 255])
    def test_valid(self, aave: AaveV3, category_id: int):
        (decoded_id,) = decode_call(aave.build_set_emode_tx(category_id), AAVE_V3_POOL.fns.setUserEMode)
        assert decoded_id == category_id

    def test_rejects_out_of_range(self, aave: AaveV3):
        with pytest.raises(ValueError):
            aave.build_set_emode_tx(256)


class TestAaveBuildFlashLoanSimpleTx:
    def test_args_encoded(self, aave: AaveV3):
        receiver = Address("0x" + "cd" * 20)
        amount = TokenAmount.from_human(USDC, "1000")
        params = b"\xde\xad\xbe\xef"
        decoded_receiver, asset, raw_amount, decoded_params, referral = decode_call(
            aave.build_flashloan_simple_tx(receiver, amount, params=params, referral_code=7),
            AAVE_V3_POOL.fns.flashLoanSimple,
        )
        assert decoded_receiver == receiver
        assert asset == USDC.address
        assert raw_amount == amount.amount
        assert decoded_params == params
        assert referral == 7


class TestAaveBuildLiquidationCallTx:
    def test_explicit_amount(self, aave: AaveV3):
        debt_to_cover = TokenAmount.from_human(USDC, "500")
        tx = aave.build_liquidation_call_tx(WETH, debt_to_cover, ETH_WHALE, receive_atoken=True)
        assert Address(tx["to"]) == AAVE_POOL_ADDR
        collateral, debt, user, raw_amount, receive_atoken = decode_call(tx, AAVE_V3_POOL.fns.liquidationCall)
        assert collateral == WETH.address
        assert debt == USDC.address
        assert user == ETH_WHALE
        assert raw_amount == debt_to_cover.amount
        assert receive_atoken is True

    def test_max_amount(self, aave: AaveV3):
        collateral, debt, _user, raw_amount, receive_atoken = decode_call(
            aave.build_liquidation_call_tx(WETH, (USDC, "max"), ETH_WHALE),
            AAVE_V3_POOL.fns.liquidationCall,
        )
        assert collateral == WETH.address
        assert debt == USDC.address
        assert raw_amount == UINT256_MAX
        assert receive_atoken is False

    def test_rejects_invalid_amount(self, aave: AaveV3):
        with pytest.raises(TypeError):
            aave.build_liquidation_call_tx(WETH, "500", ETH_WHALE)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chain", chains_for("AAVE_V3_ADDRESSES_PROVIDER"))
def test_aave_v3_deployment_pinned(chain: int):
    """The immutable ``PoolAddressesProvider`` is pinned on every supported
    chain — Pool / DataProvider / Oracle are resolved from it on-chain and
    deliberately not pinned (see ``AaveV3.from_chain``)."""
    from pydefi.deployments import get_address

    addr = get_address("AAVE_V3_ADDRESSES_PROVIDER", chain)
    assert len(addr) == 20, f"AAVE_V3_ADDRESSES_PROVIDER on {chain}: {addr.hex()!r}"


# ---------------------------------------------------------------------------
# Reserve resolution by symbol (commit a8185863)
# ---------------------------------------------------------------------------

#: Sepolia faucet-token mismatch: the registry USDC (from get_token) is NOT
#: Aave's listed reserve — a faucet token at AAVE_USDC_RESERVE_SEPOLIA — so the
#: lookup must resolve by symbol, not token.address.
SEPOLIA_USDC = get_token("USDC", ChainId.SEPOLIA)
AAVE_USDC_RESERVE_SEPOLIA = Address("0x94a9D9AC8a22534E3FaCa9F4e7F2E2cf85d5E4C8")


@contextmanager
def _patch_listed_reserves(listed: list[tuple[str, Address]]):
    """Patch ``getAllReservesTokens()`` to return *listed*; yield the ``.call``
    mock so a test can assert the map is fetched once (cached)."""
    call_mock = AsyncMock(return_value=listed)
    fn = MagicMock()
    fn.call = call_mock
    provider = MagicMock()
    provider.fns.getAllReservesTokens.return_value = fn
    with patch("pydefi.lending.aave_v3.AAVE_V3_DATA_PROVIDER", provider):
        yield call_mock


class TestAaveResolveReserve:
    """``_resolve_reserve`` maps a token's symbol to Aave's listed reserve.

    Regression for commit a8185863: testnet faucet tokens differ from the
    registry address, and ``getReserveData`` reverts on a non-reserve.
    """

    async def test_maps_symbol_to_aave_reserve(self, aave: AaveV3):
        """The fix resolves the symbol to Aave's faucet reserve, not token.address."""
        listed = [("USDC", AAVE_USDC_RESERVE_SEPOLIA), ("WETH", Address("0x" + "ab" * 20))]
        with _patch_listed_reserves(listed):
            resolved = await aave._resolve_reserve(SEPOLIA_USDC)
        assert resolved == AAVE_USDC_RESERVE_SEPOLIA
        assert resolved != SEPOLIA_USDC.address  # would have been the unresolved bug value

    async def test_mainnet_resolution_is_a_no_op(self, aave: AaveV3):
        """On mainnet the registry address IS the reserve — resolution is a no-op."""
        with _patch_listed_reserves([("USDC", USDC.address)]):
            assert await aave._resolve_reserve(USDC) == USDC.address

    async def test_unlisted_symbol_raises(self, aave: AaveV3):
        """An unlisted symbol raises PydefiError, not a downstream revert."""
        with _patch_listed_reserves([("USDC", AAVE_USDC_RESERVE_SEPOLIA)]):
            with pytest.raises(PydefiError, match="no reserve for 'WETH'"):
                await aave._resolve_reserve(WETH)

    async def test_reserve_map_is_cached(self, aave: AaveV3):
        """The ``{symbol: address}`` map is read once and reused across calls."""
        listed = [("USDC", AAVE_USDC_RESERVE_SEPOLIA), ("WETH", WETH.address)]
        with _patch_listed_reserves(listed) as call_mock:
            await aave._resolve_reserve(SEPOLIA_USDC)
            await aave._resolve_reserve(WETH)
            assert call_mock.await_count == 1


# ===========================================================================
# Compound V3 (Comet)
# ===========================================================================

CUSDC_V3 = Address("0xc3d688B66703497DAA19211EEdff47f25384cdc3")


def _apy_to_per_second_rate(target_apy: Decimal) -> int:
    """Invert :func:`per_second_rate_to_apy` for round-trip testing."""
    per_second = (Decimal(1) + target_apy) ** (Decimal(1) / Decimal(SECONDS_PER_YEAR)) - Decimal(1)
    return int(per_second * Decimal(COMET_SCALE))


# ---------------------------------------------------------------------------
# Rate model
# ---------------------------------------------------------------------------


class TestCompoundRateModel:
    def test_zero(self):
        assert per_second_rate_to_apy(0) == Decimal(0)

    @pytest.mark.parametrize("target,tolerance", [(Decimal("0.03"), "0.0001"), (Decimal("0.18"), "0.0005")])
    def test_round_trip(self, target: Decimal, tolerance: str):
        apy = per_second_rate_to_apy(_apy_to_per_second_rate(target))
        assert abs(apy - target) < Decimal(tolerance)


# ---------------------------------------------------------------------------
# Tx-builder calldata encoding
# ---------------------------------------------------------------------------


@pytest.fixture
def comet() -> CompoundV3:
    return CompoundV3(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.ETHEREUM,
        comet_address=CUSDC_V3,
        base_token=USDC,
    )


class TestCompoundBuildSupplyTx:
    @pytest.mark.parametrize("token", [USDC, WETH], ids=["base", "collateral"])
    def test_supply_dispatches_by_asset(self, comet: CompoundV3, token):
        """Comet's single ``supply`` handles both base and collateral."""
        amount = TokenAmount.from_human(token, "1")
        tx = comet.build_supply_tx(amount)
        assert Address(tx["to"]) == CUSDC_V3
        asset, raw_amount = decode_call(tx, COMPOUND_V3_COMET.fns.supply)
        assert asset == token.address
        assert raw_amount == amount.amount

    def test_supply_to(self, comet: CompoundV3):
        dst = Address("0x" + "ab" * 20)
        amount = TokenAmount.from_human(USDC, "50")
        decoded_dst, asset, raw_amount = decode_call(
            comet.build_supply_tx(amount, dst=dst), COMPOUND_V3_COMET.fns.supplyTo
        )
        assert decoded_dst == dst
        assert asset == USDC.address
        assert raw_amount == amount.amount


class TestCompoundBuildWithdrawTx:
    def test_explicit_amount(self, comet: CompoundV3):
        amount = TokenAmount.from_human(USDC, "25")
        asset, raw_amount = decode_call(comet.build_withdraw_tx(amount), COMPOUND_V3_COMET.fns.withdraw)
        assert asset == USDC.address
        assert raw_amount == amount.amount

    def test_max_amount(self, comet: CompoundV3):
        _asset, raw_amount = decode_call(comet.build_withdraw_tx((USDC, "max")), COMPOUND_V3_COMET.fns.withdraw)
        assert raw_amount == UINT256_MAX

    def test_withdraw_to(self, comet: CompoundV3):
        recipient = Address("0x" + "cd" * 20)
        decoded_to, asset, _amount = decode_call(
            comet.build_withdraw_tx(TokenAmount.from_human(WETH, "1"), to=recipient),
            COMPOUND_V3_COMET.fns.withdrawTo,
        )
        assert decoded_to == recipient
        assert asset == WETH.address

    def test_rejects_invalid_amount(self, comet: CompoundV3):
        with pytest.raises(TypeError):
            comet.build_withdraw_tx("bad")  # type: ignore[arg-type]


class TestCompoundOtherBuilders:
    def test_transfer_asset(self, comet: CompoundV3):
        dst = Address("0x" + "ef" * 20)
        amount = TokenAmount.from_human(USDC, "10")
        decoded_dst, asset, raw_amount = decode_call(
            comet.build_transfer_asset_tx(dst, amount), COMPOUND_V3_COMET.fns.transferAsset
        )
        assert decoded_dst == dst
        assert asset == USDC.address
        assert raw_amount == amount.amount

    @pytest.mark.parametrize("flag", [True, False])
    def test_allow_flag(self, comet: CompoundV3, flag: bool):
        manager = Address("0x" + "ad" * 20)
        decoded_manager, is_allowed = decode_call(comet.build_allow_tx(manager, flag), COMPOUND_V3_COMET.fns.allow)
        assert decoded_manager == manager
        assert is_allowed is flag


class TestCompoundLiquidation:
    def test_absorb_encodes_accounts(self, comet: CompoundV3):
        absorber = Address("0x" + "a1" * 20)
        accounts = [ETH_WHALE, Address("0x" + "b2" * 20)]
        tx = comet.build_absorb_tx(absorber, accounts)
        assert Address(tx["to"]) == CUSDC_V3
        decoded_absorber, decoded_accounts = decode_call(tx, COMPOUND_V3_COMET.fns.absorb)
        assert decoded_absorber == absorber
        assert [Address(a) for a in decoded_accounts] == accounts

    def test_absorb_rejects_empty_accounts(self, comet: CompoundV3):
        with pytest.raises(ValueError):
            comet.build_absorb_tx(Address("0x" + "a1" * 20), [])

    def test_buy_collateral(self, comet: CompoundV3):
        base_amount = TokenAmount.from_human(USDC, "1000")
        min_amount = TokenAmount.from_human(WETH, "0.3")
        recipient = Address("0x" + "c3" * 20)
        asset, decoded_min, decoded_base, decoded_recipient = decode_call(
            comet.build_buy_collateral_tx(base_amount, min_amount, recipient),
            COMPOUND_V3_COMET.fns.buyCollateral,
        )
        assert asset == WETH.address
        assert decoded_min == min_amount.amount
        assert decoded_base == base_amount.amount
        assert decoded_recipient == recipient


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


COMPOUND_V3_MARKETS = [(name, chain) for name in comet_contract_names() for chain in sorted(chains_for(name))]


@pytest.mark.parametrize("name,chain", COMPOUND_V3_MARKETS)
def test_compound_v3_market_pinned(name: str, chain: int):
    """Every Comet market entry resolves to a 20-byte address."""
    from pydefi.deployments import get_address

    addr = get_address(name, chain)
    assert len(addr) == 20, f"{name} on {chain}: {addr.hex()!r}"


# ===========================================================================
# Morpho Blue
# ===========================================================================

MORPHO_BLUE_ADDR = Address("0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb")
MORPHO_IRM_ADDR = Address("0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC")
MORPHO_ORACLE_ADDR = Address("0x2a01EB9496094dA03c4E364Def50f5aD1280AD72")

#: ``keccak256(abi.encode(MARKET))`` computed offline (see docs/plan_morpho.md)
#: — a fixed regression vector independent of the production code path.
EXPECTED_MARKET_ID = bytes.fromhex("68e2c12cb7f73414a223441f15f6f5e0bf3019a80ed6752f1b75a4fc3642f006")

#: A WETH-collateral / USDC-loan market used across the Morpho tests.
MARKET = MarketParams(
    loan_token=USDC,
    collateral_token=WETH,
    oracle=MORPHO_ORACLE_ADDR,
    irm=MORPHO_IRM_ADDR,
    lltv=LLTV_86,
)


# ---------------------------------------------------------------------------
# Market Id
# ---------------------------------------------------------------------------


class TestMorphoMarketId:
    def test_matches_offline_vector(self):
        assert MARKET.id == EXPECTED_MARKET_ID

    def test_cross_check_against_eth_abi(self):
        """Independent path: raw eth_abi.encode + keccak, not the ABIStruct."""
        from eth_abi import encode
        from eth_utils import keccak

        expected = keccak(
            encode(
                ["address", "address", "address", "address", "uint256"],
                [bytes(USDC.address), bytes(WETH.address), bytes(MORPHO_ORACLE_ADDR), bytes(MORPHO_IRM_ADDR), LLTV_86],
            )
        )
        assert MARKET.id == expected

    def test_field_order_matters(self):
        """Swapping loan / collateral must yield a different market."""
        swapped = MarketParams(
            loan_token=WETH,
            collateral_token=USDC,
            oracle=MORPHO_ORACLE_ADDR,
            irm=MORPHO_IRM_ADDR,
            lltv=LLTV_86,
        )
        assert swapped.id != MARKET.id


# ---------------------------------------------------------------------------
# Interest accrual
# ---------------------------------------------------------------------------


def _state(*, fee: int = 0) -> MarketState:
    return MarketState(
        total_supply_assets=1_000_000,
        total_supply_shares=1_000_000,
        total_borrow_assets=500_000,
        total_borrow_shares=500_000,
        last_update=1_700_000_000,
        fee=fee,
    )


class TestMorphoAccrueInterest:
    def test_zero_elapsed_is_noop(self):
        state = _state()
        assert accrue_interest(state, 10**9, 0) is state

    def test_no_debt_is_noop(self):
        no_debt = MarketState(1_000_000, 1_000_000, 0, 0, 1_700_000_000, 0)
        assert accrue_interest(no_debt, 10**9, 3_600) is no_debt

    def test_interest_grows_both_sides_equally(self):
        accrued = accrue_interest(_state(), 10**9, SECONDS_PER_YEAR)
        borrow_interest = accrued.total_borrow_assets - 500_000
        assert borrow_interest > 0
        # With no fee, every unit of borrow interest is credited to suppliers.
        assert accrued.total_supply_assets - 1_000_000 == borrow_interest
        assert accrued.total_supply_shares == 1_000_000

    def test_fee_mints_supply_shares(self):
        accrued = accrue_interest(_state(fee=10**17), 10**9, SECONDS_PER_YEAR)
        assert accrued.total_supply_shares > 1_000_000


# ---------------------------------------------------------------------------
# Health factor
# ---------------------------------------------------------------------------

# Oracle price of WETH (18 dec) in USDC (6 dec), 1e36-scaled: $3000 -> 3000e24.
_WETH_PRICE = 3_000 * 10**24


class TestMorphoHealthFactor:
    def test_no_debt_is_infinite(self):
        assert compute_health_factor(10**18, 0, _WETH_PRICE, LLTV_86).is_infinite()

    def test_known_value(self):
        # 1 WETH collateral @ $3000, 86% LLTV -> max borrow 2580 USDC.
        # Borrowing 1290 USDC leaves a health factor of exactly 2.
        hf = compute_health_factor(10**18, 1_290 * 10**6, _WETH_PRICE, LLTV_86)
        assert hf == Decimal(2)

    def test_liquidatable_below_one(self):
        hf = compute_health_factor(10**18, 2_600 * 10**6, _WETH_PRICE, LLTV_86)
        assert hf < Decimal(1)


# ---------------------------------------------------------------------------
# Liquidation seize math
# ---------------------------------------------------------------------------


def _morpho_position(*, borrow_assets: int, collateral: int, collateral_price: int) -> MorphoPosition:
    """A :class:`MorphoPosition` carrying just the fields ``max_liquidation``
    reads (shares stand in for assets 1:1 — the seize decision uses assets)."""
    return MorphoPosition(
        market=MARKET,
        supply_shares=0,
        borrow_shares=borrow_assets,
        supply_assets=TokenAmount(USDC, 0),
        borrow_assets=TokenAmount(USDC, borrow_assets),
        collateral=TokenAmount(WETH, collateral),
        health_factor=Decimal("0.5"),
        collateral_price=collateral_price,
    )


class TestMorphoLiquidationIncentiveFactor:
    def test_matches_known_value_for_86_lltv(self):
        # WAD / (WAD - 0.30*(WAD - 0.86 WAD)) = 1e36 // 0.958e18.
        assert liquidation_incentive_factor(LLTV_86) == 10**36 // 958_000_000_000_000_000

    def test_below_the_cap_for_a_high_lltv(self):
        assert WAD < liquidation_incentive_factor(LLTV_86) < MAX_LIQUIDATION_INCENTIVE_FACTOR

    def test_clamped_to_max_for_a_very_low_lltv(self):
        # A near-zero LLTV would push the raw factor past the cap.
        assert liquidation_incentive_factor(10**15) == MAX_LIQUIDATION_INCENTIVE_FACTOR


class TestMorphoMaxLiquidation:
    def test_solvent_position_repays_all_shares(self):
        # Collateral worth far more than borrow_assets * incentive.
        seized, repaid = max_liquidation(
            _morpho_position(borrow_assets=5000, collateral=10**24, collateral_price=ORACLE_PRICE_SCALE)
        )
        assert seized is None
        assert repaid == 5000

    def test_bad_debt_position_seizes_all_collateral(self):
        # Collateral worth less than the debt -> full repayment would over-seize.
        seized, repaid = max_liquidation(
            _morpho_position(borrow_assets=5000, collateral=1000, collateral_price=ORACLE_PRICE_SCALE)
        )
        assert repaid is None
        assert seized == TokenAmount(WETH, 1000)

    def test_no_oracle_falls_back_to_repaying_all_shares(self):
        seized, repaid = max_liquidation(_morpho_position(borrow_assets=5000, collateral=1000, collateral_price=0))
        assert seized is None
        assert repaid == 5000


# ---------------------------------------------------------------------------
# Supply APY derivation
# ---------------------------------------------------------------------------


class TestMorphoSupplyApy:
    def test_empty_market_is_zero(self):
        assert supply_apy_from_borrow(10**9, MarketState(0, 0, 0, 0, 0, 0)) == Decimal(0)

    def test_below_borrow_apy_when_partially_utilized(self):
        # 50% utilization -> suppliers earn strictly less than the borrow rate.
        half_used = MarketState(2_000_000, 2_000_000, 1_000_000, 1_000_000, 0, 0)
        supply_apy = supply_apy_from_borrow(10**9, half_used)
        assert Decimal(0) < supply_apy < per_second_rate_to_apy(10**9)


# ---------------------------------------------------------------------------
# Tx-builder calldata encoding
# ---------------------------------------------------------------------------


@pytest.fixture
def morpho() -> MorphoBlue:
    return MorphoBlue(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.ETHEREUM,
        morpho_address=MORPHO_BLUE_ADDR,
    )


def _assert_market_struct(struct: tuple) -> None:
    """Assert a decoded MarketParams struct matches :data:`MARKET`."""
    loan, collateral, oracle, irm, lltv = struct
    assert Address(loan) == USDC.address
    assert Address(collateral) == WETH.address
    assert Address(oracle) == MORPHO_ORACLE_ADDR
    assert Address(irm) == MORPHO_IRM_ADDR
    assert lltv == LLTV_86


#: supply / withdraw / borrow / repay share one calldata shape —
#: ``(MarketParams, assets, shares, onBehalf, data|receiver)``. ``extra`` is
#: the builder-specific trailing kwarg; ``trailing`` its expected decoded value.
_AMOUNT_BUILDERS = [
    pytest.param("build_supply_tx", MORPHO_BLUE.fns.supply, {}, b"", id="supply"),
    pytest.param("build_withdraw_tx", MORPHO_BLUE.fns.withdraw, {"receiver": ETH_WHALE}, ETH_WHALE, id="withdraw"),
    pytest.param("build_borrow_tx", MORPHO_BLUE.fns.borrow, {"receiver": ETH_WHALE}, ETH_WHALE, id="borrow"),
    pytest.param("build_repay_tx", MORPHO_BLUE.fns.repay, {}, b"", id="repay"),
]


@pytest.mark.parametrize("method, abi_fn, extra, trailing", _AMOUNT_BUILDERS)
class TestMorphoAmountBuilders:
    """supply / withdraw / borrow / repay — MarketParams + an assets-XOR-shares
    amount, all encoded the same way."""

    def test_by_assets(self, morpho: MorphoBlue, method, abi_fn, extra, trailing):
        amount = TokenAmount.from_human(USDC, "1000")
        tx = getattr(morpho, method)(MARKET, assets=amount, on_behalf_of=ETH_WHALE, **extra)
        assert Address(tx["to"]) == MORPHO_BLUE_ADDR
        assert tx["value"] == "0"
        struct, assets, shares, on_behalf, last = decode_call(tx, abi_fn)
        _assert_market_struct(struct)
        assert (assets, shares, on_behalf, last) == (amount.amount, 0, ETH_WHALE, trailing)

    def test_by_shares(self, morpho: MorphoBlue, method, abi_fn, extra, trailing):
        tx = getattr(morpho, method)(MARKET, shares=12_345, on_behalf_of=ETH_WHALE, **extra)
        _struct, assets, shares, *_ = decode_call(tx, abi_fn)
        assert (assets, shares) == (0, 12_345)

    def test_requires_exactly_one_amount(self, morpho: MorphoBlue, method, abi_fn, extra, trailing):
        build = getattr(morpho, method)
        with pytest.raises(ValueError):  # neither assets nor shares
            build(MARKET, on_behalf_of=ETH_WHALE, **extra)
        with pytest.raises(ValueError):  # both
            build(MARKET, assets=TokenAmount.from_human(USDC, "1"), shares=1, on_behalf_of=ETH_WHALE, **extra)

    def test_rejects_zero_amount(self, morpho: MorphoBlue, method, abi_fn, extra, trailing):
        """A zero amount would encode the assets=0, shares=0 payload Morpho rejects."""
        build = getattr(morpho, method)
        with pytest.raises(ValueError):  # zero assets
            build(MARKET, assets=TokenAmount(USDC, 0), on_behalf_of=ETH_WHALE, **extra)
        with pytest.raises(ValueError):  # zero shares
            build(MARKET, shares=0, on_behalf_of=ETH_WHALE, **extra)


#: supply / withdraw collateral — ``(MarketParams, assets, onBehalf, data|receiver)``.
_COLLATERAL_BUILDERS = [
    pytest.param("build_supply_collateral_tx", MORPHO_BLUE.fns.supplyCollateral, (ETH_WHALE,), b"", id="supply"),
    pytest.param(
        "build_withdraw_collateral_tx",
        MORPHO_BLUE.fns.withdrawCollateral,
        (ETH_WHALE, ETH_WHALE),
        ETH_WHALE,
        id="withdraw",
    ),
]


@pytest.mark.parametrize("method, abi_fn, args, trailing", _COLLATERAL_BUILDERS)
def test_morpho_collateral_builder(morpho: MorphoBlue, method, abi_fn, args, trailing):
    amount = TokenAmount.from_human(WETH, "2")
    struct, assets, on_behalf, last = decode_call(getattr(morpho, method)(MARKET, amount, *args), abi_fn)
    _assert_market_struct(struct)
    assert (assets, on_behalf, last) == (amount.amount, ETH_WHALE, trailing)


class TestMorphoBuildOtherTxs:
    @pytest.mark.parametrize("flag", [True, False])
    def test_set_authorization(self, morpho: MorphoBlue, flag: bool):
        authorized, is_authorized = decode_call(
            morpho.build_set_authorization_tx(ETH_WHALE, flag), MORPHO_BLUE.fns.setAuthorization
        )
        assert authorized == ETH_WHALE
        assert is_authorized is flag

    def test_create_market(self, morpho: MorphoBlue):
        tx = morpho.build_create_market_tx(MARKET)
        # createMarket has a single struct input, which decode_input unwraps.
        struct = MORPHO_BLUE.fns.createMarket.decode_input(HexBytes(tx["data"]))
        _assert_market_struct(struct)

    def test_flashloan(self, morpho: MorphoBlue):
        amount = TokenAmount.from_human(USDC, "10000")
        token, raw_amount, data = decode_call(morpho.build_flashloan_tx(USDC, amount), MORPHO_BLUE.fns.flashLoan)
        assert token == USDC.address
        assert raw_amount == amount.amount
        assert data == b""

    def test_liquidate_by_seized_assets(self, morpho: MorphoBlue):
        seized = TokenAmount.from_human(WETH, "2")
        tx = morpho.build_liquidate_tx(MARKET, ETH_WHALE, seized_assets=seized)
        assert Address(tx["to"]) == MORPHO_BLUE_ADDR
        assert tx["value"] == "0"
        struct, borrower, seized_assets, repaid_shares, data = decode_call(tx, MORPHO_BLUE.fns.liquidate)
        _assert_market_struct(struct)
        assert (borrower, seized_assets, repaid_shares, data) == (ETH_WHALE, seized.amount, 0, b"")

    def test_liquidate_by_repaid_shares(self, morpho: MorphoBlue):
        tx = morpho.build_liquidate_tx(MARKET, ETH_WHALE, repaid_shares=12_345)
        _struct, _borrower, seized_assets, repaid_shares, _data = decode_call(tx, MORPHO_BLUE.fns.liquidate)
        assert (seized_assets, repaid_shares) == (0, 12_345)

    def test_liquidate_encodes_flash_callback_data(self, morpho: MorphoBlue):
        """A non-empty data payload drives the onMorphoLiquidate callback."""
        payload = b"\xde\xad\xbe\xef"
        tx = morpho.build_liquidate_tx(MARKET, ETH_WHALE, repaid_shares=1, data=payload)
        _struct, _borrower, _seized, _shares, data = decode_call(tx, MORPHO_BLUE.fns.liquidate)
        assert data == payload

    def test_liquidate_requires_exactly_one_amount(self, morpho: MorphoBlue):
        with pytest.raises(ValueError):  # neither
            morpho.build_liquidate_tx(MARKET, ETH_WHALE)
        with pytest.raises(ValueError):  # both
            morpho.build_liquidate_tx(
                MARKET, ETH_WHALE, seized_assets=TokenAmount.from_human(WETH, "1"), repaid_shares=1
            )

    def test_liquidate_rejects_zero_amount(self, morpho: MorphoBlue):
        with pytest.raises(ValueError):  # zero seized assets
            morpho.build_liquidate_tx(MARKET, ETH_WHALE, seized_assets=TokenAmount(WETH, 0))
        with pytest.raises(ValueError):  # zero repaid shares
            morpho.build_liquidate_tx(MARKET, ETH_WHALE, repaid_shares=0)


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chain", chains_for("MORPHO_BLUE"))
def test_morpho_blue_deployment_pinned(chain: int):
    """Every pinned Morpho Blue address resolves to a 20-byte address."""
    from pydefi.deployments import get_address

    addr = get_address("MORPHO_BLUE", chain)
    assert len(addr) == 20, f"MORPHO_BLUE on {chain}: {addr.hex()!r}"


# ===========================================================================
# Aave V4
# ===========================================================================

#: MAIN_SPOKE on Ethereum — the Spoke the unit-test client is bound to.
AAVE_V4_SPOKE_ADDR = Address("0x94e7A5dCbE816e498b89aB752661904E2F56c485")

#: Every pinned Aave V4 Hub / Spoke deployment name.
_AAVE_V4_DEPLOYMENTS = [
    "AAVE_V4_CORE_HUB",
    "AAVE_V4_PLUS_HUB",
    "AAVE_V4_PRIME_HUB",
    "AAVE_V4_MAIN_SPOKE",
    "AAVE_V4_BLUECHIP_SPOKE",
    "AAVE_V4_TREASURY_SPOKE",
    "AAVE_V4_GOLD_SPOKE",
    "AAVE_V4_FOREX_SPOKE",
    "AAVE_V4_LOMBARD_BTC_SPOKE",
    "AAVE_V4_ETHENA_CORRELATED_SPOKE",
    "AAVE_V4_ETHENA_ECOSYSTEM_SPOKE",
    "AAVE_V4_ETHERFI_ESPOKE",
    "AAVE_V4_KELP_ESPOKE",
    "AAVE_V4_LIDO_ESPOKE",
]


# ---------------------------------------------------------------------------
# Health-factor parsing
# ---------------------------------------------------------------------------


class TestAaveV4HealthFactor:
    def test_finite(self):
        assert aave_v4.parse_health_factor(15 * 10**17) == Decimal("1.5")

    def test_infinite_when_no_debt(self):
        assert aave_v4.parse_health_factor(UINT256_MAX).is_infinite()

    def test_user_account_data_construction(self):
        data = aave_v4.V4UserAccountData(
            risk_premium=50,
            avg_collateral_factor=8 * 10**17,
            health_factor=Decimal("2"),
            total_collateral_value=1_000,
            total_debt_value_ray=500 * 10**27,
            active_collateral_count=2,
            borrow_count=1,
        )
        assert data.health_factor == Decimal("2")
        assert data.borrow_count == 1


# ---------------------------------------------------------------------------
# Tx-builder calldata encoding
# ---------------------------------------------------------------------------


@pytest.fixture
def aave_v4_spoke() -> aave_v4.AaveV4:
    return aave_v4.AaveV4(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.ETHEREUM,
        spoke_address=AAVE_V4_SPOKE_ADDR,
    )


class TestAaveV4BuildTxs:
    @pytest.mark.parametrize(
        "method, abi_fn",
        [
            ("build_supply_tx", AAVE_V4_SPOKE.fns.supply),
            ("build_withdraw_tx", AAVE_V4_SPOKE.fns.withdraw),
            ("build_borrow_tx", AAVE_V4_SPOKE.fns.borrow),
            ("build_repay_tx", AAVE_V4_SPOKE.fns.repay),
        ],
        ids=["supply", "withdraw", "borrow", "repay"],
    )
    def test_amount_builders(self, aave_v4_spoke: aave_v4.AaveV4, method, abi_fn):
        """supply / withdraw / borrow / repay all encode (reserveId, amount, onBehalfOf)."""
        amount = TokenAmount.from_human(USDC, "1000")
        tx = getattr(aave_v4_spoke, method)(7, amount, ETH_WHALE)
        assert Address(tx["to"]) == AAVE_V4_SPOKE_ADDR
        assert tx["value"] == "0"
        reserve_id, raw_amount, on_behalf = decode_call(tx, abi_fn)
        assert reserve_id == 7
        assert raw_amount == amount.amount
        assert on_behalf == ETH_WHALE

    @pytest.mark.parametrize("flag", [True, False])
    def test_set_collateral(self, aave_v4_spoke: aave_v4.AaveV4, flag: bool):
        reserve_id, use_as_collateral, on_behalf = decode_call(
            aave_v4_spoke.build_set_collateral_tx(3, flag, ETH_WHALE), AAVE_V4_SPOKE.fns.setUsingAsCollateral
        )
        assert reserve_id == 3
        assert use_as_collateral is flag
        assert on_behalf == ETH_WHALE

    def test_liquidation_call(self, aave_v4_spoke: aave_v4.AaveV4):
        amount = TokenAmount.from_human(USDC, "250")
        collat_id, debt_id, user, debt_to_cover, receive_shares = decode_call(
            aave_v4_spoke.build_liquidation_call_tx(0, 7, ETH_WHALE, amount, receive_shares=True),
            AAVE_V4_SPOKE.fns.liquidationCall,
        )
        assert (collat_id, debt_id) == (0, 7)
        assert user == ETH_WHALE
        assert debt_to_cover == amount.amount
        assert receive_shares is True

    @pytest.mark.parametrize("approve", [True, False])
    def test_set_position_manager(self, aave_v4_spoke: aave_v4.AaveV4, approve: bool):
        manager = Address("0x" + "ad" * 20)
        decoded_manager, decoded_approve = decode_call(
            aave_v4_spoke.build_set_position_manager_tx(manager, approve), AAVE_V4_SPOKE.fns.setUserPositionManager
        )
        assert decoded_manager == manager
        assert decoded_approve is approve

    def test_from_chain_resolves_named_spoke(self):
        from pydefi.deployments import get_address

        spoke = aave_v4.AaveV4.from_chain(None, ChainId.ETHEREUM, "GOLD_SPOKE")  # type: ignore[arg-type]
        assert spoke.spoke_address == get_address("AAVE_V4_GOLD_SPOKE", ChainId.ETHEREUM)


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _AAVE_V4_DEPLOYMENTS)
def test_aave_v4_deployment_pinned(name: str):
    """Every pinned Aave V4 Hub / Spoke resolves to a 20-byte address."""
    from pydefi.deployments import get_address

    addr = get_address(name, ChainId.ETHEREUM)
    assert len(addr) == 20, f"{name}: {addr.hex()!r}"


# ---------------------------------------------------------------------------
# Aave V4 — TokenizationSpoke (ERC-4626 vault)
# ---------------------------------------------------------------------------


@pytest.fixture
def aave_v4_vault() -> aave_v4.AaveV4TokenizationSpoke:
    return aave_v4.AaveV4TokenizationSpoke(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.ETHEREUM,
        spoke_address=AAVE_V4_SPOKE_ADDR,
        asset_token=USDC,
    )


class TestAaveV4TokenizationSpoke:
    def test_to_and_value(self, aave_v4_vault: aave_v4.AaveV4TokenizationSpoke):
        tx = aave_v4_vault.build_mint_tx(1_000, ETH_WHALE)
        assert Address(tx["to"]) == AAVE_V4_SPOKE_ADDR
        assert tx["value"] == "0"

    def test_deposit(self, aave_v4_vault: aave_v4.AaveV4TokenizationSpoke):
        amount = TokenAmount.from_human(USDC, "100")
        assets, receiver = decode_call(
            aave_v4_vault.build_deposit_tx(amount, ETH_WHALE), AAVE_V4_TOKENIZATION_SPOKE.fns.deposit
        )
        assert assets == amount.amount
        assert receiver == ETH_WHALE

    def test_mint(self, aave_v4_vault: aave_v4.AaveV4TokenizationSpoke):
        shares, receiver = decode_call(
            aave_v4_vault.build_mint_tx(12_345, ETH_WHALE), AAVE_V4_TOKENIZATION_SPOKE.fns.mint
        )
        assert shares == 12_345
        assert receiver == ETH_WHALE

    def test_withdraw(self, aave_v4_vault: aave_v4.AaveV4TokenizationSpoke):
        amount = TokenAmount.from_human(USDC, "50")
        assets, receiver, owner = decode_call(
            aave_v4_vault.build_withdraw_tx(amount, ETH_WHALE, ETH_WHALE), AAVE_V4_TOKENIZATION_SPOKE.fns.withdraw
        )
        assert assets == amount.amount
        assert receiver == ETH_WHALE
        assert owner == ETH_WHALE

    def test_redeem(self, aave_v4_vault: aave_v4.AaveV4TokenizationSpoke):
        shares, receiver, owner = decode_call(
            aave_v4_vault.build_redeem_tx(999, ETH_WHALE, ETH_WHALE), AAVE_V4_TOKENIZATION_SPOKE.fns.redeem
        )
        assert shares == 999
        assert receiver == ETH_WHALE
        assert owner == ETH_WHALE


# ---------------------------------------------------------------------------
# Babylon TBV — AaveAdapter (native-BTC lending on Aave V4)
# ---------------------------------------------------------------------------

#: Reserve ids on the BabylonCoreSpoke (stable indices) + their tokens.
_BB_USDC_RESERVE = 0
_BB_VAULTBTC_RESERVE = 3
_BB_USDC = get_token("USDC", ChainId.ETHEREUM)  # decimals only; amounts are raw
_BB_VBTC = get_token("vaultBTC", ChainId.SEPOLIA)
_BB_AMT = TokenAmount(_BB_USDC, 100 * 10**6)


@pytest.fixture
def babylon_adapter() -> BabylonAaveAdapter:
    """Adapter client with no live ``w3`` — enough to build (encode) txs."""
    return BabylonAaveAdapter(
        w3=None,  # type: ignore[arg-type]
        chain_id=ChainId.SEPOLIA,
        adapter_address=get_address("BABYLON_AAVE_ADAPTER", ChainId.SEPOLIA),
    )


class TestBabylonAdapterBuildTx:
    """The adapter's lending tx builders encode ``(reserveId, amount, onBehalfOf)``."""

    def test_to_and_value(self, babylon_adapter: BabylonAaveAdapter):
        tx = babylon_adapter.build_op_tx("borrow", _BB_USDC_RESERVE, _BB_AMT, ETH_WHALE)
        assert Address(tx["to"]) == babylon_adapter.adapter_address
        assert tx["value"] == "0"

    def test_supply_borrow_repay_withdraw_share_the_shape(self, babylon_adapter: BabylonAaveAdapter):
        for op, fn in (
            ("supply", AAVE_ADAPTER.fns.supply),
            ("borrow", AAVE_ADAPTER.fns.borrow),
            ("repay", AAVE_ADAPTER.fns.repay),
            ("withdraw", AAVE_ADAPTER.fns.withdraw),
        ):
            decoded = decode_call(babylon_adapter.build_op_tx(op, _BB_VAULTBTC_RESERVE, _BB_AMT, ETH_WHALE), fn)
            assert decoded == (_BB_VAULTBTC_RESERVE, _BB_AMT.amount, ETH_WHALE)

    def test_rejects_unknown_op(self, babylon_adapter: BabylonAaveAdapter):
        with pytest.raises(ValueError, match="unknown lending op"):
            babylon_adapter.build_op_tx("liquidate", _BB_USDC_RESERVE, _BB_AMT, ETH_WHALE)

    def test_set_collateral_args(self, babylon_adapter: BabylonAaveAdapter):
        reserve_id, use_as_collateral, on_behalf_of = decode_call(
            babylon_adapter.build_set_collateral_tx(_BB_VAULTBTC_RESERVE, True, ETH_WHALE),
            AAVE_ADAPTER.fns.setUsingAsCollateral,
        )
        assert reserve_id == _BB_VAULTBTC_RESERVE
        assert use_as_collateral is True
        assert on_behalf_of == ETH_WHALE

    def test_withdraw_collaterals_args(self, babylon_adapter: BabylonAaveAdapter):
        vault_ids = [b"\x11" * 32, b"\x22" * 32]
        tx = babylon_adapter.build_withdraw_collaterals_tx(vault_ids)
        # The sole bytes32[] arg decodes to a tuple of its elements.
        decoded = AAVE_ADAPTER.fns.withdrawCollaterals.decode_input(HexBytes(tx["data"]))
        assert list(decoded) == vault_ids


class TestBabylonFlow:
    """The fluent flow composes ordered adapter txs and reads naturally."""

    def test_repay_then_unlock_showcase(self, babylon_adapter: BabylonAaveAdapter):
        vault_ids = [b"\x11" * 32]
        flow = babylon_adapter.flow(ETH_WHALE).op("repay", _BB_USDC_RESERVE, _BB_AMT).unlock_btc(vault_ids)

        assert [s.label for s in flow.steps()] == ["repay:0", "withdrawCollaterals"]
        repay_tx, unlock_tx = flow.build()
        assert decode_call(repay_tx, AAVE_ADAPTER.fns.repay) == (_BB_USDC_RESERVE, _BB_AMT.amount, ETH_WHALE)
        assert list(AAVE_ADAPTER.fns.withdrawCollaterals.decode_input(HexBytes(unlock_tx["data"]))) == vault_ids

    def test_lock_then_borrow_sequences_external_activation(self, babylon_adapter: BabylonAaveAdapter):
        activation = {"to": babylon_adapter.adapter_address, "data": b"\xde\xad\xbe\xef", "value": "0"}
        flow = babylon_adapter.flow(ETH_WHALE).lock_btc_to_tbv(activation).op("borrow", _BB_USDC_RESERVE, _BB_AMT)

        steps = flow.steps()
        assert [s.label for s in steps] == ["activateVault", "borrow:0"]
        assert steps[0].tx is activation  # external activation tx passed through verbatim
        assert decode_call(flow.build()[1], AAVE_ADAPTER.fns.borrow) == (_BB_USDC_RESERVE, _BB_AMT.amount, ETH_WHALE)

    def test_build_matches_steps_order(self, babylon_adapter: BabylonAaveAdapter):
        flow = (
            babylon_adapter.flow(ETH_WHALE)
            .op("supply", _BB_VAULTBTC_RESERVE, _BB_AMT)
            .set_collateral(_BB_VAULTBTC_RESERVE, True)
            .op("borrow", _BB_USDC_RESERVE, _BB_AMT)
        )
        assert [s.label for s in flow.steps()] == ["supply:3", "setCollateral:3=True", "borrow:0"]
        assert flow.build() == [s.tx for s in flow.steps()]


class TestTBVSignetActivateVault:
    """The EVM-side activateVault tx encodes the BTCVault peg-in payload."""

    def _vault(self):
        from pydefi.lending import BTCVault, BTCVaultStatus

        return BTCVault(
            depositor=ETH_WHALE,
            depositor_btc_pubkey=b"\x01" * 32,
            depositor_signed_pegin_tx=b"raw-pegin-tx",
            amount=50_000,  # sats
            vault_provider=Address("0x" + "a1" * 20),
            status=int(BTCVaultStatus.VERIFIED),
            application_entry_point=Address("0x" + "a2" * 20),
            universal_challengers_version=1,
            app_vault_keepers_version=2,
            offchain_params_version=3,
            created_at=111,
            verified_at=222,
            depositor_wots_pk_hash=b"\x02" * 32,
            hashlock=b"\x03" * 32,
            htlc_vout=0,
            depositor_pop_signature=b"pop-sig",
            pre_pegin_tx_hash=b"\x04" * 32,
        )

    def test_builds_activate_vault_tx(self):
        from pydefi.lending import build_activate_vault_tx

        adapter = get_address("BABYLON_AAVE_ADAPTER", ChainId.SEPOLIA)
        vault_id = b"\xab" * 32
        tx = build_activate_vault_tx(adapter, vault_id, self._vault(), activation_metadata=b"meta")

        assert Address(tx["to"]) == adapter
        assert tx["value"] == "0"
        decoded_id, decoded_vault, decoded_meta = AAVE_ADAPTER.fns.activateVault.decode_input(HexBytes(tx["data"]))
        assert bytes(decoded_id) == vault_id
        assert bytes(decoded_meta) == b"meta"
        # vault tuple round-trips: depositor + amount + status land in place.
        assert Address(decoded_vault[0]) == ETH_WHALE
        assert decoded_vault[3] == 50_000


def _patch_urlopen(monkeypatch, body: bytes) -> dict[str, Any]:
    """Patch ``tbv_signet``'s ``urlopen`` to capture the request (url/data/headers)
    and return *body*. Returns the ``captured`` dict to assert against."""
    from pydefi.lending import tbv_signet

    captured: dict[str, Any] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    def _fake(req, timeout=None):
        captured["url"] = req if isinstance(req, str) else req.full_url
        captured["data"] = getattr(req, "data", None)
        captured["headers"] = getattr(req, "headers", {})
        return _Resp()

    monkeypatch.setattr(tbv_signet.urllib.request, "urlopen", _fake)
    return captured


class TestTBVVpProxy:
    """The vault-provider JSON-RPC transport builds the right request."""

    def test_rejects_bad_provider_address(self):
        from pydefi.lending.tbv_signet import vp_rpc

        with pytest.raises(ValueError, match="20-byte EVM address"):
            vp_rpc("0x1234", "anyMethod")

    def test_vp_rpc_posts_jsonrpc_with_auth(self, monkeypatch):
        import json as _json

        from pydefi.lending import tbv_signet

        captured = _patch_urlopen(monkeypatch, b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}')
        provider = "0x" + "ab" * 20
        result = tbv_signet.vp_rpc(provider, "getVaultParams", [7], auth_token="tok", vp_proxy_url="https://vp.test")

        assert result == {"ok": True}
        assert captured["url"] == f"https://vp.test/rpc/{provider}"
        body = _json.loads(captured["data"])
        assert (body["jsonrpc"], body["method"], body["params"]) == ("2.0", "getVaultParams", [7])
        assert captured["headers"].get("Authorization") == "Bearer tok"


class TestTBVSignetMempool:
    """The signet mempool helpers hit the right esplora endpoints."""

    def test_broadcast_posts_raw_hex_and_returns_txid(self, monkeypatch):
        from pydefi.lending import tbv_signet

        captured = _patch_urlopen(monkeypatch, b"deadbeeftxid\n")
        txid = tbv_signet.broadcast_pegin(b"\x01\x02", mempool_url="https://mp.test/signet")

        assert txid == "deadbeeftxid"
        assert captured["url"] == "https://mp.test/signet/api/tx"
        assert captured["data"] == b"0102"

    def test_fetch_utxos_gets_address_endpoint(self, monkeypatch):
        from pydefi.lending import tbv_signet

        captured = _patch_urlopen(monkeypatch, b'[{"txid":"aa","vout":0,"value":1000}]')
        utxos = tbv_signet.fetch_signet_utxos("tb1pexample", mempool_url="https://mp.test/signet")

        assert utxos == [{"txid": "aa", "vout": 0, "value": 1000}]
        assert captured["url"] == "https://mp.test/signet/api/address/tb1pexample/utxo"


class TestTBVVaultProviders:
    """vp-health discovery + getPeginStatus param shaping (no network)."""

    def test_vault_providers_enabled_filter(self, monkeypatch):
        from pydefi.lending import tbv_signet

        monkeypatch.setattr(
            tbv_signet,
            "vp_health",
            lambda **_: [{"address": "0xaa", "disabled": True}, {"address": "0xbb", "disabled": False}],
        )
        assert [p["address"] for p in tbv_signet.vault_providers()] == ["0xaa", "0xbb"]
        assert [p["address"] for p in tbv_signet.vault_providers(enabled_only=True)] == ["0xbb"]

    def test_get_pegin_status_requires_an_id(self):
        from pydefi.lending.tbv_signet import get_pegin_status

        with pytest.raises(ValueError, match="pegin_txid or vault_id"):
            get_pegin_status("0x" + "ab" * 20)

    def test_get_pegin_status_wraps_struct_param(self, monkeypatch):
        from pydefi.lending import tbv_signet

        captured: dict[str, Any] = {}

        def _fake_vp_rpc(provider, method, params=None, **kw):
            captured.update(provider=provider, method=method, params=params)
            return {"status": "PENDING"}

        monkeypatch.setattr(tbv_signet, "vp_rpc", _fake_vp_rpc)
        out = tbv_signet.get_pegin_status("0x" + "cd" * 20, pegin_txid="abcd")

        assert out == {"status": "PENDING"}
        assert captured["method"] == "vaultProvider_getPeginStatus"
        assert captured["params"] == [{"pegin_txid": "abcd"}]


class TestTBVAuthGating:
    """Auth-gated VP methods fail fast with guidance (no network)."""

    def test_gated_method_without_token_raises_before_network(self):
        from pydefi.lending.tbv_signet import AUTH_GATED_METHODS, vp_rpc

        method = next(iter(AUTH_GATED_METHODS))
        with pytest.raises(RuntimeError, match="auth-gated"):
            vp_rpc("0x" + "ab" * 20, method)  # no auth_token

    def test_grpc_gated_method_also_guarded(self):
        from pydefi.lending.tbv_signet import GRPC_AUTH_GATED_METHODS, vp_rpc

        method = next(iter(GRPC_AUTH_GATED_METHODS))
        with pytest.raises(RuntimeError, match="babylon-ts-sdk"):
            vp_rpc("0x" + "ab" * 20, method)

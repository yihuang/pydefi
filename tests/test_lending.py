"""Unit tests for pydefi.lending (no live node required).

Covers both protocols under the lending umbrella:

* Aave V3 — pool-per-reserve model with aTokens, health factors, E-Mode.
* Compound V3 (Comet) — one-base-asset-per-market model where supply /
  withdraw also handle borrow / repay through the same dispatch.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from hexbytes import HexBytes

from pydefi.abi.lending import AAVE_V3_POOL, COMPOUND_V3_COMET
from pydefi.lending import AaveV3, UserAccountData
from pydefi.lending.aave_v3 import (
    RAY,
    SECONDS_PER_YEAR,
    UINT256_MAX,
    parse_health_factor,
    ray_rate_to_apy,
)
from pydefi.lending.compound_v3 import (
    COMET_SCALE,
    CompoundV3,
    per_second_rate_to_apy,
)
from pydefi.types import Address, ChainId, TokenAmount
from tests.addrs import ETH_WHALE, USDC, WETH

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


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


AAVE_V3_CHAINS = [
    ChainId.ETHEREUM,
    ChainId.ARBITRUM,
    ChainId.BASE,
    ChainId.OPTIMISM,
    ChainId.POLYGON,
    ChainId.AVALANCHE,
    ChainId.BSC,
    ChainId.SCROLL,
    ChainId.LINEA,
    ChainId.ZKSYNC,
    ChainId.GNOSIS,
]

AAVE_V3_CONTRACTS = (
    "AAVE_V3_POOL",
    "AAVE_V3_DATA_PROVIDER",
    "AAVE_V3_ADDRESSES_PROVIDER",
    "AAVE_V3_ORACLE",
)


@pytest.mark.parametrize("chain", AAVE_V3_CHAINS)
@pytest.mark.parametrize("name", AAVE_V3_CONTRACTS)
def test_aave_v3_deployment_pinned(chain: int, name: str):
    """Every Aave V3 contract is pinned on every supported chain."""
    from pydefi.deployments import get_address

    addr = get_address(name, chain)
    assert len(addr) == 42 and addr.startswith("0x"), f"{name} on {chain}: {addr!r}"


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


# ---------------------------------------------------------------------------
# Deployment registry sanity
# ---------------------------------------------------------------------------


COMPOUND_V3_MARKETS = [
    ("COMPOUND_V3_USDC", ChainId.ETHEREUM),
    ("COMPOUND_V3_USDC", ChainId.ARBITRUM),
    ("COMPOUND_V3_USDC", ChainId.BASE),
    ("COMPOUND_V3_USDC", ChainId.POLYGON),
    ("COMPOUND_V3_USDC", ChainId.OPTIMISM),
    ("COMPOUND_V3_WETH", ChainId.ETHEREUM),
    ("COMPOUND_V3_WETH", ChainId.ARBITRUM),
    ("COMPOUND_V3_WETH", ChainId.BASE),
    ("COMPOUND_V3_WETH", ChainId.OPTIMISM),
    ("COMPOUND_V3_USDT", ChainId.ETHEREUM),
    ("COMPOUND_V3_USDT", ChainId.ARBITRUM),
]


@pytest.mark.parametrize("name,chain", COMPOUND_V3_MARKETS)
def test_compound_v3_market_pinned(name: str, chain: int):
    """Every Comet market entry resolves to a 20-byte address."""
    from pydefi.deployments import get_address

    addr = get_address(name, chain)
    assert len(addr) == 42 and addr.startswith("0x"), f"{name} on {chain}: {addr!r}"

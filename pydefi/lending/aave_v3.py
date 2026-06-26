"""
Aave V3 lending protocol integration.

Aave V3 exposes a single :class:`Pool` contract on each supported chain that
handles supply / withdraw / borrow / repay for every listed reserve. This
module wraps that contract plus the ``AaveProtocolDataProvider`` for the
richer configuration struct and the ``AaveOracle`` for asset prices.

Borrowing is variable-rate only — Aave removed stable-rate borrowing from
the protocol in 2023.

Docs: https://aave.com/docs/aave-v3/overview
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Literal

from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3
from web3.exceptions import Web3Exception

from pydefi._utils import resolve_amount, to_tx
from pydefi.abi.lending import (
    AAVE_V3_ADDRESSES_PROVIDER,
    AAVE_V3_DATA_PROVIDER,
    AAVE_V3_ORACLE,
    AAVE_V3_POOL,
)
from pydefi.deployments import get_address
from pydefi.exceptions import PydefiError
from pydefi.lending.utils import SECONDS_PER_YEAR, parse_health_factor
from pydefi.types import Address, Token, TokenAmount

#: 10**27 — Aave's fixed-point scale for rates and indices.
RAY: int = 10**27

#: Aave V3 ``borrow`` / ``repay`` take an ``interestRateMode`` arg; the
#: protocol now supports only variable rate, whose on-chain value is 2.
_VARIABLE_RATE_MODE: int = 2


def ray_rate_to_apy(rate_annual_ray: int) -> Decimal:
    """Convert Aave's RAY-scaled **annual** rate to a compounded APY.

    Aave stores ``currentLiquidityRate`` and ``currentVariableBorrowRate``
    as RAY (1e27)-scaled annual rates. On-chain interest accrual divides
    by :data:`SECONDS_PER_YEAR` to get a per-second rate, then compounds
    it — the APY displayed on the Aave UI uses the same formula::

        rate_per_second = rate_annual_ray / RAY / SECONDS_PER_YEAR
        apy             = (1 + rate_per_second) ** SECONDS_PER_YEAR - 1

    Args:
        rate_annual_ray: ``currentLiquidityRate`` /
            ``currentVariableBorrowRate`` as returned by Aave.

    Returns:
        APY as a :class:`~decimal.Decimal` fraction (e.g. ``0.045`` = 4.5%).
    """
    if rate_annual_ray == 0:
        return Decimal(0)
    # Bump precision for the exponentiation; the default 28 is too tight
    # when SECONDS_PER_YEAR is ~3.15e7.
    ctx = getcontext()
    prev_prec = ctx.prec
    ctx.prec = max(prev_prec, 50)
    try:
        per_second = Decimal(rate_annual_ray) / Decimal(RAY) / Decimal(SECONDS_PER_YEAR)
        apy = (Decimal(1) + per_second) ** SECONDS_PER_YEAR - Decimal(1)
    finally:
        ctx.prec = prev_prec
    return apy


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ReserveData:
    """Snapshot of one Aave V3 reserve's state.

    Attributes:
        token: The reserve asset.
        supply_apy: Variable supply rate, annualised (``0.045`` = 4.5%).
        variable_borrow_apy: Variable borrow rate, annualised.
        utilization: Fraction in ``[0, 1]`` — ``total_debt / total_supplied``.
            Returns ``0`` when the reserve has no supply.
        available_liquidity: Underlying tokens currently held by the aToken
            (i.e. immediately withdrawable / borrowable).
        total_debt: Total outstanding variable debt across all borrowers.
        liquidity_index: Aave's liquidityIndex (RAY-scaled).
        variable_borrow_index: Aave's variableBorrowIndex (RAY-scaled).
        a_token_address: The aToken receipt address.
        variable_debt_token_address: The variable debt token address.
        ltv: Loan-to-value ratio in bps (e.g. ``7500`` = 75 %).
        liquidation_threshold: Liquidation threshold in bps.
        liquidation_bonus: Liquidation bonus in bps (``10500`` = 5 % bonus).
        reserve_factor: Share of borrow interest paid to the DAO, in bps.
        is_active: Whether the reserve accepts new operations.
        is_frozen: Whether new supplies/borrows are disabled.
        is_paused: Whether the reserve is fully paused.
        borrowing_enabled: Whether borrowing is enabled on this reserve.
        usage_as_collateral_enabled: Whether the asset can be used as
            collateral by users.
    """

    token: Token
    supply_apy: Decimal
    variable_borrow_apy: Decimal
    utilization: Decimal
    available_liquidity: TokenAmount
    total_debt: TokenAmount
    liquidity_index: int
    variable_borrow_index: int
    a_token_address: Address
    variable_debt_token_address: Address
    ltv: int
    liquidation_threshold: int
    liquidation_bonus: int
    reserve_factor: int
    is_active: bool
    is_frozen: bool
    is_paused: bool
    borrowing_enabled: bool
    usage_as_collateral_enabled: bool


@dataclass
class EModeCategory:
    """Definition of an Aave V3 E-Mode category.

    E-Mode lets users opt into higher LTV/LT for *correlated* asset classes
    (e.g. all stablecoins, or all ETH-pegged tokens). The collateral and
    borrowable asset sets are independently configured as bitmaps over the
    full reserves list.

    Attributes:
        id: E-Mode category ID. ``0`` means "no E-Mode".
        label: Human-readable label (e.g. ``"ETH correlated"``).
        ltv: Max LTV for collateral inside this category, bps.
        liquidation_threshold: Liquidation threshold inside this category, bps.
        liquidation_bonus: Liquidation bonus, bps (``10100`` = 1 %).
        collateral_bitmap: Each set bit ``i`` means reserve at index ``i`` in
            ``Pool.getReservesList()`` is usable as collateral in this
            category.
        borrowable_bitmap: Each set bit ``i`` means reserve at index ``i`` is
            borrowable in this category.
    """

    id: int
    label: str
    ltv: int
    liquidation_threshold: int
    liquidation_bonus: int
    collateral_bitmap: int
    borrowable_bitmap: int


@dataclass
class UserReserveData:
    """A user's position in a single Aave V3 reserve.

    Attributes:
        token: The reserve asset.
        a_token_balance: Current supply balance (in token units).
        variable_debt: Current variable borrow balance.
        stable_debt: Current stable borrow balance (legacy; usually 0).
        usage_as_collateral_enabled: Whether the user has marked this asset
            as collateral.
    """

    token: Token
    a_token_balance: TokenAmount
    variable_debt: TokenAmount
    stable_debt: TokenAmount
    usage_as_collateral_enabled: bool


@dataclass
class UserAccountData:
    """A user's aggregate position across all Aave V3 reserves.

    All ``*_base`` values are denominated in the price oracle's base currency
    (Aave V3 mainnet uses USD with 8 decimals).

    Attributes:
        total_collateral_base: Total collateral value, oracle base units.
        total_debt_base: Total borrow value, oracle base units.
        available_borrows_base: Remaining borrowing capacity, oracle base
            units.
        current_liquidation_threshold: Effective liquidation threshold, bps.
        ltv: Effective max LTV, bps.
        health_factor: ``totalCollateralInETH * liquidationThreshold /
            totalDebtInETH``, scaled by 1e18 on-chain. Returns
            :attr:`decimal.Decimal('Infinity')` when the user has no debt
            (Aave returns ``type(uint256).max`` in that case).
    """

    total_collateral_base: int
    total_debt_base: int
    available_borrows_base: int
    current_liquidation_threshold: int
    ltv: int
    health_factor: Decimal


# ---------------------------------------------------------------------------
# AaveV3 client
# ---------------------------------------------------------------------------


class AaveV3:
    """Aave V3 lending market.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        chain_id: EVM chain ID.
        pool_address: Aave V3 ``Pool`` contract address.
        data_provider_address: ``AaveProtocolDataProvider`` address. Required
            for :meth:`get_reserve_data` (configuration bits) and
            :meth:`get_user_reserve_data`.
        oracle_address: Optional ``AaveOracle`` address (used by
            :meth:`get_asset_price`).
        protocol_name: Override the default ``"AaveV3"`` name.
    """

    protocol_name: str = "AaveV3"

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        pool_address: Address,
        data_provider_address: Address,
        oracle_address: Address | None = None,
        protocol_name: str = "AaveV3",
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.pool_address = pool_address
        self.data_provider_address = data_provider_address
        self.oracle_address = oracle_address
        self.protocol_name = protocol_name
        # Lazily-populated {symbol: reserve address} cache — see _resolve_reserve.
        self._reserves: dict[str, Address] | None = None

    @classmethod
    async def from_chain(cls, w3: AsyncWeb3, chain_id: int, protocol_name: str = "AaveV3") -> AaveV3:
        return await cls.from_addresses_provider(
            w3, chain_id, get_address("AAVE_V3_ADDRESSES_PROVIDER", chain_id), protocol_name
        )

    @classmethod
    async def from_addresses_provider(
        cls,
        w3: AsyncWeb3,
        chain_id: int,
        addresses_provider: Address,
        protocol_name: str = "AaveV3",
    ) -> "AaveV3":
        """Construct an :class:`AaveV3`, resolving ``Pool``,
        ``AaveProtocolDataProvider`` and ``AaveOracle`` from *addresses_provider*.

        Only the ``PoolAddressesProvider`` is stable enough to pin — Aave
        rotates the DataProvider on every protocol revision — so the three
        are read live. :meth:`from_chain` supplies the provider from the
        pydefi registry.
        """
        pool_str, dp_str, oracle_str = await asyncio.gather(
            AAVE_V3_ADDRESSES_PROVIDER.fns.getPool().call(w3, to=addresses_provider),
            AAVE_V3_ADDRESSES_PROVIDER.fns.getPoolDataProvider().call(w3, to=addresses_provider),
            AAVE_V3_ADDRESSES_PROVIDER.fns.getPriceOracle().call(w3, to=addresses_provider),
        )
        return cls(
            w3=w3,
            chain_id=chain_id,
            pool_address=Address(pool_str),
            data_provider_address=Address(dp_str),
            oracle_address=Address(oracle_str),
            protocol_name=protocol_name,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def _resolve_reserve(self, token: Token) -> Address:
        """Map *token*'s symbol to Aave's listed reserve address.

        Testnet markets list faucet tokens that differ from the registry
        address (``getReserveData`` reverts on a non-reserve). The
        ``getAllReservesTokens()`` map is cached; on mainnet this resolves to
        ``token.address`` unchanged.
        """
        # Fast path on mainnet, registry token address is already reserve
        # to avoid expensive getAllReservesTokens()
        if self._reserves is None and self.w3 is not None:
            try:
                reserve = await AAVE_V3_POOL.fns.getReserveData(token.address).call(self.w3, to=self.pool_address)
                if int(reserve.aTokenAddress, 16) != 0:
                    return token.address
            except Web3Exception:
                # Non-reserve address (common on testnets): fall through
                # to symbol-based resolution via DataProvider.
                pass

        if self._reserves is None:
            listed = await AAVE_V3_DATA_PROVIDER.fns.getAllReservesTokens().call(self.w3, to=self.data_provider_address)
            self._reserves = {symbol: Address(addr) for symbol, addr in listed}
        resolved = self._reserves.get(token.symbol)
        if resolved is None:
            raise PydefiError(
                f"AaveV3: no reserve for {token.symbol!r} on chain {self.chain_id} "
                f"(listed reserves: {sorted(self._reserves)})"
            )
        return resolved

    async def get_reserve_data(self, token: Token) -> ReserveData:
        """Return the current state of one reserve.

        Combines:

        * ``Pool.getReserveData`` — rates, indices, aToken / debtToken
          addresses.
        * ``DataProvider.getReserveConfigurationData`` — LTV, thresholds,
          flags.
        * ``DataProvider.getPaused`` — pause flag.
        * ``IERC20(underlying).balanceOf(aToken)`` — canonical available
          liquidity (the underlying held by the aToken contract). This
          matches the Aave UI and is correct in the presence of unbacked
          supply or treasury accrual, where ``aTokenTotalSupply - totalDebt``
          would drift.
        * ``DataProvider.getTotalDebt`` — stable + variable debt sum.
        """
        asset = await self._resolve_reserve(token)
        reserve, config, is_paused, total_debt = await asyncio.gather(
            AAVE_V3_POOL.fns.getReserveData(asset).call(self.w3, to=self.pool_address),
            AAVE_V3_DATA_PROVIDER.fns.getReserveConfigurationData(asset).call(self.w3, to=self.data_provider_address),
            AAVE_V3_DATA_PROVIDER.fns.getPaused(asset).call(self.w3, to=self.data_provider_address),
            AAVE_V3_DATA_PROVIDER.fns.getTotalDebt(asset).call(self.w3, to=self.data_provider_address),
        )

        # balanceOf needs the aToken address out of getReserveData, so it runs second.
        a_token_address = Address(reserve.aTokenAddress)
        available_liquidity = await ERC20.fns.balanceOf(a_token_address).call(self.w3, to=asset)

        (
            decimals,
            ltv,
            liquidation_threshold,
            liquidation_bonus,
            reserve_factor,
            usage_as_collateral_enabled,
            borrowing_enabled,
            _stable_enabled,
            is_active,
            is_frozen,
        ) = config

        # Aave's actual reserve asset (differs from the registry token on testnets).
        asset_token = Token(chain_id=token.chain_id, address=asset, symbol=token.symbol, decimals=decimals)

        # Aave's canonical utilization: debt / (debt + liquidity in the aToken).
        denominator = total_debt + available_liquidity
        utilization = Decimal(total_debt) / Decimal(denominator) if denominator > 0 else Decimal(0)

        return ReserveData(
            token=asset_token,
            supply_apy=ray_rate_to_apy(reserve.currentLiquidityRate),
            variable_borrow_apy=ray_rate_to_apy(reserve.currentVariableBorrowRate),
            utilization=utilization,
            available_liquidity=TokenAmount(token=asset_token, amount=available_liquidity),
            total_debt=TokenAmount(token=asset_token, amount=total_debt),
            liquidity_index=reserve.liquidityIndex,
            variable_borrow_index=reserve.variableBorrowIndex,
            a_token_address=a_token_address,
            variable_debt_token_address=Address(reserve.variableDebtTokenAddress),
            ltv=ltv,
            liquidation_threshold=liquidation_threshold,
            liquidation_bonus=liquidation_bonus,
            reserve_factor=reserve_factor,
            is_active=is_active,
            is_frozen=is_frozen,
            is_paused=is_paused,
            borrowing_enabled=borrowing_enabled,
            usage_as_collateral_enabled=usage_as_collateral_enabled,
        )

    async def get_user_account_data(self, user: Address) -> UserAccountData:
        """Return a user's aggregate position across all reserves."""
        (
            total_collateral_base,
            total_debt_base,
            available_borrows_base,
            current_liquidation_threshold,
            ltv,
            health_factor_raw,
        ) = await AAVE_V3_POOL.fns.getUserAccountData(user).call(self.w3, to=self.pool_address)

        return UserAccountData(
            total_collateral_base=total_collateral_base,
            total_debt_base=total_debt_base,
            available_borrows_base=available_borrows_base,
            current_liquidation_threshold=current_liquidation_threshold,
            ltv=ltv,
            health_factor=parse_health_factor(health_factor_raw),
        )

    async def get_user_reserve_data(self, user: Address, token: Token) -> UserReserveData:
        """Return a user's position in one reserve.

        Pulls balances and the collateral flag from
        ``DataProvider.getUserReserveData``. *token* is resolved by symbol via
        :meth:`_resolve_reserve` (a no-op on mainnet).
        """
        asset = await self._resolve_reserve(token)
        (
            current_a_token_balance,
            current_stable_debt,
            current_variable_debt,
            _principal_stable_debt,
            _scaled_variable_debt,
            _stable_borrow_rate,
            _liquidity_rate,
            _stable_rate_last_updated,
            usage_as_collateral_enabled,
        ) = await AAVE_V3_DATA_PROVIDER.fns.getUserReserveData(asset, user).call(self.w3, to=self.data_provider_address)

        return UserReserveData(
            token=token,
            a_token_balance=TokenAmount(token=token, amount=current_a_token_balance),
            variable_debt=TokenAmount(token=token, amount=current_variable_debt),
            stable_debt=TokenAmount(token=token, amount=current_stable_debt),
            usage_as_collateral_enabled=usage_as_collateral_enabled,
        )

    async def get_user_emode(self, user: Address) -> int:
        """Return the user's current E-Mode category ID (``0`` if disabled)."""
        return await AAVE_V3_POOL.fns.getUserEMode(user).call(self.w3, to=self.pool_address)

    async def get_emode_category(self, category_id: int) -> EModeCategory:
        """Return the full definition of an E-Mode category.

        On Aave V3.2+ the legacy ``getEModeCategoryData`` single getter was
        split into four (label / config / collateral bitmap / borrowable
        bitmap); this method fans them out into one :class:`EModeCategory`.
        """
        if not 0 <= category_id <= 255:
            raise ValueError("category_id must fit in uint8 (0–255)")
        label, collateral_config, coll_bitmap, borr_bitmap = await asyncio.gather(
            AAVE_V3_POOL.fns.getEModeCategoryLabel(category_id).call(self.w3, to=self.pool_address),
            AAVE_V3_POOL.fns.getEModeCategoryCollateralConfig(category_id).call(self.w3, to=self.pool_address),
            AAVE_V3_POOL.fns.getEModeCategoryCollateralBitmap(category_id).call(self.w3, to=self.pool_address),
            AAVE_V3_POOL.fns.getEModeCategoryBorrowableBitmap(category_id).call(self.w3, to=self.pool_address),
        )
        (ltv, liq_threshold, liq_bonus) = collateral_config
        return EModeCategory(
            id=category_id,
            label=label,
            ltv=ltv,
            liquidation_threshold=liq_threshold,
            liquidation_bonus=liq_bonus,
            collateral_bitmap=coll_bitmap,
            borrowable_bitmap=borr_bitmap,
        )

    async def get_asset_price(self, token: Token) -> int:
        """Return the oracle price of *token* in the oracle's base currency.

        Aave V3 mainnet uses USD with 8 decimals; some chains use ETH with
        18 decimals. Check ``AaveOracle.BASE_CURRENCY_UNIT()`` for the
        denominator.

        Raises:
            :class:`~pydefi.exceptions.PydefiError`: When the instance was
                constructed without an ``oracle_address``.
        """
        if self.oracle_address is None:
            raise PydefiError("AaveV3.get_asset_price requires oracle_address")
        return await AAVE_V3_ORACLE.fns.getAssetPrice(token.address).call(self.w3, to=self.oracle_address)

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_supply_tx(
        self,
        user: Address,
        amount: TokenAmount,
        on_behalf_of: Address | None = None,
        referral_code: int = 0,
    ) -> dict[str, Any]:
        on_behalf_of = on_behalf_of or user
        call_data = AAVE_V3_POOL.fns.supply(
            amount.token.address,
            amount.amount,
            on_behalf_of,
            referral_code,
        ).data
        return to_tx(self.pool_address, call_data)

    def build_withdraw_tx(
        self,
        user: Address,
        amount: TokenAmount | tuple[Token, Literal["max"]],
        to: Address | None = None,
    ) -> dict[str, Any]:
        recipient = to or user
        asset, raw = resolve_amount(amount)
        call_data = AAVE_V3_POOL.fns.withdraw(asset.address, raw, recipient).data
        return to_tx(self.pool_address, call_data)

    def build_borrow_tx(
        self,
        user: Address,
        amount: TokenAmount,
        on_behalf_of: Address | None = None,
        referral_code: int = 0,
    ) -> dict[str, Any]:
        on_behalf_of = on_behalf_of or user
        call_data = AAVE_V3_POOL.fns.borrow(
            amount.token.address,
            amount.amount,
            _VARIABLE_RATE_MODE,
            referral_code,
            on_behalf_of,
        ).data
        return to_tx(self.pool_address, call_data)

    def build_repay_tx(
        self,
        user: Address,
        amount: TokenAmount | tuple[Token, Literal["max"]],
        on_behalf_of: Address | None = None,
    ) -> dict[str, Any]:
        on_behalf_of = on_behalf_of or user
        asset, raw = resolve_amount(amount)
        call_data = AAVE_V3_POOL.fns.repay(asset.address, raw, _VARIABLE_RATE_MODE, on_behalf_of).data
        return to_tx(self.pool_address, call_data)

    def build_set_collateral_tx(
        self,
        token: Token,
        use_as_collateral: bool,
    ) -> dict[str, Any]:
        call_data = AAVE_V3_POOL.fns.setUserUseReserveAsCollateral(token.address, use_as_collateral).data
        return to_tx(self.pool_address, call_data)

    def build_flashloan_simple_tx(
        self,
        receiver: Address,
        amount: TokenAmount,
        params: bytes = b"",
        referral_code: int = 0,
    ) -> dict[str, Any]:
        """Build a single-asset ``flashLoanSimple`` transaction.

        The ``receiver`` contract must implement ``executeOperation`` from
        ``IFlashLoanSimpleReceiver``. Aave will mint ``amount`` of
        ``amount.token`` to ``receiver``, invoke ``executeOperation`` with
        ``params``, then pull back ``amount + premium`` via ``transferFrom``
        — the receiver is responsible for approving the Pool.
        """
        call_data = AAVE_V3_POOL.fns.flashLoanSimple(
            receiver,
            amount.token.address,
            amount.amount,
            params,
            referral_code,
        ).data
        return to_tx(self.pool_address, call_data)

    def build_set_emode_tx(self, category_id: int) -> dict[str, Any]:
        """Build a ``setUserEMode`` transaction. Pass ``0`` to disable E-Mode."""
        if not 0 <= category_id <= 255:
            raise ValueError("category_id must fit in uint8 (0–255)")
        call_data = AAVE_V3_POOL.fns.setUserEMode(category_id).data
        return to_tx(self.pool_address, call_data)

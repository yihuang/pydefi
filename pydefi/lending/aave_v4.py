"""
Aave V4 lending protocol integration.

Aave V4 splits the protocol into a per-network **Liquidity Hub** (the shared
liquidity layer) and user-facing **Spokes**. Users supply, borrow, withdraw
and repay through a Spoke — never the Hub directly — and each Spoke is scoped
to its own reserves and risk parameters.

A reserve inside a Spoke is addressed by its ``reserve_id``, an index in
``0 .. getReserveCount()-1`` rather than a token address;
:meth:`AaveV4.find_reserve_id` maps a ``(hub, asset_id)`` pair to one.

Docs: https://aave.com/docs/aave-v4
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3

from pydefi.abi.lending import (
    AAVE_V4_HUB,
    AAVE_V4_SPOKE,
    AAVE_V4_TOKENIZATION_SPOKE,
    AaveV4Asset,
)
from pydefi.deployments import get_address
from pydefi.lending.aave_v3 import RAY, ray_rate_to_apy
from pydefi.lending.utils import UINT256_MAX, WAD, to_tx
from pydefi.types import Address, Token, TokenAmount


def _supply_apy(drawn_rate: int, asset: AaveV4Asset) -> Decimal:
    """Derive a Hub asset's supply APY from its accounting state.

    Suppliers earn the drawn (borrow) rate scaled by Hub utilization and net
    of the liquidity fee: ``supplyRate = drawnRate * utilization * (1 - fee)``.
    Excludes per-position risk-premium income, so it is a slight under-estimate.
    """
    drawn = asset.drawnShares * asset.drawnIndex // RAY
    total = asset.liquidity + asset.swept + drawn
    if total == 0:
        return Decimal(0)
    supply_rate = drawn_rate * drawn // total * (10_000 - asset.liquidityFee) // 10_000
    return ray_rate_to_apy(supply_rate)


def parse_health_factor(raw: int) -> Decimal:
    """Convert Aave V4's WAD-scaled ``uint256`` health factor to a Decimal.

    ``getUserAccountData`` returns ``healthFactor`` scaled by 1e18 (so 1e18
    is a health factor of 1.00). A user with no debt has no meaningful
    ratio; Aave returns ``type(uint256).max`` there, surfaced here as
    ``Decimal("Infinity")``.
    """
    if raw == UINT256_MAX:
        return Decimal("Infinity")
    return Decimal(raw) / Decimal(WAD)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class V4Reserve:
    """Identity of one reserve inside a Spoke.

    Attributes:
        reserve_id: The reserve's index within the Spoke.
        underlying: The reserve asset. ``symbol`` is ``"?"`` unless a known
            :class:`Token` was supplied — Aave V4 exposes ``decimals`` but
            not the symbol on-chain.
        hub: The Liquidity Hub this reserve draws from.
        asset_id: The asset's identifier within that Hub.
        collateral_risk: Risk weight of the asset as collateral, in bps.
    """

    reserve_id: int
    underlying: Token
    hub: Address
    asset_id: int
    collateral_risk: int


@dataclass
class V4ReserveData:
    """Snapshot of one Spoke reserve's state.

    Attributes:
        reserve: The reserve's :class:`V4Reserve` identity.
        supplied: Total underlying supplied to the reserve.
        total_debt: Total outstanding debt (drawn + premium).
        utilization: ``total_debt / supplied`` for this reserve; ``0`` when
            the reserve has no supply. May exceed ``1`` — a Spoke reserve
            borrows from shared Hub liquidity, not just its own supply.
        borrow_apy: Annualised borrow rate, from the Hub's drawn rate
            (``0.045`` = 4.5%). Excludes any per-position risk premium.
        supply_apy: Annualised supply rate — the Hub's drawn rate scaled by
            Hub utilization, net of the liquidity fee.
        is_paused: Whether all reserve actions are blocked.
        is_frozen: Whether new supplies / borrows are blocked.
        borrowable: Whether the reserve can be borrowed.
    """

    reserve: V4Reserve
    supplied: TokenAmount
    total_debt: TokenAmount
    utilization: Decimal
    borrow_apy: Decimal
    supply_apy: Decimal
    is_paused: bool
    is_frozen: bool
    borrowable: bool


@dataclass
class V4UserReserve:
    """A user's position in a single Spoke reserve.

    Attributes:
        reserve_id: The reserve's index within the Spoke.
        supplied: The user's supplied balance, in underlying tokens.
        debt: The user's total debt (drawn + premium), in underlying tokens.
        using_as_collateral: Whether the user has enabled this reserve as
            collateral.
        borrowing: Whether the user currently borrows this reserve.
    """

    reserve_id: int
    supplied: TokenAmount
    debt: TokenAmount
    using_as_collateral: bool
    borrowing: bool


@dataclass
class V4UserAccountData:
    """A user's aggregate position on one Spoke (``getUserAccountData``).

    Attributes:
        risk_premium: The position's risk premium, in bps.
        avg_collateral_factor: Weighted-average collateral factor, WAD-scaled.
        health_factor: ``Decimal`` health factor; liquidatable below ``1``.
            ``Decimal("Infinity")`` when the user has no debt.
        total_collateral_value: Total collateral value, in oracle Value units.
        total_debt_value_ray: Total debt value, in Value units, RAY-scaled.
        active_collateral_count: Number of reserves actively collateralising.
        borrow_count: Number of reserves the user borrows.
    """

    risk_premium: int
    avg_collateral_factor: int
    health_factor: Decimal
    total_collateral_value: int
    total_debt_value_ray: int
    active_collateral_count: int
    borrow_count: int


# ---------------------------------------------------------------------------
# AaveV4 client
# ---------------------------------------------------------------------------


class AaveV4:
    """A single Aave V4 Spoke.

    Each Spoke is its own market surface (e.g. ``MAIN_SPOKE``,
    ``GOLD_SPOKE``). Construct one :class:`AaveV4` per Spoke you interact
    with — every read and write is scoped to that Spoke's reserves.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        chain_id: EVM chain ID.
        spoke_address: Address of the Spoke contract.
        protocol_name: Override the default ``"AaveV4"`` name.
    """

    protocol_name: str = "AaveV4"

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        spoke_address: Address,
        protocol_name: str = "AaveV4",
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.spoke_address = spoke_address
        self.protocol_name = protocol_name

    @classmethod
    def from_chain(cls, w3: AsyncWeb3, chain_id: int, spoke: str, protocol_name: str = "AaveV4") -> AaveV4:
        """Construct scoped to a named Spoke from the deployment registry.

        *spoke* is the registry suffix — ``"MAIN_SPOKE"``, ``"BLUECHIP_SPOKE"``,
        ``"GOLD_SPOKE"``, … — resolved to ``AAVE_V4_<spoke>``.
        """
        return cls(
            w3=w3,
            chain_id=chain_id,
            spoke_address=get_address(f"AAVE_V4_{spoke}", chain_id),
            protocol_name=protocol_name,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_reserve_count(self) -> int:
        """Return the number of reserves listed on this Spoke."""
        return await AAVE_V4_SPOKE.fns.getReserveCount().call(self.w3, to=self.spoke_address)

    async def get_reserve(self, reserve_id: int, token: Token | None = None) -> V4Reserve:
        """Return the identity of reserve *reserve_id*.

        Pass *token* to attach a known :class:`Token` (with a real symbol);
        otherwise the underlying is built from the on-chain address and
        decimals with ``symbol="?"``.
        """
        raw = await AAVE_V4_SPOKE.fns.getReserve(reserve_id).call(self.w3, to=self.spoke_address)
        underlying = token or Token(
            chain_id=self.chain_id, address=Address(raw.underlying), symbol="?", decimals=raw.decimals
        )
        return V4Reserve(
            reserve_id=reserve_id,
            underlying=underlying,
            hub=Address(raw.hub),
            asset_id=raw.assetId,
            collateral_risk=raw.collateralRisk,
        )

    async def get_reserve_data(self, reserve_id: int, token: Token | None = None) -> V4ReserveData:
        """Return the current state of reserve *reserve_id*."""
        reserve, config, supplied, total_debt = await asyncio.gather(
            self.get_reserve(reserve_id, token),
            AAVE_V4_SPOKE.fns.getReserveConfig(reserve_id).call(self.w3, to=self.spoke_address),
            AAVE_V4_SPOKE.fns.getReserveSuppliedAssets(reserve_id).call(self.w3, to=self.spoke_address),
            AAVE_V4_SPOKE.fns.getReserveTotalDebt(reserve_id).call(self.w3, to=self.spoke_address),
        )
        utilization = Decimal(total_debt) / Decimal(supplied) if supplied > 0 else Decimal(0)
        # Rates live on the Hub the reserve draws liquidity from: the fresh
        # drawn rate plus the Asset struct (for Hub utilization + the fee).
        drawn_rate, asset = await asyncio.gather(
            AAVE_V4_HUB.fns.getAssetDrawnRate(reserve.asset_id).call(self.w3, to=reserve.hub),
            AAVE_V4_HUB.fns.getAsset(reserve.asset_id).call(self.w3, to=reserve.hub),
        )
        return V4ReserveData(
            reserve=reserve,
            supplied=TokenAmount(token=reserve.underlying, amount=supplied),
            total_debt=TokenAmount(token=reserve.underlying, amount=total_debt),
            utilization=utilization,
            borrow_apy=ray_rate_to_apy(drawn_rate),
            supply_apy=_supply_apy(drawn_rate, asset),
            is_paused=config.paused,
            is_frozen=config.frozen,
            borrowable=config.borrowable,
        )

    async def find_reserve_id(self, hub: Address, asset_id: int) -> int:
        """Return the ``reserve_id`` for ``(hub, asset_id)`` on this Spoke."""
        return await AAVE_V4_SPOKE.fns.getReserveId(hub, asset_id).call(self.w3, to=self.spoke_address)

    async def get_user_reserve(self, reserve_id: int, user: Address, token: Token | None = None) -> V4UserReserve:
        """Return *user*'s position in reserve *reserve_id*."""
        reserve, supplied, debt, status = await asyncio.gather(
            self.get_reserve(reserve_id, token),
            AAVE_V4_SPOKE.fns.getUserSuppliedAssets(reserve_id, user).call(self.w3, to=self.spoke_address),
            AAVE_V4_SPOKE.fns.getUserTotalDebt(reserve_id, user).call(self.w3, to=self.spoke_address),
            AAVE_V4_SPOKE.fns.getUserReserveStatus(reserve_id, user).call(self.w3, to=self.spoke_address),
        )
        # getUserReserveStatus: (enabledAsCollateral, borrowed)
        using_as_collateral, borrowing = status
        return V4UserReserve(
            reserve_id=reserve_id,
            supplied=TokenAmount(token=reserve.underlying, amount=supplied),
            debt=TokenAmount(token=reserve.underlying, amount=debt),
            using_as_collateral=using_as_collateral,
            borrowing=borrowing,
        )

    async def get_user_account_data(self, user: Address) -> V4UserAccountData:
        """Return *user*'s aggregate position on this Spoke."""
        raw = await AAVE_V4_SPOKE.fns.getUserAccountData(user).call(self.w3, to=self.spoke_address)
        return V4UserAccountData(
            risk_premium=raw.riskPremium,
            avg_collateral_factor=raw.avgCollateralFactor,
            health_factor=parse_health_factor(raw.healthFactor),
            total_collateral_value=raw.totalCollateralValue,
            total_debt_value_ray=raw.totalDebtValueRay,
            active_collateral_count=raw.activeCollateralCount,
            borrow_count=raw.borrowCount,
        )

    async def is_position_manager(self, user: Address, position_manager: Address) -> bool:
        """Return whether *position_manager* may act on *user*'s positions."""
        return await AAVE_V4_SPOKE.fns.isPositionManager(user, position_manager).call(self.w3, to=self.spoke_address)

    async def is_position_manager_active(self, position_manager: Address) -> bool:
        """Return whether *position_manager* is enabled on this Spoke."""
        return await AAVE_V4_SPOKE.fns.isPositionManagerActive(position_manager).call(self.w3, to=self.spoke_address)

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_supply_tx(self, reserve_id: int, amount: TokenAmount, on_behalf_of: Address) -> dict[str, Any]:
        """Supply ``amount`` of reserve *reserve_id*'s asset for *on_behalf_of*.

        The caller must have approved the Spoke to pull the ERC-20 first.
        """
        call_data = AAVE_V4_SPOKE.fns.supply(reserve_id, amount.amount, on_behalf_of).data
        return to_tx(self.spoke_address, call_data)

    def build_withdraw_tx(self, reserve_id: int, amount: TokenAmount, on_behalf_of: Address) -> dict[str, Any]:
        """Withdraw ``amount`` of reserve *reserve_id*'s asset from *on_behalf_of*."""
        call_data = AAVE_V4_SPOKE.fns.withdraw(reserve_id, amount.amount, on_behalf_of).data
        return to_tx(self.spoke_address, call_data)

    def build_borrow_tx(self, reserve_id: int, amount: TokenAmount, on_behalf_of: Address) -> dict[str, Any]:
        """Borrow ``amount`` of reserve *reserve_id*'s asset for *on_behalf_of*."""
        call_data = AAVE_V4_SPOKE.fns.borrow(reserve_id, amount.amount, on_behalf_of).data
        return to_tx(self.spoke_address, call_data)

    def build_repay_tx(self, reserve_id: int, amount: TokenAmount, on_behalf_of: Address) -> dict[str, Any]:
        """Repay ``amount`` of reserve *reserve_id*'s debt for *on_behalf_of*."""
        call_data = AAVE_V4_SPOKE.fns.repay(reserve_id, amount.amount, on_behalf_of).data
        return to_tx(self.spoke_address, call_data)

    def build_set_collateral_tx(
        self, reserve_id: int, use_as_collateral: bool, on_behalf_of: Address
    ) -> dict[str, Any]:
        """Enable or disable reserve *reserve_id* as collateral for *on_behalf_of*."""
        call_data = AAVE_V4_SPOKE.fns.setUsingAsCollateral(reserve_id, use_as_collateral, on_behalf_of).data
        return to_tx(self.spoke_address, call_data)

    def build_liquidation_call_tx(
        self,
        collateral_reserve_id: int,
        debt_reserve_id: int,
        user: Address,
        debt_to_cover: TokenAmount,
        receive_shares: bool = False,
    ) -> dict[str, Any]:
        """Liquidate *user*'s unhealthy position — repay up to *debt_to_cover*
        of reserve *debt_reserve_id* and seize collateral *collateral_reserve_id*.

        With *receive_shares* the liquidator receives the seized collateral as
        Spoke supply shares instead of the underlying token.
        """
        call_data = AAVE_V4_SPOKE.fns.liquidationCall(
            collateral_reserve_id, debt_reserve_id, user, debt_to_cover.amount, receive_shares
        ).data
        return to_tx(self.spoke_address, call_data)

    def build_set_position_manager_tx(self, position_manager: Address, approve: bool) -> dict[str, Any]:
        """Grant or revoke *position_manager*'s right to act on the sender's positions."""
        call_data = AAVE_V4_SPOKE.fns.setUserPositionManager(position_manager, approve).data
        return to_tx(self.spoke_address, call_data)


# ---------------------------------------------------------------------------
# TokenizationSpoke client
# ---------------------------------------------------------------------------


@dataclass
class V4TokenizedPosition:
    """A holder's stake in a TokenizationSpoke vault.

    Attributes:
        shares: Vault share-token balance.
        assets: Value of those shares in the underlying asset.
    """

    shares: int
    assets: TokenAmount


class AaveV4TokenizationSpoke:
    """An Aave V4 TokenizationSpoke — a standard ERC-4626 vault that wraps a
    single ``(Hub, asset)`` Spoke supply position as a transferable token.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        chain_id: EVM chain ID.
        spoke_address: Address of the TokenizationSpoke contract.
        asset_token: Optional pre-resolved underlying :class:`Token`; when
            omitted it is fetched lazily by :meth:`get_asset`.
        protocol_name: Override the default name.
    """

    protocol_name: str = "AaveV4TokenizationSpoke"

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        spoke_address: Address,
        asset_token: Token | None = None,
        protocol_name: str = "AaveV4TokenizationSpoke",
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.spoke_address = spoke_address
        self.protocol_name = protocol_name
        self._asset_token = asset_token

    @classmethod
    def from_chain(
        cls,
        w3: AsyncWeb3,
        chain_id: int,
        spoke: str,
        asset_token: Token | None = None,
    ) -> AaveV4TokenizationSpoke:
        """Construct from the registry — *spoke* is a name like
        ``"CORE_WETH_TOKENIZATION_SPOKE"`` (resolved to ``AAVE_V4_<spoke>``)."""
        return cls(
            w3=w3,
            chain_id=chain_id,
            spoke_address=get_address(f"AAVE_V4_{spoke}", chain_id),
            asset_token=asset_token,
        )

    async def get_asset(self) -> Token:
        """Return the vault's underlying asset as a :class:`Token` (cached)."""
        if self._asset_token is not None:
            return self._asset_token
        address = Address(await AAVE_V4_TOKENIZATION_SPOKE.fns.asset().call(self.w3, to=self.spoke_address))
        decimals = await ERC20.fns.decimals().call(self.w3, to=address)
        self._asset_token = Token(chain_id=self.chain_id, address=address, symbol="?", decimals=decimals)
        return self._asset_token

    async def total_assets(self) -> TokenAmount:
        """Total underlying assets held by the vault."""
        asset, raw = await asyncio.gather(
            self.get_asset(),
            AAVE_V4_TOKENIZATION_SPOKE.fns.totalAssets().call(self.w3, to=self.spoke_address),
        )
        return TokenAmount(token=asset, amount=raw)

    async def convert_to_assets(self, shares: int) -> int:
        """Underlying assets *shares* vault tokens are currently worth."""
        return await AAVE_V4_TOKENIZATION_SPOKE.fns.convertToAssets(shares).call(self.w3, to=self.spoke_address)

    async def convert_to_shares(self, assets: int) -> int:
        """Vault shares minted for *assets* of the underlying."""
        return await AAVE_V4_TOKENIZATION_SPOKE.fns.convertToShares(assets).call(self.w3, to=self.spoke_address)

    async def get_position(self, user: Address) -> V4TokenizedPosition:
        """Return *user*'s vault holding — share balance and its asset value."""
        asset, shares = await asyncio.gather(
            self.get_asset(),
            AAVE_V4_TOKENIZATION_SPOKE.fns.balanceOf(user).call(self.w3, to=self.spoke_address),
        )
        assets = await self.convert_to_assets(shares)
        return V4TokenizedPosition(shares=shares, assets=TokenAmount(token=asset, amount=assets))

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_deposit_tx(self, amount: TokenAmount, receiver: Address) -> dict[str, Any]:
        """Deposit ``amount`` of the underlying, minting vault shares to *receiver*."""
        call_data = AAVE_V4_TOKENIZATION_SPOKE.fns.deposit(amount.amount, receiver).data
        return to_tx(self.spoke_address, call_data)

    def build_mint_tx(self, shares: int, receiver: Address) -> dict[str, Any]:
        """Mint exactly *shares* vault tokens to *receiver*, pulling the underlying."""
        call_data = AAVE_V4_TOKENIZATION_SPOKE.fns.mint(shares, receiver).data
        return to_tx(self.spoke_address, call_data)

    def build_withdraw_tx(self, amount: TokenAmount, receiver: Address, owner: Address) -> dict[str, Any]:
        """Withdraw ``amount`` of the underlying from *owner*'s shares to *receiver*."""
        call_data = AAVE_V4_TOKENIZATION_SPOKE.fns.withdraw(amount.amount, receiver, owner).data
        return to_tx(self.spoke_address, call_data)

    def build_redeem_tx(self, shares: int, receiver: Address, owner: Address) -> dict[str, Any]:
        """Redeem *shares* of *owner*'s vault tokens for the underlying, to *receiver*."""
        call_data = AAVE_V4_TOKENIZATION_SPOKE.fns.redeem(shares, receiver, owner).data
        return to_tx(self.spoke_address, call_data)


__all__ = [
    "AaveV4",
    "AaveV4TokenizationSpoke",
    "V4Reserve",
    "V4ReserveData",
    "V4TokenizedPosition",
    "V4UserAccountData",
    "V4UserReserve",
    "parse_health_factor",
]

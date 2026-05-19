"""
Compound III (Comet) lending market integration.

Compound III is structured very differently from Aave V3:

* Every "market" is a separate :class:`Comet` proxy with **one base asset**
  that can be both supplied and borrowed. The market's APY and utilization
  are properties of that single base asset.
* All other listed assets are **collateral-only** — they can be supplied
  (earning no interest) and must back any open base-asset borrow, but they
  cannot be borrowed.
* ``supply`` / ``withdraw`` work for both the base asset and any collateral
  asset; passing the base asset to ``withdraw`` opens (or grows) a borrow
  position once the net base balance crosses zero.

Because of those structural differences :class:`CompoundV3` shares no
abstraction with :class:`~pydefi.lending.aave_v3.AaveV3`; the two clients
live side by side in :mod:`pydefi.lending` but otherwise have no common
base.

Docs: https://docs.compound.finance/
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from web3 import AsyncWeb3

from pydefi.abi.lending import COMPOUND_V3_COMET
from pydefi.types import Address, Token, TokenAmount

#: Compound III scales rates and factors by 1e18 (no separate "RAY").
COMET_SCALE: int = 10**18

#: Seconds per calendar year — matches Aave's convention.
SECONDS_PER_YEAR: int = 31_536_000

#: ``type(uint256).max`` sentinel meaning "full balance" for ``withdraw``.
UINT256_MAX: int = (1 << 256) - 1


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class CompoundMarketData:
    """Snapshot of one Comet market.

    Attributes:
        base_token: The base (borrowable) asset.
        supply_apy: Annualised supply rate on the base asset.
        borrow_apy: Annualised borrow rate on the base asset.
        utilization: ``totalBorrow / totalSupply`` for the base asset,
            scaled by 1e18 on-chain — surfaced here as a ``Decimal`` in
            ``[0, 1]``.
        total_supply: Total base asset supplied (positive net balance only).
        total_borrow: Total base asset borrowed.
        base_price_feed: Address of the Comet-internal base asset price feed.
    """

    base_token: Token
    supply_apy: Decimal
    borrow_apy: Decimal
    utilization: Decimal
    total_supply: TokenAmount
    total_borrow: TokenAmount
    base_price_feed: Address


@dataclass
class CompoundCollateralInfo:
    """Configuration of one collateral asset within a Comet market.

    Attributes:
        offset: Index in the market's collateral list (0-based, < numAssets).
        asset: Collateral token.
        price_feed: Comet-internal price feed address.
        scale: Token scale (``10 ** decimals``).
        borrow_collateral_factor: Borrow collateral factor (1e18-scaled).
            Determines how much base asset can be borrowed against this
            collateral.
        liquidate_collateral_factor: Liquidation collateral factor
            (1e18-scaled). Once total borrows exceed
            ``collateral_value * liquidate_collateral_factor`` the account
            becomes liquidatable.
        liquidation_factor: Penalty factor applied to seized collateral
            on liquidation (1e18-scaled).
        supply_cap: Maximum total supply of this collateral, in token units.
    """

    offset: int
    asset: Token
    price_feed: Address
    scale: int
    borrow_collateral_factor: int
    liquidate_collateral_factor: int
    liquidation_factor: int
    supply_cap: int


@dataclass
class CompoundUserPosition:
    """A user's full position in a single Comet market.

    Attributes:
        base_supply: Base asset balance (positive only — see
            ``Comet.balanceOf``).
        base_borrow: Base asset debt (positive only — see
            ``Comet.borrowBalanceOf``).  Exactly one of ``base_supply`` /
            ``base_borrow`` is non-zero in steady state.
        collateral: Collateral balances keyed by asset address.
        is_liquidatable: Whether the account is currently liquidatable.
    """

    base_supply: TokenAmount
    base_borrow: TokenAmount
    collateral: dict[Address, TokenAmount]
    is_liquidatable: bool


# ---------------------------------------------------------------------------
# Rate model helper
# ---------------------------------------------------------------------------


def per_second_rate_to_apy(rate_per_second_1e18: int) -> Decimal:
    """Compound III returns rates as **per-second** values, scaled by 1e18.

    Annualised APY is then ``(1 + r) ** SECONDS_PER_YEAR - 1`` where
    ``r = rate_per_second_1e18 / 1e18``. Matches the formula on
    `compound.finance <https://docs.compound.finance/interest-rates/>`_.
    """
    if rate_per_second_1e18 == 0:
        return Decimal(0)
    from decimal import getcontext

    ctx = getcontext()
    prev = ctx.prec
    ctx.prec = max(prev, 50)
    try:
        per_second = Decimal(rate_per_second_1e18) / Decimal(COMET_SCALE)
        apy = (Decimal(1) + per_second) ** SECONDS_PER_YEAR - Decimal(1)
    finally:
        ctx.prec = prev
    return apy


# ---------------------------------------------------------------------------
# CompoundV3 client
# ---------------------------------------------------------------------------


class CompoundV3:
    """A single Compound III market.

    Each Comet proxy is its own market (e.g. ``cUSDCv3`` on Ethereum).
    Construct one :class:`CompoundV3` per market you want to interact with.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        chain_id: EVM chain ID.
        comet_address: Address of the Comet proxy.
        base_token: Optional pre-resolved base :class:`Token`. When omitted,
            decimals are fetched lazily via :meth:`get_base_token`.
        protocol_name: Override the default ``"CompoundV3"`` name.
    """

    protocol_name: str = "CompoundV3"

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        comet_address: Address,
        base_token: Token | None = None,
        protocol_name: str = "CompoundV3",
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.comet_address = comet_address
        self.protocol_name = protocol_name
        self._base_token: Token | None = base_token

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_base_token(self) -> Token:
        """Return the market's base asset as a :class:`Token`.

        Caches the result after the first call. Decimals are derived from
        ``baseScale``; symbol is set to ``"BASE"`` because Comet does not
        expose it directly — callers that need a friendlier symbol should
        pass ``base_token`` to the constructor.
        """
        if self._base_token is not None:
            return self._base_token
        addr_str = await COMPOUND_V3_COMET.fns.baseToken().call(self.w3, to=self.comet_address)
        scale = await COMPOUND_V3_COMET.fns.baseScale().call(self.w3, to=self.comet_address)
        decimals = scale.bit_length() - 1 if scale > 0 else 18
        # 10 ** decimals must equal scale; if not, fall back to log10.
        if 10**decimals != scale:
            from math import log10

            decimals = round(log10(scale))
        token = Token(
            chain_id=self.chain_id,
            address=Address(addr_str),
            symbol="BASE",
            decimals=decimals,
        )
        self._base_token = token
        return token

    async def get_market_data(self) -> CompoundMarketData:
        """Return the current state of the base asset side of this market."""
        base = await self.get_base_token()
        total_supply = await COMPOUND_V3_COMET.fns.totalSupply().call(self.w3, to=self.comet_address)
        total_borrow = await COMPOUND_V3_COMET.fns.totalBorrow().call(self.w3, to=self.comet_address)
        utilization_raw = await COMPOUND_V3_COMET.fns.getUtilization().call(self.w3, to=self.comet_address)
        supply_rate = await COMPOUND_V3_COMET.fns.getSupplyRate(utilization_raw).call(self.w3, to=self.comet_address)
        borrow_rate = await COMPOUND_V3_COMET.fns.getBorrowRate(utilization_raw).call(self.w3, to=self.comet_address)
        base_price_feed_str = await COMPOUND_V3_COMET.fns.baseTokenPriceFeed().call(self.w3, to=self.comet_address)

        return CompoundMarketData(
            base_token=base,
            supply_apy=per_second_rate_to_apy(supply_rate),
            borrow_apy=per_second_rate_to_apy(borrow_rate),
            utilization=Decimal(utilization_raw) / Decimal(COMET_SCALE),
            total_supply=TokenAmount(token=base, amount=total_supply),
            total_borrow=TokenAmount(token=base, amount=total_borrow),
            base_price_feed=Address(base_price_feed_str),
        )

    async def get_collateral_assets(self, tokens: dict[Address, Token] | None = None) -> list[CompoundCollateralInfo]:
        """List every collateral asset configured for this market.

        Args:
            tokens: Optional address → :class:`Token` map for human-readable
                symbols and decimals. When omitted, returned collateral
                tokens have ``symbol="?"`` and ``decimals`` derived from the
                ``scale`` field.
        """
        tokens = tokens or {}
        num_assets = await COMPOUND_V3_COMET.fns.numAssets().call(self.w3, to=self.comet_address)
        out: list[CompoundCollateralInfo] = []
        for i in range(num_assets):
            info = await COMPOUND_V3_COMET.fns.getAssetInfo(i).call(self.w3, to=self.comet_address)
            offset = info[0]
            asset_addr = Address(info[1])
            price_feed = Address(info[2])
            scale = info[3]
            borrow_cf = info[4]
            liq_cf = info[5]
            liq_factor = info[6]
            supply_cap = info[7]

            token = tokens.get(asset_addr)
            if token is None:
                decimals = _scale_to_decimals(scale)
                token = Token(chain_id=self.chain_id, address=asset_addr, symbol="?", decimals=decimals)

            out.append(
                CompoundCollateralInfo(
                    offset=offset,
                    asset=token,
                    price_feed=price_feed,
                    scale=scale,
                    borrow_collateral_factor=borrow_cf,
                    liquidate_collateral_factor=liq_cf,
                    liquidation_factor=liq_factor,
                    supply_cap=supply_cap,
                )
            )
        return out

    async def get_user_position(
        self,
        user: Address,
        collateral_assets: list[CompoundCollateralInfo] | None = None,
    ) -> CompoundUserPosition:
        """Return the user's base balance, borrow balance, and collateral.

        Args:
            user: Account to query.
            collateral_assets: Pre-fetched collateral list. When omitted,
                :meth:`get_collateral_assets` is called once internally.
        """
        base = await self.get_base_token()
        if collateral_assets is None:
            collateral_assets = await self.get_collateral_assets()

        base_supply = await COMPOUND_V3_COMET.fns.balanceOf(user).call(self.w3, to=self.comet_address)
        base_borrow = await COMPOUND_V3_COMET.fns.borrowBalanceOf(user).call(self.w3, to=self.comet_address)
        is_liquidatable = await COMPOUND_V3_COMET.fns.isLiquidatable(user).call(self.w3, to=self.comet_address)

        collateral: dict[Address, TokenAmount] = {}
        for info in collateral_assets:
            balance = await COMPOUND_V3_COMET.fns.collateralBalanceOf(user, info.asset.address).call(
                self.w3, to=self.comet_address
            )
            collateral[info.asset.address] = TokenAmount(token=info.asset, amount=balance)

        return CompoundUserPosition(
            base_supply=TokenAmount(token=base, amount=base_supply),
            base_borrow=TokenAmount(token=base, amount=base_borrow),
            collateral=collateral,
            is_liquidatable=is_liquidatable,
        )

    async def get_price(self, price_feed: Address) -> int:
        """Read a Comet-internal price feed. Result is 1e8-scaled USD."""
        return await COMPOUND_V3_COMET.fns.getPrice(price_feed).call(self.w3, to=self.comet_address)

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_supply_tx(self, amount: TokenAmount, dst: Address | None = None) -> dict[str, Any]:
        """Build a ``supply`` (or ``supplyTo``) transaction.

        Compound III uses the same function for base and collateral
        deposits — the contract dispatches by checking ``amount.token``
        against the market's base asset.
        """
        if dst is None:
            call_data = COMPOUND_V3_COMET.fns.supply(amount.token.address, amount.amount).data
        else:
            call_data = COMPOUND_V3_COMET.fns.supplyTo(dst, amount.token.address, amount.amount).data
        return _to_tx(self.comet_address, call_data)

    def build_withdraw_tx(
        self,
        amount: TokenAmount | tuple[Token, Literal["max"]],
        to: Address | None = None,
    ) -> dict[str, Any]:
        """Build a ``withdraw`` (or ``withdrawTo``) transaction.

        Passing ``(token, "max")`` uses the ``type(uint256).max`` sentinel.
        Note that withdrawing the **base asset** below your current supply
        balance silently opens a borrow position — Comet has no separate
        ``borrow`` function.
        """
        asset, raw = _resolve_amount(amount)
        if to is None:
            call_data = COMPOUND_V3_COMET.fns.withdraw(asset.address, raw).data
        else:
            call_data = COMPOUND_V3_COMET.fns.withdrawTo(to, asset.address, raw).data
        return _to_tx(self.comet_address, call_data)

    def build_transfer_asset_tx(self, dst: Address, amount: TokenAmount) -> dict[str, Any]:
        """Build a ``transferAsset`` transaction.

        Moves any asset (base or collateral) inside the market between two
        accounts without touching the underlying ERC-20.
        """
        call_data = COMPOUND_V3_COMET.fns.transferAsset(dst, amount.token.address, amount.amount).data
        return _to_tx(self.comet_address, call_data)

    def build_allow_tx(self, manager: Address, is_allowed: bool) -> dict[str, Any]:
        """Build an ``allow`` transaction to grant or revoke manager rights."""
        call_data = COMPOUND_V3_COMET.fns.allow(manager, is_allowed).data
        return _to_tx(self.comet_address, call_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_amount(amount: TokenAmount | tuple[Token, Literal["max"]]) -> tuple[Token, int]:
    if isinstance(amount, TokenAmount):
        return amount.token, amount.amount
    if isinstance(amount, tuple) and len(amount) == 2 and amount[1] == "max":
        return amount[0], UINT256_MAX
    raise TypeError("amount must be a TokenAmount or (Token, 'max') tuple")


def _to_tx(to: Address, call_data: bytes) -> dict[str, Any]:
    return {
        "to": "0x" + bytes(to).hex(),
        "data": "0x" + call_data.hex(),
        "value": "0",
    }


def _scale_to_decimals(scale: int) -> int:
    """Derive ``decimals`` from a Comet ``scale`` (``10 ** decimals``)."""
    if scale <= 0:
        return 18
    decimals = scale.bit_length() - 1
    if 10**decimals == scale:
        return decimals
    from math import log10

    return round(log10(scale))


__all__ = [
    "COMET_SCALE",
    "SECONDS_PER_YEAR",
    "UINT256_MAX",
    "CompoundCollateralInfo",
    "CompoundMarketData",
    "CompoundUserPosition",
    "CompoundV3",
    "per_second_rate_to_apy",
]

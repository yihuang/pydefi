"""
Lending protocol ABI definitions.

Human-readable ABI fragments and pre-built :class:`~eth_contract.Contract`
objects for lending protocols. Bind a contract to a specific on-chain address
at the call site::

    from pydefi.abi.lending import AAVE_V3_POOL

    pool = AAVE_V3_POOL(to="0xPool...")
    await pool.fns.supply(asset, amount, on_behalf_of, referral_code).transact(w3, account)
"""

from __future__ import annotations

from typing import Annotated

from eth_contract import ABIStruct, Contract

# ---------------------------------------------------------------------------
# Aave V3 — ABI struct definitions
# ---------------------------------------------------------------------------


class ReserveConfigurationMap(ABIStruct):
    """Bit-packed reserve configuration (Aave V3 ``DataTypes.ReserveConfigurationMap``)."""

    data: Annotated[int, "uint256"]


class ReserveDataLegacy(ABIStruct):
    """Aave V3 ``DataTypes.ReserveDataLegacy`` (returned by ``Pool.getReserveData``).

    Note: ``currentLiquidityRate`` / ``currentVariableBorrowRate`` /
    ``currentStableBorrowRate`` are RAY (1e27)-scaled **annual** rates;
    Aave divides by ``SECONDS_PER_YEAR`` to get a per-second rate and then
    compounds on-chain. See :func:`pydefi.lending.aave_v3.ray_rate_to_apy`.
    """

    configuration: ReserveConfigurationMap
    liquidityIndex: Annotated[int, "uint128"]
    currentLiquidityRate: Annotated[int, "uint128"]
    variableBorrowIndex: Annotated[int, "uint128"]
    currentVariableBorrowRate: Annotated[int, "uint128"]
    currentStableBorrowRate: Annotated[int, "uint128"]
    lastUpdateTimestamp: Annotated[int, "uint40"]
    id: Annotated[int, "uint16"]
    aTokenAddress: Annotated[str, "address"]
    stableDebtTokenAddress: Annotated[str, "address"]
    variableDebtTokenAddress: Annotated[str, "address"]
    interestRateStrategyAddress: Annotated[str, "address"]
    accruedToTreasury: Annotated[int, "uint128"]
    unbacked: Annotated[int, "uint128"]
    isolationModeTotalDebt: Annotated[int, "uint128"]


# ---------------------------------------------------------------------------
# Aave V3 — Pool
# ---------------------------------------------------------------------------

AAVE_V3_POOL = Contract.from_abi(
    ReserveConfigurationMap.human_readable_abi()
    + ReserveDataLegacy.human_readable_abi()
    + [
        # Writes
        "function supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external",
        "function withdraw(address asset, uint256 amount, address to) external returns (uint256)",
        "function borrow(address asset, uint256 amount, uint256 interestRateMode, uint16 referralCode, address onBehalfOf) external",
        "function repay(address asset, uint256 amount, uint256 interestRateMode, address onBehalfOf) external returns (uint256)",
        "function setUserUseReserveAsCollateral(address asset, bool useAsCollateral) external",
        "function setUserEMode(uint8 categoryId) external",
        # Flash loans (single-asset).
        "function flashLoanSimple(address receiverAddress, address asset, uint256 amount, bytes calldata params, uint16 referralCode) external",
        # Reads
        "function getReserveData(address asset) external view returns (ReserveDataLegacy)",
        "function getUserAccountData(address user) external view returns (uint256 totalCollateralBase, uint256 totalDebtBase, uint256 availableBorrowsBase, uint256 currentLiquidationThreshold, uint256 ltv, uint256 healthFactor)",
        "function getReserveNormalizedIncome(address asset) external view returns (uint256)",
        "function getReserveNormalizedVariableDebt(address asset) external view returns (uint256)",
        "function getUserEMode(address user) external view returns (uint256)",
        # E-Mode reads (Aave V3.2+ layout — legacy single-call getEModeCategoryData was split).
        "function getEModeCategoryLabel(uint8 id) external view returns (string)",
        "function getEModeCategoryCollateralConfig(uint8 id) external view returns (uint16 ltv, uint16 liquidationThreshold, uint16 liquidationBonus)",
        "function getEModeCategoryCollateralBitmap(uint8 id) external view returns (uint128)",
        "function getEModeCategoryBorrowableBitmap(uint8 id) external view returns (uint128)",
        "function getReservesList() external view returns (address[])",
    ]
)


# ---------------------------------------------------------------------------
# Aave V3 — Protocol Data Provider
# ---------------------------------------------------------------------------

AAVE_V3_DATA_PROVIDER = Contract.from_abi(
    [
        "function getUserReserveData(address asset, address user) external view returns (uint256 currentATokenBalance, uint256 currentStableDebt, uint256 currentVariableDebt, uint256 principalStableDebt, uint256 scaledVariableDebt, uint256 stableBorrowRate, uint256 liquidityRate, uint40 stableRateLastUpdated, bool usageAsCollateralEnabled)",
        "function getReserveConfigurationData(address asset) external view returns (uint256 decimals, uint256 ltv, uint256 liquidationThreshold, uint256 liquidationBonus, uint256 reserveFactor, bool usageAsCollateralEnabled, bool borrowingEnabled, bool stableBorrowRateEnabled, bool isActive, bool isFrozen)",
        "function getReserveCaps(address asset) external view returns (uint256 borrowCap, uint256 supplyCap)",
        "function getPaused(address asset) external view returns (bool isPaused)",
        "function getReserveTokensAddresses(address asset) external view returns (address aTokenAddress, address stableDebtTokenAddress, address variableDebtTokenAddress)",
        "function getATokenTotalSupply(address asset) external view returns (uint256)",
        "function getTotalDebt(address asset) external view returns (uint256)",
    ]
)


# ---------------------------------------------------------------------------
# Aave V3 — Price Oracle
# ---------------------------------------------------------------------------

AAVE_V3_ORACLE = Contract.from_abi(
    [
        "function getAssetPrice(address asset) external view returns (uint256)",
        "function getAssetsPrices(address[] calldata assets) external view returns (uint256[] memory)",
        "function BASE_CURRENCY_UNIT() external view returns (uint256)",
    ]
)


# ---------------------------------------------------------------------------
# Aave V3 — Pool Addresses Provider
# ---------------------------------------------------------------------------

AAVE_V3_ADDRESSES_PROVIDER = Contract.from_abi(
    [
        "function getPool() external view returns (address)",
        "function getPoolDataProvider() external view returns (address)",
        "function getPriceOracle() external view returns (address)",
    ]
)


# ---------------------------------------------------------------------------
# Compound III (Comet) — ABI struct definitions
# ---------------------------------------------------------------------------


class CometAssetInfo(ABIStruct):
    """``AssetInfo`` struct returned by ``Comet.getAssetInfo``.

    Compound III stores collateral configuration per asset; the ``offset``
    field is the asset's index in the market's collateral list (0-based).
    Each factor is scaled by 1e18.
    """

    offset: Annotated[int, "uint8"]
    asset: Annotated[str, "address"]
    priceFeed: Annotated[str, "address"]
    scale: Annotated[int, "uint64"]
    borrowCollateralFactor: Annotated[int, "uint64"]
    liquidateCollateralFactor: Annotated[int, "uint64"]
    liquidationFactor: Annotated[int, "uint64"]
    supplyCap: Annotated[int, "uint128"]


# ---------------------------------------------------------------------------
# Compound III — Comet proxy
# ---------------------------------------------------------------------------

COMPOUND_V3_COMET = Contract.from_abi(
    CometAssetInfo.human_readable_abi()
    + [
        # Market metadata.
        "function baseToken() external view returns (address)",
        "function baseScale() external view returns (uint256)",
        "function baseTokenPriceFeed() external view returns (address)",
        "function numAssets() external view returns (uint8)",
        "function getAssetInfo(uint8 i) external view returns (CometAssetInfo)",
        "function getAssetInfoByAddress(address asset) external view returns (CometAssetInfo)",
        # Market state.
        "function totalSupply() external view returns (uint256)",
        "function totalBorrow() external view returns (uint256)",
        "function getUtilization() external view returns (uint256)",
        "function getSupplyRate(uint256 utilization) external view returns (uint64)",
        "function getBorrowRate(uint256 utilization) external view returns (uint64)",
        "function getPrice(address priceFeed) external view returns (uint256)",
        # User state.
        "function balanceOf(address account) external view returns (uint256)",
        "function borrowBalanceOf(address account) external view returns (uint256)",
        "function collateralBalanceOf(address account, address asset) external view returns (uint128)",
        "function isLiquidatable(address account) external view returns (bool)",
        # Writes.
        "function supply(address asset, uint256 amount) external",
        "function supplyTo(address dst, address asset, uint256 amount) external",
        "function withdraw(address asset, uint256 amount) external",
        "function withdrawTo(address to, address asset, uint256 amount) external",
        "function transferAsset(address dst, address asset, uint256 amount) external",
        "function transfer(address dst, uint256 amount) external returns (bool)",
        "function allow(address manager, bool isAllowed) external",
    ]
)


# ---------------------------------------------------------------------------
# Morpho Blue — ABI struct definitions
# ---------------------------------------------------------------------------


class MorphoMarketParams(ABIStruct):
    """Morpho Blue ``MarketParams`` — the immutable definition of one market.

    The keccak256 of ``abi.encode(MarketParams)`` is the market's ``Id``
    (a ``bytes32``). Field order is part of that hash, so it must match the
    on-chain struct exactly: loanToken, collateralToken, oracle, irm, lltv.
    """

    loanToken: Annotated[str, "address"]
    collateralToken: Annotated[str, "address"]
    oracle: Annotated[str, "address"]
    irm: Annotated[str, "address"]
    lltv: Annotated[int, "uint256"]


class MorphoMarketStruct(ABIStruct):
    """Morpho Blue ``Market`` — a market's live accounting state.

    Passed by value into ``IIrm.borrowRateView``. ``totalSupplyAssets`` /
    ``totalBorrowAssets`` are the interest-bearing totals; the matching
    ``*Shares`` fields track ERC-4626-style share supply.
    """

    totalSupplyAssets: Annotated[int, "uint128"]
    totalSupplyShares: Annotated[int, "uint128"]
    totalBorrowAssets: Annotated[int, "uint128"]
    totalBorrowShares: Annotated[int, "uint128"]
    lastUpdate: Annotated[int, "uint128"]
    fee: Annotated[int, "uint128"]


# ---------------------------------------------------------------------------
# Morpho Blue — core singleton
# ---------------------------------------------------------------------------

MORPHO_BLUE = Contract.from_abi(
    MorphoMarketParams.human_readable_abi()
    + [
        # Writes — every market-scoped call carries the full MarketParams.
        # supply / withdraw / borrow / repay take (assets, shares) with
        # exactly one non-zero.
        "function supply(MorphoMarketParams marketParams, uint256 assets, uint256 shares, address onBehalf, bytes data) external returns (uint256 assetsSupplied, uint256 sharesSupplied)",
        "function withdraw(MorphoMarketParams marketParams, uint256 assets, uint256 shares, address onBehalf, address receiver) external returns (uint256 assetsWithdrawn, uint256 sharesWithdrawn)",
        "function borrow(MorphoMarketParams marketParams, uint256 assets, uint256 shares, address onBehalf, address receiver) external returns (uint256 assetsBorrowed, uint256 sharesBorrowed)",
        "function repay(MorphoMarketParams marketParams, uint256 assets, uint256 shares, address onBehalf, bytes data) external returns (uint256 assetsRepaid, uint256 sharesRepaid)",
        "function supplyCollateral(MorphoMarketParams marketParams, uint256 assets, address onBehalf, bytes data) external",
        "function withdrawCollateral(MorphoMarketParams marketParams, uint256 assets, address onBehalf, address receiver) external",
        "function createMarket(MorphoMarketParams marketParams) external",
        "function accrueInterest(MorphoMarketParams marketParams) external",
        "function setAuthorization(address authorized, bool newIsAuthorized) external",
        "function flashLoan(address token, uint256 assets, bytes data) external",
        "function liquidate(MorphoMarketParams marketParams, address borrower, uint256 seizedAssets, uint256 repaidShares, bytes data) external returns (uint256, uint256)",
        # Reads. market / position / idToMarketParams are public-mapping
        # getters: Solidity flattens the struct value into a plain tuple of
        # its members, so the ABI return is NOT a nested struct.
        "function market(bytes32 id) external view returns (uint128 totalSupplyAssets, uint128 totalSupplyShares, uint128 totalBorrowAssets, uint128 totalBorrowShares, uint128 lastUpdate, uint128 fee)",
        "function position(bytes32 id, address user) external view returns (uint256 supplyShares, uint128 borrowShares, uint128 collateral)",
        "function idToMarketParams(bytes32 id) external view returns (address loanToken, address collateralToken, address oracle, address irm, uint256 lltv)",
        "function isAuthorized(address authorizer, address authorized) external view returns (bool)",
        "function nonce(address authorizer) external view returns (uint256)",
        "function isIrmEnabled(address irm) external view returns (bool)",
        "function isLltvEnabled(uint256 lltv) external view returns (bool)",
        "function owner() external view returns (address)",
        "function feeRecipient() external view returns (address)",
        "function DOMAIN_SEPARATOR() external view returns (bytes32)",
    ]
)


# ---------------------------------------------------------------------------
# Morpho Blue — Interest Rate Model (AdaptiveCurveIRM)
# ---------------------------------------------------------------------------

MORPHO_IRM = Contract.from_abi(
    MorphoMarketParams.human_readable_abi()
    + MorphoMarketStruct.human_readable_abi()
    + [
        # Returns the borrow rate per second, scaled by WAD (1e18).
        "function borrowRateView(MorphoMarketParams marketParams, MorphoMarketStruct market) external view returns (uint256)",
    ]
)


# ---------------------------------------------------------------------------
# Morpho Blue — market oracle (IOracle)
# ---------------------------------------------------------------------------

MORPHO_ORACLE = Contract.from_abi(
    [
        # Price of 1 collateral unit quoted in loan token, scaled by 1e36.
        "function price() external view returns (uint256)",
    ]
)

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

    Note: rates are in RAY (1e27) per-second compounded.
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

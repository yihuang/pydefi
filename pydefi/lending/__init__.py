"""
Lending protocol integrations.

There is no shared abstract base — Aave V3 and Compound III have
incompatible models (per-reserve markets with aTokens vs single-base-asset
Comet proxies), so each protocol lives as its own self-contained module:

* :mod:`pydefi.lending.aave_v3` — Aave V3 client, the Aave-specific
  ``ReserveData`` / ``UserAccountData`` / ``EModeCategory`` types, and
  the RAY rate-model helpers (``ray_rate_to_apy``).
* :mod:`pydefi.lending.compound_v3` — Compound III (Comet) client plus
  its own ``CompoundMarketData`` / ``CompoundUserPosition`` types and
  per-second rate-model helper.

Quick-start::

    from pydefi.lending import AaveV3
    from pydefi.deployments import get_address, get_token
    from pydefi.types import Address, ChainId, TokenAmount
    from pydefi._utils import decode_address

    aave = AaveV3(
        w3=w3,
        chain_id=ChainId.ETHEREUM,
        pool_address=Address(get_address("AAVE_V3_POOL", ChainId.ETHEREUM)),
        data_provider_address=Address(get_address("AAVE_V3_DATA_PROVIDER", ChainId.ETHEREUM)),
        oracle_address=Address(get_address("AAVE_V3_ORACLE", ChainId.ETHEREUM)),
    )

    usdc = get_token("USDC", ChainId.ETHEREUM)
    data = await aave.get_reserve_data(usdc)
    print(data.supply_apy, data.variable_borrow_apy, data.utilization)

    tx = aave.build_supply_tx(
        user=decode_address("0x...", ChainId.ETHEREUM),
        amount=TokenAmount.from_human(usdc, "1000"),
    )

For chains that may rotate the ``PoolDataProvider``, prefer the live
constructor that resolves every dependency from the
``PoolAddressesProvider``::

    aave = await AaveV3.from_addresses_provider(
        w3=w3,
        chain_id=ChainId.ETHEREUM,
        addresses_provider=Address(get_address("AAVE_V3_ADDRESSES_PROVIDER", ChainId.ETHEREUM)),
    )
"""

from pydefi.lending.aave_v3 import (
    RAY,
    SECONDS_PER_YEAR,
    AaveV3,
    EModeCategory,
    InterestRateMode,
    ReserveData,
    UserAccountData,
    UserReserveData,
    ray_rate_to_apy,
)
from pydefi.lending.compound_v3 import (
    CompoundCollateralInfo,
    CompoundMarketData,
    CompoundUserPosition,
    CompoundV3,
)

__all__ = [
    "AaveV3",
    "CompoundCollateralInfo",
    "CompoundMarketData",
    "CompoundUserPosition",
    "CompoundV3",
    "EModeCategory",
    "InterestRateMode",
    "ReserveData",
    "UserAccountData",
    "UserReserveData",
    "RAY",
    "SECONDS_PER_YEAR",
    "ray_rate_to_apy",
]

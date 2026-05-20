"""
Lending protocol integrations.

There is no shared abstract base — Aave V3 and Compound III have
incompatible models (per-reserve markets with aTokens vs single-base-asset
Comet proxies), so each protocol lives as its own self-contained module:

* :mod:`pydefi.lending.aave_v3` — Aave V3 client, the Aave-specific
  ``ReserveData`` / ``UserAccountData`` / ``EModeCategory`` types, and
  the RAY rate-model helper (``ray_rate_to_apy``).
* :mod:`pydefi.lending.compound_v3` — Compound III (Comet) client plus
  its own ``CompoundMarketData`` / ``CompoundUserPosition`` types and
  per-second rate-model helper.
* :mod:`pydefi.lending.utils` — shared helpers (``UINT256_MAX``,
  ``SECONDS_PER_YEAR``, ``resolve_amount``, ``to_tx``).

Quick-start::

    from pydefi.lending import AaveV3
    from pydefi.deployments import get_token
    from pydefi.types import ChainId, TokenAmount
    from pydefi._utils import decode_address

    aave = await AaveV3.from_chain(w3, ChainId.ETHEREUM)

    usdc = get_token("USDC", ChainId.ETHEREUM)
    data = await aave.get_reserve_data(usdc)
    print(data.supply_apy, data.variable_borrow_apy, data.utilization)

    tx = aave.build_supply_tx(
        user=decode_address("0x...", ChainId.ETHEREUM),
        amount=TokenAmount.from_human(usdc, "1000"),
    )

Three ways to construct a client:

* ``await AaveV3.from_chain(w3, chain_id)`` — the default. Resolves Pool /
  DataProvider / Oracle on-chain from the registry's ``PoolAddressesProvider``.
* ``await AaveV3.from_addresses_provider(w3, chain_id, addresses_provider)``
  — same, from a provider you supply; for forks or unregistered chains.
* ``AaveV3(w3, chain_id, pool_address, data_provider_address, ...)`` —
  explicit addresses, no network round-trip; for tests or pinned forks.

``CompoundV3`` mirrors the first and third: ``CompoundV3.from_chain(w3,
chain_id, "USDC")`` (Comet markets are keyed by base asset) or the
explicit ``CompoundV3(w3, chain_id, comet_address)``.
"""

from pydefi.lending.aave_v3 import (
    RAY,
    AaveV3,
    EModeCategory,
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
from pydefi.lending.utils import SECONDS_PER_YEAR

__all__ = [
    "AaveV3",
    "CompoundCollateralInfo",
    "CompoundMarketData",
    "CompoundUserPosition",
    "CompoundV3",
    "EModeCategory",
    "ReserveData",
    "UserAccountData",
    "UserReserveData",
    "RAY",
    "SECONDS_PER_YEAR",
    "ray_rate_to_apy",
]

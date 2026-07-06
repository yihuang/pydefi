"""
Lending protocol integrations.

There is no shared abstract base — Aave V3, Compound III, Morpho Blue and
Aave V4 have incompatible models (per-reserve pool with aTokens vs
single-base-asset Comet proxies vs isolated markets keyed by
``MarketParams`` vs Hub-and-Spoke), so each protocol lives as its own
self-contained module:

* :mod:`pydefi.lending.aave_v3` — Aave V3 client, the Aave-specific
  ``ReserveData`` / ``UserAccountData`` / ``EModeCategory`` types, and
  the RAY rate-model helper (``ray_rate_to_apy``).
* :mod:`pydefi.lending.compound_v3` — Compound III (Comet) client plus
  its own ``CompoundMarketData`` / ``CompoundUserPosition`` types.
* :mod:`pydefi.lending.morpho` — Morpho Blue client, the ``MarketParams``
  / ``MarketState`` / ``MorphoMarket`` / ``MorphoPosition`` types, and the
  pure shares / interest / health-factor math.
* :mod:`pydefi.lending.aave_v4` — Aave V4 client (Hub-and-Spoke), the
  ``V4Reserve`` / ``V4ReserveData`` / ``V4UserReserve`` /
  ``V4UserAccountData`` types.
* :mod:`pydefi.lending.liquidation` — ``LiquidationRouter``: checks
  caller-supplied positions for liquidatability across all four protocols
  and yields ``LiquidationOpportunity`` objects carrying the prebuilt
  liquidation transaction.
* :mod:`pydefi.lending.utils` — shared helpers (``UINT256_MAX``,
  ``SECONDS_PER_YEAR``, ``resolve_amount``, ``to_tx``, and the per-second
  ``per_second_rate_to_apy`` used by both Compound III and Morpho).

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
explicit ``CompoundV3(w3, chain_id, comet_address)``. ``MorphoBlue`` is
constructed with ``MorphoBlue.from_chain(w3, chain_id)`` (the Morpho core
is one immutable singleton per chain) and every call is scoped to a
``MarketParams``.
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
from pydefi.lending.aave_v4 import (
    AaveV4,
    AaveV4TokenizationSpoke,
    V4Reserve,
    V4ReserveData,
    V4TokenizedPosition,
    V4UserAccountData,
    V4UserReserve,
)
from pydefi.lending.compound_v3 import (
    CompoundCollateralInfo,
    CompoundMarketData,
    CompoundUserPosition,
    CompoundV3,
)
from pydefi.lending.liquidation import (
    AaveV3Candidate,
    AaveV4Candidate,
    CompoundV3Candidate,
    LiquidationCandidate,
    LiquidationOpportunity,
    LiquidationRouter,
    MorphoCandidate,
)
from pydefi.lending.morpho import (
    MarketParams,
    MarketState,
    MorphoBlue,
    MorphoMarket,
    MorphoPosition,
    accrue_interest,
    compute_health_factor,
    liquidation_incentive_factor,
    max_liquidation,
    supply_apy_from_borrow,
)
from pydefi.lending.utils import SECONDS_PER_YEAR, per_second_rate_to_apy

__all__ = [
    "AaveV3",
    "AaveV3Candidate",
    "AaveV4",
    "AaveV4Candidate",
    "AaveV4TokenizationSpoke",
    "CompoundCollateralInfo",
    "CompoundMarketData",
    "CompoundUserPosition",
    "CompoundV3",
    "CompoundV3Candidate",
    "EModeCategory",
    "LiquidationCandidate",
    "LiquidationOpportunity",
    "LiquidationRouter",
    "MarketParams",
    "MarketState",
    "MorphoBlue",
    "MorphoCandidate",
    "MorphoMarket",
    "MorphoPosition",
    "ReserveData",
    "UserAccountData",
    "UserReserveData",
    "V4Reserve",
    "V4ReserveData",
    "V4TokenizedPosition",
    "V4UserAccountData",
    "V4UserReserve",
    "RAY",
    "SECONDS_PER_YEAR",
    "accrue_interest",
    "compute_health_factor",
    "liquidation_incentive_factor",
    "max_liquidation",
    "per_second_rate_to_apy",
    "ray_rate_to_apy",
    "supply_apy_from_borrow",
]

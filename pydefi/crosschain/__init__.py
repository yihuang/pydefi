"""Cross-Chain Router Builder — compose ``bridge + swap + deposit/yield`` into
broadcast-ready routes.

Two execution models:

* **atomic-compose** — the destination ``swap → supply`` is compiled into a
  DeFiVM program carried as the bridge ``hookData``; a relayer triggers the
  composer on arrival. Wired for CCTP and CCIP.
* **deferred** — broadcast the source leg, wait for settlement, then the
  destination leg (any bridge; see :mod:`pydefi.yields.router`).
"""

from pydefi.crosschain.compose import (
    AMOUNT_RECEIVED_SLOT,
    SupplyTemplate,
    build_compose_deposit_program,
    build_source_swap_bridge_program,
    build_supply_template,
)
from pydefi.crosschain.route import (
    ComposeBridge,
    CrossChainLeg,
    CrossChainRoute,
    build_cctp_swap_deposit_route,
    build_compose_route,
    build_deposit_route,
    estimate_cctp_delivered,
)

__all__ = [
    "AMOUNT_RECEIVED_SLOT",
    "build_compose_deposit_program",
    "build_source_swap_bridge_program",
    "SupplyTemplate",
    "build_supply_template",
    "ComposeBridge",
    "CrossChainLeg",
    "CrossChainRoute",
    "build_compose_route",
    "build_deposit_route",
    "build_cctp_swap_deposit_route",
    "estimate_cctp_delivered",
]

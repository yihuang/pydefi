"""Cross-chain bridge integrations."""

from typing import Any, Protocol

from pydefi.bridge.across import Across
from pydefi.bridge.ccip import CCIP, EVM_EXTRA_ARGS_V2_TAG
from pydefi.bridge.cctp import (
    CCTP,
    FINALITY_THRESHOLD_CONFIRMED,
    FINALITY_THRESHOLD_FINALIZED,
    HYPERCORE_DEX_PERP,
    HYPERCORE_DEX_SPOT,
    encode_cctp_forward_hook_data,
)
from pydefi.bridge.eureka import (
    ICS20_DEFAULT_PORT,
    Eureka,
    encode_send_transfer_calldata,
)
from pydefi.bridge.gaszip import GasZip
from pydefi.bridge.layerzero_oft import LayerZeroOFT
from pydefi.bridge.lucid import LucidBridge
from pydefi.bridge.mayan import Mayan
from pydefi.bridge.relay import Relay
from pydefi.bridge.router import BridgeRouter, RankedBridgeQuote, rank_bridge_quotes
from pydefi.bridge.stargate import Stargate
from pydefi.types import Address, BridgeQuote, Token, TokenAmount


class Bridge(Protocol):
    """Structural interface shared by every bridge integration (duck-typed,
    never inherited): one instance per (src, dst) lane, async quoting, and tx
    builders that return broadcast-ready dicts (``to``, ``data``, ``value``,
    ``gas``).

    ``spender`` is the contract the source ERC-20 must be approved to before
    :meth:`build_bridge_tx` can pull it — ``None`` for bridges that move only
    the native asset or expose no single approval target; such a bridge cannot
    carry an ERC-20 route through :func:`~pydefi.yields.build_yield_route`.

    Bridges with a compose path (CCTP, CCIP) also define
    ``build_bridge_compose_tx(amount_in, composer_address, program, **kwargs)``
    — the one-signature cross-chain entrypoint
    :func:`~pydefi.yields.build_compose_supply_route` builds on.
    """

    src_chain_id: int
    dst_chain_id: int
    protocol_name: str

    @property
    def spender(self) -> Address | None: ...

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote: ...

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


__all__ = [
    "Bridge",
    "BridgeRouter",
    "RankedBridgeQuote",
    "Stargate",
    "Across",
    "Mayan",
    "GasZip",
    "Relay",
    "LayerZeroOFT",
    "LucidBridge",
    "CCTP",
    "CCIP",
    "Eureka",
    "EVM_EXTRA_ARGS_V2_TAG",
    "FINALITY_THRESHOLD_CONFIRMED",
    "FINALITY_THRESHOLD_FINALIZED",
    "HYPERCORE_DEX_PERP",
    "HYPERCORE_DEX_SPOT",
    "ICS20_DEFAULT_PORT",
    "encode_cctp_forward_hook_data",
    "encode_send_transfer_calldata",
    "rank_bridge_quotes",
]

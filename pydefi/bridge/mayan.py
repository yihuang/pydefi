"""
Mayan Finance cross-chain bridge integration.

Mayan is a cross-chain swap protocol built on Wormhole that enables fast
bridging between EVM chains and Solana.  This module wraps the Mayan
Price API and the Mayan Forwarder / MayanSwift contracts for on-chain
execution.

Docs: https://docs.mayan.finance/
"""

from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

from eth_contract import Contract

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

_MAYAN_API_BASE = "https://price-api.mayan.finance/v3"

# Mayan chain name slugs (used in the Price API)
_CHAIN_NAMES: dict[int, str] = {
    1: "ethereum",
    56: "bsc",
    137: "polygon",
    42161: "arbitrum",
    10: "optimism",
    8453: "base",
    43114: "avalanche",
    59144: "linea",
    534352: "scroll",
    81457: "blast",
    324: "zksync",
    7777777: "zora",
}

# Wormhole chain IDs for EVM chains (source: Mayan Finance SDK)
_WORMHOLE_CHAIN_IDS: dict[int, int] = {
    1: 2,       # Ethereum
    56: 4,      # BSC
    137: 5,     # Polygon
    43114: 6,   # Avalanche
    42161: 23,  # Arbitrum
    10: 24,     # Optimism
    8453: 30,   # Base
    130: 44,    # Unichain
    59144: 38,  # Linea
}

# Mayan Forwarder contract (routes ETH/ERC-20 into the appropriate Mayan
# bridge contract).  Address is the same on all supported EVM chains.
_MAYAN_FORWARDER = "0x337685fdaB40D39bd02028545a4FfA7D287cC3E2"

# SWIFT normalize factor: SWIFT amounts are capped at 8 decimals.
_SWIFT_NORMALIZE_DECIMALS = 8

# Native ETH sentinel addresses (both the burn address and EeeE... form)
_NATIVE_SENTINELS = frozenset({
    "0x0000000000000000000000000000000000000000",
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
})

# ABI fragment for MayanForwarder.forwardEth
_FORWARDER_ABI = [
    "function forwardEth(address mayanProtocol, bytes protocolData) external payable",
]

# ABI fragment for MayanSwift (V1) createOrderWithEth
_SWIFT_ABI = [
    "function createOrderWithEth("
    "(bytes32 trader, bytes32 tokenOut, uint64 minAmountOut, uint64 gasDrop,"
    " uint64 cancelFee, uint64 refundFee, uint64 deadline, bytes32 destAddr,"
    " uint16 destChainId, bytes32 referrerAddr, uint8 referrerBps,"
    " uint8 auctionMode, bytes32 random) params"
    ") external payable returns (bytes32 orderHash)",
]


def _addr_to_bytes32(addr: str) -> bytes:
    """Left-pad an EVM address into a 32-byte Wormhole representation."""
    hex_addr = addr[2:] if addr.startswith("0x") else addr
    # EVM addresses are 40 hex chars; zero-pad to 64 chars (32 bytes)
    return bytes.fromhex(hex_addr.lower().zfill(64))


def _token_to_bytes32(token_address: str) -> bytes:
    """Convert a token address to its SWIFT bytes32 tokenOut representation.

    Native ETH (both zero-address and EeeE... sentinel) maps to 32 zero bytes
    (the Solana system program ID in the Wormhole encoding used by SWIFT).
    ERC-20 tokens are left-padded as a normal Wormhole EVM address.
    """
    if token_address.lower() in _NATIVE_SENTINELS:
        return bytes(32)
    return _addr_to_bytes32(token_address)


class Mayan(BaseBridge):
    """Mayan Finance cross-chain bridge integration.

    Args:
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        api_base_url: Override the Mayan Price API base URL.
    """

    def __init__(
        self,
        src_chain_id: int,
        dst_chain_id: int,
        api_base_url: str = _MAYAN_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self._api_base = api_base_url.rstrip("/")

    @property
    def protocol_name(self) -> str:
        return "Mayan"

    def _chain_name(self, chain_id: int) -> str:
        """Return the Mayan chain name slug for *chain_id*."""
        name = _CHAIN_NAMES.get(chain_id)
        if name is None:
            raise BridgeError(f"Mayan: unsupported chain ID {chain_id}")
        return name

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a Mayan bridge quote.

        Args:
            token_in: Source chain token.
            token_out: Destination chain token.
            amount_in: Amount to bridge.
            **kwargs: Additional query parameters forwarded to the API
                (e.g. ``slippageBps``, ``swift``).

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        from_chain = self._chain_name(self.src_chain_id)
        to_chain = self._chain_name(self.dst_chain_id)
        human_amount = str(amount_in.human_amount)

        params: dict[str, Any] = {
            "amountIn": human_amount,
            "fromToken": token_in.address,
            "toToken": token_out.address,
            "fromChain": from_chain,
            "toChain": to_chain,
            "slippageBps": "auto",
            **kwargs,
        }

        url = f"{self._api_base}/quote"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(
                        f"Mayan API error ({resp.status}): {data}"
                    )

        # The API returns a list of routes; pick the best (first) one
        routes = data if isinstance(data, list) else data.get("routes", [data])
        if not routes:
            raise BridgeError("Mayan: no routes returned from API")

        best = routes[0]
        expected_amount_out = best.get("expectedAmountOut", 0)

        # Convert human-readable output to raw units
        amount_out_raw = int(float(expected_amount_out) * (10 ** token_out.decimals))

        # Compute fee as (amount_in - effectiveAmountIn) expressed in token_in units
        effective_amount_in_str = best.get("effectiveAmountIn")
        if effective_amount_in_str is not None:
            effective_amount_in_raw = int(
                float(effective_amount_in_str) * (10 ** token_in.decimals)
            )
            fee_raw = max(0, amount_in.amount - effective_amount_in_raw)
        else:
            fee_raw = 0

        # Estimate time based on route type
        swift_routes = {"SWIFT", "swift"}
        route_type = best.get("type", "")
        estimated_time = 10 if route_type in swift_routes else 60

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=fee_raw),
            estimated_time_seconds=estimated_time,
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        slippage_bps: int = 50,
        referrer: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Mayan SWIFT bridge transaction via the Mayan Forwarder contract.

        This method:

        1. Fetches a SWIFT quote from the Mayan Price API.
        2. Encodes a ``createOrderWithEth`` call for the MayanSwift contract.
        3. Wraps it in a ``forwardEth`` call on the Mayan Forwarder contract.

        Only native-ETH input is currently supported.

        Args:
            token_in: Source token (must be native ETH).
            token_out: Destination token.
            amount_in: Amount to send.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            referrer: Optional referrer address for fee sharing.
            **kwargs: Extra query parameters forwarded to the quote API.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error or
                unsupported route type.
        """
        from_chain = self._chain_name(self.src_chain_id)
        to_chain = self._chain_name(self.dst_chain_id)
        human_amount = str(amount_in.human_amount)

        # Step 1 — fetch a SWIFT quote
        params: dict[str, Any] = {
            "amountIn": human_amount,
            "fromToken": token_in.address,
            "toToken": token_out.address,
            "fromChain": from_chain,
            "toChain": to_chain,
            "slippageBps": slippage_bps,
            "swift": True,
            **kwargs,
        }

        url = f"{self._api_base}/quote"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(
                        f"Mayan API error ({resp.status}): {data}"
                    )

        routes = data if isinstance(data, list) else data.get("routes", [data])
        if not routes:
            raise BridgeError("Mayan: no routes returned from API")

        # Find first SWIFT route
        swift_route = next(
            (r for r in routes if str(r.get("type", "")).upper() == "SWIFT"),
            None,
        )
        if swift_route is None:
            raise BridgeError("Mayan: no SWIFT route available for build_bridge_tx")

        swift_contract = swift_route.get("swiftMayanContract")
        if not swift_contract:
            raise BridgeError("Mayan: swiftMayanContract missing in quote response")

        # Resolve Wormhole chain ID for the destination chain
        dest_wh_chain = _WORMHOLE_CHAIN_IDS.get(self.dst_chain_id)
        if dest_wh_chain is None:
            raise BridgeError(
                f"Mayan: no Wormhole chain ID mapping for EVM chain {self.dst_chain_id}"
            )

        # Step 2 — decode SWIFT order parameters from the quote
        # SWIFT amounts are scaled to at most 8 decimal places.
        swift_decimals = min(token_out.decimals, _SWIFT_NORMALIZE_DECIMALS)
        min_amount_out = int(float(swift_route.get("minAmountOut", 0)) * 10**swift_decimals)
        # gasDrop on EVM chains uses the same 8-decimal SWIFT normalization
        gas_drop = int(float(swift_route.get("gasDrop", 0)) * 10**_SWIFT_NORMALIZE_DECIMALS)
        cancel_fee = int(swift_route.get("cancelRelayerFee64") or "0")
        refund_fee = int(swift_route.get("refundRelayerFee64") or "0")
        deadline = int(swift_route.get("deadline64") or "0")
        auction_mode = int(swift_route.get("swiftAuctionMode") or 0)

        # Use os.urandom for the order's random field to ensure uniqueness
        random_b32 = os.urandom(32)

        trader_b32 = _addr_to_bytes32(recipient)
        token_out_b32 = _token_to_bytes32(token_out.address)
        dest_addr_b32 = _addr_to_bytes32(recipient)
        referrer_b32 = _addr_to_bytes32(referrer) if referrer else bytes(32)

        order_tuple = (
            trader_b32,      # bytes32 trader
            token_out_b32,   # bytes32 tokenOut
            min_amount_out,  # uint64 minAmountOut
            gas_drop,        # uint64 gasDrop
            cancel_fee,      # uint64 cancelFee
            refund_fee,      # uint64 refundFee
            deadline,        # uint64 deadline
            dest_addr_b32,   # bytes32 destAddr
            dest_wh_chain,   # uint16 destChainId
            referrer_b32,    # bytes32 referrerAddr
            0,               # uint8 referrerBps
            auction_mode,    # uint8 auctionMode
            random_b32,      # bytes32 random
        )

        # Step 3 — ABI-encode the SWIFT and Forwarder calls
        swift_c = Contract.from_abi(_SWIFT_ABI, to=swift_contract)
        swift_calldata: bytes = swift_c.fns.createOrderWithEth(order_tuple).data

        forwarder = Contract.from_abi(_FORWARDER_ABI, to=_MAYAN_FORWARDER)
        forward_calldata: bytes = forwarder.fns.forwardEth(
            swift_contract, swift_calldata
        ).data

        return {
            "to": _MAYAN_FORWARDER,
            "data": "0x" + forward_calldata.hex(),
            "value": str(amount_in.amount),
            "gas": str(500_000),
        }

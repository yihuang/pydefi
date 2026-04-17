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
from decimal import Decimal
from typing import Any, Optional

import aiohttp
from hexbytes import HexBytes

from pydefi._utils import address_to_bytes32, encode_address, token_to_bytes32
from pydefi.abi.bridge import MAYAN_FORWARDER, MAYAN_SWIFT_V2, MayanSwiftOrderParams
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import NATIVE_SENTINEL, Address, BridgeQuote, Token, TokenAmount

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
    1: 2,  # Ethereum
    56: 4,  # BSC
    137: 5,  # Polygon
    43114: 6,  # Avalanche
    42161: 23,  # Arbitrum
    10: 24,  # Optimism
    8453: 30,  # Base
    130: 44,  # Unichain
    59144: 38,  # Linea
}

# Mayan Forwarder contract (routes ETH/ERC-20 into the appropriate Mayan
# bridge contract).  Address is the same on all supported EVM chains.
_MAYAN_FORWARDER = "0x337685fdaB40D39bd02028545a4FfA7D287cC3E2"

# Mayan Solana program ID (required by the Price API)
_MAYAN_SOLANA_PROGRAM = "FC4eXxkyrMPTjiYUpp4EAnkmwMbQyZ6NDCh1kfLn6vsf"

# Mayan SDK version string (required by the Price API for compatibility checks)
_MAYAN_SDK_VERSION = "13_2_0"

# SWIFT normalize factor: SWIFT amounts are capped at 8 decimals.
_SWIFT_NORMALIZE_DECIMALS = 8

# WETH ERC-20 addresses per chain (needed for native-ETH input with SWIFT V2)
_CHAIN_WETH: dict[int, Address] = {
    1: HexBytes("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),  # Ethereum
    42161: HexBytes("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"),  # Arbitrum
    10: HexBytes("0x4200000000000000000000000000000000000006"),  # Optimism
    8453: HexBytes("0x4200000000000000000000000000000000000006"),  # Base
    137: HexBytes("0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"),  # Polygon
    56: HexBytes("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),  # BSC (WBNB)
    43114: HexBytes("0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB"),  # Avalanche
    59144: HexBytes("0xe5D7C2a44FfDDf6b295A15c148167daaAf5Cf34f"),  # Linea
}

_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def _mayan_token_address(address: Address, chain_id: int) -> str:
    """Normalize a token address for the Mayan Price API.

    The Mayan API uses the zero address to represent the native gas token
    (ETH, BNB, etc.).  Any ``EeeE...`` sentinel is canonicalized here so
    that API requests always use the representation the server expects.
    """
    if address == NATIVE_SENTINEL:
        return _ZERO_ADDRESS
    return encode_address(address, chain_id)


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
            "fromToken": _mayan_token_address(token_in.address, token_in.chain_id),
            "toToken": _mayan_token_address(token_out.address, token_out.chain_id),
            "fromChain": from_chain,
            "toChain": to_chain,
            "slippageBps": "auto",
            # Bridge type flags — the API requires these to find routes
            "wormhole": "true",
            "swift": "true",
            "mctp": "true",
            "shuttle": "false",
            "fastMctp": "true",
            "gasless": "false",
            "onlyDirect": "false",
            "fullList": "false",
            "monoChain": "true",
            # Required by the Mayan v3 API
            "solanaProgram": _MAYAN_SOLANA_PROGRAM,
            "forwarderAddress": _MAYAN_FORWARDER,
            # SDK version required by the API to validate compatibility
            "sdkVersion": _MAYAN_SDK_VERSION,
            **kwargs,
        }

        url = f"{self._api_base}/quote"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    try:
                        err = await resp.json(content_type=None)
                    except Exception:
                        err = await resp.text()
                    raise BridgeError(f"Mayan API error ({resp.status}): {err}")
                data = await resp.json(content_type=None)

        # The Mayan v3 API wraps routes in a "quotes" key
        routes = data.get("quotes") if isinstance(data, dict) else data
        if not routes:
            raise BridgeError("Mayan: no routes returned from API")

        best = routes[0]
        expected_amount_out = best.get("expectedAmountOut", 0)

        # Use Decimal for exact base-10 scaling (no floating-point drift)
        amount_out_raw = int(Decimal(str(expected_amount_out)) * Decimal(10**token_out.decimals))

        # Compute fee as (amount_in - effectiveAmountIn) expressed in token_in units
        effective_amount_in_str = best.get("effectiveAmountIn")
        if effective_amount_in_str is not None:
            effective_amount_in_raw = int(Decimal(str(effective_amount_in_str)) * Decimal(10**token_in.decimals))
            fee_raw = max(0, amount_in.amount - effective_amount_in_raw)
        else:
            fee_raw = 0

        # Estimate time based on route type
        route_type = best.get("type", "")
        estimated_time = 10 if route_type.upper() == "SWIFT" else 60

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
        recipient: Address,
        slippage_bps: int = 50,
        referrer: Optional[Address] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Mayan SWIFT bridge transaction via the Mayan Forwarder contract.

        For native-ETH input this method uses the SWIFT V2 ``swapAndForwardEth``
        flow:

        1. Resolve the WETH address for the source chain.
        2. Fetch a SWIFT V2 quote using WETH as the source token.
        3. Fetch ETH→WETH swap calldata from the Mayan ``get-swap/evm`` API.
        4. Encode ``createOrderWithToken`` for the MayanSwift V2 contract.
        5. Wrap everything in ``swapAndForwardEth`` on the Mayan Forwarder.

        Only native-ETH input is currently supported (no ERC-20 approval needed).

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
        if not token_in.is_native():
            raise BridgeError("Mayan build_bridge_tx currently only supports native ETH input")

        from_chain = self._chain_name(self.src_chain_id)
        to_chain = self._chain_name(self.dst_chain_id)

        # Keep a WETH address as fallback for swiftInputContract when the
        # API does not return one (should not normally happen).
        weth_address = _CHAIN_WETH.get(self.src_chain_id)
        if weth_address is None:
            raise BridgeError(f"Mayan: no WETH address known for chain {self.src_chain_id}")

        # Step 1 — fetch a SWIFT V2 quote using the zero address (native ETH)
        # as the source token.  The Mayan API uses the zero address to represent
        # native gas tokens; the EeeE... sentinel is not recognised.
        params: dict[str, Any] = {
            "amountIn64": str(amount_in.amount),
            "fromToken": _mayan_token_address(token_in.address, token_in.chain_id),
            "toToken": _mayan_token_address(token_out.address, token_out.chain_id),
            "fromChain": from_chain,
            "toChain": to_chain,
            "slippageBps": slippage_bps,
            "wormhole": "false",
            "swift": "true",
            "mctp": "false",
            "shuttle": "false",
            "fastMctp": "false",
            "gasless": "false",
            "onlyDirect": "false",
            "fullList": "false",
            "monoChain": "false",
            "solanaProgram": _MAYAN_SOLANA_PROGRAM,
            "forwarderAddress": _MAYAN_FORWARDER,
            "sdkVersion": _MAYAN_SDK_VERSION,
            **kwargs,
        }

        url = f"{self._api_base}/quote"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    try:
                        err = await resp.json(content_type=None)
                    except Exception:
                        err = await resp.text()
                    raise BridgeError(f"Mayan API error ({resp.status}): {err}")
                data = await resp.json(content_type=None)

        routes = data.get("quotes") if isinstance(data, dict) else data
        if not routes:
            raise BridgeError("Mayan: no routes returned from API")

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
            raise BridgeError(f"Mayan: no Wormhole chain ID mapping for EVM chain {self.dst_chain_id}")

        # Step 2 — decode SWIFT V2 order parameters from the quote.
        # SWIFT amounts are scaled to at most 8 decimal places (SWIFT normalize factor).
        # Use Decimal for exact base-10 arithmetic (no floating-point drift).
        swift_decimals = min(token_out.decimals, _SWIFT_NORMALIZE_DECIMALS)
        min_amount_out = int(Decimal(str(swift_route.get("minAmountOut") or "0")) * Decimal(10**swift_decimals))
        gas_drop = int(Decimal(str(swift_route.get("gasDrop") or "0")) * Decimal(10**_SWIFT_NORMALIZE_DECIMALS))
        cancel_fee = int(swift_route.get("cancelRelayerFee64") or "0")
        refund_fee = int(swift_route.get("refundRelayerFee64") or "0")
        deadline = int(swift_route.get("deadline64") or "0")
        auction_mode = int(swift_route.get("swiftAuctionMode") or 0)
        # effectiveAmountIn64 is the token amount the SWIFT contract expects
        effective_amount_in = int(swift_route.get("effectiveAmountIn64") or amount_in.amount)

        # minMiddleAmount: minimum WETH the DEX swap must produce (in WETH decimals)
        swift_input_decimals = int(swift_route.get("swiftInputDecimals") or 18)
        min_middle_amount = int(
            Decimal(str(swift_route.get("minMiddleAmount") or "0")) * Decimal(10**swift_input_decimals)
        )
        # swiftInputContract is the ERC-20 that SWIFT V2 will receive (typically WETH).
        _raw_swift_input = swift_route.get("swiftInputContract")
        swift_input_contract: Address = HexBytes(_raw_swift_input) if _raw_swift_input else weth_address

        # Use os.urandom for the order's random field to ensure uniqueness
        random_b32 = os.urandom(32)

        trader_b32 = address_to_bytes32(recipient)
        token_out_b32 = token_to_bytes32(token_out.address)
        dest_addr_b32 = address_to_bytes32(recipient)
        referrer_b32 = address_to_bytes32(referrer) if referrer else bytes(32)

        # SWIFT V2 OrderParams (field order differs from V1)
        order_params = MayanSwiftOrderParams(
            payloadType=0,  # 0 = default, no custom payload
            trader=trader_b32,
            destAddr=dest_addr_b32,
            destChainId=dest_wh_chain,
            referrerAddr=referrer_b32,
            tokenOut=token_out_b32,
            minAmountOut=min_amount_out,
            gasDrop=gas_drop,
            cancelFee=cancel_fee,
            refundFee=refund_fee,
            deadline=deadline,
            referrerBps=0,
            auctionMode=auction_mode,
            random=random_b32,
        )

        # Step 3 — fetch ETH→WETH swap calldata from the Mayan get-swap/evm API.
        # The Mayan API uses the zero address for native ETH; normalise here.
        swap_params = {
            "forwarderAddress": _MAYAN_FORWARDER,
            "slippageBps": slippage_bps,
            "fromToken": _mayan_token_address(token_in.address, token_in.chain_id),
            "middleToken": encode_address(swift_input_contract, self.src_chain_id),  # WETH (or other swift input)
            "chainName": from_chain,
            "amountIn64": str(amount_in.amount),
            "sdkVersion": _MAYAN_SDK_VERSION,
        }
        swap_url = f"{self._api_base}/get-swap/evm"
        async with aiohttp.ClientSession() as session:
            async with session.get(swap_url, params=swap_params) as resp:
                if resp.status != 200:
                    try:
                        err = await resp.json(content_type=None)
                    except Exception:
                        err = await resp.text()
                    raise BridgeError(f"Mayan get-swap/evm API error ({resp.status}): {err}")
                swap_data = await resp.json(content_type=None)

        swap_router_address = swap_data.get("swapRouterAddress")
        swap_router_calldata = swap_data.get("swapRouterCalldata")
        if not swap_router_address or not swap_router_calldata:
            raise BridgeError("Mayan: missing swapRouterAddress or swapRouterCalldata")

        # Step 4 — ABI-encode createOrderWithToken for MayanSwift V2
        swift_calldata: bytes = MAYAN_SWIFT_V2.fns.createOrderWithToken(
            swift_input_contract,  # address tokenIn (the WETH the SWIFT contract receives)
            effective_amount_in,  # uint256 amountIn
            order_params,  # OrderParams
            b"",  # bytes customPayload (empty = default)
        ).data

        # Step 5 — ABI-encode swapAndForwardEth on the Mayan Forwarder
        forward_calldata: bytes = MAYAN_FORWARDER.fns.swapAndForwardEth(
            amount_in.amount,  # uint256 amountIn (ETH to swap)
            swap_router_address,  # address swapProtocol (DEX router)
            HexBytes(swap_router_calldata),  # bytes swapData
            swift_input_contract,  # address middleToken (WETH)
            min_middle_amount,  # uint256 minMiddleAmount
            swift_contract,  # address mayanProtocol (SWIFT V2 contract)
            swift_calldata,  # bytes mayanData (createOrderWithToken calldata)
        ).data

        return {
            "to": _MAYAN_FORWARDER,
            "data": "0x" + forward_calldata.hex(),
            "value": str(amount_in.amount),
            "gas": str(500_000),
        }

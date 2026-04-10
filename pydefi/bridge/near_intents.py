"""
NEAR Intents cross-chain bridge integration.

NEAR Intents is a cross-chain intent protocol that enables fast, low-cost
bridging of tokens across EVM chains and beyond.  This module wraps the
NEAR Intents 1Click Swap API, which abstracts intent creation, solver
coordination, and transaction execution behind a simple REST interface.

Bridge flow:
1. :meth:`NearIntents.get_quote` — call the API with ``dry=True`` to obtain
   a price estimate without creating a live intent.
2. :meth:`NearIntents.build_bridge_tx` — call the API with ``dry=False`` to
   obtain a *deposit address* for the live intent, then build the on-chain
   deposit transaction that initiates the bridge:

   * **Native ETH** — a plain ETH value transfer to the deposit address.
   * **ERC-20** — a ``transfer(depositAddress, amount)`` call on the token
     contract.

Docs: https://docs.near-intents.org/
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import aiohttp
from eth_contract.erc20 import ERC20

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

_NEAR_INTENTS_API_BASE = "https://1click.chaindefuser.com"

# EVM chain ID → NEAR Intents chain name slug used in asset IDs.
# Asset IDs use the form  ``nep141:{chain}-0x{address}.omft.near``  for
# ERC-20 tokens and  ``nep141:{chain}.omft.near``  for native gas tokens.
_CHAIN_NAMES: dict[int, str] = {
    1: "eth",
    10: "op",
    56: "bsc",
    137: "pol",
    8453: "base",
    42161: "arb",
    43114: "avax",
    59144: "linea",
    534352: "scroll",
    81457: "blast",
    324: "zksync",
    130: "unichain",
    480: "worldchain",
}

# Default quote validity window.  Intentionally short so quotes are fresh.
_DEFAULT_DEADLINE_MINUTES = 10

# Gas estimate for ERC-20 ``transfer`` (tight upper bound; actual usage ~35 k).
_ERC20_TRANSFER_GAS = 65_000

# Gas estimate for a plain native ETH transfer.
_ETH_TRANSFER_GAS = 21_000


def _asset_id(token: Token, chain_name: str) -> str:
    """Return the NEAR Intents asset ID string for *token* on *chain_name*.

    * Native tokens use  ``nep141:{chain}.omft.near``.
    * ERC-20 tokens use  ``nep141:{chain}-{address_lower}.omft.near``.
    """
    if token.is_native():
        return f"nep141:{chain_name}.omft.near"
    addr = token.address.lower()
    return f"nep141:{chain_name}-{addr}.omft.near"


class NearIntents(BaseBridge):
    """NEAR Intents 1Click cross-chain bridge integration.

    Args:
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        api_base_url: Override the 1Click API base URL.
    """

    def __init__(
        self,
        src_chain_id: int,
        dst_chain_id: int,
        api_base_url: str = _NEAR_INTENTS_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self._api_base = api_base_url.rstrip("/")

    @property
    def protocol_name(self) -> str:
        return "NEAR Intents"

    def _chain_name(self, chain_id: int) -> str:
        """Return the NEAR Intents chain name slug for *chain_id*.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If *chain_id* is not
                in the supported chain map.
        """
        name = _CHAIN_NAMES.get(chain_id)
        if name is None:
            raise BridgeError(f"NEAR Intents: unsupported chain ID {chain_id}")
        return name

    async def _request_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        refund_to: str,
        slippage_bps: int,
        dry: bool,
        deadline: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """POST to the 1Click ``/v0/quote`` endpoint.

        Args:
            token_in: Source chain token.
            token_out: Destination chain token.
            amount_in: Input amount.
            recipient: Receiver address on the destination chain.
            refund_to: Refund address on the source chain.
            slippage_bps: Slippage tolerance in basis points.
            dry: When ``True`` the API returns a price estimate only; when
                ``False`` it creates a live intent and returns a
                ``depositAddress``.
            deadline: ISO-8601 deadline string; defaults to *now* plus
                :data:`_DEFAULT_DEADLINE_MINUTES`.
            **kwargs: Extra fields forwarded to the request body.

        Returns:
            Raw API response dict.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        src_chain = self._chain_name(self.src_chain_id)
        dst_chain = self._chain_name(self.dst_chain_id)

        origin_asset = _asset_id(token_in, src_chain)
        dest_asset = _asset_id(token_out, dst_chain)

        if deadline is None:
            deadline = (datetime.now(timezone.utc) + timedelta(minutes=_DEFAULT_DEADLINE_MINUTES)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )

        payload: dict[str, Any] = {
            "swapType": "EXACT_INPUT",
            "originAsset": origin_asset,
            "depositType": "ORIGIN_CHAIN",
            "destinationAsset": dest_asset,
            "amount": str(amount_in.amount),
            "refundTo": refund_to,
            "refundType": "ORIGIN_CHAIN",
            "recipient": recipient,
            "recipientType": "DESTINATION_CHAIN",
            "dry": dry,
            "slippageTolerance": slippage_bps,
            "deadline": deadline,
            **kwargs,
        }

        url = f"{self._api_base}/v0/quote"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(f"NEAR Intents API error ({resp.status}): {data}")
        return data  # type: ignore[return-value]

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Optional[str] = None,
        refund_to: Optional[str] = None,
        slippage_bps: int = 100,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a NEAR Intents bridge quote (dry run — no live intent created).

        Args:
            token_in: Source chain token.
            token_out: Destination chain token.
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.  Defaults to
                a zero address when not provided.
            refund_to: Refund address on the source chain.  Defaults to the
                same placeholder as *recipient* when not provided.
            slippage_bps: Slippage tolerance in basis points (default 100 = 1%).
            **kwargs: Extra parameters forwarded to :meth:`_request_quote`.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error or
                unsupported chain.
        """
        _zero = "0x" + "00" * 20
        _recipient = recipient or _zero
        _refund_to = refund_to or _recipient

        data = await self._request_quote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            recipient=_recipient,
            refund_to=_refund_to,
            slippage_bps=slippage_bps,
            dry=True,
            **kwargs,
        )

        quote = data.get("quote", {})
        amount_out_raw = int(quote.get("amountOut", 0))
        min_amount_out_raw = int(quote.get("minAmountOut", amount_out_raw))
        # The bridge_fee is expressed in token_in raw units.  When token_in
        # and token_out share the same decimals we can compare raw amounts
        # directly.  When the decimals differ (e.g. USDC→ARB) the raw amounts
        # are on different scales and the comparison would be meaningless, so
        # we conservatively report 0; callers may derive a USD-denominated fee
        # from the quote's amountIn/amountOut USD fields instead.
        fee_raw = max(0, amount_in.amount - min_amount_out_raw) if token_in.decimals == token_out.decimals else 0
        estimated_time = int(quote.get("timeEstimate", 60))

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
        slippage_bps: int = 100,
        refund_to: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the on-chain deposit transaction that initiates the bridge.

        Calls the 1Click API with ``dry=False`` to create a live intent and
        obtain a *deposit address*, then constructs the on-chain transaction:

        * **Native ETH** — a plain ETH value transfer to the deposit address.
        * **ERC-20** — a ``transfer(depositAddress, amount)`` call on the token
          contract. The caller must hold enough tokens; no ERC-20 ``approve``
          step or allowance is required because NEAR Intents uses a direct
          ``transfer`` call (not ``transferFrom``).

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points (default 100 = 1%).
            refund_to: Refund address on the source chain.  Defaults to
                *recipient* when not provided.
            **kwargs: Extra parameters forwarded to :meth:`_request_quote`.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error, missing
                deposit address, or unsupported chain.
        """
        _refund_to = refund_to or recipient

        data = await self._request_quote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            recipient=recipient,
            refund_to=_refund_to,
            slippage_bps=slippage_bps,
            dry=False,
            **kwargs,
        )

        quote = data.get("quote", {})
        deposit_address = quote.get("depositAddress")
        if not deposit_address:
            raise BridgeError("NEAR Intents: missing depositAddress in API response")

        if token_in.is_native():
            # Plain ETH transfer to the deposit address.
            return {
                "to": deposit_address,
                "data": "0x",
                "value": str(amount_in.amount),
                "gas": str(_ETH_TRANSFER_GAS),
            }
        else:
            # ERC-20 transfer to the deposit address.
            transfer_calldata: bytes = ERC20.fns.transfer(
                deposit_address,
                amount_in.amount,
            ).data
            return {
                "to": token_in.address,
                "data": "0x" + transfer_calldata.hex(),
                "value": "0",
                "gas": str(_ERC20_TRANSFER_GAS),
            }

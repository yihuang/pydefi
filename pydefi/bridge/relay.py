"""
Relay cross-chain bridge integration.

Relay is a cross-chain bridge and execution protocol that enables fast,
low-cost bridging of native assets and tokens across EVM chains.

Docs: https://docs.relay.link/
"""

from __future__ import annotations

from typing import Any, Optional

import aiohttp

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

_RELAY_API_BASE = "https://api.relay.link"

# Relay uses the zero address for native ETH; the common EeeE... sentinel
# must be normalized before being sent to the API.
_RELAY_NATIVE = "0x0000000000000000000000000000000000000000"
_EEEEE_SENTINEL = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


def _relay_currency(token_address: str) -> str:
    """Return the currency address that Relay's API expects.

    The Relay API uses the zero address for native ETH.  Convert the
    common ``0xEeEe...EEeE`` sentinel to the zero address so the API
    does not reject it with ``INVALID_INPUT_CURRENCY``.
    """
    if token_address.lower() == _EEEEE_SENTINEL:
        return _RELAY_NATIVE
    return token_address


class Relay(BaseBridge):
    """Relay cross-chain bridge integration.

    Args:
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        api_base_url: Override the Relay API base URL.
    """

    def __init__(
        self,
        src_chain_id: int,
        dst_chain_id: int,
        api_base_url: str = _RELAY_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self._api_base = api_base_url.rstrip("/")

    @property
    def protocol_name(self) -> str:
        return "Relay"

    async def _request_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the Relay ``/quote`` endpoint.

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Input amount.
            recipient: Destination address.
            **kwargs: Extra fields forwarded to the request body.

        Returns:
            Raw API response dict.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        payload: dict[str, Any] = {
            "user": recipient,
            "originChainId": self.src_chain_id,
            "destinationChainId": self.dst_chain_id,
            "originCurrency": _relay_currency(token_in.address),
            "destinationCurrency": _relay_currency(token_out.address),
            "amount": str(amount_in.amount),
            "tradeType": "EXACT_INPUT",
            **kwargs,
        }

        url = f"{self._api_base}/quote"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(f"Relay API error ({resp.status}): {data}")
        return data  # type: ignore[return-value]

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Optional[str] = None,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a Relay bridge quote.

        Args:
            token_in: Source chain token.
            token_out: Destination chain token.
            amount_in: Amount to bridge.
            recipient: Receiver address (required by the Relay API; defaults
                to a zero address when not provided).
            **kwargs: Extra parameters forwarded to :meth:`_request_quote`.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        _recipient = recipient or ("0x" + "00" * 20)
        data = await self._request_quote(token_in, token_out, amount_in, _recipient, **kwargs)

        details = data.get("details", {})
        currency_out = details.get("currencyOut", {})
        amount_out_raw = int(currency_out.get("amount", 0))

        # Fee is meaningful only when token_in and token_out share the same
        # decimals (i.e. same asset bridged across chains).  When decimals
        # differ the raw amounts are not comparable, so we default to 0.
        if token_in.decimals == token_out.decimals:
            fee_raw = max(0, amount_in.amount - amount_out_raw)
        else:
            fee_raw = 0

        # Relay is typically very fast (< 30 seconds on most routes)
        estimated_time = int(details.get("timeEstimate", 30))

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Relay bridge transaction.

        Fetches the transaction data from the Relay ``/quote`` endpoint
        (which includes calldata) and returns it in the standard format.

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            **kwargs: Extra parameters forwarded to :meth:`_request_quote`.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error or missing
                transaction data.
        """
        data = await self._request_quote(
            token_in,
            token_out,
            amount_in,
            recipient,
            slippage=slippage_bps,
            **kwargs,
        )

        steps = data.get("steps", [])
        if not steps:
            raise BridgeError("Relay: no transaction steps returned from API")

        # Use the first (deposit) step's transaction
        first_step = steps[0]
        items = first_step.get("items", [])
        if not items:
            raise BridgeError("Relay: no items in first transaction step")

        tx_data = items[0].get("data", {})
        to_addr = tx_data.get("to")
        call_data = tx_data.get("data")
        if not to_addr:
            raise BridgeError("Relay: missing 'to' field in transaction data")
        if not call_data:
            raise BridgeError("Relay: missing 'data' field in transaction data")
        return {
            "to": to_addr,
            "data": call_data,
            "value": str(tx_data.get("value", "0")),
            "gas": str(tx_data.get("gas", 300_000)),
        }

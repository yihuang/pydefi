"""
Mayan Finance cross-chain bridge integration.

Mayan is a cross-chain swap protocol built on Wormhole that enables fast
bridging between EVM chains and Solana.  This module wraps the Mayan
Price API and Swift contract for on-chain execution.

Docs: https://docs.mayan.finance/
"""

from __future__ import annotations

from typing import Any, Optional

import aiohttp

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
                (e.g. ``slippage_bps``, ``swift``).

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        from_chain = self._chain_name(self.src_chain_id)
        to_chain = self._chain_name(self.dst_chain_id)
        human_amount = str(amount_in.human_amount)

        params: dict[str, Any] = {
            "amount": human_amount,
            "fromToken": token_in.address,
            "toToken": token_out.address,
            "fromChain": from_chain,
            "toChain": to_chain,
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
        """Build a Mayan bridge transaction via the swap API.

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Amount to send.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            referrer: Optional referrer address for fee sharing.
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        from_chain = self._chain_name(self.src_chain_id)
        to_chain = self._chain_name(self.dst_chain_id)
        slippage_pct = slippage_bps / 100.0
        human_amount = str(amount_in.human_amount)

        payload: dict[str, Any] = {
            "amountIn": human_amount,
            "fromToken": token_in.address,
            "toToken": token_out.address,
            "fromChain": from_chain,
            "toChain": to_chain,
            "toAddress": recipient,
            "slippage": slippage_pct,
            **kwargs,
        }
        if referrer is not None:
            payload["referrer"] = referrer

        url = f"{self._api_base}/swap"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(
                        f"Mayan swap API error ({resp.status}): {data}"
                    )

        tx = data.get("tx", data)
        return {
            "to": tx.get("to", ""),
            "data": tx.get("data", "0x"),
            "value": str(tx.get("value", "0")),
            "gas": str(tx.get("gas", 500_000)),
        }

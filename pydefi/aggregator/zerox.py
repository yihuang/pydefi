"""
0x Protocol DEX aggregator API client.

Docs: https://0x.org/docs/api#tag/Swap
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount

# Mapping from chain IDs to 0x API subdomain / base URLs
_CHAIN_URLS: dict[int, str] = {
    1: "https://api.0x.org",
    10: "https://optimism.api.0x.org",
    56: "https://bsc.api.0x.org",
    137: "https://polygon.api.0x.org",
    42161: "https://arbitrum.api.0x.org",
    43114: "https://avalanche.api.0x.org",
    8453: "https://base.api.0x.org",
}


class ZeroX(BaseAggregator):
    """0x Protocol DEX aggregator API client.

    The 0x Swap API provides smart order routing across liquidity sources.

    Args:
        chain_id: EVM chain ID.
        api_key: 0x API key (required for production use).
        base_url: Override the default chain-specific base URL.
    """

    def __init__(
        self,
        chain_id: int,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(chain_id, api_key)
        self._base_url = base_url or _CHAIN_URLS.get(chain_id, "https://api.0x.org")

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def protocol_name(self) -> str:
        return "0x"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["0x-api-key"] = self.api_key
        return headers

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    reason = data.get("reason", data.get("validationErrors", data))
                    raise AggregatorError(
                        f"0x API error: {reason}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def get_price(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the 0x ``/swap/v1/price`` (indicative quote) endpoint.

        Returns:
            Raw API response dict.
        """
        params: dict[str, Any] = {
            "sellToken": amount_in.token.address,
            "buyToken": token_out.address,
            "sellAmount": str(amount_in.amount),
            **kwargs,
        }
        return await self._get("swap/v1/price", params)

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a firm quote from the 0x ``/swap/v1/quote`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Additional query parameters (e.g. ``takerAddress``).

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` ready to broadcast.
        """
        params: dict[str, Any] = {
            "sellToken": amount_in.token.address,
            "buyToken": token_out.address,
            "sellAmount": str(amount_in.amount),
            "slippagePercentage": self._slippage_to_fraction(slippage_bps),
            **kwargs,
        }
        data = await self._get("swap/v1/quote", params)

        buy_amount = int(data["buyAmount"])
        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = buy_amount * slippage_factor // 10_000
        gas_estimate = int(data.get("estimatedGas", data.get("gas", 0)))
        price_impact_raw = float(data.get("estimatedPriceImpact", "0"))

        tx_data = {
            "to": data.get("to", ""),
            "data": data.get("data", ""),
            "value": data.get("value", "0"),
            "gas": str(gas_estimate),
            "gasPrice": data.get("gasPrice", ""),
        }

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=buy_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=Decimal(str(price_impact_raw)),
            tx_data=tx_data,
            protocol=self.protocol_name,
            route_summary=str(data.get("sources", "")),
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a 0x quote.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.

        Returns:
            A :class:`~pydefi.types.SwapRoute`.
        """
        quote = await self.get_quote(amount_in, token_out, slippage_bps, **kwargs)

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=Address(quote.tx_data["to"]) if quote.tx_data.get("to") else None,
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=quote.amount_out,
            price_impact=quote.price_impact,
        )

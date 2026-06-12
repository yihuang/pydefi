"""
ParaSwap DEX aggregator API client.

Docs: https://developers.paraswap.network/api/master
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi._math import apply_slippage
from pydefi.aggregator.base import AggregatorQuote
from pydefi.exceptions import AggregatorError
from pydefi.types import SwapRoute, Token, TokenAmount


class ParaSwap:
    """ParaSwap DEX aggregator API client.

    Args:
        chain_id: EVM chain ID.
        api_key: Optional ParaSwap API key.
        base_url: Override the default API base URL.
    """

    _DEFAULT_BASE_URL = "https://apiv5.paraswap.io"

    def __init__(
        self,
        chain_id: int,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.chain_id = chain_id
        self.api_key = api_key
        self._base_url = base_url or self._DEFAULT_BASE_URL

    @property
    def base_url(self) -> str:
        return self._base_url

    protocol_name: str = "ParaSwap"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"ParaSwap API error: {data.get('error', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def _post(self, endpoint: str, json_body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_body, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"ParaSwap API error: {data.get('error', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def get_price(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the ParaSwap ``/prices`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.

        Returns:
            Raw API response dict.
        """
        params: dict[str, Any] = {
            "srcToken": amount_in.token.encoded_address,
            "srcDecimals": amount_in.token.decimals,
            "destToken": token_out.encoded_address,
            "destDecimals": token_out.decimals,
            "amount": str(amount_in.amount),
            "network": self.chain_id,
            **kwargs,
        }
        return await self._get("prices", params)

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a quote from ParaSwap.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.
        """
        data = await self.get_price(amount_in, token_out, **kwargs)
        price_route = data.get("priceRoute", {})

        dest_amount = int(price_route.get("destAmount", 0))
        gas_cost = int(price_route.get("gasCost", 0))
        price_impact_raw = price_route.get("percentChange", "0")

        return AggregatorQuote.from_quote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out_raw=dest_amount,
            slippage_bps=slippage_bps,
            gas_estimate=gas_cost,
            price_impact=Decimal(str(price_impact_raw)),
            protocol=self.protocol_name,
            route_summary=str(price_route.get("bestRoute", "")),
            tx_data={"priceRoute": price_route},
        )

    async def build_transaction(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        from_address: str,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a ready-to-sign transaction from ParaSwap ``/transactions``.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            from_address: Sender wallet address.
            slippage_bps: Maximum slippage in basis points.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated.
        """
        # First get the price route
        price_data = await self.get_price(amount_in, token_out, **kwargs)
        price_route = price_data.get("priceRoute", {})
        dest_amount = int(price_route.get("destAmount", 0))

        min_amount_out_raw = apply_slippage(dest_amount, slippage_bps)

        body: dict[str, Any] = {
            "srcToken": amount_in.token.encoded_address,
            "srcDecimals": amount_in.token.decimals,
            "destToken": token_out.encoded_address,
            "destDecimals": token_out.decimals,
            "srcAmount": str(amount_in.amount),
            "destAmount": str(min_amount_out_raw),
            "priceRoute": price_route,
            "userAddress": from_address,
            "network": self.chain_id,
            **kwargs,
        }
        tx_data = await self._post(f"transactions/{self.chain_id}", body)

        return AggregatorQuote.from_quote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out_raw=dest_amount,
            slippage_bps=slippage_bps,
            gas_estimate=int(tx_data.get("gas", 0)),
            price_impact=Decimal(str(price_route.get("percentChange", "0"))),
            tx_data=tx_data,
            protocol=self.protocol_name,
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a ParaSwap quote."""
        quote = await self.get_quote(amount_in, token_out, slippage_bps, **kwargs)
        return quote.to_swap_route()

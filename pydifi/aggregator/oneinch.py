"""
1inch DEX aggregator API client.

Docs: https://docs.1inch.io/docs/aggregation-protocol/api/swagger/
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import aiohttp

from pydifi.aggregator.base import AggregatorQuote, BaseAggregator
from pydifi.exceptions import AggregatorError
from pydifi.types import SwapRoute, SwapStep, Token, TokenAmount


class OneInch(BaseAggregator):
    """1inch Aggregation Protocol API client.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: Optional 1inch Developer Portal API key.
        base_url: Override the default API base URL.
    """

    _DEFAULT_BASE_URL = "https://api.1inch.dev/swap/v6.0"

    def __init__(
        self,
        chain_id: int,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(chain_id, api_key)
        self._base_url = base_url or self._DEFAULT_BASE_URL

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def protocol_name(self) -> str:
        return "1inch"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chain_url(self, endpoint: str) -> str:
        return f"{self._base_url}/{self.chain_id}/{endpoint.lstrip('/')}"

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self._chain_url(endpoint)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"1inch API error: {data.get('description', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a quote from the 1inch ``/quote`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters forwarded to the API.

        Returns:
            An :class:`~pydifi.aggregator.base.AggregatorQuote`.
        """
        params: dict[str, Any] = {
            "src": amount_in.token.address,
            "dst": token_out.address,
            "amount": str(amount_in.amount),
            **kwargs,
        }
        data = await self._get("quote", params)

        dst_amount = int(data["dstAmount"])
        slippage_fraction = 10_000 - slippage_bps
        min_amount_out_raw = dst_amount * slippage_fraction // 10_000

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=dst_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=int(data.get("gas", 0)),
            price_impact=Decimal(str(data.get("estimatedPriceImpact", "0"))),
            protocol=self.protocol_name,
            route_summary=str(data.get("protocols", "")),
        )

    async def get_swap(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        from_address: str,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a fully-encoded swap transaction from the 1inch ``/swap`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            from_address: Wallet address that will execute the swap.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters.

        Returns:
            An :class:`~pydifi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated.
        """
        params: dict[str, Any] = {
            "src": amount_in.token.address,
            "dst": token_out.address,
            "amount": str(amount_in.amount),
            "from": from_address,
            "slippage": self._slippage_to_percent(slippage_bps),
            **kwargs,
        }
        data = await self._get("swap", params)

        tx = data.get("tx", {})
        dst_amount = int(data["dstAmount"])
        slippage_fraction = 10_000 - slippage_bps
        min_amount_out_raw = dst_amount * slippage_fraction // 10_000

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=dst_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=int(tx.get("gas", data.get("gas", 0))),
            price_impact=Decimal(0),
            tx_data=tx,
            protocol=self.protocol_name,
            route_summary=str(data.get("protocols", "")),
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydifi.types.SwapRoute` from a 1inch quote.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.

        Returns:
            A :class:`~pydifi.types.SwapRoute`.
        """
        quote = await self.get_quote(amount_in, token_out, slippage_bps, **kwargs)

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address="",  # 1inch routes through multiple pools
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=quote.amount_out,
            price_impact=quote.price_impact,
        )

"""
OKX DEX aggregator API client.

Docs: https://web3.okx.com/zh-hant/onchainos/dev-docs/trade/dex-swap
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import SwapRoute, SwapStep, Token, TokenAmount


class OKX(BaseAggregator):
    """OKX DEX aggregator API client.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: Optional OKX API key.
        base_url: Override the default API base URL.
    """

    _DEFAULT_BASE_URL = "https://www.okx.com/api/v6/dex/aggregator"

    def __init__(
        self,
        chain_id: int,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(chain_id, api_key)
        self._base_url = base_url or self._DEFAULT_BASE_URL

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def protocol_name(self) -> str:
        return "OKX"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["OK-ACCESS-KEY"] = self.api_key
        return headers

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200 or str(data.get("code", "0")) != "0":
                    msg = data.get("msg") or data.get("error", data)
                    raise AggregatorError(
                        f"OKX API error: {msg}",
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
        """Fetch a quote from the OKX DEX ``/quote`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters forwarded to the API.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.
        """
        params: dict[str, Any] = {
            "chainIndex": str(self.chain_id),
            "amount": str(amount_in.amount),
            "fromTokenAddress": amount_in.token.address,
            "toTokenAddress": token_out.address,
            "slippagePercent": str(self._slippage_to_percent(slippage_bps)),
            **kwargs,
        }
        data = await self._get("quote", params)
        result = data["data"]

        to_amount = int(result["toTokenAmount"])
        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = to_amount * slippage_factor // 10_000
        gas_estimate = int(result.get("estimateGasFee", 0))

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=to_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=Decimal(str(result.get("priceImpactPercentage", "0"))),
            protocol=self.protocol_name,
            route_summary=str(result.get("routerResult", "")),
        )

    async def get_swap(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        from_address: str,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a fully-encoded swap transaction from the OKX DEX ``/swap`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            from_address: Wallet address that will execute the swap.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated.
        """
        params: dict[str, Any] = {
            "chainIndex": str(self.chain_id),
            "amount": str(amount_in.amount),
            "fromTokenAddress": amount_in.token.address,
            "toTokenAddress": token_out.address,
            "slippagePercent": str(self._slippage_to_percent(slippage_bps)),
            "userWalletAddress": from_address,
            **kwargs,
        }
        data = await self._get("swap", params)
        result = data["data"]

        to_amount = int(result["routerResult"]["toTokenAmount"])
        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = to_amount * slippage_factor // 10_000

        tx_info = result.get("tx", {})
        gas_estimate = int(tx_info.get("gas", result.get("estimateGasFee", 0)))

        tx_data = {
            "to": tx_info.get("to", ""),
            "data": tx_info.get("data", ""),
            "value": tx_info.get("value", "0"),
            "gas": str(gas_estimate),
            "gasPrice": tx_info.get("gasPrice", ""),
        }

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=to_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=Decimal(str(result.get("priceImpactPercentage", "0"))),
            tx_data=tx_data,
            protocol=self.protocol_name,
            route_summary=str(result.get("routerResult", "")),
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from an OKX DEX quote.

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
            pool_address="",
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=quote.amount_out,
            price_impact=quote.price_impact,
        )

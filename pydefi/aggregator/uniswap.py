"""
Uniswap Trading API client.

Docs: https://api-docs.uniswap.org/guides/integration_guide
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import aiohttp

from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import SwapRoute, SwapStep, Token, TokenAmount


class UniswapAPI(BaseAggregator):
    """Uniswap Trading API client.

    Implements the end-to-end swap flow described in the Uniswap Trading API
    integration guide: ``POST /v1/quote`` to fetch a price quote, then
    ``POST /v1/swap`` to build a ready-to-submit transaction.

    Base URL: ``https://trade-api.gateway.uniswap.org``

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: Uniswap Trading API key (sent as ``x-api-key`` header).
        base_url: Override the default API base URL.
        origin: Optional ``Origin`` header value (rarely needed; ignored by
            the gateway unless your key has domain restrictions).
    """

    _DEFAULT_BASE_URL = "https://trade-api.gateway.uniswap.org"

    def __init__(
        self,
        chain_id: int,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> None:
        super().__init__(chain_id, api_key)
        self._base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self._origin = origin

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def protocol_name(self) -> str:
        return "Uniswap"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if self._origin:
            headers["Origin"] = self._origin
        return headers

    def _api_error_msg(self, data: dict[str, Any], status: int) -> str:
        detail = data.get("detail") or data.get("errorCode") or data.get("message")
        if detail is None:
            detail = data
        return f"Uniswap API error {status}: {detail}"

    async def _post(self, endpoint: str, json_body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=json_body, headers=self._headers()
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        self._api_error_msg(data, resp.status),
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    def _parse_classic_quote(
        self,
        data: dict[str, Any],
        token_out: Token,
        slippage_bps: int,
    ) -> tuple[int, int, int, Decimal, str]:
        """Extract amounts and metadata from a ``/v1/quote`` response.

        Returns ``(amount_out_raw, min_amount_out_raw, gas_estimate,
        price_impact, route_summary)``.
        """
        quote_data = data.get("quote", data)
        output = quote_data.get("output", {})
        amount_out_raw = int(output.get("amount", quote_data.get("amountOut", 0)))

        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = amount_out_raw * slippage_factor // 10_000

        gas_fee = quote_data.get("gasFee") or quote_data.get("gasUseEstimate", 0)
        gas_estimate = int(gas_fee) if gas_fee else 0

        price_impact_raw = quote_data.get("priceImpact", 0)
        route_summary = str(
            quote_data.get("routeString", quote_data.get("route", ""))
        )
        return (
            amount_out_raw,
            min_amount_out_raw,
            gas_estimate,
            Decimal(str(price_impact_raw)),
            route_summary,
        )

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        swapper: Optional[str] = None,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a price quote from ``POST /v1/quote``.

        The quote uses ``EXACT_INPUT`` type: the caller specifies an exact sell
        amount and receives the best possible buy amount.

        Args:
            amount_in: Exact input token amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points
                (e.g. ``50`` → 0.5 %).
            swapper: Optional wallet address.  When provided it is passed to
                the API as the ``swapper`` field, which enables on-chain
                simulation and more accurate gas estimates.
            **kwargs: Additional body fields forwarded to the API
                (e.g. ``routingPreference="BEST_PRICE"``).

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        body: dict[str, Any] = {
            "tokenIn": amount_in.token.address,
            "tokenInChainId": self.chain_id,
            "tokenOut": token_out.address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "slippageTolerance": self._slippage_to_percent(slippage_bps),
        }
        if swapper is not None:
            body["swapper"] = swapper
        body.update(kwargs)

        data = await self._post("v1/quote", body)

        amount_out_raw, min_amount_out_raw, gas_estimate, price_impact, route_summary = (
            self._parse_classic_quote(data, token_out, slippage_bps)
        )

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=price_impact,
            protocol=self.protocol_name,
            route_summary=route_summary,
            tx_data={"quoteData": data},
        )

    async def get_swap(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        wallet_address: str,
        slippage_bps: int = 50,
        deadline: Optional[int] = None,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a quote and build a ready-to-submit transaction.

        Implements the two-step flow from the Uniswap Trading API guide:

        1. ``POST /v1/quote`` — obtain a price quote (``swapper`` is set to
           *wallet_address* so the API can simulate the transaction).
        2. ``POST /v1/swap`` — convert the quote into signed calldata.

        Args:
            amount_in: Exact input token amount.
            token_out: Desired output token.
            wallet_address: Address that will execute the swap.
            slippage_bps: Maximum acceptable slippage in basis points.
            deadline: Optional UNIX timestamp for the transaction deadline.
            **kwargs: Additional body fields forwarded to the quote call.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated from the ``/v1/swap`` response.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        # Step 1: POST /v1/quote
        quote_body: dict[str, Any] = {
            "tokenIn": amount_in.token.address,
            "tokenInChainId": self.chain_id,
            "tokenOut": token_out.address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "swapper": wallet_address,
            "slippageTolerance": self._slippage_to_percent(slippage_bps),
        }
        quote_body.update(kwargs)
        quote_response = await self._post("v1/quote", quote_body)

        amount_out_raw, min_amount_out_raw, gas_estimate, price_impact, route_summary = (
            self._parse_classic_quote(quote_response, token_out, slippage_bps)
        )

        # Step 2: POST /v1/swap
        # The swap body takes the inner ``quote`` object (ClassicQuote), not
        # the full QuoteResponse.
        inner_quote = quote_response.get("quote", quote_response)
        swap_body: dict[str, Any] = {"quote": inner_quote}
        if deadline is not None:
            swap_body["deadline"] = deadline
        # Include permitData when the quote response requires a Permit2 signature.
        permit_data = quote_response.get("permitData")
        if permit_data:
            swap_body["permitData"] = permit_data

        swap_response = await self._post("v1/swap", swap_body)

        tx = swap_response.get("swap", swap_response.get("transaction", {}))

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=price_impact,
            tx_data=tx,
            protocol=self.protocol_name,
            route_summary=route_summary,
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a Uniswap quote.

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
            pool_address="",  # Uniswap API routes through multiple pools
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=quote.amount_out,
            price_impact=quote.price_impact,
        )

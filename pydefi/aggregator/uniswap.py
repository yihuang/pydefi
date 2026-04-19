"""
Uniswap Trading API client.

Docs: https://api-docs.uniswap.org/guides/integration_guide
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

import aiohttp

from pydefi._utils import encode_address
from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount

# Routing types returned by /v1/quote that are compatible with POST /v1/swap.
# UniswapX types (DUTCH_V2, DUTCH_V3, PRIORITY) must use POST /v1/order instead.
_SWAP_COMPATIBLE_ROUTING: frozenset[str] = frozenset({"CLASSIC", "WRAP", "UNWRAP", "BRIDGE"})


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
            async with session.post(url, json=json_body, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        self._api_error_msg(data, resp.status),
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    def _parse_quote_response(
        self,
        data: dict[str, Any],
        token_out: Token,
        slippage_bps: int,
    ) -> tuple[int, int, int, Decimal, str]:
        """Extract amounts and metadata from a ``/v1/quote`` response.

        Handles both CLASSIC AMM quotes and UniswapX (DUTCH_V2/V3/PRIORITY)
        quotes, which use different response shapes.

        Returns ``(amount_out_raw, min_amount_out_raw, gas_estimate,
        price_impact, route_summary)``.
        """
        routing = data.get("routing", "CLASSIC")
        quote_data = data.get("quote", data)

        if routing in _SWAP_COMPATIBLE_ROUTING:
            # CLASSIC / WRAP / UNWRAP / BRIDGE — standard AMM response shape.
            output = quote_data.get("output", {})
            amount_out_raw = int(output.get("amount", quote_data.get("amountOut", 0)))
            slippage_factor = 10_000 - slippage_bps
            min_amount_out_raw = amount_out_raw * slippage_factor // 10_000
            gas_fee = quote_data.get("gasFee") or quote_data.get("gasUseEstimate", 0)
            gas_estimate = int(gas_fee) if gas_fee else 0
            price_impact_raw = quote_data.get("priceImpact", 0)
            route_summary = str(quote_data.get("routeString", quote_data.get("route", "")))
        else:
            # UniswapX (DUTCH_V2, DUTCH_V3, PRIORITY) — amounts come from
            # aggregatedOutputs (preferred) or orderInfo.outputs.
            aggregated = quote_data.get("aggregatedOutputs", [])
            if aggregated:
                amount_out_raw = int(aggregated[0].get("amount", 0))
                min_amount_out_raw = int(aggregated[0].get("minAmount", 0))
                if not min_amount_out_raw:
                    slippage_factor = 10_000 - slippage_bps
                    min_amount_out_raw = amount_out_raw * slippage_factor // 10_000
            else:
                order_outputs = quote_data.get("orderInfo", {}).get("outputs", [])
                amount_out_raw = int(order_outputs[0]["startAmount"]) if order_outputs else 0
                end_amount = int(order_outputs[0]["endAmount"]) if order_outputs else 0
                min_amount_out_raw = end_amount or (amount_out_raw * (10_000 - slippage_bps) // 10_000)
            gas_estimate = 0
            price_impact_raw = 0
            route_summary = routing

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
        swapper: Optional[Address] = None,
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
            "tokenIn": amount_in.token.encoded_address,
            "tokenInChainId": self.chain_id,
            "tokenOut": token_out.encoded_address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "slippageTolerance": self._slippage_to_percent(slippage_bps),
        }
        if swapper is not None:
            body["swapper"] = encode_address(swapper, self.chain_id)
        body.update(kwargs)

        data = await self._post("v1/quote", body)

        amount_out_raw, min_amount_out_raw, gas_estimate, price_impact, route_summary = self._parse_quote_response(
            data, token_out, slippage_bps
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
        wallet_address: Address,
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
                Pass ``signature=<hex>`` (a signed Permit2 payload) to enable
                gasless Permit2 approval — forwarded to ``/v1/swap`` together
                with ``permitData`` from the quote response.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated from the ``/v1/swap`` response.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        # ``signature`` is a swap-only parameter; extract it before updating
        # the quote body so it is not accidentally forwarded to /v1/quote.
        signature: str | None = kwargs.pop("signature", None)
        # Step 1: POST /v1/quote
        quote_body: dict[str, Any] = {
            "tokenIn": amount_in.token.encoded_address,
            "tokenInChainId": self.chain_id,
            "tokenOut": token_out.encoded_address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "swapper": encode_address(wallet_address, self.chain_id),
            "slippageTolerance": self._slippage_to_percent(slippage_bps),
        }
        quote_body.update(kwargs)
        quote_response = await self._post("v1/quote", quote_body)

        # Routing types supported by /v1/swap (per the integration guide).
        # UniswapX types (DUTCH_V2, DUTCH_V3, PRIORITY) require /v1/order instead.
        routing = quote_response.get("routing", "CLASSIC")
        if routing not in _SWAP_COMPATIBLE_ROUTING:
            raise AggregatorError(
                f"Quote routing type '{routing}' cannot be submitted via /v1/swap. "
                f"Use protocols=['V2','V3','V4'] to request classic AMM routing, "
                f"or handle {routing!r} via /v1/order.",
                status_code=None,
            )

        amount_out_raw, min_amount_out_raw, gas_estimate, price_impact, route_summary = self._parse_quote_response(
            quote_response, token_out, slippage_bps
        )

        # Step 2: POST /v1/swap
        # The swap body takes the inner ``quote`` object (ClassicQuote), not
        # the full QuoteResponse.  Note: ``permitData`` from the quote response
        # is intentionally *not* forwarded here — it requires a paired Permit2
        # ``signature`` that only the user's wallet can produce.  Callers that
        # have already signed can pass the signature via ``signature`` kwarg
        # which will be forwarded together with permitData.
        inner_quote = quote_response.get("quote", quote_response)
        swap_body: dict[str, Any] = {"quote": inner_quote}
        if deadline is not None:
            swap_body["deadline"] = deadline
        # Forward permitData + signature together when the caller has already
        # signed the Permit2 payload.  The API requires both or neither.
        permit_data = quote_response.get("permitData")
        if permit_data and signature:
            swap_body["permitData"] = permit_data
            swap_body["signature"] = signature

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

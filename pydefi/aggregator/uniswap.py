"""
Uniswap Trading API client.

Docs: https://api-docs.uniswap.org/guides/swapping_end_to_end
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
    documentation: fetch a quote, then optionally build a ready-to-submit
    transaction via the swap endpoint.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: Uniswap Trading API key.  Sent as both ``x-api-key`` and
            ``Authorization: Bearer <key>`` headers so it works regardless of
            which auth scheme a particular API gateway deployment expects.
        base_url: Override the default API base URL.
        origin: Value for the ``Origin`` request header.  Some Uniswap API
            deployments validate the origin of server-side callers; pass the
            origin that was registered with your API key if needed.
    """

    _DEFAULT_BASE_URL = "https://api.uniswap.org"

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
            # Send the key in both formats: some API gateway deployments use
            # the AWS-style x-api-key header while others use Bearer tokens.
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._origin:
            headers["Origin"] = self._origin
        return headers

    def _api_error_msg(self, data: dict[str, Any], status: int) -> str:
        detail = data.get("detail") or data.get("errorCode") or data.get("message")
        if detail is None:
            detail = data
        return f"Uniswap API error {status}: {detail}"

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, headers=self._headers()
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        self._api_error_msg(data, resp.status),
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

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

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a quote from the Uniswap Trading API ``/v2/quote`` endpoint.

        The quote uses ``EXACT_INPUT`` type: the caller specifies an exact sell
        amount and receives the best possible buy amount.

        Args:
            amount_in: Exact input token amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
                Converted to a percentage string (e.g. ``50`` → ``"0.5"``).
            **kwargs: Additional query parameters forwarded to the API
                (e.g. ``protocols="V2,V3,V4"``).

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        params: dict[str, Any] = {
            "tokenInAddress": amount_in.token.address,
            "tokenInChainId": self.chain_id,
            "tokenOutAddress": token_out.address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "slippageTolerance": str(self._slippage_to_percent(slippage_bps)),
            **kwargs,
        }
        data = await self._get("v2/quote", params)

        # The /v2/quote response nests the output amount under quote.output.amount
        quote_data = data.get("quote", data)
        output = quote_data.get("output", {})
        amount_out_raw = int(output.get("amount", quote_data.get("amountOut", 0)))

        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = amount_out_raw * slippage_factor // 10_000

        gas_fee = quote_data.get("gasFee", quote_data.get("gasUseEstimate", 0))
        gas_estimate = int(gas_fee) if gas_fee else 0

        price_impact_raw = quote_data.get("priceImpact", "0")

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=Decimal(str(price_impact_raw)),
            protocol=self.protocol_name,
            route_summary=str(quote_data.get("route", "")),
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
        first calls ``GET /v2/quote``, then submits the quote to
        ``POST /v2/swap`` to obtain the signed transaction calldata.

        Args:
            amount_in: Exact input token amount.
            token_out: Desired output token.
            wallet_address: Address that will execute the swap.
            slippage_bps: Maximum acceptable slippage in basis points.
            deadline: Optional UNIX timestamp for the transaction deadline.
            **kwargs: Additional parameters forwarded to both API calls.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated from the swap endpoint response.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        # Step 1: get quote
        params: dict[str, Any] = {
            "tokenInAddress": amount_in.token.address,
            "tokenInChainId": self.chain_id,
            "tokenOutAddress": token_out.address,
            "tokenOutChainId": self.chain_id,
            "amount": str(amount_in.amount),
            "type": "EXACT_INPUT",
            "slippageTolerance": str(self._slippage_to_percent(slippage_bps)),
            **kwargs,
        }
        quote_response = await self._get("v2/quote", params)

        quote_data = quote_response.get("quote", quote_response)
        output = quote_data.get("output", {})
        amount_out_raw = int(output.get("amount", quote_data.get("amountOut", 0)))

        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = amount_out_raw * slippage_factor // 10_000

        gas_fee = quote_data.get("gasFee", quote_data.get("gasUseEstimate", 0))
        gas_estimate = int(gas_fee) if gas_fee else 0

        # Step 2: build transaction
        swap_body: dict[str, Any] = {
            "quote": quote_response,
            "walletAddress": wallet_address,
            "slippage": self._slippage_to_percent(slippage_bps),
        }
        if deadline is not None:
            swap_body["deadline"] = deadline

        swap_response = await self._post("v2/swap", swap_body)

        tx = swap_response.get("swap", swap_response.get("transaction", {}))

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=Decimal(str(quote_data.get("priceImpact", "0"))),
            tx_data=tx,
            protocol=self.protocol_name,
            route_summary=str(quote_data.get("route", "")),
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

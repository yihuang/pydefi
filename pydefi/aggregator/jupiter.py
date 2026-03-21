"""
Jupiter DEX aggregator API client for Solana.

Jupiter is the primary liquidity aggregator on Solana, routing swaps through
all major DEXes (Raydium, Orca, Meteora, …) to find the best price.

Docs: https://station.jup.ag/docs/apis/swap-api
API:  https://quote-api.jup.ag/v6/
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import ChainId, SwapRoute, SwapStep, Token, TokenAmount

_JUPITER_API_BASE = "https://quote-api.jup.ag/v6"


class Jupiter(BaseAggregator):
    """Jupiter DEX aggregator API client for Solana.

    Jupiter routes swaps across all major Solana DEXes.  Token addresses are
    Solana mint addresses (base-58 encoded), and amounts are raw integer
    values in the token's smallest unit (same convention as EVM integrations).

    Args:
        api_key: Optional Jupiter API key for priority access.
        base_url: Override the default Jupiter V6 API base URL.
    """

    _DEFAULT_BASE_URL = _JUPITER_API_BASE

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(chain_id=ChainId.SOLANA, api_key=api_key)
        self._base_url = base_url or self._DEFAULT_BASE_URL

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def protocol_name(self) -> str:
        return "Jupiter"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"Jupiter API error: {data.get('error', data)}",
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
        """Fetch a quote from the Jupiter ``/quote`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (Solana mint address).
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra query parameters forwarded to the API.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "slippageBps": slippage_bps,
            **kwargs,
        }
        data = await self._get("quote", params)

        out_amount = int(data["outAmount"])
        # Jupiter returns otherAmountThreshold as the minimum out after slippage
        min_out_amount = int(data.get("otherAmountThreshold", out_amount))
        # priceImpactPct is a percentage (e.g. "0.03" = 0.03%); convert to fraction
        price_impact = Decimal(str(data.get("priceImpactPct", "0"))) / Decimal(100)

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=out_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_out_amount),
            gas_estimate=0,  # Solana uses compute units, not EVM gas
            price_impact=price_impact,
            protocol=self.protocol_name,
            route_summary=str(data.get("routePlan", "")),
        )

    async def get_swap_transaction(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        user_public_key: str,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch an encoded Solana swap transaction from Jupiter.

        Calls ``/quote`` to obtain a fresh quote and then ``/swap`` to obtain
        a base-64 encoded, versioned Solana transaction ready for signing.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            user_public_key: The signer's Solana wallet address (base-58).
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra parameters forwarded to the ``/swap`` endpoint
                (e.g. ``prioritizationFeeLamports``, ``dynamicComputeUnitLimit``).

        Returns:
            Dict containing ``swapTransaction`` (base-64 encoded) and
            ``lastValidBlockHeight``.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        # Fetch the raw quote response – Jupiter /swap requires the full quote object
        quote_params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "slippageBps": slippage_bps,
        }
        quote_response = await self._get("quote", quote_params)

        url = f"{self._base_url.rstrip('/')}/swap"
        payload: dict[str, Any] = {
            "quoteResponse": quote_response,
            "userPublicKey": user_public_key,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"Jupiter swap API error: {data.get('error', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a Jupiter quote.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra query parameters forwarded to :meth:`get_quote`.

        Returns:
            A :class:`~pydefi.types.SwapRoute` with a single
            :class:`~pydefi.types.SwapStep` (Jupiter aggregates internally).
        """
        quote = await self.get_quote(amount_in, token_out, slippage_bps, **kwargs)

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address="",  # Jupiter routes across multiple pools internally
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=quote.amount_out,
            price_impact=quote.price_impact,
        )

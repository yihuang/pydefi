"""
Jupiter DEX aggregator API clients for Solana.

Two integrations are provided:

* :class:`Jupiter` — Metis V6 quote API (``https://quote-api.jup.ag/v6``).
  Free-tier friendly; no API key required for basic use.  Provides
  ``get_quote``, ``get_swap_transaction``, and ``build_swap_route``.

* :class:`JupiterSwapV2` — Unified Swap V2 API (``https://api.jup.ag/swap/v2``).
  Requires an API key from ``portal.jup.ag``.  Two flows:

  * **Order flow** (managed execution): ``get_order`` → sign →
    ``execute_order``.  Jupiter broadcasts via its proprietary Jupiter Beam
    engine and polls for confirmation.
  * **Build flow** (custom transactions): ``get_build`` returns raw
    instructions so callers can compose their own versioned transaction.

Docs: https://dev.jup.ag/api-reference/swap
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import ChainId, SwapRoute, SwapStep, Token, TokenAmount

_JUPITER_API_BASE = "https://lite-api.jup.ag/swap/v1"
_JUPITER_SWAP_V2_BASE = "https://api.jup.ag/swap/v2"


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
        # priceImpactPct is a percentage (e.g. "0.03" = 0.03%); convert to fraction.
        # Clamp to 0 — the API can return tiny negatives at favorable market conditions.
        price_impact = max(Decimal(0), Decimal(str(data.get("priceImpactPct", "0"))) / Decimal(100))

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


class JupiterSwapV2(BaseAggregator):
    """Jupiter unified Swap V2 API client.

    Integrates the ``https://api.jup.ag/swap/v2`` API.  Requires an API key
    obtainable from ``portal.jup.ag`` (passed via the ``x-api-key`` header).

    Two flows are supported:

    **Order flow** (managed execution by Jupiter):

    1. Call :meth:`get_order` with a *taker* address to receive a pre-built
       transaction plus a ``requestId``.
    2. Deserialise, sign, and re-serialise the base-64 transaction.
    3. Call :meth:`execute_order` with the signed transaction and ``requestId``.
       Jupiter broadcasts via *Jupiter Beam* and polls for confirmation.

    **Build flow** (custom transaction composition):

    Call :meth:`get_build` to receive the raw swap instructions.  Compose
    these into your own versioned Solana transaction and broadcast via your
    own RPC connection (required for CPI, custom instruction ordering, etc.).

    Args:
        api_key: Jupiter API key from ``portal.jup.ag`` (sent as ``x-api-key``).
        base_url: Override the default Swap V2 API base URL.
    """

    _DEFAULT_BASE_URL = _JUPITER_SWAP_V2_BASE

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
            headers["x-api-key"] = self.api_key
        return headers

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"Jupiter Swap V2 API error: {data.get('error', data)}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise AggregatorError(
                        f"Jupiter Swap V2 API error: {data.get('error', data)}",
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
        """Fetch a quote via ``GET /order`` (without *taker* – no transaction built).

        Calls :meth:`get_order` without a ``taker`` address.  Jupiter returns a
        quote-only response (no ``transaction`` or ``requestId`` fields).

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
        order = await self.get_order(amount_in, token_out, taker=None, slippage_bps=slippage_bps, **kwargs)

        out_amount = int(order["outAmount"])
        min_out_amount = int(order.get("otherAmountThreshold", out_amount))
        # priceImpactPct is a percentage (e.g. "0.03" = 0.03%); convert to fraction.
        # Clamp to 0 — the API can return tiny negatives at favorable market conditions.
        price_impact = max(Decimal(0), Decimal(str(order.get("priceImpactPct", "0"))) / Decimal(100))

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=out_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_out_amount),
            gas_estimate=0,  # Solana uses compute units, not EVM gas
            price_impact=price_impact,
            protocol=self.protocol_name,
            route_summary=str(order.get("routePlan", "")),
        )

    async def get_order(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        taker: str | None = None,
        slippage_bps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a swap order via ``GET /order``.

        When *taker* is provided, the response includes:

        * ``transaction`` — base-64 encoded versioned Solana transaction ready
          for signing.
        * ``requestId`` — opaque identifier required by :meth:`execute_order`.

        When *taker* is ``None``, Jupiter returns a quote-only response (no
        ``transaction`` or ``requestId`` fields).

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (Solana mint address).
            taker: Signer's Solana wallet address (base-58).  Required to
                receive a transaction; omit for a quote-only response.
            slippage_bps: Maximum acceptable slippage in basis points.  When
                ``None`` Jupiter applies its Real-Time Slippage Estimator
                (RTSE) automatically.
            **kwargs: Extra query parameters (e.g. ``referralAccount``,
                ``referralFee``).

        Returns:
            Raw order response dict.  When *taker* is provided, includes
            ``transaction`` (base-64) and ``requestId`` fields.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            **kwargs,
        }
        if taker is not None:
            params["taker"] = taker
        if slippage_bps is not None:
            params["slippageBps"] = slippage_bps
        return await self._get("order", params)

    async def execute_order(
        self,
        signed_transaction: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Execute a signed swap order via ``POST /execute``.

        Submits the signed transaction to Jupiter's proprietary *Jupiter Beam*
        execution engine.  Jupiter handles broadcasting, retry logic, priority
        fee management, and transaction confirmation polling.

        Args:
            signed_transaction: Base-64 encoded signed versioned Solana
                transaction (deserialised, signed, and re-serialised from the
                ``transaction`` field of a :meth:`get_order` response).
            request_id: The ``requestId`` returned by :meth:`get_order`.

        Returns:
            Execution result dict.  Key fields:

            * ``status`` — ``"Success"`` or ``"Failed"``.
            * ``signature`` — On-chain transaction signature (when successful).

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        return await self._post(
            "execute",
            {"signedTransaction": signed_transaction, "requestId": request_id},
        )

    async def get_build(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        taker: str,
        slippage_bps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch raw swap instructions via ``GET /build``.

        Returns the raw swap instructions needed to compose a custom versioned
        Solana transaction.  Use this flow when you need full control over
        transaction composition, such as adding your own instructions, using
        CPI, or broadcasting via your own RPC connection.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (Solana mint address).
            taker: Signer's Solana wallet address (base-58).  Required by the
                Jupiter Swap V2 ``/build`` endpoint.
            slippage_bps: Maximum acceptable slippage in basis points.  When
                ``None`` Jupiter applies its RTSE automatically.
            **kwargs: Extra query parameters forwarded to the API.

        Returns:
            Raw build response dict containing swap instructions and route
            metadata for constructing a custom transaction.

        Raises:
            :class:`~pydefi.exceptions.AggregatorError`: On API errors.
        """
        params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "taker": taker,
            **kwargs,
        }
        if slippage_bps is not None:
            params["slippageBps"] = slippage_bps
        return await self._get("build", params)

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a Swap V2 quote.

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

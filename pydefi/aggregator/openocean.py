"""
OpenOcean DEX aggregator API client.

Docs: https://apis.openocean.finance/developer/apis/swap-api/api-v4
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from typing import Any

import aiohttp

from pydefi._math import slippage_to_percent
from pydefi._utils import encode_address
from pydefi.aggregator.base import AggregatorQuote
from pydefi.exceptions import AggregatorError
from pydefi.types import Address, SwapRoute, Token, TokenAmount

# Mapping from EVM chain IDs to OpenOcean chain slugs
_CHAIN_SLUGS: dict[int, str] = {
    1: "eth",
    10: "optimism",
    56: "bsc",
    100: "xdai",
    137: "polygon",
    250: "fantom",
    1116: "core",
    1285: "moonriver",
    8453: "base",
    42161: "arbitrum",
    43114: "avax",
    59144: "linea",
    534352: "scroll",
    324: "zksync_era",
    81457: "blast",
}


def _parse_price_impact(value: Any) -> Decimal:
    """Parse a price impact value that may carry a trailing ``%`` sign."""
    raw = str(value).rstrip("%") if value is not None else "0"
    try:
        return Decimal(raw)
    except Exception:
        return Decimal("0")


def _unwrap(url: str, status: int, body: str) -> dict[str, Any]:
    """Return the JSON envelope of a response, or raise :class:`AggregatorError`."""
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise AggregatorError(
            f"OpenOcean returned a non-JSON response from {url}: HTTP {status} {body[:200].strip()}",
            status_code=status,
        )
    if status != 200 or str(data.get("code", "")) not in ("200", ""):
        msg = data.get("error") or data.get("message", data)
        raise AggregatorError(f"OpenOcean API error: {msg}", status_code=status)
    return data


class OpenOcean:
    """OpenOcean DEX aggregator API client.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: OpenOcean API key sent as ``apikey``; ``/quote`` and ``/swap`` require one.
        base_url: Override the default API base URL.
    """

    _DEFAULT_BASE_URL = "https://open-api-pro.openocean.finance/v4"

    # The API allows ~1 request/second and a quote costs two (/gasPrice, /quote),
    # so retry on 429 and reuse the gas price for a block.
    _RATE_LIMIT_RETRIES = 3
    _RATE_LIMIT_BACKOFF = 1.1
    _GAS_PRICE_TTL = 12.0

    def __init__(
        self,
        chain_id: int,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.chain_id = chain_id
        self.api_key = api_key
        self._base_url = base_url or self._DEFAULT_BASE_URL
        self._gas_price: str | None = None
        self._gas_price_at = 0.0

    @property
    def base_url(self) -> str:
        return self._base_url

    protocol_name: str = "OpenOcean"

    @property
    def chain_slug(self) -> str:
        """Return the OpenOcean chain slug for the current chain ID."""
        return _CHAIN_SLUGS.get(self.chain_id, str(self.chain_id))

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _chain_url(self, endpoint: str) -> str:
        return f"{self._base_url}/{self.chain_slug}/{endpoint.lstrip('/')}"

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self._chain_url(endpoint)
        if self.api_key:
            params = {**params, "apikey": self.api_key}
        attempt = 0
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, params=params, headers=self._headers()) as resp:
                    status, body = resp.status, await resp.text()
                if status != 429 or attempt >= self._RATE_LIMIT_RETRIES:
                    return _unwrap(url, status, body)
                attempt += 1
                await asyncio.sleep(self._RATE_LIMIT_BACKOFF * attempt)

    async def _get_gas_price(self) -> str:
        """Return the standard legacy gas price in Wei, cached for ``_GAS_PRICE_TTL``."""
        now = time.monotonic()
        if self._gas_price is not None and now - self._gas_price_at < self._GAS_PRICE_TTL:
            return self._gas_price
        data = await self._get("gasPrice", {})
        self._gas_price = str(data["data"]["standard"]["legacyGasPrice"])
        self._gas_price_at = now
        return self._gas_price

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a quote from the OpenOcean ``/quote`` endpoint.

        OpenOcean expects *amount* in human-readable units (e.g. ``"1.5"``
        for 1.5 WETH) rather than raw wei.  A ``gasPrice`` parameter (in Wei)
        is required by the API; if not supplied via ``kwargs`` it is fetched
        automatically from the OpenOcean ``/gasPrice`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters forwarded to the API.  Pass
                ``gasPrice="<wei>"`` to override the auto-fetched gas price.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote`.
        """
        if "gasPrice" not in kwargs:
            kwargs = {**kwargs, "gasPrice": await self._get_gas_price()}
        params: dict[str, Any] = {
            "inTokenAddress": amount_in.token.encoded_address,
            "outTokenAddress": token_out.encoded_address,
            "amount": str(amount_in.human_amount),
            "slippage": str(slippage_to_percent(slippage_bps)),
            **kwargs,
        }
        data = await self._get("quote", params)
        result = data["data"]

        out_amount = int(result["outAmount"])
        gas_estimate = int(result.get("estimatedGas", 0))

        return AggregatorQuote.from_quote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out_raw=out_amount,
            slippage_bps=slippage_bps,
            gas_estimate=gas_estimate,
            price_impact=_parse_price_impact(result.get("price_impact")),
            protocol=self.protocol_name,
            route_summary=str(result.get("path", "")),
        )

    async def get_swap(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        from_address: Address,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> AggregatorQuote:
        """Fetch a fully-encoded swap transaction from the OpenOcean ``/swap`` endpoint.

        A ``gasPrice`` parameter (in Wei) is required by the API; if not
        supplied via ``kwargs`` it is fetched automatically from the
        OpenOcean ``/gasPrice`` endpoint.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            from_address: Wallet address that will execute the swap.
            slippage_bps: Maximum slippage in basis points.
            **kwargs: Extra query parameters.  Pass ``gasPrice="<wei>"`` to
                override the auto-fetched gas price.

        Returns:
            An :class:`~pydefi.aggregator.base.AggregatorQuote` with
            ``tx_data`` populated.
        """
        if "gasPrice" not in kwargs:
            kwargs = {**kwargs, "gasPrice": await self._get_gas_price()}
        params: dict[str, Any] = {
            "inTokenAddress": amount_in.token.encoded_address,
            "outTokenAddress": token_out.encoded_address,
            "amount": str(amount_in.human_amount),
            "slippage": str(slippage_to_percent(slippage_bps)),
            "account": encode_address(from_address, self.chain_id),
            **kwargs,
        }
        data = await self._get("swap", params)
        result = data["data"]

        out_amount = int(result["outAmount"])

        gas_estimate = int(result.get("estimatedGas", 0))
        tx_info = result
        tx_data = {
            "to": tx_info.get("to", ""),
            "data": tx_info.get("data", ""),
            "value": tx_info.get("value", "0"),
            "gas": str(gas_estimate),
            "gasPrice": tx_info.get("gasPrice", ""),
        }

        return AggregatorQuote.from_quote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out_raw=out_amount,
            slippage_bps=slippage_bps,
            gas_estimate=gas_estimate,
            price_impact=_parse_price_impact(result.get("price_impact")),
            tx_data=tx_data,
            protocol=self.protocol_name,
            route_summary=str(result.get("path", "")),
        )

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from an OpenOcean quote."""
        quote = await self.get_quote(amount_in, token_out, slippage_bps, **kwargs)
        return quote.to_swap_route()

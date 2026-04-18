"""
OpenOcean DEX aggregator API client.

Docs: https://apis.openocean.finance/developer/apis/swap-api/api-v4
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import aiohttp

from pydefi._utils import encode_address
from pydefi.aggregator.base import AggregatorQuote, BaseAggregator
from pydefi.exceptions import AggregatorError
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount

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


class OpenOcean(BaseAggregator):
    """OpenOcean DEX aggregator API client.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).
        api_key: Optional OpenOcean API key sent as ``apikey`` query parameter.
        base_url: Override the default API base URL.
    """

    _DEFAULT_BASE_URL = "https://open-api.openocean.finance/v4"

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
        return "OpenOcean"

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
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers()) as resp:
                data = await resp.json(content_type=None)
                code = str(data.get("code", ""))
                if resp.status != 200 or code not in ("200", ""):
                    msg = data.get("error") or data.get("message", data)
                    raise AggregatorError(
                        f"OpenOcean API error: {msg}",
                        status_code=resp.status,
                    )
                return data  # type: ignore[return-value]

    @staticmethod
    def _parse_price_impact(value: Any) -> Decimal:
        """Parse a price impact value that may carry a trailing ``%`` sign."""
        raw = str(value).rstrip("%") if value is not None else "0"
        try:
            return Decimal(raw)
        except Exception:
            return Decimal("0")

    async def _get_gas_price(self) -> str:
        """Fetch the current gas price (in Wei) from the OpenOcean ``/gasPrice`` endpoint.

        Returns:
            The standard legacy gas price as a string (in Wei).
        """
        data = await self._get("gasPrice", {})
        return str(data["data"]["standard"]["legacyGasPrice"])

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
            "slippage": str(self._slippage_to_percent(slippage_bps)),
            **kwargs,
        }
        data = await self._get("quote", params)
        result = data["data"]

        out_amount = int(result["outAmount"])
        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = out_amount * slippage_factor // 10_000
        gas_estimate = int(result.get("estimatedGas", 0))

        return AggregatorQuote(
            token_in=amount_in.token,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=out_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=self._parse_price_impact(result.get("price_impact")),
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
            "slippage": str(self._slippage_to_percent(slippage_bps)),
            "account": encode_address(from_address, self.chain_id),
            **kwargs,
        }
        data = await self._get("swap", params)
        result = data["data"]

        out_amount = int(result["outAmount"])
        slippage_factor = 10_000 - slippage_bps
        min_amount_out_raw = out_amount * slippage_factor // 10_000

        gas_estimate = int(result.get("estimatedGas", 0))
        tx_info = result
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
            amount_out=TokenAmount(token=token_out, amount=out_amount),
            min_amount_out=TokenAmount(token=token_out, amount=min_amount_out_raw),
            gas_estimate=gas_estimate,
            price_impact=self._parse_price_impact(result.get("price_impact")),
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
        """Build a :class:`~pydefi.types.SwapRoute` from an OpenOcean quote.

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

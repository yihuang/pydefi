"""
Raydium AMM integration for Solana.

Raydium is the primary AMM on Solana, providing concentrated and standard
liquidity pools.  This module queries the Raydium V3 compute API for swap
quotes without requiring a Solana RPC connection.

Docs: https://docs.raydium.io/
API:  https://api-v3.raydium.io/
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import aiohttp

from pydefi.amm.base import BaseSolanaAMM
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.types import SwapRoute, SwapStep, Token, TokenAmount

_RAYDIUM_API_BASE = "https://api-v3.raydium.io"


class Raydium(BaseSolanaAMM):
    """Raydium AMM client for Solana.

    Queries the Raydium V3 compute API to obtain swap quotes.  No Solana RPC
    connection is required; all pricing data is fetched from Raydium's public
    REST API.

    Args:
        api_url: Override the default Raydium API base URL.
    """

    _DEFAULT_API_URL = _RAYDIUM_API_BASE

    def __init__(self, api_url: str | None = None) -> None:
        super().__init__(api_url or self._DEFAULT_API_URL)

    @property
    def protocol_name(self) -> str:
        return "Raydium"

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                text = await resp.text()
                if not text:
                    raise InsufficientLiquidityError(f"Raydium API returned empty response (HTTP {resp.status})")
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise InsufficientLiquidityError(
                        f"Raydium API returned non-JSON response (HTTP {resp.status}): {text[:200]}"
                    ) from exc
                if resp.status != 200 or not data.get("success", True):
                    msg = data.get("msg") or data.get("error") or str(data)
                    raise InsufficientLiquidityError(f"Raydium API error ({resp.status}): {msg}")
                return data  # type: ignore[return-value]

    async def get_quote(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> TokenAmount:
        """Get a swap quote from the Raydium compute API.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (Solana mint address).
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra query parameters forwarded to the API
                (e.g. ``txVersion="V0"``).

        Returns:
            Expected output :class:`~pydefi.types.TokenAmount`.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: When
                Raydium cannot find a route or returns an error.
        """
        params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "slippageBps": slippage_bps,
            "txVersion": kwargs.pop("txVersion", "V0"),
            **kwargs,
        }
        data = await self._get("compute/swap-base-in", params)
        out_amount = int(data["data"]["outputAmount"])
        return TokenAmount(token=token_out, amount=out_amount)

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> SwapRoute:
        """Build a :class:`~pydefi.types.SwapRoute` from a Raydium quote.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            slippage_bps: Maximum acceptable slippage in basis points.
            **kwargs: Extra query parameters forwarded to :meth:`get_quote`.

        Returns:
            A :class:`~pydefi.types.SwapRoute` with a single
            :class:`~pydefi.types.SwapStep` describing the Raydium hop.
        """
        params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "slippageBps": slippage_bps,
            "txVersion": kwargs.pop("txVersion", "V0"),
            **kwargs,
        }
        data = await self._get("compute/swap-base-in", params)
        route_data = data["data"]

        out_amount = int(route_data["outputAmount"])
        price_impact = max(Decimal(0), Decimal(str(route_data.get("priceImpactPct", "0"))) / Decimal(100))

        # Use the first pool address from the route plan if available
        route_plan = route_data.get("routePlan") or []
        pool_address = route_plan[0].get("poolId", "") if route_plan else ""

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=pool_address,
            protocol=self.protocol_name,
            fee=0,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=out_amount),
            price_impact=price_impact,
        )

    async def build_transaction(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        wallet: str,
        slippage_bps: int = 50,
        compute_unit_price_micro_lamports: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Build a Solana transaction for the swap via the Raydium API.

        Calls ``/compute/swap-base-in`` for the quote, then POSTs to
        ``/transaction/swap-base-in`` to obtain one or more base-64 encoded
        versioned Solana transactions ready for signing and broadcasting.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (Solana mint address).
            wallet: Signer's Solana wallet address (base-58 encoded).
            slippage_bps: Maximum acceptable slippage in basis points.
            compute_unit_price_micro_lamports: Priority fee in micro-lamports
                per compute unit.  ``None`` uses Raydium's default.
            **kwargs: Extra body parameters forwarded to the transaction
                endpoint (e.g. ``wrapSol``, ``unwrapSol``).

        Returns:
            List of base-64 encoded transactions to sign and broadcast.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: On API
                errors or when no route is found.
        """
        # Step 1: get compute quote (includes the full route plan Raydium needs)
        compute_params: dict[str, Any] = {
            "inputMint": amount_in.token.address,
            "outputMint": token_out.address,
            "amount": str(amount_in.amount),
            "slippageBps": slippage_bps,
            "txVersion": "V0",
        }
        compute_response = await self._get("compute/swap-base-in", compute_params)

        # Step 2: build the transaction
        url = f"{self.api_url.rstrip('/')}/transaction/swap-base-in"
        payload: dict[str, Any] = {
            "swapResponse": compute_response["data"],
            "txVersion": "V0",
            "wallet": wallet,
            **kwargs,
        }
        if compute_unit_price_micro_lamports is not None:
            payload["computeBudgetConfig"] = {"microLamports": compute_unit_price_micro_lamports}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200 or not data.get("success", True):
                    msg = data.get("msg") or data.get("error") or str(data)
                    raise InsufficientLiquidityError(f"Raydium transaction API error ({resp.status}): {msg}")
                return [item["transaction"] for item in data.get("data", [])]

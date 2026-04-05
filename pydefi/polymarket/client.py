"""
Async HTTP client for the Polymarket prediction market APIs.

Polymarket exposes two REST APIs:

* **Gamma API** (``https://gamma-api.polymarket.com``) — public market data,
  no authentication required.  Use :meth:`PolymarketClient.get_markets`,
  :meth:`PolymarketClient.get_events`, and :meth:`PolymarketClient.get_tags`
  to retrieve market information.

* **CLOB API** (``https://clob.polymarket.com``) — Central Limit Order Book
  for trading outcome tokens.  Some endpoints are public; others require L1
  or L2 authentication.

Authentication
--------------
The CLOB uses two authentication levels:

* **L1** (EIP-712, private key) — proves wallet ownership.  Required to
  create or derive API credentials with :meth:`create_api_key` /
  :meth:`derive_api_key`.

* **L2** (HMAC-SHA256, API key + secret + passphrase) — authenticates
  trading requests.  Required for :meth:`get_orders`,
  :meth:`post_order`, :meth:`cancel_order`, etc.

Docs:
    https://docs.polymarket.com/
    https://docs.polymarket.com/api-reference/authentication
    https://docs.polymarket.com/trading
"""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from pydefi.polymarket.signing import (
    BUY,
    EOA,
    POLYGON_CHAIN_ID,
    SELL,
    build_hmac_signature,
    get_order_amounts,
    sign_clob_auth,
    sign_order,
)

# ---------------------------------------------------------------------------
# Well-known endpoints
# ---------------------------------------------------------------------------

_GAMMA_API: str = "https://gamma-api.polymarket.com"
_CLOB_API: str = "https://clob.polymarket.com"


class PolymarketClient:
    """Async client for the Polymarket Gamma and CLOB APIs.

    The client can operate in three modes depending on which credentials are
    supplied:

    * **Level 0** — No credentials.  Access to all public Gamma API endpoints
      and public CLOB endpoints (orderbooks, prices, market details).

    * **Level 1** — ``private_key`` provided.  Allows creating or deriving
      API credentials in addition to all Level 0 access.

    * **Level 2** — ``private_key`` *and* ``api_key`` / ``api_secret`` /
      ``api_passphrase`` provided.  Full trading access (place orders, cancel
      orders, query open orders, etc.).

    All methods are **async** and must be awaited.

    Args:
        private_key: Hex-encoded wallet private key.  Required for L1/L2.
        api_key: CLOB API key from :meth:`create_api_key` or
            :meth:`derive_api_key`.  Required for L2.
        api_secret: CLOB API secret (base64url-encoded).  Required for L2.
        api_passphrase: CLOB API passphrase.  Required for L2.
        funder: EVM address that holds the funds.  Defaults to the address
            derived from *private_key*.  Set this when using a Polymarket
            proxy wallet (signature type 1 or 2).
        signature_type: Wallet type — ``EOA`` (0), ``POLY_PROXY`` (1), or
            ``GNOSIS_SAFE`` (2).  Defaults to ``EOA``.
        chain_id: EVM chain ID.  137 for Polygon mainnet (default), 80002
            for Polygon Amoy testnet.
        gamma_api_url: Override the Gamma API base URL.
        clob_api_url: Override the CLOB API base URL.

    Example::

        from pydefi.polymarket import PolymarketClient

        # Level 0 — read market data
        client = PolymarketClient()
        markets = await client.get_markets(limit=5)

        # Level 2 — full trading
        client = PolymarketClient(
            private_key="0x...",
            api_key="...",
            api_secret="...",
            api_passphrase="...",
        )
        creds = await client.create_api_key()
        resp = await client.post_order(
            token_id="123456...",
            price=0.50,
            size=10.0,
            side="BUY",
            tick_size="0.01",
            neg_risk=False,
        )
    """

    def __init__(
        self,
        private_key: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        funder: str | None = None,
        signature_type: int = EOA,
        chain_id: int = POLYGON_CHAIN_ID,
        gamma_api_url: str | None = None,
        clob_api_url: str | None = None,
    ) -> None:
        self._private_key = private_key
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._signature_type = signature_type
        self._chain_id = chain_id
        self._gamma_base = (gamma_api_url or _GAMMA_API).rstrip("/")
        self._clob_base = (clob_api_url or _CLOB_API).rstrip("/")

        # Derive the wallet address from the private key if available.
        if private_key is not None:
            from eth_account import Account

            self._address: str | None = Account.from_key(private_key).address
        else:
            self._address = None

        # Allow caller to override the funder address (proxy / Safe wallets).
        if funder is not None:
            self._funder = funder
        else:
            self._funder = self._address

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Perform a GET request and return the parsed JSON response."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    async def _post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Perform a POST request and return the parsed JSON response."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    async def _delete(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Perform a DELETE request and return the parsed JSON response."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, json=json, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    def _l1_headers(self, nonce: int = 0) -> dict[str, str]:
        """Build L1 authentication headers (EIP-712 ClobAuth signature).

        Args:
            nonce: Request nonce (default 0).

        Raises:
            RuntimeError: If no private key was provided.
        """
        if self._private_key is None or self._address is None:
            raise RuntimeError("private_key is required for L1 authenticated requests")
        ts = int(time.time())
        sig = sign_clob_auth(self._private_key, chain_id=self._chain_id, timestamp=ts, nonce=nonce)
        return {
            "POLY_ADDRESS": self._address,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": str(ts),
            "POLY_NONCE": str(nonce),
        }

    def _l2_headers(
        self,
        method: str,
        request_path: str,
        body: Any = None,
    ) -> dict[str, str]:
        """Build L2 authentication headers (HMAC-SHA256).

        Args:
            method: HTTP method string (``"GET"``, ``"POST"``, ``"DELETE"``).
            request_path: API path including any query string.
            body: Optional request body for POST/DELETE requests.

        Raises:
            RuntimeError: If API credentials are not set.
        """
        if self._address is None:
            raise RuntimeError("private_key is required for L2 authenticated requests")
        if self._api_key is None or self._api_secret is None or self._api_passphrase is None:
            raise RuntimeError(
                "api_key, api_secret, and api_passphrase are required for L2 "
                "authenticated requests. Call create_api_key() or derive_api_key() first."
            )
        ts = str(int(time.time()))
        sig = build_hmac_signature(self._api_secret, ts, method, request_path, body)
        return {
            "POLY_ADDRESS": self._address,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": ts,
            "POLY_API_KEY": self._api_key,
            "POLY_PASSPHRASE": self._api_passphrase,
        }

    # -----------------------------------------------------------------------
    # Gamma API — public market data
    # -----------------------------------------------------------------------

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str | None = None,
        ascending: bool = False,
        tag_id: int | None = None,
        slug: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve a list of prediction markets from the Gamma API.

        Args:
            active: Filter to active (live, tradable) markets.
            closed: Filter to closed markets.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            order: Sort field — ``"volume_24hr"``, ``"volume"``,
                ``"liquidity"``, ``"start_date"``, ``"end_date"``.
            ascending: Sort direction.  ``False`` (default) for descending.
            tag_id: Filter by tag ID.
            slug: Filter by market slug.
            **kwargs: Additional query parameters passed directly to the API.

        Returns:
            List of market dicts.  Each dict includes ``"question"``,
            ``"conditionId"``, ``"clobTokenIds"``, and many more fields.

        Example::

            markets = await client.get_markets(limit=5)
            for m in markets:
                print(m["question"], m["clobTokenIds"])
        """
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "ascending": str(ascending).lower(),
        }
        if order is not None:
            params["order"] = order
        if tag_id is not None:
            params["tag_id"] = tag_id
        if slug is not None:
            params["slug"] = slug
        params.update(kwargs)
        return await self._get(f"{self._gamma_base}/markets", params=params)

    async def get_market(self, condition_id: str) -> dict[str, Any]:
        """Retrieve a single market by its condition ID.

        Args:
            condition_id: The market's condition ID (hex string).

        Returns:
            Market dict, or ``None`` if no market with that condition ID exists.
        """
        results = await self._get(f"{self._gamma_base}/markets", params={"condition_ids": condition_id})
        if isinstance(results, list) and results:
            return results[0]
        return results

    async def get_events(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str | None = None,
        ascending: bool = False,
        tag_id: int | None = None,
        slug: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve prediction market events from the Gamma API.

        Events group related markets together (e.g. all outcomes of a single
        election).  Each event contains an embedded list of its markets.

        Args:
            active: Filter to active events.
            closed: Filter to closed events.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            order: Sort field — ``"volume_24hr"``, ``"volume"``,
                ``"liquidity"``, ``"start_date"``, ``"end_date"``.
            ascending: Sort direction.
            tag_id: Filter by tag ID.
            slug: Filter by event slug (path segment after ``/event/`` in the
                Polymarket URL).
            **kwargs: Additional query parameters.

        Returns:
            List of event dicts.

        Example::

            # Highest-volume events today
            events = await client.get_events(
                order="volume_24hr",
                limit=10,
            )
        """
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "ascending": str(ascending).lower(),
        }
        if order is not None:
            params["order"] = order
        if tag_id is not None:
            params["tag_id"] = tag_id
        if slug is not None:
            params["slug"] = slug
        params.update(kwargs)
        return await self._get(f"{self._gamma_base}/events", params=params)

    async def get_tags(self) -> list[dict[str, Any]]:
        """Retrieve all available market tags.

        Returns:
            List of tag dicts with ``"id"``, ``"label"``, and related fields.
        """
        return await self._get(f"{self._gamma_base}/tags")

    # -----------------------------------------------------------------------
    # CLOB API — public endpoints
    # -----------------------------------------------------------------------

    async def get_server_time(self) -> int:
        """Return the current server timestamp from the CLOB API.

        Returns:
            Unix timestamp as an integer.
        """
        return await self._get(f"{self._clob_base}/time")

    async def get_clob_market(self, condition_id: str) -> dict[str, Any]:
        """Retrieve market details from the CLOB API by condition ID.

        Includes ``minimum_tick_size``, ``neg_risk``, ``rewards``, and other
        trading parameters needed to construct orders.

        Args:
            condition_id: The market's condition ID (hex string).

        Returns:
            Market dict from the CLOB API.
        """
        return await self._get(f"{self._clob_base}/markets/{condition_id}")

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Retrieve the current order book for an outcome token.

        Args:
            token_id: Conditional outcome token ID (decimal string).

        Returns:
            Dict with ``"asks"`` and ``"bids"`` lists.  Each entry has
            ``"price"`` and ``"size"`` fields.
        """
        return await self._get(f"{self._clob_base}/book", params={"token_id": token_id})

    async def get_midpoint(self, token_id: str) -> dict[str, Any]:
        """Return the current midpoint price for an outcome token.

        Args:
            token_id: Conditional outcome token ID.

        Returns:
            Dict with ``"mid"`` key.
        """
        return await self._get(f"{self._clob_base}/midpoint", params={"token_id": token_id})

    async def get_price(self, token_id: str, side: str) -> dict[str, Any]:
        """Return the best price for an outcome token on a given side.

        Args:
            token_id: Conditional outcome token ID.
            side: ``"BUY"`` or ``"SELL"``.

        Returns:
            Dict with ``"price"`` key.
        """
        return await self._get(f"{self._clob_base}/price", params={"token_id": token_id, "side": side})

    async def get_spread(self, token_id: str) -> dict[str, Any]:
        """Return the current bid-ask spread for an outcome token.

        Args:
            token_id: Conditional outcome token ID.

        Returns:
            Dict with ``"spread"`` key.
        """
        return await self._get(f"{self._clob_base}/spread", params={"token_id": token_id})

    async def get_tick_size(self, token_id: str) -> dict[str, Any]:
        """Return the minimum tick size for an outcome token.

        Args:
            token_id: Conditional outcome token ID.

        Returns:
            Dict with ``"minimum_tick_size"`` key (e.g. ``"0.01"``).
        """
        return await self._get(f"{self._clob_base}/tick-size", params={"token_id": token_id})

    async def get_neg_risk(self, token_id: str) -> dict[str, Any]:
        """Return whether an outcome token uses the neg-risk exchange.

        Args:
            token_id: Conditional outcome token ID.

        Returns:
            Dict with ``"neg_risk"`` boolean key.
        """
        return await self._get(f"{self._clob_base}/neg-risk", params={"token_id": token_id})

    async def get_last_trade_price(self, token_id: str) -> dict[str, Any]:
        """Return the price of the most recent trade for an outcome token.

        Args:
            token_id: Conditional outcome token ID.

        Returns:
            Dict with ``"price"`` key.
        """
        return await self._get(f"{self._clob_base}/last-trade-price", params={"token_id": token_id})

    # -----------------------------------------------------------------------
    # CLOB API — L1 authenticated endpoints
    # -----------------------------------------------------------------------

    async def create_api_key(self, nonce: int = 0) -> dict[str, Any]:
        """Create new CLOB API credentials for this wallet.

        Requires a private key (L1 auth).  If credentials already exist for
        the given nonce, use :meth:`derive_api_key` instead.

        Args:
            nonce: Nonce used when creating the key (default 0).

        Returns:
            Dict with ``"apiKey"``, ``"secret"``, and ``"passphrase"`` keys.

        Raises:
            RuntimeError: If no private key was configured.
        """
        headers = self._l1_headers(nonce=nonce)
        return await self._post(f"{self._clob_base}/auth/api-key", headers=headers)

    async def derive_api_key(self, nonce: int = 0) -> dict[str, Any]:
        """Derive existing CLOB API credentials for this wallet.

        Requires a private key (L1 auth).  Returns the same credentials that
        were created for the given nonce, allowing recovery without re-creation.

        Args:
            nonce: Nonce used when the key was originally created (default 0).

        Returns:
            Dict with ``"apiKey"``, ``"secret"``, and ``"passphrase"`` keys.

        Raises:
            RuntimeError: If no private key was configured.
        """
        headers = self._l1_headers(nonce=nonce)
        return await self._get(f"{self._clob_base}/auth/derive-api-key", headers=headers)

    # -----------------------------------------------------------------------
    # CLOB API — L2 authenticated endpoints
    # -----------------------------------------------------------------------

    async def get_orders(
        self,
        market: str | None = None,
        asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the caller's open orders.

        Requires L2 credentials.

        Args:
            market: Filter by condition ID.
            asset_id: Filter by outcome token ID.

        Returns:
            List of open order dicts.
        """
        params: dict[str, str] = {}
        if market:
            params["market"] = market
        if asset_id:
            params["asset_id"] = asset_id

        path = "/data/orders"
        if params:
            path += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        headers = self._l2_headers("GET", path)
        return await self._get(f"{self._clob_base}{path}", headers=headers)

    async def post_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        tick_size: str,
        neg_risk: bool = False,
        order_type: str = "GTC",
        expiration: str = "0",
        funder: str | None = None,
    ) -> dict[str, Any]:
        """Create, sign, and submit a limit order to the CLOB.

        Requires L2 credentials and a private key.

        The method fetches the current *fee rate* from the market config,
        computes *maker_amount* / *taker_amount* from *price* and *size*,
        signs the order, and POSTs it to ``/order``.

        Args:
            token_id: Conditional outcome token ID (decimal string).
            price: Limit price as a probability (0–1).  E.g. ``0.50``.
            size: Order size in outcome tokens.  E.g. ``10.0``.
            side: ``"BUY"`` or ``"SELL"``.
            tick_size: Minimum tick size for the market (e.g. ``"0.01"``).
                Fetch from :meth:`get_tick_size` or :meth:`get_clob_market`.
            neg_risk: Whether the market uses the neg-risk exchange contract.
                Fetch from :meth:`get_neg_risk` or :meth:`get_clob_market`.
            order_type: ``"GTC"`` (Good-Till-Cancelled, default), ``"GTD"``
                (Good-Till-Date), or ``"FOK"`` (Fill-Or-Kill).
            expiration: Expiry timestamp in seconds (``"0"`` = no expiry).
            funder: Override the funder address for this order.

        Returns:
            CLOB API response dict with ``"orderID"`` and ``"status"`` keys.

        Raises:
            RuntimeError: If private key or API credentials are missing.
            ValueError: If *side* is not ``"BUY"`` or ``"SELL"``.

        Example::

            resp = await client.post_order(
                token_id="7132...",
                price=0.60,
                size=5.0,
                side="BUY",
                tick_size="0.01",
                neg_risk=False,
            )
            print(resp["orderID"], resp["status"])
        """
        if self._private_key is None or self._funder is None:
            raise RuntimeError("private_key is required for placing orders")

        side_int: int
        if side.upper() == "BUY":
            side_int = BUY
        elif side.upper() == "SELL":
            side_int = SELL
        else:
            raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")

        maker = funder or self._funder

        maker_amount, taker_amount = get_order_amounts(side_int, size, price, tick_size)

        order = sign_order(
            private_key=self._private_key,
            maker=maker,
            token_id=token_id,
            maker_amount=str(maker_amount),
            taker_amount=str(taker_amount),
            side=side_int,
            expiration=expiration,
            signature_type=self._signature_type,
            chain_id=self._chain_id,
            neg_risk=neg_risk,
        )

        body = {
            "order": order,
            "owner": maker,
            "orderType": order_type,
        }

        path = "/order"
        headers = self._l2_headers("POST", path, body)
        return await self._post(f"{self._clob_base}{path}", json=body, headers=headers)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order by its order ID.

        Requires L2 credentials.

        Args:
            order_id: The order ID returned by :meth:`post_order`.

        Returns:
            CLOB API response dict.
        """
        body = {"orderID": order_id}
        path = "/order"
        headers = self._l2_headers("DELETE", path, body)
        return await self._delete(f"{self._clob_base}{path}", json=body, headers=headers)

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open orders for the current wallet.

        Requires L2 credentials.

        Returns:
            CLOB API response dict.
        """
        path = "/cancel-all"
        headers = self._l2_headers("DELETE", path)
        return await self._delete(f"{self._clob_base}{path}", headers=headers)

    async def get_trades(
        self,
        market: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve the caller's trade history.

        Requires L2 credentials.

        Args:
            market: Filter by condition ID.
            asset_id: Filter by outcome token ID.
            limit: Maximum number of results.

        Returns:
            List of trade dicts.
        """
        params: dict[str, Any] = {"limit": limit}
        if market:
            params["market"] = market
        if asset_id:
            params["asset_id"] = asset_id

        path = "/data/trades"
        if params:
            path += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        headers = self._l2_headers("GET", path)
        return await self._get(f"{self._clob_base}{path}", headers=headers)

"""Polymarket prediction market integrations.

This module provides an async Python client for the Polymarket prediction
market platform — the world's largest decentralized prediction market.

* :class:`~pydefi.polymarket.client.PolymarketClient` — async HTTP client for
  the Polymarket Gamma API (market data) and CLOB API (trading).

* Signing utilities in :mod:`~pydefi.polymarket.signing` — EIP-712 helpers
  for CLOB L1 authentication and order signing; HMAC-SHA256 for L2 auth.

Quick-start — reading market data (no credentials needed)::

    from pydefi.polymarket import PolymarketClient

    client = PolymarketClient()
    markets = await client.get_markets(limit=5)

    for m in markets:
        print(m["question"])
        # ["Yes token ID", "No token ID"]
        print(m["clobTokenIds"])

Full trading example::

    import os
    from pydefi.polymarket import PolymarketClient

    # Initialise with credentials
    client = PolymarketClient(
        private_key=os.getenv("PRIVATE_KEY"),
        api_key=os.getenv("POLY_API_KEY"),
        api_secret=os.getenv("POLY_API_SECRET"),
        api_passphrase=os.getenv("POLY_API_PASSPHRASE"),
    )

    # Fetch market details — needed to get tick_size and neg_risk
    market = await client.get_clob_market("YOUR_CONDITION_ID")
    tick_size = market["minimum_tick_size"]  # e.g. "0.01"
    neg_risk = market["neg_risk"]            # e.g. False

    # Place a limit order for a Yes outcome token
    resp = await client.post_order(
        token_id="YOUR_TOKEN_ID",   # from market["clobTokenIds"][0]
        price=0.50,
        size=10.0,
        side="BUY",
        tick_size=tick_size,
        neg_risk=neg_risk,
    )
    print(resp["orderID"], resp["status"])

Polymarket: https://docs.polymarket.com/
"""

from pydefi.polymarket.client import PolymarketClient
from pydefi.polymarket.signing import (
    BUY,
    CLOB_AUTH_TYPES,
    EOA,
    EXCHANGE_ADDRESSES,
    GNOSIS_SAFE,
    ORDER_STRUCTURE,
    POLY_PROXY,
    POLYGON_AMOY_CHAIN_ID,
    POLYGON_CHAIN_ID,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    SELL,
    build_hmac_signature,
    get_order_amounts,
    sign_clob_auth,
    sign_order,
    to_token_decimals,
)

__all__ = [
    # Client
    "PolymarketClient",
    # Signing helpers
    "sign_clob_auth",
    "sign_order",
    "build_hmac_signature",
    "get_order_amounts",
    "to_token_decimals",
    # Constants
    "BUY",
    "SELL",
    "EOA",
    "POLY_PROXY",
    "GNOSIS_SAFE",
    "POLYGON_CHAIN_ID",
    "POLYGON_AMOY_CHAIN_ID",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "EXCHANGE_ADDRESSES",
    # EIP-712 type definitions
    "CLOB_AUTH_TYPES",
    "ORDER_STRUCTURE",
]

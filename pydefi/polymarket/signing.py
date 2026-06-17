"""
EIP-712 and HMAC signing utilities for the Polymarket CLOB API.

Two authentication levels are used by the Polymarket CLOB:

1. **L1 authentication** (EIP-712, private key)
   Used to create or derive API credentials.  A ``ClobAuth`` struct is signed
   with the wallet's private key to prove key ownership.

2. **L2 authentication** (HMAC-SHA256, API credentials)
   Used for all trading operations (place/cancel orders, query open orders,
   etc.).  Each request is signed with the API secret via HMAC-SHA256.

Order signing also uses EIP-712 to produce a ``SignedOrder`` struct that is
submitted to the CLOB for matching and onchain settlement.

Refs:
    https://docs.polymarket.com/trading
    https://github.com/Polymarket/clob-client/blob/main/src/signing/eip712.ts
    https://github.com/Polymarket/clob-client/blob/main/src/order-utils/exchange.order.const.ts
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import time
from typing import Any

from eth_account import Account
from eth_account.messages import encode_typed_data

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: EIP-712 domain name for the CTF Exchange contract.
PROTOCOL_NAME: str = "Polymarket CTF Exchange"
#: EIP-712 domain version for the CTF Exchange contract.
PROTOCOL_VERSION: str = "1"

#: Message signed to prove ownership of a wallet.
CLOB_AUTH_MSG: str = "This message attests that I control the given wallet"
#: EIP-712 domain name used for CLOB authentication.
CLOB_AUTH_DOMAIN_NAME: str = "ClobAuthDomain"
#: EIP-712 domain version used for CLOB authentication.
CLOB_AUTH_VERSION: str = "1"

#: Polygon mainnet chain ID.
POLYGON_CHAIN_ID: int = 137
#: Polygon Amoy testnet chain ID.
POLYGON_AMOY_CHAIN_ID: int = 80002

# ---------------------------------------------------------------------------
# Exchange contract addresses  (source: py-clob-client config.py)
# ---------------------------------------------------------------------------

#: CTF Exchange addresses keyed by (chain_id, neg_risk).
EXCHANGE_ADDRESSES: dict[tuple[int, bool], str] = {
    (POLYGON_CHAIN_ID, False): "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    (POLYGON_CHAIN_ID, True): "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    (POLYGON_AMOY_CHAIN_ID, False): "0xdFE02Eb6733538f8Ea35D585af8DE5958AD99E40",
    (POLYGON_AMOY_CHAIN_ID, True): "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

# ---------------------------------------------------------------------------
# EIP-712 type definitions
# ---------------------------------------------------------------------------

#: EIP-712 ``ClobAuth`` type used for L1 authentication.
CLOB_AUTH_TYPES: list[dict[str, str]] = [
    {"name": "address", "type": "address"},
    {"name": "timestamp", "type": "string"},
    {"name": "nonce", "type": "uint256"},
    {"name": "message", "type": "string"},
]

#: EIP-712 ``Order`` type used for order signing.
ORDER_STRUCTURE: list[dict[str, str]] = [
    {"name": "salt", "type": "uint256"},
    {"name": "maker", "type": "address"},
    {"name": "signer", "type": "address"},
    {"name": "taker", "type": "address"},
    {"name": "tokenId", "type": "uint256"},
    {"name": "makerAmount", "type": "uint256"},
    {"name": "takerAmount", "type": "uint256"},
    {"name": "expiration", "type": "uint256"},
    {"name": "nonce", "type": "uint256"},
    {"name": "feeRateBps", "type": "uint256"},
    {"name": "side", "type": "uint8"},
    {"name": "signatureType", "type": "uint8"},
]

# ---------------------------------------------------------------------------
# Side / SignatureType constants (mirrors py-order-utils)
# ---------------------------------------------------------------------------

#: Order side: buy.
BUY: int = 0
#: Order side: sell.
SELL: int = 1

#: Signature type: EOA (externally owned account).
EOA: int = 0
#: Signature type: Polymarket proxy wallet.
POLY_PROXY: int = 1
#: Signature type: Gnosis Safe smart contract wallet.
GNOSIS_SAFE: int = 2


# ---------------------------------------------------------------------------
# L1 auth: EIP-712 ClobAuth signing
# ---------------------------------------------------------------------------


def sign_clob_auth(
    private_key: str,
    chain_id: int = POLYGON_CHAIN_ID,
    timestamp: int | None = None,
    nonce: int = 0,
) -> str:
    """Sign the CLOB authentication message with an EIP-712 ``ClobAuth`` struct.

    The returned hex signature should be supplied in the ``POLY_SIGNATURE``
    header when creating or deriving API credentials.

    Args:
        private_key: Hex-encoded private key of the wallet.
        chain_id: EVM chain ID (137 for Polygon mainnet).
        timestamp: Unix timestamp in seconds.  Defaults to current time.
        nonce: Request nonce (default 0).

    Returns:
        Hex-encoded EIP-712 signature string (``0x...``).
    """
    if timestamp is None:
        timestamp = int(time.time())

    account = Account.from_key(private_key)
    address = account.address

    domain: dict[str, Any] = {
        "name": CLOB_AUTH_DOMAIN_NAME,
        "version": CLOB_AUTH_VERSION,
        "chainId": chain_id,
    }
    # message_types must NOT include EIP712Domain when using the 3-arg form
    message_types: dict[str, Any] = {"ClobAuth": CLOB_AUTH_TYPES}
    message: dict[str, Any] = {
        "address": address,
        "timestamp": str(timestamp),
        "nonce": nonce,
        "message": CLOB_AUTH_MSG,
    }

    payload = encode_typed_data(domain, message_types, message)
    signed = account.sign_message(payload)
    sig_hex = signed.signature.hex()
    return sig_hex if sig_hex.startswith("0x") else "0x" + sig_hex


# ---------------------------------------------------------------------------
# L2 auth: HMAC-SHA256 request signing
# ---------------------------------------------------------------------------


def build_hmac_signature(
    secret: str,
    timestamp: str | int,
    method: str,
    request_path: str,
    body: Any = None,
) -> str:
    """Build an HMAC-SHA256 signature for a CLOB API L2 request.

    Args:
        secret: Base64url-encoded API secret from the API credentials.
        timestamp: Unix timestamp string or int.
        method: HTTP method (``"GET"``, ``"POST"``, ``"DELETE"``).
        request_path: API path including query string, e.g. ``"/orders"``.
        body: Optional request body dict or string.  Dict values are
            JSON-serialised with double quotes (required by the CLOB API).

    Returns:
        Base64url-encoded HMAC-SHA256 signature string.
    """
    base64_secret = base64.urlsafe_b64decode(secret)
    message = str(timestamp) + str(method) + str(request_path)
    if body:
        message += body if isinstance(body, str) else json.dumps(body)

    h = hmac.new(base64_secret, bytes(message, "utf-8"), hashlib.sha256)
    return base64.urlsafe_b64encode(h.digest()).decode("utf-8")


# ---------------------------------------------------------------------------
# Order signing
# ---------------------------------------------------------------------------


def _generate_salt() -> int:
    """Return a cryptographically random 32-bit salt for order uniqueness."""
    return int.from_bytes(os.urandom(4), "big")


def sign_order(
    private_key: str,
    maker: str,
    token_id: str,
    maker_amount: str,
    taker_amount: str,
    side: int,
    *,
    taker: str = "0x0000000000000000000000000000000000000000",
    signer: str | None = None,
    expiration: str = "0",
    nonce: str = "0",
    fee_rate_bps: str = "0",
    signature_type: int = EOA,
    chain_id: int = POLYGON_CHAIN_ID,
    neg_risk: bool = False,
    salt: int | None = None,
) -> dict[str, Any]:
    """Create and sign a Polymarket CLOB order.

    Produces a :class:`dict` whose shape matches the ``SignedOrder`` expected by
    ``POST /order`` on the CLOB API.

    Args:
        private_key: Hex-encoded private key of the signing wallet.
        maker: EVM address of the order maker (holds the collateral / tokens).
        token_id: Conditional token ID (outcome token) as a decimal string.
        maker_amount: Amount given by the maker in base units (6 decimals USDC).
        taker_amount: Amount received by the maker in base units.
        side: ``BUY`` (0) or ``SELL`` (1).
        taker: EVM address of the taker (default zero address — open order).
        signer: EVM address of the signing key.  Defaults to *maker*.
        expiration: Expiry timestamp in seconds as a decimal string.
            ``"0"`` means no expiry (GTC).
        nonce: On-chain nonce for replay protection (decimal string).
        fee_rate_bps: Fee rate in basis points (decimal string).
        signature_type: ``EOA`` (0), ``POLY_PROXY`` (1), or ``GNOSIS_SAFE`` (2).
        chain_id: EVM chain ID (137 for Polygon mainnet).
        neg_risk: Whether to use the neg-risk exchange contract.
        salt: Custom salt for deterministic testing.  Defaults to a random value.

    Returns:
        Signed order dict ready to be POST-ed to ``/order``.
    """
    account = Account.from_key(private_key)
    if signer is None:
        signer = maker

    if salt is None:
        salt = _generate_salt()

    exchange_key = (chain_id, neg_risk)
    exchange_address = EXCHANGE_ADDRESSES.get(exchange_key)
    if exchange_address is None:
        supported_keys = ", ".join(str(k) for k in sorted(EXCHANGE_ADDRESSES.keys()))
        raise ValueError(
            f"Unsupported Polymarket exchange configuration {exchange_key}. "
            f"Supported (chain_id, neg_risk) values: {supported_keys}"
        )

    domain: dict[str, Any] = {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "chainId": chain_id,
        "verifyingContract": exchange_address,
    }
    # message_types must NOT include EIP712Domain when using the 3-arg form
    message_types: dict[str, Any] = {"Order": ORDER_STRUCTURE}
    message: dict[str, Any] = {
        "salt": salt,
        "maker": maker,
        "signer": signer,
        "taker": taker,
        "tokenId": int(token_id),
        "makerAmount": int(maker_amount),
        "takerAmount": int(taker_amount),
        "expiration": int(expiration),
        "nonce": int(nonce),
        "feeRateBps": int(fee_rate_bps),
        "side": side,
        "signatureType": signature_type,
    }

    payload = encode_typed_data(domain, message_types, message)
    signed = account.sign_message(payload)
    sig_hex: str = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex

    return {
        "salt": str(salt),
        "maker": maker,
        "signer": signer,
        "taker": taker,
        "tokenId": token_id,
        "makerAmount": maker_amount,
        "takerAmount": taker_amount,
        "expiration": expiration,
        "nonce": nonce,
        "feeRateBps": fee_rate_bps,
        "side": side,
        "signatureType": signature_type,
        "signature": sig_hex,
    }


# ---------------------------------------------------------------------------
# Amount helpers  (mirrors py-clob-client order_builder/helpers.py)
# ---------------------------------------------------------------------------

#: USDC uses 6 decimal places on Polygon.
USDC_DECIMALS: int = 6
_USDC_SCALE: int = 10**USDC_DECIMALS


def to_token_decimals(amount: float) -> int:
    """Convert a human-readable USDC amount to integer base units (6 decimals)."""
    return int(round(amount * _USDC_SCALE))


def _round_down(value: float, decimals: int) -> float:
    """Round *value* down to *decimals* decimal places."""
    factor = 10**decimals
    return int(value * factor) / factor


def _round_normal(value: float, decimals: int) -> float:
    """Round *value* to *decimals* decimal places (standard rounding)."""
    return round(value, decimals)


def _round_up(value: float, decimals: int) -> float:
    """Round *value* up to *decimals* decimal places."""
    factor = 10**decimals
    return math.ceil(value * factor) / factor


def _decimal_places(value: float) -> int:
    """Return the number of decimal places in *value*."""
    s = f"{value:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


# Tick-size → (price_decimals, size_decimals, amount_decimals)
_TICK_SIZE_CONFIG: dict[str, tuple[int, int, int]] = {
    "0.1": (1, 2, 3),
    "0.01": (2, 2, 4),
    "0.001": (3, 2, 5),
    "0.0001": (4, 2, 6),
}


def get_order_amounts(
    side: int,
    size: float,
    price: float,
    tick_size: str,
) -> tuple[int, int]:
    """Return ``(maker_amount, taker_amount)`` in USDC base units.

    Mirrors the logic in the official ``py-clob-client`` order builder so that
    amounts match what the CLOB expects.

    Args:
        side: ``BUY`` (0) or ``SELL`` (1).
        size: Order size in outcome tokens (human-readable).
        price: Limit price (probability, 0–1).
        tick_size: Minimum tick size string (e.g. ``"0.01"``).

    Returns:
        ``(maker_amount, taker_amount)`` as integers in USDC base units.
    """
    if tick_size not in _TICK_SIZE_CONFIG:
        raise ValueError(f"Unsupported tick_size: {tick_size!r}")

    price_dec, size_dec, amount_dec = _TICK_SIZE_CONFIG[tick_size]
    raw_price = _round_normal(price, price_dec)

    if side == BUY:
        raw_taker = _round_down(size, size_dec)
        raw_maker = raw_taker * raw_price
        if _decimal_places(raw_maker) > amount_dec:
            raw_maker = _round_up(raw_maker, amount_dec + 4)
            if _decimal_places(raw_maker) > amount_dec:
                raw_maker = _round_down(raw_maker, amount_dec)
        return to_token_decimals(raw_maker), to_token_decimals(raw_taker)

    elif side == SELL:
        raw_maker = _round_down(size, size_dec)
        raw_taker = raw_maker * raw_price
        if _decimal_places(raw_taker) > amount_dec:
            raw_taker = _round_up(raw_taker, amount_dec + 4)
            if _decimal_places(raw_taker) > amount_dec:
                raw_taker = _round_down(raw_taker, amount_dec)
        return to_token_decimals(raw_maker), to_token_decimals(raw_taker)

    else:
        raise ValueError(f"side must be BUY (0) or SELL (1), got {side!r}")

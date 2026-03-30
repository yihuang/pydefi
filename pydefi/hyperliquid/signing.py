"""
EIP-712 signing utilities for Hyperliquid L1 actions.

Hyperliquid uses two distinct signing schemes:

1. **Phantom-agent signing** (for trading actions: ``order``, ``cancel``, etc.)
   The action dict is serialised with MessagePack, hashed together with the
   nonce, and the resulting hash is embedded in an EIP-712 ``Agent`` typed-data
   payload before signing.  This ensures the signed bytes are human-readable in
   wallet UIs.

2. **User-signed action signing** (for transfer/withdrawal actions: ``usdSend``,
   ``spotSend``, ``withdraw3``, etc.)
   The action dict is signed directly as EIP-712 typed-data with an
   action-specific type definition (e.g. ``HyperliquidTransaction:UsdSend``).
   The ``signatureChainId`` and ``hyperliquidChain`` fields are added to the
   action automatically before hashing.

The ``sign_inner()`` helper signs any fully-formed EIP-712 payload dict using an
``eth_account.Account`` object (or any object with a ``sign_message`` method).

Refs:
    https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
    https://github.com/hyperliquid-dex/hyperliquid-python-sdk (signing.py)
"""

from __future__ import annotations

from typing import Any

import msgpack
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_hex

# ---------------------------------------------------------------------------
# EIP-712 type definitions for user-signed actions
# ---------------------------------------------------------------------------

USD_SEND_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "destination", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "time", "type": "uint64"},
]

SPOT_TRANSFER_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "destination", "type": "string"},
    {"name": "token", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "time", "type": "uint64"},
]

WITHDRAW_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "destination", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "time", "type": "uint64"},
]

USD_CLASS_TRANSFER_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "toPerp", "type": "bool"},
    {"name": "nonce", "type": "uint64"},
]

SEND_ASSET_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "destination", "type": "string"},
    {"name": "sourceDex", "type": "string"},
    {"name": "destinationDex", "type": "string"},
    {"name": "token", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "fromSubAccount", "type": "string"},
    {"name": "nonce", "type": "uint64"},
]

APPROVE_AGENT_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "agentAddress", "type": "address"},
    {"name": "agentName", "type": "string"},
    {"name": "nonce", "type": "uint64"},
]

APPROVE_BUILDER_FEE_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "maxFeeRate", "type": "string"},
    {"name": "builder", "type": "address"},
    {"name": "nonce", "type": "uint64"},
]

# EIP-712 type for sendToEvmWithData — withdraw USDC from HyperCore to an EVM chain via CCTP.
# Note: signatureChainId is the destination EVM chain ID (NOT the fixed Arbitrum Sepolia).
# Docs: https://developers.circle.com/cctp/howtos/withdraw-usdc-from-hypercore-to-evm
SEND_TO_EVM_WITH_DATA_SIGN_TYPES: list[dict[str, str]] = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "token", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "sourceDex", "type": "string"},
    {"name": "destinationRecipient", "type": "string"},
    {"name": "addressEncoding", "type": "string"},
    {"name": "destinationChainId", "type": "uint32"},
    {"name": "gasLimit", "type": "uint64"},
    {"name": "data", "type": "bytes"},
    {"name": "nonce", "type": "uint64"},
]

# Signature chain ID used by standard Hyperliquid L1 EIP-712 actions
# (excluding sendToEvmWithData, which uses the destination EVM chain ID).
# Hyperliquid uses Arbitrum Sepolia (chainId 421614 = 0x66eee) as the
# canonical chain ID for signing these actions.
_SIGNATURE_CHAIN_ID: str = "0x66eee"  # 421614 — Arbitrum Sepolia

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EIP712_DOMAIN_TYPES: list[dict[str, str]] = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]


def _hyperliquid_chain_name(is_mainnet: bool) -> str:
    """Return the ``hyperliquidChain`` value for EIP-712 signing."""
    return "Mainnet" if is_mainnet else "Testnet"


def action_hash(
    action: Any,
    vault_address: str | None,
    nonce: int,
    expires_after: int | None = None,
) -> bytes:
    """Compute the keccak256 hash used for phantom-agent signing.

    The hash covers: ``msgpack(action) + nonce (8 bytes big-endian)`` plus an
    optional ``vault_address`` byte and an optional ``expires_after`` field.

    Args:
        action: The L1 action dict (or list for multi-sig).
        vault_address: Vault / sub-account address, or ``None`` for the master
            account.
        nonce: Millisecond timestamp nonce.
        expires_after: Optional expiry timestamp in milliseconds.

    Returns:
        32-byte keccak256 hash.
    """
    data = msgpack.packb(action)
    data += nonce.to_bytes(8, "big")
    if vault_address is None:
        data += b"\x00"
    else:
        data += b"\x01"
        addr = vault_address[2:] if vault_address.startswith("0x") else vault_address
        data += bytes.fromhex(addr)
    if expires_after is not None:
        data += b"\x00"
        data += expires_after.to_bytes(8, "big")
    return keccak(data)


def _l1_payload(phantom_agent: dict[str, Any]) -> dict[str, Any]:
    """Build the EIP-712 payload dict for phantom-agent signing."""
    return {
        "domain": {
            "chainId": 1337,
            "name": "Exchange",
            "verifyingContract": "0x0000000000000000000000000000000000000000",
            "version": "1",
        },
        "types": {
            "Agent": [
                {"name": "source", "type": "string"},
                {"name": "connectionId", "type": "bytes32"},
            ],
            "EIP712Domain": _EIP712_DOMAIN_TYPES,
        },
        "primaryType": "Agent",
        "message": phantom_agent,
    }


def _user_signed_payload(
    primary_type: str,
    payload_types: list[dict[str, str]],
    action: dict[str, Any],
) -> dict[str, Any]:
    """Build the EIP-712 payload dict for user-signed action signing."""
    chain_id = int(action["signatureChainId"], 16)
    return {
        "domain": {
            "name": "HyperliquidSignTransaction",
            "version": "1",
            "chainId": chain_id,
            "verifyingContract": "0x0000000000000000000000000000000000000000",
        },
        "types": {
            primary_type: payload_types,
            "EIP712Domain": _EIP712_DOMAIN_TYPES,
        },
        "primaryType": primary_type,
        "message": action,
    }


def sign_inner(wallet: Account, data: dict[str, Any]) -> dict[str, str | int]:
    """Sign an EIP-712 ``data`` payload and return ``{r, s, v}``.

    Args:
        wallet: An :class:`~eth_account.Account` instance (or any object with
            a ``sign_message`` method compatible with ``eth_account``).
        data: A fully-formed EIP-712 payload dict with ``domain``, ``types``,
            ``primaryType``, and ``message`` keys.

    Returns:
        ``{"r": "0x...", "s": "0x...", "v": 27|28}``
    """
    structured_data = encode_typed_data(full_message=data)
    signed = wallet.sign_message(structured_data)
    return {"r": to_hex(signed.r), "s": to_hex(signed.s), "v": signed.v}


# ---------------------------------------------------------------------------
# Public signing functions
# ---------------------------------------------------------------------------


def sign_l1_action(
    private_key: str,
    action: Any,
    nonce: int,
    vault_address: str | None = None,
    expires_after: int | None = None,
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a Hyperliquid L1 trading action using phantom-agent EIP-712.

    Use this for trading actions: ``order``, ``cancel``, ``modify``, etc.

    Args:
        private_key: Hex-encoded private key (with or without ``0x`` prefix).
        action: The L1 action dict (e.g. ``{"type": "order", ...}``).
        nonce: Millisecond-precision timestamp (unique per signer per action).
        vault_address: Address of the vault / sub-account, or ``None`` for the
            master account.
        expires_after: Optional expiry timestamp in milliseconds.
        is_mainnet: ``True`` for mainnet, ``False`` for testnet.

    Returns:
        ``{"r": "0x...", "s": "0x...", "v": 27|28}``
    """
    h = action_hash(action, vault_address, nonce, expires_after)
    phantom_agent = {"source": "a" if is_mainnet else "b", "connectionId": h}
    data = _l1_payload(phantom_agent)
    wallet = Account.from_key(private_key)
    return sign_inner(wallet, data)


def sign_user_signed_action(
    private_key: str,
    action: dict[str, Any],
    payload_types: list[dict[str, str]],
    primary_type: str,
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a Hyperliquid L1 user-signed action (transfers, withdrawals, etc.).

    Mutates *action* in place to add ``signatureChainId`` and
    ``hyperliquidChain`` fields before signing.

    Args:
        private_key: Hex-encoded private key.
        action: The action dict (will be mutated to add signing metadata).
        payload_types: EIP-712 type definition list for this action type.
        primary_type: EIP-712 primary type string (e.g.
            ``"HyperliquidTransaction:UsdSend"``).
        is_mainnet: ``True`` for mainnet, ``False`` for testnet.

    Returns:
        ``{"r": "0x...", "s": "0x...", "v": 27|28}``
    """
    action["signatureChainId"] = _SIGNATURE_CHAIN_ID
    action["hyperliquidChain"] = _hyperliquid_chain_name(is_mainnet)
    data = _user_signed_payload(primary_type, payload_types, action)
    wallet = Account.from_key(private_key)
    return sign_inner(wallet, data)


def sign_usd_transfer_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``usdSend`` action (Core USDC transfer between accounts)."""
    return sign_user_signed_action(
        private_key,
        action,
        USD_SEND_SIGN_TYPES,
        "HyperliquidTransaction:UsdSend",
        is_mainnet,
    )


def sign_spot_transfer_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``spotSend`` action (spot token transfer between accounts)."""
    return sign_user_signed_action(
        private_key,
        action,
        SPOT_TRANSFER_SIGN_TYPES,
        "HyperliquidTransaction:SpotSend",
        is_mainnet,
    )


def sign_withdraw_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``withdraw3`` action (bridge withdrawal to L1 address)."""
    return sign_user_signed_action(
        private_key,
        action,
        WITHDRAW_SIGN_TYPES,
        "HyperliquidTransaction:Withdraw",
        is_mainnet,
    )


def sign_usd_class_transfer_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``usdClassTransfer`` action (spot ↔ perp balance transfer)."""
    return sign_user_signed_action(
        private_key,
        action,
        USD_CLASS_TRANSFER_SIGN_TYPES,
        "HyperliquidTransaction:UsdClassTransfer",
        is_mainnet,
    )


def sign_send_asset_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``sendAsset`` action (generalised cross-account asset transfer)."""
    return sign_user_signed_action(
        private_key,
        action,
        SEND_ASSET_SIGN_TYPES,
        "HyperliquidTransaction:SendAsset",
        is_mainnet,
    )


def sign_approve_agent_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign an ``approveAgent`` action (register an API wallet)."""
    return sign_user_signed_action(
        private_key,
        action,
        APPROVE_AGENT_SIGN_TYPES,
        "HyperliquidTransaction:ApproveAgent",
        is_mainnet,
    )


def sign_approve_builder_fee_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign an ``approveBuilderFee`` action."""
    return sign_user_signed_action(
        private_key,
        action,
        APPROVE_BUILDER_FEE_SIGN_TYPES,
        "HyperliquidTransaction:ApproveBuilderFee",
        is_mainnet,
    )


def sign_send_to_evm_with_data_action(
    private_key: str,
    action: dict[str, Any],
    is_mainnet: bool = True,
) -> dict[str, str | int]:
    """Sign a ``sendToEvmWithData`` action (withdraw USDC from HyperCore to EVM).

    Unlike other user-signed actions, the ``signatureChainId`` for
    ``sendToEvmWithData`` is the **destination** EVM chain ID (e.g.
    ``"0xa4b1"`` for Arbitrum mainnet), not the fixed Arbitrum Sepolia ID.
    The caller is responsible for setting ``action["signatureChainId"]`` to
    the correct destination chain ID before calling this function.

    Args:
        private_key: Hex-encoded private key.
        action: The ``sendToEvmWithData`` action dict.  Must already contain
            ``signatureChainId`` (destination chain ID in hex, e.g.
            ``"0xa4b1"``).  The ``hyperliquidChain`` field will be set
            automatically.
        is_mainnet: ``True`` for mainnet, ``False`` for testnet.

    Returns:
        ``{"r": "0x...", "s": "0x...", "v": 27|28}``
    """
    action["hyperliquidChain"] = _hyperliquid_chain_name(is_mainnet)
    data = _user_signed_payload(
        "HyperliquidTransaction:SendToEvmWithData",
        SEND_TO_EVM_WITH_DATA_SIGN_TYPES,
        action,
    )
    wallet = Account.from_key(private_key)
    return sign_inner(wallet, data)

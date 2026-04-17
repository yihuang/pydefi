"""
Internal conversion utilities for pydefi.

Common helpers for converting between hex strings, bytes, and canonical
representations for EVM addresses and hashes/topics.

Chain-aware encoding
--------------------
Use :func:`encode_address` and :func:`decode_address` whenever an ``Address``
must be serialised to or from a human-readable string.  The encoding depends
on the chain:

* **EVM chains** — ``0x``-prefixed lowercase hex (20 bytes).
* **Solana** — base-58 encoding (32-byte public key).

Only call ``HexBytes(str)`` or ``to_checksum_address`` directly if EVM chains only,
otherwise always go through these helpers so that Solana and future non-EVM chains
are handled correctly.
"""

from __future__ import annotations

import base58
from hexbytes import HexBytes

from pydefi.types import NATIVE_ADDRESSES, ZERO_HASH, Address, ChainId, Hash


def address_to_bytes32(address: Address) -> Hash:
    """Left-pad an EVM address to a 32-byte big-endian value.

    The 20-byte address is placed in the rightmost 20 bytes.

    Args:
        address: canonical ``Address``.

    Returns:
        32 bytes with the address right-aligned (left zero-padded).
    """
    return Hash(b"\x00" * (32 - len(address)) + address)


def token_to_bytes32(address: Address) -> Hash:
    """Convert a token ``Address`` to its Wormhole/SWIFT ``bytes32`` representation.

    Native tokens (zero address or ``0xEeEe…`` sentinel) map to 32 zero bytes,
    which is the Solana system program ID in Wormhole encoding.
    ERC-20 tokens are left-padded as a normal EVM address (see
    :func:`address_to_bytes32`).

    Args:
        address: canonical ``Address``.

    Returns:
        A 32-byte :class:`~hexbytes.HexBytes` (``Hash``) value.
    """
    if address in NATIVE_ADDRESSES:
        return ZERO_HASH
    return address_to_bytes32(address)


def encode_address(address: Address, chain_id: int) -> str:
    """Encode a raw ``Address`` to its chain-specific string representation.

    * **EVM chains** — returns ``"0x"``-prefixed lowercase hex (20 bytes).
    * **Solana** (:attr:`~pydefi.types.ChainId.SOLANA`) — returns
      base-58 encoded string (32-byte public key).

    Use this at true peripheries (HTTP API payloads, log messages, JSON
    serialisation) instead of calling ``"0x" + addr.hex()`` directly.

    Args:
        address: Raw address bytes (:class:`~hexbytes.HexBytes`).
        chain_id: The chain ID (use :class:`~pydefi.types.ChainId` constants).

    Returns:
        Human-readable string representation suitable for the target chain.
    """
    if chain_id == ChainId.SOLANA:
        if isinstance(address, str):
            return address
        return base58.b58encode(bytes(address)).decode()
    return "0x" + address.hex()


def decode_address(addr_str: str, chain_id: int) -> Address:
    """Decode a chain-specific string representation to a raw ``Address``.

    * **EVM chains** — parses ``"0x"``-prefixed hex; accepts checksummed or
      lower-case strings.
    * **Solana** (:attr:`~pydefi.types.ChainId.SOLANA`) — base-58 decodes
      the public key.

    Use this at true peripheries (CLI arguments, JSON input, test fixtures)
    instead of calling ``HexBytes(str)`` directly.

    Args:
        addr_str: Human-readable address string.
        chain_id: The chain ID (use :class:`~pydefi.types.ChainId` constants).

    Returns:
        Raw address as :class:`~hexbytes.HexBytes` (``Address``).
    """
    if chain_id == ChainId.SOLANA:
        bz = base58.b58decode(addr_str)
        if len(bz) != 32:
            raise ValueError(f"Invalid Solana address: {addr_str!r}")
    else:
        bz = HexBytes(addr_str)
        if len(bz) != 20:
            raise ValueError(f"Invalid EVM address: {addr_str!r}")
    return Address(bz)

"""
Internal conversion utilities for pydefi.

Common helpers for converting between hex strings, bytes, and canonical
``HexBytes`` representations for EVM addresses and hashes/topics.
"""

from __future__ import annotations

from pydefi.types import Address, Hash

# Bytes representation of the two common native-token sentinel addresses:
#   zero address (0x000…000) and the "EeeE…" burn address.
_NATIVE_ADDRESS_SENTINELS: frozenset[bytes] = frozenset(
    {
        b"\x00" * 20,
        b"\xee" * 20,
    }
)


def address_to_bytes32(address: Address) -> Hash:
    """Left-pad an EVM address to a 32-byte big-endian value.

    The 20-byte address is placed in the rightmost 20 bytes.

    Args:
        address: An EVM address as :class:`~hexbytes.HexBytes` (``Address``).

    Returns:
        32 bytes with the address right-aligned (left zero-padded).
    """
    return Hash(address.rjust(32, b"\x00"))


def token_to_bytes32(address: Address) -> Hash:
    """Convert a token ``Address`` to its Wormhole/SWIFT ``bytes32`` representation.

    Native tokens (zero address or ``0xEeEe…`` sentinel) map to 32 zero bytes,
    which is the Solana system program ID in Wormhole encoding.
    ERC-20 tokens are left-padded as a normal EVM address (see
    :func:`address_to_bytes32`).

    Args:
        address: A 20-byte token address as :class:`~hexbytes.HexBytes`
            (``Address``).

    Returns:
        A 32-byte :class:`~hexbytes.HexBytes` (``Hash``) value.
    """
    if bytes(address) in _NATIVE_ADDRESS_SENTINELS:
        return Hash(b"\x00" * 32)
    return address_to_bytes32(address)

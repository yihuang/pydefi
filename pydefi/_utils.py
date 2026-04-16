"""
Internal conversion utilities for pydefi.

Common helpers for converting between hex strings, bytes, and canonical
``HexBytes`` representations for EVM addresses and hashes/topics.
"""

from __future__ import annotations

from hexbytes import HexBytes


def address_to_bytes32(address: str | bytes) -> bytes:
    """Left-pad an EVM address to a 32-byte big-endian value.

    Accepts a 0x-prefixed hex string, a bare hex string, or raw bytes.
    The 20-byte address is placed in the rightmost 20 bytes.

    Args:
        address: An EVM address as a hex string or bytes.

    Returns:
        32 bytes with the address right-aligned (left zero-padded).
    """
    return HexBytes(address).rjust(32, b"\x00")

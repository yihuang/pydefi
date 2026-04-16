"""
Internal conversion utilities for pydefi.

Common helpers for converting between hex strings, bytes, and canonical
``HexBytes`` representations for EVM addresses and hashes/topics.
"""

from __future__ import annotations

from pydefi.types import Address, Hash


def address_to_bytes32(address: Address) -> Hash:
    """Left-pad an EVM address to a 32-byte big-endian value.

    The 20-byte address is placed in the rightmost 20 bytes.

    Args:
        address: An EVM address as :class:`~hexbytes.HexBytes` (``Address``).

    Returns:
        32 bytes with the address right-aligned (left zero-padded).
    """
    return Hash(address.rjust(32, b"\x00"))

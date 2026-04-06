"""ABI encode/decode helpers for DeFiVM.

Provides Python-level ABI encoding and decoding compatible with the Ethereum
ABI specification and usable alongside the DeFiVM program builder.

- :func:`abi_encode` — standard ABI encoding (``abi.encode`` in Solidity).
- :func:`abi_encode_packed` — non-standard packed encoding (``abi.encodePacked``).
- :func:`abi_decode` — standard ABI decoding (``abi.decode`` in Solidity).

All three functions operate on the same type language used throughout the
Ethereum ABI spec and ``eth_abi``:

- Primitive types: ``uint<M>``, ``int<M>``, ``address``, ``bool``,
  ``bytes<M>`` (M = 1–32), ``bytes``, ``string``.
- Fixed-size arrays: ``T[N]`` — N elements of type T.
- Dynamic arrays: ``T[]`` — variable-length sequence of T elements.
- Tuples (structs): ``(T1,T2,...)`` — ordered fields.
- Nested combinations of all of the above.

Standard encoding head/tail layout
------------------------------------
``abi_encode`` follows the canonical ABI encoding:

* Each top-level value contributes exactly one 32-byte *head* slot.
* Static types (uintN, intN, address, bool, bytesN, fixed-size arrays and
  tuples of static types) are encoded in-place inside their head slot.
* Dynamic types (bytes, string, dynamic arrays, or tuples/arrays that
  contain a dynamic member) store a 32-byte byte-offset in the head and
  place the actual data in the *tail* section that follows all head slots.

This mirrors exactly what the Solidity compiler generates for
``abi.encode(...)`` calls.

Packed encoding
---------------
``abi_encode_packed`` packs each value using its *natural* size:

* ``uint<M>`` / ``int<M>`` → M/8 bytes (big-endian, no padding).
* ``address`` → 20 bytes.
* ``bool`` → 1 byte.
* ``bytes<M>`` → M bytes (left-aligned, no right-padding).
* ``bytes`` / ``string`` → raw bytes with **no** length prefix.
* ``T[N]`` / ``T[]`` → elements packed sequentially, **no** length prefix.
* ``(T1,T2,...)`` → fields packed sequentially.

Dynamic array lengths are omitted in packed mode, making the encoding
ambiguous when multiple dynamic types follow each other.  Use packed
encoding only for single dynamic types or when the layout is unambiguous.

Integration example::

    from pydefi.vm.abi import abi_encode, abi_encode_packed, abi_decode
    from pydefi.vm import Program

    # Build a data buffer to pass as callback data (abi.encode style)
    callback_data = abi_encode(['address', 'uint256'], [TOKEN_IN, amount_owed])

    # Use the buffer inside a DeFiVM program
    program = (
        Program()
        .push_bytes(callback_data)
        ...
        .build()
    )

    # Decode return data from a completed call
    (amount_out,) = abi_decode(['uint256'], returndata)
"""

from __future__ import annotations

from typing import Any

from eth_abi import decode as _decode
from eth_abi import encode as _encode
from eth_abi.packed import encode_packed as _encode_packed

__all__ = [
    "abi_encode",
    "abi_encode_packed",
    "abi_decode",
]


def abi_encode(types: list[str], values: list[Any]) -> bytes:
    """Standard ABI encode — equivalent to Solidity's ``abi.encode()``.

    Encodes *values* according to the canonical Ethereum ABI specification.
    Each value is encoded with a 32-byte head slot; dynamic types (``bytes``,
    ``string``, dynamic arrays, or types that contain dynamic members) store a
    tail-section offset in their head slot and their actual data in the tail
    that follows all head slots.

    The result can be decoded by :func:`abi_decode` using the same type list,
    or passed to a Solidity function expecting ABI-encoded data via
    :func:`~pydefi.vm.program.push_bytes`.

    Args:
        types:  List of ABI type strings, e.g.
                ``['uint256', 'address', 'bytes']``.
        values: List of Python values corresponding to each type.

    Returns:
        Canonical ABI-encoded bytes (no function selector prefix).

    Raises:
        ValueError: If ``len(types) != len(values)``.
        eth_abi.exceptions.EncodingError: If a value cannot be encoded as the
            specified type.

    Examples::

        # Static types only — output is 3 × 32 bytes = 96 bytes
        abi_encode(['uint256', 'address', 'bool'], [42, ADDR, True])

        # Dynamic type — output includes head + tail sections
        abi_encode(['bytes', 'uint256'], [b'\\xde\\xad', 1])

        # Tuple (struct)
        abi_encode(['(address,uint256)'], [(ADDR, 1000)])

        # Array
        abi_encode(['uint256[]'], [[1, 2, 3]])
    """
    if len(types) != len(values):
        raise ValueError(
            f"abi_encode: types and values must have equal length, "
            f"got {len(types)} types and {len(values)} values"
        )
    return _encode(types, values)


def abi_encode_packed(types: list[str], values: list[Any]) -> bytes:
    """Packed ABI encode — equivalent to Solidity's ``abi.encodePacked()``.

    Encodes *values* using their minimal *natural* sizes without padding:

    * ``uint<M>`` / ``int<M>`` → M/8 bytes big-endian.
    * ``address`` → 20 bytes.
    * ``bool`` → 1 byte.
    * ``bytes<M>`` → M bytes (no right-padding).
    * ``bytes`` / ``string`` → raw bytes, **no** length prefix.
    * ``T[N]`` / ``T[]`` → packed elements, **no** length prefix.
    * ``(T1,T2,...)`` → packed fields.

    .. warning::

        Packed encoding is **ambiguous** when multiple dynamic types
        (``bytes``, ``string``, or dynamic arrays) follow each other,
        because no length delimiter is emitted.  Use it only when the
        layout is unambiguous (e.g. a single dynamic type, or all static
        types, or when the sizes are known at decode time).

    Args:
        types:  List of ABI type strings.
        values: List of Python values corresponding to each type.

    Returns:
        Packed ABI-encoded bytes.

    Raises:
        ValueError: If ``len(types) != len(values)``.
        eth_abi.exceptions.EncodingError: If a value cannot be encoded as the
            specified type.

    Examples::

        # Single uint8 — 1 byte
        abi_encode_packed(['uint8'], [255])   # → b'\\xff'

        # Mixed static types — compact encoding
        abi_encode_packed(['uint8', 'address'], [1, ADDR])  # 21 bytes

        # Dynamic bytes — no length prefix
        abi_encode_packed(['bytes'], [b'\\xde\\xad'])  # → b'\\xde\\xad'

        # Keccak hash of packed values (common Solidity pattern)
        import hashlib
        data = abi_encode_packed(['address', 'uint256'], [ADDR, nonce])
        key = hashlib.new('sha3_256', data).digest()  # illustrative only
    """
    if len(types) != len(values):
        raise ValueError(
            f"abi_encode_packed: types and values must have equal length, "
            f"got {len(types)} types and {len(values)} values"
        )
    return _encode_packed(types, values)


def abi_decode(types: list[str], data: bytes | bytearray) -> tuple[Any, ...]:
    """Standard ABI decode — equivalent to Solidity's ``abi.decode()``.

    Decodes *data* that was produced by :func:`abi_encode` (or by Solidity's
    ``abi.encode``) back into a Python tuple of values.  Handles all static
    and dynamic ABI types, including nested tuples and arrays.

    Args:
        types: List of ABI type strings matching the original encoding.
        data:  Canonical ABI-encoded bytes (no selector prefix).

    Returns:
        A ``tuple`` of decoded Python values in the same order as *types*.

    Raises:
        eth_abi.exceptions.DecodingError: If *data* is too short or
            structurally invalid for the given *types*.

    Examples::

        values = abi_decode(['uint256', 'address'], encoded_bytes)
        amount, recipient = values

        # Round-trip static types
        data = abi_encode(['uint256', 'bool'], [999, True])
        (n, flag) = abi_decode(['uint256', 'bool'], data)
        assert n == 999 and flag is True

        # Round-trip dynamic type
        data = abi_encode(['bytes', 'string'], [b'\\x01\\x02', 'hi'])
        (raw, text) = abi_decode(['bytes', 'string'], data)
    """
    return _decode(types, bytes(data))

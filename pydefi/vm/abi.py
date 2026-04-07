"""ABI encoding bytecode generators for DeFiVM.

Generate EVM opcodes that, when executed inside the VM, perform ABI encoding
of values taken from the EVM stack and write the result into free memory.

**Types are known at compile time** (when the Python code generates opcodes);
**values are provided at runtime** on the EVM stack.

This module is the "in-VM abi.encode" equivalent — analogous to how the
Solidity compiler generates opcodes for the built-in ``abi.encode()`` /
``abi.encodePacked()`` functions.

Standard (canonical) encoding — :func:`emit_abi_encode`
-------------------------------------------------------
Produces the same byte layout as Solidity's ``abi.encode()``: each leaf
scalar occupies one 32-byte slot, and the slots follow the flattened
(depth-first) order of the type list.  Static tuples and fixed-size arrays
are flattened; dynamic types are not supported.

Packed encoding — :func:`emit_abi_encode_packed`
------------------------------------------------
Produces the same byte layout as Solidity's ``abi.encodePacked()``: each
value uses its *natural* size with no padding.  Values are written in
forward (left-to-right) order using overlapping 32-byte MSTOREs so that
each write's trailing zeros are overwritten by the next.

Usage with the fluent builder::

    from pydefi.vm import Program

    # ABI-encode two runtime values from the stack
    bytecode = (
        Program()
        .load_reg(0)                           # arg 0: uint256 (deepest)
        .push_addr("0x" + "ab" * 20)           # arg 1: address (TOS)
        .abi_encode(["uint256", "address"])
        # Stack: [argsOffset(TOS), argsLen(2nd)]  — ready for CALL args
        ...
        .build()
    )

    # With a 4-byte function selector prefix
    bytecode = (
        Program()
        .load_reg(0)                           # arg 0: uint256
        .load_reg(1)                           # arg 1: uint256
        .abi_encode(["uint256", "uint256"], selector=b"\\x12\\x34\\x56\\x78")
        .push_u256(0).push_addr(TARGET).gas_opcode().call().pop()
        .build()
    )

Low-level functional style::

    from pydefi.vm.abi import emit_abi_encode
    from pydefi.vm.program import push_u256, push_addr

    opcodes = (
        push_u256(42) + push_addr("0x" + "ab" * 20)
        + emit_abi_encode(["uint256", "address"])
    )
"""

from __future__ import annotations

import re
from typing import Sequence

from pydefi.vm.program import (
    _DUP2,
    _PUSH1,
    OP_ADD,
    OP_AND,
    OP_DUP,
    OP_ISZERO,
    OP_MSTORE,
    OP_POP,
    OP_PUSH_U256,
    OP_SHL,
    OP_SWAP,
    fp_init,
)

# Internal EVM opcode not exposed from program.py
_SIGNEXTEND: int = 0x0B

# ---------------------------------------------------------------------------
# ABI type parsing helpers
# ---------------------------------------------------------------------------

_UINT_RE = re.compile(r"^uint(\d+)$")
_INT_RE = re.compile(r"^int(\d+)$")
_BYTES_FIXED_RE = re.compile(r"^bytes(\d+)$")
_FIXED_ARRAY_RE = re.compile(r"^(.+)\[(\d+)\]$")


def _split_tuple_types(inner: str) -> list[str]:
    """Split a comma-separated type string, respecting nested parentheses."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in inner:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _is_dynamic(typ: str) -> bool:
    """Return ``True`` if *typ* is a dynamic ABI type."""
    typ = typ.strip()
    if typ in ("bytes", "string"):
        return True
    if typ.endswith("[]"):
        return True
    if typ.startswith("(") and typ.endswith(")"):
        return any(_is_dynamic(m) for m in _split_tuple_types(typ[1:-1]))
    m = _FIXED_ARRAY_RE.match(typ)
    if m:
        return _is_dynamic(m.group(1))
    return False


def _validate_scalar(typ: str) -> None:
    """Raise if *typ* is not a recognised scalar type."""
    if typ in ("uint256", "int256", "address", "bool"):
        return
    m = _UINT_RE.match(typ)
    if m:
        bits = int(m.group(1))
        if bits % 8 != 0 or not (8 <= bits <= 256):
            raise ValueError(f"Invalid uint width: {typ}")
        return
    m = _INT_RE.match(typ)
    if m:
        bits = int(m.group(1))
        if bits % 8 != 0 or not (8 <= bits <= 256):
            raise ValueError(f"Invalid int width: {typ}")
        return
    m = _BYTES_FIXED_RE.match(typ)
    if m:
        n = int(m.group(1))
        if not (1 <= n <= 32):
            raise ValueError(f"Invalid bytesN width: {typ}")
        return
    raise ValueError(f"Unsupported ABI scalar type: {typ!r}")


def _flatten_static_types(types: Sequence[str]) -> list[str]:
    """Flatten static types into a list of leaf scalars.

    Static tuples are expanded into their members; fixed-size arrays of
    static types are expanded into repeated copies of the base type.

    Each leaf is a scalar: ``uint<M>``, ``int<M>``, ``address``, ``bool``,
    or ``bytes<M>``.

    Raises:
        ValueError: If any type is dynamic (``bytes``, ``string``, ``T[]``).
    """
    result: list[str] = []
    for t in types:
        t = t.strip()
        if _is_dynamic(t):
            raise ValueError(
                f"Dynamic type {t!r} not supported in emit_abi_encode; "
                "use push_bytes + patch_value for dynamic calldata"
            )
        if t.startswith("(") and t.endswith(")"):
            members = _split_tuple_types(t[1:-1])
            result.extend(_flatten_static_types(members))
        else:
            m = _FIXED_ARRAY_RE.match(t)
            if m:
                base, count = m.group(1), int(m.group(2))
                result.extend(_flatten_static_types([base]) * count)
            else:
                _validate_scalar(t)
                result.append(t)
    return result


def _natural_byte_size(typ: str) -> int:
    """Return the natural byte size of a scalar type (for packed encoding)."""
    typ = typ.strip()
    if typ == "address":
        return 20
    if typ == "bool":
        return 1
    m = _UINT_RE.match(typ)
    if m:
        return int(m.group(1)) // 8
    m = _INT_RE.match(typ)
    if m:
        return int(m.group(1)) // 8
    m = _BYTES_FIXED_RE.match(typ)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine natural size for type: {typ!r}")


# ---------------------------------------------------------------------------
# Small opcode helpers
# ---------------------------------------------------------------------------


def _push_small(v: int) -> bytes:
    """Smallest PUSH opcode for a non-negative integer *v* (up to 256 bits)."""
    if v == 0:
        return bytes([_PUSH1, 0x00])
    n = (v.bit_length() + 7) // 8
    if n > 32:
        raise ValueError(f"_push_small: value {v:#x} exceeds 256 bits")
    return bytes([0x5F + n]) + v.to_bytes(n, "big")


def _emit_clean(typ: str) -> bytes:
    """Emit opcodes to clean/mask TOS for standard ABI encoding (no-op → b"")."""
    typ = typ.strip()
    if typ in ("uint256", "int256", "bytes32"):
        return b""
    if typ == "bool":
        # normalise any non-zero to 1
        return bytes([OP_ISZERO, OP_ISZERO])
    if typ == "address":
        mask = (1 << 160) - 1
        return _push_small(mask) + bytes([OP_AND])
    m = _UINT_RE.match(typ)
    if m:
        bits = int(m.group(1))
        if bits == 256:
            return b""
        mask = (1 << bits) - 1
        return _push_small(mask) + bytes([OP_AND])
    m = _INT_RE.match(typ)
    if m:
        bits = int(m.group(1))
        if bits == 256:
            return b""
        # SIGNEXTEND(b, x): b = byte_width - 1
        b = bits // 8 - 1
        return _push_small(b) + bytes([_SIGNEXTEND])
    m = _BYTES_FIXED_RE.match(typ)
    if m:
        n = int(m.group(1))
        if n == 32:
            return b""
        # zero-out trailing (32−n) bytes
        mask = ((1 << (n * 8)) - 1) << ((32 - n) * 8)
        return _push_small(mask) + bytes([OP_AND])
    return b""


# ---------------------------------------------------------------------------
# Memory preamble / epilogue — shared between encode and encode_packed
# ---------------------------------------------------------------------------


def _emit_fp_update(padded_size: int) -> bytes:
    """``mem[0x40] = TOS + padded_size``.  TOS (max_fp) is preserved."""
    return (
        bytes([OP_DUP])  # [max_fp, max_fp]
        + _push_small(padded_size)  # [padded_size, max_fp, max_fp]
        + bytes(
            [
                OP_ADD,  # [new_fp, max_fp]
                _PUSH1,
                0x40,  # [0x40, new_fp, max_fp]
                OP_MSTORE,  # mem[0x40]=new_fp; [max_fp]
            ]
        )
    )


# =========================================================================
# emit_abi_encode — standard canonical ABI encoding in EVM bytecodes
# =========================================================================


def emit_abi_encode(
    types: Sequence[str],
    *,
    selector: bytes | None = None,
) -> bytes:
    """Generate EVM opcodes that ABI-encode stack values into memory.

    Like Solidity's ``abi.encode()``, but the values come from the EVM stack
    at runtime rather than from constants.  The type list is known at
    compile time and determines the generated opcode sequence.

    **Stack convention** — values are pushed in type-list order::

        push(val_0)   # types[0] — pushed first, ends up deepest
        push(val_1)   # types[1]
        …
        push(val_{n-1})  # types[-1] — pushed last, TOS

    Static tuples and fixed-size arrays are **flattened**: each leaf scalar
    occupies one stack slot.

    **Stack after execution**::

        [argsOffset(TOS), argsLen(2nd), <rest…>]

    This is the same layout that :func:`~pydefi.vm.program.push_bytes`
    produces, so the result plugs directly into ``CALL`` preparation or
    :func:`~pydefi.vm.program.patch_value`.

    Args:
        types:    ABI type strings.  Only static scalars, static tuples,
                  and fixed-size arrays of static types are supported.
                  Dynamic types (``bytes``, ``string``, ``T[]``) raise
                  :class:`ValueError`.
        selector: Optional 4-byte function selector.  When given, the
                  memory buffer is ``selector ‖ abi.encode(values)`` and
                  ``argsLen`` includes the 4 selector bytes.

    Returns:
        Raw EVM bytecodes.

    Raises:
        ValueError: If *types* is empty (and no *selector*), or contains a
            dynamic type, or *selector* length ≠ 4.
    """
    if selector is not None and len(selector) != 4:
        raise ValueError(f"emit_abi_encode: selector must be exactly 4 bytes, got {len(selector)}")
    if not types and selector is None:
        raise ValueError("emit_abi_encode: types must not be empty (or provide a selector)")

    flat = _flatten_static_types(types) if types else []
    n = len(flat)

    prefix_len = 4 if selector is not None else 0
    total_bytes = prefix_len + n * 32
    padded = (total_bytes + 31) & ~31  # always 32-aligned for static types

    parts: list[bytes] = []

    # ── 1. Initialise free-memory pointer ──────────────────────────────
    # Stack after (TOS first): [max_fp, val_{n-1}, …, val_0, <rest>]
    parts.append(fp_init())

    # ── 2. Write function selector (if any) ────────────────────────────
    if selector is not None:
        # Left-align 4-byte selector in a 32-byte word, then MSTORE at fp.
        sel_word = int.from_bytes(selector, "big") << 224
        parts.append(bytes([OP_PUSH_U256]) + sel_word.to_bytes(32, "big"))
        parts.append(bytes([_DUP2, OP_MSTORE]))  # mem[max_fp] = sel_word

    # ── 3. Write values in reverse order (TOS first → deepest last) ────
    # Processing TOS first is natural for a stack machine and avoids deep
    # stack access.  For standard encoding every slot is 32-byte-aligned,
    # so writes never overlap (except selector bytes 4-31, which are
    # correctly overwritten by the last-processed arg_0 at offset 4).
    for i in range(n - 1, -1, -1):
        offset = prefix_len + i * 32
        clean = _emit_clean(flat[i])

        # SWAP1: bring val_i to TOS, push max_fp down
        parts.append(bytes([OP_SWAP]))
        if clean:
            parts.append(clean)

        # DUP2 max_fp; compute store address; MSTORE
        parts.append(bytes([_DUP2]))
        if offset > 0:
            parts.append(_push_small(offset))
            parts.append(bytes([OP_ADD]))
        parts.append(bytes([OP_MSTORE]))
        # Stack restored to: [max_fp, remaining_vals…]

    # ── 4. Update free-memory pointer ──────────────────────────────────
    parts.append(_emit_fp_update(padded))

    # ── 5. Leave [argsOffset(TOS), argsLen(2nd)] ──────────────────────
    parts.append(_push_small(total_bytes))  # [total_bytes, max_fp]
    parts.append(bytes([OP_SWAP]))  # [max_fp, total_bytes]

    return b"".join(parts)


# =========================================================================
# emit_abi_encode_packed — packed ABI encoding in EVM bytecodes
# =========================================================================


def emit_abi_encode_packed(types: Sequence[str]) -> bytes:
    """Generate EVM opcodes that packed-encode stack values into memory.

    Like Solidity's ``abi.encodePacked()``, but values come from the EVM
    stack.  Each scalar is encoded at its natural byte size (no padding).

    **Stack convention** — same as :func:`emit_abi_encode`::

        push(val_0)      # types[0] — deepest
        …
        push(val_{n-1})  # types[-1] — TOS

    **Stack after execution**::

        [argsOffset(TOS), argsLen(2nd)]

    Implementation writes values in **forward** (left-to-right) order using
    ``DUP`` to copy each value from the stack without consuming it.  Each
    value is left-aligned via ``SHL`` and written with ``MSTORE`` (32 bytes);
    subsequent writes overwrite the trailing zeros.  After all writes the
    original values are removed from the stack.

    Args:
        types: Static scalar ABI type strings.  Max 15 leaf types (EVM
               ``DUP`` depth limit).

    Returns:
        Raw EVM bytecodes.

    Raises:
        ValueError: Empty types, dynamic types, or > 15 leaf values.
    """
    if not types:
        raise ValueError("emit_abi_encode_packed: types must not be empty")

    flat = _flatten_static_types(types)
    n = len(flat)

    if n > 15:
        raise ValueError(f"emit_abi_encode_packed: too many leaf values ({n}); max 15 (limited by EVM DUP depth)")

    # Pre-compute cumulative byte offsets and total size.
    offsets: list[int] = []
    cum = 0
    for t in flat:
        offsets.append(cum)
        cum += _natural_byte_size(t)
    total_bytes = cum
    padded = (total_bytes + 31) & ~31

    parts: list[bytes] = []

    # ── 1. Initialise free-memory pointer ──────────────────────────────
    # Stack after (TOS first): [max_fp, val_{n-1}, …, val_0, <rest>]
    parts.append(fp_init())

    # ── 2. Write values in FORWARD order (val_0 first) ─────────────────
    # Forward order is required because each MSTORE writes 32 bytes — the
    # trailing zeros of an earlier write must be overwritten by the next.
    #
    # Stack positions (1-indexed from TOS):
    #   max_fp  → depth 1
    #   val_{n-1} → depth 2
    #   val_{n-2} → depth 3
    #   …
    #   val_i     → depth (n - i + 1)
    #   …
    #   val_0     → depth (n + 1)
    for i in range(n):
        off = offsets[i]
        nat = _natural_byte_size(flat[i])
        dup_depth = n - i + 1  # 1-indexed depth of val_i

        # DUP{dup_depth}: copy val_i to TOS
        parts.append(bytes([0x7F + dup_depth]))  # DUP1=0x80, DUP2=0x81, …

        # Left-align the value for packed MSTORE.
        is_bytes_fixed = _BYTES_FIXED_RE.match(flat[i])
        if is_bytes_fixed:
            # bytes<M> is already left-aligned; just mask trailing bytes.
            bsize = int(is_bytes_fixed.group(1))
            if bsize < 32:
                mask = ((1 << (bsize * 8)) - 1) << ((32 - bsize) * 8)
                parts.append(_push_small(mask) + bytes([OP_AND]))
        else:
            # Right-aligned types: shift left to fill the MSB.
            shift = 256 - nat * 8
            if shift > 0:
                # For bool, normalise to 0/1 first.
                if flat[i] == "bool":
                    parts.append(bytes([OP_ISZERO, OP_ISZERO]))
                parts.append(_push_small(shift) + bytes([OP_SHL]))
            # If shift==0 (uint256/int256), no transformation needed.

        # DUP2 to get max_fp, add offset, MSTORE
        parts.append(bytes([_DUP2]))
        if off > 0:
            parts.append(_push_small(off))
            parts.append(bytes([OP_ADD]))
        parts.append(bytes([OP_MSTORE]))
        # Stack restored: [max_fp, val_{n-1}, …, val_0, <rest>]

    # ── 3. Remove all n original values from the stack ─────────────────
    for _ in range(n):
        parts.append(bytes([OP_SWAP, OP_POP]))

    # ── 4. Update free-memory pointer ──────────────────────────────────
    parts.append(_emit_fp_update(padded))

    # ── 5. Leave [argsOffset(TOS), argsLen(2nd)] ──────────────────────
    parts.append(_push_small(total_bytes))  # [total_bytes, max_fp]
    parts.append(bytes([OP_SWAP]))  # [max_fp, total_bytes]

    return b"".join(parts)

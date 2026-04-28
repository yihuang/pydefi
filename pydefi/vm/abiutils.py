# Default max length for ABI types with no bound (bytes, string, dynamic arrays).
# Required for vyper types.
import re
from typing import Any, Sequence

from eth_typing import ABIComponent
from hexbytes import HexBytes
from vyper.codegen_venom.context import VenomCodegenContext
from vyper.codegen_venom.value import VyperValue
from vyper.exceptions import CompilerPanic
from vyper.semantics.types.base import VyperType
from vyper.semantics.types.bytestrings import BytesT, StringT, _BytestringT
from vyper.semantics.types.primitives import AddressT, BoolT, BytesM_T, DecimalT, IntegerT
from vyper.semantics.types.subscriptable import DArrayT, SArrayT, TupleT
from vyper.utils import DECIMAL_DIVISOR
from vyper.venom.basicblock import IRLiteral, IROperand, IRVariable

_DEFAULT_MAX_LEN = 1024

# Regex for splitting a type string into base + array suffixes.
# E.g. "uint256[10][]" → base="uint256", arrays=["[10]", "[]"]
_ARRAY_SUFFIX_RE = re.compile(r"(\[[0-9]*\])")


# =============================================================================
# Canonical conversion: TypedDict → VyperType
# =============================================================================


def _split_type_and_arrays(type_str: str) -> tuple[str, list[int | None]]:
    """
    Split a type string into base name and list of array dimensions.

    Returns ``(base_name, [length, ...])`` where ``length`` is
    ``None`` for dynamic arrays (``[]``) and ``int N`` for static (``[N]``).
    """
    parts = _ARRAY_SUFFIX_RE.split(type_str)
    base = parts[0]
    arrays: list[int | None] = []
    for raw in parts[1:]:
        if not raw:
            continue
        inner = raw[1:-1]  # strip [ ]
        if inner:
            arrays.append(int(inner))
        else:
            arrays.append(None)
    return base, arrays


def abi_to_vyper(comp: str | ABIComponent) -> VyperType:
    """
    Convert an ABI type description to a VyperType.

    Accepts a plain ABI type string (``"uint256"``), a JSON ABI component
    dict (``{"type": "uint256"}``), or a nested dict for tuple types.
    This is the canonical path for all type conversion.
    """
    if isinstance(comp, str):
        type_str: str = comp
        components: Sequence[ABIComponent] = []
    else:
        type_str = comp["type"]
        components = comp.get("components", [])

    base, arrays = _split_type_and_arrays(type_str)

    if base == "tuple":
        typ: VyperType = TupleT(tuple(abi_to_vyper(c) for c in components))
    else:
        typ = _base_to_vyper(base)

    # Apply array suffixes left-to-right (innermost first).
    # E.g. "uint256[3][4]" → arrays=[3, 4]
    # → SArrayT(SArrayT(uint256,3),4)
    for length in arrays:
        if length is not None:
            typ = SArrayT(typ, length)
        else:
            typ = DArrayT(typ, _DEFAULT_MAX_LEN)

    return typ


def _base_to_vyper(base: str) -> VyperType:
    """Map a base type name (no arrays) to a VyperType."""
    # Integers (also handle "uint256", "int8", bare "uint"/"int" etc.)
    if base == "uint" or base == "int":
        bits = 256  # bare "uint"/"int" alias uint256/int256
        is_signed = base == "int"
        return IntegerT(is_signed, bits)
    if base.startswith("uint"):
        bits = int(base[4:])
        return IntegerT(False, bits)
    if base.startswith("int"):
        bits = int(base[3:])
        return IntegerT(True, bits)

    if base == "address":
        return AddressT()
    if base == "bool":
        return BoolT()
    if base == "decimal":
        return DecimalT()
    if base == "string":
        return StringT(_DEFAULT_MAX_LEN)

    # bytesM (bytes1..bytes32) or dynamic bytes
    if base.startswith("bytes"):
        rest = base[5:]
        if rest:
            return BytesM_T(int(rest))
        return BytesT(_DEFAULT_MAX_LEN)

    raise ValueError(f"Unknown type: {base}")


def load_object(
    ctx: VenomCodegenContext,
    value: Any,
    typ: VyperType | str | ABIComponent,
) -> VyperValue:
    """
    Lower a Python object to a VyperValue using the given type.

    Accepts a VyperType, a plain ABI type string (e.g. ``"uint256"``),
    or a JSON ABI component dict.  Non-VyperType inputs are converted
    via :func:`abi_to_vyper`.
    """
    if not isinstance(typ, VyperType):
        typ = abi_to_vyper(typ)

    if typ._is_prim_word:
        return _load_primitive(value, typ)

    if isinstance(typ, _BytestringT):
        return _load_bytestring(ctx, value, typ)

    if isinstance(typ, TupleT):
        return _load_tuple(ctx, value, typ)

    if isinstance(typ, SArrayT):
        return _load_sarray(ctx, value, typ)

    if isinstance(typ, DArrayT):
        return _load_darray(ctx, value, typ)

    raise CompilerPanic(f"load_object: unsupported type {typ}")


def _load_primitive(value: Any, typ: VyperType) -> VyperValue:
    """Load a primitive word value. Accepts int, IROperand, bool, hex str."""
    if isinstance(value, IROperand):
        return VyperValue.from_stack_op(value, typ)

    if isinstance(value, int):
        if isinstance(typ, DecimalT):
            value = int(value * DECIMAL_DIVISOR)
        return VyperValue.from_stack_op(IRLiteral(value), typ)

    if isinstance(value, bool):
        return VyperValue.from_stack_op(IRLiteral(int(value)), typ)

    if isinstance(value, float) and isinstance(typ, DecimalT):
        return VyperValue.from_stack_op(IRLiteral(int(value * DECIMAL_DIVISOR)), typ)

    if isinstance(value, str):
        int_val = int.from_bytes(HexBytes(value), "big")

        if isinstance(typ, AddressT):
            return VyperValue.from_stack_op(IRLiteral(int_val), typ)

        if isinstance(typ, BytesM_T):
            n_bytes = typ.m
            int_val = int_val << 8 * (32 - n_bytes)
            return VyperValue.from_stack_op(IRLiteral(int_val), typ)

    raise CompilerPanic(f"load_object: cannot convert {type(value).__name__}({value!r}) to {typ}")


def _load_bytestring(
    ctx: VenomCodegenContext,
    value: Any,
    typ: _BytestringT,
) -> VyperValue:
    """Load a bytestring (Bytes or String) from Python bytes or str."""
    b = ctx.builder

    if isinstance(value, str):
        bytez: bytes = value.encode("utf-8")
    elif isinstance(value, bytes):
        bytez = value
    elif isinstance(value, bytearray):
        bytez = bytes(value)
    else:
        raise CompilerPanic(f"load_object: expected bytes/str for {typ}, got {type(value).__name__}")

    if len(bytez) > typ.length:
        raise CompilerPanic(f"load_object: value length {len(bytez)} exceeds {typ} max length {typ.length}")

    val = ctx.new_temporary_value(typ)
    assert isinstance(val.operand, IRVariable)

    ctx.ptr_store(val.ptr(), IRLiteral(len(bytez)))

    padded = bytez.ljust(((len(bytez) + 31) // 32) * 32, b"\x00")

    for i in range(0, len(bytez), 32):
        chunk = padded[i : i + 32]
        word = int.from_bytes(chunk, "big")
        offset = b.add(val.operand, IRLiteral(32 + i))
        b.mstore(offset, IRLiteral(word))

    return val


def _load_tuple(
    ctx: VenomCodegenContext,
    value: Any,
    typ: TupleT,
) -> VyperValue:
    """Load a tuple from a sequence of values."""
    b = ctx.builder

    if isinstance(value, (list, tuple)):
        elements = list(value)
    else:
        raise CompilerPanic(f"load_object: expected list/tuple for {typ}, got {type(value).__name__}")

    if len(elements) != len(typ.member_types):
        raise CompilerPanic(f"load_object: expected {len(typ.member_types)} elements for {typ}, got {len(elements)}")

    val = ctx.new_temporary_value(typ)
    assert isinstance(val.operand, IRVariable)

    offset = 0
    for elem_val, elem_typ in zip(elements, typ.member_types):
        elem_vv = load_object(ctx, elem_val, elem_typ)

        if offset == 0:
            dst: IROperand = val.operand
        else:
            dst = b.add(val.operand, IRLiteral(offset))

        ctx.store_vyper_value(elem_vv, dst, elem_typ)
        offset += elem_typ.memory_bytes_required

    return val


def _load_sarray(
    ctx: VenomCodegenContext,
    value: Any,
    typ: SArrayT,
) -> VyperValue:
    """Load a static array from a sequence of values."""
    b = ctx.builder

    if isinstance(value, (list, tuple)):
        elements = list(value)
    else:
        raise CompilerPanic(f"load_object: expected list/tuple for {typ}, got {type(value).__name__}")

    if len(elements) != typ.count:
        raise CompilerPanic(f"load_object: expected {typ.count} elements for {typ}, got {len(elements)}")

    val = ctx.new_temporary_value(typ)
    assert isinstance(val.operand, IRVariable)

    elem_typ = typ.value_type
    elem_size = elem_typ.memory_bytes_required

    for i, elem_val in enumerate(elements):
        elem_vv = load_object(ctx, elem_val, elem_typ)
        dst = b.add(val.operand, IRLiteral(i * elem_size))
        ctx.store_vyper_value(elem_vv, dst, elem_typ)

    return val


def _load_darray(
    ctx: VenomCodegenContext,
    value: Any,
    typ: DArrayT,
) -> VyperValue:
    """Load a dynamic array from a sequence of values."""
    b = ctx.builder

    if isinstance(value, (list, tuple)):
        elements = list(value)
    else:
        raise CompilerPanic(f"load_object: expected list/tuple for {typ}, got {type(value).__name__}")

    if len(elements) > typ.count:
        raise CompilerPanic(f"load_object: {len(elements)} elements exceeds {typ} max length {typ.count}")

    val = ctx.new_temporary_value(typ)
    assert isinstance(val.operand, IRVariable)

    elem_typ = typ.value_type
    elem_size = elem_typ.memory_bytes_required

    ctx.ptr_store(val.ptr(), IRLiteral(len(elements)))

    for i, elem_val in enumerate(elements):
        elem_vv = load_object(ctx, elem_val, elem_typ)
        dst = b.add(val.operand, IRLiteral(32 + i * elem_size))
        ctx.store_vyper_value(elem_vv, dst, elem_typ)

    return val

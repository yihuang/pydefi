"""High-level Venom IR program builder.

:class:`ProgramContext` inherits from :class:`~vyper.codegen_venom.context.VenomCodegenContext`
and is self-contained — it creates the :class:`~vyper.venom.context.IRContext` and
:class:`~vyper.venom.builder.VenomBuilder` internally, provides a dummy
:class:`~vyper.semantics.types.module.ModuleT`, and adds:

* :meth:`abi_encode` / :meth:`abi_decode` — type-safe helpers that don't need AST nodes
* :meth:`compile` — runs the full Venom optimisation pipeline and emits EVM bytecode

Usage::

    from pydefi.vm.context import ProgramContext
    from vyper.semantics.types.shortcuts import UINT256_T
    from vyper.semantics.types.primitives import AddressT

    ctx = ProgramContext()
    enc = ctx.abi_encode(
        (AddressT(), UINT256_T),
        (recipient, 10**18),
        method_id=bytes.fromhex("a9059cbb"),
    )
    ctx.builder.return_(enc.buf, enc.length)
    bytecode = ctx.compile()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Union

from vyper import ast as vy_ast
from vyper.codegen_venom.abi.abi_decoder import abi_decode_to_buf
from vyper.codegen_venom.abi.abi_encoder import abi_encode_to_buf
from vyper.codegen_venom.buffer import Buffer, Ptr
from vyper.codegen_venom.context import VenomCodegenContext
from vyper.codegen_venom.value import VyperValue
from vyper.compiler.settings import Settings, VenomOptimizationFlags, anchor_settings
from vyper.evm.assembler.core import assembly_to_evm
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import BytesT, DArrayT, SArrayT, StringT, TupleT, VyperType
from vyper.semantics.types.module import ModuleT
from vyper.semantics.types.primitives import AddressT, BoolT, BytesM_T, IntegerT
from vyper.venom import VenomCompiler, run_passes_on
from vyper.venom.basicblock import IRLabel, IRLiteral, IROperand, IRVariable
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

#: Handle to an SSA value.  Returned by builder methods (an ``IRVariable``)
#: or used as a compile-time constant (an ``IRLiteral``).  Both are accepted
#: anywhere ``IROperand`` is.
Value = Union[IRVariable, IRLiteral]

# Module-level dummy ModuleT shared by all ProgramContext instances.
# VenomCodegenContext requires a ModuleT, but our operations (abi encode/decode,
# variable management) only use ctx.builder, ctx.new_temporary_value(),
# ctx.unwrap(), and ctx.copy_memory() — none of which touch module_ctx.
_dummy_ast = vy_ast.Module(body=[], name="", doc_string=None, source_id=0)
_dummy_ast.path = ""  # required by ModuleT.__init__
_DUMMY_MODULE_T: ModuleT = ModuleT(_dummy_ast)


# ---------------------------------------------------------------------------
# Sig-string parsing
# ---------------------------------------------------------------------------
#
# Mini grammar (supports the cases pydefi needs today; nested tuples NYI):
#
#   atom = "uint" [bits]
#        | "int"  [bits]
#        | "address"
#        | "bool"
#        | "bytes" digits           # fixed bytesM, e.g. "bytes32"
#        | "bytes" ":" maxlen       # variable bytes, e.g. "bytes:64"
#   sig  = atom ("[" length "]")?   # optional dynamic-array dim
#
# Examples:
#   "uint256"      -> IntegerT(False, 256)
#   "address"      -> AddressT()
#   "bytes32"      -> BytesM_T(32)
#   "bytes:64"     -> BytesT(64)
#   "bytes:64[4]"  -> DArrayT(BytesT(64), 4)
#   "address[10]"  -> DArrayT(AddressT(), 10)

_SIG_RE = re.compile(r"^([a-z]+)(\d*)?(?::(\d+))?(?:\[(\d+)\])?$")


def _split_top_level_commas(s: str) -> list[str]:
    """Split *s* on top-level commas, respecting nested ``()`` / ``[]``."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"parse_sig: unbalanced brackets in {s!r}")
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        raise ValueError(f"parse_sig: unbalanced brackets in {s!r}")
    last = "".join(cur).strip()
    if last:
        parts.append(last)
    return parts


def parse_sig(sig: str) -> VyperType:
    """Parse a single ABI-style sig string into a :class:`VyperType`.

    Variable-length bytestrings require an explicit max length via
    ``bytes:N`` (the standard ABI ``bytes`` is ambiguous about capacity,
    which the codec needs for buffer allocation).

    Tuples are written ``(t1, t2, ...)``; nest as needed.
    """
    sig = sig.strip()
    if sig.startswith("("):
        if not sig.endswith(")"):
            raise ValueError(f"parse_sig: unterminated tuple {sig!r}")
        inner = sig[1:-1].strip()
        if not inner:
            raise ValueError(f"parse_sig: empty tuple {sig!r}")
        members = tuple(parse_sig(p) for p in _split_top_level_commas(inner))
        return TupleT(members)

    m = _SIG_RE.match(sig)
    if not m:
        raise ValueError(f"parse_sig: unrecognised sig {sig!r}")
    name, bits, maxlen, dim = m.groups()

    if name == "uint":
        n = int(bits) if bits else 256
        base: VyperType = IntegerT(False, n)
    elif name == "int":
        n = int(bits) if bits else 256
        base = IntegerT(True, n)
    elif name == "address" and not bits:
        base = AddressT()
    elif name == "bool" and not bits:
        base = BoolT()
    elif name == "bytes":
        if bits:
            base = BytesM_T(int(bits))
        elif maxlen:
            base = BytesT(int(maxlen))
        else:
            raise ValueError(f"parse_sig: bare 'bytes' is ambiguous; use 'bytes:N' for max length (got {sig!r})")
    else:
        raise ValueError(f"parse_sig: unknown atom {name!r} in {sig!r}")

    if dim:
        base = DArrayT(base, int(dim))
    return base


def parse_sigs(sigs: Sequence[str]) -> tuple[VyperType, ...]:
    """Parse a sequence of sig strings; returns a tuple of :class:`VyperType`
    (the natural input shape for :meth:`ProgramContext.abi_encode` /
    :meth:`ProgramContext.abi_decode`)."""
    return tuple(parse_sig(s) for s in sigs)


# ---------------------------------------------------------------------------
# Codec result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeBytes:
    """Runtime-sourced bytes payload for a ``BytesT`` slot in
    :meth:`ProgramContext.abi_encode`.  The data lives at *data_ptr* in
    memory; *length* is the runtime byte count (must be ``<= BytesT.length``
    of the slot).  The bytes do not need to be zero-padded.
    """

    length: Value
    data_ptr: Value


@dataclass(frozen=True)
class EncodedBuffer:
    """Result of :meth:`ProgramContext.abi_encode`: a memory pointer plus
    the encoded length, both as runtime SSA handles.  Pair with
    :meth:`pydefi.vm.Program.call_contract` ``(buf, length)`` form to send
    the encoded data, or feed to ``builder.return_(buf, length)``.
    """

    buf: Value
    length: Value


@dataclass
class DecodedBuffer:
    """Result of :meth:`ProgramContext.abi_decode`: a memory pointer to the
    decoded Vyper-internal-layout buffer for the wrapping tuple, plus the
    per-field byte offsets.  Use :meth:`read_word` for primitive-word fields
    (uint/int/address/bool/bytesM) and :meth:`read_bytes` for ``BytesT``
    fields.  Raw access via *buf* / *offsets* is still supported.
    """

    buf: Value
    offsets: tuple[int, ...]
    types: tuple[VyperType, ...]
    _ctx: "ProgramContext"

    def _field_ptr(self, idx: int) -> Value:
        if not 0 <= idx < len(self.types):
            raise IndexError(f"DecodedBuffer: field index {idx} out of range")
        off = self.offsets[idx]
        if off == 0:
            return self.buf
        return self._ctx.builder.add(self.buf, off)

    def read_word(self, idx: int) -> Value:
        """Read the 32-byte word at field *idx*.  Valid for primitive-word
        types (uint*, int*, address, bool, bytesN)."""
        typ = self.types[idx]
        if not typ._is_prim_word:
            raise TypeError(
                f"DecodedBuffer.read_word: field {idx} is {typ!r}, "
                f"not a primitive-word type; use read_bytes for bytestrings"
            )
        return self._ctx.builder.mload(self._field_ptr(idx))

    def read_bytes(self, idx: int) -> tuple[Value, Value]:
        """Return ``(length, data_ptr)`` for a ``BytesT`` field."""
        typ = self.types[idx]
        if not isinstance(typ, BytesT):
            raise TypeError(f"DecodedBuffer.read_bytes: field {idx} is {typ!r}, not BytesT")
        b = self._ctx.builder
        slot_ptr = self._field_ptr(idx)
        length = b.mload(slot_ptr)
        data_ptr = b.add(slot_ptr, 32)
        return length, data_ptr


class ProgramContext(VenomCodegenContext):
    """High-level Venom IR program builder.

    Inherits all of :class:`VenomCodegenContext` (``new_variable``, ``unwrap``,
    ``store_vyper_value``, ``allocate_buffer``, ``copy_memory``, ``block_scope``,
    etc.) and adds self-contained construction, ABI encode/decode helpers, and
    ``compile()``.
    """

    def __init__(
        self,
        ir_ctx: IRContext | None = None,
        fn_name: str = "main",
        *,
        evm_version: str = "shanghai",
    ) -> None:
        """Create a ProgramContext for a function within an IRContext.

        Args:
            ir_ctx: Shared IRContext to add the function to.  When ``None``
                (default) a fresh context is created.
            fn_name: Name for the function.  Defaults to ``"main"``; the
                first function added to an IRContext that has no entry
                function is automatically set as the entry point.
            evm_version: Target EVM hard fork.  Read by emitters that branch
                on opcode availability (e.g. MCOPY in Cancun+); the codec
                methods :meth:`abi_encode` / :meth:`abi_decode` wrap their
                IR emission in ``anchor_settings(Settings(evm_version=...))``.
                Default ``"shanghai"`` matches pydefi's ``mini_evm`` fixture.
        """
        if ir_ctx is None:
            ir_ctx = IRContext()
        self._ir_ctx = ir_ctx
        self._evm_version = evm_version
        fn = ir_ctx.create_function(fn_name)
        if ir_ctx.entry_function is None:
            ir_ctx.entry_function = fn
            # VenomCompiler emits functions in dict insertion order. Ensure
            # the entry function is first once at construction time.
            functions = ir_ctx.functions
            if next(iter(functions)) != fn.name:
                entry_fn = functions.pop(fn.name)
                reordered = {fn.name: entry_fn, **functions}
                functions.clear()
                functions.update(reordered)
        builder = VenomBuilder(ir_ctx, fn)
        super().__init__(module_ctx=_DUMMY_MODULE_T, builder=builder)

    # ------------------------------------------------------------------
    # ABI codec (Python-value staging, nested tuples, selector)
    # ------------------------------------------------------------------

    def abi_encode(
        self,
        types: tuple[VyperType, ...],
        values: tuple[object, ...],
        *,
        method_id: bytes | None = None,
    ) -> EncodedBuffer:
        """ABI-encode *values* according to *types* into a runtime memory
        buffer, returning ``EncodedBuffer(buf, length)``.

        Wraps the inputs in a tuple type, allocates source/destination buffers,
        stages each value into Vyper's internal layout (recursively, so nested
        tuples / dynamic arrays / Bytes work), then calls
        :func:`abi_encode_to_buf`.  The destination holds the standard ABI
        encoding; its length is returned as a runtime value (depends on
        runtime contents for dynamic types).

        Args:
            types: VyperType per argument.  Use objects directly
                (``UINT256_T``, ``BytesT(64)``) or :func:`parse_sigs`.
            values: Parallel tuple of values.  Each value may be:

                * a Python ``int`` / ``bool`` / ``bytes`` (primitive-word slots)
                * a ``bytes`` / :class:`RuntimeBytes` (BytesT slots)
                * a list / tuple of element values (DArrayT / TupleT slots)
                * an :class:`IRVariable` / :class:`IRLiteral` (already-staged
                  runtime SSA handle)
            method_id: Optional 4-byte function selector to prepend.  When set,
                ``buf`` points at the selector and ``length`` includes its 4
                bytes — the result is ready for a CALL's ``argsOffset/argsLen``.

        Returns:
            ``EncodedBuffer`` with ``buf`` (memory pointer) and ``length``
            (runtime SSA value of the encoded byte count).
        """
        if len(types) != len(values):
            raise ValueError(f"abi_encode: types/values length mismatch ({len(types)} vs {len(values)})")
        if method_id is not None and len(method_id) != 4:
            raise ValueError(f"abi_encode: method_id must be exactly 4 bytes, got {len(method_id)}")

        b = self.builder
        with anchor_settings(Settings(evm_version=self._evm_version)):  # type: ignore[call-arg]
            wrapped = TupleT(types)

            src_buf = self.allocate_buffer(wrapped.memory_bytes_required, annotation="abi_encode_src")
            cur = 0
            for typ, val in zip(types, values):
                self._write_into_internal_layout(src_buf._ptr, cur, typ, val)
                cur += typ.memory_bytes_required

            extra = 4 if method_id is not None else 0
            dst_buf = self.allocate_buffer(wrapped.abi_type.size_bound() + extra, annotation="abi_encode_dst")

            if method_id is not None:
                # Selector lives in the high 4 bytes of the word at offset 0.
                sel_word = int.from_bytes(method_id, "big") << ((32 - 4) * 8)
                b.mstore(dst_buf._ptr, IRLiteral(sel_word))
                encode_dst = b.add(dst_buf._ptr, IRLiteral(4))
                encoded_len = abi_encode_to_buf(self, encode_dst, src_buf._ptr, wrapped)
                total_len = b.add(encoded_len, IRLiteral(4))
                return EncodedBuffer(buf=dst_buf._ptr, length=total_len)
            encoded_len = abi_encode_to_buf(self, dst_buf._ptr, src_buf._ptr, wrapped)
            return EncodedBuffer(buf=dst_buf._ptr, length=encoded_len)

    def abi_decode(
        self,
        src: Value,
        src_len: Value,
        types: tuple[VyperType, ...],
    ) -> DecodedBuffer:
        """ABI-decode data at *src* (length *src_len*, in memory) into Vyper's
        internal layout for ``TupleT(types)``.

        *src_len* is the upper bound for bounds-checked decode — pydefi callers
        usually pass ``returndatasize()`` or the calldata length.  The returned
        :class:`DecodedBuffer` exposes typed accessors (:meth:`read_word`,
        :meth:`read_bytes`) so callers don't manually compute field offsets.
        """
        with anchor_settings(Settings(evm_version=self._evm_version)):  # type: ignore[call-arg]
            wrapped = TupleT(types)

            src_buffer = Buffer(_ptr=src, size=wrapped.memory_bytes_required, annotation="abi_decode_src")
            src_ptr = Ptr(operand=src, location=DataLocation.MEMORY, buf=src_buffer)
            src_val = VyperValue.from_ptr(src_ptr, wrapped)

            dst_buf = self.allocate_buffer(wrapped.memory_bytes_required, annotation="abi_decode_dst")

            hi = self.builder.add(src, src_len)
            abi_decode_to_buf(self, dst_buf._ptr, src_val, hi=hi)

            offsets = []
            cur = 0
            for typ in types:
                offsets.append(cur)
                cur += typ.memory_bytes_required

            return DecodedBuffer(
                buf=dst_buf._ptr,
                offsets=tuple(offsets),
                types=types,
                _ctx=self,
            )

    def _coerce_value(self, v: object) -> IROperand | int:
        """Coerce a Python value into something acceptable as a Venom operand.

        * ``IRVariable`` / ``IRLiteral`` → returned as-is.
        * ``int`` / ``bool`` → returned as ``int`` (Venom accepts directly).
        * ``bytes`` len 20 → big-endian ``int`` (an EVM address).
        """
        if isinstance(v, (IRVariable, IRLiteral)):
            return v
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            if v < 0:
                raise ValueError(f"_coerce_value: must be non-negative, got {v}")
            return v
        if isinstance(v, (bytes, bytearray, memoryview)):
            if len(v) != 20:
                raise ValueError(f"_coerce_value: bytes must be 20 (an address), got {len(v)} bytes")
            return int.from_bytes(bytes(v), "big")
        raise TypeError(f"_coerce_value: unsupported type {type(v).__name__}")

    def _write_into_internal_layout(
        self,
        base: IRVariable,
        offset: int,
        typ: VyperType,
        value: object,
    ) -> None:
        """Recursively write *value* into Vyper's internal memory layout for
        *typ* at ``base + offset``.  Supports primitive-word types, ``BytesT``
        / ``StringT`` (literal ``bytes``/``str`` or :class:`RuntimeBytes`),
        ``SArrayT``, ``DArrayT``, and nested ``TupleT``.
        """
        b = self.builder

        def _mstore_at(off: int, v: IROperand | int) -> None:
            addr = base if off == 0 else b.add(base, off)
            b.mstore(addr, v)

        if typ._is_prim_word:
            if isinstance(typ, BytesM_T) and isinstance(value, (bytes, bytearray, memoryview)):
                payload = bytes(value)
                if len(payload) != typ.m:
                    raise ValueError(f"{typ!r} expects exactly {typ.m} bytes, got {len(payload)}")
                word = int.from_bytes(payload, "big") << ((32 - typ.m) * 8)
                _mstore_at(offset, word)
                return
            _mstore_at(offset, self._coerce_value(value))
            return

        # BytesT and StringT share the same internal layout: [len][padded data].
        if isinstance(typ, (BytesT, StringT)):
            if isinstance(value, RuntimeBytes):
                # Mirror Vyper's Bytes[N] invariant for runtime payloads.
                b.assert_(b.iszero(b.gt(value.length, typ.length)))
                _mstore_at(offset, value.length)
                data_dst = b.add(base, offset + 32) if offset + 32 != 0 else base
                self.copy_memory_dynamic(data_dst, value.data_ptr, value.length)
                return
            if isinstance(value, str):
                payload: bytes = value.encode("utf-8")
            elif isinstance(value, (bytes, bytearray)):
                payload = bytes(value)
            else:
                raise TypeError(f"{type(typ).__name__} slot expects bytes/str/RuntimeBytes; got {type(value).__name__}")
            if len(payload) > typ.length:
                raise ValueError(f"payload exceeds {type(typ).__name__}[{typ.length}] cap: {len(payload)}B")
            _mstore_at(offset, len(payload))
            padded = payload + b"\x00" * ((32 - len(payload) % 32) % 32)
            for i in range(0, len(padded), 32):
                chunk = int.from_bytes(padded[i : i + 32], "big")
                _mstore_at(offset + 32 + i, chunk)
            return

        if isinstance(typ, SArrayT):
            # layout: [slot_0][slot_1]... (no count prefix; size is fixed).
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"SArrayT slot expects list/tuple; got {type(value).__name__}")
            if len(value) != typ.length:
                raise ValueError(f"SArrayT[{typ.length}] arity mismatch: expected {typ.length}, got {len(value)}")
            elem_stride = typ.value_type.memory_bytes_required
            for i, item in enumerate(value):
                self._write_into_internal_layout(base, offset + i * elem_stride, typ.value_type, item)
            return

        if isinstance(typ, DArrayT):
            # layout: [count][slot_0][slot_1]...
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"DArrayT slot expects list/tuple; got {type(value).__name__}")
            if len(value) > typ.length:
                raise ValueError(f"list exceeds DynArray[..., {typ.length}] cap: {len(value)}")
            elem_stride = typ.value_type.memory_bytes_required
            _mstore_at(offset, len(value))
            for i, item in enumerate(value):
                self._write_into_internal_layout(base, offset + 32 + i * elem_stride, typ.value_type, item)
            return

        if isinstance(typ, TupleT):
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"TupleT slot expects list/tuple; got {type(value).__name__}")
            if len(value) != len(typ.member_types):
                raise ValueError(
                    f"tuple arity mismatch for {typ!r}: expected {len(typ.member_types)} members, got {len(value)}"
                )
            cur = offset
            for member_typ, member_val in zip(typ.member_types, value):
                self._write_into_internal_layout(base, cur, member_typ, member_val)
                cur += member_typ.memory_bytes_required
            return

        raise NotImplementedError(f"_write_into_internal_layout: unsupported type {typ!r}")

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(
        self,
        flags: VenomOptimizationFlags | None = None,
    ) -> bytes:
        """Compile this context to EVM bytecode.

        Runs the full Venom IR optimisation pipeline and generates EVM
        bytecode via :class:`~vyper.venom.VenomCompiler`.

        Returns:
            Compiled EVM bytecode as :class:`bytes`.
        """
        if flags is None:
            flags = VenomOptimizationFlags()
        run_passes_on(self._ir_ctx, flags)

        compiler = VenomCompiler(self._ir_ctx)
        asm = compiler.generate_evm_assembly()
        bytecode, _ = assembly_to_evm(asm)
        return bytecode

    # ------------------------------------------------------------------
    # Data sections
    # ------------------------------------------------------------------

    def append_data_section(self, name: str) -> None:
        """Append a named data section."""
        self._ir_ctx.append_data_section(IRLabel(name))

    def append_data_item(self, data: IRLabel | bytes) -> None:
        """Append a data item to the most-recently opened data section."""
        self._ir_ctx.append_data_item(data)

    def runtime_buffer(self, size: int) -> IRVariable:
        """Allocate a buffer and return its pointer.

        The returned buffer has undefined contents — use calldatacopy or
        codecopy to fill it at runtime.  The ``alloca`` is tracked by the
        memory allocator like any other static allocation.

        Compared to :meth:`allocate_buffer`, this is the raw ``alloca``
        output without a ``Buffer`` wrapper.
        """
        return self.builder.alloca(size)

    def embed_and_load(self, data: bytes) -> IRVariable:
        """Embed *data* in a data section and copy it into a memory buffer
        at runtime using a volatile ``codecopy``.

        Unlike ``mstore`` with compile-time constants, ``codecopy`` cannot
        be eliminated by the optimiser, making this safe for use as input
        to ``abi_decode`` even when the data is known at build time.
        """
        b = self.builder
        label_name = f"_data_{len(self._ir_ctx.data_segment)}"
        label = IRLabel(label_name)
        self._ir_ctx.append_data_section(label)
        self._ir_ctx.append_data_item(data)
        buf = b.alloca(len(data))
        src = b.offset(IRLiteral(0), label)
        b.codecopy(buf, src, IRLiteral(len(data)))
        return buf

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def ir_ctx(self) -> IRContext:
        """The underlying :class:`~vyper.venom.context.IRContext`."""
        return self._ir_ctx

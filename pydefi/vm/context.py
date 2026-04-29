"""DeFiVM ProgramContext — SSA-style builder over Vyper's Venom IR.

A Pythonic value-flow API: methods that produce values return :class:`Operand`
handles that the user threads through subsequent calls, and Venom's stack
allocator generates DUP / SWAP / POP automatically.  Quick example::

    from pydefi.vm import ProgramContext

    ctx = ProgramContext()
    success = ctx.call_contract(ROUTER, ROUTER_FN, recipient, amount)
    ctx.assert_(success)
    ctx.stop()
    bytecode = ctx.build()

Patched call (raw calldata + runtime overlay)::

    ctx = ProgramContext()
    quote_ok = ctx.call_raw(QUOTER, quote_calldata)
    ctx.assert_(quote_ok)
    amount = ctx.returndata_word(0)
    swap_ok = ctx.call_raw(
        ROUTER,
        swap_template,
        patches={36: amount},  # write amount into the calldata at offset 36
    )
    ctx.assert_(swap_ok)
    ctx.stop()
    bytecode = ctx.build()

Design notes
------------
``ProgramContext`` owns one :class:`vyper.venom.context.IRContext` and one
``main`` :class:`IRFunction` for its lifetime.  Each method that emits IR
appends instructions to the *current* basic block, exposed via
``self.builder.current_block()``.

**Calldata buffers.**  Static calldata for external calls is appended to a
Venom data section; the body emits ``codecopy`` from the section into a
fresh allocation, applies optional patches via ``mstore``, then ``call``.
Venom resolves the data-section label to its absolute byte position in
the compiled output, so no post-processing patches are needed.

**Termination.**  A ``ProgramContext`` is "open" until the user calls one
of :meth:`stop`, :meth:`return_`, :meth:`return_word`, :meth:`revert`, or
a control-flow primitive that terminates the current BB.  :meth:`build`
adds an implicit ``stop`` if the current BB is still open.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence, Union

from eth_contract.contract import ContractFunction
from eth_typing import ABIComponent
from vyper import ast as vy_ast
from vyper.codegen_venom.abi.abi_decoder import abi_decode_to_buf
from vyper.codegen_venom.abi.abi_encoder import abi_encode_to_buf
from vyper.codegen_venom.buffer import Buffer, Ptr
from vyper.codegen_venom.context import VenomCodegenContext
from vyper.codegen_venom.value import VyperValue
from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
from vyper.evm.assembler.core import assembly_to_evm
from vyper.evm.assembler.instructions import (
    CONST,
    DATA_ITEM,
    PUSH_OFST,
    PUSHLABEL,
    DataHeader,
)
from vyper.evm.assembler.instructions import Label as _AsmLabel
from vyper.evm.assembler.symbols import SYMBOL_SIZE
from vyper.evm.opcodes import OPCODES
from vyper.semantics.data_locations import DataLocation
from vyper.semantics.types import BytesT, TupleT, VyperType
from vyper.semantics.types.module import ModuleT
from vyper.venom import VenomCompiler, generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRLabel, IRLiteral, IRVariable
from vyper.venom.builder import VenomBuilder
from vyper.venom.context import IRContext

from pydefi.vm.abiutils import abi_to_vyper, load_object

if TYPE_CHECKING:
    from pydefi.types import Address


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: A Venom operand — SSA handle (``IRVariable`` / ``IRLiteral``) or a plain
#: ``int`` literal that Venom accepts directly.  Returned by value-producing
#: methods on :class:`ProgramContext` and passed between internal helpers.
Operand = Union[IRVariable, IRLiteral, int]

#: Anything callers may pass in an :data:`Operand` slot — an :data:`Operand`,
#: or 20-byte ``bytes`` interpreted as a big-endian address.  Coerced to
#: :data:`Operand` once at the public-method boundary via
#: :meth:`ProgramContext._to_operand`.
ValueLike = Union[Operand, bytes]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Module-level dummy ModuleT shared by all ProgramContext instances.
# VenomCodegenContext requires a ModuleT, but our operations (abi encode/decode,
# variable management) only use ctx.builder, ctx.new_temporary_value(),
# ctx.unwrap(), and ctx.copy_memory() — none of which touch module_ctx.
_dummy_ast = vy_ast.Module(body=[], name="", doc_string=None, source_id=0)
_dummy_ast.path = ""  # required by ModuleT.__init__
_DUMMY_MODULE_T: ModuleT = ModuleT(_dummy_ast)


# ---------------------------------------------------------------------------
# Internal helpers — assembly / bytecode post-processing
# ---------------------------------------------------------------------------


def _asm_item_size(item: object) -> int:
    """Byte size that *item* contributes when assembled to EVM bytecode.

    Mirrors :func:`vyper.evm.assembler.core._assembly_to_evm` for the asm item
    types Venom emits.  Used to walk the asm list and locate byte positions of
    label-resolved PUSH_OFST / PUSHLABEL instructions when post-processing the
    bytecode.
    """
    if item == "DEBUG" or isinstance(item, (CONST, DataHeader)):
        return 0
    if isinstance(item, PUSHLABEL):
        return 1 + SYMBOL_SIZE
    if isinstance(item, _AsmLabel):
        return 1  # JUMPDEST
    if isinstance(item, PUSH_OFST):
        if isinstance(item.label, _AsmLabel):
            return 1 + SYMBOL_SIZE
        raise NotImplementedError(f"_asm_item_size: CONSTREF PUSH_OFST not supported: {item!r}")
    if isinstance(item, int):
        return 1
    if isinstance(item, str):
        up = item.upper()
        if up.startswith("PUSH") and up != "PUSH":
            return 1
        if up.startswith("DUP") or up.startswith("SWAP"):
            return 1
        if up in OPCODES:
            return 1
        raise ValueError(f"_asm_item_size: unrecognised asm string item: {item!r}")
    if isinstance(item, DATA_ITEM):
        if isinstance(item.data, bytes):
            return len(item.data)
        if isinstance(item.data, _AsmLabel):
            return SYMBOL_SIZE
        raise ValueError(f"_asm_item_size: unrecognised DATA_ITEM payload: {item.data!r}")
    raise ValueError(f"_asm_item_size: unknown asm item type: {type(item).__name__}")


def _shift_label_pushes(asm: list, bytecode: bytes, shift: int) -> bytes:
    """Add *shift* to every label-resolved ``PUSH_OFST`` / ``PUSHLABEL`` immediate.

    Used when the compiled program will be embedded inside a larger bytecode
    buffer at runtime — e.g. CCTP / OFT composer contracts prepend a 66-byte
    PUSH32 prologue before our program runs.  Venom resolves data-section and
    jump-target labels to absolute byte offsets at compile time; if the
    program is then shifted, those references read / jump to the wrong place.
    Adding *shift* to each one keeps them correct.
    """
    if shift == 0:
        return bytecode
    buf = bytearray(bytecode)
    pos = 0
    for item in asm:
        size = _asm_item_size(item)
        if isinstance(item, (PUSH_OFST, PUSHLABEL)) and isinstance(getattr(item, "label", None), _AsmLabel):
            imm_pos = pos + 1  # immediate sits after the 1-byte PUSH<SYMBOL_SIZE> opcode
            current = int.from_bytes(buf[imm_pos : imm_pos + SYMBOL_SIZE], "big")
            shifted = current + shift
            if shifted >> (SYMBOL_SIZE * 8) != 0:
                raise ValueError(
                    f"_shift_label_pushes: shifted offset {shifted} does not fit in "
                    f"{SYMBOL_SIZE} bytes (label={item.label!r}, shift={shift})"
                )
            buf[imm_pos : imm_pos + SYMBOL_SIZE] = shifted.to_bytes(SYMBOL_SIZE, "big")
        pos += size
    return bytes(buf)


# ---------------------------------------------------------------------------
# ProgramContext
# ---------------------------------------------------------------------------


class ProgramContext(VenomCodegenContext):
    """SSA-style fluent Venom IR builder.

    Inherits :class:`VenomCodegenContext` (``new_variable``, ``unwrap``,
    ``store_vyper_value``, ``allocate_buffer``, ``copy_memory``,
    ``block_scope``, …) and adds:

    * Self-contained construction (creates ``IRContext``/``VenomBuilder``)
    * Pythonic SSA helpers — :meth:`const`, :meth:`add`, :meth:`call_contract`,
      :meth:`assert_`, :meth:`return_word`, …
    * ABI helpers — :meth:`abi_encode`, :meth:`abi_decode`
    * Bytecode emission — :meth:`build` (with ``prefix_length`` shift) and
      :meth:`compile` (no shift, used by stdlib / abi tests)

    Methods either:

    * **Produce a value** — return a :class:`Operand` handle.  Examples:
      :meth:`const`, :meth:`add`, :meth:`call_contract`, :meth:`load_slot`.
    * **Have a side effect** — return ``None``.  Examples: :meth:`store_slot`,
      :meth:`assert_`, :meth:`stop`.

    Wherever a method takes a ``ValueLike`` argument, you may pass a
    :class:`Operand` returned by another method, a Python ``int`` (auto-wrapped
    as a constant), or 20 raw bytes (interpreted as a big-endian address).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, ir_ctx: IRContext | None = None, fn_name: str = "main", *, set_entry: bool = True) -> None:
        """Create a ProgramContext for a function within an IRContext.

        Args:
            ir_ctx: Shared IRContext to add the function to.  When ``None``
                (default) a fresh context is created.
            fn_name: Name for the function.  Defaults to ``"main"``.
            set_entry: If ``True`` (default) and *ir_ctx* has no entry
                function yet, this function is set as the entry point and
                the venom stdlib helpers (``stdlib_revert_if`` /
                ``stdlib_assert_ge``) are added.  Pass ``False`` when
                building library/utility functions inside an existing
                context (e.g. while constructing the stdlib itself).
        """
        if ir_ctx is None:
            ir_ctx = IRContext()
        self._ir_ctx = ir_ctx
        fn = ir_ctx.create_function(fn_name)
        if set_entry and ir_ctx.entry_function is None:
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

        # Auto-load stdlib helpers for entry contexts.  Lazy import avoids
        # the circular pydefi.vm.context ↔ pydefi.vm.stdlib module load.
        # FunctionInlinerPass + DCE eliminate unused stdlib at build time,
        # so there's no bytecode cost.
        if set_entry:
            from pydefi.vm.stdlib import build_stdlib

            build_stdlib(ir_ctx)

    # ------------------------------------------------------------------
    # Operand coercion
    # ------------------------------------------------------------------

    def _to_operand(self, v: ValueLike) -> Operand:
        """Coerce a :class:`ValueLike` into something acceptable as a Venom operand.

        * :class:`Operand` (``IRVariable`` / ``IRLiteral``) → returned as-is.
        * ``int``            → Venom accepts plain ``int`` literals directly.
        * ``bytes`` (len 20) → big-endian ``int`` (an EVM address).
        """
        if isinstance(v, (IRVariable, IRLiteral)):
            return v
        if isinstance(v, int):
            if v < 0:
                raise ValueError(f"operand must be non-negative, got {v}")
            return v
        if isinstance(v, (bytes, bytearray, memoryview)):
            if len(v) != 20:
                raise ValueError(
                    f"operand: bytes must be 20 (an address), got {len(v)} bytes; "
                    "use ProgramContext.const(int.from_bytes(...)) for arbitrary widths"
                )
            return int.from_bytes(bytes(v), "big")
        raise TypeError(f"operand: unsupported type {type(v).__name__}")

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def const(self, n: int) -> Operand:
        """Return a :class:`Operand` wrapping the constant uint256 *n*."""
        if n < 0:
            raise ValueError(f"const: value must be non-negative, got {n}")
        if n >= 1 << 256:
            raise ValueError(f"const: value {n} does not fit in uint256")
        return IRLiteral(n)

    def addr(self, a: Address) -> Operand:
        """Return a :class:`Operand` for a 20-byte EVM address."""
        if len(a) != 20:
            raise ValueError(f"addr: address must be 20 bytes, got {len(a)}")
        return IRLiteral(int.from_bytes(a, "big"))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def add(self, a: ValueLike, b: ValueLike) -> Operand:
        """Wrapping uint256 ``a + b``."""
        return self.builder.add(self._to_operand(a), self._to_operand(b))

    def sub(self, a: ValueLike, b: ValueLike) -> Operand:
        """Saturating ``max(a - b, 0)``.

        Implemented as ``(a - b) * (a >= b)`` since EVM SUB wraps modulo 2^256.
        """
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        # raw_diff = a - b (wraps if a < b)
        raw_diff = self.builder.sub(a_op, b_op)
        # not_underflow = 1 if a >= b else 0  ==  iszero(a < b)
        not_underflow = self.builder.iszero(self.builder.lt(a_op, b_op))
        return self.builder.mul(raw_diff, not_underflow)

    def mul(self, a: ValueLike, b: ValueLike) -> Operand:
        """Wrapping uint256 ``a * b``."""
        return self.builder.mul(self._to_operand(a), self._to_operand(b))

    def div(self, a: ValueLike, b: ValueLike) -> Operand:
        """Unsigned ``a // b``; EVM DIV returns 0 when ``b == 0``."""
        return self.builder.div(self._to_operand(a), self._to_operand(b))

    def mod(self, a: ValueLike, b: ValueLike) -> Operand:
        """Unsigned ``a % b``; EVM MOD returns 0 when ``b == 0``."""
        return self.builder.mod(self._to_operand(a), self._to_operand(b))

    # ------------------------------------------------------------------
    # Comparison / boolean
    # ------------------------------------------------------------------

    def lt(self, a: ValueLike, b: ValueLike) -> Operand:
        """Unsigned ``1 if a < b else 0``."""
        return self.builder.lt(self._to_operand(a), self._to_operand(b))

    def gt(self, a: ValueLike, b: ValueLike) -> Operand:
        """Unsigned ``1 if a > b else 0``."""
        return self.builder.gt(self._to_operand(a), self._to_operand(b))

    def eq(self, a: ValueLike, b: ValueLike) -> Operand:
        """``1 if a == b else 0``."""
        return self.builder.eq(self._to_operand(a), self._to_operand(b))

    def is_zero(self, a: ValueLike) -> Operand:
        """``1 if a == 0 else 0``."""
        return self.builder.iszero(self._to_operand(a))

    # ------------------------------------------------------------------
    # Bitwise
    # ------------------------------------------------------------------

    def bit_and(self, a: ValueLike, b: ValueLike) -> Operand:
        return self.builder.and_(self._to_operand(a), self._to_operand(b))

    def bit_or(self, a: ValueLike, b: ValueLike) -> Operand:
        return self.builder.or_(self._to_operand(a), self._to_operand(b))

    def bit_xor(self, a: ValueLike, b: ValueLike) -> Operand:
        return self.builder.xor(self._to_operand(a), self._to_operand(b))

    def bit_not(self, a: ValueLike) -> Operand:
        return self.builder.not_(self._to_operand(a))

    def shl(self, value: ValueLike, shift: ValueLike) -> Operand:
        """``value << shift`` (Venom signature: ``shl(bits, val)``)."""
        return self.builder.shl(self._to_operand(shift), self._to_operand(value))

    def shr(self, value: ValueLike, shift: ValueLike) -> Operand:
        """``value >> shift`` (Venom signature: ``shr(bits, val)``)."""
        return self.builder.shr(self._to_operand(shift), self._to_operand(value))

    # ------------------------------------------------------------------
    # Memory slots
    # ------------------------------------------------------------------

    def alloc_slot(self) -> IRVariable:
        """Allocate a 32-byte memory slot and return a handle to its address.

        Use :meth:`store_slot` and :meth:`load_slot` to read and write it.
        """
        return self.builder.alloca(32)

    def load_slot(self, slot: IRVariable) -> Operand:
        """Load the 32-byte word stored at *slot*."""
        return self.builder.mload(slot)

    def store_slot(self, slot: IRVariable, v: ValueLike) -> None:
        """Store *v* (32 bytes) into *slot*."""
        self.builder.mstore(slot, self._to_operand(v))

    # ------------------------------------------------------------------
    # Self / context
    # ------------------------------------------------------------------

    def self_addr(self) -> Operand:
        """EVM ``ADDRESS`` — the running program's own address."""
        return self.builder.address()

    def gas_left(self) -> Operand:
        """EVM ``GAS`` — remaining gas."""
        return self.builder.gas()

    def eth_balance(self, account: ValueLike) -> Operand:
        """EVM ``BALANCE(account)`` — ETH balance of *account*."""
        return self.builder.balance(self._to_operand(account))

    def erc20_balance_of(self, token: ValueLike, account: ValueLike) -> Operand:
        """ERC-20 ``token.balanceOf(account)`` via STATICCALL.

        Reverts if the staticcall fails.  Use :meth:`eth_balance` for native
        ETH balance (calls the BALANCE opcode directly, no staticcall).
        """
        # ABI: balanceOf(address) selector 0x70a08231, arg at offset 4..36.
        selector = 0x70A08231 << (28 * 8)  # left-align to byte 0..3 of a 32-byte word
        scratch = self._alloc(36)
        # mem[scratch] = selector word (bytes 0..3 hold the selector, 4..31 are zero)
        self.builder.mstore(scratch, selector)
        # mem[scratch+4] = account (right-aligned in a 32-byte word; MSTORE at
        # scratch+4 writes bytes scratch+4..scratch+36, with 12 leading zeros
        # then the 20-byte address).
        self.builder.mstore(self.builder.add(scratch, 4), self._to_operand(account))
        # STATICCALL(gas, token, scratch, 36, scratch, 32) — reuse scratch for retdata.
        success = self.builder.staticcall(
            self.builder.gas(),
            self._to_operand(token),
            scratch,
            36,
            scratch,
            32,
        )
        self.builder.assert_(success)
        return self.builder.mload(scratch)

    # ------------------------------------------------------------------
    # ABI helpers (no AST nodes)
    # ------------------------------------------------------------------

    def abi_encode(
        self,
        args: Sequence[Any],
        types: Sequence[VyperType | ABIComponent],
        *,
        method_id: bytes | None = None,
        ensure_tuple: bool = True,
    ) -> VyperValue:
        if method_id is not None and len(method_id) != 4:
            raise ValueError("method_id must be 4 bytes")
        if len(args) != len(types):
            raise ValueError("args and types must have the same length")

        b = self.builder

        vyper_types = [abi_to_vyper(comp) if not isinstance(comp, VyperType) else comp for comp in types]

        if len(args) == 1 and not ensure_tuple:
            value = args[0]
            vyper_type = vyper_types[0]
        else:
            value = args
            vyper_type = TupleT(tuple(vyper_types))

        vyper_value = load_object(self, value, vyper_type)

        # abi_encode_to_buf expects a memory pointer for src, even for
        # primitives.  If load_object returned a stack value (primitives),
        # store it to a temporary first.
        if vyper_value.is_stack_value:
            tmp = self.new_temporary_value(vyper_type)
            assert isinstance(tmp.operand, IRVariable)
            b.mstore(tmp.operand, vyper_value.operand)
            encode_src = tmp.operand
        else:
            encode_src = vyper_value.operand

        # Allocate output buffer: [length word] [method_id?] [data]
        offset = 4 if method_id is not None else 0
        maxlen = vyper_type.abi_type.size_bound() + offset
        buf_t = BytesT(maxlen)
        buf_val = self.new_temporary_value(buf_t)
        assert isinstance(buf_val.operand, IRVariable)

        if method_id is not None:
            method_id_word = int.from_bytes(method_id.ljust(32, b"\x00"), "big")
            b.mstore(b.add(buf_val.operand, IRLiteral(32)), IRLiteral(method_id_word))
            data_dst = b.add(buf_val.operand, IRLiteral(36))
        else:
            data_dst = b.add(buf_val.operand, IRLiteral(32))

        encoded_len = abi_encode_to_buf(self, data_dst, encode_src, vyper_type)
        if offset > 0:
            encoded_len = b.add(encoded_len, IRLiteral(offset))
        b.mstore(buf_val.operand, encoded_len)

        return buf_val

    def abi_decode(
        self,
        data: IRVariable,
        output_type: VyperType,
        *,
        unwrap_tuple: bool = True,
    ) -> VyperValue:
        """ABI-decode a Bytes buffer and return the decoded value in memory.

        Args:
            data: Memory pointer to the Bytes buffer (length word + ABI data).
            output_type: The VyperType to decode into.
            unwrap_tuple: If True (default), single-element tuples are
                unwrapped to the element type.

        Returns:
            ``VyperValue`` pointing to the decoded value in Vyper memory layout.
        """
        b = self.builder

        # The Bytes buffer: [length_word][ABI data ...]
        data_len = b.mload(data)
        data_ptr = b.add(data, IRLiteral(32))

        # Determine the ABI-level type (may be wrapped in a tuple).
        wrapped_typ = output_type
        # For ABI conformance, external return types are wrapped in tuples.
        # We only wrap if the output_type itself isn't already a tuple and
        # unwrap_tuple is True (meaning the caller wants us to handle the
        # tuple wrapping/unwrapping automatically).
        if unwrap_tuple and not isinstance(output_type, TupleT):
            wrapped_typ = TupleT((output_type,))

        # Validate size bounds.
        abi_min_size = wrapped_typ.abi_type.static_size()
        abi_max_size = wrapped_typ.abi_type.size_bound()
        if abi_min_size == abi_max_size:
            b.assert_(b.eq(data_len, IRLiteral(abi_min_size)))
        else:
            ge_min = b.iszero(b.lt(data_len, IRLiteral(abi_min_size)))
            le_max = b.iszero(b.gt(data_len, IRLiteral(abi_max_size)))
            b.assert_(b.and_(ge_min, le_max))

        # Allocate output buffer and decode.
        output_val = self.new_temporary_value(wrapped_typ)
        assert isinstance(output_val.operand, IRVariable)

        hi = b.add(data_ptr, data_len)
        buf = Buffer(_ptr=data_ptr, size=wrapped_typ.memory_bytes_required, annotation="abi_decode_src")
        ptr = Ptr(operand=data_ptr, location=DataLocation.MEMORY, buf=buf)
        src = VyperValue.from_ptr(ptr, wrapped_typ)
        abi_decode_to_buf(self, output_val.operand, src, hi=hi)

        # Unwrap single-element tuple if requested.
        if unwrap_tuple and isinstance(wrapped_typ, TupleT) and wrapped_typ != output_type:
            return VyperValue.from_ptr(output_val.ptr(), output_type)
        return output_val

    # ------------------------------------------------------------------
    # Internal: memory + data-section helpers (Venom alloca)
    # ------------------------------------------------------------------

    def _alloc_calldata(self, calldata: bytes) -> tuple[IRVariable, int]:
        """Embed *calldata* in a data section and return ``(base, len)``.

        Thin wrapper around :meth:`embed_and_load` that returns the unpadded
        byte length alongside the buffer pointer (the value to pass as
        ``argsLen`` to ``CALL``).
        """
        if len(calldata) > 0xFFFF:
            raise ValueError(f"calldata too large ({len(calldata)} bytes, max 65535)")
        return self.embed_and_load(calldata), len(calldata)

    def _alloc(self, size: int) -> IRVariable:
        """Allocate *size* bytes (rounded up to 32) and return the base pointer."""
        return self.builder.alloca((size + 31) & ~31)

    # ------------------------------------------------------------------
    # External calls
    # ------------------------------------------------------------------

    def call_contract(
        self,
        to: ValueLike,
        fn: ContractFunction | str,
        *args: object,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
    ) -> Operand:
        """Emit a CALL with calldata built from a :class:`ContractFunction`.

        *fn* may also be a human-readable signature string
        (``"transfer(address,uint256)"`` or ``"function transfer(...)"``);
        it will be converted to a :class:`ContractFunction` internally.
        Prefer hoisting a :class:`ContractFunction` to module scope to avoid
        re-parsing on every call.

        Each arg in *args* may be a plain Python constant (``int``, address
        ``bytes``/``str``, ``bool``, nested ``tuple``/``list``) or a runtime
        :class:`Operand` (``IRVariable``/``IRLiteral``); both are encoded
        uniformly by :meth:`abi_encode`.

        For raw pre-encoded calldata + ``patches`` overlays, use
        :meth:`call_raw` instead.

        Returns:
            A :class:`Operand` holding the CALL success flag.
        """
        if isinstance(fn, str):
            normalised = fn if fn.lstrip().startswith("function ") else "function " + fn
            fn = ContractFunction.from_abi(normalised)
        param_types = [abi_to_vyper(t) for t in fn.input_types]
        if len(args) != len(param_types):
            raise ValueError(
                f"call_contract: expected {len(param_types)} argument(s) for {fn.signature!r}, got {len(args)}"
            )

        buf_val = self.abi_encode(list(args), param_types, method_id=bytes(fn.selector))

        # buf_val points to a Bytes buffer: [length_word][selector + abi data].
        buf_ptr = buf_val.operand
        blen = self.builder.mload(buf_ptr)
        base = self.builder.add(buf_ptr, IRLiteral(32))
        gas_op = self.builder.gas() if gas is None else self._to_operand(gas)
        return self.builder.call(
            gas_op,
            self._to_operand(to),
            self._to_operand(value),
            base,
            blen,
            0,
            0,
        )

    def call_raw(
        self,
        to: ValueLike,
        calldata: bytes,
        *,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
        patches: Mapping[int, ValueLike] | None = None,
    ) -> Operand:
        """Emit a CALL with pre-encoded calldata and optional runtime patches.

        Args:
            to:        Target contract address.
            calldata:  Pre-encoded calldata template; stored in a Venom data
                       section and copied into memory before the call.
            value:     ETH value to forward (wei).
            gas:       Gas limit; ``None`` forwards all remaining gas.
            patches:   Optional ``{calldata_offset: value}`` overlay.  Each
                       entry overwrites a 32-byte word in the in-memory
                       calldata buffer at the given offset before CALL.

        Returns:
            A :class:`Operand` holding the CALL success flag (1 on success,
            0 on failure).  Use :meth:`assert_` to revert on failure.
        """
        base_fp, blen = self._alloc_calldata(calldata)

        if patches:
            for offset, val in patches.items():
                if not 0 <= offset <= blen - 32:
                    raise ValueError(
                        f"call_raw: patch offset {offset} out of bounds "
                        f"(calldata length {blen}, must satisfy 0 <= offset <= len-32)"
                    )
                target_addr = self.builder.add(base_fp, offset)
                self.builder.mstore(target_addr, self._to_operand(val))

        gas_op = self.builder.gas() if gas is None else self._to_operand(gas)
        success = self.builder.call(
            gas_op,
            self._to_operand(to),
            self._to_operand(value),
            base_fp,
            blen,
            0,
            0,
        )
        return success

    # ------------------------------------------------------------------
    # Returndata
    # ------------------------------------------------------------------

    def returndata_word(self, offset: int = 0) -> Operand:
        """Read 32 bytes from the last call's returndata at *offset*."""
        if offset < 0:
            raise ValueError(f"returndata_word: offset must be non-negative, got {offset}")
        scratch = self._alloc(32)
        self.builder.returndatacopy(scratch, offset, 32)
        return self.builder.mload(scratch)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_(self, cond: ValueLike, msg: str = "") -> None:
        """Revert if *cond* is zero.

        Empty *msg* emits a bare ``REVERT(0, 0)``; non-empty *msg* (≤ 32 bytes)
        invokes ``stdlib_revert_if`` for the Solidity ``Error(string)`` payload.
        """
        if not msg:
            self.builder.assert_(self._to_operand(cond))
            return
        cond_zero = self.builder.iszero(self._to_operand(cond))
        self._invoke_revert_if(cond_zero, msg)

    def assert_ge(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a < b`` (i.e. require ``a >= b``)."""
        a_op = self._to_operand(a)
        b_op = self._to_operand(b)
        if msg:
            self._invoke_assert_ge(a_op, b_op, msg)
            return
        not_lt = self.builder.iszero(self.builder.lt(a_op, b_op))
        self.builder.assert_(not_lt)

    def assert_le(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a > b`` (i.e. require ``a <= b``) — ``assert_ge(b, a)``."""
        self.assert_ge(b, a, msg)

    def _invoke_revert_if(self, cond: Operand, msg: str) -> None:
        from pydefi.vm.stdlib import encode_msg

        msg_len, msg_word = encode_msg(msg)
        self.builder.invoke(
            IRLabel("stdlib_revert_if"),
            [cond, IRLiteral(msg_len), IRLiteral(msg_word)],
            returns=0,
        )

    def _invoke_assert_ge(self, a: Operand, b: Operand, msg: str) -> None:
        from pydefi.vm.stdlib import encode_msg

        msg_len, msg_word = encode_msg(msg)
        self.builder.invoke(
            IRLabel("stdlib_assert_ge"),
            [a, b, IRLiteral(msg_len), IRLiteral(msg_word)],
            returns=0,
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Terminate the current block with ``STOP`` (halt, no return data)."""
        self.builder.stop()

    def return_(self, offset: ValueLike, length: ValueLike) -> None:
        """Terminate with ``RETURN(offset, length)`` — return memory slice."""
        self.builder.return_(self._to_operand(offset), self._to_operand(length))

    def return_word(self, value: ValueLike) -> None:
        """Convenience: store *value* into a fresh slot and ``RETURN`` it (32 bytes)."""
        slot = self.builder.alloca(32)
        self.builder.mstore(slot, self._to_operand(value))
        self.builder.return_(slot, IRLiteral(32))

    def revert(self, offset: ValueLike = 0, length: ValueLike = 0) -> None:
        """Terminate with ``REVERT(offset, length)`` — revert with memory slice."""
        self.builder.revert(self._to_operand(offset), self._to_operand(length))

    # ------------------------------------------------------------------
    # Build / compile
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        optimize: OptimizationLevel = OptimizationLevel.GAS,
        disable_constant_folding: bool = False,
        prefix_length: int = 0,
    ) -> bytes:
        """Compile the accumulated IR via Venom and return EVM bytecode.

        If the current basic block is not terminated, an implicit ``STOP`` is
        inserted before compilation.

        Args:
            optimize: Venom optimization level.  Defaults to ``GAS`` for
                production builds.
            disable_constant_folding: Disable SCCP, algebraic optimization,
                and assert elimination.  Useful in tests that exercise
                runtime behavior with constant inputs — without this,
                ``assert_(ctx.const(0))`` raises a Venom
                ``StaticAssertionException`` at compile time because SCCP
                proves the assertion will fail.  Real programs whose
                conditions depend on call returndata or other runtime data
                are unaffected.
            prefix_length: When the compiled program will be embedded at
                byte offset *prefix_length* of a larger bytecode buffer at
                runtime (e.g. the CCTP / OFT composer prepends a 66-byte
                ``PUSH32`` prologue before our program), every absolute
                label reference Venom resolved (``CODECOPY`` data-section
                offsets, ``jmp`` / ``jnz`` targets) is shifted by
                *prefix_length* so it stays correct.
        """
        if not self.builder.is_terminated():
            self.builder.stop()

        if disable_constant_folding:
            flags = VenomOptimizationFlags(
                level=optimize,
                disable_sccp=True,
                disable_algebraic_optimization=True,
                disable_assert_elimination=True,
                disable_branch_optimization=True,
            )
        else:
            flags = VenomOptimizationFlags(level=optimize)
        run_passes_on(self._ir_ctx, flags, disable_mem_checks=True)
        asm = generate_assembly_experimental(self._ir_ctx, optimize=optimize)
        bytecode, _ = generate_bytecode(asm)
        if prefix_length:
            bytecode = _shift_label_pushes(asm, bytecode, prefix_length)
        return bytecode

    def compile(
        self,
        flags: VenomOptimizationFlags | None = None,
    ) -> bytes:
        """Compile via VenomCompiler.generate_evm_assembly (no prefix-shift).

        Used by stdlib / abi tests that don't need the optimisation flags or
        ``prefix_length`` shift exposed by :meth:`build`.
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
    # Accessors / dunders
    # ------------------------------------------------------------------

    @property
    def ir_ctx(self) -> IRContext:
        """The underlying :class:`~vyper.venom.context.IRContext`."""
        return self._ir_ctx

    def __bytes__(self) -> bytes:
        return self.build()

    def __repr__(self) -> str:
        return f"ProgramContext(data_sections={len(self._ir_ctx.data_segment)})"


# Backwards-compat alias.  ``Program`` was previously a thin subclass that
# only added the stdlib auto-load — that's now in ProgramContext.__init__.
Program = ProgramContext

"""DeFiVM Program — SSA-style builder over Vyper's Venom IR.

A Pythonic value-flow API: methods that produce values return :class:`Value`
handles that the user threads through subsequent calls, and Venom's stack
allocator generates DUP / SWAP / POP automatically.  Quick example::

    from pydefi.vm import Program

    prog = Program()
    success = prog.call_contract(ROUTER, calldata)
    prog.assert_(success)
    prog.stop()
    bytecode = prog.build()

Patched call (the calldata pattern)::

    prog = Program()
    quote_ok = prog.call_contract(QUOTER, quote_calldata)
    prog.assert_(quote_ok)
    amount = prog.returndata_word(0)
    swap_ok = prog.call_contract(
        ROUTER,
        swap_template,
        patches={36: amount},  # write amount into the calldata at offset 36
    )
    prog.assert_(swap_ok)
    prog.stop()
    bytecode = prog.build()

Design notes
------------
``Program`` owns one :class:`vyper.venom.context.IRContext` and one
``main`` :class:`IRFunction` for its lifetime.  Each method that emits IR
appends instructions to the *current* basic block, exposed via
``self.builder.current_block()``.

**Calldata buffers.**  Static calldata for external calls is appended to a
Venom data section; the body emits ``codecopy`` from the section into a
fresh allocation, applies optional patches via ``mstore``, then ``call``.
Venom resolves the data-section label to its absolute byte position in
the compiled output, so no post-processing patches are needed.

**Termination.**  A ``Program`` is "open" until the user calls one of
:meth:`stop`, :meth:`return_`, :meth:`return_word`, :meth:`revert`, or a
control-flow primitive that terminates the current BB.  :meth:`build` adds
an implicit ``stop`` if the current BB is still open.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

from eth_contract.contract import ContractFunction
from vyper.compiler.phases import generate_bytecode
from vyper.compiler.settings import OptimizationLevel, VenomOptimizationFlags
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
from vyper.venom import generate_assembly_experimental, run_passes_on
from vyper.venom.basicblock import IRLabel, IRLiteral, IRVariable

from pydefi.vm.abiutils import abi_to_vyper
from pydefi.vm.context import ProgramContext
from pydefi.vm.stdlib import build_stdlib, encode_msg

if TYPE_CHECKING:
    from pydefi.types import Address


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: Handle to an SSA value (an EVM word produced at runtime).
#:
#: Returned by methods on :class:`Program` (e.g. :meth:`Program.const`,
#: :meth:`Program.add`, :meth:`Program.call_contract`).  Pass these handles
#: as arguments to other ``Program`` methods to build a value-flow graph
#: that Venom compiles to EVM bytecode.
#:
#: Currently an alias for Venom's operand types — an ``IRVariable`` for
#: computed values, an ``IRLiteral`` for constants.  The union is what the
#: Venom builder accepts as an operand, so no wrapper is needed.
Value = Union[IRVariable, IRLiteral]

#: Anything acceptable in a ``Value`` slot — runtime SSA handle, plain ``int``
#: literal, or 20-byte ``bytes`` address (interpreted as big-endian uint256).
ValueLike = Union[Value, int, bytes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class Placeholder:
    """Marks a runtime :class:`Value` embedded in a ``call_contract_abi``
    argument list.

    Wrap any SSA :class:`Value` that should be substituted into the encoded
    calldata at runtime::

        prog.call_contract_abi(
            token, "transfer(address,uint256)", recipient, Placeholder(amount),
        )

    Plain Python values (``int`` / ``str`` / ``bytes`` / ``bool``) are static
    and don't need wrapping.
    """

    def __init__(self, value: Value) -> None:
        self.value = value


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


def _unwrap_placeholders(arg: object) -> object:
    """Recursively replace :class:`Placeholder` with its inner :class:`Value`.

    ``ProgramContext.abi_encode`` accepts ``IROperand`` arguments directly and
    encodes them at runtime, so the wrapper is only a marker for the API.
    """
    if isinstance(arg, Placeholder):
        return arg.value
    if isinstance(arg, (tuple, list)):
        return type(arg)(_unwrap_placeholders(item) for item in arg)
    return arg


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


class Program(ProgramContext):
    """SSA-style fluent builder over Vyper's Venom IR.

    Subclasses :class:`ProgramContext`, so each instance owns one
    :class:`IRContext` and one ``main`` :class:`IRFunction`; method calls
    append IR to the current basic block.  Call :meth:`build` to compile the
    accumulated IR to EVM bytecode.

    Methods either:

    * **Produce a value** — return a :class:`Value` handle.  Examples:
      :meth:`const`, :meth:`add`, :meth:`call_contract`, :meth:`load_slot`.
    * **Have a side effect** — return ``None``.  Examples: :meth:`store_slot`,
      :meth:`assert_`, :meth:`stop`, :meth:`jump`.

    Wherever a method takes a ``ValueLike`` argument, you may pass a
    :class:`Value` returned by another method, a Python ``int`` (auto-wrapped
    as a constant), or 20 raw bytes (interpreted as a big-endian address).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        # Stdlib helpers (stdlib_revert_if, stdlib_assert_ge) are added
        # unconditionally; FunctionInlinerPass + DCE eliminate them when
        # unused, so there's no bytecode cost.
        build_stdlib(self.ir_ctx)

    # ------------------------------------------------------------------
    # Operand coercion
    # ------------------------------------------------------------------

    def _to_operand(self, v: ValueLike) -> Union[IRVariable, IRLiteral, int]:
        """Coerce a :class:`ValueLike` into something acceptable as a Venom operand.

        * :class:`Value` (``IRVariable`` / ``IRLiteral``) → returned as-is.
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
                    "use Program.const(int.from_bytes(...)) for arbitrary widths"
                )
            return int.from_bytes(bytes(v), "big")
        raise TypeError(f"operand: unsupported type {type(v).__name__}")

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def const(self, n: int) -> Value:
        """Return a :class:`Value` wrapping the constant uint256 *n*."""
        if n < 0:
            raise ValueError(f"const: value must be non-negative, got {n}")
        if n >= 1 << 256:
            raise ValueError(f"const: value {n} does not fit in uint256")
        return IRLiteral(n)

    def addr(self, a: "Address") -> Value:
        """Return a :class:`Value` for a 20-byte EVM address."""
        if len(a) != 20:
            raise ValueError(f"addr: address must be 20 bytes, got {len(a)}")
        return IRLiteral(int.from_bytes(bytes(a), "big"))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def stack_param(self) -> Value:
        """Consume a value that was already on the EVM stack when the program started.

        Used when the caller (e.g. the CCTP / OFT composer prologue in
        ``DeFiVM.sol``) pushes values onto the stack before dispatching to
        the user program.  Each call consumes one pre-existing stack slot
        in **Venom ``param`` order**: the first call returns the *deepest*
        slot (the value pushed first by the prologue), the second call
        returns the slot above it, and so on — last call returns TOS.
        Callers must therefore request ``stack_param()`` values in the same
        order the prologue pushed them.

        Lowers to Venom's ``param`` instruction which emits no bytecode —
        it's a dataflow marker telling Venom's stack allocator that the
        variable is already on the stack.  Must be called before emitting
        any other value-producing instruction in the entry basic block.
        """
        return self.builder.param()

    def add(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a + b``."""
        return self.builder.add(self._to_operand(a), self._to_operand(b))

    def sub(self, a: ValueLike, b: ValueLike) -> Value:
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

    def mul(self, a: ValueLike, b: ValueLike) -> Value:
        """Wrapping uint256 ``a * b``."""
        return self.builder.mul(self._to_operand(a), self._to_operand(b))

    def div(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a // b``; EVM DIV returns 0 when ``b == 0``."""
        return self.builder.div(self._to_operand(a), self._to_operand(b))

    def mod(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``a % b``; EVM MOD returns 0 when ``b == 0``."""
        return self.builder.mod(self._to_operand(a), self._to_operand(b))

    # ------------------------------------------------------------------
    # Comparison / boolean
    # ------------------------------------------------------------------

    def lt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a < b else 0``."""
        return self.builder.lt(self._to_operand(a), self._to_operand(b))

    def gt(self, a: ValueLike, b: ValueLike) -> Value:
        """Unsigned ``1 if a > b else 0``."""
        return self.builder.gt(self._to_operand(a), self._to_operand(b))

    def eq(self, a: ValueLike, b: ValueLike) -> Value:
        """``1 if a == b else 0``."""
        return self.builder.eq(self._to_operand(a), self._to_operand(b))

    def is_zero(self, a: ValueLike) -> Value:
        """``1 if a == 0 else 0``."""
        return self.builder.iszero(self._to_operand(a))

    # ------------------------------------------------------------------
    # Bitwise
    # ------------------------------------------------------------------

    def bit_and(self, a: ValueLike, b: ValueLike) -> Value:
        return self.builder.and_(self._to_operand(a), self._to_operand(b))

    def bit_or(self, a: ValueLike, b: ValueLike) -> Value:
        return self.builder.or_(self._to_operand(a), self._to_operand(b))

    def bit_xor(self, a: ValueLike, b: ValueLike) -> Value:
        return self.builder.xor(self._to_operand(a), self._to_operand(b))

    def bit_not(self, a: ValueLike) -> Value:
        return self.builder.not_(self._to_operand(a))

    def shl(self, value: ValueLike, shift: ValueLike) -> Value:
        """``value << shift`` (Venom signature: ``shl(bits, val)``)."""
        return self.builder.shl(self._to_operand(shift), self._to_operand(value))

    def shr(self, value: ValueLike, shift: ValueLike) -> Value:
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

    def load_slot(self, slot: IRVariable) -> Value:
        """Load the 32-byte word stored at *slot*."""
        return self.builder.mload(slot)

    def store_slot(self, slot: IRVariable, v: ValueLike) -> None:
        """Store *v* (32 bytes) into *slot*."""
        self.builder.mstore(slot, self._to_operand(v))

    # ------------------------------------------------------------------
    # Self / context
    # ------------------------------------------------------------------

    def self_addr(self) -> Value:
        """EVM ``ADDRESS`` — the running program's own address."""
        return self.builder.address()

    def gas_left(self) -> Value:
        """EVM ``GAS`` — remaining gas."""
        return self.builder.gas()

    def eth_balance(self, account: ValueLike) -> Value:
        """EVM ``BALANCE(account)`` — ETH balance of *account*."""
        return self.builder.balance(self._to_operand(account))

    def erc20_balance_of(self, token: ValueLike, account: ValueLike) -> Value:
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
    # Internal: memory + data-section helpers (Venom alloca)
    # ------------------------------------------------------------------

    def _alloc_calldata(self, calldata: bytes) -> tuple[IRVariable, int]:
        """Embed *calldata* in a data section and return ``(base, len)``.

        Thin wrapper around :meth:`ProgramContext.embed_and_load` that returns
        the unpadded byte length alongside the buffer pointer (the value to
        pass as ``argsLen`` to ``CALL``).
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
        calldata: bytes,
        *,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
        patches: Mapping[int, ValueLike] | None = None,
    ) -> Value:
        """Emit a CALL with static calldata and optional runtime patches.

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
            A :class:`Value` holding the CALL success flag (1 on success,
            0 on failure).  Use :meth:`assert_` to revert on failure.
        """
        base_fp, blen = self._alloc_calldata(calldata)

        if patches:
            for offset, val in patches.items():
                if not 0 <= offset <= blen - 32:
                    raise ValueError(
                        f"call_contract: patch offset {offset} out of bounds "
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

    def call_contract_abi(
        self,
        to: ValueLike,
        abi_sig: str,
        *args: object,
        value: ValueLike = 0,
        gas: ValueLike | None = None,
    ) -> Value:
        """Emit a CALL with calldata built from a human-readable ABI signature.

        Plain Python values (``int``, ``str`` address, ``bool``, ``bytes``,
        nested ``tuple``/``list``) are encoded as constants.  Wrap any runtime
        :class:`Value` in :class:`Placeholder` to have it embedded into the
        encoded calldata at runtime.

        The ``function`` keyword in *abi_sig* is optional; both bare
        ``"transfer(address,uint256)"`` and qualified ``"function transfer(...)"``
        forms are accepted.

        Returns:
            A :class:`Value` holding the CALL success flag.
        """
        normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
        fn = ContractFunction.from_abi(normalised)
        param_types = [abi_to_vyper(t) for t in fn.input_types]

        if len(args) != len(param_types):
            raise ValueError(
                f"call_contract_abi: expected {len(param_types)} argument(s) for signature {abi_sig!r}, got {len(args)}"
            )

        # ProgramContext.abi_encode handles plain Python values and IROperands
        # uniformly — Placeholder is just a marker we strip here.
        encoded_args = [_unwrap_placeholders(a) for a in args]
        buf_val = self.abi_encode(encoded_args, param_types, method_id=bytes(fn.selector))

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

    # ------------------------------------------------------------------
    # Returndata
    # ------------------------------------------------------------------

    def returndata_word(self, offset: int = 0) -> Value:
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
        if msg:
            self._invoke_assert_ge(a, b, msg)
            return
        not_lt = self.builder.iszero(self.builder.lt(self._to_operand(a), self._to_operand(b)))
        self.builder.assert_(not_lt)

    def assert_le(self, a: ValueLike, b: ValueLike, msg: str = "") -> None:
        """Revert if ``a > b`` (i.e. require ``a <= b``) — ``assert_ge(b, a)``."""
        self.assert_ge(b, a, msg)

    def _invoke_revert_if(self, cond: ValueLike, msg: str) -> None:
        msg_len, msg_word = encode_msg(msg)
        self.builder.invoke(
            IRLabel("stdlib_revert_if"),
            [self._to_operand(cond), IRLiteral(msg_len), IRLiteral(msg_word)],
            returns=0,
        )

    def _invoke_assert_ge(self, a: ValueLike, b: ValueLike, msg: str) -> None:
        msg_len, msg_word = encode_msg(msg)
        self.builder.invoke(
            IRLabel("stdlib_assert_ge"),
            [self._to_operand(a), self._to_operand(b), IRLiteral(msg_len), IRLiteral(msg_word)],
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
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        optimize: OptimizationLevel = OptimizationLevel.GAS,  # type: ignore[valid-type]
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
                ``assert_(prog.const(0))`` raises a Venom
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
            flags = VenomOptimizationFlags(  # type: ignore[call-arg]
                level=optimize,
                disable_sccp=True,
                disable_algebraic_optimization=True,
                disable_assert_elimination=True,
                disable_branch_optimization=True,
            )
        else:
            flags = VenomOptimizationFlags(level=optimize)  # type: ignore[call-arg]
        run_passes_on(self.ir_ctx, flags, disable_mem_checks=True)  # type: ignore[misc]
        asm = generate_assembly_experimental(self.ir_ctx, optimize=optimize)  # type: ignore[misc]
        bytecode, _ = generate_bytecode(asm)  # type: ignore[misc]
        if prefix_length:
            bytecode = _shift_label_pushes(asm, bytecode, prefix_length)
        return bytecode

    # ------------------------------------------------------------------
    # Convenience dunders
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        return self.build()

    def __repr__(self) -> str:
        return f"Program(data_sections={len(self.ir_ctx.data_segment)})"

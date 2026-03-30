"""DeFiVM fluent program builder.

:class:`Program` provides a method-chaining interface over the low-level
instruction builders in :mod:`pydefi.vm.program`.  It adds three higher-level
features that are awkward with the raw byte-concatenation approach:

1. **Label-based jumps** — define named positions with :meth:`label` and
   reference them in :meth:`jump` / :meth:`jumpi` without computing byte
   offsets by hand.  Labels are resolved when :meth:`build` is called.

2. **``call_contract`` helper** — wraps the four-item stack protocol required
   by the ``CALL`` opcode into a single method call.

3. **Program composition** — combine independent sub-programs with
   :meth:`extend` / ``+`` / ``+=`` or :meth:`compose`.

4. **Calldata surgery** — :meth:`call_with_patches` embeds runtime values
   (static, from returndata, or from a register) into a calldata template
   before dispatching the external call.

Basic usage::

    from pydefi.vm import Program
    from pydefi.vm.abi import erc20_approve

    ROUTER  = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    TOKEN   = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    AMOUNT  = 10 ** 18

    bytecode = (
        Program()
        # approve router to spend tokens
        .call_contract(TOKEN, erc20_approve(ROUTER, AMOUNT))
        # swap (pre-built calldata)
        .call_contract(ROUTER, swap_calldata, value=0, gas=0)
        # check minimum output
        .push_addr(RECIPIENT)
        .push_addr(TOKEN)
        .push_u256(MIN_OUT)
        .assert_ge("slippage: amount_out too low")
        .build()
    )

Label-based conditional example::

    bytecode = (
        Program()
        .push_u256(condition_value)
        .jumpi("skip")          # jump if condition != 0
        .push_bytes(calldata_a)
        .push_u256(0).push_addr(CONTRACT_A).push_u256(0)
        .call()
        .pop()
        .label("skip")
        .build()
    )

Composition example::

    approve = Program().call_contract(TOKEN, erc20_approve(ROUTER, MAX_U256)).pop()
    swap    = Program().call_contract(ROUTER, swap_calldata).pop()

    full = approve + swap            # returns a new Program
    # or: approve.extend(swap)       # in-place
    # or: Program.compose([approve, swap])

Calldata surgery example — embed amount from last returndata::

    # double_sel(5) → 10; patch that into double_sel(0) template → double_sel(10) → 20
    bytecode = (
        Program()
        .call_contract(ADAPTER, double_calldata)
        .pop()
        .call_with_patches(
            ADAPTER,
            template_calldata,               # double(0) placeholder template
            patches=[
                ("u256", 4, ("ret_u256", 0)),  # offset 4, value from last returndata[0:32]
            ],
        )
        .pop()
        .build()
    )

Calldata surgery with a register source::

    # Amount was saved to reg 0 earlier in the program
    bytecode = (
        Program()
        .store_reg(0)                        # save amount from stack top
        .call_with_patches(
            ROUTER,
            swap_template,
            patches=[
                ("u256", 36, ("reg", 0)),    # offset 36, value from register 0
            ],
        )
        .pop()
        .build()
    )
"""

from __future__ import annotations

import struct

from pydefi.vm.program import (
    OP_JUMP,
    OP_JUMPI,
    assert_ge,
    assert_le,
    balance_of,
    call,
    dup,
    jump,
    jumpi,
    load_reg,
    patch_addr,
    patch_u256,
    pop,
    push_addr,
    push_bytes,
    push_u256,
    ret_slice,
    ret_u256,
    revert_if,
    self_addr,
    store_reg,
    sub,
    swap,
)

# ---------------------------------------------------------------------------
# Patch source types
# ---------------------------------------------------------------------------

#: A *patch source* describes where the runtime value for a calldata field comes
#: from.  Supported forms:
#:
#: ``int``
#:     Static ``uint256`` value embedded directly into the program bytecode.
#:
#: ``str``
#:     Static Ethereum address (hex with ``0x`` prefix).
#:
#: ``("ret_u256", offset)``
#:     ``uint256`` read from the last external call's returndata at ``offset``.
#:     Emits a ``RET_U256 <offset>`` instruction before the patch.
#:
#: ``("reg", reg_idx)``
#:     Value loaded from VM register *reg_idx*.  Emits ``LOAD_REG <reg_idx>``.
#:     Works for both ``"u256"`` and ``"addr"`` patch kinds.
PatchSource = int | str | tuple[str, int]

#: A single patch descriptor: ``(kind, calldata_offset, source)`` where:
#:
#: - *kind*: ``"u256"`` (patch a 32-byte word) or ``"addr"`` (patch a 20-byte address).
#: - *calldata_offset*: byte offset inside the calldata template to overwrite.
#: - *source*: a :data:`PatchSource`.
PatchSpec = tuple[str, int, PatchSource]

# ---------------------------------------------------------------------------
# Program builder
# ---------------------------------------------------------------------------


class Program:
    """Fluent DeFiVM bytecode builder with label support.

    All instruction methods return ``self`` so calls can be chained.
    Call :meth:`build` at the end to obtain the final ``bytes`` bytecode.
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str]] = []  # (u16 offset in _buf, label name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, data: bytes) -> "Program":
        self._buf.extend(data)
        return self

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def label(self, name: str) -> "Program":
        """Mark the current program position with *name*.

        Use the same name as the target in :meth:`jump` or :meth:`jumpi`
        to create a labelled branch without computing byte offsets.

        Raises :exc:`ValueError` if the label has already been defined.
        """
        if name in self._labels:
            raise ValueError(f"Program: duplicate label {name!r}")
        self._labels[name] = len(self._buf)
        return self

    # ------------------------------------------------------------------
    # Stack / register instructions
    # ------------------------------------------------------------------

    def push_u256(self, n: int) -> "Program":
        """Emit PUSH_U256."""
        return self._emit(push_u256(n))

    def push_addr(self, a: str) -> "Program":
        """Emit PUSH_ADDR."""
        return self._emit(push_addr(a))

    def push_bytes(self, data: bytes) -> "Program":
        """Emit PUSH_BYTES."""
        return self._emit(push_bytes(data))

    def dup(self) -> "Program":
        """Emit DUP."""
        return self._emit(dup())

    def swap(self) -> "Program":
        """Emit SWAP."""
        return self._emit(swap())

    def pop(self) -> "Program":
        """Emit POP."""
        return self._emit(pop())

    def load_reg(self, i: int) -> "Program":
        """Emit LOAD_REG *i*."""
        return self._emit(load_reg(i))

    def store_reg(self, i: int) -> "Program":
        """Emit STORE_REG *i*."""
        return self._emit(store_reg(i))

    # ------------------------------------------------------------------
    # Control flow instructions
    # ------------------------------------------------------------------

    def jump(self, target: str | int) -> "Program":
        """Emit JUMP.

        *target* may be either a raw byte offset (``int``) or a label name
        (``str``).  Label references are resolved at :meth:`build` time.
        """
        if isinstance(target, int):
            return self._emit(jump(target))
        self._buf.append(OP_JUMP)
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")
        return self

    def jumpi(self, target: str | int) -> "Program":
        """Emit JUMPI.

        *target* may be a raw byte offset (``int``) or a label name (``str``).
        JUMPI pops the condition from the top of the stack and jumps if it is
        non-zero.
        """
        if isinstance(target, int):
            return self._emit(jumpi(target))
        self._buf.append(OP_JUMPI)
        self._fixups.append((len(self._buf), target))
        self._buf.extend(b"\x00\x00")
        return self

    def revert_if(self, msg: str) -> "Program":
        """Emit REVERT_IF with message *msg*."""
        return self._emit(revert_if(msg))

    def assert_ge(self, msg: str = "") -> "Program":
        """Emit ASSERT_GE — revert if top-of-stack ``a < b``."""
        return self._emit(assert_ge(msg))

    def assert_le(self, msg: str = "") -> "Program":
        """Emit ASSERT_LE — revert if top-of-stack ``a > b``."""
        return self._emit(assert_le(msg))

    # ------------------------------------------------------------------
    # External / introspection instructions
    # ------------------------------------------------------------------

    def call(self, require_success: bool = True) -> "Program":
        """Emit CALL.

        The caller must have pushed (top to bottom):
        ``gasLimit``, ``to``, ``value``, ``calldataBufIdx``.
        """
        return self._emit(call(require_success))

    def balance_of(self) -> "Program":
        """Emit BALANCE_OF — pop ``token``, ``account``; push ERC-20 balance."""
        return self._emit(balance_of())

    def self_addr(self) -> "Program":
        """Emit SELF_ADDR — push the VM contract's own address."""
        return self._emit(self_addr())

    def sub(self) -> "Program":
        """Emit SUB — pop ``a`` (top), ``b``; push ``a - b`` (saturates to 0)."""
        return self._emit(sub())

    # ------------------------------------------------------------------
    # ABI / data instructions
    # ------------------------------------------------------------------

    def patch_u256(self, offset: int) -> "Program":
        """Emit PATCH_U256 at *offset*."""
        return self._emit(patch_u256(offset))

    def patch_addr(self, offset: int) -> "Program":
        """Emit PATCH_ADDR at *offset*."""
        return self._emit(patch_addr(offset))

    def ret_u256(self, offset: int) -> "Program":
        """Emit RET_U256 — push uint256 from last returndata at *offset*."""
        return self._emit(ret_u256(offset))

    def ret_slice(self, offset: int, length: int) -> "Program":
        """Emit RET_SLICE — push bytes slice from last returndata."""
        return self._emit(ret_slice(offset, length))

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def call_contract(
        self,
        to: str,
        calldata: bytes,
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a complete external-call sequence for a pre-built calldata buffer.

        This is a convenience wrapper that pushes the four items required by
        the ``CALL`` opcode in the correct stack order::

            push_bytes(calldata)   # calldataBufIdx (bottom)
            push_u256(value)
            push_addr(to)
            push_u256(gas)         # gasLimit (top)
            CALL

        Args:
            to: Target contract address (checksummed or lowercase hex).
            calldata: Pre-encoded ABI calldata (use :mod:`pydefi.vm.abi` helpers).
            value: ETH value to forward with the call (wei), default 0.
            gas: Gas limit for the sub-call (0 = forward all remaining gas).
            require_success: If ``True`` (default), revert if the sub-call fails.

        Returns:
            ``self`` for chaining.
        """
        return (
            self._emit(push_bytes(calldata))
            ._emit(push_u256(value))
            ._emit(push_addr(to))
            ._emit(push_u256(gas))
            ._emit(call(require_success))
        )

    def call_with_patches(
        self,
        to: str,
        calldata: bytes,
        patches: list[PatchSpec],
        *,
        value: int = 0,
        gas: int = 0,
        require_success: bool = True,
    ) -> "Program":
        """Emit a patched external call — embed runtime values into a calldata template.

        This is the **calldata surgery** helper.  It pushes a mutable copy of
        *calldata* as a buffer, applies each patch from *patches* (each one
        overwrites a field at a specific byte offset using a value obtained at
        runtime), then issues the ``CALL`` opcode.

        Each entry in *patches* is a 3-tuple ``(kind, offset, source)``:

        - *kind* — ``"u256"`` to overwrite a 32-byte word, ``"addr"`` for 20 bytes.
        - *offset* — byte offset in the calldata template to overwrite.
        - *source* — where the value comes from at runtime:

          - ``int`` — static ``uint256`` literal (only valid for ``kind="u256"``).
          - ``str`` — static address hex string (only valid for ``kind="addr"``).
          - ``("ret_u256", retdata_offset)`` — ``uint256`` from the last call's
            returndata at *retdata_offset*.
          - ``("reg", reg_idx)`` — value from VM register *reg_idx*.

        Stack contract — the stack must be clean (no leftover items from the
        current patch value) when each patch instruction runs.  This is
        automatically satisfied when all sources are static, from returndata, or
        from registers.

        Example::

            # Embed the output of a previous call (from returndata) as amountIn
            program = (
                Program()
                .call_contract(QUOTER, quote_calldata)
                .pop()
                .call_with_patches(
                    ROUTER,
                    swap_template,          # swap(0, ...) — amount placeholder at offset 36
                    patches=[
                        ("u256", 36, ("ret_u256", 0)),  # fill amount from last retdata
                    ],
                )
                .pop()
                .build()
            )

        Args:
            to: Target contract address.
            calldata: Mutable calldata template bytes.
            patches: List of ``(kind, offset, source)`` patch descriptors.
            value: ETH value to forward (wei), default 0.
            gas: Sub-call gas limit (0 = forward all remaining gas).
            require_success: Revert if the sub-call fails (default ``True``).

        Returns:
            ``self`` for chaining.
        """
        self._emit(push_bytes(calldata))   # [bufIdx]

        for kind, offset, source in patches:
            if kind not in ("u256", "addr"):
                raise ValueError(
                    f"call_with_patches: unknown patch kind {kind!r}; expected 'u256' or 'addr'"
                )

            if isinstance(source, tuple):
                src_type = source[0]
                if src_type == "ret_u256":
                    retdata_offset = source[1]
                    self._emit(ret_u256(retdata_offset))
                elif src_type == "reg":
                    reg_idx = source[1]
                    self._emit(load_reg(reg_idx))
                else:
                    raise ValueError(
                        f"call_with_patches: unknown source type {src_type!r}; "
                        "expected 'ret_u256' or 'reg'"
                    )
            elif isinstance(source, int):
                if kind != "u256":
                    raise ValueError(
                        f"call_with_patches: int source requires kind='u256', got {kind!r}"
                    )
                self._emit(push_u256(source))
            elif isinstance(source, str):
                if kind != "addr":
                    raise ValueError(
                        f"call_with_patches: str source requires kind='addr', got {kind!r}"
                    )
                self._emit(push_addr(source))
            else:
                raise TypeError(
                    f"call_with_patches: unsupported source type {type(source).__name__!r}"
                )

            if kind == "u256":
                self._emit(patch_u256(offset))
            else:  # kind == "addr"
                self._emit(patch_addr(offset))

        # Stack now: [bufIdx] — ready for CALL prologue
        self._emit(push_u256(value))
        self._emit(push_addr(to))
        self._emit(push_u256(gas))
        self._emit(call(require_success))
        return self

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def extend(self, other: "Program") -> "Program":
        """Append *other*'s instructions to this program **in-place**.

        All byte offsets in *other*'s labels and fixup table are adjusted by
        the current length of ``self`` so that label references remain correct
        after merging.

        Raises :exc:`ValueError` if *other* defines a label that already exists
        in ``self``.

        Returns:
            ``self`` for chaining.
        """
        offset = len(self._buf)
        self._buf.extend(other._buf)
        for name, pos in other._labels.items():
            if name in self._labels:
                raise ValueError(
                    f"Program: duplicate label {name!r} during extend"
                )
            self._labels[name] = pos + offset
        for fixup_off, name in other._fixups:
            self._fixups.append((fixup_off + offset, name))
        return self

    def __add__(self, other: "Program") -> "Program":
        """Return a new :class:`Program` that concatenates *self* and *other*.

        Neither ``self`` nor ``other`` is modified.

        Raises :exc:`ValueError` on duplicate label names.
        """
        result = Program()
        result._buf.extend(self._buf)
        result._labels.update(self._labels)
        result._fixups.extend(self._fixups)
        result.extend(other)
        return result

    def __iadd__(self, other: "Program") -> "Program":
        """Extend this program in-place (``self += other``)."""
        return self.extend(other)

    @classmethod
    def compose(cls, programs: list["Program"]) -> "Program":
        """Compose a sequence of programs into a single :class:`Program`.

        Equivalent to reducing the list with ``+``, but more efficient for
        large numbers of sub-programs.

        Example::

            parts = [approve_prog, wrap_prog, swap_prog, unwrap_prog]
            bytecode = Program.compose(parts).build()

        Raises :exc:`ValueError` on duplicate label names across sub-programs.
        """
        result = cls()
        for prog in programs:
            result.extend(prog)
        return result

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> bytes:
        """Resolve label fixups and return the final bytecode.

        Raises :exc:`ValueError` if any label referenced in a jump has not
        been defined.
        """
        buf = bytearray(self._buf)
        for fixup_offset, name in self._fixups:
            if name not in self._labels:
                raise ValueError(f"Program: undefined label {name!r}")
            target = self._labels[name]
            struct.pack_into(">H", buf, fixup_offset, target)
        return bytes(buf)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        """Allow ``bytes(program)`` as an alias for ``program.build()``."""
        return self.build()

    def __len__(self) -> int:
        """Return the current (unresolved) byte length of the program."""
        return len(self._buf)

    def __repr__(self) -> str:
        return f"Program(len={len(self._buf)}, labels={list(self._labels)!r})"

"""DeFiVM fluent program builder.

:class:`Program` provides a method-chaining interface over the low-level
instruction builders in :mod:`pydefi.vm.program`.  It adds two higher-level
features that are awkward with the raw byte-concatenation approach:

1. **Label-based jumps** — define named positions with :meth:`label` and
   reference them in :meth:`jump` / :meth:`jumpi` without computing byte
   offsets by hand.  Labels are resolved when :meth:`build` is called.

2. **``call_contract`` helper** — wraps the four-item stack protocol required
   by the ``CALL`` opcode into a single method call.

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

"""Venom IR stdlib — utility functions built with :class:`ProgramContext`.

This module provides factory functions to build named venom IR functions
(``stdlib_revert_if``, ``stdlib_assert_ge``) into a shared
:class:`~vyper.venom.context.IRContext` using :class:`~pydefi.vm.context.ProgramContext`.

Stdlib functions
----------------

``stdlib_revert_if(cond, msg_len, msg_word, ret_pc)``
    Conditionally reverts with a 100-byte ``Error(string)`` ABI payload when
    *cond* is non-zero, otherwise returns to the caller.

``stdlib_assert_ge(a, b, msg_len, msg_word, ret_pc)``
    Reverts with ``Error(string)`` when ``a < b``, otherwise returns.
"""

from __future__ import annotations

from collections.abc import Callable

from vyper.venom.basicblock import IRLabel, IRLiteral
from vyper.venom.context import IRContext

from pydefi.vm.context import ProgramContext

__all__ = ["build_stdlib", "encode_msg"]

# keccak256("Error(string)")[:4] stored as a full 32-byte MSTORE word
_ERROR_SELECTOR_WORD: int = 0x08C379A000000000000000000000000000000000000000000000000000000000

# Maximum encoded message length (must fit in a single EVM word)
_MAX_MSG_BYTES: int = 32


def encode_msg(msg: str) -> tuple[int, int]:
    """Encode a UTF-8 string into ``(msg_len, msg_word)`` for the stdlib functions.

    Args:
        msg: Error message string.  Must be ≤ 32 bytes when UTF-8 encoded.

    Returns:
        ``(msg_len, msg_word)`` — the byte length and the left-justified 32-byte
        word representation, both as Python :class:`int` suitable for
        :class:`~vyper.venom.basicblock.IRLiteral`.

    Raises:
        ValueError: If *msg* encodes to more than 32 bytes.
    """
    raw = msg.encode("utf-8")
    if len(raw) > _MAX_MSG_BYTES:
        raise ValueError(f"message too long ({len(raw)} bytes, max {_MAX_MSG_BYTES})")
    msg_len = len(raw)
    msg_word = int.from_bytes(raw.ljust(_MAX_MSG_BYTES, b"\x00"), "big")
    return msg_len, msg_word


def _build_revert_if(ctx: ProgramContext) -> None:
    b = ctx.builder
    cond = b.param()
    msg_len = b.param()
    msg_word = b.param()
    ret_pc = b.param()

    bb_revert = b.create_block("revert")
    bb_ok = b.create_block("ok")
    b.jnz(cond, bb_revert.label, bb_ok.label)

    # --- revert path ---
    b.append_block(bb_revert)
    b.set_block(bb_revert)
    buf = b.alloca(100)
    b.mstore(buf, IRLiteral(_ERROR_SELECTOR_WORD))
    ptr4 = b.add(buf, IRLiteral(4))
    b.mstore(ptr4, IRLiteral(32))
    ptr36 = b.add(buf, IRLiteral(36))
    b.mstore(ptr36, msg_len)
    ptr68 = b.add(buf, IRLiteral(68))
    b.mstore(ptr68, msg_word)
    b.revert(buf, IRLiteral(100))

    # --- ok path ---
    b.append_block(bb_ok)
    b.set_block(bb_ok)
    b.ret(ret_pc)


def _build_assert_ge(ctx: ProgramContext) -> None:
    b = ctx.builder
    a_val = b.param()
    b_val = b.param()
    msg_len = b.param()
    msg_word = b.param()
    ret_pc = b.param()

    cond = b.lt(a_val, b_val)
    b.invoke(IRLabel("stdlib_revert_if"), [cond, msg_len, msg_word], returns=0)
    b.ret(ret_pc)


#: Registry of built-in functions.
STDLIB_FUNCTIONS: dict[str, Callable[[ProgramContext], None]] = {
    "stdlib_revert_if": _build_revert_if,
    "stdlib_assert_ge": _build_assert_ge,
}


def build_stdlib(ir_ctx: IRContext) -> None:
    for name, builder in STDLIB_FUNCTIONS.items():
        ctx = ProgramContext(ir_ctx, name, set_entry=False)
        builder(ctx)

"""Venom IR stdlib — utility functions defined as proper IR functions.

This module builds a pre-compiled ``STDLIB`` :class:`~pydefi.vm.venom.ModuleBuilder`
containing ``stdlib_revert_if`` and ``stdlib_assert_ge`` as named venom IR functions.

Rather than inlining error-handling logic directly into the calling module, callers
**invoke the functions by label** after merging the stdlib context.  The
:class:`~vyper.venom.passes.function_inliner.FunctionInlinerPass` automatically
inlines them at each call site during compilation, so there is no runtime call
overhead.

Usage pattern
-------------

::

    from pydefi.vm.venom import ModuleBuilder
    from pydefi.vm.stdlib import STDLIB, encode_msg
    from vyper.venom.basicblock import IRLabel, IRLiteral

    mod = ModuleBuilder("example")
    amount = mod.calldataload(IRLiteral(4))

    # Conditional revert:
    is_zero = mod.iszero(amount)
    msg_len, msg_word = encode_msg("amount is zero")
    mod.invoke(
        IRLabel("stdlib_revert_if"),
        [is_zero, IRLiteral(msg_len), IRLiteral(msg_word)],
        returns=0,
    )

    # Assertion:
    msg_len2, msg_word2 = encode_msg("amount too small")
    mod.invoke(
        IRLabel("stdlib_assert_ge"),
        [amount, IRLiteral(1000), IRLiteral(msg_len2), IRLiteral(msg_word2)],
        returns=0,
    )

    mod.stop()
    mod.merge(STDLIB.ctx)
    bytecode = mod.compile()

Stdlib functions
----------------

``stdlib_revert_if(cond, msg_len, msg_word, ret_pc)``
    Conditionally reverts with a 100-byte ``Error(string)`` ABI payload when
    *cond* is non-zero, otherwise returns to the caller.

``stdlib_assert_ge(a, b, msg_len, msg_word, ret_pc)``
    Reverts with ``Error(string)`` when ``a < b``, otherwise returns.

Error encoding
--------------

Reverts use the standard Solidity ``Error(string)`` ABI encoding (selector
``0x08c379a0``)::

    [4 bytes]  0x08c379a0            — keccak256("Error(string)")[:4]
    [32 bytes] 0x00..0020            — ABI offset = 32
    [32 bytes] 0x00..000N            — string length N
    [32 bytes] <string bytes>        — UTF-8 data, zero-padded on the right

Total revert data: 100 bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vyper.venom.basicblock import IRLabel, IRLiteral

if TYPE_CHECKING:
    from pydefi.vm.venom import ModuleBuilder

__all__ = ["STDLIB", "encode_msg"]

# keccak256("Error(string)")[:4] stored as a full 32-byte MSTORE word
# (4-byte selector in the high bits, 28 zero bytes in the low bits)
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

    Example::

        msg_len, msg_word = encode_msg("amount is zero")
        mod.invoke(
            IRLabel("stdlib_revert_if"),
            [cond, IRLiteral(msg_len), IRLiteral(msg_word)],
            returns=0,
        )
        mod.merge(STDLIB.ctx)
    """
    raw = msg.encode("utf-8")
    if len(raw) > _MAX_MSG_BYTES:
        raise ValueError(f"message too long ({len(raw)} bytes, max {_MAX_MSG_BYTES})")
    msg_len = len(raw)
    msg_word = int.from_bytes(raw.ljust(_MAX_MSG_BYTES, b"\x00"), "big")
    return msg_len, msg_word


def _build_revert_if(mod: ModuleBuilder) -> None:
    """Add ``stdlib_revert_if(cond, msg_len, msg_word, ret_pc)`` to *mod*."""
    fn = mod.create_function("revert_if")
    mod.set_block(fn.entry)
    cond = mod.param()
    msg_len = mod.param()
    msg_word = mod.param()
    ret_pc = mod.param()

    bb_revert = mod.create_block("revert")
    bb_ok = mod.create_block("ok")
    mod.jnz(cond, bb_revert.label, bb_ok.label)

    # --- revert path ---
    mod.append_block(bb_revert)
    mod.set_block(bb_revert)
    buf = mod.alloca(100)
    mod.mstore(buf, IRLiteral(_ERROR_SELECTOR_WORD))
    ptr4 = mod.add(buf, IRLiteral(4))
    mod.mstore(ptr4, IRLiteral(32))
    ptr36 = mod.add(buf, IRLiteral(36))
    mod.mstore(ptr36, msg_len)
    ptr68 = mod.add(buf, IRLiteral(68))
    mod.mstore(ptr68, msg_word)
    mod.revert(buf, IRLiteral(100))

    # --- ok path ---
    mod.append_block(bb_ok)
    mod.set_block(bb_ok)
    mod.ret(ret_pc)


def _build_assert_ge(mod: ModuleBuilder) -> None:
    """Add ``stdlib_assert_ge(a, b, msg_len, msg_word, ret_pc)`` to *mod*."""
    fn = mod.create_function("assert_ge")
    mod.set_block(fn.entry)
    a = mod.param()
    b = mod.param()
    msg_len = mod.param()
    msg_word = mod.param()
    ret_pc = mod.param()

    # Assertion fails when a < b (lt returns 1); delegate revert to revert_if.
    cond = mod.lt(a, b)
    mod.invoke(IRLabel("stdlib_revert_if"), [cond, msg_len, msg_word], returns=0)
    mod.ret(ret_pc)


def _build_stdlib() -> ModuleBuilder:
    """Build the stdlib ModuleBuilder with revert_if and assert_ge IR functions.

    Returns a fresh :class:`~pydefi.vm.venom.ModuleBuilder` whose context
    contains two named functions:

    * ``stdlib_revert_if`` — conditional revert with ``Error(string)`` encoding
    * ``stdlib_assert_ge`` — revert with ``Error(string)`` when ``a < b``

    The entry function (``stdlib_main``) contains only a ``stop`` instruction.
    """
    from pydefi.vm.venom import ModuleBuilder

    mod = ModuleBuilder("stdlib")
    mod.stop()  # terminate the auto-created stdlib_main entry function
    _build_revert_if(mod)
    _build_assert_ge(mod)
    del mod.ctx.functions[mod.named_label("main")]
    mod.ctx.entry_function = None
    return mod


#: Pre-built stdlib module.  Merge ``STDLIB.ctx`` into your builder before
#: :meth:`~pydefi.vm.venom.ModuleBuilder.compile` to make the stdlib functions
#: available for inlining.
STDLIB: ModuleBuilder = _build_stdlib()

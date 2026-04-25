"""Venom IR stdlib — utility functions defined as proper IR functions.

This module builds a pre-compiled ``STDLIB`` :class:`~pydefi.vm.venom.ModuleBuilder`
containing ``stdlib.revert_if`` and ``stdlib.assert_ge`` as named venom IR functions.

Rather than inlining error-handling logic directly into the calling module, callers
**invoke the functions by label** after merging the stdlib context.  The
:class:`~vyper.venom.passes.function_inliner.FunctionInlinerPass` automatically
inlines them at each call site during compilation, so there is no runtime call
overhead.

Usage pattern
-------------

::

    from pydefi.vm.venom import ModuleBuilder
    from vyper.venom.basicblock import IRLiteral

    mod = ModuleBuilder("example")
    amount = mod.calldataload(IRLiteral(4))

    # Via the convenience methods (auto-merges stdlib):
    is_zero = mod.iszero(amount)
    mod.revert_if(is_zero, "amount is zero")
    mod.assert_ge(amount, IRLiteral(1000), "amount too small")

    mod.return_(mod.alloca(32), IRLiteral(32))
    bytecode = mod.compile()

    # Or explicitly, using the stdlib module:
    from pydefi.vm.stdlib import STDLIB, revert_if, assert_ge
    mod2 = ModuleBuilder("example2")
    revert_if(mod2, IRLiteral(1), "always fails")
    mod2.return_(mod2.alloca(32), IRLiteral(32))
    mod2.merge(STDLIB.ctx)
    bytecode2 = mod2.compile()

Stdlib functions
----------------

``stdlib.revert_if(cond, msg_len, msg_word, ret_pc)``
    Conditionally reverts with a 100-byte ``Error(string)`` ABI payload when
    *cond* is non-zero, otherwise returns to the caller.

``stdlib.assert_ge(a, b, msg_len, msg_word, ret_pc)``
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

from vyper.venom.basicblock import IRLabel, IRLiteral
from vyper.venom.builder import VenomBuilder

__all__ = ["STDLIB", "revert_if", "assert_ge"]

# keccak256("Error(string)")[:4] stored as a full 32-byte MSTORE word
# (4-byte selector in the high bits, 28 zero bytes in the low bits)
_ERROR_SELECTOR_WORD: int = (
    0x08C379A0_00000000_00000000_00000000_00000000_00000000_00000000_00000000
)

# Maximum encoded message length (must fit in a single EVM word)
_MAX_MSG_BYTES: int = 32


def _build_stdlib() -> "ModuleBuilder":
    """Build the stdlib ModuleBuilder with revert_if and assert_ge IR functions.

    Returns a fresh :class:`~pydefi.vm.venom.ModuleBuilder` whose context
    contains two named functions:

    * ``stdlib.revert_if`` — conditional revert with ``Error(string)`` encoding
    * ``stdlib.assert_ge`` — revert with ``Error(string)`` when ``a < b``

    The entry function (``stdlib.main``) contains only a ``stop`` instruction.
    """
    from pydefi.vm.venom import ModuleBuilder

    mod = ModuleBuilder("stdlib")
    mod.stop()  # terminate the auto-created stdlib.main entry function

    # ------------------------------------------------------------------
    # stdlib.revert_if(cond, msg_len, msg_word, ret_pc)
    # ------------------------------------------------------------------
    fn_ri = mod.create_function("revert_if")
    mod.set_block(fn_ri.entry)
    ri_cond = mod.param()
    ri_msg_len = mod.param()
    ri_msg_word = mod.param()
    ri_ret_pc = mod.param()

    bb_revert = mod.create_block("revert")
    bb_ok = mod.create_block("ok")
    mod.jnz(ri_cond, bb_revert.label, bb_ok.label)

    # --- revert path ---
    mod.append_block(bb_revert)
    mod.set_block(bb_revert)
    buf = mod.alloca(100)
    mod.mstore(buf, IRLiteral(_ERROR_SELECTOR_WORD))
    ptr4 = mod.add(buf, IRLiteral(4))
    mod.mstore(ptr4, IRLiteral(32))
    ptr36 = mod.add(buf, IRLiteral(36))
    mod.mstore(ptr36, ri_msg_len)
    ptr68 = mod.add(buf, IRLiteral(68))
    mod.mstore(ptr68, ri_msg_word)
    mod.revert(buf, IRLiteral(100))

    # --- ok path ---
    mod.append_block(bb_ok)
    mod.set_block(bb_ok)
    mod.ret(ri_ret_pc)

    # ------------------------------------------------------------------
    # stdlib.assert_ge(a, b, msg_len, msg_word, ret_pc)
    # ------------------------------------------------------------------
    fn_age = mod.create_function("assert_ge")
    mod.set_block(fn_age.entry)
    age_a = mod.param()
    age_b = mod.param()
    age_msg_len = mod.param()
    age_msg_word = mod.param()
    age_ret_pc = mod.param()

    # Assertion fails when a < b (lt returns 1).
    age_cond = mod.lt(age_a, age_b)

    bb_revert2 = mod.create_block("revert")
    bb_ok2 = mod.create_block("ok")
    mod.jnz(age_cond, bb_revert2.label, bb_ok2.label)

    # --- revert path ---
    mod.append_block(bb_revert2)
    mod.set_block(bb_revert2)
    buf2 = mod.alloca(100)
    mod.mstore(buf2, IRLiteral(_ERROR_SELECTOR_WORD))
    ptr4b = mod.add(buf2, IRLiteral(4))
    mod.mstore(ptr4b, IRLiteral(32))
    ptr36b = mod.add(buf2, IRLiteral(36))
    mod.mstore(ptr36b, age_msg_len)
    ptr68b = mod.add(buf2, IRLiteral(68))
    mod.mstore(ptr68b, age_msg_word)
    mod.revert(buf2, IRLiteral(100))

    # --- ok path ---
    mod.append_block(bb_ok2)
    mod.set_block(bb_ok2)
    mod.ret(age_ret_pc)

    return mod


def _encode_msg(msg: str) -> tuple[int, int]:
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
        raise ValueError(
            f"message too long ({len(raw)} bytes, max {_MAX_MSG_BYTES})"
        )
    msg_len = len(raw)
    msg_word = int.from_bytes(raw + b"\x00" * (_MAX_MSG_BYTES - len(raw)), "big")
    return msg_len, msg_word


#: Pre-built stdlib module.  Merge ``STDLIB.ctx`` into your builder before
#: :meth:`~pydefi.vm.venom.ModuleBuilder.compile` to make the stdlib functions
#: available for inlining.  The :class:`~pydefi.vm.venom.ModuleBuilder` convenience
#: methods (:meth:`~pydefi.vm.venom.ModuleBuilder.revert_if` and
#: :meth:`~pydefi.vm.venom.ModuleBuilder.assert_ge`) merge it automatically.
STDLIB: "ModuleBuilder" = _build_stdlib()


def revert_if(builder: VenomBuilder, cond: object, msg: str) -> None:
    """Emit an ``invoke stdlib.revert_if`` instruction.

    Encodes *msg* at Python build time into its length and word representation,
    then emits an :ref:`invoke <venom-invoke>` instruction targeting
    ``stdlib.revert_if``.  The
    :class:`~vyper.venom.passes.function_inliner.FunctionInlinerPass` will inline
    the function at the call site during compilation.

    The calling module must have the stdlib context merged before
    :meth:`~pydefi.vm.venom.ModuleBuilder.compile` is called.  When *builder* is a
    :class:`~pydefi.vm.venom.ModuleBuilder`, the merge is performed automatically.

    Args:
        builder: Venom IR builder to emit the invoke instruction into.
        cond:    Venom IR operand — non-zero triggers the revert.
        msg:     Error message string.  Must be ≤ 32 bytes when UTF-8 encoded.

    Raises:
        ValueError: If *msg* encodes to more than 32 bytes.

    Example::

        from pydefi.vm.stdlib import revert_if, STDLIB
        is_zero = mod.iszero(amount)
        revert_if(mod, is_zero, "amount is zero")
        mod.merge(STDLIB.ctx)   # or use ModuleBuilder.revert_if which auto-merges
        bytecode = mod.compile()
    """
    msg_len, msg_word = _encode_msg(msg)
    builder.invoke(
        IRLabel("stdlib.revert_if"),
        [cond, IRLiteral(msg_len), IRLiteral(msg_word)],
        returns=0,
    )
    if hasattr(builder, "_ensure_stdlib_merged"):
        builder._ensure_stdlib_merged()


def assert_ge(builder: VenomBuilder, a: object, b: object, msg: str) -> None:
    """Emit an ``invoke stdlib.assert_ge`` instruction.

    Encodes *msg* at Python build time into its length and word representation,
    then emits an :ref:`invoke <venom-invoke>` instruction targeting
    ``stdlib.assert_ge``.  Reverts with ``Error(string)`` when ``a < b`` at
    runtime; otherwise execution continues after the invoke.

    The calling module must have the stdlib context merged before
    :meth:`~pydefi.vm.venom.ModuleBuilder.compile` is called.  When *builder* is a
    :class:`~pydefi.vm.venom.ModuleBuilder`, the merge is performed automatically.

    Args:
        builder: Venom IR builder to emit the invoke instruction into.
        a:       Venom IR operand — left-hand side of the comparison.
        b:       Venom IR operand — right-hand side of the comparison.
        msg:     Error message string.  Must be ≤ 32 bytes when UTF-8 encoded.

    Raises:
        ValueError: If *msg* encodes to more than 32 bytes.

    Example::

        from pydefi.vm.stdlib import assert_ge, STDLIB
        assert_ge(mod, amount_out, IRLiteral(min_out), "slippage too high")
        mod.merge(STDLIB.ctx)   # or use ModuleBuilder.assert_ge which auto-merges
        bytecode = mod.compile()
    """
    msg_len, msg_word = _encode_msg(msg)
    builder.invoke(
        IRLabel("stdlib.assert_ge"),
        [a, b, IRLiteral(msg_len), IRLiteral(msg_word)],
        returns=0,
    )
    if hasattr(builder, "_ensure_stdlib_merged"):
        builder._ensure_stdlib_merged()

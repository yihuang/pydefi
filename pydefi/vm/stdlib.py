"""Venom IR stdlib — utility functions defined as proper IR functions.

``STDLIB`` is a lazy proxy: ``STDLIB.ctx`` returns a fresh
:class:`~vyper.venom.context.IRContext` containing ``stdlib.revert_if``
and ``stdlib.assert_ge`` as named venom IR functions.

Callers **invoke the functions by label** after merging the stdlib context into
their own.  :class:`~vyper.venom.passes.function_inliner.FunctionInlinerPass`
inlines them at each call site during compilation, so there is no runtime call
overhead.

A fresh context per merge is required because :meth:`IRContext.merge
<vyper.venom.context.IRContext.merge>` empties its sources.

Usage pattern
-------------

::

    from pydefi.vm.venom import ModuleBuilder
    from pydefi.vm.stdlib import STDLIB, encode_msg
    from vyper.venom.basicblock import IRLabel, IRLiteral

    mod = ModuleBuilder("example")
    amount = mod.calldataload(IRLiteral(4))

    msg_len, msg_word = encode_msg("amount is zero")
    mod.invoke(
        IRLabel("stdlib.revert_if"),
        [mod.iszero(amount), IRLiteral(msg_len), IRLiteral(msg_word)],
        returns=0,
    )

    mod.return_(mod.alloca(32), IRLiteral(32))
    mod.ctx.merge(STDLIB.ctx)
    bytecode = mod.compile()

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

if TYPE_CHECKING:
    from pydefi.vm.venom import ModuleBuilder

__all__ = ["STDLIB", "encode_msg"]

# keccak256("Error(string)")[:4] stored as a full 32-byte MSTORE word
# (4-byte selector in the high bits, 28 zero bytes in the low bits)
_ERROR_SELECTOR_WORD: int = 0x08C379A000000000000000000000000000000000000000000000000000000000

_MAX_MSG_BYTES: int = 32


def encode_msg(msg: str) -> tuple[int, int]:
    """Encode a UTF-8 string ≤ 32 bytes into ``(msg_len, msg_word)`` literals."""
    raw = msg.encode("utf-8")
    if len(raw) > _MAX_MSG_BYTES:
        raise ValueError(f"message too long ({len(raw)} bytes, max {_MAX_MSG_BYTES})")
    msg_word = int.from_bytes(raw.ljust(_MAX_MSG_BYTES, b"\x00"), "big")
    return len(raw), msg_word


def _emit_error_revert(mod: "ModuleBuilder", msg_len, msg_word) -> None:
    """Emit MSTOREs + REVERT for a 100-byte ``Error(string)`` payload."""
    buf = mod.alloca(100)
    mod.mstore(buf, _ERROR_SELECTOR_WORD)
    mod.mstore(mod.add(buf, 4), 32)
    mod.mstore(mod.add(buf, 36), msg_len)
    mod.mstore(mod.add(buf, 68), msg_word)
    mod.revert(buf, 100)


def _build_revert_branch(mod: "ModuleBuilder", cond, msg_len, msg_word, ret_pc) -> None:
    """Emit ``if cond: revert(Error(msg)) else: ret(ret_pc)``."""
    bb_revert = mod.create_block("revert")
    bb_ok = mod.create_block("ok")
    mod.jnz(cond, bb_revert.label, bb_ok.label)

    mod.append_block(bb_revert)
    mod.set_block(bb_revert)
    _emit_error_revert(mod, msg_len, msg_word)

    mod.append_block(bb_ok)
    mod.set_block(bb_ok)
    mod.ret(ret_pc)


def _build_revert_if(mod: "ModuleBuilder") -> None:
    """``stdlib.revert_if(cond, msg_len, msg_word, ret_pc)``."""
    fn = mod.ctx.create_function("revert_if")
    mod.set_block(fn.entry)
    cond = mod.param()
    msg_len = mod.param()
    msg_word = mod.param()
    ret_pc = mod.param()
    _build_revert_branch(mod, cond, msg_len, msg_word, ret_pc)


def _build_assert_ge(mod: "ModuleBuilder") -> None:
    """``stdlib.assert_ge(a, b, msg_len, msg_word, ret_pc)`` — reverts when ``a < b``."""
    fn = mod.ctx.create_function("assert_ge")
    mod.set_block(fn.entry)
    a = mod.param()
    b = mod.param()
    msg_len = mod.param()
    msg_word = mod.param()
    ret_pc = mod.param()
    _build_revert_branch(mod, mod.lt(a, b), msg_len, msg_word, ret_pc)


def _build_stdlib() -> "ModuleBuilder":
    from pydefi.vm.venom import ModuleBuilder

    mod = ModuleBuilder("stdlib")
    mod.stop()  # terminate the auto-created stdlib.main entry
    _build_revert_if(mod)
    _build_assert_ge(mod)
    return mod


class _StdlibProxy:
    """Lazy stdlib proxy — ``STDLIB.ctx`` returns a fresh context each access.

    A fresh build per merge is required because :meth:`IRContext.merge
    <vyper.venom.context.IRContext.merge>` empties its sources.
    """

    @property
    def ctx(self):
        return _build_stdlib().ctx


#: Stdlib module proxy.  Use ``mod.ctx.merge(STDLIB.ctx)`` before
#: :meth:`~pydefi.vm.venom.ModuleBuilder.compile`.
STDLIB = _StdlibProxy()

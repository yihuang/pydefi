"""Venom IR stdlib — shared utility snippets.

Functions in this module emit venom IR instructions into a
:class:`~vyper.venom.builder.VenomBuilder` (or :class:`~pydefi.vm.venom.ModuleBuilder`)
to implement common patterns that would otherwise require duplicated boilerplate.
All functions emit instructions into the **current block** of the given builder.
After the call, the builder's active block is the "continuation" block (the path
taken when no revert occurs).

Error encoding
--------------

Reverts use the standard Solidity ``Error(string)`` ABI encoding (selector
``0x08c379a0``), matching what EVM tooling and frontends expect::

    [4 bytes]  0x08c379a0            — keccak256("Error(string)")[:4]
    [32 bytes] 0x00..0020            — ABI offset = 32
    [32 bytes] 0x00..000N            — string length N
    [32 bytes] <string bytes>        — UTF-8 data, zero-padded on the right

Total revert data: 100 bytes.

Usage
-----

These functions are most conveniently accessed as methods on
:class:`~pydefi.vm.venom.ModuleBuilder`::

    from pydefi.vm.venom import ModuleBuilder
    from vyper.venom.basicblock import IRLiteral

    mod = ModuleBuilder("example")
    amount = mod.calldataload(IRLiteral(4))

    # Revert if amount is zero
    is_zero = mod.iszero(amount)
    mod.revert_if(is_zero, "amount is zero")

    # Assert amount >= min_amount
    mod.assert_ge(amount, IRLiteral(1000), "amount too small")

    mod.return_(mod.alloca(32), IRLiteral(32))
    bytecode = mod.compile()

They can also be called standalone::

    from pydefi.vm.stdlib import revert_if, assert_ge
    revert_if(builder, cond, "error message")
    assert_ge(builder, a, b, "a must be >= b")
"""

from __future__ import annotations

from vyper.venom.basicblock import IRLiteral
from vyper.venom.builder import VenomBuilder

__all__ = ["revert_if", "assert_ge"]

# keccak256("Error(string)")[:4] stored as a full 32-byte MSTORE word
# (4-byte selector in the high bits, 28 zero bytes in the low bits)
_ERROR_SELECTOR_WORD: int = 0x08C379A0_00000000_00000000_00000000_00000000_00000000_00000000_00000000

# Maximum encoded message length (must fit in a single EVM word)
_MAX_MSG_BYTES: int = 32


def revert_if(builder: VenomBuilder, cond: object, msg: str) -> None:
    """Emit a conditional revert with ``Error(string)`` ABI encoding.

    If *cond* is non-zero at runtime the function builds a standard
    ``Error(string)`` payload in memory and executes ``REVERT``.  When *cond*
    is zero, execution continues normally after the call.

    The builder's active block is updated to the continuation ("ok") block on
    return, so any instructions emitted afterwards follow the non-revert path.

    Args:
        builder: Venom IR builder to emit instructions into.
        cond:    Venom IR operand — the condition to test.  A non-zero value
                 triggers the revert.
        msg:     Error message string.  Must be ≤ 32 bytes when UTF-8 encoded.

    Raises:
        ValueError: If *msg* encodes to more than 32 bytes.

    Example::

        is_zero = mod.iszero(amount)
        revert_if(mod, is_zero, "amount is zero")
        # instructions here run only when amount != 0
    """
    raw = msg.encode("utf-8")
    if len(raw) > _MAX_MSG_BYTES:
        raise ValueError(f"revert_if: message too long ({len(raw)} bytes, max {_MAX_MSG_BYTES})")

    msg_len = len(raw)
    # Left-justified in a 32-byte word (high bytes = msg, low bytes = zeros)
    msg_word = int.from_bytes(raw + b"\x00" * (_MAX_MSG_BYTES - len(raw)), "big")

    # Create the revert and continuation blocks.
    bb_revert = builder.create_block("revert")
    bb_ok = builder.create_block("ok")

    # Conditional branch: non-zero → revert, zero → continue.
    builder.jnz(cond, bb_revert.label, bb_ok.label)

    # --- Revert block ---
    builder.append_block(bb_revert)
    builder.set_block(bb_revert)

    # Allocate 100 bytes for the Error(string) payload:
    #   [ 4 bytes selector ][ 32 bytes offset ][ 32 bytes length ][ 32 bytes data ]
    buf = builder.alloca(100)

    # mem[buf + 0..31]:  selector word  (0x08c379a0 + 28 zero bytes)
    builder.mstore(buf, IRLiteral(_ERROR_SELECTOR_WORD))
    # mem[buf + 4..35]:  ABI offset = 32  (overwrites the trailing zeros above)
    ptr4 = builder.add(buf, IRLiteral(4))
    builder.mstore(ptr4, IRLiteral(32))
    # mem[buf + 36..67]: string length
    ptr36 = builder.add(buf, IRLiteral(36))
    builder.mstore(ptr36, IRLiteral(msg_len))
    # mem[buf + 68..99]: string bytes, zero-padded on the right to 32 bytes
    ptr68 = builder.add(buf, IRLiteral(68))
    builder.mstore(ptr68, IRLiteral(msg_word))

    # REVERT with the 100-byte payload.
    builder.revert(buf, IRLiteral(100))

    # --- Continuation block ---
    builder.append_block(bb_ok)
    builder.set_block(bb_ok)


def assert_ge(builder: VenomBuilder, a: object, b: object, msg: str) -> None:
    """Emit an assertion that *a* ≥ *b*, reverting with *msg* if it fails.

    Equivalent to ``if a < b: revert(Error(msg))``.

    The builder's active block is updated to the continuation block on return,
    so subsequent instructions run only when ``a >= b``.

    Args:
        builder: Venom IR builder to emit instructions into.
        a:       Venom IR operand — left-hand side of the comparison.
        b:       Venom IR operand — right-hand side of the comparison.
        msg:     Error message string.  Must be ≤ 32 bytes when UTF-8 encoded.

    Raises:
        ValueError: If *msg* encodes to more than 32 bytes.

    Example::

        assert_ge(mod, amount_out, IRLiteral(min_out), "slippage too high")
    """
    # lt(a, b) = 1 if a < b (assertion violated), 0 otherwise
    cond = builder.lt(a, b)
    revert_if(builder, cond, msg)

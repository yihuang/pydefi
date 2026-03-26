"""DeFiVM — minimal register-based macro-assembler for on-chain DeFi flows.

The :mod:`pydefi.vm.program` module provides a Python DSL for building DeFiVM
bytecode programs::

    from pydefi.vm.program import push_u256, push_addr, push_bytes, call, assert_ge

    program = (
        push_bytes(swap_calldata)
        + push_u256(0)
        + push_addr(SWAP_ADAPTER)
        + push_u256(0)
        + call()
    )
"""

from pydefi.vm.program import (
    OP_ASSERT_GE,
    OP_ASSERT_LE,
    OP_BALANCE_OF,
    OP_CALL,
    OP_DUP,
    OP_JUMP,
    OP_JUMPI,
    OP_LOAD_REG,
    OP_PATCH_ADDR,
    OP_PATCH_U256,
    OP_POP,
    OP_PUSH_ADDR,
    OP_PUSH_BYTES,
    OP_PUSH_U256,
    OP_RET_SLICE,
    OP_RET_U256,
    OP_REVERT_IF,
    OP_SELF_ADDR,
    OP_STORE_REG,
    OP_SUB,
    OP_SWAP,
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

__all__ = [
    # Opcode constants
    "OP_PUSH_U256",
    "OP_PUSH_ADDR",
    "OP_PUSH_BYTES",
    "OP_DUP",
    "OP_SWAP",
    "OP_POP",
    "OP_LOAD_REG",
    "OP_STORE_REG",
    "OP_JUMP",
    "OP_JUMPI",
    "OP_REVERT_IF",
    "OP_ASSERT_GE",
    "OP_ASSERT_LE",
    "OP_CALL",
    "OP_BALANCE_OF",
    "OP_SELF_ADDR",
    "OP_SUB",
    "OP_PATCH_U256",
    "OP_PATCH_ADDR",
    "OP_RET_U256",
    "OP_RET_SLICE",
    # Program builder helpers
    "push_u256",
    "push_addr",
    "push_bytes",
    "dup",
    "swap",
    "pop",
    "load_reg",
    "store_reg",
    "jump",
    "jumpi",
    "revert_if",
    "assert_ge",
    "assert_le",
    "call",
    "balance_of",
    "self_addr",
    "sub",
    "patch_u256",
    "patch_addr",
    "ret_u256",
    "ret_slice",
]

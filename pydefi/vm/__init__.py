"""DeFiVM — minimal register-based macro-assembler for on-chain DeFi flows.

Two complementary interfaces are provided:

**Functional (low-level)**
    Import individual instruction builders from :mod:`pydefi.vm.program` and
    concatenate them with ``+``::

        from pydefi.vm.program import push_u256, push_addr, push_bytes, call, assert_ge

        program = (
            push_bytes(swap_calldata)
            + push_u256(0)
            + push_addr(SWAP_ADAPTER)
            + push_u256(0)
            + call()
        )

**Fluent builder (high-level)**
    Use :class:`~pydefi.vm.builder.Program` for method chaining, label-based
    jumps, and the :meth:`~pydefi.vm.builder.Program.call_contract` helper::

        from pydefi.vm import Program
        from eth_contract.erc20 import ERC20

        bytecode = (
            Program()
            .call_contract(TOKEN, ERC20.fns.approve(ROUTER, amount_in).data)
            .pop()  # consume CALL success flag
            .call_contract(ROUTER, swap_calldata)
            .pop()  # consume CALL success flag
            .push_addr(RECIPIENT)
            .push_addr(TOKEN)
            .push_u256(min_out)
            .assert_ge("slippage: amount_out too low")
            .build()
        )
"""

from pydefi.vm.builder import PatchSource, PatchSpec, Program
from pydefi.vm.program import (
    OP_ADD,
    OP_ASSERT_GE,
    OP_ASSERT_LE,
    OP_BALANCE_OF,
    OP_CALL,
    OP_DIV,
    OP_DUP,
    OP_JUMP,
    OP_JUMPI,
    OP_LOAD_REG,
    OP_MOD,
    OP_MUL,
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
    add,
    assert_ge,
    assert_le,
    balance_of,
    call,
    div,
    dup,
    jump,
    jumpi,
    load_reg,
    mod,
    mul,
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
    # Fluent builder
    "Program",
    # Patch type aliases
    "PatchSource",
    "PatchSpec",
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
    "OP_ADD",
    "OP_MUL",
    "OP_DIV",
    "OP_MOD",
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
    "add",
    "mul",
    "div",
    "mod",
    "patch_u256",
    "patch_addr",
    "ret_u256",
    "ret_slice",
]

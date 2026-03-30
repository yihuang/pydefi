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
        from pydefi.vm.abi import erc20_approve

        bytecode = (
            Program()
            .call_contract(TOKEN, erc20_approve(ROUTER, amount_in))
            .call_contract(ROUTER, swap_calldata)
            .push_addr(RECIPIENT)
            .push_addr(TOKEN)
            .push_u256(min_out)
            .assert_ge("slippage: amount_out too low")
            .build()
        )
"""

from pydefi.vm.abi import (
    encode_calldata,
    erc20_approve,
    erc20_balance_of,
    erc20_transfer,
    erc20_transfer_from,
)
from pydefi.vm.builder import Program, PatchSource, PatchSpec
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
    OP_ADD,
    OP_MUL,
    OP_DIV,
    OP_MOD,
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
    add,
    mul,
    div,
    mod,
    swap,
)

__all__ = [
    # Fluent builder
    "Program",
    # Patch type aliases
    "PatchSource",
    "PatchSpec",
    # ABI helpers
    "encode_calldata",
    "erc20_transfer",
    "erc20_approve",
    "erc20_transfer_from",
    "erc20_balance_of",
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

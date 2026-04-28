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

from pydefi.vm.abi import emit_abi_encode, emit_abi_encode_packed
from pydefi.vm.builder import Patch, PatchSpec, Program
from pydefi.vm.context import ProgramContext
from pydefi.vm.dag import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.program import (
    OP_ADD,
    OP_AND,
    OP_BALANCE,
    OP_CALL,
    OP_DIV,
    OP_DUP,
    OP_EQ,
    OP_GAS,
    OP_GT,
    OP_ISZERO,
    OP_JUMP,
    OP_JUMPDEST,
    OP_JUMPI,
    OP_LT,
    OP_MLOAD,
    OP_MOD,
    OP_MSTORE,
    OP_MUL,
    OP_NOT,
    OP_OR,
    OP_POP,
    OP_PUSH_ADDR,
    OP_PUSH_U256,
    OP_RETURNDATACOPY,
    OP_REVERT,
    OP_SELF_ADDR,
    OP_SHL,
    OP_SHR,
    OP_STATICCALL,
    OP_SUB,
    OP_SWAP,
    OP_XOR,
    add,
    assert_ge,
    assert_le,
    balance_of,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    call,
    div,
    dup,
    dup_n,
    eq,
    gas_opcode,
    gt,
    iszero,
    jump,
    jumpi,
    load_reg,
    lt,
    mod,
    mul,
    pop,
    push_addr,
    push_bytes,
    push_u256,
    ret_slice,
    ret_u256,
    revert_if,
    self_addr,
    shl,
    shr,
    store_reg,
    sub,
    swap,
)
from pydefi.vm.stdlib import build_stdlib, encode_msg
from pydefi.vm.swap import (
    V2_AMOUNT_OUT_OFFSET,
    V3_AMOUNT_OUT_OFFSET,
    SwapHop,
    SwapProtocol,
    encode_v2_callback_data,
    encode_v3_callback_data,
    encode_v3_path,
    swap_route_to_hops,
    v3_pool_swap_calldata,
)

__all__ = [
    # In-VM ABI encoding bytecode generators
    "emit_abi_encode",
    "emit_abi_encode_packed",
    # Fluent builder
    "Program",
    # Venom IR program builder (high-level, typed)
    "ProgramContext",
    "encode_msg",
    "build_stdlib",
    # Patch type aliases and Patch class
    "Patch",
    "PatchSpec",
    # Opcode constants
    "OP_PUSH_U256",
    "OP_PUSH_ADDR",
    "OP_DUP",
    "OP_SWAP",
    "OP_POP",
    "OP_JUMP",
    "OP_JUMPI",
    "OP_JUMPDEST",
    "OP_CALL",
    "OP_SELF_ADDR",
    "OP_BALANCE",
    "OP_GAS",
    "OP_MLOAD",
    "OP_MSTORE",
    "OP_REVERT",
    "OP_RETURNDATACOPY",
    "OP_STATICCALL",
    "OP_SUB",
    "OP_ADD",
    "OP_MUL",
    "OP_DIV",
    "OP_MOD",
    "OP_LT",
    "OP_GT",
    "OP_EQ",
    "OP_ISZERO",
    "OP_AND",
    "OP_OR",
    "OP_XOR",
    "OP_NOT",
    "OP_SHL",
    "OP_SHR",
    # Program builder helpers
    "push_u256",
    "push_addr",
    "push_bytes",
    "dup",
    "dup_n",
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
    "gas_opcode",
    "balance_of",
    "self_addr",
    "sub",
    "add",
    "mul",
    "div",
    "mod",
    "lt",
    "gt",
    "eq",
    "iszero",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "shl",
    "shr",
    "ret_u256",
    "ret_slice",
    # Swap composer
    "SwapHop",
    "SwapProtocol",
    "V2_AMOUNT_OUT_OFFSET",
    "V3_AMOUNT_OUT_OFFSET",
    "build_execution_program_for_dag",
    "build_quote_program_for_dag",
    "encode_v2_callback_data",
    "encode_v3_callback_data",
    "encode_v3_path",
    "swap_route_to_hops",
    "v3_pool_swap_calldata",
]

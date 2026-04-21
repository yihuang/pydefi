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
from pydefi.vm.builder import Patch, PatchSpec
from pydefi.vm.dag import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.program import Program
from pydefi.vm.swap import (
    V2_AMOUNT_OUT_OFFSET,
    V3_AMOUNT_OUT_OFFSET,
    SwapHop,
    SwapProtocol,
    encode_v2_callback_data,
    encode_v3_callback_data,
    encode_v3_path,
    v3_pool_swap_calldata,
)

__all__ = [
    # In-VM ABI encoding bytecode generators
    "emit_abi_encode",
    "emit_abi_encode_packed",
    # Fluent builder
    "Program",
    # Patch type aliases and Patch class
    "Patch",
    "PatchSpec",
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
    "v3_pool_swap_calldata",
]

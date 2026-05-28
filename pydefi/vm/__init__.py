"""DeFiVM — SSA-style program builder over Vyper's Venom IR.

Usage::

    from eth_contract.erc20 import ERC20
    from pydefi.vm import Program

    prog = Program()
    success = prog.call_contract(token, ERC20.fns.approve, ROUTER, 10**18)
    prog.assert_(success)
    prog.builder.stop()
    bytecode = prog.build()
"""

from pydefi.abi.bridge import IIBC_SENDER_CALLBACKS_INTERFACE_ID
from pydefi.vm.context import Operand, Program, ProgramContext
from pydefi.vm.dag import build_execution_program_for_dag, build_quote_program_for_dag
from pydefi.vm.eureka import (
    approve_then_send_transfer,
    encode_send_and_compose_calldata,
    send_transfer,
)
from pydefi.vm.swap import (
    V2_AMOUNT_OUT_OFFSET,
    V3_AMOUNT_OUT_OFFSET,
    SwapHop,
    SwapProtocol,
    build_swap_transaction,
    encode_v2_callback_data,
    encode_v3_callback_data,
    encode_v3_path,
    swap_route_to_hops,
    v3_pool_swap_calldata,
)

__all__ = [
    "Operand",
    "Program",
    # Venom IR program builder (high-level, typed)
    "ProgramContext",
    # Swap composer
    "SwapHop",
    "SwapProtocol",
    "IIBC_SENDER_CALLBACKS_INTERFACE_ID",
    "V2_AMOUNT_OUT_OFFSET",
    "V3_AMOUNT_OUT_OFFSET",
    "approve_then_send_transfer",
    "build_execution_program_for_dag",
    "encode_send_and_compose_calldata",
    "build_quote_program_for_dag",
    "build_swap_transaction",
    "encode_v2_callback_data",
    "encode_v3_callback_data",
    "encode_v3_path",
    "send_transfer",
    "swap_route_to_hops",
    "v3_pool_swap_calldata",
]

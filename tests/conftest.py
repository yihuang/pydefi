"""Shared test utilities and fixtures for pydefi tests.

Provides :func:`mini_evm` — a lightweight `py-evm`-based EVM executor for
running DeFiVM program bytecode entirely in-process without requiring Anvil,
any external process, or network access.

Intended for fast unit tests that verify program logic.  It is significantly
faster to set up than JSON-RPC fork tests and allows direct inspection of EVM
execution results such as output bytes, gas usage, and revert messages.

Usage example::

    from tests.conftest import mini_evm, RETURN_TOP
    from pydefi.vm.program import push_u256, add

    result = mini_evm(push_u256(3) + push_u256(5) + add() + RETURN_TOP)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 8

Memory layout note
------------------
``memory[0x40]`` (the Solidity free-memory pointer) starts at ``0`` in the
standalone EVM context, rather than ``0x280`` as in the on-chain DeFiVM
interpreter context.  :func:`~pydefi.vm.program.push_bytes` handles this
transparently — it initialises the pointer to ``0x280`` on first use.
Programs that rely on :func:`~pydefi.vm.program.ret_u256` or
:func:`~pydefi.vm.program.ret_slice` require returndata from an outer call
and therefore cannot run standalone; test those paths via Anvil fork tests
instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from eth import constants
from eth.chains.base import MiningChain
from eth.db.atomic import AtomicDB
from eth.vm.forks.london import LondonVM
from eth.vm.message import Message
from eth.vm.transaction_context import BaseTransactionContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Caller address used in all mini_evm executions.
MINI_EVM_SENDER: bytes = b"\xaa" * 20

#: Contract address used as the execution target in mini_evm.
MINI_EVM_RECEIVER: bytes = b"\xbb" * 20

#: Initial ETH balance credited to :data:`MINI_EVM_SENDER` in the genesis
#: state.  Tests can query this value via ``balance_of(0, SENDER_INT)``.
MINI_EVM_SENDER_BALANCE: int = 10**21

#: Bytecode snippet that stores the top-of-stack value at ``memory[0]`` and
#: returns 32 bytes — effectively converting a ``uint256`` stack result into
#: a 32-byte return value that :func:`mini_evm` exposes as ``result.output``.
#:
#: Append this to any program that leaves a ``uint256`` on the stack::
#:
#:     result = mini_evm(push_u256(42) + RETURN_TOP)
#:     assert int.from_bytes(result.output, "big") == 42
#:
#: Opcodes: ``PUSH1 0x00  MSTORE  PUSH1 0x20  PUSH1 0x00  RETURN``
RETURN_TOP: bytes = bytes(
    [
        0x60,
        0x00,  # PUSH1 0x00   → offset for MSTORE
        0x52,  # MSTORE        → mem[0] = TOS-value
        0x60,
        0x20,  # PUSH1 0x20   → size = 32
        0x60,
        0x00,  # PUSH1 0x00   → offset = 0
        0xF3,  # RETURN
    ]
)

# ---------------------------------------------------------------------------
# Module-level EVM setup (shared state, read-only from execution perspective)
# ---------------------------------------------------------------------------

_GENESIS_PARAMS: dict = {
    "difficulty": constants.GENESIS_DIFFICULTY,
    "gas_limit": 30_000_000,
    "timestamp": 1,
    "coinbase": b"\x00" * 20,
    "extra_data": b"",
    "nonce": constants.GENESIS_NONCE,
    "mix_hash": constants.GENESIS_MIX_HASH,
}

_GENESIS_STATE: dict = {
    MINI_EVM_SENDER: {
        "balance": MINI_EVM_SENDER_BALANCE,
        "nonce": 0,
        "code": b"",
        "storage": {},
    }
}

# Build chain and vm once; memory is ephemeral per computation so the shared
# state is effectively read-only across test calls.
_chain = MiningChain.configure(
    __name__="MiniEVMChain",
    vm_configuration=((0, LondonVM),),
).from_genesis(AtomicDB(), _GENESIS_PARAMS, _GENESIS_STATE)
_vm = _chain.get_vm()
_TX_CTX = BaseTransactionContext(gas_price=1, origin=MINI_EVM_SENDER)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class EVMResult:
    """Result of executing EVM bytecode via :func:`mini_evm`.

    Attributes:
        output:   Return data produced by ``RETURN``, or revert data produced
                  by ``REVERT``.  For a successful execution that ends with
                  :data:`RETURN_TOP` appended, ``output`` holds the 32-byte
                  big-endian encoding of the top-of-stack value.
        gas_used: Number of EVM gas units consumed.
        is_error: ``True`` if the computation ended with ``REVERT`` or ran
                  out of gas; ``False`` for a successful ``RETURN``.
    """

    output: bytes
    gas_used: int
    is_error: bool


def mini_evm(
    bytecode: bytes,
    *,
    calldata: bytes = b"",
    gas: int = 1_000_000,
) -> EVMResult:
    """Execute *bytecode* using py-evm (LondonVM) and return the result.

    Runs EVM bytecode in-process without any external processes or network
    access, making it suitable for fast unit tests that verify program logic.

    Args:
        bytecode: EVM bytecode to execute.
        calldata: Optional calldata bytes (default empty).
        gas:      Gas limit for the execution (default 1 000 000).

    Returns:
        :class:`EVMResult` with ``.output``, ``.gas_used``, ``.is_error``.

    Example::

        from pydefi.vm.program import push_u256, mul

        result = mini_evm(push_u256(6) + push_u256(7) + mul() + RETURN_TOP)
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 42
    """
    msg = Message(
        gas=gas,
        to=MINI_EVM_RECEIVER,
        sender=MINI_EVM_SENDER,
        value=0,
        data=calldata,
        code=bytecode,
    )
    comp = _vm.state.computation_class.apply_computation(_vm.state, msg, _TX_CTX)
    return EVMResult(
        output=comp.output,
        gas_used=comp.get_gas_used(),
        is_error=comp.is_error,
    )

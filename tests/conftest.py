"""Shared test utilities and fixtures for pydefi tests.

Provides two complementary EVM testing facilities:

1. :func:`mini_evm` / :data:`RETURN_TOP` — stateless, single-shot executor
   for quick bytecode tests (no contract setup needed).

2. :class:`MiniEVMContext` — stateful EVM context backed by py-evm
   ``ShanghaiVM`` with real DeFiVM + Analog-Labs interpreter deployed.
   Contracts deployed via :meth:`~MiniEVMContext.deploy` persist across
   all subsequent calls.

Usage::

    # Stateless
    result = mini_evm(push_u256(3) + push_u256(5) + add() + RETURN_TOP)
    assert int.from_bytes(result.output, "big") == 8

    # Stateful
    ctx = MiniEVMContext()
    token = ctx.deploy_mock_token()
    ctx.mint_token(token, ctx.program_executor, 500 * 10**18)
    result = ctx.run_program(self_addr() + push_addr(token.hex()) + balance_of() + RETURN_TOP)
    assert int.from_bytes(result.output, "big") == 500 * 10**18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
from eth import constants
from eth._utils.address import generate_contract_address
from eth.chains.base import MiningChain
from eth.db.atomic import AtomicDB
from eth.vm.forks.london import LondonVM
from eth.vm.forks.shanghai import ShanghaiVM
from eth.vm.message import Message
from eth.vm.transaction_context import BaseTransactionContext
from eth_contract.contract import Contract
from eth_contract.erc20 import ERC20
from eth_contract.utils import get_initcode
from eth_keys import keys

from tests.live.sol_utils import (
    MOCK_TOKEN_SOL,
    compile_interpreter_sync,
    compile_sol_file,
    compile_sol_source,
    ensure_solc,
)

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
# Module-level EVM setup for mini_evm() (shared, read-only across calls)
# ---------------------------------------------------------------------------

_GENESIS_PARAMS_LONDON: dict = {
    "difficulty": constants.GENESIS_DIFFICULTY,
    "gas_limit": 30_000_000,
    "timestamp": 1,
    "coinbase": b"\x00" * 20,
    "extra_data": b"",
    "nonce": constants.GENESIS_NONCE,
    "mix_hash": constants.GENESIS_MIX_HASH,
}

_GENESIS_STATE_LONDON: dict = {
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
).from_genesis(AtomicDB(), _GENESIS_PARAMS_LONDON, _GENESIS_STATE_LONDON)
_vm = _chain.get_vm()
_TX_CTX = BaseTransactionContext(gas_price=1, origin=MINI_EVM_SENDER)


# ---------------------------------------------------------------------------
# Public API — EVMResult
# ---------------------------------------------------------------------------


@dataclass
class EVMResult:
    """Result of executing EVM bytecode via :func:`mini_evm` or
    :class:`MiniEVMContext`.

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


# ---------------------------------------------------------------------------
# Public API — mini_evm() stateless executor
# ---------------------------------------------------------------------------


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
    snapshot = _vm.state.snapshot()
    try:
        comp = _vm.state.computation_class.apply_computation(_vm.state, msg, _TX_CTX)
        return EVMResult(
            output=comp.output,
            gas_used=comp.get_gas_used(),
            is_error=comp.is_error,
        )
    finally:
        _vm.state.revert(snapshot)


# ---------------------------------------------------------------------------
# Public API — MiniEVMContext stateful EVM context
# ---------------------------------------------------------------------------

#: EVM-version flag used when compiling Solidity sources inside
#: :class:`MiniEVMContext`.  ``"shanghai"`` enables PUSH0 (required by the
#: Analog-Labs interpreter) while remaining compatible with Solidity 0.8.24.
_SOLC_EVM_VERSION: str = "shanghai"

#: Genesis parameters for the ShanghaiVM chain used by :class:`MiniEVMContext`.
#: Post-merge: ``difficulty=0``, ``nonce=b'\x00'*8``, ``mix_hash`` used as
#: ``prevrandao`` (zero is fine for tests).
_GENESIS_PARAMS_SHANGHAI: dict = {
    "difficulty": 0,
    "gas_limit": 30_000_000,
    "timestamp": 1,
    "coinbase": b"\x00" * 20,
    "extra_data": b"",
    "nonce": b"\x00" * 8,
    "mix_hash": b"\x00" * 32,
}

#: EIP-4844 is not used; gas price for deploy transactions in MiniEVMContext.
_CTX_GAS_PRICE: int = 10**9

#: Default gas limit used for ``run_program`` and ``call`` inside a
#: :class:`MiniEVMContext`.  Large enough for programs that do several
#: ERC-20 calls; lower it explicitly when testing gas-bounded behaviour.
_CTX_DEFAULT_GAS: int = 3_000_000

# Cached compiled MockToken bytecode (creation code only).
# Uses MOCK_TOKEN_SOL from tests.live.sol_utils to avoid duplication.
_mock_token_bin: Optional[bytes] = None


def _get_mock_token_bin() -> bytes:
    """Lazily compile MockToken and return its creation bytecode."""
    global _mock_token_bin
    if _mock_token_bin is None:
        ensure_solc("0.8.24")
        compiled = compile_sol_source(
            MOCK_TOKEN_SOL,
            "MockToken",
            evm_version=_SOLC_EVM_VERSION,
        )
        _mock_token_bin = bytes.fromhex(compiled["bin"])
    return _mock_token_bin


# Cached compiled interpreter creation bytecode.
_interpreter_bin: Optional[bytes] = None


def _get_interpreter_bin() -> bytes:
    """Lazily compile the Analog-Labs interpreter and return its creation bytecode."""
    global _interpreter_bin
    if _interpreter_bin is None:
        compiled = compile_interpreter_sync()
        _interpreter_bin = bytes.fromhex(compiled["<stdin>:Interpreter"]["bin"])
    return _interpreter_bin


# ---------------------------------------------------------------------------
# DeFiVM contract binding (for ABI-encoding execute() calls)
# ---------------------------------------------------------------------------

_DeFiVM: Contract = Contract.from_abi(["function execute(bytes calldata program) external payable"])

#: Path to the real DeFiVM.sol (relative to repo root).
_DEFI_VM_SOL_PATH = Path(__file__).parent.parent / "pydefi" / "vm" / "DeFiVM.sol"

# Cached compiled DeFiVM bytecode.
_defi_vm_compiled: Optional[dict] = None


def _get_defi_vm_compiled() -> dict:
    """Lazily compile DeFiVM.sol and return ``{"bin": ..., "abi": ...}``."""
    global _defi_vm_compiled
    if _defi_vm_compiled is None:
        _defi_vm_compiled = compile_sol_file(_DEFI_VM_SOL_PATH, "DeFiVM")
    return _defi_vm_compiled


@dataclass
class MiniEVMContext:
    """Stateful in-process EVM context backed by py-evm ``ShanghaiVM``.

    Deploys the real Analog-Labs interpreter and DeFiVM on construction so
    :meth:`run_program` exercises the authentic dispatch path.  Contract
    deployments via :meth:`deploy` persist across all subsequent calls.

    Use the :func:`evm_ctx` pytest fixture for per-test isolation::

        def test_transfer(evm_ctx):
            token = evm_ctx.deploy_mock_token()
            evm_ctx.mint_token(token, evm_ctx.program_executor, 1000 * 10**18)
            ...

    Attributes:
        deployer:         20-byte address of the funded deployer account.
        program_executor: 20-byte address of the deployed DeFiVM contract.
    """

    deployer: bytes = field(init=False)
    program_executor: bytes = field(init=False)

    # ---- private fields (not exposed to callers) --------------------------
    _chain: MiningChain = field(init=False, repr=False)
    _deployer_key: keys.PrivateKey = field(init=False, repr=False)
    _nonce: int = field(init=False, repr=False)
    _vm: object = field(init=False, repr=False)  # cached vm, invalidated on deploy

    def __post_init__(self) -> None:
        self._deployer_key = keys.PrivateKey(b"\x01" * 32)
        self.deployer = self._deployer_key.public_key.to_canonical_address()
        self._nonce = 0
        self._chain = MiningChain.configure(
            __name__="MiniEVMContextChain",
            vm_configuration=((0, ShanghaiVM),),
        ).from_genesis(
            AtomicDB(),
            _GENESIS_PARAMS_SHANGHAI,
            {
                self.deployer: {
                    "balance": 10**22,
                    "nonce": 0,
                    "code": b"",
                    "storage": {},
                }
            },
        )
        self._vm = self._chain.get_vm()
        # Deploy the real Analog-Labs EVM interpreter + DeFiVM.
        interp_addr = self.deploy(_get_interpreter_bin())
        defi_vm_info = _get_defi_vm_compiled()
        artifact = {"bytecode": defi_vm_info["bin"], "abi": defi_vm_info["abi"]}
        defivm_addr = self.deploy(get_initcode(artifact, "0x" + interp_addr.hex()))
        self.program_executor = defivm_addr

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _tx_ctx(self) -> BaseTransactionContext:
        return BaseTransactionContext(gas_price=_CTX_GAS_PRICE, origin=self.deployer)

    def _apply_computation(
        self,
        to: bytes,
        sender: bytes,
        calldata: bytes,
        value: int,
        gas: int,
    ) -> object:
        """Run a computation on the cached vm.state and return the result."""
        vm = self._vm
        code = vm.state.get_code(to)
        msg = Message(
            gas=gas,
            to=to,
            sender=sender,
            value=value,
            data=calldata,
            code=code,
        )
        return vm.state.computation_class.apply_computation(vm.state, msg, self._tx_ctx())

    # -----------------------------------------------------------------------
    # Contract deployment
    # -----------------------------------------------------------------------

    def deploy(self, creation_code: bytes, *, value: int = 0) -> bytes:
        """Deploy a contract via a CREATE transaction.

        The creation code is executed and the resulting runtime bytecode is
        committed to the chain database.  This guarantees that the deployed
        contract is visible to all subsequent :meth:`call` and
        :meth:`run_program` executions.

        Args:
            creation_code: EVM creation bytecode (constructor + runtime).
            value:         ETH (in wei) to send with the deployment transaction.

        Returns:
            The 20-byte canonical address of the newly deployed contract.
        """
        txn = self._chain.create_unsigned_transaction(
            nonce=self._nonce,
            gas_price=_CTX_GAS_PRICE,
            gas=5_000_000,
            to=b"",
            value=value,
            data=creation_code,
        )
        _, _, computation = self._chain.apply_transaction(txn.as_signed_transaction(self._deployer_key))
        addr = generate_contract_address(self.deployer, self._nonce)
        self._nonce += 1
        # Refresh cached vm so deployed code is visible.
        self._vm = self._chain.get_vm()
        deployed_code = self._vm.state.get_code(addr)
        if computation.is_error or deployed_code == b"":
            error = getattr(computation, "error", None)
            details = f": {error}" if error is not None else ""
            raise AssertionError(f"Contract deployment failed at 0x{addr.hex()}{details}")
        return addr

    def compile_and_deploy(self, source: str, contract_name: str, *constructor_args: object) -> bytes:
        """Compile a Solidity source string and deploy the named contract.

        Uses ``py-solc-x`` with Solidity 0.8.24 and the Shanghai EVM target.
        Constructor arguments are ABI-encoded and appended to the creation
        bytecode.

        Args:
            source:          Solidity source code (including pragma directive).
            contract_name:   Name of the contract to deploy.
            *constructor_args: ABI-encodable constructor arguments.

        Returns:
            The 20-byte canonical address of the newly deployed contract.
        """
        compiled = compile_sol_source(source, contract_name, evm_version=_SOLC_EVM_VERSION)
        # get_initcode expects 'bytecode' key; compile_sol_source returns 'bin'.
        artifact = {"bytecode": compiled["bin"], "abi": compiled["abi"]}
        creation_code = get_initcode(artifact, *constructor_args)
        return self.deploy(creation_code)

    # -----------------------------------------------------------------------
    # Direct state manipulation
    # -----------------------------------------------------------------------

    def set_code(self, address: bytes, code: bytes) -> None:
        """Inject runtime bytecode at *address* without running a constructor.

        Changes applied via this method are visible to subsequent
        :meth:`call` and :meth:`run_program` calls **as long as** no further
        :meth:`deploy` call intervenes (each :meth:`deploy` refreshes the VM
        from the chain database, discarding any previous :meth:`set_code`
        mutations).  For contracts that require a constructor, use
        :meth:`deploy` instead.

        Args:
            address: 20-byte address at which to place the bytecode.
            code:    Runtime bytecode to store.
        """
        self._vm.state.set_code(address, code)

    def set_balance(self, address: bytes, amount: int) -> None:
        """Set the ETH balance of *address* to *amount* wei.

        .. note::
            Like :meth:`set_code`, this mutation is applied directly to
            ``self._vm.state`` and is **not** persisted to the chain database.
            It will be lost if a subsequent :meth:`deploy` call refreshes the
            VM from the chain DB.

        Args:
            address: 20-byte address.
            amount:  New balance in wei.
        """
        self._vm.state.set_balance(address, amount)

    def set_storage(self, address: bytes, slot: int, value: int) -> None:
        """Write *value* to contract storage slot *slot* at *address*.

        Useful for pre-populating contract state without going through
        function calls.

        .. note::
            Like :meth:`set_code`, this mutation is applied directly to
            ``self._vm.state`` and is **not** persisted to the chain database.
            It will be lost if a subsequent :meth:`deploy` call refreshes the
            VM from the chain DB.

        Args:
            address: 20-byte contract address.
            slot:    Storage slot index.
            value:   Value to store (uint256).
        """
        self._vm.state.set_storage(address, slot, value)

    def get_storage(self, address: bytes, slot: int) -> int:
        """Read a contract storage slot.

        Args:
            address: 20-byte contract address.
            slot:    Storage slot index.

        Returns:
            The stored uint256 value (0 for uninitialised slots).
        """
        return self._vm.state.get_storage(address, slot)

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def call(
        self,
        to: bytes,
        calldata: bytes = b"",
        *,
        value: int = 0,
        gas: int = _CTX_DEFAULT_GAS,
    ) -> EVMResult:
        """Call a deployed contract and return the result.

        The call originates from :attr:`deployer`.  Storage mutations made by
        the call persist within the current context (they are visible to
        subsequent :meth:`call` and :meth:`run_program` invocations).

        Args:
            to:       20-byte target contract address.
            calldata: ABI-encoded calldata bytes.
            value:    ETH value (in wei) to forward with the call.
            gas:      Gas limit (default :data:`_CTX_DEFAULT_GAS`).

        Returns:
            :class:`EVMResult` with ``.output``, ``.gas_used``, ``.is_error``.
        """
        comp = self._apply_computation(to, self.deployer, calldata, value, gas)
        return EVMResult(
            output=comp.output,
            gas_used=comp.get_gas_used(),
            is_error=comp.is_error,
        )

    def run_program(
        self,
        bytecode: bytes,
        *,
        sender: Optional[bytes] = None,
        value: int = 0,
        gas: int = _CTX_DEFAULT_GAS,
    ) -> EVMResult:
        """Execute a DeFiVM program via the deployed :class:`DeFiVM` contract.

        Calls ``DeFiVM.execute(bytecode)``, which DELEGATECALLs to the
        Analog-Labs EVM interpreter.  ``address(this)`` inside the program
        equals :attr:`program_executor`.  Storage mutations persist and are
        visible to subsequent :meth:`call` and :meth:`run_program` calls.

        Args:
            bytecode:  EVM bytecode to execute.
            sender:    ``msg.sender`` for the call; defaults to :attr:`deployer`.
            value:     ETH value (in wei) forwarded to the execution.
            gas:       Gas limit (default :data:`_CTX_DEFAULT_GAS`).

        Returns:
            :class:`EVMResult` with ``.output``, ``.gas_used``, ``.is_error``.
        """
        effective_sender = sender if sender is not None else self.deployer
        execute_calldata = _DeFiVM.fns.execute(bytecode).data
        comp = self._apply_computation(
            to=self.program_executor,
            sender=effective_sender,
            calldata=execute_calldata,
            value=value,
            gas=gas,
        )
        return EVMResult(
            output=comp.output,
            gas_used=comp.get_gas_used(),
            is_error=comp.is_error,
        )

    # -----------------------------------------------------------------------
    # Convenience helpers — MockToken
    # -----------------------------------------------------------------------

    def deploy_mock_token(self) -> bytes:
        """Compile and deploy a minimal mintable ERC-20 token.

        The token supports ``mint``, ``transfer``, ``transferFrom``, and
        ``approve`` / ``allowance``.  It has no owner checks — any caller
        can mint.

        Returns:
            The 20-byte address of the deployed ``MockToken`` contract.
        """
        return self.deploy(_get_mock_token_bin())

    def mint_token(self, token: bytes, recipient: bytes, amount: int) -> None:
        """Mint *amount* of *token* to *recipient*.

        Calls ``MockToken.mint(recipient, amount)`` on the deployed *token*
        contract.  Raises ``AssertionError`` if the call reverts.

        Args:
            token:     20-byte address of a :meth:`deploy_mock_token` token.
            recipient: 20-byte address of the recipient.
            amount:    Token amount (in the token's smallest unit).
        """
        calldata = bytes(ERC20.fns.mint("0x" + recipient.hex(), amount).data)
        result = self.call(token, calldata)
        assert not result.is_error, f"mint_token failed: {result.output.hex()}"

    def token_balance(self, token: bytes, holder: bytes) -> int:
        """Return the ERC-20 balance of *holder* in *token*.

        Args:
            token:  20-byte address of the token contract.
            holder: 20-byte address of the account to query.

        Returns:
            Token balance as a Python ``int``.
        """
        fn = ERC20.fns.balanceOf("0x" + holder.hex())
        result = self.call(token, bytes(fn.data))
        assert not result.is_error, f"token_balance failed: {result.output.hex()}"
        return fn.decode(result.output)


# ---------------------------------------------------------------------------
# pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def evm_ctx() -> MiniEVMContext:
    """Create a fresh :class:`MiniEVMContext` for each test.

    Tests that need to deploy contracts or set up token balances before
    running DeFiVM programs should use this fixture::

        def test_token_balance(evm_ctx):
            token = evm_ctx.deploy_mock_token()
            evm_ctx.mint_token(token, evm_ctx.program_executor, 1000 * 10**18)

            result = evm_ctx.run_program(
                self_addr() + push_addr(token.hex()) + balance_of() + RETURN_TOP
            )
            assert not result.is_error
    """
    return MiniEVMContext()

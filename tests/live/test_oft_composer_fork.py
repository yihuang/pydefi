"""Fork tests for OFTComposer — LayerZero OFT compose receiver backed by DeFiVM.

These tests compile OFTComposer.sol and DeFiVM.sol with py-solc-x, deploy them
alongside mock contracts on a local Anvil fork of Ethereum mainnet, and exercise
the full ``lzCompose`` flow including:

 - Basic compose execution via a mock LayerZero endpoint
 - Multi-call compose execution (two CALL instructions in one program)
 - Compose execution carrying ETH value to a sub-call
 - Dynamic access to ``amountLD`` inside the program (PATCH_U256)
 - Dynamic access to ``_from`` (OFT address) inside the program (PATCH_ADDR)
 - Revert when the caller is not the authorised endpoint
 - Revert when a sub-call inside the compose fails
 - Owner rescue of stuck ETH and ERC-20 tokens

Run with::

    pytest -m fork tests/live/test_oft_composer_fork.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm.program import (
    call,
    load_reg,
    patch_addr,
    patch_u256,
    pop,
    push_addr,
    push_bytes,
    push_u256,
    store_reg,
)

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "OFTComposer.sol"
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Compile + deploy helpers
# ---------------------------------------------------------------------------


def _ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc once (no-op if already installed)."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def _compile_oft_composer() -> dict:
    """Compile OFTComposer.sol and return the ABI + bytecode."""
    _ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(SOL_FILE)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(":OFTComposer"))
    return result[key]


def _compile_defi_vm() -> dict:
    """Compile DeFiVM.sol and return the ABI + bytecode."""
    _ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(DEFI_VM_SOL_FILE)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(":DeFiVM"))
    return result[key]


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: str, *args) -> str:
    """Deploy a contract and return its address."""
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Mock contracts Solidity source (compiled inline at test-module load time)
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mock LayerZero EndpointV2.
///         Allows tests to manually trigger lzCompose on a registered composer.
contract MockEndpoint {
    /// @notice Deliver a compose message to a composer contract.
    function deliverCompose(
        address _composer,
        address _from,
        bytes32 _guid,
        bytes calldata _message
    ) external payable {
        (bool ok, bytes memory err) = _composer.call{value: msg.value}(
            abi.encodeWithSignature(
                "lzCompose(address,bytes32,bytes,address,bytes)",
                _from,
                _guid,
                _message,
                address(this),
                bytes("")
            )
        );
        if (!ok) {
            assembly { revert(add(err, 32), mload(err)) }
        }
    }
}

/// @notice Minimal mock OFT token (ERC-20 subset with mint).
///         token() returns address(this) — simulates a native OFT that IS an ERC-20.
contract MockOFT {
    string public name = "Mock OFT";
    string public symbol = "MOFT";
    uint8 public decimals = 18;

    mapping(address => uint256) public balanceOf;
    uint256 public totalSupply;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    /// @notice Returns address(this) — this OFT contract is itself the ERC-20.
    function token() external view returns (address) {
        return address(this);
    }
}

/// @notice Mock OFT Adapter — wraps a separate ERC-20 token.
///         token() returns the underlying ERC-20 address (not address(this)).
///         Simulates an OFT Adapter for a pre-existing ERC-20 token.
contract MockOFTAdapter {
    address private immutable _token;

    constructor(address token_) {
        _token = token_;
    }

    function token() external view returns (address) {
        return _token;
    }
}

/// @notice Mock target contract — records the most recent call and emits an event.
contract MockTarget {
    event Called(address sender, uint256 value, bytes data);

    uint256 public callCount;
    bytes public lastData;
    uint256 public lastValue;

    function execute(bytes calldata data) external payable returns (bool) {
        callCount++;
        lastData = data;
        lastValue = msg.value;
        emit Called(msg.sender, msg.value, data);
        return true;
    }

    receive() external payable {}
}

/// @notice Mock target that always reverts — used to test sub-call failure handling.
contract RevertingTarget {
    error AlwaysReverts();

    fallback() external payable {
        revert AlwaysReverts();
    }
}
"""


def _compile_mock_contracts() -> dict[str, dict]:
    """Compile mock contracts and return {name: {abi, bin}} mapping."""
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_CONTRACTS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return {
        "MockEndpoint": result["<stdin>:MockEndpoint"],
        "MockOFT": result["<stdin>:MockOFT"],
        "MockOFTAdapter": result["<stdin>:MockOFTAdapter"],
        "MockTarget": result["<stdin>:MockTarget"],
        "RevertingTarget": result["<stdin>:RevertingTarget"],
    }


# ---------------------------------------------------------------------------
# Compose-message helpers
# ---------------------------------------------------------------------------


def make_compose_message(
    nonce: int,
    src_eid: int,
    amount_ld: int,
    program: bytes,
) -> bytes:
    """Build a LayerZero OFTComposeMsgCodec-encoded message with a DeFiVM program.

    Layout::

        | 8B nonce | 4B srcEid | 32B amountLD | DeFiVM program |

    Args:
        nonce:     uint64 message nonce.
        src_eid:   uint32 source endpoint ID.
        amount_ld: uint256 amount of OFT tokens delivered (in local decimals).
        program:   Raw DeFiVM bytecode (``_from`` and ``amountLD`` are pre-pushed
                   by OFTComposer before the program runs).

    Returns:
        Raw bytes ready to pass as ``_message`` in ``lzCompose``.
    """
    return (
        struct.pack(">Q", nonce)  # 8 bytes  — uint64 nonce
        + struct.pack(">I", src_eid)  # 4 bytes  — uint32 srcEid
        + amount_ld.to_bytes(32, "big")  # 32 bytes — uint256 amountLD
        + program  # DeFiVM bytecode
    )


# ---------------------------------------------------------------------------
# Module-scoped Anvil fork fixture (shared across all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def oft_fork_w3(fork_w3_module):
    """Module-scoped Anvil mainnet fork, shared across all tests in this module."""
    return fork_w3_module


# ---------------------------------------------------------------------------
# Module-scoped setup: compile + deploy once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_oft_composer():
    return _compile_oft_composer()


@pytest.fixture(scope="module")
def compiled_mocks():
    return _compile_mock_contracts()


@pytest.fixture(scope="module")
def compiled_defi_vm():
    return _compile_defi_vm()


@pytest.fixture(scope="module")
async def ctx(oft_fork_w3, compiled_oft_composer, compiled_mocks, compiled_defi_vm):
    """Deploy OFTComposer, DeFiVM, and mock contracts once; return shared context."""
    w3 = oft_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy mock endpoint (controls which address may call lzCompose).
    endpoint_address = await _deploy(w3, compiled_mocks["MockEndpoint"], deployer)

    # Deploy DeFiVM (no constructor arguments).
    vm_address = await _deploy(w3, compiled_defi_vm, deployer)

    # Deploy OFT composer pointing it at the mock endpoint and DeFiVM.
    composer_address = await _deploy(
        w3,
        compiled_oft_composer,
        deployer,
        endpoint_address,  # _endpoint
        vm_address,  # _vm
        deployer,  # _owner
    )
    composer = w3.eth.contract(address=composer_address, abi=compiled_oft_composer["abi"])
    endpoint = w3.eth.contract(address=endpoint_address, abi=compiled_mocks["MockEndpoint"]["abi"])

    # Deploy mock OFT (no approval needed — any OFT may trigger compose).
    oft_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)

    # Deploy mock target contracts.
    target_address = await _deploy(w3, compiled_mocks["MockTarget"], deployer)
    reverting_address = await _deploy(w3, compiled_mocks["RevertingTarget"], deployer)

    target = w3.eth.contract(address=target_address, abi=compiled_mocks["MockTarget"]["abi"])

    return {
        "w3": w3,
        "accounts": accounts,
        "deployer": deployer,
        "composer": composer,
        "composer_address": composer_address,
        "endpoint": endpoint,
        "endpoint_address": endpoint_address,
        "vm_address": vm_address,
        "oft_address": oft_address,
        "target": target,
        "target_address": target_address,
        "reverting_address": reverting_address,
        "compiled_mocks": compiled_mocks,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _abidata(hex_or_bytes: str | bytes) -> bytes:
    """Convert encode_abi() hex output to raw bytes."""
    if isinstance(hex_or_bytes, bytes):
        return hex_or_bytes
    return bytes.fromhex(hex_or_bytes.removeprefix("0x"))


@pytest.mark.fork
class TestOFTComposerFork:
    """Fork-level tests for OFTComposer.sol backed by DeFiVM on a local Anvil fork."""

    # ------------------------------------------------------------------
    # Helper: build a single-CALL DeFiVM snippet (no OFT param setup)
    # ------------------------------------------------------------------

    @staticmethod
    def _call_target(target_address: str, calldata: bytes, value: int = 0) -> bytes:
        """Return DeFiVM instructions for one external call (discards success flag)."""
        return (
            push_bytes(calldata)  # buf N; stack: [..., N]
            + push_u256(value)  # stack: [..., N, value]
            + push_addr(target_address)  # stack: [..., N, value, to]
            + push_u256(0)  # gasLimit=0 (all gas); stack: [..., N, value, to, 0]
            + call()  # pops top-4; requireSuccess=True -> reverts on failure; pushes 1
            + pop()  # discard success flag
        )

    # ------------------------------------------------------------------
    # Basic single-call compose
    # ------------------------------------------------------------------

    async def test_single_call_compose(self, ctx):
        """lzCompose runs a DeFiVM program that calls MockTarget.execute()."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        calldata = _abidata(target.encode_abi("execute", [b"hello"]))
        # The composer pre-pushes amountLD and _from onto the stack.
        # Save them to R0/_from and R1/amountLD so the stack is clean for the call.
        program = store_reg(0) + store_reg(1) + self._call_target(target_address, calldata)
        amount_ld = 10**18
        message = make_compose_message(nonce=1, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        call_count = await target.functions.callCount().call()
        assert call_count == 1
        assert await target.functions.lastData().call() == b"hello"

    # ------------------------------------------------------------------
    # Multi-call compose (two CALL instructions in one program)
    # ------------------------------------------------------------------

    async def test_multi_call_compose(self, ctx):
        """Program with two sequential CALL instructions increments callCount by 2."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        before = await target.functions.callCount().call()

        calldata_a = _abidata(target.encode_abi("execute", [b"call_a"]))
        calldata_b = _abidata(target.encode_abi("execute", [b"call_b"]))
        program = (
            store_reg(0)
            + store_reg(1)
            + self._call_target(target_address, calldata_a)
            + self._call_target(target_address, calldata_b)
        )
        amount_ld = 5 * 10**17
        message = make_compose_message(nonce=2, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 31 + b"\x01",
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        after = await target.functions.callCount().call()
        assert after == before + 2

    # ------------------------------------------------------------------
    # Compose with ETH value forwarded to a sub-call
    # ------------------------------------------------------------------

    async def test_compose_with_eth_value(self, ctx):
        """ETH sent with deliverCompose is forwarded to a sub-call via msg.value."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        eth_amount = 10**16  # 0.01 ETH
        amount_ld = 10**18

        before_balance = await w3.eth.get_balance(target_address)

        calldata = _abidata(target.encode_abi("execute", [b"with_eth"]))
        # Pass eth_amount as the call value; DeFiVM forwards it from its own balance
        # (received via vm.execute{value: msg.value}).
        program = (
            store_reg(0)
            + store_reg(1)
            + push_bytes(calldata)
            + push_u256(eth_amount)  # value for sub-call
            + push_addr(target_address)
            + push_u256(0)  # gasLimit=0
            + call()
            + pop()
        )
        message = make_compose_message(nonce=3, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        # Send ETH with the compose delivery; it flows: endpoint -> lzCompose -> vm.execute
        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 31 + b"\x02",
            message,
        ).transact({"from": deployer, "value": eth_amount})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(target_address) == before_balance + eth_amount
        assert await target.functions.lastValue().call() == eth_amount

    # ------------------------------------------------------------------
    # Composed event is emitted
    # ------------------------------------------------------------------

    async def test_composed_event_emitted(self, ctx):
        """lzCompose emits Composed(from, guid, amountLD) with correct values."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        amount_ld = 777 * 10**18
        guid = b"\xde\xad" + b"\x00" * 30
        calldata = _abidata(target.encode_abi("execute", [b"event_test"]))
        program = store_reg(0) + store_reg(1) + self._call_target(target_address, calldata)
        message = make_compose_message(nonce=4, src_eid=30184, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            guid,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        events = composer.events.Composed().process_receipt(receipt)
        assert len(events) == 1
        evt = events[0]["args"]
        assert evt["from"] == oft_address
        assert evt["guid"] == guid
        assert evt["amountLD"] == amount_ld

    # ------------------------------------------------------------------
    # Program can read amountLD from the initial stack
    # ------------------------------------------------------------------

    async def test_compose_accesses_amount_ld(self, ctx):
        """Program uses PATCH_U256 to write amountLD (from the stack) into calldata."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        amount_ld = 42 * 10**18

        # Build a calldata template for MockTarget.execute(bytes data) with 32 zero bytes.
        # ABI layout:
        #   [0:4]    selector
        #   [4:36]   ABI offset = 0x20 (32)
        #   [36:68]  data length = 32
        #   [68:100] data content (32 zero bytes -- will be patched with amountLD)
        template = _abidata(target.encode_abi("execute", [b"\x00" * 32]))

        # Program:
        #   Stack start: [amountLD, _from]  (_from on top)
        #   STORE_REG 0 -> R0 = _from;   stack: [amountLD]
        #   STORE_REG 1 -> R1 = amountLD; stack: []
        #   PUSH_BYTES template -> buf 0; stack: [0]
        #   LOAD_REG 1  -> stack: [0, amountLD]
        #   PATCH_U256 68 -> pops amountLD (top) and bufIdx 0; patches; stack: [0]
        #   push value=0, push to, push gasLimit=0 -> CALL
        program = (
            store_reg(0)  # R0 = _from
            + store_reg(1)  # R1 = amountLD
            + push_bytes(template)  # buf 0; stack: [0]
            + load_reg(1)  # stack: [0, amountLD]
            + patch_u256(68)  # patch amountLD at offset 68; stack: [0]
            + push_u256(0)  # value=0
            + push_addr(target_address)  # to
            + push_u256(0)  # gasLimit=0
            + call()
            + pop()
        )
        message = make_compose_message(nonce=5, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # MockTarget.lastData is the `data` argument -- the 32-byte content that was patched.
        last_data = await target.functions.lastData().call()
        assert last_data == amount_ld.to_bytes(32, "big")

    # ------------------------------------------------------------------
    # Program can read _from (OFT address) from the initial stack
    # ------------------------------------------------------------------

    async def test_compose_accesses_from_address(self, ctx):
        """Program uses PATCH_ADDR to write _from (OFT address) into calldata."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        # Same calldata template; we'll patch the 20-byte address at offset 68.
        template = _abidata(target.encode_abi("execute", [b"\x00" * 32]))

        # Program:
        #   STORE_REG 0 -> R0 = _from   (pop from top)
        #   STORE_REG 1 -> R1 = amountLD
        #   PUSH_BYTES template -> buf 0; stack: [0]
        #   LOAD_REG 0  -> stack: [0, _from]
        #   PATCH_ADDR 68 -> writes bytes20(_from) at offset 68; stack: [0]
        #   CALL
        program = (
            store_reg(0)  # R0 = _from
            + store_reg(1)  # R1 = amountLD
            + push_bytes(template)  # buf 0; stack: [0]
            + load_reg(0)  # stack: [0, _from]
            + patch_addr(68)  # write 20 bytes of _from at offset 68; stack: [0]
            + push_u256(0)  # value=0
            + push_addr(target_address)  # to
            + push_u256(0)  # gasLimit=0
            + call()
            + pop()
        )
        amount_ld = 10**18
        message = make_compose_message(nonce=6, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=ctx["compiled_mocks"]["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # PATCH_ADDR writes exactly 20 bytes at offset 68; the remaining 12 bytes stay zero.
        last_data = await target.functions.lastData().call()
        expected_addr_bytes = bytes.fromhex(oft_address.lower().removeprefix("0x"))
        assert last_data[:20] == expected_addr_bytes
        assert last_data[20:] == b"\x00" * 12

    # ------------------------------------------------------------------
    # Security: unauthorized endpoint
    # ------------------------------------------------------------------

    async def test_unauthorized_endpoint_reverts(self, ctx):
        """lzCompose reverts when called by an address that is not the endpoint."""
        composer = ctx["composer"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]

        program = store_reg(0) + store_reg(1)  # minimal no-op program
        message = make_compose_message(nonce=7, src_eid=30101, amount_ld=10**18, program=program)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.functions.lzCompose(
                oft_address,
                b"\x00" * 32,
                message,
                deployer,
                b"",
            ).transact({"from": deployer})

    # ------------------------------------------------------------------
    # Security: sub-call failure rolls back all state changes
    # ------------------------------------------------------------------

    async def test_sub_call_failure_reverts_compose(self, ctx):
        """A failing CALL in the program reverts the entire compose transaction."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        reverting_address = ctx["reverting_address"]
        target = ctx["target"]
        compiled_mocks = ctx["compiled_mocks"]

        before_count = await target.functions.callCount().call()

        calldata_ok = _abidata(target.encode_abi("execute", [b"before_fail"]))
        # First call succeeds; second call (to RevertingTarget) always reverts.
        # DeFiVM's requireSuccess=True causes the whole execute() to revert.
        program = (
            store_reg(0)
            + store_reg(1)
            + self._call_target(target_address, calldata_ok)  # succeeds, pops success
            + push_bytes(b"")  # empty calldata for fallback
            + push_u256(0)
            + push_addr(reverting_address)
            + push_u256(0)
            + call()  # requireSuccess=True; target reverts -> DeFiVM reverts
        )
        amount_ld = 10**18
        message = make_compose_message(nonce=8, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = w3.eth.contract(address=oft_address, abi=compiled_mocks["MockOFT"]["abi"])
        await oft.functions.mint(composer.address, amount_ld).transact({"from": deployer})

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await endpoint.functions.deliverCompose(
                composer.address,
                oft_address,
                b"\x00" * 32,
                message,
            ).transact({"from": deployer})

        # callCount increment from the first CALL must have been rolled back.
        after_count = await target.functions.callCount().call()
        assert after_count == before_count

    # ------------------------------------------------------------------
    # Admin: any OFT can trigger compose (no whitelist)
    # ------------------------------------------------------------------

    async def test_any_oft_can_compose(self, ctx):
        """Any OFT address forwarded by the endpoint may trigger lzCompose."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        random_oft = w3.eth.account.create().address

        calldata = _abidata(target.encode_abi("execute", [b"random_oft"]))
        program = store_reg(0) + store_reg(1) + self._call_target(target_address, calldata)
        # amount_ld=0 skips token transfer; this test only verifies there is no OFT whitelist.
        message = make_compose_message(nonce=9, src_eid=30101, amount_ld=0, program=program)

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            random_oft,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Admin: rescue stuck ETH
    # ------------------------------------------------------------------

    async def test_rescue_eth(self, ctx):
        """Owner can rescue ETH stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]

        eth_amount = 5 * 10**16  # 0.05 ETH

        await w3.eth.send_transaction({"from": deployer, "to": composer.address, "value": eth_amount})

        before_composer = await w3.eth.get_balance(composer.address)
        assert before_composer >= eth_amount

        fresh_recipient = w3.eth.account.create().address
        assert await w3.eth.get_balance(fresh_recipient) == 0

        tx = await composer.functions.rescueETH(fresh_recipient, eth_amount).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(composer.address) == before_composer - eth_amount
        assert await w3.eth.get_balance(fresh_recipient) == eth_amount

    async def test_non_owner_cannot_rescue_eth(self, ctx):
        """rescueETH reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        fresh_recipient = w3.eth.account.create().address

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.functions.rescueETH(fresh_recipient, 1).transact({"from": non_owner})

    # ------------------------------------------------------------------
    # Admin: rescue stuck ERC-20 tokens
    # ------------------------------------------------------------------

    async def test_rescue_token(self, ctx):
        """Owner can rescue ERC-20 tokens stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]
        compiled_mocks = ctx["compiled_mocks"]

        fresh_recipient = w3.eth.account.create().address
        token_amount = 100 * 10**18

        token_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
        token = w3.eth.contract(address=token_address, abi=compiled_mocks["MockOFT"]["abi"])
        await token.functions.mint(composer.address, token_amount).transact({"from": deployer})

        assert await token.functions.balanceOf(composer.address).call() == token_amount

        tx = await composer.functions.rescueToken(token_address, fresh_recipient, token_amount).transact(
            {"from": deployer}
        )
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert await token.functions.balanceOf(composer.address).call() == 0
        assert await token.functions.balanceOf(fresh_recipient).call() == token_amount

    async def test_non_owner_cannot_rescue_token(self, ctx):
        """rescueToken reverts when called by a non-owner."""
        composer = ctx["composer"]
        accounts = ctx["accounts"]
        oft_address = ctx["oft_address"]

        non_owner = accounts[1]
        fresh_recipient = "0x000000000000000000000000000000000000dEaD"

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.functions.rescueToken(oft_address, fresh_recipient, 1).transact({"from": non_owner})

    # ------------------------------------------------------------------
    # Token transfer: composer forwards OFT tokens to DeFiVM before execute
    # ------------------------------------------------------------------

    async def test_token_transfer_to_vm(self, ctx):
        """lzCompose transfers OFT tokens from the composer to DeFiVM before execution.

        Flow:
          1. Tokens arrive at the composer (minted here to simulate OFT bridge delivery).
          2. ``lzCompose`` transfers ``amountLD`` tokens from composer → DeFiVM.
          3. The DeFiVM program forwards the tokens to a fresh recipient via a CALL.
          4. After execution the recipient holds the tokens; composer and DeFiVM are empty.
        """
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        vm_address = ctx["vm_address"]
        compiled_mocks = ctx["compiled_mocks"]

        token_amount = 50 * 10**18
        fresh_recipient = w3.eth.account.create().address

        # Wrap MockOFT in a contract object so we can encode ABI and check balances.
        oft = w3.eth.contract(address=oft_address, abi=compiled_mocks["MockOFT"]["abi"])

        # Record pre-test balances; the module-scoped fixture may carry residual
        # tokens from earlier tests (e.g. a reverted lzCompose that left tokens at
        # the composer) or accumulated tokens at the vm.
        pre_composer = await oft.functions.balanceOf(composer.address).call()
        pre_vm = await oft.functions.balanceOf(vm_address).call()

        # Simulate OFT bridge: tokens land in the composer before lzCompose is called.
        await oft.functions.mint(composer.address, token_amount).transact({"from": deployer})
        assert await oft.functions.balanceOf(composer.address).call() == pre_composer + token_amount

        # Build calldata for token.transfer(fresh_recipient, token_amount).
        # After the composer's token transfer, DeFiVM holds the tokens and can use them.
        vm_forward_calldata = _abidata(oft.encode_abi("transfer", [fresh_recipient, token_amount]))
        program = (
            store_reg(0)  # R0 = _from (OFT address)
            + store_reg(1)  # R1 = amountLD
            + push_bytes(vm_forward_calldata)  # push calldata buffer; stack: [buf_idx]
            + push_u256(0)  # value = 0 ETH
            + push_addr(oft_address)  # call the OFT token contract
            + push_u256(0)  # gasLimit = 0 (all gas)
            + call()
            + pop()  # discard success flag
        )
        message = make_compose_message(nonce=10, src_eid=30101, amount_ld=token_amount, program=program)

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,  # _from = the OFT contract that delivered the tokens
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Tokens must have been forwarded through DeFiVM to the fresh recipient.
        assert await oft.functions.balanceOf(fresh_recipient).call() == token_amount
        # Composer lost exactly amountLD; its residual from earlier tests is unchanged.
        assert await oft.functions.balanceOf(composer.address).call() == pre_composer
        # DeFiVM gained exactly amountLD then spent it all; its prior balance is unchanged.
        assert await oft.functions.balanceOf(vm_address).call() == pre_vm

    async def test_token_transfer_to_vm_oft_adapter(self, ctx):
        """lzCompose resolves the ERC-20 token via IOFT.token() for an OFT Adapter.

        An OFT Adapter wraps a pre-existing ERC-20 token; its ``token()`` method returns
        the address of that underlying ERC-20, not ``address(this)``.  This test verifies
        that ``OFTComposer`` calls ``IOFT(_from).token()`` and transfers the correct token.

        Flow:
          1. Deploy a standalone ERC-20 token (MockOFT used as plain ERC-20).
          2. Deploy MockOFTAdapter wrapping that token — ``adapter.token()`` returns the
             standalone ERC-20 address.
          3. Mint tokens to the composer (simulating OFT Adapter bridge delivery).
          4. Call ``lzCompose`` with ``_from = adapter``; the composer resolves the ERC-20
             via ``adapter.token()`` and transfers it to DeFiVM.
          5. The DeFiVM program forwards the tokens to a fresh recipient.
          6. Assert the recipient holds all tokens; composer and DeFiVM are empty.
        """
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        compiled_mocks = ctx["compiled_mocks"]

        token_amount = 75 * 10**18
        fresh_recipient = w3.eth.account.create().address

        # Deploy a standalone ERC-20 token (reuse MockOFT — it IS an ERC-20).
        token_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
        token = w3.eth.contract(address=token_address, abi=compiled_mocks["MockOFT"]["abi"])

        # Deploy an OFT Adapter that wraps the standalone token.
        adapter_address = await _deploy(w3, compiled_mocks["MockOFTAdapter"], deployer, token_address)

        # Simulate OFT Adapter bridge: tokens land in the composer.
        await token.functions.mint(composer.address, token_amount).transact({"from": deployer})
        assert await token.functions.balanceOf(composer.address).call() == token_amount

        # Build calldata for token.transfer(fresh_recipient, token_amount).
        vm_forward_calldata = _abidata(token.encode_abi("transfer", [fresh_recipient, token_amount]))
        program = (
            store_reg(0)  # R0 = _from (adapter address)
            + store_reg(1)  # R1 = amountLD
            + push_bytes(vm_forward_calldata)  # push calldata buffer; stack: [buf_idx]
            + push_u256(0)  # value = 0 ETH
            + push_addr(token_address)  # call the underlying ERC-20 contract
            + push_u256(0)  # gasLimit = 0 (all gas)
            + call()
            + pop()  # discard success flag
        )
        message = make_compose_message(nonce=11, src_eid=30101, amount_ld=token_amount, program=program)

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            adapter_address,  # _from = the OFT Adapter (not the ERC-20 itself)
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Tokens must have been forwarded through DeFiVM to the fresh recipient.
        assert await token.functions.balanceOf(fresh_recipient).call() == token_amount
        # Neither the composer nor DeFiVM should retain any tokens.
        assert await token.functions.balanceOf(composer.address).call() == 0
        assert await token.functions.balanceOf(vm_address).call() == 0

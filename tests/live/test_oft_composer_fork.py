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
from collections.abc import Sequence
from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3, Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.types import Address
from pydefi.vm import Program
from tests.live.sol_utils import compile_sol_file, deploy, deploy_mock_v3_pool, ensure_solc


def _start_program() -> tuple[Program, "object", "object"]:
    """Start a compose program and read the two transient-storage params.

    The OFTComposer stages OFT parameters in DeFiVM's transient store:
    slot 0 = amountLD, slot 1 = _from.  The program reads them via TLOAD.

    Returns ``(prog, from_val, amount_val)``.
    """
    prog = Program()
    amount_val = prog.builder.tload(IRLiteral(0))  # amountLD
    from_val = prog.builder.tload(IRLiteral(1))  # _from
    return prog, from_val, amount_val


def _compose_single_call(target_address: Address, calldata: bytes, *, value: int = 0) -> bytes:
    """Build a compose program that issues a single external call.

    Matches legacy ``call(require_success=True)`` semantics: a failed sub-call
    propagates a revert to the outer ``lzCompose`` / ``receiveAndExecute``.
    """
    prog, _from_val, _amount_val = _start_program()
    success = prog.call_raw(target_address, calldata, value=value)
    prog.assert_(success)
    prog.builder.stop()
    return prog.build()


def _compose_multi_call(calls: Sequence[tuple[Address, bytes]]) -> bytes:
    """Build a compose program that issues N external calls in sequence."""
    prog, _from_val, _amount_val = _start_program()
    for target, calldata in calls:
        success = prog.call_raw(target, calldata)
        prog.assert_(success)
    prog.builder.stop()
    return prog.build()


def _compose_noop() -> bytes:
    """Build a minimal no-op compose program (prologue stores only)."""
    prog, _from_val, _amount_val = _start_program()
    prog.builder.stop()
    return prog.build()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "OFTComposer.sol"
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Local compile + deploy wrappers
# ---------------------------------------------------------------------------


def _compile_oft_composer() -> dict:
    """Compile OFTComposer.sol and return the ABI + bytecode."""
    return compile_sol_file(SOL_FILE, "OFTComposer")


def _compile_defi_vm() -> dict:
    """Compile DeFiVM.sol and return the ABI + bytecode."""
    return compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: Address, *args) -> Address:
    """Deploy a contract and return its address."""
    return await deploy(w3, compiled, deployer, *args)


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
    ensure_solc("0.8.24")
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
async def ctx(oft_fork_w3, compiled_oft_composer, compiled_mocks, compiled_defi_vm, interpreter_addr):
    """Deploy OFTComposer, DeFiVM, and mock contracts once; return shared context."""
    w3 = oft_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy mock endpoint (controls which address may call lzCompose).
    endpoint_address = await _deploy(w3, compiled_mocks["MockEndpoint"], deployer)

    # Deploy DeFiVM (kept for tests that exercise DeFiVM.execute directly).
    vm_address = await _deploy(w3, compiled_defi_vm, deployer, interpreter_addr)

    # Deploy OFT composer.  Composer DELEGATECALLs the interpreter directly,
    # so it takes the interpreter address (not the VM address).
    composer_address = await _deploy(
        w3,
        compiled_oft_composer,
        deployer,
        endpoint_address,  # _endpoint
        interpreter_addr,  # _interpreter
        deployer,  # _owner
    )
    composer = Contract(abi=compiled_oft_composer["abi"], tx={"to": Web3.to_checksum_address(composer_address)})
    endpoint = Contract(
        abi=compiled_mocks["MockEndpoint"]["abi"], tx={"to": Web3.to_checksum_address(endpoint_address)}
    )

    # Deploy mock OFT (no approval needed — any OFT may trigger compose).
    oft_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)

    # Deploy mock target contracts.
    target_address = await _deploy(w3, compiled_mocks["MockTarget"], deployer)
    reverting_address = await _deploy(w3, compiled_mocks["RevertingTarget"], deployer)

    target = Contract(abi=compiled_mocks["MockTarget"]["abi"], tx={"to": Web3.to_checksum_address(target_address)})

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


@pytest.mark.fork
class TestOFTComposerFork:
    """Fork-level tests for OFTComposer.sol backed by DeFiVM on a local Anvil fork."""

    # ------------------------------------------------------------------
    # Basic single-call compose
    # ------------------------------------------------------------------

    async def test_single_call_compose(self, ctx):
        """lzCompose runs a DeFiVM program that calls MockTarget.execute()."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        calldata = target.fns.execute(b"hello").data
        program = _compose_single_call(target_address, calldata)
        amount_ld = 10**18
        message = make_compose_message(nonce=1, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        call_count = await target.fns.callCount().call(w3)
        assert call_count == 1
        assert await target.fns.lastData().call(w3) == b"hello"

    # ------------------------------------------------------------------
    # Multi-call compose (two CALL instructions in one program)
    # ------------------------------------------------------------------

    async def test_multi_call_compose(self, ctx):
        """Program with two sequential CALL instructions increments callCount by 2."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        before = await target.fns.callCount().call(w3)

        calldata_a = target.fns.execute(b"call_a").data
        calldata_b = target.fns.execute(b"call_b").data
        program = _compose_multi_call([(target_address, calldata_a), (target_address, calldata_b)])
        amount_ld = 5 * 10**17
        message = make_compose_message(nonce=2, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 31 + b"\x01",
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        after = await target.fns.callCount().call(w3)
        assert after == before + 2

    # ------------------------------------------------------------------
    # Compose with ETH value forwarded to a sub-call
    # ------------------------------------------------------------------

    async def test_compose_with_eth_value(self, ctx):
        """ETH sent with deliverCompose is forwarded to a sub-call via msg.value."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        eth_amount = 10**16  # 0.01 ETH
        amount_ld = 10**18

        before_balance = await w3.eth.get_balance(target_address)

        calldata = target.fns.execute(b"with_eth").data
        # Pass eth_amount as the call value; DeFiVM forwards it from its own balance
        # (received via vm.execute{value: msg.value}).
        program = _compose_single_call(target_address, calldata, value=eth_amount)
        message = make_compose_message(nonce=3, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        # Send ETH with the compose delivery; it flows: endpoint -> lzCompose -> vm.execute
        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 31 + b"\x02",
            message,
        ).transact(w3, deployer, value=eth_amount)

        receipt = tx
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(target_address) == before_balance + eth_amount
        assert await target.fns.lastValue().call(w3) == eth_amount

    # ------------------------------------------------------------------
    # Composed event is emitted
    # ------------------------------------------------------------------

    async def test_composed_event_emitted(self, ctx):
        """lzCompose emits Composed(from, guid, amountLD) with correct values."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        amount_ld = 777 * 10**18
        guid = b"\xde\xad" + b"\x00" * 30
        calldata = target.fns.execute(b"event_test").data
        program = _compose_single_call(target_address, calldata)
        message = make_compose_message(nonce=4, src_eid=30184, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            guid,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        events = composer.events.Composed.parse_logs(receipt["logs"])
        assert len(events) == 1
        evt = events[0]["args"]
        assert HexBytes(evt["from"]) == oft_address
        assert evt["guid"] == guid
        assert evt["amountLD"] == amount_ld

    # ------------------------------------------------------------------
    # Program can read amountLD from the initial stack
    # ------------------------------------------------------------------

    async def test_compose_accesses_amount_ld(self, ctx):
        """Program uses PATCH_U256 to write amountLD (from the stack) into calldata."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
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
        template = target.fns.execute(b"\x00" * 32).data

        # Patch amountLD (stored in R1 by the prologue) into offset 68 of the
        # calldata buffer before calling target.execute().
        prog, _from_val, amount_val = _start_program()
        success = prog.call_raw(target_address, template, patches={68: amount_val})
        prog.assert_(success)
        prog.builder.stop()
        program = prog.build()
        message = make_compose_message(nonce=5, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        # MockTarget.lastData is the `data` argument -- the 32-byte content that was patched.
        last_data = await target.fns.lastData().call(w3)
        assert last_data == amount_ld.to_bytes(32, "big")

    # ------------------------------------------------------------------
    # Program can read _from (OFT address) from the initial stack
    # ------------------------------------------------------------------

    async def test_compose_accesses_from_address(self, ctx):
        """Program uses PATCH_ADDR to write _from (OFT address) into calldata."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        # Same calldata template; patch the 20-byte address into data[12..31].
        template = target.fns.execute(b"\x00" * 32).data

        # Patch _from (R0 from prologue, a uint160 address) into offset 68.
        # SSA patches always MSTORE 32 bytes at the slot start, which writes
        # 12 zero bytes then the 20-byte address — equivalent to the legacy
        # patch_value(80, 20) which used mstore_off = 80 - 12 = 68.
        prog, from_val, _amount_val = _start_program()
        success = prog.call_raw(target_address, template, patches={68: from_val})
        prog.assert_(success)
        prog.builder.stop()
        program = prog.build()
        amount_ld = 10**18
        message = make_compose_message(nonce=6, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=ctx["compiled_mocks"]["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        # PATCH_ADDR(80) does MSTORE(buf+68, _from): writes 12 leading zeros then
        # the 20-byte address into data[0..31].
        last_data = await target.fns.lastData().call(w3)
        assert last_data[:12] == b"\x00" * 12
        assert last_data[12:32] == oft_address

    # ------------------------------------------------------------------
    # Security: unauthorized endpoint
    # ------------------------------------------------------------------

    async def test_unauthorized_endpoint_reverts(self, ctx):
        """lzCompose reverts when called by an address that is not the endpoint."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        ctx["composer_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]

        program = _compose_noop()  # minimal no-op program
        message = make_compose_message(nonce=7, src_eid=30101, amount_ld=10**18, program=program)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.lzCompose(
                oft_address,
                b"\x00" * 32,
                message,
                deployer,
                b"",
            ).transact(w3, deployer)

    # ------------------------------------------------------------------
    # Security: sub-call failure rolls back all state changes
    # ------------------------------------------------------------------

    async def test_sub_call_failure_reverts_compose(self, ctx):
        """A failing CALL in the program reverts the entire compose transaction."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        reverting_address = ctx["reverting_address"]
        target = ctx["target"]
        compiled_mocks = ctx["compiled_mocks"]

        before_count = await target.fns.callCount().call(w3)

        calldata_ok = target.fns.execute(b"before_fail").data
        # First call succeeds; second call (to RevertingTarget) always reverts.
        # require_success=True on the second call causes the whole execute() to revert.
        program = _compose_multi_call([(target_address, calldata_ok), (reverting_address, b"")])
        amount_ld = 10**18
        message = make_compose_message(nonce=8, src_eid=30101, amount_ld=amount_ld, program=program)

        # Simulate OFT bridge: mint tokens to the composer before lzCompose is called.
        oft = Contract(abi=compiled_mocks["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})
        await oft.fns.mint(composer_address, amount_ld).transact(w3, deployer)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await endpoint.fns.deliverCompose(
                composer_address,
                oft_address,
                b"\x00" * 32,
                message,
            ).transact(w3, deployer)

        # callCount increment from the first CALL must have been rolled back.
        after_count = await target.fns.callCount().call(w3)
        assert after_count == before_count

    # ------------------------------------------------------------------
    # Admin: any OFT can trigger compose (no whitelist)
    # ------------------------------------------------------------------

    async def test_any_oft_can_compose(self, ctx):
        """Any OFT address forwarded by the endpoint may trigger lzCompose."""
        w3 = ctx["w3"]
        ctx["composer"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        ctx["endpoint_address"]
        deployer = ctx["deployer"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        random_oft = w3.eth.account.create().address

        calldata = target.fns.execute(b"random_oft").data
        program = _compose_single_call(target_address, calldata)
        # amount_ld=0 skips token transfer; this test only verifies there is no OFT whitelist.
        message = make_compose_message(nonce=9, src_eid=30101, amount_ld=0, program=program)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            random_oft,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Admin: rescue stuck ETH
    # ------------------------------------------------------------------

    async def test_rescue_eth(self, ctx):
        """Owner can rescue ETH stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]

        eth_amount = 5 * 10**16  # 0.05 ETH

        await w3.eth.send_transaction({"from": deployer, "to": composer_address, "value": eth_amount})

        before_composer = await w3.eth.get_balance(composer_address)
        assert before_composer >= eth_amount

        fresh_recipient = w3.eth.account.create().address
        assert await w3.eth.get_balance(fresh_recipient) == 0

        tx = await composer.fns.rescueETH(fresh_recipient, eth_amount).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(composer_address) == before_composer - eth_amount
        assert await w3.eth.get_balance(fresh_recipient) == eth_amount

    async def test_non_owner_cannot_rescue_eth(self, ctx):
        """rescueETH reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        ctx["composer_address"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        fresh_recipient = w3.eth.account.create().address

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.rescueETH(fresh_recipient, 1).transact(w3, non_owner)

    # ------------------------------------------------------------------
    # Admin: rescue stuck ERC-20 tokens
    # ------------------------------------------------------------------

    async def test_rescue_token(self, ctx):
        """Owner can rescue ERC-20 tokens stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        compiled_mocks = ctx["compiled_mocks"]

        fresh_recipient = w3.eth.account.create().address
        token_amount = 100 * 10**18

        token_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
        token = Contract(abi=compiled_mocks["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(token_address)})
        await token.fns.mint(composer_address, token_amount).transact(w3, deployer)

        assert await token.fns.balanceOf(composer_address).call(w3) == token_amount

        tx = await composer.fns.rescueToken(token_address, fresh_recipient, token_amount).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        assert await token.fns.balanceOf(composer_address).call(w3) == 0
        assert await token.fns.balanceOf(fresh_recipient).call(w3) == token_amount

    async def test_non_owner_cannot_rescue_token(self, ctx):
        """rescueToken reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        ctx["composer_address"]
        accounts = ctx["accounts"]
        oft_address = ctx["oft_address"]

        non_owner = accounts[1]
        fresh_recipient: Address = Address("0x000000000000000000000000000000000000dEaD")

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.rescueToken(oft_address, fresh_recipient, 1).transact(w3, non_owner)

    # ------------------------------------------------------------------
    # Token transfer: composer forwards OFT tokens to DeFiVM before execute
    # ------------------------------------------------------------------

    async def test_program_spends_tokens_held_by_composer(self, ctx):
        """lzCompose runs the program in composer's context; tokens delivered by the
        OFT land at the composer and are spent by the program.

        Flow:
          1. Tokens arrive at the composer (minted here to simulate OFT delivery).
          2. ``lzCompose`` DELEGATECALLs the interpreter — program runs as composer.
          3. The program calls ``token.transfer(recipient, amountLD)`` which sends
             tokens from the composer to a fresh recipient.
        """
        w3 = ctx["w3"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        compiled_mocks = ctx["compiled_mocks"]

        token_amount = 50 * 10**18
        fresh_recipient = w3.eth.account.create().address

        oft = Contract(abi=compiled_mocks["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})

        pre_composer = await oft.fns.balanceOf(composer_address).call(w3)

        # Simulate OFT delivery: tokens land at the composer before lzCompose.
        await oft.fns.mint(composer_address, token_amount).transact(w3, deployer)

        transfer_calldata = oft.fns.transfer(fresh_recipient, token_amount).data
        program = _compose_single_call(oft_address, transfer_calldata)
        message = make_compose_message(nonce=10, src_eid=30101, amount_ld=token_amount, program=program)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        assert await oft.fns.balanceOf(fresh_recipient).call(w3) == token_amount
        # Composer received then sent — net delta is zero.
        assert await oft.fns.balanceOf(composer_address).call(w3) == pre_composer

    async def test_program_spends_tokens_via_oft_adapter_delivery(self, ctx):
        """OFT Adapter delivery: underlying ERC-20 lands at composer; program spends it.

        With the DELEGATECALL design the composer no longer asks the OFT for its
        underlying token via ``IOFT.token()`` — whichever ERC-20 the bridge actually
        delivered is already on the composer's balance.  ``_from`` is exposed to
        the program through transient slot 1 if it needs to identify the source.
        """
        w3 = ctx["w3"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        compiled_mocks = ctx["compiled_mocks"]

        token_amount = 75 * 10**18
        fresh_recipient = w3.eth.account.create().address

        # Standalone ERC-20 (reuse MockOFT — it IS an ERC-20).
        token_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
        token = Contract(abi=compiled_mocks["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(token_address)})

        # OFT Adapter is only used as ``_from`` in the compose call now; the
        # composer no longer dereferences it.
        adapter_address = await _deploy(w3, compiled_mocks["MockOFTAdapter"], deployer, token_address)

        # Simulate adapter delivery: underlying ERC-20 lands directly at composer.
        await token.fns.mint(composer_address, token_amount).transact(w3, deployer)
        assert await token.fns.balanceOf(composer_address).call(w3) == token_amount

        transfer_calldata = token.fns.transfer(fresh_recipient, token_amount).data
        program = _compose_single_call(token_address, transfer_calldata)
        message = make_compose_message(nonce=11, src_eid=30101, amount_ld=token_amount, program=program)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            adapter_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)

        receipt = tx
        assert receipt["status"] == 1

        assert await token.fns.balanceOf(fresh_recipient).call(w3) == token_amount
        assert await token.fns.balanceOf(composer_address).call(w3) == 0

    async def test_v3_swap_callback_routed_through_composer(self, ctx):
        """Regression: program runs in composer context and triggers a V3 swap;
        the pool callbacks back into the composer, whose inherited
        ``DEXCallbackRouter.fallback`` must dispatch ``uniswapV3SwapCallback``
        and pay ``amountIn`` of ``tokenIn`` to the pool.
        """
        w3 = ctx["w3"]
        composer_address = ctx["composer_address"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        compiled_mocks = ctx["compiled_mocks"]

        oft = Contract(abi=compiled_mocks["MockOFT"]["abi"], tx={"to": Web3.to_checksum_address(oft_address)})

        token1_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
        pool_address, pool_abi = await deploy_mock_v3_pool(w3, deployer, oft_address, token1_address)
        pool = Contract(abi=pool_abi, tx={"to": Web3.to_checksum_address(pool_address)})

        amount_in = 50 * 10**18

        # Simulate OFT delivery: tokens land at composer.
        await oft.fns.mint(composer_address, amount_in).transact(w3, deployer)

        from eth_abi.abi import encode as abi_encode

        callback_data = abi_encode(["address"], [oft_address])
        swap_calldata = pool.fns.swap(composer_address, True, amount_in, 0, callback_data).data
        program = _compose_single_call(pool_address, swap_calldata)
        message = make_compose_message(nonce=12, src_eid=30101, amount_ld=amount_in, program=program)

        pre_composer = await oft.fns.balanceOf(composer_address).call(w3)
        pre_pool = await oft.fns.balanceOf(pool_address).call(w3)

        tx = await endpoint.fns.deliverCompose(
            composer_address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact(w3, deployer)
        assert tx["status"] == 1

        # Composer paid amount_in via the V3 callback; pool received it.
        assert await oft.fns.balanceOf(composer_address).call(w3) == pre_composer - amount_in
        assert await oft.fns.balanceOf(pool_address).call(w3) == pre_pool + amount_in

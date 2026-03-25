"""Fork tests for DeFiVM — a minimal register-based macro-assembler for DeFi flows.

These tests compile DeFiVM.sol with py-solc-x, deploy it on a local Anvil fork
of Ethereum mainnet, and exercise the full instruction set including:

 - Stack / register instructions (PUSH_U256, PUSH_ADDR, PUSH_BYTES, DUP, SWAP, POP,
   LOAD_REG, STORE_REG)
 - Control flow (JUMP, JUMPI, REVERT_IF, ASSERT_GE, ASSERT_LE)
 - External calls to a whitelisted mock adapter (CALL)
 - Balance introspection (BALANCE_OF, SELF_BAL, DELTA_START, DELTA_LOAD) using
   real on-chain WETH contract on the forked chain
 - ABI patching (PATCH_U256, PATCH_ADDR, RET_U256, RET_SLICE)

Run with::

    pytest -m fork tests/live/test_defi_vm_fork.py
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

import pytest
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, Web3RPCError

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

from .conftest import ETH_RPC_URL  # noqa: E402

# Well-known mainnet addresses used in fork tests
WETH_MAINNET = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
# Coinbase 8 — a well-funded address on mainnet (used for introspection only)
WHALE = "0x77134cbC06cB00b66F4c7e623D5fdBF6777635EC"

# ---------------------------------------------------------------------------
# Opcodes (mirrors the constants in DeFiVM.sol)
# ---------------------------------------------------------------------------
OP_PUSH_U256 = 0x01
OP_PUSH_ADDR = 0x02
OP_PUSH_BYTES = 0x03
OP_DUP = 0x04
OP_SWAP = 0x05
OP_POP = 0x06
OP_LOAD_REG = 0x10
OP_STORE_REG = 0x11
OP_JUMP = 0x20
OP_JUMPI = 0x21
OP_REVERT_IF = 0x22
OP_ASSERT_GE = 0x23
OP_ASSERT_LE = 0x24
OP_CALL = 0x30
OP_BALANCE_OF = 0x31
OP_SELF_BAL = 0x32
OP_DELTA_START = 0x33
OP_DELTA_LOAD = 0x34
OP_PATCH_U256 = 0x40
OP_PATCH_ADDR = 0x41
OP_RET_U256 = 0x42
OP_RET_SLICE = 0x43


# ---------------------------------------------------------------------------
# Program builder helpers
# ---------------------------------------------------------------------------


def u256(n: int) -> bytes:
    """Encode a uint256 as 32 big-endian bytes."""
    return n.to_bytes(32, "big")


def addr(a: str) -> bytes:
    """Encode a checksummed / hex Ethereum address as 20 bytes."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    assert len(raw) == 20, f"bad address length: {a!r}"
    return raw


def u16(n: int) -> bytes:
    return struct.pack(">H", n)


def push_u256(n: int) -> bytes:
    return bytes([OP_PUSH_U256]) + u256(n)


def push_addr(a: str) -> bytes:
    return bytes([OP_PUSH_ADDR]) + addr(a)


def push_bytes(data: bytes) -> bytes:
    assert len(data) <= 0xFFFF
    return bytes([OP_PUSH_BYTES]) + u16(len(data)) + data


def dup() -> bytes:
    return bytes([OP_DUP])


def swap() -> bytes:
    return bytes([OP_SWAP])


def pop() -> bytes:
    return bytes([OP_POP])


def load_reg(i: int) -> bytes:
    return bytes([OP_LOAD_REG, i])


def store_reg(i: int) -> bytes:
    return bytes([OP_STORE_REG, i])


def jump(target: int) -> bytes:
    return bytes([OP_JUMP]) + u16(target)


def jumpi(target: int) -> bytes:
    return bytes([OP_JUMPI]) + u16(target)


def revert_if(msg: str) -> bytes:
    raw = msg.encode()
    assert len(raw) <= 255
    return bytes([OP_REVERT_IF, len(raw)]) + raw


def assert_ge(msg: str = "") -> bytes:
    raw = msg.encode()
    assert len(raw) <= 255
    return bytes([OP_ASSERT_GE, len(raw)]) + raw


def assert_le(msg: str = "") -> bytes:
    raw = msg.encode()
    assert len(raw) <= 255
    return bytes([OP_ASSERT_LE, len(raw)]) + raw


def call(require_success: bool = True) -> bytes:
    """Emit a CALL opcode.

    Caller must have pushed on the stack (top to bottom):
        gasLimit (uint256)  <- top
        to       (address)
        value    (uint256)
        calldataBufIdx (bytes)
    """
    flags = 0x01 if require_success else 0x00
    return bytes([OP_CALL, flags])


def balance_of() -> bytes:
    """BALANCE_OF – pop: token, account → push balance."""
    return bytes([OP_BALANCE_OF])


def self_bal() -> bytes:
    return bytes([OP_SELF_BAL])


def delta_start() -> bytes:
    """DELTA_START – pop: token, account → snapshot."""
    return bytes([OP_DELTA_START])


def delta_load() -> bytes:
    """DELTA_LOAD – pop: token, account → push delta."""
    return bytes([OP_DELTA_LOAD])


def patch_u256(offset: int) -> bytes:
    return bytes([OP_PATCH_U256]) + u16(offset)


def patch_addr(offset: int) -> bytes:
    return bytes([OP_PATCH_ADDR]) + u16(offset)


def ret_u256(offset: int) -> bytes:
    return bytes([OP_RET_U256]) + u16(offset)


def ret_slice(offset: int, length: int) -> bytes:
    return bytes([OP_RET_SLICE]) + u16(offset) + u16(length)


# ---------------------------------------------------------------------------
# Compile + deploy helpers
# ---------------------------------------------------------------------------


def _ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc once (no-op if already installed)."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def _compile_defi_vm() -> dict:
    """Compile DeFiVM.sol and return the ABI + bytecode."""
    _ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(SOL_FILE)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    # solcx may return relative or absolute path as the key prefix;
    # find the DeFiVM entry regardless.
    key = next(k for k in result if k.endswith(":DeFiVM"))
    return result[key]


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: str) -> str:
    """Deploy a contract and return its address."""
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor().transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Mock adapter Solidity source (compiled inline)
# ---------------------------------------------------------------------------

MOCK_ADAPTER_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mock adapter for DeFiVM tests.
contract MockAdapter {
    event Called(address sender, uint256 value, bytes data);

    /// Echoes back its calldata as returndata, and emits an event.
    fallback() external payable {
        emit Called(msg.sender, msg.value, msg.data);
        assembly {
            calldatacopy(0, 0, calldatasize())
            return(0, calldatasize())
        }
    }

    receive() external payable {}

    /// Helper: returns uint256(42) always, for RET_U256 tests.
    function getFortyTwo() external pure returns (uint256) {
        return 42;
    }
}
"""


def _compile_mock_adapter() -> dict:
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        MOCK_ADAPTER_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    key = "<stdin>:MockAdapter"
    return result[key]


# ---------------------------------------------------------------------------
# Module-scoped Anvil fork fixture (shared across all tests in this file)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
async def vm_fork_w3():
    """Start a single Anvil mainnet fork for the entire test module."""
    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        ["anvil", "--fork-url", ETH_RPC_URL, "--port", str(port), "--silent"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            await w3.eth.chain_id
            break
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.25)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        pytest.fail("Anvil did not start within 30 seconds")

    yield w3

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Module-scoped setup: deploy once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_vm():
    return _compile_defi_vm()


@pytest.fixture(scope="module")
def compiled_adapter():
    return _compile_mock_adapter()


@pytest.fixture(scope="module")
async def ctx(vm_fork_w3, compiled_vm, compiled_adapter):
    """Deploy DeFiVM + MockAdapter once, whitelist the adapter, return context dict."""
    w3 = vm_fork_w3

    accounts = await w3.eth.accounts
    deployer = accounts[0]

    vm_address = await _deploy(w3, compiled_vm, deployer)
    adapter_address = await _deploy(w3, compiled_adapter, deployer)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])

    tx = await vm.functions.setAdapter(adapter_address, True).transact({"from": deployer})
    await w3.eth.get_transaction_receipt(tx)

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "adapter_address": adapter_address,
        "deployer": deployer,
        "accounts": accounts,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestDeFiVMFork:
    """Fork-level tests for DeFiVM.sol on a local Anvil mainnet fork."""

    # ------------------------------------------------------------------
    # Deployment / admin
    # ------------------------------------------------------------------

    async def test_deploy_and_ownership(self, ctx):
        """DeFiVM deploys correctly and owner is the deployer."""
        vm = ctx["vm"]
        owner = await vm.functions.owner().call()
        assert owner.lower() == ctx["deployer"].lower()

    async def test_set_adapter_whitelists(self, ctx):
        """setAdapter enables the mock adapter."""
        vm = ctx["vm"]
        enabled = await vm.functions.adapters(ctx["adapter_address"]).call()
        assert enabled is True

    async def test_non_owner_cannot_set_adapter(self, ctx):
        """Non-owner address cannot whitelist adapters."""
        vm = ctx["vm"]
        other = ctx["accounts"][1]
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.setAdapter(ctx["adapter_address"], False).transact({"from": other})

    # ------------------------------------------------------------------
    # Stack / register instructions
    # ------------------------------------------------------------------

    async def test_push_and_store_load_register(self, ctx):
        """PUSH_U256 / STORE_REG / LOAD_REG — basic register round-trip."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(0xDEADBEEF) + store_reg(0) + load_reg(0) + pop()

        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_dup_swap_pop(self, ctx):
        """DUP, SWAP, POP instructions execute without revert."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(1) + push_u256(2) + dup() + swap() + pop() + pop() + pop()
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    async def test_jump_forward(self, ctx):
        """JUMP skips over a subsequent unknown opcode that would otherwise revert."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(1)  # 33 bytes
        bad_byte = bytes([0xFF])  # unknown opcode — would revert if reached
        pop_part = pop()

        target = len(push_part) + 3 + len(bad_byte)
        jump_part = jump(target)

        program = push_part + jump_part + bad_byte + pop_part
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_jumpi_taken(self, ctx):
        """JUMPI jumps when the condition is non-zero."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(1)  # 33 bytes
        bad_byte = bytes([0xFF])

        target = len(push_part) + 3 + len(bad_byte)
        jumpi_part = jumpi(target)

        program = push_part + jumpi_part + bad_byte
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_jumpi_not_taken(self, ctx):
        """JUMPI does not jump when the condition is zero."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(0)  # condition = False
        target = len(push_u256(99)) + len(push_part) + 3 + 1
        jumpi_part = jumpi(target)
        skip_pop = pop()  # pops the dummy 99 when condition is False

        program = push_u256(99) + push_part + jumpi_part + skip_pop

        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_revert_if_triggers(self, ctx):
        """REVERT_IF causes a revert when condition is non-zero."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(1) + revert_if("slippage exceeded")
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

    async def test_revert_if_no_trigger(self, ctx):
        """REVERT_IF does nothing when condition is zero."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(0) + revert_if("should not trigger")
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_assert_ge_pass(self, ctx):
        """ASSERT_GE passes when a >= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        # push b=100, push a=200 -> assert a >= b -> ok
        program = push_u256(100) + push_u256(200) + assert_ge("min not met")
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_assert_ge_fail(self, ctx):
        """ASSERT_GE reverts when a < b."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(200) + push_u256(100) + assert_ge("min not met")
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

    async def test_assert_le_pass(self, ctx):
        """ASSERT_LE passes when a <= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_u256(200) + push_u256(100) + assert_le("max exceeded")
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def test_self_balance(self, ctx):
        """SELF_BAL executes and emits ProgramExecuted."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = self_bal() + pop()
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_balance_of_weth(self, ctx):
        """BALANCE_OF can read WETH balance of a mainnet whale."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = push_addr(WHALE) + push_addr(WETH_MAINNET) + balance_of() + pop()
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_delta_balance_weth(self, ctx):
        """DELTA_START / DELTA_LOAD measures zero delta when no transfer occurs."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = (
            push_addr(WHALE)
            + push_addr(WETH_MAINNET)
            + delta_start()
            + push_addr(WHALE)
            + push_addr(WETH_MAINNET)
            + delta_load()
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # External CALL
    # ------------------------------------------------------------------

    async def test_call_whitelisted_adapter(self, ctx):
        """CALL succeeds for a whitelisted adapter."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        selector = keccak(b"getFortyTwo()")[:4]
        calldata = bytes(selector)

        program = (
            push_bytes(calldata) + push_u256(0) + push_addr(adapter) + push_u256(0) + call(require_success=True) + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_call_non_whitelisted_reverts(self, ctx):
        """CALL to a non-whitelisted address must revert."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        random_addr = "0x000000000000000000000000000000000000dEaD"

        program = (
            push_bytes(b"\x00\x00\x00\x00")
            + push_u256(0)
            + push_addr(random_addr)
            + push_u256(0)
            + call(require_success=False)
            + pop()
        )
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

    async def test_ret_u256_from_adapter(self, ctx):
        """RET_U256 reads a uint256 from the last call's returndata."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        selector = keccak(b"getFortyTwo()")[:4]
        calldata = bytes(selector)

        program = (
            push_bytes(calldata)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            + ret_u256(0)
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_ret_slice(self, ctx):
        """RET_SLICE extracts a bytes chunk from the last call's returndata."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        selector = keccak(b"getFortyTwo()")[:4]
        calldata = bytes(selector)

        program = (
            push_bytes(calldata)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            + ret_slice(0, 32)
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # ABI patching
    # ------------------------------------------------------------------

    async def test_patch_u256_and_call(self, ctx):
        """PATCH_U256 mutates a calldata template before calling the adapter."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        selector = keccak(b"getFortyTwo()")[:4]
        template = bytearray(selector + b"\x00" * 32)

        program = (
            push_bytes(bytes(template))
            + push_u256(0xABCD)
            + patch_u256(4)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_patch_addr(self, ctx):
        """PATCH_ADDR writes a 20-byte address into a calldata template."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        selector = keccak(b"getFortyTwo()")[:4]
        template = bytearray(selector + b"\x00" * 32)

        patch_offset = 4 + 12

        program = (
            push_bytes(bytes(template))
            + push_addr(adapter)
            + patch_addr(patch_offset)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Safety / limits
    # ------------------------------------------------------------------

    async def test_unknown_opcode_reverts(self, ctx):
        """An unrecognised opcode causes a revert."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = bytes([0xFF])
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

    async def test_stack_overflow_reverts(self, ctx):
        """Pushing 33 values onto the 32-element stack must revert."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        program = b"".join(push_u256(i) for i in range(33))
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

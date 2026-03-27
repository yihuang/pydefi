"""Fork tests for DeFiVM — a minimal register-based macro-assembler for DeFi flows.

These tests compile DeFiVM.sol with py-solc-x, deploy it on a local Anvil fork
of Ethereum mainnet, and exercise the full instruction set including:

 - Stack / register instructions (PUSH_U256, PUSH_ADDR, PUSH_BYTES, DUP, SWAP, POP,
   LOAD_REG, STORE_REG)
 - Control flow (JUMP, JUMPI, REVERT_IF, ASSERT_GE, ASSERT_LE)
 - External calls to a mock adapter (CALL)
 - Balance introspection (BALANCE_OF, SELF_ADDR, SUB) using
   real on-chain WETH contract on the forked chain
 - ABI patching (PATCH_U256, PATCH_ADDR, RET_U256, RET_SLICE)

Run with::

    pytest -m fork tests/live/test_defi_vm_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm.program import (
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
    swap,
)

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# Well-known mainnet addresses used in fork tests
WETH_MAINNET = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
# Coinbase 8 — a well-funded address on mainnet (used for introspection only)
WHALE = "0x77134cbC06cB00b66F4c7e623D5fdBF6777635EC"

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

    /// Returns the double of the input.  Used in chained-call tests.
    function double(uint256 x) external pure returns (uint256) {
        return x * 2;
    }

    /// Returns the sum of two inputs.  Used in multi-patch chained-call tests.
    function addInputs(uint256 a, uint256 b) external pure returns (uint256) {
        return a + b;
    }

    /// Returns the ABI-encoded calldata needed to call double(x) on this contract.
    /// Used to exercise the ret_slice calldata-surgery approach.
    function encodeDouble(uint256 x) external pure returns (bytes memory) {
        return abi.encodeWithSelector(MockAdapter.double.selector, x);
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


@pytest.fixture(scope="module")
async def vm_fork_w3(fork_w3_module):
    """Module-scoped Anvil mainnet fork, shared across all tests in this module."""
    return fork_w3_module


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
    """Deploy DeFiVM + MockAdapter once and return context dict."""
    w3 = vm_fork_w3

    accounts = await w3.eth.accounts
    deployer = accounts[0]

    vm_address = await _deploy(w3, compiled_vm, deployer)
    adapter_address = await _deploy(w3, compiled_adapter, deployer)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])

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
        """SELF_ADDR + BALANCE_OF gives the VM contract's own ETH balance."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        # SELF_ADDR pushes address(this); push_u256(0) = ETH token; BALANCE_OF pops token then account
        program = self_addr() + push_u256(0) + balance_of() + pop()
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
        """SUB computes a zero balance delta when no transfer occurs.

        Pattern: balance_of (pre) → balance_of (post) → sub
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        # Stack after each step:
        #   push WHALE + WETH + BALANCE_OF → [pre_bal]
        #   push WHALE + WETH + BALANCE_OF → [pre_bal, post_bal]
        #   sub → [post_bal - pre_bal]  (== 0 here since no transfer happened)
        program = (
            push_addr(WHALE)
            + push_addr(WETH_MAINNET)
            + balance_of()
            + push_addr(WHALE)
            + push_addr(WETH_MAINNET)
            + balance_of()
            + sub()
            + pop()
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # External CALL
    # ------------------------------------------------------------------

    async def test_call_adapter(self, ctx):
        """CALL succeeds for a deployed mock adapter."""
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

        # Use a dummy 4-byte selector that doesn't match any explicit function,
        # so the call is routed to MockAdapter.fallback() which emits Called and echoes calldata.
        selector = b"\xde\xad\xbe\xef"
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

        # Verify PATCH_U256 actually wrote 0xABCD by decoding the Called event.
        # MockAdapter.fallback() emits Called(sender, value, data) and echoes calldata.
        called_topic = keccak(b"Called(address,uint256,bytes)")
        adapter_log = None
        for log in receipt["logs"]:
            if log["address"].lower() == adapter.lower() and log["topics"][0] == called_topic:
                adapter_log = log
                break
        assert adapter_log is not None, "Expected Called event from adapter"
        # ABI layout of data: sender(32) + value(32) + offset(32) + length(32) + calldata
        encoded = bytes(adapter_log["data"])
        calldata_len = int.from_bytes(encoded[96:128], "big")
        received_calldata = encoded[128 : 128 + calldata_len]
        expected_calldata = selector + (0xABCD).to_bytes(32, "big")
        assert received_calldata == expected_calldata

    async def test_patch_addr(self, ctx):
        """PATCH_ADDR writes a 20-byte address into a calldata template."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        # Use a dummy 4-byte selector to trigger MockAdapter.fallback()
        # which emits Called(sender, value, data) and echoes calldata.
        selector = b"\xca\xfe\xba\xbe"
        template = bytearray(selector + b"\x00" * 32)

        # patch_offset=16: write the 20-byte address at byte 16 of the buffer,
        # which fills the right half of the 32-byte ABI slot starting at offset 4.
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

        # Verify PATCH_ADDR wrote the address bytes at the correct offset.
        called_topic = keccak(b"Called(address,uint256,bytes)")
        adapter_log = None
        for log in receipt["logs"]:
            if log["address"].lower() == adapter.lower() and log["topics"][0] == called_topic:
                adapter_log = log
                break
        assert adapter_log is not None, "Expected Called event from adapter"
        encoded = bytes(adapter_log["data"])
        calldata_len = int.from_bytes(encoded[96:128], "big")
        received_calldata = encoded[128 : 128 + calldata_len]
        # Expected: selector + 12 zero bytes + 20-byte address (raw byte-for-byte patch)
        addr_bytes = bytes.fromhex(adapter.removeprefix("0x"))
        expected_calldata = selector + b"\x00" * 12 + addr_bytes
        assert received_calldata == expected_calldata

    # ------------------------------------------------------------------
    # Chained actions (calldata surgery)
    # ------------------------------------------------------------------

    async def test_chained_calls_patch_u256(self, ctx):
        """Chain two adapter calls by patching calldata with the previous output.

        Surgery approach 1 — ret_u256 + patch_u256:
          double(5) → 10, then patch template double(0) → double(10) → 20.
        The final retdata is verified in-program using ASSERT_GE / ASSERT_LE.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        double_sel = keccak(b"double(uint256)")[:4]
        calldata1 = double_sel + (5).to_bytes(32, "big")
        template2 = double_sel + (0).to_bytes(32, "big")  # placeholder; patched at runtime

        program = (
            # --- Call 1: double(5) → retdata = abi.encode(10) ---
            push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- Calldata surgery: embed call-1 output into call-2 template ---
            + push_bytes(template2)  # stack: [buf1]
            + ret_u256(0)  # push 10 from retdata; stack: [buf1, 10]
            + patch_u256(4)  # patch buf1[4..36] = abi.encode(10); stack: [buf1]
            # --- Call 2: double(10) → retdata = abi.encode(20) ---
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- In-program assertion: result == 20 ---
            + push_u256(20)
            + ret_u256(0)
            + assert_ge("result below expected")
            + push_u256(20)
            + ret_u256(0)
            + assert_le("result above expected")
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_chained_calls_multi_patch_u256(self, ctx):
        """Chain two adapter calls patching the first calldata slot of a two-argument template.

        Surgery approach 1 — ret_u256 + patch_u256:
          double(7) → 14, then patch first slot of addInputs(0, 3) → addInputs(14, 3) → 17.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        double_sel = keccak(b"double(uint256)")[:4]
        add_sel = keccak(b"addInputs(uint256,uint256)")[:4]

        calldata1 = double_sel + (7).to_bytes(32, "big")
        # Template: addInputs(0, 3) — first arg is a placeholder patched at runtime.
        template2 = add_sel + (0).to_bytes(32, "big") + (3).to_bytes(32, "big")

        program = (
            # --- Call 1: double(7) → retdata = abi.encode(14) ---
            push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- Calldata surgery: embed call-1 output into the first arg of call-2 ---
            + push_bytes(template2)  # stack: [buf1]
            + ret_u256(0)  # push 14; stack: [buf1, 14]
            + patch_u256(4)  # patch buf1[4..36] = abi.encode(14); stack: [buf1]
            # --- Call 2: addInputs(14, 3) → retdata = abi.encode(17) ---
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- In-program assertion: result == 17 ---
            + push_u256(17)
            + ret_u256(0)
            + assert_ge("result below expected")
            + push_u256(17)
            + ret_u256(0)
            + assert_le("result above expected")
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_chained_calls_ret_slice(self, ctx):
        """Chain two adapter calls by forwarding a raw calldata slice.

        Surgery approach 2 — ret_slice used directly as calldata:
          encodeDouble(5) returns the ABI calldata for double(5).
          ret_slice extracts that payload and feeds it straight into the next CALL,
          so double(5) is invoked without any additional patching.  Result: 10.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        from eth_utils import keccak

        # encodeDouble(uint256) returns bytes memory.
        # ABI layout of its returndata:
        #   [0..32)  offset of bytes value = 0x20
        #   [32..64) length of inner bytes = 36 (4-byte selector + 32-byte arg)
        #   [64..100) inner bytes = double(uint256) selector + abi.encode(x)
        encode_double_sel = keccak(b"encodeDouble(uint256)")[:4]
        calldata1 = encode_double_sel + (5).to_bytes(32, "big")

        program = (
            # --- Call 1: encodeDouble(5) → retdata carries calldata for double(5) ---
            push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- Surgery: extract the inner bytes as a new buffer ---
            # retdata[64..100] = selector(4) + abi.encode(5) = calldata for double(5)
            + ret_slice(64, 36)  # stack: [buf1]
            # --- Call 2: double(5) using the slice from call-1's returndata ---
            + push_u256(0)
            + push_addr(adapter)
            + push_u256(0)
            + call(require_success=True)
            + pop()
            # --- In-program assertion: result == 10 ---
            + push_u256(10)
            + ret_u256(0)
            + assert_ge("result below expected")
            + push_u256(10)
            + ret_u256(0)
            + assert_le("result above expected")
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

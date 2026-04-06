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
import solcx
from eth_contract.erc20 import ERC20
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm import Patch, Program
from pydefi.vm.program import (
    assert_ge,
    assert_le,
    balance_of,
    call,
    dup,
    jump,
    jumpi,
    load_reg,
    patch_value,
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
from tests.live.sol_utils import MOCK_TOKEN_SOL, compile_sol_file, compile_sol_source, deploy, ensure_solc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"
APPROVE_PROXY_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "ApproveProxy.sol"

# Well-known mainnet addresses used in fork tests
WETH_MAINNET = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
# Coinbase 8 — a well-funded address on mainnet (used for introspection only)
WHALE = "0x77134cbC06cB00b66F4c7e623D5fdBF6777635EC"


def _compile_defi_vm() -> dict:
    """Compile DeFiVM.sol and return the ABI + bytecode."""
    return compile_sol_file(SOL_FILE, "DeFiVM")


def _compile_approve_proxy() -> dict:
    """Compile ApproveProxy.sol and return ABI + bytecode."""
    return compile_sol_file(APPROVE_PROXY_SOL_FILE, "ApproveProxy")


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
    ensure_solc("0.8.24")
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
async def ctx(vm_fork_w3, compiled_vm, compiled_adapter, interpreter_addr):
    """Deploy DeFiVM + MockAdapter once and return context dict."""
    w3 = vm_fork_w3

    accounts = await w3.eth.accounts
    deployer = accounts[0]

    vm_address = await deploy(w3, compiled_vm, deployer, interpreter_addr)
    adapter_address = await deploy(w3, compiled_adapter, deployer)

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

    # jump() / jumpi() each emit PUSH2 hi lo JUMP/JUMPI = 4 bytes.
    _JUMP_INSTR_SIZE = 4

    async def test_jump_forward(self, ctx):
        """JUMP skips over a subsequent unknown opcode that would otherwise revert."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(1)  # 33 bytes (value stays on stack; consumed by pop_part)
        bad_byte = bytes([0xFF])  # unknown opcode — would revert if reached
        jumpdest = bytes([0x5B])  # JUMPDEST required by EVM at every jump target
        pop_part = pop()

        # Target must point to the JUMPDEST byte that follows the bad opcode.
        target = len(push_part) + self._JUMP_INSTR_SIZE + len(bad_byte)  # 33 + 4 + 1 = 38
        jump_part = jump(target)

        program = push_part + jump_part + bad_byte + jumpdest + pop_part
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_jumpi_taken(self, ctx):
        """JUMPI jumps when the condition is non-zero."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(1)  # 33 bytes — non-zero condition
        bad_byte = bytes([0xFF])
        jumpdest = bytes([0x5B])  # JUMPDEST required at jump target

        # Target must point to the JUMPDEST byte that follows the bad opcode.
        target = len(push_part) + self._JUMP_INSTR_SIZE + len(bad_byte)  # 33 + 4 + 1 = 38
        jumpi_part = jumpi(target)

        program = push_part + jumpi_part + bad_byte + jumpdest
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_jumpi_not_taken(self, ctx):
        """JUMPI does not jump when the condition is zero."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        push_part = push_u256(0)  # condition = False (33 bytes)
        # EVM only validates the jump destination when the condition is non-zero;
        # with condition=0 JUMPI simply falls through.  Any destination is safe here.
        UNUSED_TARGET = 0
        jumpi_part = jumpi(UNUSED_TARGET)
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for CALL
            + push_bytes(calldata)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
            + call(require_success=True)
            + pop()
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for CALL
            + push_bytes(calldata)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for CALL
            + push_bytes(calldata)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for CALL
            + push_bytes(bytes(template))
            + push_u256(0xABCD)
            + patch_value(4, 32)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for CALL
            + push_bytes(bytes(template))
            + push_addr(adapter)
            + patch_value(patch_offset, 20)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0
            + push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
            + call(require_success=True)
            + pop()
            # --- Calldata surgery: embed call-1 output into call-2 template ---
            # Push retLen/retOffset first so they sit below the call-2 buffer on stack
            + push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for call 2
            + push_bytes(template2)  # argsOffset, argsLen above retOffset/retLen
            + ret_u256(0)  # push 10 from retdata
            + patch_value(4, 32)  # patch template2[4..36] = 10; leaves [argsOffset, argsLen, retOffset, retLen]
            # --- Call 2: double(10) → retdata = abi.encode(20) ---
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0
            + push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
            + call(require_success=True)
            + pop()
            # --- Calldata surgery: embed call-1 output into the first arg of call-2 ---
            # Push retLen/retOffset first so they sit below the call-2 buffer on stack
            + push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for call 2
            + push_bytes(template2)  # argsOffset, argsLen above retOffset/retLen
            + ret_u256(0)  # push 14
            + patch_value(4, 32)  # patch template2[4..36] = 14; leaves [argsOffset, argsLen, retOffset, retLen]
            # --- Call 2: addInputs(14, 3) → retdata = abi.encode(17) ---
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
            push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0
            + push_bytes(calldata1)
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
            + call(require_success=True)
            + pop()
            # --- Surgery: extract the inner bytes as a new buffer ---
            # Push retLen/retOffset first so they sit below the slice buffer on stack
            + push_u256(0)
            + push_u256(0)  # retLen=0, retOffset=0 for call 2
            # retdata[64..100] = selector(4) + abi.encode(5) = calldata for double(5)
            + ret_slice(64, 36)  # argsOffset, argsLen above retOffset/retLen
            # --- Call 2: double(5) using the slice from call-1's returndata ---
            + push_u256(0)
            + push_addr(adapter)
            + bytes([0x5A])
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
        """Pushing more than 1024 values onto the EVM stack must revert."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        # The EVM allows up to 1024 stack items; 1025 PUSH1 instructions overflow it.
        program = bytes([0x60, 0x00]) * 1025  # 1025x PUSH1 0x00
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(program).transact({"from": deployer})

    # ------------------------------------------------------------------
    # call_contract_abi with Patch (high-level ABI builder)
    # ------------------------------------------------------------------

    async def test_call_contract_abi_patch_single_uint256(self, ctx):
        """call_contract_abi with a single Patch arg auto-detects the calldata offset.

        Stores the input value (7) in register 0, then calls double(7) via the
        high-level ABI builder with a Patch sourced from register 0.  Verifies
        the on-chain result equals 14.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        program = (
            push_u256(7)
            + store_reg(0)
            + Program()
            .call_contract_abi(
                adapter,
                "function double(uint256 x) external pure returns (uint256)",
                Patch(load_reg(0)),
            )
            .pop()
            .build()
            + push_u256(14)
            + ret_u256(0)
            + assert_ge("result below 14")
            + push_u256(14)
            + ret_u256(0)
            + assert_le("result above 14")
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_call_contract_abi_patch_two_uint256_args(self, ctx):
        """call_contract_abi with two Patch args patches both calldata slots.

        Stores 11 in register 0 and 6 in register 1, then calls addInputs(11, 6)
        via the high-level ABI builder with both args as Patch instances.
        Verifies the on-chain result equals 17.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        program = (
            push_u256(11)
            + store_reg(0)
            + push_u256(6)
            + store_reg(1)
            + Program()
            .call_contract_abi(
                adapter,
                "function addInputs(uint256 a, uint256 b) external pure returns (uint256)",
                Patch(load_reg(0)),
                Patch(load_reg(1)),
            )
            .pop()
            .build()
            + push_u256(17)
            + ret_u256(0)
            + assert_ge("result below 17")
            + push_u256(17)
            + ret_u256(0)
            + assert_le("result above 17")
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_call_contract_abi_patch_chained(self, ctx):
        """Chain two calls: first via static call_contract_abi, second with a Patch.

        double(5) → 10 stored in register 0, then double(10) via call_contract_abi
        with Patch(load_reg(0)).  Verifies the final result equals 20.
        """
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        double_sig = "function double(uint256 x) external pure returns (uint256)"

        program = (
            # Call 1: double(5) → result in retdata
            Program().call_contract_abi(adapter, double_sig, 5).pop().build()
            # Store result from retdata into register 0
            + ret_u256(0)
            + store_reg(0)
            # Call 2: double(reg0) using Patch
            + Program().call_contract_abi(adapter, double_sig, Patch(load_reg(0))).pop().build()
            # Verify result == 20
            + push_u256(20)
            + ret_u256(0)
            + assert_ge("result below 20")
            + push_u256(20)
            + ret_u256(0)
            + assert_le("result above 20")
        )
        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1


# ---------------------------------------------------------------------------
# Module-scoped fixture: deploy ApproveProxy + two MockTokens alongside DeFiVM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def proxy_ctx(vm_fork_w3, compiled_vm, interpreter_addr):
    """Deploy DeFiVM, ApproveProxy, and two MockTokens; return shared context."""
    w3 = vm_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]
    user = accounts[1]
    recipient = accounts[2]

    vm_address = await deploy(w3, compiled_vm, deployer, interpreter_addr)

    compiled_proxy = _compile_approve_proxy()
    proxy_address = await deploy(w3, compiled_proxy, deployer, vm_address)

    compiled_token = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    token_a_address = await deploy(w3, compiled_token, deployer)
    token_b_address = await deploy(w3, compiled_token, deployer)
    token_a = w3.eth.contract(address=token_a_address, abi=compiled_token["abi"])
    token_b = w3.eth.contract(address=token_b_address, abi=compiled_token["abi"])

    MINT_AMOUNT = 1_000 * 10**18
    for fn in [token_a.functions.mint(user, MINT_AMOUNT), token_b.functions.mint(user, MINT_AMOUNT)]:
        tx = await fn.transact({"from": deployer})
        await w3.eth.get_transaction_receipt(tx)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])
    proxy = w3.eth.contract(address=proxy_address, abi=compiled_proxy["abi"])

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "proxy": proxy,
        "proxy_address": proxy_address,
        "token_a": token_a,
        "token_a_address": token_a_address,
        "token_b": token_b,
        "token_b_address": token_b_address,
        "deployer": deployer,
        "user": user,
        "recipient": recipient,
        "mint_amount": MINT_AMOUNT,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestApproveProxyFork:
    """Fork-level tests for ApproveProxy.sol on a local Anvil mainnet fork."""

    async def test_single_deposit_and_transfer(self, proxy_ctx):
        """Proxy deposits one token into DeFiVM; program transfers it to recipient."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        vm_address = proxy_ctx["vm_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        AMOUNT = 100 * 10**18

        tx = await token_a.functions.approve(proxy_address, AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        bal_user_before = await token_a.functions.balanceOf(user).call()
        bal_recipient_before = await token_a.functions.balanceOf(recipient).call()
        bal_vm_before = await token_a.functions.balanceOf(vm_address).call()

        program = Program().call_contract(token_a_address, ERC20.fns.transfer(recipient, AMOUNT).data).pop().build()
        deposits = [{"token": token_a_address, "amount": AMOUNT}]

        tx = await proxy.functions.execute(program, deposits).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert await token_a.functions.balanceOf(user).call() == bal_user_before - AMOUNT
        assert await token_a.functions.balanceOf(recipient).call() == bal_recipient_before + AMOUNT
        assert await token_a.functions.balanceOf(vm_address).call() == bal_vm_before

    async def test_multiple_deposits(self, proxy_ctx):
        """Proxy deposits two different tokens into DeFiVM in one execute() call."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        token_b = proxy_ctx["token_b"]
        token_b_address = proxy_ctx["token_b_address"]
        vm_address = proxy_ctx["vm_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        AMOUNT_A = 50 * 10**18
        AMOUNT_B = 75 * 10**18

        for token, amount in [(token_a, AMOUNT_A), (token_b, AMOUNT_B)]:
            tx = await token.functions.approve(proxy_address, amount).transact({"from": user})
            await w3.eth.get_transaction_receipt(tx)

        bal_a_user_before = await token_a.functions.balanceOf(user).call()
        bal_b_user_before = await token_b.functions.balanceOf(user).call()
        bal_a_recipient_before = await token_a.functions.balanceOf(recipient).call()
        bal_b_recipient_before = await token_b.functions.balanceOf(recipient).call()

        program = (
            Program()
            .call_contract(token_a_address, ERC20.fns.transfer(recipient, AMOUNT_A).data)
            .pop()
            .call_contract(token_b_address, ERC20.fns.transfer(recipient, AMOUNT_B).data)
            .pop()
            .build()
        )
        deposits = [
            {"token": token_a_address, "amount": AMOUNT_A},
            {"token": token_b_address, "amount": AMOUNT_B},
        ]

        tx = await proxy.functions.execute(program, deposits).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert await token_a.functions.balanceOf(user).call() == bal_a_user_before - AMOUNT_A
        assert await token_b.functions.balanceOf(user).call() == bal_b_user_before - AMOUNT_B
        assert await token_a.functions.balanceOf(recipient).call() == bal_a_recipient_before + AMOUNT_A
        assert await token_b.functions.balanceOf(recipient).call() == bal_b_recipient_before + AMOUNT_B
        assert await token_a.functions.balanceOf(vm_address).call() == 0
        assert await token_b.functions.balanceOf(vm_address).call() == 0

    async def test_empty_deposits_succeeds(self, proxy_ctx):
        """execute() with an empty deposits list still runs the program."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        user = proxy_ctx["user"]

        program = push_u256(0) + pop()
        tx = await proxy.functions.execute(program, []).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_insufficient_allowance_reverts(self, proxy_ctx):
        """execute() reverts before running the program if a deposit allowance is too low."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        APPROVE_AMOUNT = 1
        DEPOSIT_AMOUNT = 1_000 * 10**18

        tx = await token_a.functions.approve(proxy_address, APPROVE_AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        program = (
            Program().call_contract(token_a_address, ERC20.fns.transfer(recipient, DEPOSIT_AMOUNT).data).pop().build()
        )
        deposits = [{"token": token_a_address, "amount": DEPOSIT_AMOUNT}]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.execute(program, deposits).transact({"from": user})

    async def test_vm_accessor(self, proxy_ctx):
        """proxy.vm() returns the paired DeFiVM address."""
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]

        stored_vm = await proxy.functions.vm().call()
        assert stored_vm.lower() == vm_address.lower()

    async def test_eth_forwarding(self, proxy_ctx):
        """ETH sent to proxy.execute() is forwarded to DeFiVM."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]
        user = proxy_ctx["user"]

        ETH_VALUE = 10**16  # 0.01 ETH

        vm_balance_before = await w3.eth.get_balance(vm_address)

        program = push_u256(0) + pop()
        tx = await proxy.functions.execute(program, []).transact({"from": user, "value": ETH_VALUE})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        vm_balance_after = await w3.eth.get_balance(vm_address)
        assert vm_balance_after - vm_balance_before == ETH_VALUE

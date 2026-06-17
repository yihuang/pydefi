"""Fork tests for DeFiVM — a minimal register-based macro-assembler for DeFi flows.

Compiles DeFiVM.sol and a mock adapter with py-solc-x, deploys on a local
Anvil mainnet fork, and exercises the SSA :class:`pydefi.vm.Program`
across:

 - Slot round-trips (alloc_slot / store_slot / load_slot)
 - Assertions (assert_, assert_ge, assert_le) with and without Error(string) msgs
 - ETH and ERC-20 balance introspection (eth_balance / erc20_balance_of)
 - External CALL with static calldata
 - Returndata access (returndata_word)
 - Runtime patching of calldata templates (``call_raw`` + ``patches=`` kwarg)
 - Chained calls (first call's output → second call's patched input)
 - High-level ABI builder (``call_contract``) with runtime-value args

Run with::

    pytest -m fork tests/live/test_defi_vm_fork.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import solcx
from eth_abi.abi import decode as abi_decode
from eth_contract.contract import ContractFunction
from eth_contract.erc20 import ERC20
from eth_contract.utils import send_transaction as eth_send_transaction
from eth_utils import keccak
from hexbytes import HexBytes
from web3.exceptions import ContractLogicError, Web3RPCError
from web3.types import Wei

from pydefi.abi.amm import UNISWAP_V3_POOL
from pydefi.amm.uniswap_v4 import UniswapV4
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.pathfinder.dag import RouteDAG
from pydefi.pathfinder.graph import PoolGraph, V3PoolEdge, V4PoolEdge
from pydefi.pathfinder.router import Router, _dag_leg_weights
from pydefi.types import ZERO_ADDRESS, Address, Token, TokenAmount
from pydefi.vm import Program
from pydefi.vm.swap import build_swap_transaction
from tests.addrs import (
    POOL_WETH_USDC_500,
    POOL_WETH_USDC_3000,
    UNISWAP_V4_POOL_MANAGER,
    UNISWAP_V4_QUOTER,
    UNISWAP_V4_STATE_VIEW,
)
from tests.live.sol_utils import (
    MOCK_TOKEN_SOL,
    compile_sol_file,
    compile_sol_source,
    deploy,
    deploy_mock_v3_pool,
    ensure_solc,
)
from tests.test_aggregator import USDC, WETH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"
APPROVE_PROXY_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "ApproveProxy.sol"

# Coinbase 8 — a well-funded address on mainnet (used for introspection only)
WHALE: Address = Address("0x77134cbC06cB00b66F4c7e623D5fdBF6777635EC")

_DOUBLE_FN = ContractFunction.from_abi("function double(uint256 x) external pure returns (uint256)")
_ADD_INPUTS_FN = ContractFunction.from_abi("function addInputs(uint256 a, uint256 b) external pure returns (uint256)")


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

    async def test_store_load_slot(self, ctx):
        """store_slot / load_slot round-trip a value through an alloca'd slot."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        slot = prog.alloc_slot()
        prog.store_slot(slot, 0xDEADBEEF)
        _ = prog.load_slot(slot)  # side-effect check: program compiles and runs
        prog.builder.stop()
        program = prog.build(disable_constant_folding=True)

        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    async def test_assert_triggers_with_message(self, ctx):
        """assert_(0, msg) reverts with an Error(string) payload."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_(0, "slippage exceeded")
        prog.builder.stop()
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})

    async def test_assert_passes_when_nonzero(self, ctx):
        """assert_(nonzero, msg) does not revert."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_(1, "should not trigger")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_assert_ge_pass(self, ctx):
        """assert_ge passes when a >= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_ge(200, 100, "min not met")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_assert_ge_fail(self, ctx):
        """assert_ge reverts when a < b."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_ge(100, 200, "min not met")
        prog.builder.stop()
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})

    async def test_assert_le_pass(self, ctx):
        """assert_le passes when a <= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_le(100, 200, "max exceeded")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def test_self_eth_balance(self, ctx):
        """eth_balance(self_addr()) — the VM contract's own ETH balance."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        _ = prog.eth_balance(prog.builder.address())
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_erc20_balance_of_weth_whale(self, ctx):
        """erc20_balance_of(WETH, whale) reads WETH balance of a mainnet address."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        _ = prog.erc20_balance_of(WETH.address, WHALE)
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_delta_balance_weth(self, ctx):
        """Compute a zero balance delta when no transfer occurs (pre - post == 0)."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        pre = prog.erc20_balance_of(WETH.address, WHALE)
        post = prog.erc20_balance_of(WETH.address, WHALE)
        delta = prog.sub(post, pre)  # saturating; == 0 since no transfer happened
        prog.assert_(prog.is_zero(delta), "expected zero delta")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
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

        calldata = bytes(keccak(b"getFortyTwo()")[:4])

        prog = Program()
        success = prog.call_raw(adapter, calldata)
        prog.assert_(success)
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_returndata_word_from_adapter(self, ctx):
        """returndata_word(0) reads a uint256 from the last call's returndata."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        calldata = bytes(keccak(b"getFortyTwo()")[:4])

        prog = Program()
        success = prog.call_raw(adapter, calldata)
        prog.assert_(success)
        result = prog.returndata_word(0)
        prog.assert_(prog.eq(result, 42), "expected 42")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # ABI patching
    # ------------------------------------------------------------------

    async def test_patch_u256_and_call(self, ctx):
        """patches={offset: const} mutates a calldata template before CALL."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        # Use a dummy selector → routed to MockAdapter.fallback() which emits
        # Called(sender, value, data) and echoes calldata.
        selector = b"\xde\xad\xbe\xef"
        template = bytes(selector + b"\x00" * 32)

        prog = Program()
        success = prog.call_raw(adapter, template, patches={4: 0xABCD})
        prog.assert_(success)
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        # Verify the patch wrote 0xABCD by decoding the Called event.
        called_topic = keccak(b"Called(address,uint256,bytes)")
        adapter_log = None
        for log in receipt["logs"]:
            if HexBytes(log["address"]) == adapter and log["topics"][0] == called_topic:
                adapter_log = log
                break
        assert adapter_log is not None, "Expected Called event from adapter"
        encoded = bytes(adapter_log["data"])
        calldata_len = int.from_bytes(encoded[96:128], "big")
        received_calldata = encoded[128 : 128 + calldata_len]
        assert received_calldata == selector + (0xABCD).to_bytes(32, "big")

    async def test_patch_addr(self, ctx):
        """A 32-byte MSTORE patch with a uint160 value writes a right-aligned address."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        selector = b"\xca\xfe\xba\xbe"
        template = bytes(selector + b"\x00" * 32)

        # patches={4: addr} MSTOREs 32 bytes at offset 4 with the uint160 address
        # right-aligned: 12 leading zeros then the 20-byte address.  Equivalent
        # to legacy patch_value(4 + 12, 20).
        prog = Program()
        success = prog.call_raw(adapter, template, patches={4: prog.addr(adapter)})
        prog.assert_(success)
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        called_topic = keccak(b"Called(address,uint256,bytes)")
        adapter_log = None
        for log in receipt["logs"]:
            if HexBytes(log["address"]) == adapter and log["topics"][0] == called_topic:
                adapter_log = log
                break
        assert adapter_log is not None, "Expected Called event from adapter"
        encoded = bytes(adapter_log["data"])
        calldata_len = int.from_bytes(encoded[96:128], "big")
        received_calldata = encoded[128 : 128 + calldata_len]
        assert received_calldata == selector + b"\x00" * 12 + adapter

    # ------------------------------------------------------------------
    # Chained actions (calldata surgery)
    # ------------------------------------------------------------------

    async def test_chained_calls_patch_u256(self, ctx):
        """Chain two adapter calls; second call's input = first call's output."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        double_sel = keccak(b"double(uint256)")[:4]
        calldata1 = double_sel + (5).to_bytes(32, "big")
        template2 = double_sel + (0).to_bytes(32, "big")

        prog = Program()
        # Call 1: double(5) → retdata = 10
        s1 = prog.call_raw(adapter, calldata1)
        prog.assert_(s1)
        amount = prog.returndata_word(0)
        # Call 2: double(amount) — patch template2 at offset 4 with `amount`
        s2 = prog.call_raw(adapter, template2, patches={4: amount})
        prog.assert_(s2)
        # Final assertion: result == 20
        result = prog.returndata_word(0)
        prog.assert_(prog.eq(result, 20), "expected 20")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_chained_calls_multi_patch_u256(self, ctx):
        """Chain into a 2-arg template, patching only the first slot."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        double_sel = keccak(b"double(uint256)")[:4]
        add_sel = keccak(b"addInputs(uint256,uint256)")[:4]
        calldata1 = double_sel + (7).to_bytes(32, "big")
        # Template: addInputs(0, 3) — first slot patched at runtime.
        template2 = add_sel + (0).to_bytes(32, "big") + (3).to_bytes(32, "big")

        prog = Program()
        s1 = prog.call_raw(adapter, calldata1)
        prog.assert_(s1)
        amount = prog.returndata_word(0)  # 14
        s2 = prog.call_raw(adapter, template2, patches={4: amount})
        prog.assert_(s2)
        # addInputs(14, 3) == 17
        result = prog.returndata_word(0)
        prog.assert_(prog.eq(result, 17), "expected 17")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # call_contract (high-level ABI builder) with runtime-value args
    # ------------------------------------------------------------------

    async def test_call_contract_runtime_uint256(self, ctx):
        """call_contract with a Value handle for the uint256 arg encodes it at runtime."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        prog = Program()
        amount = prog.const(7)
        success = prog.call_contract(adapter, _DOUBLE_FN, amount)
        prog.assert_(success)
        prog.assert_(prog.eq(prog.returndata_word(0), 14), "expected 14")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_call_contract_two_runtime_args(self, ctx):
        """call_contract with two Value args encodes both at runtime."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        prog = Program()
        a = prog.const(6)
        b = prog.const(11)
        success = prog.call_contract(adapter, _ADD_INPUTS_FN, a, b)
        prog.assert_(success)
        prog.assert_(prog.eq(prog.returndata_word(0), 17), "expected 17")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

    async def test_call_contract_chained(self, ctx):
        """Chain two calls: first uses a literal arg; second uses the first's result."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]
        adapter = ctx["adapter_address"]

        prog = Program()
        # Call 1: double(5) → 10  (constant arg)
        s1 = prog.call_contract(adapter, _DOUBLE_FN, 5)
        prog.assert_(s1)
        amount = prog.returndata_word(0)
        # Call 2: double(amount) → 20  (runtime SSA value)
        s2 = prog.call_contract(adapter, _DOUBLE_FN, amount)
        prog.assert_(s2)
        prog.assert_(prog.eq(prog.returndata_word(0), 20), "expected 20")
        prog.builder.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
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
    # ApproveProxy DELEGATECALLs the interpreter directly (not the VM).
    proxy_address = await deploy(w3, compiled_proxy, deployer, interpreter_addr)

    compiled_token = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    token_a_address = await deploy(w3, compiled_token, deployer)
    token_b_address = await deploy(w3, compiled_token, deployer)
    token_a = w3.eth.contract(address=token_a_address, abi=compiled_token["abi"])
    token_b = w3.eth.contract(address=token_b_address, abi=compiled_token["abi"])

    MINT_AMOUNT = 1_000 * 10**18
    for fn in [token_a.functions.mint(user, MINT_AMOUNT), token_b.functions.mint(user, MINT_AMOUNT)]:
        tx = await fn.transact({"from": deployer})
        await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])
    proxy = w3.eth.contract(address=proxy_address, abi=compiled_proxy["abi"])

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "interpreter_addr": interpreter_addr,
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
        """Proxy pulls one token from user; program transfers it from proxy to recipient."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        AMOUNT = 100 * 10**18

        tx = await token_a.functions.approve(proxy_address, AMOUNT).transact({"from": user})
        await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)

        bal_user_before = await token_a.functions.balanceOf(user).call()
        bal_recipient_before = await token_a.functions.balanceOf(recipient).call()
        bal_proxy_before = await token_a.functions.balanceOf(proxy_address).call()

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, AMOUNT).data)
        prog.builder.stop()
        program = prog.build()
        deposits = [{"token": token_a_address, "amount": AMOUNT}]

        tx = await proxy.functions.execute(program, deposits, []).transact({"from": user})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        assert await token_a.functions.balanceOf(user).call() == bal_user_before - AMOUNT
        assert await token_a.functions.balanceOf(recipient).call() == bal_recipient_before + AMOUNT
        # Proxy held the deposit briefly then forwarded it via the program — net delta zero.
        assert await token_a.functions.balanceOf(proxy_address).call() == bal_proxy_before

    async def test_multiple_deposits(self, proxy_ctx):
        """Proxy pulls two tokens from user; program forwards both to recipient."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        token_b = proxy_ctx["token_b"]
        token_b_address = proxy_ctx["token_b_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        AMOUNT_A = 50 * 10**18
        AMOUNT_B = 75 * 10**18

        for token, amount in [(token_a, AMOUNT_A), (token_b, AMOUNT_B)]:
            tx = await token.functions.approve(proxy_address, amount).transact({"from": user})
            await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)

        bal_a_user_before = await token_a.functions.balanceOf(user).call()
        bal_b_user_before = await token_b.functions.balanceOf(user).call()
        bal_a_recipient_before = await token_a.functions.balanceOf(recipient).call()
        bal_b_recipient_before = await token_b.functions.balanceOf(recipient).call()

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, AMOUNT_A).data)
        prog.call_raw(token_b_address, ERC20.fns.transfer(recipient, AMOUNT_B).data)
        prog.builder.stop()
        program = prog.build()
        deposits = [
            {"token": token_a_address, "amount": AMOUNT_A},
            {"token": token_b_address, "amount": AMOUNT_B},
        ]

        tx = await proxy.functions.execute(program, deposits, []).transact({"from": user})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        assert await token_a.functions.balanceOf(user).call() == bal_a_user_before - AMOUNT_A
        assert await token_b.functions.balanceOf(user).call() == bal_b_user_before - AMOUNT_B
        assert await token_a.functions.balanceOf(recipient).call() == bal_a_recipient_before + AMOUNT_A
        assert await token_b.functions.balanceOf(recipient).call() == bal_b_recipient_before + AMOUNT_B
        assert await token_a.functions.balanceOf(proxy_address).call() == 0
        assert await token_b.functions.balanceOf(proxy_address).call() == 0

    async def test_empty_deposits_succeeds(self, proxy_ctx):
        """execute() with an empty deposits list still runs the program."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        user = proxy_ctx["user"]

        program = Program().build()  # empty no-op program
        tx = await proxy.functions.execute(program, [], []).transact({"from": user})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
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
        await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, DEPOSIT_AMOUNT).data)
        prog.builder.stop()
        program = prog.build()
        deposits = [{"token": token_a_address, "amount": DEPOSIT_AMOUNT}]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.execute(program, deposits, []).transact({"from": user})

    async def test_interpreter_accessor(self, proxy_ctx):
        """proxy.interpreter() returns the paired EVM interpreter address."""
        proxy = proxy_ctx["proxy"]
        interpreter_addr = proxy_ctx["interpreter_addr"]

        stored = await proxy.functions.interpreter().call()
        assert HexBytes(stored) == interpreter_addr

    async def test_eth_held_by_proxy(self, proxy_ctx):
        """ETH sent with proxy.execute() lands at the proxy (program runs in proxy context)."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        user = proxy_ctx["user"]

        ETH_VALUE = 10**16  # 0.01 ETH

        proxy_balance_before = await w3.eth.get_balance(proxy_address)

        program = Program().build()  # empty no-op program
        tx = await proxy.functions.execute(program, [], []).transact({"from": user, "value": ETH_VALUE})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        proxy_balance_after = await w3.eth.get_balance(proxy_address)
        assert proxy_balance_after - proxy_balance_before == ETH_VALUE

    async def test_v3_swap_callback_routed_through_proxy(self, proxy_ctx):
        """Regression: program runs in proxy context and triggers a V3 swap;
        the pool callbacks back into the proxy, whose inherited
        ``DEXCallbackRouter.fallback`` must dispatch ``uniswapV3SwapCallback``
        and pay ``amountIn`` of ``tokenIn`` to the pool.
        """
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token_a = proxy_ctx["token_a"]
        token_a_address = proxy_ctx["token_a_address"]
        token_b_address = proxy_ctx["token_b_address"]
        deployer = proxy_ctx["deployer"]
        user = proxy_ctx["user"]

        AMOUNT_IN = 50 * 10**18

        pool_address, pool_abi = await deploy_mock_v3_pool(w3, deployer, token_a_address, token_b_address)
        pool = w3.eth.contract(address=pool_address, abi=pool_abi)

        # User approves the proxy for the deposit; proxy will pull tokens to itself
        # and the program will trigger the swap.
        tx = await token_a.functions.approve(proxy_address, AMOUNT_IN).transact({"from": user})
        await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)

        bal_proxy_before = await token_a.functions.balanceOf(proxy_address).call()
        bal_pool_before = await token_a.functions.balanceOf(pool_address).call()

        from eth_abi.abi import encode as abi_encode

        callback_data = abi_encode(["address"], [token_a_address])
        # ``encode_abi`` returns a 0x-prefixed hex str; ``call_raw`` needs bytes.
        swap_calldata = bytes.fromhex(
            pool.encode_abi("swap", args=[proxy_address, True, AMOUNT_IN, 0, callback_data])[2:]
        )
        prog = Program()
        prog.call_raw(pool_address, swap_calldata)
        prog.builder.stop()
        program = prog.build()
        deposits = [{"token": token_a_address, "amount": AMOUNT_IN}]

        tx = await proxy.functions.execute(program, deposits, []).transact({"from": user})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        # Proxy ended at the same balance: pulled in `AMOUNT_IN` from user, then
        # paid the pool exactly `AMOUNT_IN` via the V3 callback.
        assert await token_a.functions.balanceOf(proxy_address).call() == bal_proxy_before
        assert await token_a.functions.balanceOf(pool_address).call() == bal_pool_before + AMOUNT_IN


# ---------------------------------------------------------------------------
# Swap execution tests — fork with real V3 pools
# ---------------------------------------------------------------------------


async def _v3_pool_edge(w3, pool_address: Address, token_in, token_out) -> V3PoolEdge:
    pool = UNISWAP_V3_POOL(to=POOL_WETH_USDC_500)
    token0_addr = await pool.fns.token0().call(w3)
    slot0 = await pool.fns.slot0().call(w3)
    liquidity = await pool.fns.liquidity().call(w3)
    fee = await pool.fns.fee().call(w3)
    return V3PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=pool_address,
        protocol="UniswapV3",
        fee_bps=fee // 100,
        sqrt_price_x96=slot0[0],
        liquidity=liquidity,
        is_token0_in=token0_addr.lower() == token_in.address.lower(),
    )


_WETH_DEPOSIT = ContractFunction.from_abi("function deposit() external payable")

# V4 fee tiers probed in order of popularity: 0.05% / 0.3% / 0.01% / 1%.
_V4_FEE_TIERS = ((500, 10), (3000, 60), (100, 1), (10000, 200))


async def _v4_pool_edge_or_skip(w3, token_in: Token, tiers=_V4_FEE_TIERS) -> V4PoolEdge:
    """Return the first initialised *token_in*->USDC V4 pool edge, or skip."""
    v4 = UniswapV4(
        w3=w3,
        pool_manager_address=UNISWAP_V4_POOL_MANAGER,
        state_view_address=UNISWAP_V4_STATE_VIEW,
        quoter_address=UNISWAP_V4_QUOTER,
    )
    for fee, tick in tiers:
        try:
            return await v4.get_pool_edge(token_in, USDC, fee=fee, tick_spacing=tick)
        except InsufficientLiquidityError:
            continue
    pytest.skip(f"no initialised {token_in.symbol}/USDC V4 pool on this fork")


async def _execute_v4_swap(ctx, token_in: Token, edge: V4PoolEdge, amount_in: int = 10**16) -> int:
    """Fund the VM, broadcast a *token_in*->USDC V4 swap, and return the USDC delta.

    Native ETH (``address(0)``) is pre-funded with a bare transfer (exercising
    ``receive()`` + the ``settle{value}`` path); an ERC-20 is wrapped and sent in.
    """
    w3, vm_address, deployer = ctx["w3"], ctx["vm_address"], ctx["deployer"]
    swap_tx = build_swap_transaction(RouteDAG().from_token(token_in).swap(USDC, edge), amount_in, vm_address, deployer)

    if token_in.address == ZERO_ADDRESS:
        await eth_send_transaction(w3, deployer, to=vm_address, value=Wei(amount_in))
    else:
        await _WETH_DEPOSIT().transact(w3, deployer, to=token_in.address, value=Wei(amount_in))
        await ERC20.fns.transfer(vm_address, amount_in).transact(w3, deployer, to=token_in.address)

    bal_before = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)
    await eth_send_transaction(w3, deployer, to=swap_tx.to, data=swap_tx.data)
    bal_after = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)
    return bal_after - bal_before


@pytest.mark.fork
class TestBuildSwapTransactionFork:
    """Fork tests for build_swap_transaction(RouteDAG) end-to-end.

    Exercises the full path: find_best_split → RouteDAG →
    build_swap_transaction → vm.execute() on a mainnet fork with real V3 pools.
    Unlike TestQuoteFork (which quotes each leg via QuoterV2), these tests
    execute the compiled swap program and verify non-zero token output.
    """

    async def test_split_route_build_and_execute(self, ctx) -> None:
        """build_swap_transaction(RouteDAG) executes a 2-leg split on a mainnet fork.

        Uses fee-equalized synthetic liquidity to force find_best_split into a
        2-leg split DAG, compiles it via build_swap_transaction, and executes
        against real V3 pools, verifying the deployer receives non-zero USDC.
        """
        w3 = ctx["w3"]
        vm_address = ctx["vm_address"]
        deployer = ctx["deployer"]
        amount_in = 10**18  # 1 WETH — large enough for split to improve on single-pool route

        # Symmetric graph (equal fee + price) so a split can improve on single-pool routing.
        graph = PoolGraph()
        ref_edge = await _v3_pool_edge(w3, POOL_WETH_USDC_500, WETH, USDC)
        for pool_addr in (POOL_WETH_USDC_500, POOL_WETH_USDC_3000):
            edge = await _v3_pool_edge(w3, pool_addr, WETH, USDC)
            graph.add_pool(
                V3PoolEdge(
                    token_in=WETH,
                    token_out=USDC,
                    pool_address=pool_addr,
                    protocol="UniswapV3",
                    fee_bps=5,
                    sqrt_price_x96=ref_edge.sqrt_price_x96,
                    liquidity=10**15,
                    is_token0_in=edge.is_token0_in,
                )
            )

        dag = Router(graph).find_best_split(TokenAmount(WETH, amount_in), USDC, step_bps=1000)
        assert len(_dag_leg_weights(dag)) >= 1, "expected at least one leg in split DAG"

        # min_final_out=0: actual output verified by balance check below.
        swap_tx = build_swap_transaction(dag, amount_in, vm_address, deployer)

        await _WETH_DEPOSIT().transact(w3, deployer, to=WETH.address, value=Wei(amount_in))
        await ERC20.fns.transfer(vm_address, amount_in).transact(w3, deployer, to=WETH.address)

        bal_before = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)
        await eth_send_transaction(w3, deployer, to=swap_tx.to, data=swap_tx.data)
        bal_after = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)

        assert bal_after > bal_before, f"Expected USDC > 0 after split swap, got {bal_after - bal_before}"

    async def test_v4_swap_build_and_execute(self, ctx) -> None:
        """WETH->USDC V4 swap via DeFiVM's ``unlockCallback``.

        The VM calls ``PoolManager.unlock()``; the callback performs the swap,
        settles WETH from the VM's balance (sync→transfer→settle), and ``take``s
        USDC to the deployer.
        """
        edge = await _v4_pool_edge_or_skip(ctx["w3"], WETH)
        received = await _execute_v4_swap(ctx, WETH, edge)
        assert received > 0, f"expected USDC out from V4 swap, got {received}"

    async def test_v4_native_eth_swap_build_and_execute(self, ctx) -> None:
        """Native ETH->USDC V4 swap: exercises the ``settle{value}`` branch.

        Native ETH is ``address(0)`` (currency0), driving ``zeroForOne=true`` and
        the native-settle path the WETH test does not; the bare ETH pre-fund also
        exercises ``receive()``.
        """
        eth = Token(chain_id=1, address=ZERO_ADDRESS, symbol="ETH", decimals=18)
        edge = await _v4_pool_edge_or_skip(ctx["w3"], eth)
        assert edge.is_token0_in, "native ETH must be currency0 (zeroForOne=true path)"
        received = await _execute_v4_swap(ctx, eth, edge)
        assert received > 0, f"expected USDC out from native ETH V4 swap, got {received}"

    async def test_v4_pool_key_fee_uses_exact_lp_fee(self, ctx) -> None:
        """The V4 PoolKey fee must be the pool's exact lp_fee_pips, not the
        truncated fee_bps*100 — which derives a non-existent poolId and reverts
        for fees that are not a multiple of 100 pips.
        """
        edge = await _v4_pool_edge_or_skip(ctx["w3"], WETH)

        def program_of(e: V4PoolEdge) -> bytes:
            dag = RouteDAG().from_token(e.token_in).swap(e.token_out, e)
            tx = build_swap_transaction(dag, 10**16, ctx["vm_address"], ctx["deployer"])
            (program,) = abi_decode(["bytes"], tx.data[4:])
            return program

        # The real pool's exact on-chain fee is embedded in compiled program.
        assert edge.lp_fee_pips.to_bytes(32, "big") in program_of(edge)

        # Clone same pool with a non-100-multiple fee: exact value must be encoded.
        odd = edge.lp_fee_pips + 50
        odd_edge = dataclasses.replace(edge, lp_fee_pips=odd, fee_bps=odd // 100)
        program = program_of(odd_edge)
        assert odd.to_bytes(32, "big") in program
        assert (odd_edge.fee_bps * 100).to_bytes(32, "big") not in program

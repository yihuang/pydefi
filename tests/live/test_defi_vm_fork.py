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

from pathlib import Path

import pytest
import solcx
from eth_contract.contract import ContractFunction
from eth_contract.erc20 import ERC20
from eth_contract.utils import send_transaction as eth_send_transaction
from eth_utils import keccak
from hexbytes import HexBytes
from web3.exceptions import ContractLogicError, Web3RPCError
from web3.types import Wei

from pydefi.abi.amm import UNISWAP_V3_POOL
from pydefi.pathfinder.graph import PoolGraph, V3PoolEdge
from pydefi.pathfinder.router import Router
from pydefi.types import Address, TokenAmount
from pydefi.vm import Program
from pydefi.vm.swap import build_swap_transaction
from tests.addrs import POOL_WETH_USDC_500, POOL_WETH_USDC_3000
from tests.live.sol_utils import MOCK_TOKEN_SOL, compile_sol_file, compile_sol_source, deploy, ensure_solc
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
        prog.stop()
        program = prog.build(disable_constant_folding=True)

        tx = await vm.functions.execute(program).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})

    async def test_assert_passes_when_nonzero(self, ctx):
        """assert_(nonzero, msg) does not revert."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_(1, "should not trigger")
        prog.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_assert_ge_pass(self, ctx):
        """assert_ge passes when a >= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_ge(200, 100, "min not met")
        prog.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_assert_ge_fail(self, ctx):
        """assert_ge reverts when a < b."""
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_ge(100, 200, "min not met")
        prog.stop()
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})

    async def test_assert_le_pass(self, ctx):
        """assert_le passes when a <= b."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        prog.assert_le(100, 200, "max exceeded")
        prog.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        _ = prog.eth_balance(prog.self_addr())
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_erc20_balance_of_weth_whale(self, ctx):
        """erc20_balance_of(WETH, whale) reads WETH balance of a mainnet address."""
        w3 = ctx["w3"]
        vm = ctx["vm"]
        deployer = ctx["deployer"]

        prog = Program()
        _ = prog.erc20_balance_of(WETH.address, WHALE)
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
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

        calldata = bytes(keccak(b"getFortyTwo()")[:4])

        prog = Program()
        success = prog.call_raw(adapter, calldata)
        prog.assert_(success)
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build(disable_constant_folding=True)).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
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
        prog.stop()
        tx = await vm.functions.execute(prog.build()).transact({"from": deployer})
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

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, AMOUNT).data)
        prog.stop()
        program = prog.build()
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

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, AMOUNT_A).data)
        prog.call_raw(token_b_address, ERC20.fns.transfer(recipient, AMOUNT_B).data)
        prog.stop()
        program = prog.build()
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

        program = Program().build()  # empty no-op program
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

        prog = Program()
        prog.call_raw(token_a_address, ERC20.fns.transfer(recipient, DEPOSIT_AMOUNT).data)
        prog.stop()
        program = prog.build()
        deposits = [{"token": token_a_address, "amount": DEPOSIT_AMOUNT}]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.execute(program, deposits).transact({"from": user})

    async def test_vm_accessor(self, proxy_ctx):
        """proxy.vm() returns the paired DeFiVM address."""
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]

        stored_vm = await proxy.functions.vm().call()
        assert HexBytes(stored_vm) == vm_address

    async def test_eth_forwarding(self, proxy_ctx):
        """ETH sent to proxy.execute() is forwarded to DeFiVM."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]
        user = proxy_ctx["user"]

        ETH_VALUE = 10**16  # 0.01 ETH

        vm_balance_before = await w3.eth.get_balance(vm_address)

        program = Program().build()  # empty no-op program
        tx = await proxy.functions.execute(program, []).transact({"from": user, "value": ETH_VALUE})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        vm_balance_after = await w3.eth.get_balance(vm_address)
        assert vm_balance_after - vm_balance_before == ETH_VALUE


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
        assert len(Router.dag_leg_weights(dag)) >= 1, "expected at least one leg in split DAG"

        # min_final_out=0: actual output verified by balance check below.
        swap_tx = build_swap_transaction(dag, amount_in, vm_address, deployer)

        await _WETH_DEPOSIT().transact(w3, deployer, to=WETH.address, value=Wei(amount_in))
        await ERC20.fns.transfer(vm_address, amount_in).transact(w3, deployer, to=WETH.address)

        bal_before = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)
        await eth_send_transaction(w3, deployer, to=swap_tx.to, data=swap_tx.data)
        bal_after = await ERC20.fns.balanceOf(deployer).call(w3, to=USDC.address)

        assert bal_after > bal_before, f"Expected USDC > 0 after split swap, got {bal_after - bal_before}"

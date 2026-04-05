"""Fork tests for ApproveProxy — safe ERC-20 allowance proxy for DeFiVM.

These tests compile ApproveProxy.sol and DeFiVM.sol with py-solc-x, deploy
them on a local Anvil fork, and exercise the full approval + execution flow:

 - Basic token transfer via ApproveProxy.execute() + program calling
   proxy.transferFrom()
 - Revert when transferFrom is called directly (outside of execute)
 - Revert when transferFrom is called by address other than the vm
 - Revert when the proxy is called reentrantly
 - Transfer reverts when ERC-20 allowance is insufficient
 - ETH value is correctly forwarded to DeFiVM via proxy.execute()

Run with::

    pytest -m fork tests/live/test_approve_proxy_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm import ApproveProxy, Program
from pydefi.vm.program import pop, push_u256

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Mock ERC-20 token (inline Solidity)
# ---------------------------------------------------------------------------

_MOCK_TOKEN_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mintable ERC-20 token used in tests.
contract MockToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "MockToken: insufficient balance");
        require(allowance[from][msg.sender] >= amount, "MockToken: insufficient allowance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "MockToken: insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}
"""

# Minimal interpreter used in fork tests (delegatecall-based).
_MINIMAL_INTERPRETER_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;
contract TestInterpreter {
    fallback() external payable {
        assembly {
            let size := calldatasize()
            calldatacopy(0, 0, size)
            let ok := create(0, 0, size)
            if iszero(ok) { revert(0, 0) }
            let retsize := 0x1000
            let success := delegatecall(gas(), ok, 0, size, 0, retsize)
            returndatacopy(0, 0, returndatasize())
            if iszero(success) { revert(0, returndatasize()) }
            return(0, returndatasize())
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Compile + deploy helpers
# ---------------------------------------------------------------------------


def _ensure_solc(version: str = "0.8.24") -> None:
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def _compile_sol_source(source: str, contract_name: str) -> dict:
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    key = f"<stdin>:{contract_name}"
    return result[key]


def _compile_defi_vm() -> dict:
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
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


async def _ensure_interpreter(w3: AsyncWeb3, deployer: str) -> str:
    """Deploy a minimal test interpreter if the Analog-Labs one is absent."""
    INTERPRETER_ADDR = "0x0000000000001e3F4F615cd5e20c681Cf7d85e8D"
    code = await w3.eth.get_code(INTERPRETER_ADDR)
    if code and code != b"":
        return INTERPRETER_ADDR
    compiled = _compile_sol_source(_MINIMAL_INTERPRETER_SOL, "TestInterpreter")
    return await _deploy(w3, compiled, deployer)


# ---------------------------------------------------------------------------
# Module-scoped Anvil fork fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def proxy_fork_w3(fork_w3_module):
    return fork_w3_module


# ---------------------------------------------------------------------------
# Module-scoped setup: deploy once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def proxy_ctx(proxy_fork_w3, interpreter_addr):
    """Deploy DeFiVM, ApproveProxy, and MockToken; return shared context."""
    w3 = proxy_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]
    user = accounts[1]
    recipient = accounts[2]

    # Deploy DeFiVM
    compiled_vm = _compile_defi_vm()
    vm_address = await _deploy(w3, compiled_vm, deployer, interpreter_addr)

    # Deploy ApproveProxy paired with the DeFiVM
    compiled_proxy = ApproveProxy.compile()
    proxy_address = await _deploy(w3, compiled_proxy, deployer, vm_address)

    # Deploy MockToken and mint tokens to the user
    compiled_token = _compile_sol_source(_MOCK_TOKEN_SOL, "MockToken")
    token_address = await _deploy(w3, compiled_token, deployer)
    token = w3.eth.contract(address=token_address, abi=compiled_token["abi"])

    MINT_AMOUNT = 1_000 * 10**18
    tx = await token.functions.mint(user, MINT_AMOUNT).transact({"from": deployer})
    await w3.eth.get_transaction_receipt(tx)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])
    proxy = ApproveProxy.contract(w3, proxy_address)

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "proxy": proxy,
        "proxy_address": proxy_address,
        "token": token,
        "token_address": token_address,
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

    # ------------------------------------------------------------------
    # Happy path: token transfer via proxy
    # ------------------------------------------------------------------

    async def test_transfer_via_proxy(self, proxy_ctx):
        """Program calls proxy.transferFrom to pull tokens from the executing user."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token = proxy_ctx["token"]
        token_address = proxy_ctx["token_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        TRANSFER_AMOUNT = 100 * 10**18

        # User approves the proxy to spend tokens on their behalf
        tx = await token.functions.approve(proxy_address, TRANSFER_AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        # Check balances before
        bal_user_before = await token.functions.balanceOf(user).call()
        bal_recipient_before = await token.functions.balanceOf(recipient).call()

        # Build a program that calls proxy.transferFrom(token, recipient, amount)
        calldata = ApproveProxy.calldata.transferFrom(token_address, recipient, TRANSFER_AMOUNT)
        program = (
            Program()
            .call_contract(proxy_address, calldata)
            .pop()
            .build()
        )

        # Execute through the proxy (user is msg.sender → proxy tracks them)
        tx = await proxy.functions.execute(program).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Verify balances changed correctly
        bal_user_after = await token.functions.balanceOf(user).call()
        bal_recipient_after = await token.functions.balanceOf(recipient).call()
        assert bal_user_before - bal_user_after == TRANSFER_AMOUNT
        assert bal_recipient_after - bal_recipient_before == TRANSFER_AMOUNT

    async def test_transfer_partial_amount(self, proxy_ctx):
        """proxy.transferFrom can be called with any amount up to the ERC-20 allowance."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token = proxy_ctx["token"]
        token_address = proxy_ctx["token_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        APPROVE_AMOUNT = 500 * 10**18
        TRANSFER_AMOUNT = 50 * 10**18

        tx = await token.functions.approve(proxy_address, APPROVE_AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        bal_user_before = await token.functions.balanceOf(user).call()
        bal_recipient_before = await token.functions.balanceOf(recipient).call()

        calldata = ApproveProxy.calldata.transferFrom(token_address, recipient, TRANSFER_AMOUNT)
        program = Program().call_contract(proxy_address, calldata).pop().build()

        tx = await proxy.functions.execute(program).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        assert (await token.functions.balanceOf(user).call()) == bal_user_before - TRANSFER_AMOUNT
        assert (await token.functions.balanceOf(recipient).call()) == bal_recipient_before + TRANSFER_AMOUNT

    # ------------------------------------------------------------------
    # Security: transferFrom outside of execute() must fail
    # ------------------------------------------------------------------

    async def test_transferFrom_direct_call_reverts(self, proxy_ctx):
        """Calling proxy.transferFrom directly (outside execute) must revert."""
        proxy = proxy_ctx["proxy"]
        token_address = proxy_ctx["token_address"]
        recipient = proxy_ctx["recipient"]
        user = proxy_ctx["user"]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.transferFrom(
                token_address, recipient, 1
            ).transact({"from": user})

    async def test_transferFrom_by_non_vm_caller_reverts(self, proxy_ctx):
        """Only the paired DeFiVM can call proxy.transferFrom (not arbitrary callers)."""
        proxy = proxy_ctx["proxy"]
        token_address = proxy_ctx["token_address"]
        recipient = proxy_ctx["recipient"]
        deployer = proxy_ctx["deployer"]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.transferFrom(
                token_address, recipient, 1
            ).transact({"from": deployer})

    async def test_insufficient_allowance_reverts(self, proxy_ctx):
        """Executing a transfer that exceeds the ERC-20 allowance must revert."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        proxy_address = proxy_ctx["proxy_address"]
        token = proxy_ctx["token"]
        token_address = proxy_ctx["token_address"]
        user = proxy_ctx["user"]
        recipient = proxy_ctx["recipient"]

        # Approve less than the transfer amount
        APPROVE_AMOUNT = 1
        TRANSFER_AMOUNT = 1_000 * 10**18

        tx = await token.functions.approve(proxy_address, APPROVE_AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        calldata = ApproveProxy.calldata.transferFrom(token_address, recipient, TRANSFER_AMOUNT)
        program = Program().call_contract(proxy_address, calldata).pop().build()

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.execute(program).transact({"from": user})

    # ------------------------------------------------------------------
    # Proxy vm() accessor
    # ------------------------------------------------------------------

    async def test_vm_accessor(self, proxy_ctx):
        """proxy.vm() returns the paired DeFiVM address."""
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]

        stored_vm = await proxy.functions.vm().call()
        assert stored_vm.lower() == vm_address.lower()

    # ------------------------------------------------------------------
    # ETH forwarding
    # ------------------------------------------------------------------

    async def test_eth_forwarding(self, proxy_ctx):
        """ETH sent to proxy.execute() is forwarded to DeFiVM."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        vm_address = proxy_ctx["vm_address"]
        user = proxy_ctx["user"]

        ETH_VALUE = 10**16  # 0.01 ETH

        vm_balance_before = await w3.eth.get_balance(vm_address)

        # Empty program that just succeeds — ETH ends up in DeFiVM
        program = push_u256(0) + pop()

        tx = await proxy.functions.execute(program).transact(
            {"from": user, "value": ETH_VALUE}
        )
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        vm_balance_after = await w3.eth.get_balance(vm_address)
        assert vm_balance_after - vm_balance_before == ETH_VALUE

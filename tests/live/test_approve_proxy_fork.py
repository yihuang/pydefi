"""Fork tests for ApproveProxy — safe ERC-20 allowance proxy for DeFiVM.

These tests compile ApproveProxy.sol and DeFiVM.sol with py-solc-x, deploy
them on a local Anvil fork, and exercise the full deposit + execution flow:

 - Single token deposit: tokens move user → DeFiVM, program transfers them
   to a recipient
 - Multiple token deposits in one call
 - Deposit with no tokens (empty deposits list) + ETH forwarding
 - Insufficient ERC-20 allowance reverts before program runs
 - vm() accessor returns the correct paired DeFiVM address

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
    """Deploy DeFiVM, ApproveProxy, and two MockTokens; return shared context."""
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

    # Deploy two MockTokens and mint tokens to the user
    compiled_token = _compile_sol_source(_MOCK_TOKEN_SOL, "MockToken")
    token_a_address = await _deploy(w3, compiled_token, deployer)
    token_b_address = await _deploy(w3, compiled_token, deployer)
    token_a = w3.eth.contract(address=token_a_address, abi=compiled_token["abi"])
    token_b = w3.eth.contract(address=token_b_address, abi=compiled_token["abi"])

    MINT_AMOUNT = 1_000 * 10**18
    for fn in [token_a.functions.mint(user, MINT_AMOUNT), token_b.functions.mint(user, MINT_AMOUNT)]:
        tx = await fn.transact({"from": deployer})
        await w3.eth.get_transaction_receipt(tx)

    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])
    proxy = ApproveProxy.contract(w3, proxy_address)

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
# ERC-20 transfer calldata helper
# ---------------------------------------------------------------------------


def _transfer_calldata(recipient: str, amount: int) -> bytes:
    """Encode calldata for ERC-20 ``transfer(address,uint256)``."""
    from eth_abi import encode
    from eth_utils import keccak

    selector = keccak(b"transfer(address,uint256)")[:4]
    return selector + encode(["address", "uint256"], [recipient, amount])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestApproveProxyFork:
    """Fork-level tests for ApproveProxy.sol on a local Anvil mainnet fork."""

    # ------------------------------------------------------------------
    # Happy path: single token deposit + program transfers to recipient
    # ------------------------------------------------------------------

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

        # User approves proxy to pull tokens
        tx = await token_a.functions.approve(proxy_address, AMOUNT).transact({"from": user})
        await w3.eth.get_transaction_receipt(tx)

        bal_user_before = await token_a.functions.balanceOf(user).call()
        bal_recipient_before = await token_a.functions.balanceOf(recipient).call()
        bal_vm_before = await token_a.functions.balanceOf(vm_address).call()

        # Program: DeFiVM already holds the tokens after deposit, transfer to recipient
        program = Program().call_contract(token_a_address, _transfer_calldata(recipient, AMOUNT)).pop().build()
        deposits = [{"token": token_a_address, "amount": AMOUNT}]

        tx = await proxy.functions.execute(program, deposits).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # User lost AMOUNT tokens; recipient gained AMOUNT; VM balance unchanged (transit)
        assert await token_a.functions.balanceOf(user).call() == bal_user_before - AMOUNT
        assert await token_a.functions.balanceOf(recipient).call() == bal_recipient_before + AMOUNT
        assert await token_a.functions.balanceOf(vm_address).call() == bal_vm_before

    # ------------------------------------------------------------------
    # Multiple token deposits in one call
    # ------------------------------------------------------------------

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

        # Program: transfer both tokens from VM to recipient
        program = (
            Program()
            .call_contract(token_a_address, _transfer_calldata(recipient, AMOUNT_A))
            .pop()
            .call_contract(token_b_address, _transfer_calldata(recipient, AMOUNT_B))
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
        # VM balance is net-zero for both tokens (deposited then transferred out)
        assert await token_a.functions.balanceOf(vm_address).call() == 0
        assert await token_b.functions.balanceOf(vm_address).call() == 0

    # ------------------------------------------------------------------
    # Empty deposits list (program runs without any token deposit)
    # ------------------------------------------------------------------

    async def test_empty_deposits_succeeds(self, proxy_ctx):
        """execute() with an empty deposits list still runs the program."""
        w3 = proxy_ctx["w3"]
        proxy = proxy_ctx["proxy"]
        user = proxy_ctx["user"]

        # Trivial program: push 0 + pop (no-op)
        program = push_u256(0) + pop()
        tx = await proxy.functions.execute(program, []).transact({"from": user})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    # ------------------------------------------------------------------
    # Security: insufficient ERC-20 allowance reverts
    # ------------------------------------------------------------------

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

        program = Program().call_contract(token_a_address, _transfer_calldata(recipient, DEPOSIT_AMOUNT)).pop().build()
        deposits = [{"token": token_a_address, "amount": DEPOSIT_AMOUNT}]

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await proxy.functions.execute(program, deposits).transact({"from": user})

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

        program = push_u256(0) + pop()
        tx = await proxy.functions.execute(program, []).transact(
            {"from": user, "value": ETH_VALUE}
        )
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        vm_balance_after = await w3.eth.get_balance(vm_address)
        assert vm_balance_after - vm_balance_before == ETH_VALUE

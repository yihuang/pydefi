"""Fork-based tests for the transaction simulation utilities.

These tests spin up a local Anvil fork of Ethereum mainnet and exercise the
simulation functions in ``pydefi.simulate`` against live contract state.

How the tests work
------------------
1. An Anvil fork is started via the ``fork_w3_module`` fixture in ``conftest.py``.
2. The ``sim_ctx`` fixture deploys:

   - A **DeFiVM** contract — the VM contract under test.
   - A **MockToken** with a known minted balance for controlled testing.

3. ERC-20 calldata is built with :class:`eth_contract.erc20.ERC20` rather than
   raw byte selectors.
4. Each simulation method is exercised and the results are verified:

   - :func:`~pydefi.simulate.simulate_with_eth_call` — success/failure only.
   - :func:`~pydefi.simulate.simulate_with_debug_trace_call` — logs + gas.
   - :func:`~pydefi.simulate.simulate_with_eth_simulate_v1` — multi-message.
   - :func:`~pydefi.simulate.build_allowance_state_override` — slot detection.
   - :func:`~pydefi.simulate.simulate_tx` — auto-selection.

Requirements
------------
- ``anvil`` must be on ``$PATH`` (install via Foundry).
- ``ETH_RPC_URL`` must point to a working Ethereum mainnet RPC endpoint.

Run with::

    pytest -m fork -v tests/live/test_simulate_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3, Web3

from pydefi.simulate import (
    SimulationResult,
    build_allowance_state_override,
    detect_allowance_slot,
    simulate_tx,
    simulate_with_debug_trace_call,
    simulate_with_eth_call,
    simulate_with_eth_simulate_v1,
)
from pydefi.vm import Program

from .conftest import ETH_WHALE, USDC
from .sol_utils import MOCK_TOKEN_SOL, compile_sol_file, compile_sol_source, deploy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the DeFiVM.sol contract
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _impersonate(w3: AsyncWeb3, address: str) -> None:
    """Ask Anvil to impersonate *address*."""
    await w3.provider.make_request("anvil_impersonateAccount", [address])


async def _set_eth_balance(w3: AsyncWeb3, address: str, amount: int) -> None:
    """Set ETH balance via Anvil."""
    await w3.provider.make_request("anvil_setBalance", [address, hex(amount)])


# ---------------------------------------------------------------------------
# Module-scoped fixture: deploy DeFiVM + MockToken once for this test module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def sim_ctx(fork_w3_module, interpreter_addr):
    """Deploy DeFiVM + MockToken once for the entire test module.

    Returns a dict containing:

    - ``w3``: the shared :class:`~web3.AsyncWeb3` instance
    - ``vm`` / ``vm_address``: the deployed DeFiVM contract
    - ``token`` / ``token_address``: the deployed MockToken contract
    - ``deployer`` / ``user`` / ``recipient``: Anvil test accounts
    - ``mint_amount``: tokens minted to both ``user`` and ``vm_address``
    """
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer = accounts[0]
    user = accounts[1]
    recipient = accounts[2]

    # Deploy DeFiVM
    compiled_vm = compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")
    vm_address = await deploy(w3, compiled_vm, deployer, interpreter_addr)
    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])

    # Deploy MockToken and mint tokens to the user and the VM contract so that
    # composition programs executed from inside DeFiVM have tokens to spend.
    compiled_token = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    token_address = await deploy(w3, compiled_token, deployer)
    token = w3.eth.contract(address=token_address, abi=compiled_token["abi"])

    mint_amount = 1_000 * 10**18
    for addr in [user, vm_address]:
        tx = await token.functions.mint(addr, mint_amount).transact({"from": deployer})
        await w3.eth.get_transaction_receipt(tx)

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "token": token,
        "token_address": token_address,
        "deployer": deployer,
        "user": user,
        "recipient": recipient,
        "mint_amount": mint_amount,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestSimulateEthCall:
    """Tests for simulate_with_eth_call."""

    async def test_eth_call_success(self, sim_ctx):
        """Calling balanceOf returns success and non-empty return data."""
        w3 = sim_ctx["w3"]
        result = await simulate_with_eth_call(
            w3,
            {"to": USDC.address, "data": ERC20.fns.balanceOf(ETH_WHALE).data},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        assert len(result.return_data) == 32
        # Balance should be parseable as a uint256
        balance = int.from_bytes(result.return_data, "big")
        assert balance >= 0
        # eth_call does not expose logs
        assert result.transfers == []
        assert result.balance_changes == []

    async def test_eth_call_revert(self, sim_ctx):
        """Calling a non-existent function returns failure."""
        w3 = sim_ctx["w3"]
        result = await simulate_with_eth_call(
            w3,
            {"to": USDC.address, "data": "0xdeadbeef"},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is False

    async def test_eth_call_state_override_balance(self, sim_ctx):
        """State override injects an artificial ETH balance."""
        w3 = sim_ctx["w3"]
        fake_balance = 10**21  # 1000 ETH
        result = await simulate_with_eth_call(
            w3,
            {"from": ETH_WHALE, "to": ETH_WHALE, "value": fake_balance, "data": "0x"},
            state_overrides={
                ETH_WHALE: {"balance": fake_balance * 2},
            },
        )
        assert result.success is True

    async def test_eth_call_allowance_state_override(self, sim_ctx):
        """build_allowance_state_override injects an ERC-20 allowance via eth_call.

        Uses the deployed DeFiVM as the spender — the realistic case where we
        want to let the VM spend USDC on behalf of the user before executing a
        composition program.
        """
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        amount = 10**12  # 1 million USDC
        state_override = await build_allowance_state_override(
            w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=vm_address,
            amount=amount,
        )

        # Read back the allowance via eth_call with the state override applied.
        result = await simulate_with_eth_call(
            w3,
            {"to": USDC.address, "data": ERC20.fns.allowance(ETH_WHALE, vm_address).data},
            state_overrides=state_override,
        )
        assert result.success is True
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == amount


@pytest.mark.fork
class TestDetectAllowanceSlot:
    """Tests for detect_allowance_slot and build_allowance_state_override.

    Uses the deployed DeFiVM as the spender — simulating the approval that
    would be needed before executing a DeFiVM composition program that
    transfers USDC.
    """

    async def test_detect_allowance_slot_usdc(self, sim_ctx):
        """Detect the allowance mapping slot for USDC."""
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        slot = await detect_allowance_slot(
            w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=vm_address,
        )
        assert slot is not None, "detect_allowance_slot returned None — is debug_traceCall available?"
        # USDC stores allowances at slot 9 (proxy impl); just check we get a valid slot
        assert len(slot.slot) == 32

    async def test_build_allowance_state_override_structure(self, sim_ctx):
        """build_allowance_state_override returns a valid state-override dict."""
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        amount = 5 * 10**12  # 5 million USDC
        override = await build_allowance_state_override(
            w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=vm_address,
            amount=amount,
        )
        assert isinstance(override, dict)
        token_key = Web3.to_checksum_address(USDC.address)
        assert token_key in override
        assert "stateDiff" in override[token_key]
        # stateDiff should have exactly one entry
        state_diff = override[token_key]["stateDiff"]
        assert len(state_diff) == 1
        # The stored value should encode the requested amount
        slot_val = list(state_diff.values())[0]
        assert int(slot_val, 16) == amount

    async def test_allowance_state_override_read_back(self, sim_ctx):
        """Injected allowance is readable via eth_call with the override applied."""
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        target_amount = 42 * 10**6  # 42 USDC
        override = await build_allowance_state_override(
            w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=vm_address,
            amount=target_amount,
        )
        result = await simulate_with_eth_call(
            w3,
            {"to": USDC.address, "data": ERC20.fns.allowance(ETH_WHALE, vm_address).data},
            state_overrides=override,
        )
        assert result.success is True
        assert int.from_bytes(result.return_data, "big") == target_amount


@pytest.mark.fork
class TestSimulateDebugTraceCall:
    """Tests for simulate_with_debug_trace_call."""

    async def test_debug_trace_call_balanceof(self, sim_ctx):
        """debug_traceCall on balanceOf returns success and gas usage."""
        w3 = sim_ctx["w3"]
        result = await simulate_with_debug_trace_call(
            w3,
            {"to": USDC.address, "data": ERC20.fns.balanceOf(ETH_WHALE).data},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        assert result.gas_used is not None and result.gas_used > 0

    async def test_debug_trace_call_erc20_approve(self, sim_ctx):
        """debug_traceCall on an ERC-20 approve captures gas and succeeds.

        Impersonates the whale and simulates approving the DeFiVM contract to
        spend USDC — the first step in a DeFiVM-composed swap flow.
        """
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        await _impersonate(w3, ETH_WHALE)
        result = await simulate_with_debug_trace_call(
            w3,
            {"from": ETH_WHALE, "to": USDC.address, "data": ERC20.fns.approve(vm_address, 10**18).data},
        )
        assert result.success is True
        # approve() emits an Approval event (not Transfer), so no ERC-20 transfers
        # but gas should still be reported
        assert result.gas_used is not None

    async def test_debug_trace_call_state_override_allowance(self, sim_ctx):
        """debug_traceCall with state override injects an allowance for the VM."""
        w3 = sim_ctx["w3"]
        vm_address = sim_ctx["vm_address"]
        override = await build_allowance_state_override(
            w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=vm_address,
            amount=10**10,
        )
        result = await simulate_with_debug_trace_call(
            w3,
            {"to": USDC.address, "data": ERC20.fns.allowance(ETH_WHALE, vm_address).data},
            state_overrides=override,
        )
        assert result.success is True
        assert len(result.return_data) == 32
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == 10**10


@pytest.mark.fork
class TestSimulateEthSimulateV1:
    """Tests for simulate_with_eth_simulate_v1."""

    async def test_simulate_v1_single_call(self, sim_ctx):
        """eth_simulateV1 on a view call returns success and logs."""
        w3 = sim_ctx["w3"]
        results = await simulate_with_eth_simulate_v1(
            w3,
            [{"to": USDC.address, "data": ERC20.fns.balanceOf(ETH_WHALE).data}],
        )
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SimulationResult)
        assert result.success is True

    async def test_simulate_v1_approve_then_check(self, sim_ctx):
        """eth_simulateV1 multi-message: approve MockToken + check allowance for VM.

        Simulates the approve-before-swap pattern using the deployed MockToken
        and DeFiVM contract as the spender.
        """
        w3 = sim_ctx["w3"]
        user = sim_ctx["user"]
        vm_address = sim_ctx["vm_address"]
        token_address = sim_ctx["token_address"]
        approve_call = {
            "from": user,
            "to": token_address,
            "data": ERC20.fns.approve(vm_address, 10**18).data,
        }
        check_call = {
            "to": token_address,
            "data": ERC20.fns.allowance(user, vm_address).data,
        }
        results = await simulate_with_eth_simulate_v1(
            w3,
            [approve_call, check_call],
        )
        assert len(results) == 2
        approve_result, check_result = results

        assert approve_result.success is True
        assert check_result.success is True
        # After the simulated approve, allowance should be 10**18
        allowance = int.from_bytes(check_result.return_data, "big")
        assert allowance == 10**18

    async def test_simulate_v1_erc20_transfer_event(self, sim_ctx):
        """eth_simulateV1 captures ERC-20 Transfer events for MockToken.

        No state override needed — the ``sim_ctx`` fixture mints tokens directly
        to the user so the test is deterministic.
        """
        w3 = sim_ctx["w3"]
        user = sim_ctx["user"]
        token_address = sim_ctx["token_address"]
        recipient = sim_ctx["recipient"]
        transfer_amount = 10**18  # 1 MockToken

        results = await simulate_with_eth_simulate_v1(
            w3,
            [{"from": user, "to": token_address, "data": ERC20.fns.transfer(recipient, transfer_amount).data}],
        )
        assert len(results) == 1
        result = results[0]
        assert result.success is True, f"Transfer failed: {result.revert_reason}"

        # There should be a Transfer event
        token_transfers = [t for t in result.transfers if t.token == Web3.to_checksum_address(token_address)]
        assert len(token_transfers) >= 1
        t = token_transfers[0]
        assert t.from_address == Web3.to_checksum_address(user)
        assert t.to_address == Web3.to_checksum_address(recipient)
        assert t.amount == transfer_amount

        # balance_changes should reflect the transfer
        changes = {(c.address, c.token): c.delta for c in result.balance_changes}
        from_key = (Web3.to_checksum_address(user), Web3.to_checksum_address(token_address))
        to_key = (Web3.to_checksum_address(recipient), Web3.to_checksum_address(token_address))
        assert changes.get(from_key, 0) == -transfer_amount
        assert changes.get(to_key, 0) == transfer_amount

    async def test_simulate_v1_composition_program(self, sim_ctx):
        """eth_simulateV1 simulates a DeFiVM composition program (token transfer via VM).

        Builds a :class:`~pydefi.vm.Program` that calls
        ``MockToken.transfer(recipient, amount)`` from inside the DeFiVM
        contract, then simulates the ``execute(program)`` call.  The Transfer
        event should appear in the simulation result — demonstrating that
        ``simulate_tx`` can preview the side effects of a composed DeFi
        operation before it is broadcast.
        """
        w3 = sim_ctx["w3"]
        vm = sim_ctx["vm"]
        vm_address = sim_ctx["vm_address"]
        token_address = sim_ctx["token_address"]
        deployer = sim_ctx["deployer"]
        recipient = sim_ctx["recipient"]
        transfer_amount = 10 * 10**18

        # Build DeFiVM composition bytecodes: call token.transfer(recipient, amount)
        program = (
            Program().call_contract(token_address, ERC20.fns.transfer(recipient, transfer_amount).data).pop().build()
        )

        # Encode DeFiVM.execute(program) calldata
        execute_calldata = vm.functions.execute(program)._encode_transaction_data()

        results = await simulate_with_eth_simulate_v1(
            w3,
            [{"from": deployer, "to": vm_address, "data": execute_calldata}],
        )
        assert len(results) == 1
        result = results[0]
        assert result.success is True, f"Program execution failed: {result.revert_reason}"

        # The Transfer event emitted by MockToken should be captured
        token_transfers = [t for t in result.transfers if t.token == Web3.to_checksum_address(token_address)]
        assert len(token_transfers) >= 1
        t = token_transfers[0]
        assert t.from_address == Web3.to_checksum_address(vm_address)
        assert t.to_address == Web3.to_checksum_address(recipient)
        assert t.amount == transfer_amount

    async def test_simulate_v1_native_eth_transfer(self, sim_ctx):
        """eth_simulateV1 with traceTransfers captures native ETH transfers."""
        w3 = sim_ctx["w3"]
        await _set_eth_balance(w3, ETH_WHALE, 10**20)  # 100 ETH
        recipient = "0x000000000000000000000000000000000000dEaD"
        eth_amount = 10**18  # 1 ETH

        results = await simulate_with_eth_simulate_v1(
            w3,
            [{"from": ETH_WHALE, "to": recipient, "value": hex(eth_amount), "data": "0x"}],
            trace_transfers=True,
        )
        assert len(results) == 1
        result = results[0]
        assert result.success is True

        # With traceTransfers, native ETH shows as a Transfer from zero-address token
        assert result.transfers, "Expected transfers when trace_transfers=True"
        eth_transfers = [t for t in result.transfers if t.token == "0x0000000000000000000000000000000000000000"]
        assert eth_transfers, "Expected at least one synthetic ETH transfer"
        assert any(
            t.from_address == Web3.to_checksum_address(ETH_WHALE)
            and t.to_address == Web3.to_checksum_address(recipient)
            and t.amount == eth_amount
            for t in eth_transfers
        )


@pytest.mark.fork
class TestSimulateTx:
    """Tests for the high-level simulate_tx auto-selection function."""

    async def test_simulate_tx_basic(self, sim_ctx):
        """simulate_tx on a view call selects a method automatically."""
        w3 = sim_ctx["w3"]
        result = await simulate_tx(
            w3,
            {"to": USDC.address, "data": ERC20.fns.balanceOf(ETH_WHALE).data},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True

    async def test_simulate_tx_with_approvals(self, sim_ctx):
        """simulate_tx handles ERC-20 approval automatically via the approvals param.

        Simulates an ``approve`` on MockToken before checking the allowance for
        the VM contract — the realistic approve-before-swap pattern.  The
        implementation picks the right backend encoding (prepend-call for
        eth_simulateV1, state-override for debug_traceCall/eth_call) without the
        caller needing to know which method is used.
        """
        w3 = sim_ctx["w3"]
        user = sim_ctx["user"]
        vm_address = sim_ctx["vm_address"]
        token_address = sim_ctx["token_address"]
        check_call = {
            "to": token_address,
            "data": ERC20.fns.allowance(user, vm_address).data,
        }
        result = await simulate_tx(
            w3,
            check_call,
            approvals=[{"token": token_address, "owner": user, "spender": vm_address, "amount": 10**18}],
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        # The main call is the allowance check after the simulated approve
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == 10**18

    async def test_simulate_tx_composition_program(self, sim_ctx):
        """simulate_tx on a DeFiVM composition program captures token transfers.

        Builds a :class:`~pydefi.vm.Program` that transfers MockToken from
        inside the VM, then uses ``simulate_tx`` to preview the side effects
        before broadcasting.
        """
        w3 = sim_ctx["w3"]
        vm = sim_ctx["vm"]
        vm_address = sim_ctx["vm_address"]
        token_address = sim_ctx["token_address"]
        deployer = sim_ctx["deployer"]
        recipient = sim_ctx["recipient"]
        transfer_amount = 5 * 10**18

        # Build DeFiVM composition bytecodes
        program = (
            Program().call_contract(token_address, ERC20.fns.transfer(recipient, transfer_amount).data).pop().build()
        )
        execute_calldata = vm.functions.execute(program)._encode_transaction_data()

        result = await simulate_tx(
            w3,
            {"from": deployer, "to": vm_address, "data": execute_calldata},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        # Transfer events are exposed by eth_simulateV1 / debug_traceCall
        assert result.transfers, "Expected simulate_tx to return parsed transfers for the VM-emitted MockToken transfer"
        token_transfers = [t for t in result.transfers if t.token == Web3.to_checksum_address(token_address)]
        assert token_transfers, "Expected simulate_tx to include at least one MockToken transfer"
        assert any(t.amount == transfer_amount for t in token_transfers)

    async def test_simulate_tx_revert(self, sim_ctx):
        """simulate_tx reports failure for a reverted call."""
        w3 = sim_ctx["w3"]
        result = await simulate_tx(
            w3,
            {"to": USDC.address, "data": "0xdeadbeef"},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is False

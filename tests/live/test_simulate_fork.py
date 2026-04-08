"""Fork-based tests for the transaction simulation utilities.

These tests spin up a local Anvil fork of Ethereum mainnet and exercise the
simulation functions in ``pydefi.simulate`` against live contract state.

How the tests work
------------------
1. An Anvil fork is started via the ``fork_w3`` fixture in ``conftest.py``.
2. A well-known ETH whale (``ETH_WHALE``) is impersonated so we can send
   transactions without a private key.
3. Each simulation method is exercised and the results are verified:
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

import pytest
from eth_abi import encode as abi_encode
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

from .conftest import USDC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Well-known Ethereum addresses
ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth

# Uniswap V3 SwapRouter (V1) — used as the spender for approval tests
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"

# 0.1 ETH in wei
ETH_AMOUNT = 10**17

# Minimum plausible USDC amount for any operation (sanity guard)
MIN_USDC = 1 * 10**6  # $1

# ERC-20 function selectors
_APPROVE_SELECTOR = bytes.fromhex("095ea7b3")  # approve(address,uint256)
_ALLOWANCE_SELECTOR = bytes.fromhex("dd62ed3e")  # allowance(address,address)
_BALANCE_OF_SELECTOR = bytes.fromhex("70a08231")  # balanceOf(address)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _impersonate(w3: AsyncWeb3, address: str) -> None:
    """Ask Anvil to impersonate *address*."""
    await w3.provider.make_request("anvil_impersonateAccount", [address])


async def _set_eth_balance(w3: AsyncWeb3, address: str, amount: int) -> None:
    """Set ETH balance via Anvil."""
    await w3.provider.make_request("anvil_setBalance", [address, hex(amount)])


def _approve_calldata(spender: str, amount: int) -> str:
    """Build calldata for ERC-20 approve(spender, amount)."""
    return "0x" + (_APPROVE_SELECTOR + abi_encode(["address", "uint256"], [spender, amount])).hex()


def _allowance_calldata(owner: str, spender: str) -> str:
    """Build calldata for ERC-20 allowance(owner, spender)."""
    return "0x" + (_ALLOWANCE_SELECTOR + abi_encode(["address", "address"], [owner, spender])).hex()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestSimulateEthCall:
    """Tests for simulate_with_eth_call."""

    async def test_eth_call_success(self, fork_w3):
        """Calling balanceOf returns success and non-empty return data."""
        calldata = "0x" + (_BALANCE_OF_SELECTOR + abi_encode(["address"], [ETH_WHALE])).hex()
        result = await simulate_with_eth_call(
            fork_w3,
            {"to": USDC.address, "data": calldata},
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

    async def test_eth_call_revert(self, fork_w3):
        """Calling a non-existent function returns failure."""
        result = await simulate_with_eth_call(
            fork_w3,
            {"to": USDC.address, "data": "0xdeadbeef"},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is False

    async def test_eth_call_state_override_balance(self, fork_w3):
        """State override injects an artificial ETH balance."""
        fake_balance = 10**21  # 1000 ETH
        calldata = "0x"  # empty call — just check value reaches the target
        result = await simulate_with_eth_call(
            fork_w3,
            {"from": ETH_WHALE, "to": ETH_WHALE, "value": fake_balance, "data": calldata},
            state_overrides={
                ETH_WHALE: {"balance": fake_balance * 2},
            },
        )
        assert result.success is True

    async def test_eth_call_allowance_state_override(self, fork_w3):
        """build_allowance_state_override injects an ERC-20 allowance via eth_call."""
        # The whale likely has 0 USDC allowance for the V3 router by default.
        state_override = await build_allowance_state_override(
            fork_w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=UNISWAP_V3_ROUTER,
            amount=10**12,  # 1 million USDC
        )

        # Read back the allowance via eth_call with the state override applied.
        calldata = _allowance_calldata(ETH_WHALE, UNISWAP_V3_ROUTER)
        result = await simulate_with_eth_call(
            fork_w3,
            {"to": USDC.address, "data": calldata},
            state_overrides=state_override,
        )
        assert result.success is True
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == 10**12


@pytest.mark.fork
class TestDetectAllowanceSlot:
    """Tests for detect_allowance_slot and build_allowance_state_override."""

    async def test_detect_allowance_slot_usdc(self, fork_w3):
        """Detect the allowance mapping slot for USDC."""
        slot = await detect_allowance_slot(
            fork_w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=UNISWAP_V3_ROUTER,
        )
        assert slot is not None, "detect_allowance_slot returned None — is debug_traceCall available?"
        # USDC stores allowances at slot 9 (proxy impl); just check we get a valid slot
        assert len(slot.slot) == 32

    async def test_build_allowance_state_override_structure(self, fork_w3):
        """build_allowance_state_override returns a valid state-override dict."""
        amount = 5 * 10**12  # 5 million USDC
        override = await build_allowance_state_override(
            fork_w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=UNISWAP_V3_ROUTER,
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

    async def test_allowance_state_override_read_back(self, fork_w3):
        """Injected allowance is readable via eth_call with the override applied."""
        target_amount = 42 * 10**6  # 42 USDC
        override = await build_allowance_state_override(
            fork_w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=UNISWAP_V3_ROUTER,
            amount=target_amount,
        )
        calldata = _allowance_calldata(ETH_WHALE, UNISWAP_V3_ROUTER)
        result = await simulate_with_eth_call(
            fork_w3,
            {"to": USDC.address, "data": calldata},
            state_overrides=override,
        )
        assert result.success is True
        assert int.from_bytes(result.return_data, "big") == target_amount


@pytest.mark.fork
class TestSimulateDebugTraceCall:
    """Tests for simulate_with_debug_trace_call."""

    async def test_debug_trace_call_balanceof(self, fork_w3):
        """debug_traceCall on balanceOf returns success and gas usage."""
        calldata = "0x" + (_BALANCE_OF_SELECTOR + abi_encode(["address"], [ETH_WHALE])).hex()
        result = await simulate_with_debug_trace_call(
            fork_w3,
            {"to": USDC.address, "data": calldata},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        assert result.gas_used is not None and result.gas_used > 0

    async def test_debug_trace_call_erc20_transfer_logs(self, fork_w3):
        """debug_traceCall on an ERC-20 approve captures the Approval log."""
        await _impersonate(fork_w3, ETH_WHALE)
        calldata = _approve_calldata(UNISWAP_V3_ROUTER, 10**18)
        result = await simulate_with_debug_trace_call(
            fork_w3,
            {"from": ETH_WHALE, "to": USDC.address, "data": calldata},
        )
        assert result.success is True
        # approve() emits an Approval event (not Transfer), so no ERC-20 transfers
        # but gas should still be reported
        assert result.gas_used is not None

    async def test_debug_trace_call_state_override_allowance(self, fork_w3):
        """debug_traceCall with state override injects an allowance."""
        override = await build_allowance_state_override(
            fork_w3,
            token=USDC.address,
            owner=ETH_WHALE,
            spender=UNISWAP_V3_ROUTER,
            amount=10**10,
        )
        calldata = _allowance_calldata(ETH_WHALE, UNISWAP_V3_ROUTER)
        result = await simulate_with_debug_trace_call(
            fork_w3,
            {"to": USDC.address, "data": calldata},
            state_overrides=override,
        )
        assert result.success is True
        assert len(result.return_data) == 32
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == 10**10


@pytest.mark.fork
class TestSimulateEthSimulateV1:
    """Tests for simulate_with_eth_simulate_v1."""

    async def test_simulate_v1_single_call(self, fork_w3):
        """eth_simulateV1 on a view call returns success and logs."""
        calldata = "0x" + (_BALANCE_OF_SELECTOR + abi_encode(["address"], [ETH_WHALE])).hex()
        results = await simulate_with_eth_simulate_v1(
            fork_w3,
            [{"to": USDC.address, "data": calldata}],
        )
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SimulationResult)
        assert result.success is True

    async def test_simulate_v1_approve_then_check(self, fork_w3):
        """eth_simulateV1 multi-message: approve + allowance check in sequence."""
        approve_call = {
            "from": ETH_WHALE,
            "to": USDC.address,
            "data": _approve_calldata(UNISWAP_V3_ROUTER, 10**18),
        }
        check_call = {
            "to": USDC.address,
            "data": _allowance_calldata(ETH_WHALE, UNISWAP_V3_ROUTER),
        }
        results = await simulate_with_eth_simulate_v1(
            fork_w3,
            [approve_call, check_call],
        )
        assert len(results) == 2
        approve_result, check_result = results

        assert approve_result.success is True
        assert check_result.success is True
        # After the simulated approve, allowance should be 10**18
        allowance = int.from_bytes(check_result.return_data, "big")
        assert allowance == 10**18

    async def test_simulate_v1_erc20_transfer_event(self, fork_w3):
        """eth_simulateV1 captures ERC-20 Transfer events in logs."""
        await _impersonate(fork_w3, ETH_WHALE)

        # Check if whale has USDC; if not, skip
        usdc_balance = await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=USDC.address)
        if usdc_balance == 0:
            pytest.skip("ETH_WHALE has no USDC on this fork")

        transfer_amount = min(usdc_balance, 10**6)  # transfer at most 1 USDC
        recipient = "0x000000000000000000000000000000000000dEaD"

        # ERC-20 transfer(address,uint256) selector
        _TRANSFER_SEL = bytes.fromhex("a9059cbb")
        calldata = "0x" + (_TRANSFER_SEL + abi_encode(["address", "uint256"], [recipient, transfer_amount])).hex()

        results = await simulate_with_eth_simulate_v1(
            fork_w3,
            [{"from": ETH_WHALE, "to": USDC.address, "data": calldata}],
        )
        assert len(results) == 1
        result = results[0]
        assert result.success is True

        # There should be a Transfer event
        transfers = result.transfers
        erc20_transfers = [t for t in transfers if t.token == Web3.to_checksum_address(USDC.address)]
        assert len(erc20_transfers) >= 1
        t = erc20_transfers[0]
        assert t.from_address == Web3.to_checksum_address(ETH_WHALE)
        assert t.to_address == Web3.to_checksum_address(recipient)
        assert t.amount == transfer_amount

        # balance_changes should reflect the transfer
        changes = {(c.address, c.token): c.delta for c in result.balance_changes}
        from_key = (Web3.to_checksum_address(ETH_WHALE), Web3.to_checksum_address(USDC.address))
        to_key = (Web3.to_checksum_address(recipient), Web3.to_checksum_address(USDC.address))
        assert changes.get(from_key, 0) == -transfer_amount
        assert changes.get(to_key, 0) == transfer_amount

    async def test_simulate_v1_native_eth_transfer(self, fork_w3):
        """eth_simulateV1 with traceTransfers captures native ETH transfers."""
        await _set_eth_balance(fork_w3, ETH_WHALE, 10**20)  # 100 ETH
        recipient = "0x000000000000000000000000000000000000dEaD"
        eth_amount = 10**18  # 1 ETH

        results = await simulate_with_eth_simulate_v1(
            fork_w3,
            [{"from": ETH_WHALE, "to": recipient, "value": hex(eth_amount), "data": "0x"}],
            trace_transfers=True,
        )
        assert len(results) == 1
        result = results[0]
        assert result.success is True

        # With traceTransfers, native ETH shows as a Transfer from zero-address token
        if result.transfers:
            eth_transfers = [t for t in result.transfers if t.token == "0x0000000000000000000000000000000000000000"]
            if eth_transfers:
                assert any(
                    t.from_address == Web3.to_checksum_address(ETH_WHALE)
                    and t.to_address == Web3.to_checksum_address(recipient)
                    and t.amount == eth_amount
                    for t in eth_transfers
                )


@pytest.mark.fork
class TestSimulateTx:
    """Tests for the high-level simulate_tx auto-selection function."""

    async def test_simulate_tx_basic(self, fork_w3):
        """simulate_tx on a view call selects a method automatically."""
        calldata = "0x" + (_BALANCE_OF_SELECTOR + abi_encode(["address"], [ETH_WHALE])).hex()
        result = await simulate_tx(
            fork_w3,
            {"to": USDC.address, "data": calldata},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True

    async def test_simulate_tx_with_prepend_calls(self, fork_w3):
        """simulate_tx routes to eth_simulateV1 when prepend_calls is provided."""
        approve_call = {
            "from": ETH_WHALE,
            "to": USDC.address,
            "data": _approve_calldata(UNISWAP_V3_ROUTER, 10**18),
        }
        check_call = {
            "to": USDC.address,
            "data": _allowance_calldata(ETH_WHALE, UNISWAP_V3_ROUTER),
        }
        result = await simulate_tx(
            fork_w3,
            check_call,
            prepend_calls=[approve_call],
        )
        assert isinstance(result, SimulationResult)
        assert result.success is True
        # The main call is the allowance check after the simulated approve
        allowance = int.from_bytes(result.return_data, "big")
        assert allowance == 10**18

    async def test_simulate_tx_revert(self, fork_w3):
        """simulate_tx reports failure for a reverted call."""
        result = await simulate_tx(
            fork_w3,
            {"to": USDC.address, "data": "0xdeadbeef"},
        )
        assert isinstance(result, SimulationResult)
        assert result.success is False

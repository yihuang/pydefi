"""Regression: patched interpreter rejects forbidden opcodes at runtime (issue #138).

Five opcodes — ``SLOAD``, ``SSTORE``, ``CALLCODE``, ``DELEGATECALL``,
``SELFDESTRUCT`` — must revert.  ``TSTORE``/``TLOAD`` and dynamic JUMPs must not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from eth_utils import keccak
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm import Program
from tests.addrs import INTERPRETER_ADDR
from tests.live.sol_utils import compile_sol_file, deploy

_REVERT = (ContractLogicError, Web3RPCError)

_DEFI_VM_SOL = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "DeFiVM.sol"
_VENDOR_CONSTANTS = Path(__file__).resolve().parents[2] / "vendor" / "evm-interpreter" / "Constants.sol"
_FAKE_TARGET = b"\xde" * 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_defi_vm() -> dict:
    return compile_sol_file(_DEFI_VM_SOL, "DeFiVM")


@pytest.fixture(scope="module")
async def ctx(fork_w3_module, compiled_defi_vm, interpreter_addr) -> dict:
    """Deploy DeFiVM pointed at the (fixture-managed) patched interpreter."""
    w3 = fork_w3_module
    deployer = (await w3.eth.accounts)[0]
    vm_addr = await deploy(w3, compiled_defi_vm, deployer, interpreter_addr)
    return {
        "w3": w3,
        "deployer": deployer,
        "vm": w3.eth.contract(address=vm_addr, abi=compiled_defi_vm["abi"]),
        "vm_addr": vm_addr,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _execute(ctx: dict, program: bytes) -> dict:
    """Send ``vm.execute(program)`` and return the tx receipt."""
    tx = await ctx["vm"].functions.execute(program).transact({"from": ctx["deployer"]})
    return await ctx["w3"].eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestPatchedInterpreter:
    """Issue #138 regression suite: interpreter-level opcode rejection."""

    @pytest.mark.parametrize(
        "program,name",
        [
            (bytes([0x60, 0x00, 0x54, 0x00]), "SLOAD"),
            (bytes([0x60, 0x00, 0x60, 0x00, 0x55, 0x00]), "SSTORE"),
            (bytes([0x60, 0] * 7 + [0xF2, 0x00]), "CALLCODE"),
            (bytes([0x60, 0] * 6 + [0xF4, 0x00]), "DELEGATECALL"),
            (bytes([0x60, 0, 0xFF]), "SELFDESTRUCT"),
        ],
    )
    async def test_rejects_forbidden_opcode_at_runtime(self, ctx, program, name):
        with pytest.raises(_REVERT):
            await _execute(ctx, program)

    async def test_dynamic_jump_accepted(self, ctx):
        # PUSH1 5; PUSH1 5; ADD; JUMP; STOP×4; JUMPDEST; STOP — computed target.
        program = bytes([0x60, 0x05, 0x60, 0x05, 0x01, 0x56, 0x00, 0x00, 0x00, 0x00, 0x5B, 0x00])
        assert (await _execute(ctx, program))["status"] == 1

    @pytest.mark.parametrize(
        "trailer,name",
        [
            (bytes([0x60, 0x00, 0x54, 0x00]), "SLOAD"),
            (bytes([0x60, 0x00, 0x60, 0x00, 0x55, 0x00]), "SSTORE"),
            (bytes([0x60, 0] * 7 + [0xF2, 0x00]), "CALLCODE"),
            (bytes([0x60, 0] * 6 + [0xF4, 0x00]), "DELEGATECALL"),
            (bytes([0x60, 0, 0xFF]), "SELFDESTRUCT"),
        ],
    )
    async def test_dynamic_jump_into_forbidden_opcode_reverts(self, ctx, trailer, name):
        """Computed-target JUMP must not bypass interpreter opcode rejection.

        The patch sits at the dispatch-table level — any path that executes
        a forbidden opcode reverts, regardless of how control got there.
        """
        # PUSH1 6; PUSH1 7; ADD; JUMP; STOP×7; JUMPDEST  → JUMPDEST at offset 0x0d
        prefix = bytes([0x60, 0x06, 0x60, 0x07, 0x01, 0x56] + [0x00] * 7 + [0x5B])
        with pytest.raises(_REVERT):
            await _execute(ctx, prefix + trailer)

    @pytest.mark.parametrize(
        "program,name",
        [
            (bytes([0x00]), "STOP-only"),
            (bytes([0x5B, 0x00]), "JUMPDEST+STOP"),
            (bytes([0x60, 0x01, 0x50, 0x00]), "PUSH1+POP+STOP"),
            (bytes([0x60, 0x02, 0x60, 0x03, 0x01, 0x50, 0x00]), "ADD+POP+STOP"),
        ],
    )
    async def test_short_safe_programs_not_false_killed(self, ctx, program, name):
        """Programs without forbidden opcodes should execute successfully."""
        receipt = await _execute(ctx, program)
        assert receipt["status"] == 1, f"{name} unexpectedly failed"

    async def test_compile_shape_consistency(self, ctx):
        """Equivalent safe programs from different build options must both pass."""
        prog_a = Program()
        sum_a = prog_a.add(1, 2)
        prog_a.assert_(prog_a.eq(sum_a, 3), "sum mismatch")
        prog_a.builder.stop()

        prog_b = Program()
        sum_b = prog_b.add(1, 2)
        prog_b.assert_(prog_b.eq(sum_b, 3), "sum mismatch")
        prog_b.builder.stop()

        receipt_a = await _execute(ctx, prog_a.build(disable_constant_folding=True))
        receipt_b = await _execute(ctx, prog_b.build(disable_constant_folding=False))
        assert receipt_a["status"] == 1
        assert receipt_b["status"] == 1

    async def test_tstore_tload_accepted(self, ctx):
        # PUSH1 0x42; PUSH1 0x07; TSTORE; PUSH1 0x07; TLOAD; POP; STOP
        program = bytes([0x60, 0x42, 0x60, 0x07, 0x5D, 0x60, 0x07, 0x5C, 0x50, 0x00])
        assert (await _execute(ctx, program))["status"] == 1

    async def test_arithmetic_runs(self, ctx):
        # PUSH1 1; PUSH1 2; ADD; PUSH1 4; MUL; POP; STOP  → (1+2)*4, discard.
        program = bytes([0x60, 0x01, 0x60, 0x02, 0x01, 0x60, 0x04, 0x02, 0x50, 0x00])
        assert (await _execute(ctx, program))["status"] == 1

    async def test_realistic_swap_program(self, ctx):
        """Gas-cost regression sentinel: single V3-style CALL stays < 33k.

        The patched interpreter runs this at ~30k (spike-measured 29,870);
        the < 33k bound catches regressions without being flaky.
        """
        prog = Program()
        prog.call_raw(_FAKE_TARGET, b"\x12\x8a\xcb\x08" + b"\x00" * (32 * 8))
        prog.builder.stop()
        receipt = await _execute(ctx, prog.build())
        assert receipt["status"] == 1
        assert receipt["gasUsed"] < 33_000, f"single-hop swap took {receipt['gasUsed']} gas; expected < 33k"

    async def test_defi_vm_default_interpreter_is_patched_address(self, ctx, compiled_defi_vm, interpreter_addr):
        """``DeFiVM(address(0))`` resolves to ``INTERPRETER_ADDR`` (= patched).

        On a fresh Anvil fork the deterministic CREATE2 address has no code,
        so ``DeFiVM(0)`` construction would hit the ``InterpreterNotDeployed``
        guard.  Mirror the fixture-deployed patched runtime to
        ``INTERPRETER_ADDR`` via ``anvil_setCode``, then deploy ``DeFiVM(0)``
        and check the immutable INTERPRETER (embedded in deployed code)
        equals ``INTERPRETER_ADDR``.  Confirms both the constant wiring and
        the existence check.
        """
        w3 = ctx["w3"]
        patched_runtime = await w3.eth.get_code(interpreter_addr)
        await w3.provider.make_request("anvil_setCode", [INTERPRETER_ADDR, patched_runtime.hex()])

        vm_addr = await deploy(w3, compiled_defi_vm, ctx["deployer"], "0x" + "00" * 20)
        runtime = bytes(await w3.eth.get_code(vm_addr))
        assert bytes(INTERPRETER_ADDR) in runtime, (
            f"expected {INTERPRETER_ADDR} (raw 20 bytes) embedded in DeFiVM runtime "
            f"via immutable INTERPRETER, but not found"
        )

    async def test_vendored_upstream_matches_onchain_deployment(self, ctx):
        src = _VENDOR_CONSTANTS.read_text()
        upstream_addr = re.search(r"constant INTERPRETER_ADDRESS = (0x[0-9a-fA-F]{40})", src).group(1)
        upstream_codehash = re.search(r"constant INTERPRETER_CODEHASH =\s*\n?\s*(0x[0-9a-fA-F]{64})", src).group(1)

        code = bytes(await ctx["w3"].eth.get_code(upstream_addr))
        if len(code) <= 1:
            pytest.skip("upstream Analog-Labs interpreter not deployed on this fork")
        assert keccak(code) == bytes.fromhex(upstream_codehash[2:]), (
            "vendored Constants.sol does not match the on-chain upstream interpreter"
        )

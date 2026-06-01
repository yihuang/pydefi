"""Fork tests for EurekaComposer — IBC v2 (Eureka) sender backed by DeFiVM.

EurekaComposer registers a DeFiVM follow-up program at send time and runs it on
the ack/timeout callback. Its program is loaded from storage into ``bytes
memory``, so it runs via the seam's memory variant ``_runProgramMemory``
(ProgramExecutor → InterpreterRunner). These tests drive that path end-to-end
against the real Analog-Labs interpreter on an Anvil mainnet fork — the
ack/timeout callbacks, ``onlyTransfer`` access control, and sub-call revert
bubbling — with a mock ICS20Transfer standing in for the IBC router (its
fireAck/fireTimeout helpers invoke the callbacks as msg.sender).

Run with::

    pytest -m fork tests/live/test_eureka_composer_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from vyper.venom.basicblock import IRLiteral
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.vm import Program
from tests.live.sol_utils import (
    MOCK_REVERTING_TARGET_SOL,
    MOCK_TARGET_SOL,
    MOCK_TOKEN_SOL,
    compile_sol_file,
    compile_sol_source,
    deploy,
    ensure_solc,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EUREKA_SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "EurekaComposer.sol"

_SOURCE_CLIENT = "client-0"


def _compose_program(target_address, target_calldata: bytes, *, value: int = 0) -> bytes:
    """DeFiVM program: read the staged transient params then call the target.

    EurekaComposer stages slot 0 = success, slot 1 = sequence; the program may
    read them via TLOAD. On CALL failure the callback reverts.
    """
    prog = Program()
    prog.builder.tload(IRLiteral(0))  # success (staged by the composer)
    prog.builder.tload(IRLiteral(1))  # sequence
    success = prog.call_raw(target_address, target_calldata, value=value)
    prog.assert_(success)
    prog.builder.stop()
    return prog.build()


# ---------------------------------------------------------------------------
# Inline mock contracts
# ---------------------------------------------------------------------------

_MOCKS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// Mirrors EurekaComposer's IIBCAppCallbacks_minimal callback structs so the
// ABI-encoded selector + tuple layout match exactly.
interface IEurekaComposer {
    struct Payload { string sourcePort; string destPort; string version; string encoding; bytes value; }
    struct OnAcknowledgementPacketCallback {
        string sourceClient; string destinationClient; uint64 sequence;
        Payload payload; bytes acknowledgement; address relayer;
    }
    struct OnTimeoutPacketCallback {
        string sourceClient; string destinationClient; uint64 sequence;
        Payload payload; address relayer;
    }
    function onAckPacket(bool success, OnAcknowledgementPacketCallback calldata msg_) external;
    function onTimeoutPacket(OnTimeoutPacketCallback calldata msg_) external;
}

// Stand-in for the IBC router: returns sequences and drives the callbacks so
// they originate from this address (satisfying EurekaComposer.onlyTransfer).
contract MockICS20Transfer {
    struct SendTransferMsg {
        address denom; uint256 amount; string receiver; string sourceClient;
        string destPort; uint64 timeoutTimestamp; string memo;
    }
    uint64 public nextSequence = 1;

    function sendTransfer(SendTransferMsg calldata) external returns (uint64) {
        return nextSequence++;
    }

    function fireAck(address composer, bool success, string calldata sourceClient, uint64 sequence) external {
        IEurekaComposer.Payload memory p = IEurekaComposer.Payload("", "", "", "", "");
        IEurekaComposer.OnAcknowledgementPacketCallback memory cb =
            IEurekaComposer.OnAcknowledgementPacketCallback(sourceClient, "", sequence, p, "", address(0));
        IEurekaComposer(composer).onAckPacket(success, cb);
    }

    function fireTimeout(address composer, string calldata sourceClient, uint64 sequence) external {
        IEurekaComposer.Payload memory p = IEurekaComposer.Payload("", "", "", "", "");
        IEurekaComposer.OnTimeoutPacketCallback memory cb =
            IEurekaComposer.OnTimeoutPacketCallback(sourceClient, "", sequence, p, address(0));
        IEurekaComposer(composer).onTimeoutPacket(cb);
    }
}
"""


def _compile_mocks() -> dict[str, dict]:
    ensure_solc("0.8.24")
    result = solcx.compile_source(_MOCKS_SOL, output_values=["abi", "bin"], solc_version="0.8.24")
    mocks = {name: result[f"<stdin>:{name}"] for name in ("MockICS20Transfer",)}
    # MockToken/MockTarget/RevertingTarget come from the shared sol_utils sources.
    mocks["MockToken"] = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    mocks["MockTarget"] = compile_sol_source(MOCK_TARGET_SOL, "MockTarget")
    mocks["RevertingTarget"] = compile_sol_source(MOCK_REVERTING_TARGET_SOL, "RevertingTarget")
    return mocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_eureka_composer():
    return compile_sol_file(EUREKA_SOL_FILE, "EurekaComposer")


@pytest.fixture(scope="module")
def compiled_eureka_mocks():
    return _compile_mocks()


@pytest.fixture(scope="module")
async def ctx(fork_w3_module, compiled_eureka_composer, compiled_eureka_mocks, interpreter_addr):
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    token_addr = await deploy(w3, compiled_eureka_mocks["MockToken"], deployer)
    transfer_addr = await deploy(w3, compiled_eureka_mocks["MockICS20Transfer"], deployer)
    target_addr = await deploy(w3, compiled_eureka_mocks["MockTarget"], deployer)
    reverting_addr = await deploy(w3, compiled_eureka_mocks["RevertingTarget"], deployer)

    # EurekaComposer DELEGATECALLs / RUNCODEs through the seam — it takes the
    # interpreter address (default backend).
    composer_addr = await deploy(w3, compiled_eureka_composer, deployer, transfer_addr, interpreter_addr)

    def c(compiled, addr):
        return Contract(abi=compiled["abi"], tx={"to": Web3.to_checksum_address(addr)})

    return {
        "w3": w3,
        "deployer": deployer,
        "accounts": accounts,
        "token": c(compiled_eureka_mocks["MockToken"], token_addr),
        "token_addr": token_addr,
        "transfer": c(compiled_eureka_mocks["MockICS20Transfer"], transfer_addr),
        "transfer_addr": transfer_addr,
        "target": c(compiled_eureka_mocks["MockTarget"], target_addr),
        "target_addr": target_addr,
        "reverting_addr": reverting_addr,
        "composer": c(compiled_eureka_composer, composer_addr),
        "composer_addr": composer_addr,
    }


async def _register(ctx, program: bytes, amount: int = 10**18) -> int:
    """Approve, call sendTransferAndCompose, return the assigned sequence."""
    w3, deployer = ctx["w3"], ctx["deployer"]
    await ctx["token"].fns.mint(deployer, amount).transact(w3, deployer)
    await ctx["token"].fns.approve(ctx["composer_addr"], amount).transact(w3, deployer)

    transfer_msg = (
        Web3.to_checksum_address(ctx["token_addr"]),  # denom
        amount,  # amount
        "cosmos1receiver",  # receiver
        _SOURCE_CLIENT,  # sourceClient
        "transfer",  # destPort
        0,  # timeoutTimestamp
        "",  # memo
    )
    receipt = await ctx["composer"].fns.sendTransferAndCompose(transfer_msg, program).transact(w3, deployer)
    assert receipt["status"] == 1, "sendTransferAndCompose reverted"
    evt = ctx["composer"].events.Composed.parse_logs(receipt["logs"])[0]
    return evt["args"]["sequence"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestEurekaComposerCompose:
    async def test_ack_runs_registered_program(self, ctx):
        """onAckPacket runs the stored program via _runProgramMemory and clears the slot."""
        w3, deployer = ctx["w3"], ctx["deployer"]
        target, composer, transfer = ctx["target"], ctx["composer"], ctx["transfer"]

        calldata = target.fns.execute(b"\xde\xad\xbe\xef").data
        program = _compose_program(ctx["target_addr"], calldata)
        seq = await _register(ctx, program)

        pre = await target.fns.callCount().call(w3)
        # program is registered (non-empty) before the callback
        assert len(await composer.fns.programs(_SOURCE_CLIENT, seq).call(w3)) > 0

        receipt = await transfer.fns.fireAck(ctx["composer_addr"], True, _SOURCE_CLIENT, seq).transact(w3, deployer)
        assert receipt["status"] == 1

        assert await target.fns.callCount().call(w3) == pre + 1
        assert bytes(await target.fns.lastData().call(w3)) == b"\xde\xad\xbe\xef"
        # slot cleared after execution
        assert len(await composer.fns.programs(_SOURCE_CLIENT, seq).call(w3)) == 0

    async def test_timeout_runs_registered_program(self, ctx):
        """onTimeoutPacket also runs the stored program through the memory seam."""
        w3, deployer = ctx["w3"], ctx["deployer"]
        target, transfer = ctx["target"], ctx["transfer"]

        calldata = target.fns.execute(b"timeout").data
        program = _compose_program(ctx["target_addr"], calldata)
        seq = await _register(ctx, program)

        pre = await target.fns.callCount().call(w3)
        receipt = await transfer.fns.fireTimeout(ctx["composer_addr"], _SOURCE_CLIENT, seq).transact(w3, deployer)
        assert receipt["status"] == 1
        assert await target.fns.callCount().call(w3) == pre + 1

    async def test_sub_call_revert_bubbles(self, ctx):
        """A reverting sub-call inside the program makes the callback revert."""
        w3, deployer = ctx["w3"], ctx["deployer"]
        transfer = ctx["transfer"]

        program = _compose_program(ctx["reverting_addr"], b"\xde\xad")
        seq = await _register(ctx, program)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await transfer.fns.fireAck(ctx["composer_addr"], True, _SOURCE_CLIENT, seq).transact(w3, deployer)


@pytest.mark.fork
class TestEurekaComposerAccessControl:
    async def test_only_transfer_can_callback(self, ctx):
        """Direct onAckPacket from a non-transfer caller reverts (onlyTransfer)."""
        w3, deployer = ctx["w3"], ctx["deployer"]
        composer = ctx["composer"]

        program = _compose_program(ctx["target_addr"], ctx["target"].fns.execute(b"x").data)
        seq = await _register(ctx, program)

        cb = (_SOURCE_CLIENT, "", seq, ("", "", "", "", b""), b"", "0x" + "00" * 20)
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.onAckPacket(True, cb).transact(w3, deployer)

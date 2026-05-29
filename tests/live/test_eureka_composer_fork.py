"""Fork tests for EurekaComposer — IBC v2 (Eureka) sender + DeFiVM compose.

Deploy on a local Anvil fork against mock tokens and a mock ICS20Transfer to
exercise the runtime paths the compile-only ABI test in tests/test_program.py
can't reach. Per-test docstrings cover the specifics.

Run with::

    pytest -m fork tests/live/test_eureka_composer_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.types import Address
from pydefi.vm import Program
from tests.live.sol_utils import (
    MOCK_TARGET_SOL,
    MOCK_TOKEN_SOL,
    compile_sol_file,
    compile_sol_source,
    deploy,
    ensure_solc,
)

# Reusable "this should revert" matcher for both anvil-direct and forked reverts.
_REVERT = (ContractLogicError, Web3RPCError)
_SOURCE_CLIENT = "07-tendermint-0"
_DEST_PORT = "transfer"
_RELAYER = "0x" + "11" * 20

REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "EurekaComposer.sol"


# ---------------------------------------------------------------------------
# Mock contracts Solidity source
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Non-standard ERC-20 (classic USDT): approve / transferFrom return
/// nothing. Forces the SafeERC20-style empty-returndata path in _erc20Call.
contract MockUSDT {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external { balanceOf[to] += amount; }

    function approve(address spender, uint256 amount) external {
        allowance[msg.sender][spender] = amount;
    }

    function transferFrom(address from, address to, uint256 amount) external {
        require(balanceOf[from] >= amount, "balance");
        if (from != msg.sender) {
            require(allowance[from][msg.sender] >= amount, "allowance");
            allowance[from][msg.sender] -= amount;
        }
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
    }
}

/// @notice Minimal ICS20Transfer stand-in. sendTransfer just hands back an
/// incrementing sequence (it does not escrow — the composer's job ends at
/// approve + submit). ``forward`` lets a test invoke the composer's
/// onlyTransfer callbacks as if from the real transfer app.
contract MockICS20Transfer {
    uint64 public seq;

    struct SendTransferMsg {
        address denom;
        uint256 amount;
        string receiver;
        string sourceClient;
        string destPort;
        uint64 timeoutTimestamp;
        string memo;
    }

    function sendTransfer(SendTransferMsg calldata) external returns (uint64) {
        return ++seq;
    }

    function forward(address to, bytes calldata data) external returns (bool ok, bytes memory ret) {
        (ok, ret) = to.call(data);
        require(ok, "forward failed");
    }
}
"""


@pytest.fixture(scope="module")
async def eureka_fork_w3(fork_w3_module):
    return fork_w3_module


@pytest.fixture(scope="module")
def compiled_eureka_composer():
    return compile_sol_file(SOL_FILE, "EurekaComposer")


@pytest.fixture(scope="module")
def compiled_mocks():
    ensure_solc("0.8.24")
    result = solcx.compile_source(_MOCK_CONTRACTS_SOL, output_values=["abi", "bin"], solc_version="0.8.24")
    return {
        # Reuse the shared generic bool-ERC-20 rather than redefining one.
        "MockToken": compile_sol_source(MOCK_TOKEN_SOL, "MockToken"),
        "MockUSDT": result["<stdin>:MockUSDT"],
        "MockICS20Transfer": result["<stdin>:MockICS20Transfer"],
        "MockTarget": compile_sol_source(MOCK_TARGET_SOL, "MockTarget"),
    }


@pytest.fixture(scope="module")
async def ctx(eureka_fork_w3, compiled_eureka_composer, compiled_mocks, interpreter_addr):
    """Deploy EurekaComposer and mock contracts once; share across tests."""
    w3 = eureka_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    ics20_address = await deploy(w3, compiled_mocks["MockICS20Transfer"], deployer)
    composer_address = await deploy(
        w3,
        compiled_eureka_composer,
        deployer,
        ics20_address,  # _ics20Transfer
        interpreter_addr,  # _interpreter
    )
    token_address = await deploy(w3, compiled_mocks["MockToken"], deployer)
    usdt_address = await deploy(w3, compiled_mocks["MockUSDT"], deployer)
    target_address = await deploy(w3, compiled_mocks["MockTarget"], deployer)

    composer = Contract(abi=compiled_eureka_composer["abi"], tx={"to": Web3.to_checksum_address(composer_address)})
    ics20 = Contract(abi=compiled_mocks["MockICS20Transfer"]["abi"], tx={"to": Web3.to_checksum_address(ics20_address)})
    token = Contract(abi=compiled_mocks["MockToken"]["abi"], tx={"to": Web3.to_checksum_address(token_address)})
    usdt = Contract(abi=compiled_mocks["MockUSDT"]["abi"], tx={"to": Web3.to_checksum_address(usdt_address)})
    target = Contract(abi=compiled_mocks["MockTarget"]["abi"], tx={"to": Web3.to_checksum_address(target_address)})

    return {
        "w3": w3,
        "accounts": accounts,
        "deployer": deployer,
        "composer": composer,
        "composer_address": composer_address,
        "ics20": ics20,
        "ics20_address": ics20_address,
        "token": token,
        "token_address": token_address,
        "usdt": usdt,
        "usdt_address": usdt_address,
        "target": target,
        "target_address": target_address,
    }


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _registered_program() -> bytes:
    """A minimal, non-empty DeFiVM program (just stops). Stored verbatim by the
    composer for send-path tests; never executed there."""
    prog = Program()
    prog.builder.stop()
    return prog.build()


def _send_msg(denom, amount: int, *, receiver: str = "mantra1qq", timeout: int = 1, memo: str = "") -> tuple:
    """SendTransferMsg struct tuple in ABI field order."""
    return (Web3.to_checksum_address(denom), amount, receiver, _SOURCE_CLIENT, _DEST_PORT, timeout, memo)


def _ack_msg(sequence: int) -> tuple:
    """OnAcknowledgementPacketCallback struct tuple."""
    payload = (_DEST_PORT, _DEST_PORT, "ics20-2", "abi", b"")  # sourcePort, destPort, version, encoding, value
    return (_SOURCE_CLIENT, "07-tendermint-1", sequence, payload, b"\x01", Web3.to_checksum_address(_RELAYER))


async def _send_and_compose(ctx: dict, program: bytes, *, token, token_address: Address, amount: int):
    """Mint *amount* to the deployer, approve the composer, and submit a
    ``sendTransferAndCompose``. Returns the transaction receipt (the send is
    what reverts in the negative cases)."""
    w3 = ctx["w3"]
    deployer = ctx["deployer"]
    await token.fns.mint(deployer, amount).transact(w3, deployer)
    await token.fns.approve(ctx["composer_address"], amount).transact(w3, deployer)
    return (
        await ctx["composer"]
        .fns.sendTransferAndCompose(_send_msg(token_address, amount), HexBytes(program))
        .transact(w3, deployer)
    )


async def _register(ctx: dict, program: bytes) -> int:
    """Submit a send so *program* is registered; return its sequence."""
    receipt = await _send_and_compose(
        ctx, program, token=ctx["token"], token_address=ctx["token_address"], amount=7 * 10**6
    )
    (evt,) = ctx["composer"].events.Composed.parse_logs(receipt["logs"])
    return evt["args"]["sequence"]


@pytest.mark.fork
class TestSendTransferAndCompose:
    async def test_standard_erc20_happy_path(self, ctx):
        """Bool-returning token: composer pulls funds, approves ICS20Transfer,
        and registers the program under the returned sequence."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        token = ctx["token"]
        token_address = ctx["token_address"]
        ics20_address = ctx["ics20_address"]

        amount = 1000 * 10**6
        program = _registered_program()
        receipt = await _send_and_compose(ctx, program, token=token, token_address=token_address, amount=amount)
        assert receipt["status"] == 1

        # Caller funds moved into the composer; ICS20Transfer was approved.
        assert await token.fns.balanceOf(composer_address).call(w3) == amount
        assert await token.fns.balanceOf(deployer).call(w3) == 0
        assert await token.fns.allowance(composer_address, ics20_address).call(w3) == amount

        # Program registered verbatim under the sequence from the Composed event.
        (evt,) = composer.events.Composed.parse_logs(receipt["logs"])
        sequence = evt["args"]["sequence"]
        assert bytes(await composer.fns.programs(_SOURCE_CLIENT, sequence).call(w3)) == program

    async def test_nonbool_erc20_happy_path(self, ctx):
        """Classic-USDT token (no bool return) works via the SafeERC20-style
        empty-returndata path."""
        w3 = ctx["w3"]
        composer_address = ctx["composer_address"]
        usdt = ctx["usdt"]
        usdt_address = ctx["usdt_address"]
        ics20_address = ctx["ics20_address"]

        amount = 500 * 10**6
        receipt = await _send_and_compose(
            ctx, _registered_program(), token=usdt, token_address=usdt_address, amount=amount
        )
        assert receipt["status"] == 1
        assert await usdt.fns.balanceOf(composer_address).call(w3) == amount
        assert await usdt.fns.allowance(composer_address, ics20_address).call(w3) == amount

    async def test_rejects_empty_program(self, ctx):
        """Zero-length program is rejected at send time (EmptyProgram)."""
        with pytest.raises(_REVERT):
            await _send_and_compose(ctx, b"", token=ctx["token"], token_address=ctx["token_address"], amount=10 * 10**6)

    async def test_rejects_no_code_denom(self, ctx):
        """A denom at an EOA (no code) must revert (TransferFromFailed) rather
        than 'succeed' with no funds moved — _erc20Call's code-length guard. A
        fresh key is used since default Anvil accounts may carry an EIP-7702
        delegation on a live fork."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]

        eoa_denom = w3.eth.account.create().address
        with pytest.raises(_REVERT):
            await composer.fns.sendTransferAndCompose(
                _send_msg(eoa_denom, 1), HexBytes(_registered_program())
            ).transact(w3, deployer)


@pytest.mark.fork
class TestCallbacks:
    async def test_onack_rejects_unauthorized_caller(self, ctx):
        """onAckPacket reverts (UnauthorizedCallback) when not called by the
        ICS20Transfer proxy."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]

        seq = await _register(ctx, _registered_program())
        with pytest.raises(_REVERT):
            # deployer != ics20Transfer → onlyTransfer gate trips.
            await composer.fns.onAckPacket(True, _ack_msg(seq)).transact(w3, deployer)

    async def test_onack_runs_and_clears_registered_program(self, ctx):
        """A genuine ack from the transfer app DELEGATECALLs the interpreter to
        run the registered program, then deletes it from the mapping."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        ics20 = ctx["ics20"]
        target = ctx["target"]
        target_address = ctx["target_address"]

        # Ack program (run in the composer's context): read the staged
        # success/sequence from transient slots 0/1, then CALL the target.
        prog = Program()
        _success = prog.builder.tload(IRLiteral(0))
        _sequence = prog.builder.tload(IRLiteral(1))
        prog.assert_(prog.call_raw(target_address, target.fns.execute(b"\xde\xad\xbe\xef").data))
        prog.builder.stop()
        seq = await _register(ctx, prog.build())

        pre_count = await target.fns.callCount().call(w3)
        # Invoke onAckPacket *as* the ICS20Transfer proxy via its forwarder.
        ack_calldata = composer.fns.onAckPacket(True, _ack_msg(seq)).data
        receipt = await ics20.fns.forward(composer_address, HexBytes(ack_calldata)).transact(w3, deployer)
        assert receipt["status"] == 1

        assert await target.fns.callCount().call(w3) == pre_count + 1
        # Program cleared after execution.
        assert bytes(await composer.fns.programs(_SOURCE_CLIENT, seq).call(w3)) == b""

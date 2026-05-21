"""Fork tests for CCTPComposer — Circle CCTP v2 compose receiver backed by DeFiVM.

These tests compile CCTPComposer.sol and DeFiVM.sol with py-solc-x, deploy them
alongside mock contracts on a local Anvil fork, and exercise the full
``receiveAndExecute`` flow including:

 - Basic compose execution (program embedded as hookData in CCTP v2 message)
 - Compose execution carrying ETH value to a sub-call
 - Prologue values (amountReceived, sourceDomain) pushed onto the stack
 - Fee deduction: amountReceived = amount - feeExecuted
 - Revert when CCTP ``receiveMessage`` fails (bad attestation)
 - Revert when message is too short
 - Revert when a sub-call inside the compose fails
 - Owner rescue of stuck ETH and ERC-20 tokens
 - Ownership transfer

Run with::

    pytest -m fork tests/live/test_cctp_composer_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3, Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.types import ZERO_ADDRESS, Address, Hash
from pydefi.vm import Program
from tests.live.sol_utils import compile_sol_file, deploy, deploy_mock_v3_pool, ensure_solc


def _compose_program(target_address: Address, target_calldata: bytes, *, value: int = 0) -> bytes:
    """Build the DeFiVM program embedded as CCTP hookData.

    The composer stages bridged parameters in DeFiVM's transient store:
    slot 0 = amountReceived, slot 1 = sourceDomain.  The program reads them
    via TLOAD when needed.

    On CALL failure the outer ``receiveAndExecute`` reverts (matching legacy
    ``call(require_success=True)``).
    """
    prog = Program()
    _amount = prog.builder.tload(IRLiteral(0))  # amountReceived
    _source_domain = prog.builder.tload(IRLiteral(1))  # sourceDomain
    success = prog.call_raw(target_address, target_calldata, value=value)
    prog.assert_(success)
    prog.builder.stop()
    return prog.build()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "CCTPComposer.sol"
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Mock contracts (inline Solidity)
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mock USDC / ERC-20 token with mint capability.
contract MockUSDC {
    string public name = "USD Coin";
    string public symbol = "USDC";
    uint8 public decimals = 6;

    mapping(address => uint256) public balanceOf;
    uint256 public totalSupply;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "insufficient");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

/// @notice Mock CCTP v2 MessageTransmitterV2.
///
/// Decodes the v2 BurnMessageV2 body (starts at byte 148 in the full message):
///   mintRecipient at body+36 (absolute: 184)
///   amount        at body+68 (absolute: 216)
///   feeExecuted   at body+164 (absolute: 312)
///
/// On success it mints (amount - feeExecuted) USDC to mintRecipient.
/// The ``fail`` flag simulates a bad-attestation failure.
contract MockMessageTransmitterV2 {
    // CCTP v2 message offsets (mirrors CCTPComposer.sol constants)
    uint256 private constant MSG_BODY_OFFSET          = 148;
    uint256 private constant MINT_RECIPIENT_OFFSET    = MSG_BODY_OFFSET + 36;   // 184
    uint256 private constant AMOUNT_OFFSET            = MSG_BODY_OFFSET + 68;   // 216
    uint256 private constant FEE_EXECUTED_OFFSET      = MSG_BODY_OFFSET + 164;  // 312
    uint256 private constant MIN_MESSAGE_LENGTH       = MSG_BODY_OFFSET + 228;  // 376

    address public immutable usdc;
    bool public fail;

    constructor(address _usdc) {
        usdc = _usdc;
    }

    function setFail(bool _fail) external {
        fail = _fail;
    }

    function receiveMessage(bytes calldata message, bytes calldata /*attestation*/) external returns (bool) {
        require(!fail, "MockMessageTransmitterV2: bad attestation");
        require(message.length >= MIN_MESSAGE_LENGTH, "MockMessageTransmitterV2: message too short");

        address mintRecipient = address(
            uint160(uint256(bytes32(message[MINT_RECIPIENT_OFFSET:MINT_RECIPIENT_OFFSET + 32])))
        );
        uint256 amount = uint256(bytes32(message[AMOUNT_OFFSET:AMOUNT_OFFSET + 32]));
        uint256 feeExecuted = uint256(bytes32(message[FEE_EXECUTED_OFFSET:FEE_EXECUTED_OFFSET + 32]));

        uint256 amountReceived = amount - feeExecuted;
        if (amountReceived > 0) {
            (bool ok, ) = usdc.call(
                abi.encodeWithSignature("mint(address,uint256)", mintRecipient, amountReceived)
            );
            require(ok, "MockMessageTransmitterV2: mint failed");
        }
        return true;
    }
}

/// @notice Mock target contract — records the most recent call.
contract MockTarget {
    event Called(address sender, uint256 value, bytes data);

    uint256 public callCount;
    bytes public lastData;
    uint256 public lastValue;

    function execute(bytes calldata data) external payable returns (bool) {
        callCount++;
        lastData = data;
        lastValue = msg.value;
        emit Called(msg.sender, msg.value, data);
        return true;
    }

    receive() external payable {}
}

/// @notice Mock target that always reverts.
contract RevertingTarget {
    error AlwaysReverts();

    fallback() external payable {
        revert AlwaysReverts();
    }
}
"""

# ---------------------------------------------------------------------------
# CCTP v2 message builder
# ---------------------------------------------------------------------------

_ETHEREUM_DOMAIN = 0  # CCTP domain ID for Ethereum


def make_cctp_v2_message(
    source_domain: int,
    nonce: bytes,
    amount: int,
    mint_recipient: Address,
    hook_data: bytes = b"",
    fee_executed: int = 0,
    destination_domain: int = 6,  # Base
    burn_token: Address = ZERO_ADDRESS,
) -> bytes:
    """Build a synthetic CCTP v2 burn message.

    MessageV2 header (148 bytes):
      [0:4]    version                    = 0
      [4:8]    sourceDomain
      [8:12]   destinationDomain
      [12:44]  nonce                      (bytes32)
      [44:76]  sender                     = zero-padded
      [76:108] recipient                  = zero-padded
      [108:140] destinationCaller         = zero-padded
      [140:144] minFinalityThreshold      = 1000
      [144:148] finalityThresholdExecuted = 1000

    BurnMessageV2 body (starts at 148):
      [0:4]    burnMessageVersion = 1
      [4:36]   burnToken          (32 bytes, right-aligned address)
      [36:68]  mintRecipient      (32 bytes, right-aligned address)
      [68:100] amount             (uint256)
      [100:132] messageSender     = zero-padded
      [132:164] maxFee            = zero-padded
      [164:196] feeExecuted       (uint256)
      [196:228] expirationBlock   = 0 (no expiry)
      [228:]   hookData           (DeFiVM program)
    """
    if isinstance(nonce, int):
        nonce = nonce.to_bytes(32, "big")
    assert len(nonce) == 32, "nonce must be 32 bytes"

    # MessageV2 header
    version = (0).to_bytes(4, "big")
    src_domain_bytes = source_domain.to_bytes(4, "big")
    dst_domain_bytes = destination_domain.to_bytes(4, "big")
    nonce_bytes = nonce  # bytes32
    sender_bytes = (0).to_bytes(32, "big")
    recipient_bytes = (0).to_bytes(32, "big")
    dst_caller_bytes = (0).to_bytes(32, "big")
    min_finality = (1000).to_bytes(4, "big")
    finality_executed = (1000).to_bytes(4, "big")

    header = (
        version
        + src_domain_bytes
        + dst_domain_bytes
        + nonce_bytes
        + sender_bytes
        + recipient_bytes
        + dst_caller_bytes
        + min_finality
        + finality_executed
    )
    assert len(header) == 148, f"header length {len(header)}"

    # BurnMessageV2 body
    burn_msg_version = (1).to_bytes(4, "big")
    burn_token_bytes = int.from_bytes(burn_token, "big").to_bytes(32, "big")
    mint_recipient_bytes = int.from_bytes(mint_recipient, "big").to_bytes(32, "big")
    amount_bytes = amount.to_bytes(32, "big")
    msg_sender_bytes = (0).to_bytes(32, "big")
    max_fee_bytes = (0).to_bytes(32, "big")
    fee_executed_bytes = fee_executed.to_bytes(32, "big")
    expiration_block_bytes = (0).to_bytes(32, "big")

    body = (
        burn_msg_version
        + burn_token_bytes
        + mint_recipient_bytes
        + amount_bytes
        + msg_sender_bytes
        + max_fee_bytes
        + fee_executed_bytes
        + expiration_block_bytes
        + hook_data
    )
    assert len(body) == 228 + len(hook_data), f"body length {len(body)}"

    return header + body


# ---------------------------------------------------------------------------
# Compile + deploy helpers
# ---------------------------------------------------------------------------


def _compile_cctp_composer() -> dict:
    return compile_sol_file(SOL_FILE, "CCTPComposer")


def _compile_defi_vm() -> dict:
    return compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")


def _compile_mock_contracts() -> dict[str, dict]:
    ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_CONTRACTS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return {
        "MockUSDC": result["<stdin>:MockUSDC"],
        "MockMessageTransmitterV2": result["<stdin>:MockMessageTransmitterV2"],
        "MockTarget": result["<stdin>:MockTarget"],
        "RevertingTarget": result["<stdin>:RevertingTarget"],
    }


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: Address, *args) -> Address:
    return await deploy(w3, compiled, deployer, *args)


# ---------------------------------------------------------------------------
# Module-scoped Anvil fork fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def cctp_fork_w3(fork_w3_module):
    return fork_w3_module


# ---------------------------------------------------------------------------
# Module-scoped setup: compile + deploy once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_cctp_composer():
    return _compile_cctp_composer()


@pytest.fixture(scope="module")
def compiled_mocks():
    return _compile_mock_contracts()


@pytest.fixture(scope="module")
def compiled_defi_vm():
    return _compile_defi_vm()


@pytest.fixture(scope="module")
async def ctx(cctp_fork_w3, compiled_cctp_composer, compiled_mocks, compiled_defi_vm, interpreter_addr):
    """Deploy CCTPComposer, DeFiVM, and mock contracts once; return shared context."""
    w3 = cctp_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy mock USDC token.
    usdc_address = await _deploy(w3, compiled_mocks["MockUSDC"], deployer)

    # Deploy mock MessageTransmitterV2 (mints USDC on receiveMessage).
    transmitter_address = await _deploy(w3, compiled_mocks["MockMessageTransmitterV2"], deployer, usdc_address)

    # Deploy DeFiVM (kept for tests that exercise DeFiVM.execute directly).
    vm_address = await _deploy(w3, compiled_defi_vm, deployer, interpreter_addr)

    # Deploy CCTPComposer.  Composer DELEGATECALLs the interpreter directly,
    # so it takes the interpreter address (not the VM address).
    composer_address = await _deploy(
        w3,
        compiled_cctp_composer,
        deployer,
        transmitter_address,  # _messageTransmitter
        usdc_address,  # _usdc
        interpreter_addr,  # _interpreter
        deployer,  # _owner
    )

    usdc = Contract(abi=compiled_mocks["MockUSDC"]["abi"], tx={"to": Web3.to_checksum_address(usdc_address)})
    transmitter = Contract(
        abi=compiled_mocks["MockMessageTransmitterV2"]["abi"], tx={"to": Web3.to_checksum_address(transmitter_address)}
    )
    composer = Contract(abi=compiled_cctp_composer["abi"], tx={"to": Web3.to_checksum_address(composer_address)})

    # Deploy mock targets.
    target_address = await _deploy(w3, compiled_mocks["MockTarget"], deployer)
    reverting_address = await _deploy(w3, compiled_mocks["RevertingTarget"], deployer)

    target = Contract(abi=compiled_mocks["MockTarget"]["abi"], tx={"to": Web3.to_checksum_address(target_address)})

    return {
        "w3": w3,
        "accounts": accounts,
        "deployer": deployer,
        "usdc": usdc,
        "usdc_address": usdc_address,
        "transmitter": transmitter,
        "transmitter_address": transmitter_address,
        "vm_address": vm_address,
        "composer": composer,
        "composer_address": composer_address,
        "target": target,
        "target_address": target_address,
        "reverting_address": reverting_address,
        "compiled_mocks": compiled_mocks,
    }


# ---------------------------------------------------------------------------
# Helper: build CCTP v2 message + attestation
# ---------------------------------------------------------------------------


def _make_message_and_attestation(
    composer_address: Address,
    amount: int,
    hook_data: Hash = b"",
    nonce: int = 1,
    fee_executed: int = 0,
) -> tuple[bytes, bytes]:
    """Return (message_bytes, attestation_bytes) for a CCTP v2 compose call."""
    message = make_cctp_v2_message(
        source_domain=_ETHEREUM_DOMAIN,
        nonce=nonce.to_bytes(32, "big"),
        amount=amount,
        mint_recipient=composer_address,
        hook_data=hook_data,
        fee_executed=fee_executed,
    )
    attestation = b"\x00" * 65  # mock: MessageTransmitterV2 ignores attestation content
    return message, attestation


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestCCTPComposerBasic:
    """Basic receiveAndExecute flow tests."""

    async def test_receive_and_execute_basic_call(self, ctx):
        """receiveAndExecute mints USDC and executes the program from hookData.

        Composer DELEGATECALLs the interpreter directly, so the program runs
        in the composer's context.  The minted USDC stays at the composer
        unless the program explicitly transfers it elsewhere — this program
        only calls MockTarget.execute() and does not move USDC.
        """
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]
        usdc = ctx["usdc"]

        amount = 1000 * 10**6  # 1000 USDC

        target_calldata = target.fns.execute(b"\xde\xad\xbe\xef").data
        program = _compose_program(target_address, target_calldata)

        message, attestation = _make_message_and_attestation(
            composer_address, amount, hook_data=HexBytes(program), nonce=1
        )

        pre_call_count = await target.fns.callCount().call(w3)
        pre_composer = await usdc.fns.balanceOf(composer_address).call(w3)

        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        # Verify target was called.
        assert await target.fns.callCount().call(w3) == pre_call_count + 1

        # USDC stays at the composer (program did not transfer it).
        assert await usdc.fns.balanceOf(composer_address).call(w3) == pre_composer + amount

    async def test_fee_deduction(self, ctx):
        """amountReceived = amount - feeExecuted; staged in transient slot 0."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]
        usdc = ctx["usdc"]

        amount = 1000 * 10**6  # 1000 USDC
        fee = 1 * 10**6  # 1 USDC relayer fee
        expected_received = amount - fee

        target_calldata = target.fns.execute(b"fee_test").data
        program = _compose_program(target_address, target_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=2, fee_executed=fee
        )

        pre_composer = await usdc.fns.balanceOf(composer_address).call(w3)
        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        # Composer received only amountReceived = amount - fee.
        assert await usdc.fns.balanceOf(composer_address).call(w3) == pre_composer + expected_received

    async def test_receive_and_execute_with_eth_value(self, ctx):
        """receiveAndExecute forwards ETH to the DeFiVM sub-call."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]

        amount = 100 * 10**6  # 100 USDC
        eth_value = 10**16  # 0.01 ETH

        target_calldata = target.fns.execute(b"with eth").data
        program = _compose_program(target_address, target_calldata, value=eth_value)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=3
        )

        pre_target_bal = await w3.eth.get_balance(target_address)
        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer, value=eth_value)
        receipt = tx
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(target_address) == pre_target_bal + eth_value
        assert await target.fns.lastValue().call(w3) == eth_value

    async def test_receive_and_execute_zero_amount(self, ctx):
        """receiveAndExecute with amount=0 skips the token transfer but still runs the program."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]

        amount = 0
        target_calldata = target.fns.execute(b"zero").data
        program = _compose_program(target_address, target_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=4
        )

        pre_count = await target.fns.callCount().call(w3)
        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1
        assert await target.fns.callCount().call(w3) == pre_count + 1

    async def test_emits_composed_event(self, ctx):
        """receiveAndExecute emits Composed(sourceDomain, nonce, amountReceived)."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]

        amount = 250 * 10**6
        fee = 500_000  # 0.5 USDC
        expected_received = amount - fee
        nonce_int = 5
        nonce_bytes32 = nonce_int.to_bytes(32, "big")

        target_calldata = target.fns.execute(b"event").data
        program = _compose_program(target_address, target_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=nonce_int, fee_executed=fee
        )

        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        events = composer.events.Composed.parse_logs(receipt["logs"])
        assert len(events) == 1
        evt = events[0]
        assert evt["args"]["sourceDomain"] == _ETHEREUM_DOMAIN
        assert bytes(evt["args"]["nonce"]) == nonce_bytes32
        assert evt["args"]["amountReceived"] == expected_received

    async def test_usdc_held_by_composer_then_spent(self, ctx):
        """Program runs in composer context, transfers minted USDC to recipient."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        usdc = ctx["usdc"]
        usdc_address = ctx["usdc_address"]

        amount = 300 * 10**6
        fresh_recipient = w3.eth.account.create().address

        pre_composer = await usdc.fns.balanceOf(composer_address).call(w3)

        transfer_calldata = usdc.fns.transfer(fresh_recipient, amount).data
        program = _compose_program(usdc_address, transfer_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=6
        )

        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        # Recipient received amount; composer net delta is zero (received then sent).
        assert await usdc.fns.balanceOf(fresh_recipient).call(w3) == amount
        assert await usdc.fns.balanceOf(composer_address).call(w3) == pre_composer

    async def test_v3_swap_callback_routed_through_composer(self, ctx):
        """Regression: program runs in composer context and triggers a V3 swap;
        the pool callbacks back into the composer, whose inherited
        ``DEXCallbackRouter.fallback`` must dispatch ``uniswapV3SwapCallback``
        and pay ``amountIn`` of ``tokenIn`` to the pool.
        """
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        usdc = ctx["usdc"]
        usdc_address = ctx["usdc_address"]

        token1_address = await _deploy(w3, ctx["compiled_mocks"]["MockUSDC"], deployer)
        pool_address, pool_abi = await deploy_mock_v3_pool(w3, deployer, usdc_address, token1_address)
        pool = Contract(abi=pool_abi, tx={"to": Web3.to_checksum_address(pool_address)})

        from eth_abi.abi import encode as abi_encode

        amount_in = 100 * 10**6  # USDC
        callback_data = abi_encode(["address"], [usdc_address])
        swap_calldata = pool.fns.swap(composer_address, True, amount_in, 0, callback_data).data
        program = _compose_program(pool_address, swap_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount_in, hook_data=HexBytes(program), nonce=7
        )

        pre_composer_usdc = await usdc.fns.balanceOf(composer_address).call(w3)
        pre_pool_usdc = await usdc.fns.balanceOf(pool_address).call(w3)

        tx = await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)
        assert tx["status"] == 1

        # Composer received `amount_in` (CCTP mint) and paid it back via the
        # V3 callback; pool received exactly `amount_in` of USDC.
        assert await usdc.fns.balanceOf(composer_address).call(w3) == pre_composer_usdc
        assert await usdc.fns.balanceOf(pool_address).call(w3) == pre_pool_usdc + amount_in


@pytest.mark.fork
class TestCCTPComposerErrors:
    """Error handling tests."""

    async def test_revert_when_receive_message_fails(self, ctx):
        """receiveAndExecute reverts when the MessageTransmitterV2 rejects the attestation."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        transmitter = ctx["transmitter"]
        deployer = ctx["deployer"]
        target = ctx["target"]
        target_address = ctx["target_address"]

        await transmitter.fns.setFail(True).transact(w3, deployer)

        amount = 100 * 10**6
        target_calldata = target.fns.execute(b"fail").data
        program = _compose_program(target_address, target_calldata)

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=50
        )

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)

        await transmitter.fns.setFail(False).transact(w3, deployer)

    async def test_revert_when_message_too_short(self, ctx):
        """receiveAndExecute reverts when the CCTP v2 message is shorter than 376 bytes."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]

        short_message = b"\x00" * 200  # less than 376 bytes minimum
        attestation = b"\x00" * 65

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.receiveAndExecute(short_message, attestation).transact(w3, deployer)

    async def test_revert_when_sub_call_fails(self, ctx):
        """receiveAndExecute reverts when a DeFiVM sub-call inside the program fails."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        reverting_address = ctx["reverting_address"]

        amount = 50 * 10**6

        # Call a reverting target — the outer receiveAndExecute should revert.
        program = _compose_program(reverting_address, b"\xde\xad")

        message, attestation = _make_message_and_attestation(
            HexBytes(composer_address), amount, hook_data=HexBytes(program), nonce=51
        )

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.receiveAndExecute(message, attestation).transact(w3, deployer)


@pytest.mark.fork
class TestCCTPComposerAdmin:
    """Ownership and rescue tests."""

    async def test_owner_can_rescue_eth(self, ctx):
        """Owner can rescue ETH stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]

        eth_amount = 5 * 10**15  # 0.005 ETH
        tx_hash = await w3.eth.send_transaction({"from": deployer, "to": composer_address, "value": eth_amount})
        await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=0.1)
        before = await w3.eth.get_balance(composer_address)
        assert before >= eth_amount

        fresh_recipient = w3.eth.account.create().address
        tx = await composer.fns.rescueETH(fresh_recipient, eth_amount).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        assert await w3.eth.get_balance(composer_address) == before - eth_amount
        assert await w3.eth.get_balance(fresh_recipient) == eth_amount

    async def test_non_owner_cannot_rescue_eth(self, ctx):
        """rescueETH reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.rescueETH(non_owner, 1).transact(w3, non_owner)

    async def test_owner_can_rescue_token(self, ctx):
        """Owner can rescue ERC-20 tokens stuck in the composer contract."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        usdc = ctx["usdc"]
        usdc_address = ctx["usdc_address"]

        token_amount = 99 * 10**6
        fresh_recipient = w3.eth.account.create().address

        await usdc.fns.mint(composer_address, token_amount).transact(w3, deployer)
        before_composer = await usdc.fns.balanceOf(composer_address).call(w3)

        tx = await composer.fns.rescueToken(usdc_address, fresh_recipient, token_amount).transact(w3, deployer)
        receipt = tx
        assert receipt["status"] == 1

        assert await usdc.fns.balanceOf(composer_address).call(w3) == before_composer - token_amount
        assert await usdc.fns.balanceOf(fresh_recipient).call(w3) == token_amount

    async def test_non_owner_cannot_rescue_token(self, ctx):
        """rescueToken reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        usdc_address = ctx["usdc_address"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.rescueToken(usdc_address, non_owner, 1).transact(w3, non_owner)

    async def test_transfer_ownership(self, ctx):
        """Owner can transfer ownership to a new address."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        deployer = ctx["deployer"]
        accounts = ctx["accounts"]

        new_owner = accounts[1]
        receipt = await composer.fns.transferOwnership(new_owner).transact(w3, deployer)
        assert receipt["status"] == 1
        owner_after = Web3.to_checksum_address(await composer.fns.owner().call(w3))
        await composer.fns.transferOwnership(deployer).transact(w3, new_owner)

        assert owner_after == new_owner
        assert Web3.to_checksum_address(await composer.fns.owner().call(w3)) == deployer

    async def test_non_owner_cannot_transfer_ownership(self, ctx):
        """transferOwnership reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.fns.transferOwnership(non_owner).transact(w3, non_owner)

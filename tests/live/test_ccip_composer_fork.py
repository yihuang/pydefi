"""Fork tests for CCIPComposer — Chainlink CCIP compose receiver backed by DeFiVM.

These tests compile CCIPComposer.sol and DeFiVM.sol with py-solc-x, deploy them
alongside mock contracts on a local Anvil fork of Ethereum mainnet, and exercise
the full ``ccipReceive`` flow including:

 - Basic compose execution via a mock CCIP Router
 - Multi-call compose execution (two CALL instructions in one program)
 - Compose execution carrying ETH value to a sub-call
 - The prologue pushes (amountReceived, sourceChainSelector) onto the stack
 - Composed event payload
 - Revert when the caller is not the authorised Router
 - Revert when ``destTokenAmounts`` has an unexpected length
 - Revert when a sub-call inside the compose fails
 - Owner rescue of stuck ETH and ERC-20 tokens

Run with::

    pytest -m fork tests/live/test_ccip_composer_fork.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from hexbytes import HexBytes
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3, Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.types import Address
from pydefi.vm import Program
from tests.live.sol_utils import compile_sol_file, deploy, ensure_solc

# Reusable "this should revert" matcher for both anvil-direct and forked reverts.
_REVERT = (ContractLogicError, Web3RPCError)
_ZERO_ADDRESS = "0x" + "00" * 20
_DEFAULT_MESSAGE_ID = b"\x00" * 32

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "CCIPComposer.sol"

# Spot-check a CCIP chain selector (Ethereum mainnet).
_ETHEREUM_SELECTOR = 5009297550715157269


# ---------------------------------------------------------------------------
# Mock contracts Solidity source
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

struct EVMTokenAmount {
    address token;
    uint256 amount;
}

struct Any2EVMMessage {
    bytes32 messageId;
    uint64 sourceChainSelector;
    bytes sender;
    bytes data;
    EVMTokenAmount[] destTokenAmounts;
}

interface ICCIPComposer {
    function ccipReceive(Any2EVMMessage calldata message) external payable;
}

/// @notice Minimal mock CCIP Router.  Controls which address may call
///         ``ccipReceive`` on the composer and lets tests assemble inbound
///         ``Any2EVMMessage`` structs from individual fields.
contract MockRouter {
    /// @notice Deliver a CCIP compose message to a composer contract.
    ///
    /// In real CCIP, the destination Router transfers the bridged token to the
    /// receiver and then invokes ``ccipReceive``.  The mock router does only
    /// the second step — the test mints the token to the composer directly.
    function deliverCompose(
        address _composer,
        bytes32 _messageId,
        uint64 _sourceChainSelector,
        bytes calldata _sender,
        bytes calldata _data,
        address _token,
        uint256 _amount
    ) external payable {
        EVMTokenAmount[] memory tokens = new EVMTokenAmount[](1);
        tokens[0] = EVMTokenAmount({token: _token, amount: _amount});

        Any2EVMMessage memory message = Any2EVMMessage({
            messageId: _messageId,
            sourceChainSelector: _sourceChainSelector,
            sender: _sender,
            data: _data,
            destTokenAmounts: tokens
        });

        ICCIPComposer(_composer).ccipReceive{value: msg.value}(message);
    }

    /// @notice Deliver a CCIP message with an arbitrary ``destTokenAmounts``
    ///         length so tests can exercise the ``UnexpectedTokenCount`` path.
    function deliverComposeManyTokens(
        address _composer,
        bytes32 _messageId,
        uint64 _sourceChainSelector,
        bytes calldata _sender,
        bytes calldata _data,
        EVMTokenAmount[] calldata _tokens
    ) external payable {
        Any2EVMMessage memory message = Any2EVMMessage({
            messageId: _messageId,
            sourceChainSelector: _sourceChainSelector,
            sender: _sender,
            data: _data,
            destTokenAmounts: _tokens
        });

        ICCIPComposer(_composer).ccipReceive{value: msg.value}(message);
    }
}

/// @notice Minimal mintable ERC-20 (matches MockToken in sol_utils).
contract MockERC20 {
    string public name = "Mock";
    string public symbol = "MOCK";
    uint8 public decimals = 18;

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


def _compile_mock_contracts() -> dict[str, dict]:
    ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_CONTRACTS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return {
        "MockRouter": result["<stdin>:MockRouter"],
        "MockERC20": result["<stdin>:MockERC20"],
        "MockTarget": result["<stdin>:MockTarget"],
        "RevertingTarget": result["<stdin>:RevertingTarget"],
    }


def _compile_ccip_composer() -> dict:
    return compile_sol_file(SOL_FILE, "CCIPComposer")


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: Address, *args) -> Address:
    return await deploy(w3, compiled, deployer, *args)


# ---------------------------------------------------------------------------
# Module-scoped fork + deploy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def ccip_fork_w3(fork_w3_module):
    return fork_w3_module


@pytest.fixture(scope="module")
def compiled_ccip_composer():
    return _compile_ccip_composer()


@pytest.fixture(scope="module")
def compiled_mocks():
    return _compile_mock_contracts()


@pytest.fixture(scope="module")
async def ctx(ccip_fork_w3, compiled_ccip_composer, compiled_mocks, interpreter_addr):
    """Deploy CCIPComposer and mock contracts once; share across tests."""
    w3 = ccip_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    router_address = await _deploy(w3, compiled_mocks["MockRouter"], deployer)
    composer_address = await _deploy(
        w3,
        compiled_ccip_composer,
        deployer,
        router_address,  # _router
        interpreter_addr,  # _interpreter
        deployer,  # _owner
        False,  # _allowlistEnabled — module-scoped fixture keeps the open
        #                                  default so existing tests can drive
        #                                  arbitrary senders.  A dedicated test
        #                                  deploys a separate composer with the
        #                                  allowlist enabled at construction.
    )

    token_address = await _deploy(w3, compiled_mocks["MockERC20"], deployer)
    target_address = await _deploy(w3, compiled_mocks["MockTarget"], deployer)
    reverting_address = await _deploy(w3, compiled_mocks["RevertingTarget"], deployer)

    composer = Contract(abi=compiled_ccip_composer["abi"], tx={"to": Web3.to_checksum_address(composer_address)})
    router = Contract(abi=compiled_mocks["MockRouter"]["abi"], tx={"to": Web3.to_checksum_address(router_address)})
    token = Contract(abi=compiled_mocks["MockERC20"]["abi"], tx={"to": Web3.to_checksum_address(token_address)})
    target = Contract(abi=compiled_mocks["MockTarget"]["abi"], tx={"to": Web3.to_checksum_address(target_address)})

    return {
        "w3": w3,
        "accounts": accounts,
        "deployer": deployer,
        "composer": composer,
        "composer_address": composer_address,
        "router": router,
        "router_address": router_address,
        "interpreter_addr": interpreter_addr,
        "token": token,
        "token_address": token_address,
        "target": target,
        "target_address": target_address,
        "reverting_address": reverting_address,
    }


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _start_program() -> tuple[Program, "object", "object"]:
    """Start a compose program and read the two transient-storage params.

    The CCIPComposer stages bridged parameters in transient storage:
    slot 0 = amountReceived, slot 1 = sourceChainSelector.  The program
    reads them via TLOAD.

    Returns ``(prog, amount_received, source_selector)``.
    """
    prog = Program()
    amount_received = prog.builder.tload(IRLiteral(0))
    source_selector = prog.builder.tload(IRLiteral(1))
    return prog, amount_received, source_selector


def _compose_noop() -> bytes:
    """Minimal compose program: read prologue values, do nothing."""
    prog, _amount, _selector = _start_program()
    prog.builder.stop()
    return prog.build()


def _compose_single_call(target_address: Address, calldata: bytes, *, value: int = 0) -> bytes:
    """Build a compose program that issues a single external call.

    On CALL failure the outer ``ccipReceive`` reverts (matches legacy
    ``call(require_success=True)``).
    """
    prog, _amount, _selector = _start_program()
    success = prog.call_raw(target_address, calldata, value=value)
    prog.assert_(success)
    prog.builder.stop()
    return prog.build()


def _compose_multi_call(calls: list[tuple[Address, bytes]]) -> bytes:
    """Build a compose program that issues N external calls in sequence."""
    prog, _amount, _selector = _start_program()
    for target, data in calls:
        success = prog.call_raw(target, data)
        prog.assert_(success)
    prog.builder.stop()
    return prog.build()


async def _deliver(
    ctx: dict,
    program: bytes,
    *,
    amount: int = 10**18,
    sender: bytes = b"",
    message_id: bytes = _DEFAULT_MESSAGE_ID,
    value: int = 0,
    composer_address: Address | None = None,
    mint: bool = True,
):
    """Mint *amount* tokens to the composer (when ``mint=True``) and deliver a
    compose message through the mock Router.

    Returns the transaction receipt.  Pass ``composer_address`` to target a
    composer other than the one in the shared fixture (used by the
    deploy-time-hardened test).
    """
    w3 = ctx["w3"]
    deployer = ctx["deployer"]
    composer_addr = composer_address or ctx["composer_address"]
    if mint and amount > 0:
        await ctx["token"].fns.mint(composer_addr, amount).transact(w3, deployer)
    transact_kwargs = {"value": value} if value else {}
    return (
        await ctx["router"]
        .fns.deliverCompose(
            composer_addr,
            message_id,
            _ETHEREUM_SELECTOR,
            sender,
            HexBytes(program),
            ctx["token_address"],
            amount,
        )
        .transact(w3, deployer, **transact_kwargs)
    )


@asynccontextmanager
async def _allowlist_on(ctx: dict, *, register: tuple[int, bytes] | None = None):
    """Enable the source-side allowlist for the duration of the block.

    When *register* is ``(selector, sender)``, also add that pair before
    yielding.  Both flags / entries are reset on exit so the module-scoped
    fixture stays clean for the next test.
    """
    composer = ctx["composer"]
    w3 = ctx["w3"]
    deployer = ctx["deployer"]
    await composer.fns.setAllowlistEnabled(True).transact(w3, deployer)
    if register is not None:
        await composer.fns.setAllowed(*register, True).transact(w3, deployer)
    try:
        yield
    finally:
        if register is not None:
            await composer.fns.setAllowed(*register, False).transact(w3, deployer)
        await composer.fns.setAllowlistEnabled(False).transact(w3, deployer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestCCIPComposerFork:
    """Fork-level tests for CCIPComposer.sol backed by DeFiVM on a local Anvil fork."""

    # ------------------------------------------------------------------
    # Happy-path compose flow
    # ------------------------------------------------------------------

    async def test_single_call_compose(self, ctx):
        """ccipReceive runs a DeFiVM program that calls MockTarget.execute()."""
        target = ctx["target"]
        amount = 5 * 10**18
        program = _compose_single_call(ctx["target_address"], target.fns.execute(b"ccip-hello").data)

        pre_count = await target.fns.callCount().call(ctx["w3"])
        pre_composer_bal = await ctx["token"].fns.balanceOf(ctx["composer_address"]).call(ctx["w3"])

        receipt = await _deliver(ctx, program, amount=amount)
        assert receipt["status"] == 1

        assert await target.fns.callCount().call(ctx["w3"]) == pre_count + 1
        assert await target.fns.lastData().call(ctx["w3"]) == b"ccip-hello"

        # Program runs in the composer's own context via DELEGATECALL, so the
        # router-delivered tokens stay on the composer until the program moves
        # them out.  This compose program is a no-op transfer, so the balance
        # increment exactly matches the bridged amount.
        assert await ctx["token"].fns.balanceOf(ctx["composer_address"]).call(ctx["w3"]) == pre_composer_bal + amount

    async def test_multi_call_compose(self, ctx):
        """A program with two sequential CALL instructions increments callCount by 2."""
        target = ctx["target"]
        target_address = ctx["target_address"]
        program = _compose_multi_call(
            [
                (target_address, target.fns.execute(b"call_a").data),
                (target_address, target.fns.execute(b"call_b").data),
            ]
        )

        before = await target.fns.callCount().call(ctx["w3"])
        receipt = await _deliver(ctx, program, amount=2 * 10**18, message_id=b"\x00" * 31 + b"\x01")
        assert receipt["status"] == 1
        assert await target.fns.callCount().call(ctx["w3"]) == before + 2

    async def test_compose_with_eth_value(self, ctx):
        """ETH attached to ccipReceive is forwarded to the DeFiVM sub-call."""
        target = ctx["target"]
        target_address = ctx["target_address"]
        eth_value = 5 * 10**15  # 0.005 ETH
        program = _compose_single_call(target_address, target.fns.execute(b"with-eth").data, value=eth_value)

        pre_target = await ctx["w3"].eth.get_balance(target_address)
        receipt = await _deliver(ctx, program, value=eth_value)
        assert receipt["status"] == 1

        assert await ctx["w3"].eth.get_balance(target_address) == pre_target + eth_value
        assert await target.fns.lastValue().call(ctx["w3"]) == eth_value

    async def test_zero_amount_runs_program(self, ctx):
        """ccipReceive with amount=0 still runs the program."""
        target = ctx["target"]
        program = _compose_single_call(ctx["target_address"], target.fns.execute(b"zero").data)

        before = await target.fns.callCount().call(ctx["w3"])
        receipt = await _deliver(ctx, program, amount=0, mint=False)
        assert receipt["status"] == 1
        assert await target.fns.callCount().call(ctx["w3"]) == before + 1

    async def test_composed_event(self, ctx):
        """Composed(sourceChainSelector, messageId, token, amountReceived) is emitted."""
        amount = 333 * 10**18
        message_id = b"\xab" * 32
        program = _compose_single_call(ctx["target_address"], ctx["target"].fns.execute(b"event").data)

        receipt = await _deliver(ctx, program, amount=amount, message_id=message_id)
        assert receipt["status"] == 1

        events = ctx["composer"].events.Composed.parse_logs(receipt["logs"])
        assert len(events) == 1
        evt = events[0]["args"]
        assert evt["sourceChainSelector"] == _ETHEREUM_SELECTOR
        assert HexBytes(evt["messageId"]) == HexBytes(message_id)
        assert HexBytes(evt["token"]) == HexBytes(ctx["token_address"])
        assert evt["amountReceived"] == amount

    # ------------------------------------------------------------------
    # Negative paths: caller + payload validation, sub-call rollback
    # ------------------------------------------------------------------

    async def test_unauthorized_router_reverts(self, ctx):
        """Direct call to ccipReceive from a non-router address reverts."""
        msg = (
            _DEFAULT_MESSAGE_ID,
            _ETHEREUM_SELECTOR,
            b"",  # sender
            HexBytes(_compose_noop()),
            ((ctx["token_address"], 0),),
        )
        with pytest.raises(_REVERT):
            await ctx["composer"].fns.ccipReceive(msg).transact(ctx["w3"], ctx["deployer"])

    @pytest.mark.parametrize("tokens", [(), "many"], ids=["zero", "many"])
    async def test_unexpected_token_count_reverts(self, ctx, tokens):
        """ccipReceive reverts when destTokenAmounts is empty or has >1 entry."""
        if tokens == "many":
            tokens = ((ctx["token_address"], 1), (ctx["token_address"], 2))
        with pytest.raises(_REVERT):
            await (
                ctx["router"]
                .fns.deliverComposeManyTokens(
                    ctx["composer_address"],
                    _DEFAULT_MESSAGE_ID,
                    _ETHEREUM_SELECTOR,
                    b"",
                    HexBytes(_compose_noop()),
                    tokens,
                )
                .transact(ctx["w3"], ctx["deployer"])
            )

    async def test_sub_call_failure_reverts(self, ctx):
        """A failing CALL in the program reverts the entire compose transaction."""
        target = ctx["target"]
        target_address = ctx["target_address"]
        reverting_address = ctx["reverting_address"]

        # First sub-call succeeds; second goes to a reverting target.  The
        # second ``call_raw`` + ``assert_`` enforces require-success semantics,
        # bubbling the revert up through ``ccipReceive``.
        program = _compose_multi_call(
            [
                (target_address, target.fns.execute(b"before-fail").data),
                (reverting_address, b""),
            ]
        )

        before = await target.fns.callCount().call(ctx["w3"])
        with pytest.raises(_REVERT):
            await _deliver(ctx, program)
        # The first CALL's increment must have been rolled back.
        assert await target.fns.callCount().call(ctx["w3"]) == before

    # ------------------------------------------------------------------
    # Owner rescue + ownership transfer
    # ------------------------------------------------------------------

    async def test_rescue_eth(self, ctx):
        """Owner can rescue ETH stuck in the composer.

        Mirrors the proven idiom from :mod:`tests.live.test_oft_composer_fork`:
        fund the composer via a normal ``send_transaction`` (rather than
        ``anvil_setBalance``, whose behaviour varies between anvil builds and
        was producing a silent no-op rescue here), send the rescue to a fresh
        ephemeral recipient that no prior test has touched, and assert both
        the composer and recipient balances move by exactly ``eth_amount``.
        """
        w3 = ctx["w3"]
        composer_address = ctx["composer_address"]
        deployer = ctx["deployer"]
        eth_amount = 5 * 10**17  # 0.5 ETH

        tx_hash = await w3.eth.send_transaction(
            {"from": deployer, "to": Web3.to_checksum_address(composer_address), "value": eth_amount}
        )
        await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=0.1)

        before_composer = await w3.eth.get_balance(composer_address)
        assert before_composer >= eth_amount

        fresh_recipient = w3.eth.account.create().address
        assert await w3.eth.get_balance(fresh_recipient) == 0

        receipt = await ctx["composer"].fns.rescueETH(fresh_recipient, eth_amount).transact(w3, deployer)
        assert receipt["status"] == 1
        assert await w3.eth.get_balance(composer_address) == before_composer - eth_amount
        assert await w3.eth.get_balance(fresh_recipient) == eth_amount

    async def test_rescue_token(self, ctx):
        """Owner can rescue ERC-20 tokens stuck in the composer."""
        token = ctx["token"]
        amount = 7 * 10**18

        await token.fns.mint(ctx["composer_address"], amount).transact(ctx["w3"], ctx["deployer"])

        # Use a fresh ephemeral recipient so any tokens left over from earlier
        # tests in this module's ctx cannot influence the post-balance check.
        fresh_recipient = ctx["w3"].eth.account.create().address
        assert await token.fns.balanceOf(fresh_recipient).call(ctx["w3"]) == 0

        receipt = (
            await ctx["composer"]
            .fns.rescueToken(ctx["token_address"], fresh_recipient, amount)
            .transact(ctx["w3"], ctx["deployer"])
        )
        assert receipt["status"] == 1
        assert await token.fns.balanceOf(fresh_recipient).call(ctx["w3"]) == amount

    @pytest.mark.parametrize(
        "fn_name,args_idx",
        [
            ("rescueETH", "non_owner_then_1"),
            ("setAllowlistEnabled", "true"),
            ("setAllowed", "default_entry"),
        ],
    )
    async def test_only_owner(self, ctx, fn_name, args_idx):
        """Owner-gated functions revert when called by anyone else."""
        non_owner = ctx["accounts"][2]
        args = {
            "non_owner_then_1": (non_owner, 1),
            "true": (True,),
            "default_entry": (_ETHEREUM_SELECTOR, b"\xcc" * 32, True),
        }[args_idx]
        with pytest.raises(_REVERT):
            await getattr(ctx["composer"].fns, fn_name)(*args).transact(ctx["w3"], non_owner)

    async def test_transfer_ownership(self, ctx):
        """transferOwnership updates owner and emits OwnershipTransferred.

        Deploys a dedicated composer rather than reusing the module-scoped
        fixture: the ownership mutation must not leak into later tests if an
        assertion below fails partway, which (combined with ``--reruns``) would
        otherwise cascade ``CCIPComposer: not owner`` into every owner-gated
        test that follows.
        """
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        new_owner = ctx["accounts"][3]

        compiled = _compile_ccip_composer()
        composer_address = await deploy(
            w3,
            compiled,
            deployer,
            Web3.to_checksum_address(ctx["router_address"]),
            Web3.to_checksum_address(ctx["interpreter_addr"]),
            deployer,  # _owner
            False,  # _allowlistEnabled
        )
        composer = Contract(abi=compiled["abi"], tx={"to": Web3.to_checksum_address(composer_address)})

        old_owner = await composer.fns.owner().call(w3)
        receipt = await composer.fns.transferOwnership(new_owner).transact(w3, deployer)
        assert receipt["status"] == 1
        assert HexBytes(await composer.fns.owner().call(w3)) == HexBytes(new_owner)

        events = composer.events.OwnershipTransferred.parse_logs(receipt["logs"])
        assert len(events) == 1
        assert HexBytes(events[0]["args"]["previousOwner"]) == HexBytes(old_owner)
        assert HexBytes(events[0]["args"]["newOwner"]) == HexBytes(new_owner)

    # ------------------------------------------------------------------
    # Constructor guards
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("zero_arg", ["router", "owner"])
    async def test_constructor_rejects_zero_address(self, ctx, zero_arg):
        """Deploying CCIPComposer with router/owner = 0 reverts.

        The interpreter argument intentionally accepts ``address(0)`` as a
        sentinel meaning "use the well-known Analog-Labs interpreter"
        (see :class:`InterpreterRunner`), so it is not validated here.
        """
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        compiled = _compile_ccip_composer()

        ctor_args: dict[str, str] = {
            "router": Web3.to_checksum_address(ctx["router_address"]),
            "owner": deployer,
        }
        ctor_args[zero_arg] = _ZERO_ADDRESS

        contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
        with pytest.raises(_REVERT):
            tx_hash = await contract.constructor(
                ctor_args["router"], ctx["interpreter_addr"], ctor_args["owner"], False
            ).transact({"from": deployer})
            await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30, poll_latency=0.1)

    async def test_constructor_can_enable_allowlist_at_deploy(self, ctx):
        """Deploying with ``_allowlistEnabled=true`` boots the composer in a
        fail-closed posture: an inbound message from an unregistered
        ``(selector, sender)`` reverts immediately.
        """
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        compiled = _compile_ccip_composer()
        hardened_address = await deploy(
            w3,
            compiled,
            deployer,
            Web3.to_checksum_address(ctx["router_address"]),
            Web3.to_checksum_address(ctx["interpreter_addr"]),
            deployer,
            True,  # _allowlistEnabled
        )
        hardened = Contract(abi=compiled["abi"], tx={"to": Web3.to_checksum_address(hardened_address)})

        assert await hardened.fns.allowlistEnabled().call(w3) is True

        # An inbound message with an unregistered sender must revert.
        with pytest.raises(_REVERT):
            await _deliver(
                ctx,
                _compose_noop(),
                composer_address=hardened_address,
                sender=b"\xaa" * 32,
            )

    # ------------------------------------------------------------------
    # Source-side allowlist
    # ------------------------------------------------------------------

    async def test_allowlist_disabled_by_default(self, ctx):
        """allowlistEnabled is false right after deployment."""
        assert await ctx["composer"].fns.allowlistEnabled().call(ctx["w3"]) is False

    async def test_allowlist_blocks_unregistered_sender(self, ctx):
        """When the allowlist is on, an unregistered sender cannot trigger compose."""
        async with _allowlist_on(ctx):
            with pytest.raises(_REVERT):
                await _deliver(ctx, _compose_noop(), sender=b"\xaa" * 32)

    async def test_allowlist_admits_registered_sender(self, ctx):
        """A registered (selector, sender) pair passes the allowlist check."""
        sender_bytes = b"\xbb" * 32
        target = ctx["target"]
        program = _compose_single_call(ctx["target_address"], target.fns.execute(b"allowed").data)

        async with _allowlist_on(ctx, register=(_ETHEREUM_SELECTOR, sender_bytes)):
            receipt = await _deliver(ctx, program, sender=sender_bytes)
            assert receipt["status"] == 1

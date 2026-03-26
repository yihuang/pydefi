"""Fork tests for OFTComposer — LayerZero OFT compose receiver.

These tests compile OFTComposer.sol with py-solc-x, deploy it alongside mock
contracts on a local Anvil fork of Ethereum mainnet, and exercise the full
``lzCompose`` flow including:

 - Single-call compose execution via a mock LayerZero endpoint
 - Multi-call compose execution (sequential calls)
 - Compose execution carrying ETH value to a sub-call
 - Revert when the caller is not the authorised endpoint
 - Revert when the originating OFT is not in the approved list
 - Revert when a sub-call inside the compose fails

Run with::

    pytest -m fork tests/live/test_oft_composer_fork.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from eth_abi import encode as abi_encode
from web3 import AsyncWeb3
from web3.exceptions import ContractLogicError, Web3RPCError

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_FILE = REPO_ROOT / "pydefi" / "bridge" / "OFTComposer.sol"

# ---------------------------------------------------------------------------
# Compile + deploy helpers
# ---------------------------------------------------------------------------


def _ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc once (no-op if already installed)."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def _compile_oft_composer() -> dict:
    """Compile OFTComposer.sol and return the ABI + bytecode."""
    _ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(SOL_FILE)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(":OFTComposer"))
    return result[key]


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: str, *args) -> str:
    """Deploy a contract and return its address."""
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Mock contracts Solidity source (compiled inline at test-module load time)
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mock LayerZero EndpointV2.
///         Allows tests to manually trigger lzCompose on a registered composer.
contract MockEndpoint {
    /// @notice Deliver a compose message to a composer contract.
    function deliverCompose(
        address _composer,
        address _from,
        bytes32 _guid,
        bytes calldata _message
    ) external payable {
        (bool ok, bytes memory err) = _composer.call{value: msg.value}(
            abi.encodeWithSignature(
                "lzCompose(address,bytes32,bytes,address,bytes)",
                _from,
                _guid,
                _message,
                address(this),
                bytes("")
            )
        );
        if (!ok) {
            assembly { revert(add(err, 32), mload(err)) }
        }
    }
}

/// @notice Minimal mock OFT token (ERC-20 subset with mint).
contract MockOFT {
    string public name = "Mock OFT";
    string public symbol = "MOFT";
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

/// @notice Mock target contract — records the most recent call and emits an event.
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

/// @notice Mock target that always reverts — used to test sub-call failure handling.
contract RevertingTarget {
    error AlwaysReverts();

    fallback() external payable {
        revert AlwaysReverts();
    }
}
"""


def _compile_mock_contracts() -> dict[str, dict]:
    """Compile mock contracts and return {name: {abi, bin}} mapping."""
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_CONTRACTS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return {
        "MockEndpoint": result["<stdin>:MockEndpoint"],
        "MockOFT": result["<stdin>:MockOFT"],
        "MockTarget": result["<stdin>:MockTarget"],
        "RevertingTarget": result["<stdin>:RevertingTarget"],
    }


# ---------------------------------------------------------------------------
# Compose-message helpers
# ---------------------------------------------------------------------------


def _to_bytes(hex_or_bytes: str | bytes) -> bytes:
    """Convert a hex string or bytes to raw bytes."""
    if isinstance(hex_or_bytes, bytes):
        return hex_or_bytes
    return bytes.fromhex(hex_or_bytes.removeprefix("0x"))


def make_compose_message(
    nonce: int,
    src_eid: int,
    amount_ld: int,
    calls: list[tuple],
) -> bytes:
    """Build a LayerZero OFTComposeMsgCodec-encoded message.

    Layout::

        | 8B nonce | 4B srcEid | 32B amountLD | abi.encode(Call[]) |

    Args:
        nonce:     uint64 message nonce.
        src_eid:   uint32 source endpoint ID.
        amount_ld: uint256 amount of OFT tokens delivered (in local decimals).
        calls:     List of (target, value, data) tuples.

    Returns:
        Raw bytes ready to pass as ``_message`` in ``lzCompose``.
    """
    payload = abi_encode(
        ["(address,uint256,bytes)[]"],
        [[(t, v, _to_bytes(d)) for t, v, d in calls]],
    )
    return (
        struct.pack(">Q", nonce)  # 8 bytes  — uint64 nonce
        + struct.pack(">I", src_eid)  # 4 bytes  — uint32 srcEid
        + amount_ld.to_bytes(32, "big")  # 32 bytes — uint256 amountLD
        + payload  # ABI-encoded Call[]
    )


# ---------------------------------------------------------------------------
# Module-scoped Anvil fork fixture (shared across all tests in this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def oft_fork_w3(fork_w3_module):
    """Module-scoped Anvil mainnet fork, shared across all tests in this module."""
    return fork_w3_module


# ---------------------------------------------------------------------------
# Module-scoped setup: compile + deploy once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_oft_composer():
    return _compile_oft_composer()


@pytest.fixture(scope="module")
def compiled_mocks():
    return _compile_mock_contracts()


@pytest.fixture(scope="module")
async def ctx(oft_fork_w3, compiled_oft_composer, compiled_mocks):
    """Deploy OFTComposer and mock contracts once, return shared context."""
    w3 = oft_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy mock endpoint (controls which address may call lzCompose).
    endpoint_address = await _deploy(w3, compiled_mocks["MockEndpoint"], deployer)

    # Deploy OFT composer, pointing it at the mock endpoint.
    composer_address = await _deploy(
        w3,
        compiled_oft_composer,
        deployer,
        endpoint_address,  # _endpoint
        deployer,  # _owner
    )
    composer = w3.eth.contract(address=composer_address, abi=compiled_oft_composer["abi"])
    endpoint = w3.eth.contract(address=endpoint_address, abi=compiled_mocks["MockEndpoint"]["abi"])

    # Deploy mock OFT and approve it in the composer.
    oft_address = await _deploy(w3, compiled_mocks["MockOFT"], deployer)
    await composer.functions.approveOFT(oft_address).transact({"from": deployer})

    # Deploy mock target contracts.
    target_address = await _deploy(w3, compiled_mocks["MockTarget"], deployer)
    reverting_address = await _deploy(w3, compiled_mocks["RevertingTarget"], deployer)

    target = w3.eth.contract(address=target_address, abi=compiled_mocks["MockTarget"]["abi"])

    return {
        "w3": w3,
        "accounts": accounts,
        "deployer": deployer,
        "composer": composer,
        "composer_address": composer_address,
        "endpoint": endpoint,
        "endpoint_address": endpoint_address,
        "oft_address": oft_address,
        "target": target,
        "target_address": target_address,
        "reverting_address": reverting_address,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestOFTComposerFork:
    """Fork-level tests for OFTComposer.sol on a local Anvil mainnet fork."""

    # ------------------------------------------------------------------
    # Basic single-call compose
    # ------------------------------------------------------------------

    async def test_single_call_compose(self, ctx):
        """lzCompose executes a single call to MockTarget.execute()."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        # Encode the calldata for MockTarget.execute(bytes)
        call_data = target.encode_abi("execute", [b"hello"])
        calls = [(target_address, 0, call_data)]
        message = make_compose_message(
            nonce=1,
            src_eid=30101,
            amount_ld=10**18,
            calls=calls,
        )

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Verify the mock target received exactly one call.
        call_count = await target.functions.callCount().call()
        assert call_count == 1

        last_data = await target.functions.lastData().call()
        assert last_data == b"hello"

    # ------------------------------------------------------------------
    # Multi-call compose
    # ------------------------------------------------------------------

    async def test_multi_call_compose(self, ctx):
        """lzCompose executes multiple calls in sequence."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        before = await target.functions.callCount().call()

        call_data_a = target.encode_abi("execute", [b"call_a"])
        call_data_b = target.encode_abi("execute", [b"call_b"])
        calls = [
            (target_address, 0, call_data_a),
            (target_address, 0, call_data_b),
        ]
        message = make_compose_message(
            nonce=2,
            src_eid=30101,
            amount_ld=5 * 10**17,
            calls=calls,
        )

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 31 + b"\x01",
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        after = await target.functions.callCount().call()
        # Both calls should have been executed, incrementing count by 2.
        assert after == before + 2

    # ------------------------------------------------------------------
    # Compose with ETH value forwarded to a sub-call
    # ------------------------------------------------------------------

    async def test_compose_with_eth_value(self, ctx):
        """lzCompose forwards ETH to a sub-call correctly."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        eth_amount = 10**16  # 0.01 ETH

        # Fund the composer so it can forward ETH.
        await w3.eth.send_transaction({"from": deployer, "to": composer.address, "value": eth_amount})

        before_balance = await w3.eth.get_balance(target_address)

        call_data = target.encode_abi("execute", [b"with_eth"])
        calls = [(target_address, eth_amount, call_data)]
        message = make_compose_message(
            nonce=3,
            src_eid=30101,
            amount_ld=10**18,
            calls=calls,
        )

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            b"\x00" * 31 + b"\x02",
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        after_balance = await w3.eth.get_balance(target_address)
        assert after_balance == before_balance + eth_amount

        last_value = await target.functions.lastValue().call()
        assert last_value == eth_amount

    # ------------------------------------------------------------------
    # Composed event is emitted
    # ------------------------------------------------------------------

    async def test_composed_event_emitted(self, ctx):
        """lzCompose emits the Composed event with correct arguments."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        amount_ld = 777 * 10**18
        guid = b"\xde\xad" + b"\x00" * 30
        call_data = target.encode_abi("execute", [b"event_test"])
        calls = [(target_address, 0, call_data)]
        message = make_compose_message(nonce=4, src_eid=30184, amount_ld=amount_ld, calls=calls)

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            oft_address,
            guid,
            message,
        ).transact({"from": deployer})

        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Parse the Composed event from the receipt.
        events = composer.events.Composed().process_receipt(receipt)
        assert len(events) == 1
        evt = events[0]["args"]
        assert evt["from"] == oft_address
        assert evt["guid"] == guid
        assert evt["amountLD"] == amount_ld
        assert evt["numCalls"] == 1

    # ------------------------------------------------------------------
    # Security: unauthorized endpoint
    # ------------------------------------------------------------------

    async def test_unauthorized_endpoint_reverts(self, ctx):
        """lzCompose reverts when called by an address that is not the endpoint."""
        composer = ctx["composer"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        call_data = target.encode_abi("execute", [b"unauthorized"])
        calls = [(target_address, 0, call_data)]
        message = make_compose_message(nonce=5, src_eid=30101, amount_ld=10**18, calls=calls)

        # Call lzCompose directly from the deployer (not the endpoint).
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.functions.lzCompose(
                oft_address,
                b"\x00" * 32,
                message,
                deployer,
                b"",
            ).transact({"from": deployer})

    # ------------------------------------------------------------------
    # Security: unapproved OFT
    # ------------------------------------------------------------------

    async def test_unapproved_oft_reverts(self, ctx):
        """lzCompose reverts when the originating OFT is not approved."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        # Use a random address that has never been approved.
        unapproved_oft = w3.eth.account.create().address

        call_data = target.encode_abi("execute", [b"unapproved"])
        calls = [(target_address, 0, call_data)]
        message = make_compose_message(nonce=6, src_eid=30101, amount_ld=10**18, calls=calls)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await endpoint.functions.deliverCompose(
                composer.address,
                unapproved_oft,
                b"\x00" * 32,
                message,
            ).transact({"from": deployer})

    # ------------------------------------------------------------------
    # Security: sub-call failure propagates
    # ------------------------------------------------------------------

    async def test_sub_call_failure_reverts_compose(self, ctx):
        """lzCompose reverts when a sub-call fails, rolling back all state changes."""
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        oft_address = ctx["oft_address"]
        target_address = ctx["target_address"]
        reverting_address = ctx["reverting_address"]
        target = ctx["target"]

        before_count = await target.functions.callCount().call()

        # First call succeeds; second call always reverts.
        call_data = target.encode_abi("execute", [b"before_fail"])
        calls = [
            (target_address, 0, call_data),
            (reverting_address, 0, b""),  # always reverts
        ]
        message = make_compose_message(nonce=7, src_eid=30101, amount_ld=10**18, calls=calls)

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await endpoint.functions.deliverCompose(
                composer.address,
                oft_address,
                b"\x00" * 32,
                message,
            ).transact({"from": deployer})

        # State change from the first call must be rolled back.
        after_count = await target.functions.callCount().call()
        assert after_count == before_count

    # ------------------------------------------------------------------
    # Admin: approve / revoke OFT
    # ------------------------------------------------------------------

    async def test_approve_and_revoke_oft(self, ctx):
        """approveOFT and revokeOFT correctly update the approved list."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        endpoint = ctx["endpoint"]
        deployer = ctx["deployer"]
        target_address = ctx["target_address"]
        target = ctx["target"]

        new_oft = w3.eth.account.create().address

        # Approve the new OFT.
        tx = await composer.functions.approveOFT(new_oft).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1
        assert await composer.functions.approvedOFTs(new_oft).call()

        # Compose should now succeed using the newly approved OFT.
        call_data = target.encode_abi("execute", [b"new_oft"])
        calls = [(target_address, 0, call_data)]
        message = make_compose_message(nonce=8, src_eid=30101, amount_ld=10**18, calls=calls)

        tx = await endpoint.functions.deliverCompose(
            composer.address,
            new_oft,
            b"\x00" * 32,
            message,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        # Revoke the OFT.
        await composer.functions.revokeOFT(new_oft).transact({"from": deployer})
        assert not await composer.functions.approvedOFTs(new_oft).call()

        # Compose should now revert.
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await endpoint.functions.deliverCompose(
                composer.address,
                new_oft,
                b"\x00" * 32,
                message,
            ).transact({"from": deployer})

    # ------------------------------------------------------------------
    # Admin: only owner may approve OFT
    # ------------------------------------------------------------------

    async def test_non_owner_cannot_approve_oft(self, ctx):
        """approveOFT reverts when called by a non-owner."""
        w3 = ctx["w3"]
        composer = ctx["composer"]
        accounts = ctx["accounts"]

        non_owner = accounts[1]
        random_oft = w3.eth.account.create().address

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await composer.functions.approveOFT(random_oft).transact({"from": non_owner})

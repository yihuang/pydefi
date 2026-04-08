"""
Transaction simulation utilities.

Provides utilities to simulate EVM transactions before broadcasting, using
three different RPC methods in decreasing order of capability:

1. :func:`simulate_with_eth_simulate_v1` — multi-message support, native ETH
   balance tracking via synthetic Transfer logs.
2. :func:`simulate_with_debug_trace_call` — full call-tree trace, logs, gas;
   requires an RPC that supports ``debug_traceCall``.
3. :func:`simulate_with_eth_call` — widest RPC compatibility; return data and
   success/failure status only (no logs).

ERC-20 Approval Simulation
--------------------------

Often we need to simulate a swap before the on-chain approval has been sent.
Two helper functions cover this:

* :func:`build_allowance_state_override` — produces an ``eth_call`` /
  ``debug_traceCall`` state-override dict that injects the required allowance
  into the token's storage without an on-chain transaction.  Uses
  :func:`detect_allowance_slot` internally (requires ``debug_traceCall``).
* With :func:`simulate_with_eth_simulate_v1` you can instead prepend the
  ``approve`` call as an additional message inside the same block call.

Quick-start example::

    from pydefi.simulate import simulate_tx

    result = await simulate_tx(
        w3,
        {"from": "0x…", "to": "0x…", "data": "0x…", "value": 0},
    )
    print(result.success, result.transfers, result.balance_changes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3
from web3.exceptions import ContractLogicError, Web3RPCError
from web3.types import BlockIdentifier

__all__ = [
    "BalanceChange",
    "SimulationResult",
    "TokenTransfer",
    "build_allowance_state_override",
    "detect_allowance_slot",
    "simulate_tx",
    "simulate_with_debug_trace_call",
    "simulate_with_eth_call",
    "simulate_with_eth_simulate_v1",
]

logger = logging.getLogger(__name__)

# keccak256("Transfer(address,address,uint256)")
_TRANSFER_TOPIC: str = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# keccak256("Error(string)")[:4]
_ERROR_SELECTOR: bytes = bytes.fromhex("08c379a0")
# keccak256("Panic(uint256)")[:4]
_PANIC_SELECTOR: bytes = bytes.fromhex("4e487b71")

# ERC-20 function selectors
_ALLOWANCE_SELECTOR: bytes = bytes.fromhex("dd62ed3e")  # allowance(address,address)

# Sentinel for zero address (used in synthetic ETH transfer events)
_ZERO_ADDRESS: str = "0x0000000000000000000000000000000000000000"


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TokenTransfer:
    """An ERC-20 Transfer event parsed from transaction logs.

    When :func:`simulate_with_eth_simulate_v1` is called with
    ``trace_transfers=True``, native ETH value transfers also appear as
    synthetic Transfer events where :attr:`token` is the zero address
    (``"0x0000000000000000000000000000000000000000"``).

    Attributes:
        token: Checksum address of the ERC-20 contract (or the zero address
            for native ETH synthetic events).
        from_address: Sender checksum address (``address(0)`` for mints).
        to_address: Recipient checksum address (``address(0)`` for burns).
        amount: Raw token amount (in the token's smallest unit).
    """

    token: str
    from_address: str
    to_address: str
    amount: int


@dataclass
class BalanceChange:
    """Net balance change for a single address.

    Attributes:
        address: Checksum address of the account.
        token: ERC-20 contract address, or ``None`` for the native gas token
            (ETH, MATIC, …).
        delta: Signed integer change (positive = received, negative = sent).
    """

    address: str
    token: str | None  # None = native ETH
    delta: int


@dataclass
class SimulationResult:
    """Result of a simulated transaction.

    Attributes:
        success: ``True`` if the call completed without reverting.
        revert_reason: Human-readable revert reason if ``success`` is
            ``False``.  ``None`` when the reason cannot be decoded.
        return_data: Raw bytes returned by the call.
        transfers: ERC-20 Transfer events emitted during the call.  Empty
            when the RPC method does not expose logs (e.g. ``eth_call``).
        balance_changes: Aggregated net balance changes derived from
            *transfers*.
        gas_used: Estimated gas consumed, or ``None`` when not available.
        logs: Raw log entry dicts as returned by the RPC.
    """

    success: bool
    revert_reason: str | None
    return_data: bytes
    transfers: list[TokenTransfer] = field(default_factory=list)
    balance_changes: list[BalanceChange] = field(default_factory=list)
    gas_used: int | None = None
    logs: list[dict] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _decode_revert_reason(data: bytes) -> str | None:
    """Decode an ABI-encoded revert reason from raw revert data.

    Handles the two standard Solidity revert encodings:

    * ``Error(string)`` — produced by ``revert "message"`` and ``require``.
    * ``Panic(uint256)`` — produced by arithmetic overflows, array bounds
      errors, etc.

    Returns the human-readable string, or a hex representation of the raw
    data when neither pattern matches.  Returns ``None`` for empty input.
    """
    if not data:
        return None
    if data[:4] == _ERROR_SELECTOR and len(data) >= 4:
        try:
            (msg,) = abi_decode(["string"], data[4:])
            return str(msg)
        except Exception:  # noqa: BLE001
            pass
    elif data[:4] == _PANIC_SELECTOR and len(data) >= 4:
        try:
            (code,) = abi_decode(["uint256"], data[4:])
            return f"Panic(0x{code:02x})"
        except Exception:  # noqa: BLE001
            pass
    return f"0x{data.hex()}"


def _topic_to_address(topic: Any) -> str:
    """Convert a log topic (bytes or hex str) to a checksum address."""
    if isinstance(topic, (bytes, HexBytes)):
        raw = bytes(topic)[-20:]
    else:
        raw = bytes.fromhex(str(topic).removeprefix("0x"))[-20:]
    return Web3.to_checksum_address(raw)


def _parse_transfer_logs(logs: list) -> list[TokenTransfer]:
    """Parse ERC-20 Transfer events from a list of raw log entries.

    Each entry is a dict with ``address``, ``topics``, and ``data`` fields
    (the standard format returned by ``eth_simulateV1`` and ``debug_traceCall``
    with ``callTracer + withLog``).
    """
    transfers: list[TokenTransfer] = []
    for log in logs:
        topics = log.get("topics", [])
        if not topics:
            continue

        topic0 = topics[0]
        if isinstance(topic0, (bytes, HexBytes)):
            topic0_hex = "0x" + bytes(topic0).hex()
        else:
            topic0_hex = str(topic0)

        if topic0_hex.lower() != _TRANSFER_TOPIC:
            continue
        if len(topics) < 3:
            continue

        from_addr = _topic_to_address(topics[1])
        to_addr = _topic_to_address(topics[2])

        raw_data = log.get("data", "0x") or "0x"
        if isinstance(raw_data, (bytes, HexBytes)):
            amount_bytes = bytes(raw_data)
        else:
            amount_bytes = bytes.fromhex(str(raw_data).removeprefix("0x")) if raw_data not in ("0x", "") else b""

        amount = int.from_bytes(amount_bytes[:32], "big") if len(amount_bytes) >= 32 else 0

        token_raw = log.get("address", _ZERO_ADDRESS)
        token = Web3.to_checksum_address(token_raw)

        transfers.append(
            TokenTransfer(
                token=token,
                from_address=from_addr,
                to_address=to_addr,
                amount=amount,
            )
        )
    return transfers


def _aggregate_balance_changes(transfers: list[TokenTransfer]) -> list[BalanceChange]:
    """Compute net balance changes from a list of :class:`TokenTransfer` events.

    Native ETH synthetic transfers (token == zero address) produce a
    ``BalanceChange`` with ``token=None``.
    """
    # (address, token_or_None) -> signed delta
    deltas: dict[tuple[str, str | None], int] = {}

    for t in transfers:
        token: str | None = t.token if t.token != _ZERO_ADDRESS else None

        key_from = (t.from_address, token)
        deltas[key_from] = deltas.get(key_from, 0) - t.amount

        key_to = (t.to_address, token)
        deltas[key_to] = deltas.get(key_to, 0) + t.amount

    return [
        BalanceChange(address=addr, token=tok, delta=delta)
        for (addr, tok), delta in deltas.items()
        if delta != 0
    ]


def _collect_call_tree_logs(call: dict) -> list[dict]:
    """Recursively collect all logs from a ``callTracer`` call tree.

    Logs from reverted sub-calls are excluded since they are discarded by
    the EVM on revert.
    """
    logs: list[dict] = list(call.get("logs", []))
    for sub_call in call.get("calls", []):
        if not sub_call.get("error"):
            logs.extend(_collect_call_tree_logs(sub_call))
    return logs


# ── Storage slot detection ────────────────────────────────────────────────────


async def detect_allowance_slot(
    w3: AsyncWeb3,
    token: str,
    owner: str,
    spender: str,
    block: BlockIdentifier = "latest",
):
    """Detect the ERC-20 ``allowances`` mapping storage slot via ``debug_traceCall``.

    Traces an ``allowance(owner, spender)`` call on *token* and parses the
    resulting structLogs with :func:`eth_contract.slots.parse_allowance_slot`
    to find the base slot of the ``allowances`` nested mapping.

    Args:
        w3: Async web3 instance.
        token: Checksum address of the ERC-20 token contract.
        owner: Checksum address of the token owner.
        spender: Checksum address of the approved spender.
        block: Block identifier to trace against.

    Returns:
        A :class:`~eth_contract.slots.MappingSlot` representing the base slot
        of the ``allowances`` mapping, or ``None`` if detection fails (e.g.
        the RPC does not support ``debug_traceCall``).
    """
    from eth_contract.slots import parse_allowance_slot

    calldata = _ALLOWANCE_SELECTOR + abi_encode(["address", "address"], [owner, spender])
    block_param = block if isinstance(block, str) else hex(block)

    try:
        response = await w3.provider.make_request(
            "debug_traceCall",
            [
                {"from": owner, "to": token, "data": "0x" + calldata.hex()},
                block_param,
                {"disableStorage": True, "disableReturnData": True},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("debug_traceCall failed for detect_allowance_slot: %s", exc)
        return None

    if "error" in response:
        logger.debug("debug_traceCall error: %s", response["error"])
        return None

    result = response.get("result", {})
    struct_logs = result.get("structLogs", [])

    token_bytes = bytes.fromhex(Web3.to_checksum_address(token).removeprefix("0x"))
    owner_bytes = bytes.fromhex(Web3.to_checksum_address(owner).removeprefix("0x"))
    spender_bytes = bytes.fromhex(Web3.to_checksum_address(spender).removeprefix("0x"))

    return parse_allowance_slot(token_bytes, owner_bytes, spender_bytes, struct_logs)


async def build_allowance_state_override(
    w3: AsyncWeb3,
    token: str,
    owner: str,
    spender: str,
    amount: int = 2**256 - 1,
    block: BlockIdentifier = "latest",
) -> dict:
    """Build an ``eth_call`` / ``debug_traceCall`` state-override dict for an ERC-20 allowance.

    Uses :func:`detect_allowance_slot` (which requires ``debug_traceCall``) to
    find where the token stores ``allowances[owner][spender]``, then returns a
    state override that injects *amount* into that slot without sending an
    on-chain approval transaction.

    This works for any standard ERC-20 token whose allowances are stored in a
    Solidity or Vyper nested mapping.

    Args:
        w3: Async web3 instance.
        token: Checksum address of the ERC-20 token.
        owner: Checksum address of the token owner (the ``from`` of the tx
            that needs the allowance).
        spender: Checksum address of the contract being approved (the
            ``to`` of the downstream call that will call ``transferFrom``).
        amount: Allowance amount to inject.  Defaults to ``uint256`` max
            (unlimited approval).
        block: Block to detect the slot against.

    Returns:
        A state-override dict suitable for the *state_overrides* parameter of
        :func:`simulate_with_eth_call` or :func:`simulate_with_debug_trace_call`.

    Raises:
        ValueError: If the storage slot cannot be detected.
    """
    slot = await detect_allowance_slot(w3, token, owner, spender, block)
    if slot is None:
        raise ValueError(
            f"Could not detect allowance storage slot for token {token}. "
            "Ensure the RPC supports debug_traceCall."
        )

    owner_padded = bytes.fromhex(Web3.to_checksum_address(owner).removeprefix("0x")).rjust(32, b"\x00")
    spender_padded = bytes.fromhex(Web3.to_checksum_address(spender).removeprefix("0x")).rjust(32, b"\x00")

    # Compute the final storage slot: allowances[owner][spender]
    allowance_slot = slot.value(owner_padded).value(spender_padded)
    slot_hex = "0x" + allowance_slot.slot.hex()
    amount_hex = "0x" + amount.to_bytes(32, "big").hex()

    return {
        Web3.to_checksum_address(token): {
            "stateDiff": {slot_hex: amount_hex},
        }
    }


# ── Simulation functions ──────────────────────────────────────────────────────


async def simulate_with_eth_call(
    w3: AsyncWeb3,
    tx: dict,
    *,
    state_overrides: dict | None = None,
    block: BlockIdentifier = "latest",
) -> SimulationResult:
    """Simulate a transaction using ``eth_call``.

    This is the most widely supported simulation method but provides the least
    information — only the call's success/failure status and return data are
    available.  No logs or balance changes are returned.

    For ERC-20 approval simulation without an on-chain approval, use
    :func:`build_allowance_state_override` to generate a *state_overrides* dict
    that injects the required allowance into the token's storage.

    Args:
        w3: Async web3 instance.
        tx: Transaction dict (``from``, ``to``, ``data``, ``value``, …).
        state_overrides: Optional EVM state overrides (balances, nonces, code,
            storage slots).
        block: Block identifier to simulate against.

    Returns:
        A :class:`SimulationResult` with ``transfers`` and ``balance_changes``
        always empty (``eth_call`` does not expose logs).
    """
    try:
        return_data = await w3.eth.call(
            tx,
            block_identifier=block,
            state_override=state_overrides,
        )
        return SimulationResult(
            success=True,
            revert_reason=None,
            return_data=bytes(return_data),
        )
    except ContractLogicError as exc:
        reason = str(exc.message) if exc.message else None
        if not reason and exc.data:
            raw = str(exc.data).removeprefix("0x")
            if raw:
                reason = _decode_revert_reason(bytes.fromhex(raw))
        return SimulationResult(
            success=False,
            revert_reason=reason,
            return_data=b"",
        )
    except Web3RPCError as exc:
        return SimulationResult(
            success=False,
            revert_reason=str(exc),
            return_data=b"",
        )


async def simulate_with_eth_simulate_v1(
    w3: AsyncWeb3,
    calls: list[dict],
    *,
    state_overrides: dict | None = None,
    block: BlockIdentifier = "latest",
    trace_transfers: bool = True,
) -> list[SimulationResult]:
    """Simulate one or more transactions using ``eth_simulateV1``.

    ``eth_simulateV1`` supports multiple sequential messages in a single
    request and (when *trace_transfers* is ``True``) emits synthetic Transfer
    events for native ETH value transfers, enabling native balance tracking.

    This method is ideal for simulating interactions that require a prior ERC-20
    approval: prepend an ``approve(spender, amount)`` call to *calls* so the
    target transaction sees the allowance without an on-chain tx.

    Args:
        w3: Async web3 instance.
        calls: List of transaction dicts to simulate in sequence within the
            same block call.
        state_overrides: EVM state overrides applied before executing the calls.
        block: Block identifier to simulate against.
        trace_transfers: When ``True`` native ETH transfers appear as synthetic
            Transfer events (token == zero address) in each call's logs.

    Returns:
        A :class:`SimulationResult` for each entry in *calls*, in order.

    Raises:
        Exception: Propagates any RPC error (e.g. method not supported).
    """
    from web3.types import BlockStateCallV1, SimulateV1Payload

    block_state_call: BlockStateCallV1 = {"calls": calls}
    if state_overrides:
        block_state_call["stateOverrides"] = state_overrides

    payload: SimulateV1Payload = {
        "blockStateCalls": [block_state_call],
        "traceTransfers": trace_transfers,
    }

    sim_blocks = await w3.eth.simulate_v1(payload, block)

    results: list[SimulationResult] = []
    for block_result in sim_blocks:
        for call_result in block_result.get("calls", []):
            success = int(call_result.get("status", 0)) == 1

            raw_return = call_result.get("returnData", b"")
            if isinstance(raw_return, str):
                raw_return = bytes.fromhex(raw_return.removeprefix("0x"))
            else:
                raw_return = bytes(raw_return)

            revert_reason: str | None = None
            if not success:
                err = call_result.get("error")
                if isinstance(err, dict):
                    revert_reason = err.get("message") or err.get("data")
                elif err:
                    revert_reason = str(err)
                if not revert_reason and raw_return:
                    revert_reason = _decode_revert_reason(raw_return)

            raw_logs = list(call_result.get("logs", []))
            transfers = _parse_transfer_logs(raw_logs)
            balance_changes = _aggregate_balance_changes(transfers)

            gas_used_raw = call_result.get("gasUsed")
            if isinstance(gas_used_raw, str):
                gas_used: int | None = int(gas_used_raw, 16)
            else:
                gas_used = gas_used_raw

            results.append(
                SimulationResult(
                    success=success,
                    revert_reason=revert_reason,
                    return_data=raw_return if success else b"",
                    transfers=transfers,
                    balance_changes=balance_changes,
                    gas_used=gas_used,
                    logs=raw_logs,
                )
            )

    return results


async def simulate_with_debug_trace_call(
    w3: AsyncWeb3,
    tx: dict,
    *,
    state_overrides: dict | None = None,
    block: BlockIdentifier = "latest",
) -> SimulationResult:
    """Simulate a transaction using ``debug_traceCall`` with the call tracer.

    This provides richer information than :func:`simulate_with_eth_call` —
    including logs and gas usage — but requires an RPC node that supports the
    ``debug_traceCall`` method (e.g. Geth, Erigon, Anvil).

    For ERC-20 approval simulation, use :func:`build_allowance_state_override`
    to generate the required *state_overrides*.

    Args:
        w3: Async web3 instance.
        tx: Transaction dict.
        state_overrides: Optional EVM state overrides (same format as
            ``eth_call`` state override, passed in the tracer config).
        block: Block identifier to simulate against.

    Returns:
        A :class:`SimulationResult` including logs collected from the full
        call tree (reverted sub-calls are excluded).

    Raises:
        Exception: Propagates if the RPC returns an unexpected error.
    """
    block_param = block if isinstance(block, str) else hex(block)

    tracer_config: dict = {
        "tracer": "callTracer",
        "tracerConfig": {"withLog": True},
    }
    if state_overrides:
        tracer_config["stateOverrides"] = state_overrides

    response = await w3.provider.make_request(
        "debug_traceCall",
        [tx, block_param, tracer_config],
    )

    if "error" in response:
        err = response["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return SimulationResult(
            success=False,
            revert_reason=msg,
            return_data=b"",
        )

    result = response.get("result", {})
    success = not result.get("error") and not result.get("failed", False)

    raw_output = result.get("output") or result.get("returnValue") or ""
    if isinstance(raw_output, (bytes, HexBytes)):
        return_data = bytes(raw_output)
    else:
        return_data = bytes.fromhex(str(raw_output).removeprefix("0x")) if raw_output else b""

    revert_reason: str | None = None
    if not success:
        revert_reason = result.get("revertReason") or result.get("error")
        if not revert_reason and return_data:
            revert_reason = _decode_revert_reason(return_data)

    raw_logs = _collect_call_tree_logs(result)
    transfers = _parse_transfer_logs(raw_logs)
    balance_changes = _aggregate_balance_changes(transfers)

    gas_used_raw = result.get("gasUsed", "0x0")
    gas_used = int(str(gas_used_raw), 16) if isinstance(gas_used_raw, str) else int(gas_used_raw or 0)

    return SimulationResult(
        success=success,
        revert_reason=revert_reason,
        return_data=return_data if success else b"",
        transfers=transfers,
        balance_changes=balance_changes,
        gas_used=gas_used,
        logs=raw_logs,
    )


async def simulate_tx(
    w3: AsyncWeb3,
    tx: dict,
    *,
    state_overrides: dict | None = None,
    prepend_calls: list[dict] | None = None,
    block: BlockIdentifier = "latest",
) -> SimulationResult:
    """Simulate a transaction using the best available RPC method.

    Auto-detects which simulation method the connected RPC supports and
    chooses accordingly:

    1. **eth_simulateV1** — when *prepend_calls* is provided (multi-message).
    2. **debug_traceCall** — when available (logs + gas, single message).
    3. **eth_call** — always available (return data only, no logs).

    Args:
        w3: Async web3 instance.
        tx: Transaction dict (``from``, ``to``, ``data``, ``value``, …).
        state_overrides: Optional EVM state overrides applied before the call.
        prepend_calls: Additional calls to execute *before* the main *tx*
            within the same ``eth_simulateV1`` block call.  Useful for
            simulating an ERC-20 ``approve`` before the main swap transaction
            without sending a real approval on chain.
        block: Block identifier to simulate against.

    Returns:
        A :class:`SimulationResult` for the main transaction *tx*.
    """
    # ── 1. eth_simulateV1 (multi-message) ────────────────────────────────────
    if prepend_calls:
        all_calls = list(prepend_calls) + [tx]
        try:
            results = await simulate_with_eth_simulate_v1(
                w3,
                all_calls,
                state_overrides=state_overrides,
                block=block,
            )
            if results:
                return results[-1]
        except Exception as exc:  # noqa: BLE001
            logger.debug("eth_simulateV1 failed (%s); falling back to debug_traceCall", exc)

    # ── 2. debug_traceCall ────────────────────────────────────────────────────
    try:
        return await simulate_with_debug_trace_call(
            w3,
            tx,
            state_overrides=state_overrides,
            block=block,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("debug_traceCall failed (%s); falling back to eth_call", exc)

    # ── 3. eth_call (fallback) ────────────────────────────────────────────────
    return await simulate_with_eth_call(
        w3,
        tx,
        state_overrides=state_overrides,
        block=block,
    )

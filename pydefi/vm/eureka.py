"""DeFiVM helpers for IBC v2 (Eureka) sends — Program-side analogue to
:class:`pydefi.bridge.Eureka`. Both share
:func:`pydefi.bridge.encode_send_transfer_calldata`, so off-chain quotes
and on-chain calldata round-trip cleanly.

Uses ``call_raw`` + offset patches rather than ``call_contract`` because
``abi_to_vyper`` can't parse the bare ``(...)`` tuple-type string
``ContractFunction.from_abi`` returns for ``SendTransferMsg``. Runtime
fields (``amount``, optionally ``timeoutTimestamp``) are patched into the
encoded calldata at known ABI offsets.
"""

from __future__ import annotations

from typing import Any

from pydefi.abi.bridge import EUREKA_COMPOSER
from pydefi.bridge.eureka import (
    ICS20_DEFAULT_PORT,
    encode_send_transfer_calldata,
    validate_send_transfer_fields,
)
from pydefi.types import Address
from pydefi.vm.context import Operand, Program, ValueLike
from pydefi.vm.erc20 import emit_approve


def encode_send_and_compose_calldata(
    *,
    denom: Address | str,
    amount: int,
    receiver: str,
    source_client: str,
    timeout_timestamp: int,
    program: bytes,
    dest_port: str = ICS20_DEFAULT_PORT,
    memo: str = "",
) -> bytes:
    """ABI-encode ``EurekaComposer.sendTransferAndCompose(SendTransferMsg, program)``.
    Mirrors :func:`pydefi.bridge.encode_send_transfer_calldata` with an extra
    ``program`` argument: the DeFiVM bytecode the composer runs on
    ack/timeout."""
    denom_bytes = validate_send_transfer_fields(denom, amount, timeout_timestamp)
    return bytes(
        EUREKA_COMPOSER.fns.sendTransferAndCompose(
            (denom_bytes, amount, receiver, source_client, dest_port, timeout_timestamp, memo),
            program,
        ).data
    )


# ABI offsets within the encoded ``sendTransfer((address,uint256,string,string,string,uint64,string))``
# calldata. Layout:
#
#   [0 :4)   selector
#   [4 :36)  outer tuple pointer (0x20)
#   [36 :68)  denom              (address, right-padded in uint256 slot)
#   [68 :100) amount             (uint256)
#   [100:132) receiver           (offset pointer)
#   [132:164) sourceClient       (offset pointer)
#   [164:196) destPort           (offset pointer)
#   [196:228) timeoutTimestamp   (uint64, right-padded in uint256 slot)
#   [228:260) memo               (offset pointer)
#   [260: )   dynamic string bodies
_AMOUNT_OFFSET = 68
_TIMEOUT_OFFSET = 196


def _resolve_timeout(
    prog: Program,
    *,
    timeout_seconds: int | None,
    timeout_timestamp: ValueLike | None,
) -> ValueLike:
    """Return the value for the ``timeoutTimestamp`` slot. Exactly one of
    ``timeout_timestamp`` (absolute) or ``timeout_seconds`` (relative —
    resolved to ``block.timestamp + n`` at runtime) must be supplied."""
    if (timeout_timestamp is None) == (timeout_seconds is None):
        raise ValueError("supply exactly one of timeout_timestamp / timeout_seconds")
    if timeout_timestamp is not None:
        return timeout_timestamp
    assert timeout_seconds is not None  # XOR above guarantees this
    return prog.add(prog.builder.timestamp(), timeout_seconds)


def send_transfer(
    prog: Program,
    *,
    transfer_addr: Address,
    denom: Address,
    amount: ValueLike,
    receiver: str,
    source_client: str,
    timeout_seconds: int | None = None,
    timeout_timestamp: ValueLike | None = None,
    dest_port: str = ICS20_DEFAULT_PORT,
    memo: str = "",
    gas: ValueLike | None = None,
) -> Operand:
    """Emit ``ICS20Transfer.sendTransfer(SendTransferMsg)`` and return the
    CALL success operand.

    Caller must approve ``transfer_addr`` for ``amount`` of ``denom`` first
    (or use :func:`approve_then_send_transfer`). ``amount`` accepts a Python
    ``int`` (baked into the calldata) or an :class:`Operand` (patched in at
    runtime). String fields are compile-time only.
    """
    timeout_op = _resolve_timeout(
        prog,
        timeout_seconds=timeout_seconds,
        timeout_timestamp=timeout_timestamp,
    )

    # Bake the compile-time fields into a calldata template; mark the runtime
    # slots with placeholders that we'll overwrite via patches.
    static_amount = amount if isinstance(amount, int) else 0
    static_timeout = timeout_op if isinstance(timeout_op, int) else 0

    calldata = encode_send_transfer_calldata(
        denom=denom,
        amount=static_amount,
        receiver=receiver,
        source_client=source_client,
        timeout_timestamp=static_timeout,
        dest_port=dest_port,
        memo=memo,
    )

    patches: dict[int, ValueLike] = {}
    if not isinstance(amount, int):
        patches[_AMOUNT_OFFSET] = amount
    if not isinstance(timeout_op, int):
        patches[_TIMEOUT_OFFSET] = timeout_op

    return prog.call_raw(
        transfer_addr,
        calldata,
        gas=gas,
        patches=patches or None,
    )


def approve_then_send_transfer(
    prog: Program,
    *,
    transfer_addr: Address,
    denom: Address,
    amount: ValueLike,
    receiver: str,
    source_client: str,
    timeout_seconds: int | None = None,
    timeout_timestamp: ValueLike | None = None,
    dest_port: str = ICS20_DEFAULT_PORT,
    memo: str = "",
    approve_amount: ValueLike | None = None,
    **send_kwargs: Any,
) -> Operand:
    """``ERC20.approve(transfer_addr, amount)`` then :func:`send_transfer`,
    asserting the approve succeeded and returning the send's success
    operand. ``approve_amount`` defaults to ``amount`` — pass e.g.
    ``2**256 - 1`` for an unlimited allowance."""
    approve_val: ValueLike = approve_amount if approve_amount is not None else amount
    emit_approve(prog, denom, transfer_addr, approve_val)
    return send_transfer(
        prog,
        transfer_addr=transfer_addr,
        denom=denom,
        amount=amount,
        receiver=receiver,
        source_client=source_client,
        timeout_seconds=timeout_seconds,
        timeout_timestamp=timeout_timestamp,
        dest_port=dest_port,
        memo=memo,
        **send_kwargs,
    )

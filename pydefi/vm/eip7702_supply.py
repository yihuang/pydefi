"""Toolkit for :sol:`EIP7702BatchExecutor` — single-signature gasless deposits.

The owner delegates its EOA and signs an EIP-712 ``Batch``; a sponsor submits the
type-4 tx and ``execute`` runs ``[approve, supply]`` in the EOA's context, so the
plain ``supply`` credits the owner — no router, no Permit2, no ``onBehalfOf``. A
*call* is the standard pydefi ``{to, data, value}`` tx dict. UNAUDITED.
"""

from __future__ import annotations

from typing import Any

from eth_account import Account
from eth_contract import Contract

from pydefi.abi.vm import EIP7702_EXECUTOR
from pydefi.types import ZERO_ADDRESS, Address

_EXECUTOR_NAME = "EIP7702BatchExecutor"
_BATCH_NONCE = Contract.from_abi(["function batchNonce() view returns (uint256)"])
# A 7702-delegated EOA carries code ``0xef0100 || delegate`` (the delegation designator).
_DELEGATION_PREFIX = b"\xef\x01\x00"
_EXEC_GAS = 400_000
_REVOKE_GAS = 60_000


def _data_bytes(call: dict[str, Any]) -> bytes:
    data = call["data"]
    return data if isinstance(data, bytes) else bytes.fromhex(data.removeprefix("0x"))


def _to_hex(call: dict[str, Any]) -> str:
    to = call["to"]
    return to if isinstance(to, str) else "0x" + bytes(to).hex()


async def batch_nonce(w3, eoa: Address) -> int:
    """The executor's per-account replay counter for *eoa* — zero until the EOA
    is delegated (an undelegated account has no code to read)."""
    if not bytes(await w3.eth.get_code(bytes(eoa))):
        return 0
    return await _BATCH_NONCE.fns.batchNonce().call(w3, to=bytes(eoa))


async def is_delegated_to(w3, eoa: Address, delegate: Address) -> bool:
    """True if *eoa*'s code is the 7702 delegation designator pointing at *delegate*."""
    return bytes(await w3.eth.get_code(bytes(eoa))) == _DELEGATION_PREFIX + bytes(delegate)


def build_batch_typed_data(
    calls: list[dict[str, Any]], nonce: int, deadline: int, eoa: Address, chain_id: int
) -> dict[str, Any]:
    """EIP-712 ``Batch`` typed-data. ``verifyingContract`` is the EOA itself —
    the delegated code recovers the signer against ``address(this)``."""
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Batch": [
                {"name": "calls", "type": "Call[]"},
                {"name": "nonce", "type": "uint256"},
                {"name": "deadline", "type": "uint256"},
            ],
            "Call": [
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "data", "type": "bytes"},
            ],
        },
        "domain": {"name": _EXECUTOR_NAME, "chainId": chain_id, "verifyingContract": "0x" + bytes(eoa).hex()},
        "primaryType": "Batch",
        "message": {
            "calls": [{"to": _to_hex(c), "value": int(c.get("value", 0) or 0), "data": _data_bytes(c)} for c in calls],
            "nonce": nonce,
            "deadline": deadline,
        },
    }


def sign_authorization(private_key: str, delegate: Address, chain_id: int, nonce: int):
    """Sign a 7702 authorization tuple delegating the signer's EOA to *delegate*.
    *nonce* is the EOA's account nonce at submission time (the auth consumes it)."""
    return Account.from_key(private_key).sign_authorization(
        {"chainId": chain_id, "address": "0x" + bytes(delegate).hex(), "nonce": nonce}
    )


def build_revoke_authorization(
    private_key: str, chain_id: int, nonce: int, *, gas: int = _REVOKE_GAS
) -> dict[str, Any]:
    """Build a broadcast-ready type-4 tx that clears the signer EOA's 7702 delegation.

    Signs a 7702 authorization to the zero address (the revocation designator) and
    targets the now code-less EOA with an empty call, so once the authorization
    applies the account holds no delegated code. A sponsor can still submit it (the
    EOA pays no gas); *nonce* is the EOA's account nonce at submission time."""
    acct = Account.from_key(private_key)
    authorization = acct.sign_authorization({"chainId": chain_id, "address": "0x" + ZERO_ADDRESS.hex(), "nonce": nonce})
    return {
        "to": acct.address,
        "data": "0x",
        "value": "0",
        "gas": str(gas),
        "type": 4,
        "authorizationList": [authorization],
    }


def build_execute_tx(
    eoa: Address,
    calls: list[dict[str, Any]],
    nonce: int,
    deadline: int,
    signature: bytes,
    *,
    authorization: Any = None,
    gas: int = _EXEC_GAS,
) -> dict[str, Any]:
    """Encode ``EIP7702BatchExecutor.execute(...)`` as a tx targeting the delegated *eoa*.

    Pass *authorization* (from :func:`sign_authorization`) on the first deposit to set
    the EOA's code — that makes it a type-4 tx. Omit it once the EOA is already
    delegated; a plain call to the EOA then runs the delegated code."""
    call_tuples = [(_to_hex(c), int(c.get("value", 0) or 0), _data_bytes(c)) for c in calls]
    data = EIP7702_EXECUTOR.fns.execute(call_tuples, nonce, deadline, signature).data
    tx: dict[str, Any] = {"to": "0x" + bytes(eoa).hex(), "data": "0x" + data.hex(), "value": "0", "gas": str(gas)}
    if authorization is not None:
        tx["type"] = 4
        tx["authorizationList"] = [authorization]
    return tx

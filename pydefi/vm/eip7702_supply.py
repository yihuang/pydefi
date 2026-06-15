"""EIP-7702 gasless deposits via the audited Uniswap Calibur delegate
(https://github.com/Uniswap/calibur).

The owner delegates its EOA to Calibur and signs an EIP-712
``SignedBatchedCall``; a sponsor submits the tx and the batch runs in the
EOA's own context, so a plain ``supply`` credits the owner. A *call* is the
standard pydefi ``{to, data, value}`` tx dict.
"""

from __future__ import annotations

import asyncio
from typing import Any

from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import keccak
from hexbytes import HexBytes

from pydefi._utils import to_tx
from pydefi.abi.vm import CALIBUR as _CALIBUR_ABI
from pydefi.deployments import get_address
from pydefi.types import ZERO_ADDRESS, Address, ChainId

#: Calibur v1.0.0 singleton — one address on every chain it is deployed on;
#: see ``chains_for("CALIBUR")`` in :mod:`pydefi.deployments` for the list.
CALIBUR = get_address("CALIBUR", ChainId.ETHEREUM)

#: Calibur's root key (the EOA itself): ``KeyLib.ROOT_KEY_HASH == bytes32(0)``.
_ROOT_KEY_HASH = b"\x00" * 32

# A 7702-delegated EOA carries code ``0xef0100 || delegate`` (the delegation designator).
_DELEGATION_PREFIX = b"\xef\x01\x00"
# Calibur lays out ALL state at its ERC-7201 root (``CaliburEntry is Calibur
# layout at …``, namespace "Uniswap.Calibur.1.0.0"); ``nonceSequenceNumber``
# is the 5th state variable. Probed against the deployed v1.0.0 singleton.
_CALIBUR_STORAGE_ROOT = 0x3B86514C5C56B21F08D8E56AB090292E07C2483B3E667A2A45849DCB71368600
_NONCE_BASE_SLOT = _CALIBUR_STORAGE_ROOT + 4
# Covers the worst first deposit: authorization + approve + a first-time
# Aave v3 supply (~280k) + Calibur verification overhead.
_EXEC_GAS = 550_000
_REVOKE_GAS = 60_000


def _data_bytes(call: dict[str, Any]) -> bytes:
    return bytes(HexBytes(call["data"]))


def _to_hex(call: dict[str, Any]) -> str:
    # build_approve_tx emits "to" as Address bytes; the lending builders as hex str.
    to = call["to"]
    return to if isinstance(to, str) else Address(to).to_0x_hex()


async def is_delegated_to(w3, eoa: Address, delegate: Address) -> bool:
    """True if *eoa*'s code is the 7702 delegation designator pointing at *delegate*."""
    return bytes(await w3.eth.get_code(bytes(eoa))) == _DELEGATION_PREFIX + bytes(delegate)


def _nonce_slot(key: int) -> int:
    """Storage slot of ``nonceSequenceNumber[key]`` in the EOA's own storage."""
    return int.from_bytes(keccak(abi_encode(["uint256", "uint256"], [key, _NONCE_BASE_SLOT])), "big")


async def delegation_status(w3, eoa: Address, delegate: Address = CALIBUR) -> tuple[int, bool]:
    """``(nonce, needs_auth)`` for *eoa* against the Calibur *delegate*.

    Calibur nonces are 2-D (``key << 64 | seq``); the next key-0 seq is read
    straight from the EOA's storage, which works even before the EOA is
    delegated (storage persists across delegations) — so deposits use warm,
    sequential key-0 nonces from the very first one. Raises ``ValueError``
    when *delegate* has no code on this chain (delegating to it would make
    every sponsored call a silent no-op) or on a foreign delegation — revoke
    that first (:func:`build_revoke_authorization`)."""
    code, delegate_code, seq = await asyncio.gather(
        w3.eth.get_code(bytes(eoa)),
        w3.eth.get_code(bytes(delegate)),
        w3.eth.get_storage_at(bytes(eoa), _nonce_slot(0)),
    )
    if not bytes(delegate_code):
        raise ValueError(f"delegate {delegate.to_0x_hex()} has no code on this chain")
    code = bytes(code)
    if code and code != _DELEGATION_PREFIX + bytes(delegate):
        raise ValueError(
            f"EOA 0x{bytes(eoa).hex()} carries foreign code 0x{code.hex()}; "
            "revoke the existing EIP-7702 delegation before using the gasless path"
        )
    return int.from_bytes(bytes(seq), "big"), not code


def build_batch_typed_data(
    calls: list[dict[str, Any]],
    nonce: int,
    deadline: int,
    eoa: Address,
    chain_id: int,
    delegate: Address = CALIBUR,
    executor: Address = ZERO_ADDRESS,
) -> dict[str, Any]:
    """EIP-712 ``SignedBatchedCall`` typed-data for Calibur: ``verifyingContract``
    is the EOA, the domain ``salt`` is the left-padded *delegate* address (binding
    the implementation), and ``keyHash`` is the root key — the EOA's own key.
    *executor* pins who may submit (zero address = anyone)."""
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
                {"name": "salt", "type": "bytes32"},
            ],
            "SignedBatchedCall": [
                {"name": "batchedCall", "type": "BatchedCall"},
                {"name": "nonce", "type": "uint256"},
                {"name": "keyHash", "type": "bytes32"},
                {"name": "executor", "type": "address"},
                {"name": "deadline", "type": "uint256"},
            ],
            "BatchedCall": [
                {"name": "calls", "type": "Call[]"},
                {"name": "revertOnFailure", "type": "bool"},
            ],
            "Call": [
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "data", "type": "bytes"},
            ],
        },
        "domain": {
            "name": "Calibur",
            "version": "1.0.0",
            "chainId": chain_id,
            "verifyingContract": eoa.to_0x_hex(),
            "salt": "0x" + (b"\x00" * 12 + bytes(delegate)).hex(),
        },
        "primaryType": "SignedBatchedCall",
        "message": {
            "batchedCall": {
                "calls": [
                    {"to": _to_hex(c), "value": int(c.get("value", 0) or 0), "data": _data_bytes(c)} for c in calls
                ],
                "revertOnFailure": True,
            },
            "nonce": nonce,
            "keyHash": _ROOT_KEY_HASH,
            "executor": executor.to_0x_hex(),
            "deadline": deadline,
        },
    }


def _auth_to_dict(auth: Any) -> dict[str, str]:
    """A signed 7702 authorization as the JSON-RPC wire dict — tx dicts must stay
    JSON-serializable, and eth_account's pydantic object is not."""
    address = auth.address if isinstance(auth.address, str) else Address(auth.address).to_0x_hex()
    return {
        "chainId": hex(auth.chain_id),
        "address": address,
        "nonce": hex(auth.nonce),
        "yParity": hex(auth.y_parity),
        "r": hex(auth.r),
        "s": hex(auth.s),
    }


def sign_authorization(private_key: str, delegate: Address, chain_id: int, nonce: int):
    """Sign a 7702 authorization tuple delegating the signer's EOA to *delegate*.
    *nonce* is the EOA's account nonce at submission time (the auth consumes it)."""
    return Account.from_key(private_key).sign_authorization(
        {"chainId": chain_id, "address": delegate.to_0x_hex(), "nonce": nonce}
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
    authorization = acct.sign_authorization({"chainId": chain_id, "address": ZERO_ADDRESS.to_0x_hex(), "nonce": nonce})
    return {
        "to": acct.address,
        "data": "0x",
        "value": "0",
        "gas": str(gas),
        "type": 4,
        "authorizationList": [_auth_to_dict(authorization)],
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
    executor: Address = ZERO_ADDRESS,
) -> dict[str, Any]:
    """Encode ``Calibur.execute(signedBatchedCall, wrappedSignature)`` as a tx
    targeting the delegated *eoa*.

    Pass *authorization* (from :func:`sign_authorization`) on the first deposit to set
    the EOA's code — that makes it a type-4 tx. Omit it once the EOA is already
    delegated; a plain call to the EOA then runs the delegated code."""
    batched_call = ([(_to_hex(c), int(c.get("value", 0) or 0), _data_bytes(c)) for c in calls], True)
    signed_batched_call = (batched_call, nonce, _ROOT_KEY_HASH, executor.to_0x_hex(), deadline)
    # Calibur expects ``abi.encode(bytes signature, bytes hookData)``.
    wrapped_signature = abi_encode(["bytes", "bytes"], [signature, b""])
    data = _CALIBUR_ABI.fns.execute(signed_batched_call, wrapped_signature).data
    tx: dict[str, Any] = to_tx(eoa, data, gas=gas)
    if authorization is not None:
        tx["type"] = 4
        tx["authorizationList"] = [_auth_to_dict(authorization)]
    return tx

"""Unit tests for :mod:`pydefi.vm.eip7702_supply` — the Calibur toolkit (no network)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak

from pydefi.abi.vm import CALIBUR as CALIBUR_ABI
from pydefi.types import ZERO_ADDRESS, Address
from pydefi.vm.eip7702_supply import (
    _DELEGATION_PREFIX,
    CALIBUR,
    build_batch_typed_data,
    build_execute_tx,
    build_revoke_authorization,
    delegation_status,
    sign_authorization,
)

_EOA = Address("0x" + "11" * 20)
_FOREIGN = Address("0x" + "33" * 20)
_CALL = {"to": "0x" + "44" * 20, "value": "0", "data": "0xdeadbeef"}
_SIG = b"\x01" * 65

# Calibur type strings, verbatim from CallLib / BatchedCallLib / SignedBatchedCallLib.
_CALL_TYPE = b"Call(address to,uint256 value,bytes data)"
_BATCHED_TYPE = b"BatchedCall(Call[] calls,bool revertOnFailure)" + _CALL_TYPE
_SIGNED_TYPE = (
    b"SignedBatchedCall(BatchedCall batchedCall,uint256 nonce,bytes32 keyHash,address executor,uint256 deadline)"
    + _BATCHED_TYPE
)
_DOMAIN_TYPE = b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract,bytes32 salt)"


def _w3(code: bytes, seq: int = 0, delegate_code: bytes = b"\xfe") -> MagicMock:
    """*code* is the EOA's; the delegate (Calibur) carries *delegate_code*."""
    w3 = MagicMock()
    w3.eth.get_code = AsyncMock(side_effect=lambda a: delegate_code if bytes(a) == bytes(CALIBUR) else code)
    w3.eth.get_storage_at = AsyncMock(return_value=seq.to_bytes(32, "big"))  # nonceSequenceNumber[0]
    return w3


def _hash(types: list[str], values: list) -> bytes:
    return keccak(abi_encode(types, values))


@pytest.mark.asyncio
async def test_delegation_status_undelegated_reads_key0_seq_from_storage():
    """A code-less EOA needs an authorization; the key-0 seq comes straight
    from storage, which works even before delegation and stays correct across
    re-delegations (storage persists)."""
    assert await delegation_status(_w3(b""), _EOA) == (0, True)
    assert await delegation_status(_w3(b"", seq=3), _EOA) == (3, True)


@pytest.mark.asyncio
async def test_delegation_status_delegated_reads_key0_seq():
    w3 = _w3(_DELEGATION_PREFIX + bytes(CALIBUR), seq=5)
    assert await delegation_status(w3, _EOA) == (5, False)


@pytest.mark.asyncio
async def test_delegation_status_rejects_foreign_delegate():
    with pytest.raises(ValueError, match="foreign code"):
        await delegation_status(_w3(_DELEGATION_PREFIX + bytes(_FOREIGN)), _EOA)


@pytest.mark.asyncio
async def test_delegation_status_requires_deployed_delegate():
    """Calibur only exists on some chains; delegating to a code-less address
    would make every sponsored call a silent no-op."""
    with pytest.raises(ValueError, match="no code on this chain"):
        await delegation_status(_w3(b"", delegate_code=b""), _EOA)


def test_calibur_address_comes_from_deployments():
    from pydefi.deployments import chains_for, get_address
    from pydefi.types import ChainId

    assert get_address("CALIBUR", ChainId.BASE) == CALIBUR
    assert ChainId.ETHEREUM in chains_for("CALIBUR")


def test_batch_typed_data_matches_calibur_digest():
    """The typed-data must hash exactly as Calibur's ``_digest``: domain
    (name, version, chainId, verifyingContract=EOA, salt=implementation) and
    the nested SignedBatchedCall/BatchedCall/Call structs."""
    msg = encode_typed_data(full_message=build_batch_typed_data([_CALL], 7, 123, _EOA, 1))

    salt = b"\x00" * 12 + bytes(CALIBUR)  # PrefixedSaltLib with the default 0 prefix
    domain_sep = _hash(
        ["bytes32", "bytes32", "bytes32", "uint256", "address", "bytes32"],
        [keccak(_DOMAIN_TYPE), keccak(b"Calibur"), keccak(b"1.0.0"), 1, _EOA.to_0x_hex(), salt],
    )
    assert bytes(msg.header) == domain_sep

    call_hash = _hash(
        ["bytes32", "address", "uint256", "bytes32"],
        [keccak(_CALL_TYPE), _CALL["to"], 0, keccak(bytes.fromhex(_CALL["data"][2:]))],
    )
    batched_hash = _hash(["bytes32", "bytes32", "bool"], [keccak(_BATCHED_TYPE), keccak(call_hash), True])
    struct_hash = _hash(
        ["bytes32", "bytes32", "uint256", "bytes32", "address", "uint256"],
        [keccak(_SIGNED_TYPE), batched_hash, 7, b"\x00" * 32, ZERO_ADDRESS.to_0x_hex(), 123],
    )
    assert bytes(msg.body) == struct_hash


def test_execute_tx_encodes_calibur_call():
    """Calibur selector, tx targeting the EOA, abi.encode(signature, hookData)."""
    tx = build_execute_tx(_EOA, [_CALL], 7, 123, _SIG)
    data = bytes.fromhex(tx["data"][2:])
    assert tx["to"] == _EOA.to_0x_hex()
    assert data[:4] == CALIBUR_ABI.fns.execute.selector
    assert abi_encode(["bytes", "bytes"], [_SIG, b""]) in data  # hookData empty


def test_execute_and_revoke_txs_are_json_serializable():
    """The embedded 7702 authorization must be the JSON-RPC wire dict — tx dicts
    cross JSON boundaries — and eth_account must still accept that form when the
    sponsor signs the type-4 tx."""
    acct = Account.create()
    auth = sign_authorization(acct.key.hex(), CALIBUR, 1, 0)
    execute = build_execute_tx(Address(acct.address), [_CALL], 0, 9_999_999_999, _SIG, authorization=auth)
    revoke = build_revoke_authorization(acct.key.hex(), 1, 1)

    for tx in (execute, revoke):
        (entry,) = json.loads(json.dumps(tx))["authorizationList"]
        assert set(entry) == {"chainId", "address", "nonce", "yParity", "r", "s"}
        Account.create().sign_transaction(
            {
                "to": acct.address,
                "value": 0,
                "data": "0x",
                "chainId": 1,
                "nonce": 0,
                "gas": 100_000,
                "maxFeePerGas": 10**9,
                "maxPriorityFeePerGas": 10**9,
                "type": 4,
                "authorizationList": [entry],
            }
        )

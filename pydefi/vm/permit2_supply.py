"""Toolkit for :sol:`Permit2SupplyRouter` — single-signature gasless deposits.

The owner signs one Permit2 witness transfer binding the action (``protocol`` +
``keccak(supply_data)``); the router pulls the token and calls ``protocol`` with
that exact calldata. ``supply_data`` is the protocol's supply call crediting the
owner (the ``data`` the ``pydefi.lending`` builders already produce).

One-time setup per token: owner ``approve(PERMIT2)``; the router is ``prime``d
once per (token, protocol). Works for any token Permit2 supports, including USDT.

UNAUDITED — audit before mainnet.
"""

from __future__ import annotations

import secrets
from typing import Any

from eth_account.messages import encode_typed_data
from eth_contract import Contract
from eth_utils import keccak

from pydefi.abi.vm import PERMIT2_SUPPLY_ROUTER
from pydefi.types import Address

PERMIT2 = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
_PERMIT2 = Contract.from_abi(["function nonceBitmap(address owner, uint256 wordPos) view returns (uint256)"])
_FULL_WORD = 2**256 - 1
_SUPPLY_GAS = 350_000


def random_nonce() -> int:
    """A fresh Permit2 unordered nonce (256-bit; collision negligible)."""
    return int.from_bytes(secrets.token_bytes(32), "big")


async def pick_nonce(w3, owner: Address, scan_words: int = 8) -> int:
    """A Permit2 nonce reusing an already-touched bitmap word — a warm ``nonzero→
    nonzero`` SSTORE (~5k) vs ~22k for a fresh word, saving ~17k on the 2nd+ deposit.

    Picks the lowest non-full word (``nonce>>8``) and a random free bit in it; the
    owner's first deposit still hits a fresh word. Falls back to :func:`random_nonce`."""
    for word_pos in range(scan_words):
        bitmap = await _PERMIT2.fns.nonceBitmap(bytes(owner), word_pos).call(w3, to=PERMIT2)
        if bitmap != _FULL_WORD:
            free = [b for b in range(256) if not (bitmap >> b) & 1]
            return (word_pos << 8) | secrets.choice(free)
    return random_nonce()


def build_witness_typed_data(
    token: Address,
    amount: int,
    router: Address,
    nonce: int,
    deadline: int,
    protocol: Address,
    supply_data: bytes,
    chain_id: int,
) -> dict[str, Any]:
    """EIP-712 ``PermitWitnessTransferFrom`` for Permit2, witness =
    ``Witness(protocol, keccak(supply_data))``.  ``spender`` is the router."""
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "PermitWitnessTransferFrom": [
                {"name": "permitted", "type": "TokenPermissions"},
                {"name": "spender", "type": "address"},
                {"name": "nonce", "type": "uint256"},
                {"name": "deadline", "type": "uint256"},
                {"name": "witness", "type": "Witness"},
            ],
            "TokenPermissions": [{"name": "token", "type": "address"}, {"name": "amount", "type": "uint256"}],
            "Witness": [{"name": "protocol", "type": "address"}, {"name": "supplyDataHash", "type": "bytes32"}],
        },
        "domain": {"name": "Permit2", "chainId": chain_id, "verifyingContract": PERMIT2},
        "primaryType": "PermitWitnessTransferFrom",
        "message": {
            "permitted": {"token": "0x" + bytes(token).hex(), "amount": amount},
            "spender": "0x" + bytes(router).hex(),
            "nonce": nonce,
            "deadline": deadline,
            "witness": {"protocol": "0x" + bytes(protocol).hex(), "supplyDataHash": "0x" + keccak(supply_data).hex()},
        },
    }


def sign_witness(typed_data: dict[str, Any], private_key: str) -> bytes:
    """Sign the Permit2 witness typed-data; returns the 65-byte signature."""
    from eth_account import Account

    signed = Account.from_key(private_key).sign_message(encode_typed_data(full_message=typed_data))
    return bytes(signed.signature)


def build_supply_tx(
    router: Address,
    token: Address,
    amount: int,
    nonce: int,
    deadline: int,
    owner: Address,
    signature: bytes,
    protocol: Address,
    supply_data: bytes,
    gas: int = _SUPPLY_GAS,
) -> dict[str, Any]:
    """Encode ``Permit2SupplyRouter.supply(...)`` into a broadcast-ready tx dict."""
    permit = (("0x" + bytes(token).hex(), amount), nonce, deadline)
    data = PERMIT2_SUPPLY_ROUTER.fns.supply(
        permit, "0x" + bytes(owner).hex(), signature, "0x" + bytes(protocol).hex(), supply_data
    ).data
    return {"to": "0x" + bytes(router).hex(), "data": "0x" + data.hex(), "value": "0", "gas": str(gas)}


def build_prime_tx(router: Address, token: Address, protocol: Address) -> dict[str, Any]:
    """One-time ``router.prime(token, protocol)`` — max-approve the protocol to pull from the router."""
    data = PERMIT2_SUPPLY_ROUTER.fns.prime("0x" + bytes(token).hex(), "0x" + bytes(protocol).hex()).data
    return {"to": "0x" + bytes(router).hex(), "data": "0x" + data.hex(), "value": "0", "gas": "60000"}

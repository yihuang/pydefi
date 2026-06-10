"""Gasless deposits via :sol:`DeFiVM.executeWithPermit2`.

The owner signs one Permit2 witness transfer bound to ``keccak(program)``;
``executeWithPermit2`` pulls the token and runs exactly that program, so a
relayer can submit but cannot redirect. One-time setup per token: owner
``approve(PERMIT2)``.
"""

from __future__ import annotations

import secrets
from typing import Any

from eth_contract import Contract
from eth_contract.erc20 import ERC20
from eth_utils import keccak

from pydefi.abi.vm import DeFiVM
from pydefi.lending.utils import UINT256_MAX, to_tx
from pydefi.types import Address
from pydefi.vm.context import Program

PERMIT2 = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
_PERMIT2 = Contract.from_abi(["function nonceBitmap(address owner, uint256 wordPos) view returns (uint256)"])
_FULL_WORD = 2**256 - 1
# Covers the worst path: Permit2 pull + in-program approve + a first-time
# Aave v3 supply (~280k) + interpreter overhead.
_SUPPLY_GAS = 550_000


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


def build_supply_program(
    token: Address, protocol: Address, supply_data: bytes, *, approve: bool = True, reset: bool = False
) -> bytes:
    """Compile the program a gasless deposit runs after the Permit2 pull:
    optionally max-approve ``protocol``, then ``protocol.call(supply_data)``.
    ``supply_data`` must credit the owner (the VM is ``msg.sender``).

    Pass ``approve=False`` when the VM already carries a sufficient allowance
    (the common repeat case) — skipping the ~25k approve leg. The max approval
    is set once; a standing VM→protocol allowance is harmless because the VM
    never holds a balance between transactions. ``reset`` prepends ``approve(0)``
    for USDT-style tokens that forbid a nonzero→nonzero change — only needed
    when a nonzero-but-insufficient allowance is standing."""
    prog = Program()
    if reset:
        prog.assert_(prog.call_contract(bytes(token), ERC20.fns.approve, bytes(protocol), 0), "approve reset failed")
    if approve:
        prog.assert_(
            prog.call_contract(bytes(token), ERC20.fns.approve, bytes(protocol), UINT256_MAX), "approve failed"
        )
    prog.assert_(prog.call_raw(bytes(protocol), supply_data), "supply failed")
    prog.builder.stop()
    return prog.build()


def build_witness_typed_data(
    token: Address,
    amount: int,
    defivm: Address,
    nonce: int,
    deadline: int,
    program: bytes,
    chain_id: int,
) -> dict[str, Any]:
    """EIP-712 ``PermitWitnessTransferFrom`` for Permit2, witness =
    ``Witness(keccak(program))``. ``spender`` is the DeFiVM."""
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
            "Witness": [{"name": "programHash", "type": "bytes32"}],
        },
        "domain": {"name": "Permit2", "chainId": chain_id, "verifyingContract": PERMIT2},
        "primaryType": "PermitWitnessTransferFrom",
        "message": {
            "permitted": {"token": token.to_0x_hex(), "amount": amount},
            "spender": defivm.to_0x_hex(),
            "nonce": nonce,
            "deadline": deadline,
            "witness": {"programHash": "0x" + keccak(program).hex()},
        },
    }


def build_supply_tx(
    defivm: Address,
    token: Address,
    amount: int,
    nonce: int,
    deadline: int,
    owner: Address,
    signature: bytes,
    program: bytes,
    gas: int = _SUPPLY_GAS,
) -> dict[str, Any]:
    """Encode ``DeFiVM.executeWithPermit2(...)`` into a broadcast-ready tx dict."""
    permit = ((token.to_0x_hex(), amount), nonce, deadline)
    data = DeFiVM.fns.executeWithPermit2(permit, owner.to_0x_hex(), signature, program).data
    return {**to_tx(defivm, data), "gas": str(gas)}

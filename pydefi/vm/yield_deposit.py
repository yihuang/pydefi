"""Counterfactual deposit addresses via Safe + DeFiVM — zero signatures, zero
custom contracts.

The deposit address is a counterfactual 1/1 Safe whose ``setup`` initializer
delegatecalls :sol:`DeFiVM.execute` with a committed program; the canonical
``SafeProxyFactory`` derives the address from ``keccak(initializer)``, so it
commits to ``(owner, program)`` — the Permit2 witness commitment moved into
the address.

Fund it by plain transfer (CEX withdrawal, bridge output); anyone may deploy
the Safe, which runs the program in its own context: pay the committed fee to
``tx.origin``, supply the rest crediting the owner. Each address is single-shot
(``setup`` runs once) — recurring inflows rotate ``salt_nonce``; leftovers sit
in a Safe the owner controls.
"""

from __future__ import annotations

from typing import Any

from eth_contract.erc20 import ERC20
from eth_utils import keccak, to_checksum_address

from pydefi._utils import to_tx
from pydefi.abi.safe import SAFE as _SAFE_ABI
from pydefi.abi.safe import SAFE_PROXY_FACTORY as _FACTORY_ABI
from pydefi.abi.vm import DeFiVM
from pydefi.deployments import get_address
from pydefi.types import ZERO_ADDRESS, Address, ChainId
from pydefi.vm.context import Program
from pydefi.vm.erc20 import emit_approve

#: Safe v1.4.1 canonical deployments — one address on every supported chain;
#: see ``chains_for("SAFE_PROXY_FACTORY")`` in :mod:`pydefi.deployments`.
SAFE_PROXY_FACTORY = get_address("SAFE_PROXY_FACTORY", ChainId.ETHEREUM)
SAFE_SINGLETON = get_address("SAFE_SINGLETON", ChainId.ETHEREUM)

#: Place this as the amount in a protocol supply tx to mark where the runtime
#: balance must be patched in (12 random-looking bytes; collision negligible).
AMOUNT_SENTINEL = int.from_bytes(keccak(b"pydefi.yield_deposit.amount")[:12], "big")

# Safe proxy deployment + setup + the supply program (worst case ~280k for a
# first-time Aave v3 supply) + interpreter overhead.
_SWEEP_GAS = 900_000

# SafeProxyFactory.proxyCreationCode() — immutable and identical across the
# canonical v1.4.1 deployments, so fetched once per process.
_proxy_creation_code: bytes | None = None


def find_amount_offset(template: bytes, sentinel: int = AMOUNT_SENTINEL) -> int:
    """Byte offset of the 32-byte *sentinel* word inside *template* — the sweep
    patches the runtime amount there. Raises unless it occurs exactly once."""
    word = sentinel.to_bytes(32, "big")
    first = template.find(word)
    if first < 0 or template.find(word, first + 1) >= 0:
        raise ValueError("amount sentinel must occur exactly once in the supply calldata")
    return first


def build_deposit_program(
    token: Address, protocol: Address, supply_template: bytes, execution_fee: int, *, approve_reset: bool = False
) -> bytes:
    """Compile the sweep program: pay *execution_fee* to ``tx.origin`` (the
    relayer), approve *protocol* for the remaining *token* balance, and run
    *supply_template* with the remainder patched over the :data:`AMOUNT_SENTINEL`.
    The template must credit the owner — the Safe is ``msg.sender``.
    ``approve_reset`` prepends a USDT-style ``approve(0)``."""
    amount_offset = find_amount_offset(supply_template)
    prog = Program()
    prog.assert_(prog.call_contract(bytes(token), ERC20.fns.balanceOf, prog.builder.address()), "balance read failed")
    balance = prog.returndata_word(0)
    prog.assert_ge(balance, execution_fee + 1, "balance below fee")
    amount = prog.builder.sub(balance, execution_fee)
    if execution_fee:
        origin = prog.builder.origin()
        prog.assert_(prog.call_contract(bytes(token), ERC20.fns.transfer, origin, execution_fee), "fee failed")
    emit_approve(prog, token, protocol, amount, reset=approve_reset)
    prog.assert_(prog.call_raw(bytes(protocol), supply_template, patches={amount_offset: amount}), "supply failed")
    prog.builder.stop()
    return prog.build()


def build_initializer(owner: Address, defivm: Address, program: bytes) -> bytes:
    """``Safe.setup`` calldata committed into the deposit address: a 1/1 Safe
    owned by *owner* whose setup delegatecalls ``DeFiVM.execute(program)``."""
    zero = ZERO_ADDRESS.to_0x_hex()
    calls = ([owner.to_0x_hex()], 1, defivm.to_0x_hex(), DeFiVM.fns.execute(program).data, zero, zero, 0, zero)
    return _SAFE_ABI.fns.setup(*calls).data


async def deposit_address(w3, owner: Address, defivm: Address, program: bytes, salt_nonce: int = 0) -> Address:
    """The counterfactual deposit address: CREATE2 over the committed
    initializer, exactly as ``SafeProxyFactory.createProxyWithNonce`` derives it."""
    global _proxy_creation_code
    if _proxy_creation_code is None:
        factory = to_checksum_address(bytes(SAFE_PROXY_FACTORY))
        _proxy_creation_code = bytes(await _FACTORY_ABI.fns.proxyCreationCode().call(w3, to=factory))
    salt = keccak(keccak(build_initializer(owner, defivm, program)) + salt_nonce.to_bytes(32, "big"))
    init_hash = keccak(_proxy_creation_code + b"\x00" * 12 + bytes(SAFE_SINGLETON))
    return Address(keccak(b"\xff" + bytes(SAFE_PROXY_FACTORY) + salt + init_hash)[12:])


def build_sweep_tx(
    owner: Address, defivm: Address, program: bytes, salt_nonce: int = 0, gas: int = _SWEEP_GAS
) -> dict[str, Any]:
    """Encode ``createProxyWithNonce`` deploying the committed Safe — its setup
    sweeps the balance; broadcast by any relayer (which earns the committed fee)."""
    initializer = build_initializer(owner, defivm, program)
    data = _FACTORY_ABI.fns.createProxyWithNonce(SAFE_SINGLETON.to_0x_hex(), initializer, salt_nonce).data
    return to_tx(SAFE_PROXY_FACTORY, data, gas=gas)

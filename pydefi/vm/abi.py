"""DeFiVM ABI helpers — calldata builders for common ERC-20 operations.

These helpers produce raw ``bytes`` calldata suitable for use with
:func:`pydefi.vm.program.push_bytes` or :meth:`pydefi.vm.builder.Program.call_contract`.

Example — approve then swap::

    from pydefi.vm import Program
    from pydefi.vm.abi import erc20_approve, erc20_transfer

    bytecode = (
        Program()
        .call_contract(TOKEN, erc20_approve(ROUTER, amount_in))
        .call_contract(ROUTER, swap_calldata)
        .call_contract(TOKEN, erc20_transfer(RECIPIENT, 0))
        .build()
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _u256(n: int) -> bytes:
    if n < 0:
        raise ValueError(f"_u256: value must be non-negative, got {n}")
    return n.to_bytes(32, "big")


def _addr_word(a: str) -> bytes:
    """Encode an Ethereum address as a 32-byte ABI word (12 zero bytes + 20 address bytes)."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError(f"bad address length: expected 20 bytes, got {len(raw)} from {a!r}")
    return b"\x00" * 12 + raw


# ---------------------------------------------------------------------------
# ERC-20 calldata builders
# ---------------------------------------------------------------------------


def erc20_transfer(to: str, amount: int) -> bytes:
    """Build ``transfer(address,uint256)`` calldata.

    Selector: ``0xa9059cbb``
    """
    return bytes.fromhex("a9059cbb") + _addr_word(to) + _u256(amount)


def erc20_approve(spender: str, amount: int) -> bytes:
    """Build ``approve(address,uint256)`` calldata.

    Selector: ``0x095ea7b3``
    """
    return bytes.fromhex("095ea7b3") + _addr_word(spender) + _u256(amount)


def erc20_transfer_from(from_addr: str, to: str, amount: int) -> bytes:
    """Build ``transferFrom(address,address,uint256)`` calldata.

    Selector: ``0x23b872dd``
    """
    return bytes.fromhex("23b872dd") + _addr_word(from_addr) + _addr_word(to) + _u256(amount)


def erc20_balance_of(account: str) -> bytes:
    """Build ``balanceOf(address)`` calldata.

    Selector: ``0x70a08231``
    """
    return bytes.fromhex("70a08231") + _addr_word(account)

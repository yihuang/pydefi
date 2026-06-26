"""Uniswap V4 hook-permission helpers.

A V4 hook's permissions are encoded in the **low 14 bits of its address**
(v4-core ``Hooks.sol``), so what a hook *can* do is knowable locally from the
pool key alone. Only the ``*_RETURNS_DELTA`` permissions (hook fees / custom
curves) and, on dynamic-fee pools, ``BEFORE_SWAP`` lpFee overrides can change
swap economics — every other permission can at most veto a swap, leaving the
local V3-style math exact.
"""

from __future__ import annotations

from enum import IntFlag

from pydefi.types import Address


class HookFlag(IntFlag):
    """Permission bits from v4-core ``Hooks.sol`` (low address bit upward)."""

    AFTER_REMOVE_LIQUIDITY_RETURNS_DELTA = 1 << 0
    AFTER_ADD_LIQUIDITY_RETURNS_DELTA = 1 << 1
    AFTER_SWAP_RETURNS_DELTA = 1 << 2
    BEFORE_SWAP_RETURNS_DELTA = 1 << 3
    AFTER_DONATE = 1 << 4
    BEFORE_DONATE = 1 << 5
    AFTER_SWAP = 1 << 6
    BEFORE_SWAP = 1 << 7
    AFTER_REMOVE_LIQUIDITY = 1 << 8
    BEFORE_REMOVE_LIQUIDITY = 1 << 9
    AFTER_ADD_LIQUIDITY = 1 << 10
    BEFORE_ADD_LIQUIDITY = 1 << 11
    AFTER_INITIALIZE = 1 << 12
    BEFORE_INITIALIZE = 1 << 13


#: Permissions that let a hook change the amounts of a swap.
SWAP_DELTA_FLAGS = HookFlag.BEFORE_SWAP_RETURNS_DELTA | HookFlag.AFTER_SWAP_RETURNS_DELTA


def hook_flags(hooks: Address) -> HookFlag:
    """Return the permission bits encoded in *hooks*' low 14 address bits."""
    return HookFlag(int.from_bytes(hooks, "big") & ((1 << 14) - 1))


def has_swap_hook(hooks: Address) -> bool:
    """True if the hook runs on swaps at all (it may veto them, e.g. allowlists)."""
    return bool(hook_flags(hooks) & (HookFlag.BEFORE_SWAP | HookFlag.AFTER_SWAP))


def affects_swap_pricing(hooks: Address, *, is_dynamic_fee: bool = False) -> bool:
    """True if the hook can change swap amounts, making local math an estimate.

    Hook fees / custom curves need ``*_RETURNS_DELTA``; per-swap ``lpFee``
    overrides from ``beforeSwap`` are only honoured on dynamic-fee pools.
    False → the local math is exact; True → rank locally, quote on-chain.
    """
    flags = hook_flags(hooks)
    return bool(flags & SWAP_DELTA_FLAGS) or (is_dynamic_fee and HookFlag.BEFORE_SWAP in flags)

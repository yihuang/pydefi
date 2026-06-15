"""Shared Uniswap V4 ``PoolKey`` primitives.

V4 pools are keyed by ``PoolKey = (currency0, currency1, fee, tickSpacing,
hooks)`` with ``currency0 < currency1`` by address value. These helpers are
used by both the on-chain quoter client
(:class:`~pydefi.amm.uniswap_v4.UniswapV4`) and the Universal Router calldata
encoder (:class:`~pydefi.amm.universal_router.UniversalRouter`).
"""

from __future__ import annotations

from eth_abi import encode as abi_encode
from eth_utils import keccak

from pydefi.types import Address


def sort_currencies(a: Address, b: Address) -> tuple[Address, Address]:
    """Return ``(currency0, currency1)`` ordered by address value (lower first)."""
    return (a, b) if int.from_bytes(bytes(a), "big") < int.from_bytes(bytes(b), "big") else (b, a)


def pool_id(token_a: Address, token_b: Address, fee: int, tick_spacing: int, hooks: Address) -> bytes:
    """Return the 32-byte ``poolId`` (keccak256 of the ABI-encoded PoolKey).

    Currencies are sorted internally, so argument order does not matter.
    """
    c0, c1 = sort_currencies(token_a, token_b)
    return keccak(
        abi_encode(
            ["address", "address", "uint24", "int24", "address"],
            [c0.to_0x_hex(), c1.to_0x_hex(), fee, tick_spacing, hooks.to_0x_hex()],
        )
    )

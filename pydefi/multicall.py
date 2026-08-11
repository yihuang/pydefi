"""Batch read-only calls through Multicall3.

Anything fanning out over pools or ticks belongs here: providers rate-limit on
request count, not work done, so a burst of individual calls draws a 429 long
before it draws a timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from eth_contract.contract import ContractFunction
from eth_contract.multicall3 import MULTICALL3, Call3
from web3 import AsyncWeb3

from pydefi.deployments import get_address
from pydefi.types import Address

__all__ = ["batch_call"]

#: Calls per ``aggregate3``. Providers cap an ``eth_call`` by response size and
#: gas, not by call count, so this is a conservative split, not a hard limit.
DEFAULT_CHUNK_SIZE = 250


def _decode(fn: ContractFunction, success: bool, return_data: bytes, allow_failure: bool) -> Any:
    """Decode one entry, or ``None`` where the caller tolerates failure."""
    if not success:
        if allow_failure:
            return None
        raise ValueError(f"{fn.name} reverted inside a multicall batch")
    try:
        return fn.decode(bytes(return_data))
    except Exception:
        if allow_failure:
            return None  # e.g. bytes32 where a string was declared
        raise


async def batch_call(
    w3: AsyncWeb3,
    calls: Sequence[tuple[Address | str, ContractFunction]],
    *,
    multicall_address: Address | str | None = None,
    allow_failure: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[Any]:
    """Run *calls* in as few round trips as possible and return decoded results.

    Args:
        w3: Web3 client; its chain decides the Multicall3 deployment.
        calls: ``(target, bound_function)`` pairs, e.g.
            ``(pool, UNISWAP_V3_POOL.fns.ticks(t))``. A target may be an
            :class:`Address` or a hex string — the encoder takes either.
        multicall_address: Override the Multicall3 address, skipping the
            per-chain lookup.
        allow_failure: When ``True`` a call that reverts *or* decodes badly
            yields ``None`` rather than failing the batch — a non-standard
            ERC-20 returning ``bytes32`` from ``symbol()`` is the usual case.
            When ``False`` (default) both raise, so a ``None`` can never be
            mistaken for a real value.
        chunk_size: Calls per ``aggregate3``.

    Returns:
        Decoded results in the order given, ``None`` for calls that reverted
        under ``allow_failure``.

    Raises:
        ValueError: If a call reverted and *allow_failure* is ``False``, or a
            chunk came back short.
        KeyError: If Multicall3 has no known deployment on this chain — raised
            before any call, so a caller can fall back to individual reads.
    """
    if not calls:
        return []
    if multicall_address is None:
        multicall_address = get_address("MULTICALL3", await w3.eth.chain_id)

    chunks = [calls[i : i + chunk_size] for i in range(0, len(calls), chunk_size)]
    responses = await asyncio.gather(
        *(
            MULTICALL3.fns.aggregate3([Call3(target, allow_failure, fn.data) for target, fn in chunk]).call(
                w3, to=multicall_address
            )
            for chunk in chunks
        )
    )

    results: list[Any] = []
    for chunk, response in zip(chunks, responses):
        # Callers line results up with their calls, so a short response must
        # raise rather than let zip truncate it into a silent off-by-one.
        if len(response) != len(chunk):
            raise ValueError(f"multicall returned {len(response)} results for {len(chunk)} calls")
        results.extend(
            _decode(fn, success, data, allow_failure) for (_target, fn), (success, data) in zip(chunk, response)
        )
    return results

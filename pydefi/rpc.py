"""
Public RPC endpoint discovery and multi-RPC web3 provider.

This module provides:

- :func:`fetch_chain_rpcs` – fetch HTTP(S) RPC endpoints for a chain from
  `chainlist.org <https://chainlist.org/rpcs.json>`_ (result is cached in
  memory for the lifetime of the process).
- :class:`MultiRpcProvider` – an ``AsyncWeb3``-compatible provider that holds
  several JSON-RPC endpoints and automatically falls back to the next one
  whenever a request raises an exception (e.g. timeout, connection error, or
  rate-limit).
- :func:`get_w3` – convenience factory that returns an :class:`~web3.AsyncWeb3`
  instance backed by all public endpoints for a given chain ID.

Example::

    from pydefi.rpc import get_w3
    from pydefi.types import ChainId

    w3 = await get_w3(ChainId.ETHEREUM)
    block = await w3.eth.block_number
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
from web3 import AsyncWeb3
from web3.providers.async_base import AsyncJSONBaseProvider
from web3.types import RPCEndpoint, RPCResponse

__all__ = [
    "MultiRpcProvider",
    "fetch_chain_rpcs",
    "get_w3",
]

logger = logging.getLogger(__name__)

_CHAINLIST_URL = "https://chainlist.org/rpcs.json"

# Module-level cache: chain_id -> list of HTTP(S) RPC URLs
_rpc_cache: dict[int, list[str]] = {}
# Raw chainlist data cache (None until first fetch)
_chainlist_data: list[dict] | None = None


async def _fetch_chainlist_data() -> list[dict]:
    """Fetch and cache the raw chainlist RPC data from chainlist.org."""
    global _chainlist_data
    if _chainlist_data is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(_CHAINLIST_URL) as resp:
                resp.raise_for_status()
                _chainlist_data = await resp.json(content_type=None)
    return _chainlist_data


async def fetch_chain_rpcs(chain_id: int) -> list[str]:
    """Return a list of public HTTP(S) JSON-RPC endpoints for *chain_id*.

    The chainlist data is downloaded once from ``https://chainlist.org/rpcs.json``
    and then cached in-process.  Subsequent calls for the same or a different
    chain ID reuse the cached payload.

    Only ``http://`` and ``https://`` endpoints are returned (WebSocket and
    IPC entries are excluded).  Endpoints that contain template placeholders
    (``${...}``) are also excluded because they require an API key.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).

    Returns:
        List of HTTP(S) RPC URLs.  May be empty if the chain is not found or
        has no public HTTP endpoints.
    """
    if chain_id in _rpc_cache:
        return _rpc_cache[chain_id]

    data = await _fetch_chainlist_data()

    urls: list[str] = []
    for entry in data:
        if entry.get("chainId") != chain_id:
            continue
        for rpc in entry.get("rpc", []):
            # rpc entries can be strings or dicts like {"url": "...", ...}
            url: str = rpc if isinstance(rpc, str) else rpc.get("url", "")
            if not url:
                continue
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            # Skip endpoints that require an API key (contain template vars)
            if "${" in url:
                continue
            urls.append(url)
        break  # found the matching chain entry

    _rpc_cache[chain_id] = urls
    return urls


class MultiRpcProvider(AsyncJSONBaseProvider):
    """An async web3 provider that tries multiple JSON-RPC endpoints in order.

    When a request to the current endpoint raises an exception (e.g. a network
    error or a connection timeout), the provider transparently retries the same
    request against the next endpoint in the list.  If *all* endpoints fail the
    last exception is re-raised.

    Args:
        endpoints: One or more JSON-RPC endpoint URLs.  At least one must be
            provided.

    Raises:
        ValueError: If *endpoints* is empty.

    Example::

        from web3 import AsyncWeb3
        from pydefi.rpc import MultiRpcProvider

        provider = MultiRpcProvider(
            "https://eth.drpc.org",
            "https://cloudflare-eth.com",
        )
        w3 = AsyncWeb3(provider)
        block = await w3.eth.block_number
    """

    def __init__(self, *endpoints: str, **kwargs: Any) -> None:
        if not endpoints:
            raise ValueError("MultiRpcProvider requires at least one endpoint")
        super().__init__(**kwargs)
        self._endpoints: list[str] = list(endpoints)
        # Index of the currently active endpoint.
        self._current_index: int = 0
        # The single cached provider for the active endpoint; None until first use.
        self._current_provider: AsyncWeb3.AsyncHTTPProvider | None = None

    def _build_provider(self, url: str) -> AsyncWeb3.AsyncHTTPProvider:
        """Create a new :class:`AsyncHTTPProvider` for *url*."""
        return AsyncWeb3.AsyncHTTPProvider(url)

    async def _close_provider(self, provider: AsyncWeb3.AsyncHTTPProvider) -> None:
        """Best-effort close of a provider that is no longer needed."""
        try:
            await provider.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AsyncBaseProvider interface
    # ------------------------------------------------------------------

    async def make_request(self, method: RPCEndpoint, params: Any) -> RPCResponse:
        last_exc: Exception | None = None
        n = len(self._endpoints)
        for i in range(n):
            index = (self._current_index + i) % n
            # Reuse the cached provider on the first iteration; create a fresh
            # temporary one for every fallback so unused endpoint objects are
            # never retained.
            is_current = i == 0 and self._current_provider is not None
            candidate = self._current_provider if is_current else self._build_provider(self._endpoints[index])
            try:
                response = await candidate.make_request(method, params)
                # Success – adopt this provider as the new current one and drop the old.
                if not is_current:
                    old = self._current_provider
                    self._current_provider = candidate
                    # asyncio is single-threaded so this assignment is safe from
                    # data races; concurrent coroutines that both succeed will each
                    # set _current_index to their own successful endpoint, which is
                    # an acceptable last-write-wins outcome.
                    self._current_index = index
                    if old is not None:
                        await self._close_provider(old)
                return response
            except Exception as exc:
                logger.debug(
                    "RPC endpoint %s failed for %s: %s – trying next",
                    self._endpoints[index],
                    method,
                    exc,
                )
                last_exc = exc
                # Release temporary providers that won't be cached.
                if not is_current:
                    await self._close_provider(candidate)

        assert last_exc is not None  # guaranteed: self._endpoints is non-empty
        raise last_exc

    async def is_connected(self, show_traceback: bool = False) -> bool:
        """Return ``True`` if *any* endpoint is reachable."""
        n = len(self._endpoints)
        for i in range(n):
            index = (self._current_index + i) % n
            is_current = i == 0 and self._current_provider is not None
            p = self._current_provider if is_current else self._build_provider(self._endpoints[index])
            try:
                connected = await p.is_connected(show_traceback=show_traceback)
            except Exception:
                connected = False
            finally:
                if not is_current:
                    await self._close_provider(p)
            if connected:
                return True
        return False


async def get_w3(chain_id: int) -> AsyncWeb3:
    """Create an :class:`~web3.AsyncWeb3` instance for *chain_id*.

    RPC endpoints are discovered automatically from chainlist.org.

    Args:
        chain_id: EVM chain ID (e.g. ``1`` for Ethereum mainnet).

    Returns:
        An :class:`~web3.AsyncWeb3` instance backed by a
        :class:`MultiRpcProvider` containing all public HTTP(S) endpoints
        listed for the chain.

    Raises:
        ValueError: If no public HTTP(S) endpoints are found for *chain_id*.

    Example::

        from pydefi.rpc import get_w3
        from pydefi.types import ChainId

        w3 = await get_w3(ChainId.ETHEREUM)
        print(await w3.eth.block_number)
    """
    endpoints = await fetch_chain_rpcs(chain_id)
    if not endpoints:
        raise ValueError(f"No public HTTP RPC endpoints found for chain_id={chain_id}")
    return AsyncWeb3(MultiRpcProvider(*endpoints))

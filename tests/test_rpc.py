"""Tests for pydefi.rpc – multi-RPC provider and chainlist integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from web3.types import RPCEndpoint, RPCResponse

from pydefi.rpc import (
    MultiRpcProvider,
    _rpc_cache,
    fetch_chain_rpcs,
    get_w3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal chainlist-style JSON payload
_MOCK_CHAINLIST = [
    {
        "name": "Ethereum Mainnet",
        "chain": "ETH",
        "chainId": 1,
        "rpc": [
            "https://eth.drpc.org",
            "https://cloudflare-eth.com",
            "wss://mainnet.infura.io/ws/v3/${INFURA_KEY}",  # ws -- excluded
            "https://mainnet.infura.io/v3/${INFURA_KEY}",  # template -- excluded
        ],
    },
    {
        "name": "Arbitrum One",
        "chain": "ARB1",
        "chainId": 42161,
        "rpc": [
            "https://arb1.arbitrum.io/rpc",
            {"url": "https://arbitrum.drpc.org"},  # dict form
        ],
    },
]


def _make_mock_aiohttp(payload: Any):
    """Return a context-manager mock that yields a response with *payload*."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=payload)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return mock_session


# ---------------------------------------------------------------------------
# fetch_chain_rpcs
# ---------------------------------------------------------------------------


class TestFetchChainRpcs:
    def setup_method(self):
        # Clear module-level caches before each test
        _rpc_cache.clear()
        import pydefi.rpc as rpc_module

        rpc_module._chainlist_data = None

    async def test_returns_http_endpoints_only(self):
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session):
            urls = await fetch_chain_rpcs(1)

        assert "https://eth.drpc.org" in urls
        assert "https://cloudflare-eth.com" in urls
        # WebSocket endpoint must be excluded
        assert not any(u.startswith("wss://") for u in urls)
        # Template endpoints must be excluded
        assert not any("${" in u for u in urls)

    async def test_dict_rpc_entries(self):
        """RPC entries given as dicts (with a 'url' key) are handled."""
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session):
            urls = await fetch_chain_rpcs(42161)

        assert "https://arb1.arbitrum.io/rpc" in urls
        assert "https://arbitrum.drpc.org" in urls

    async def test_unknown_chain_returns_empty_list(self):
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session):
            urls = await fetch_chain_rpcs(99999)

        assert urls == []

    async def test_result_is_cached(self):
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session) as mock_cls:
            await fetch_chain_rpcs(1)
            await fetch_chain_rpcs(1)  # second call – should use cache

        # ClientSession should only have been instantiated once (for the
        # initial chainlist download); the second fetch_chain_rpcs call should
        # hit the _rpc_cache without touching the network.
        assert mock_cls.call_count == 1

    async def test_chainlist_fetched_once_for_different_chains(self):
        """The raw chainlist JSON should be downloaded only once."""
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session) as mock_cls:
            await fetch_chain_rpcs(1)
            await fetch_chain_rpcs(42161)

        assert mock_cls.call_count == 1


# ---------------------------------------------------------------------------
# MultiRpcProvider
# ---------------------------------------------------------------------------


def _make_rpc_response(result: Any = "0x1") -> RPCResponse:
    return {"jsonrpc": "2.0", "id": 1, "result": result}  # type: ignore[return-value]


class TestMultiRpcProvider:
    def test_requires_at_least_one_endpoint(self):
        with pytest.raises(ValueError, match="at least one endpoint"):
            MultiRpcProvider()

    async def test_succeeds_on_first_provider(self):
        provider = MultiRpcProvider(
            "https://endpoint-1.example.com",
            "https://endpoint-2.example.com",
        )
        expected = _make_rpc_response("0x42")
        provider._providers[0].make_request = AsyncMock(return_value=expected)
        provider._providers[1].make_request = AsyncMock(return_value=_make_rpc_response("0x99"))

        response = await provider.make_request(RPCEndpoint("eth_blockNumber"), [])

        assert response == expected
        provider._providers[0].make_request.assert_called_once()
        provider._providers[1].make_request.assert_not_called()

    async def test_falls_back_to_second_provider_on_exception(self):
        provider = MultiRpcProvider(
            "https://endpoint-1.example.com",
            "https://endpoint-2.example.com",
        )
        expected = _make_rpc_response("0x42")
        provider._providers[0].make_request = AsyncMock(side_effect=ConnectionError("timeout"))
        provider._providers[1].make_request = AsyncMock(return_value=expected)

        response = await provider.make_request(RPCEndpoint("eth_blockNumber"), [])

        assert response == expected

    async def test_raises_when_all_providers_fail(self):
        provider = MultiRpcProvider(
            "https://endpoint-1.example.com",
            "https://endpoint-2.example.com",
        )
        provider._providers[0].make_request = AsyncMock(side_effect=ConnectionError("fail-1"))
        provider._providers[1].make_request = AsyncMock(side_effect=ConnectionError("fail-2"))

        with pytest.raises(ConnectionError, match="fail-2"):
            await provider.make_request(RPCEndpoint("eth_blockNumber"), [])

    async def test_is_connected_returns_true_if_any_provider_is_up(self):
        provider = MultiRpcProvider(
            "https://endpoint-1.example.com",
            "https://endpoint-2.example.com",
        )
        provider._providers[0].is_connected = AsyncMock(return_value=False)
        provider._providers[1].is_connected = AsyncMock(return_value=True)

        assert await provider.is_connected() is True

    async def test_is_connected_returns_false_if_all_down(self):
        provider = MultiRpcProvider("https://endpoint-1.example.com")
        provider._providers[0].is_connected = AsyncMock(return_value=False)

        assert await provider.is_connected() is False

    async def test_tries_all_providers_before_raising(self):
        provider = MultiRpcProvider(
            "https://endpoint-1.example.com",
            "https://endpoint-2.example.com",
            "https://endpoint-3.example.com",
        )
        for p in provider._providers:
            p.make_request = AsyncMock(side_effect=OSError("down"))

        with pytest.raises(OSError):
            await provider.make_request(RPCEndpoint("eth_chainId"), [])

        for p in provider._providers:
            p.make_request.assert_called_once()


# ---------------------------------------------------------------------------
# get_w3
# ---------------------------------------------------------------------------


class TestGetW3:
    def setup_method(self):
        _rpc_cache.clear()
        import pydefi.rpc as rpc_module

        rpc_module._chainlist_data = None

    async def test_returns_async_web3_instance(self):
        from web3 import AsyncWeb3

        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session):
            w3 = await get_w3(1)

        assert isinstance(w3, AsyncWeb3)
        assert isinstance(w3.provider, MultiRpcProvider)

    async def test_raises_for_unknown_chain(self):
        mock_session = _make_mock_aiohttp(_MOCK_CHAINLIST)
        with patch("pydefi.rpc.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ValueError, match="No public HTTP RPC endpoints"):
                await get_w3(99999)

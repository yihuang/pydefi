"""Shared fixtures for live integration tests.

Live tests require network access to:
- A public Ethereum JSON-RPC endpoint (default: https://eth.drpc.org)
- Public APIs that do not require an API key (e.g. ParaSwap /prices)

Set the ``ETH_RPC_URL`` environment variable to override the default RPC.
All live tests are marked with ``@pytest.mark.live`` and are excluded from
the regular ``pytest`` run.  Run them explicitly with::

    pytest -m live
"""

import os
import time

import pytest
from web3 import AsyncWeb3

from pydifi.types import ChainId, Token

# ---------------------------------------------------------------------------
# Public RPC
# ---------------------------------------------------------------------------

ETH_RPC_URL = os.environ.get("ETH_RPC_URL", "https://eth.drpc.org")

# ---------------------------------------------------------------------------
# Well-known Ethereum mainnet tokens
# ---------------------------------------------------------------------------

WETH = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    symbol="WETH",
    decimals=18,
)
USDC = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    symbol="USDC",
    decimals=6,
)
DAI = Token(
    chain_id=ChainId.ETHEREUM,
    address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
    symbol="DAI",
    decimals=18,
)
USDT = Token(
    chain_id=ChainId.ETHEREUM,
    address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
    symbol="USDT",
    decimals=6,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eth_w3() -> AsyncWeb3:
    """Return an :class:`~web3.AsyncWeb3` instance backed by a public RPC."""
    return AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(ETH_RPC_URL))


@pytest.fixture(autouse=True)
def _throttle_live_requests(request: pytest.FixtureRequest) -> None:
    """Insert a small delay before each live test to avoid rate-limiting on free RPCs."""
    if request.node.get_closest_marker("live"):
        time.sleep(1)

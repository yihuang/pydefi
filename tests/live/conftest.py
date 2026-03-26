"""Shared fixtures for live integration tests.

Live tests require network access to:
- A public Ethereum JSON-RPC endpoint (default: https://eth.drpc.org)
- Public APIs that do not require an API key (e.g. ParaSwap /prices)

Set the ``ETH_RPC_URL`` environment variable to override the default RPC.
All live tests are marked with ``@pytest.mark.live`` and are excluded from
the regular ``pytest`` run.  Run them explicitly with::

    pytest -m live

Fork tests (``@pytest.mark.fork``) require `Anvil
<https://book.getfoundry.sh/anvil/>`_ (part of the Foundry toolchain) to be
installed and available on ``$PATH``.  They spin up a temporary Anvil process
that forks the configured ``ETH_RPC_URL`` and execute *real* transactions
against that local fork.  Run them with::

    pytest -m fork
"""

import asyncio
import os
import socket
import subprocess
import time

import pytest
from web3 import AsyncWeb3

from pydefi.types import ChainId, Token

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


def _free_port() -> int:
    """Return an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def fork_w3(request: pytest.FixtureRequest):
    """Start a temporary Anvil fork of Ethereum mainnet and return an AsyncWeb3 client.

    The fixture:

    1. Finds a free local TCP port.
    2. Launches ``anvil --fork-url <ETH_RPC_URL>`` as a subprocess.
    3. Polls the JSON-RPC endpoint until the node is ready (up to 30 s).
    4. Yields an :class:`~web3.AsyncWeb3` instance connected to the fork.
    5. Terminates the Anvil process on fixture teardown.

    The fixture is automatically skipped when the ``anvil`` binary is not
    found on ``$PATH`` so that the test suite can still run in environments
    where Foundry is not installed.
    """
    import shutil

    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            "anvil",
            "--fork-url",
            ETH_RPC_URL,
            "--port",
            str(port),
            "--silent",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until anvil is ready (up to 30 seconds).  The connection will
    # actively fail (ConnectionRefusedError, HTTP errors, web3 wrapping
    # exceptions) until the process is fully started, so we intentionally
    # swallow all exceptions here.
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            await w3.eth.chain_id
            break
        except Exception:  # noqa: BLE001 — expected during startup
            await asyncio.sleep(0.25)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        pytest.fail("Anvil did not start within 30 seconds")

    yield w3

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture(scope="module")
async def fork_w3_module():
    """Module-scoped Anvil mainnet fork.  Same as ``fork_w3`` but shared across
    an entire test module to avoid per-test process startup costs."""
    import shutil

    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            "anvil",
            "--fork-url",
            ETH_RPC_URL,
            "--port",
            str(port),
            "--silent",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            await w3.eth.chain_id
            break
        except Exception:  # noqa: BLE001 — expected during startup
            await asyncio.sleep(0.25)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        pytest.fail("Anvil did not start within 30 seconds")

    yield w3

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

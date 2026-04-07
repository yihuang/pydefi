"""Shared fixtures for live integration tests.

Live tests require network access to:
- A public Ethereum JSON-RPC endpoint (default: https://eth.drpc.org)
- Public APIs that do not require an API key (e.g. ParaSwap /prices)

Set the ``ETH_RPC_URL`` environment variable to override the default RPC.
All live tests are marked with ``@pytest.mark.live`` and are excluded from
the regular ``pytest`` run.  Run them explicitly with::

    pytest -m live

Fork tests (``@pytest.mark.fork``) require either:

- **EVM fork**: `Anvil <https://book.getfoundry.sh/anvil/>`_ (part of the
  Foundry toolchain) installed on ``$PATH``.  They spin up a temporary Anvil
  process that forks the configured ``ETH_RPC_URL``.

- **Solana fork**: `surfpool <https://github.com/txtx/surfpool>`_ installed
  on ``$PATH``.  They spin up a local surfpool process that forks Solana
  mainnet state via ``SOLANA_RPC_URL`` (default:
  ``https://api.mainnet-beta.solana.com``).

Run fork tests with::

    pytest -m fork
"""

import asyncio
import os
import socket
import subprocess
import time

import aiohttp
import pytest
from web3 import AsyncWeb3

from pydefi.rpc import get_w3
from pydefi.types import ChainId, Token
from tests.live.sol_utils import compile_interpreter_sync

# ---------------------------------------------------------------------------
# Public RPC
# ---------------------------------------------------------------------------

ETH_RPC_URL = os.environ.get("ETH_RPC_URL", "https://eth.drpc.org")

# ---------------------------------------------------------------------------
# Solana public RPC (used for simulation and as the surfpool upstream)
# ---------------------------------------------------------------------------

SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

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
# Analog-Labs EVM interpreter (shared by all DeFiVM fork tests)
# ---------------------------------------------------------------------------

# Well-known mainnet address where the interpreter is pre-deployed via CREATE2.
INTERPRETER_ADDR = "0x0000000000001e3F4F615cd5e20c681Cf7d85e8D"


async def _ensure_interpreter(w3: AsyncWeb3, deployer: str) -> str:
    """Return EVM interpreter address, compiling + deploying one if needed.

    If the Analog-Labs interpreter is pre-deployed on this fork, returns its
    well-known address.  Otherwise compiles and deploys a fresh copy of
    ``Interpreter.sol`` so tests can run on any fork network.
    """
    code = await w3.eth.get_code(INTERPRETER_ADDR)
    if code and len(code) > 1:
        return INTERPRETER_ADDR

    compiled = await asyncio.to_thread(compile_interpreter_sync)
    key = "<stdin>:Interpreter"
    contract = w3.eth.contract(abi=compiled[key]["abi"], bytecode=compiled[key]["bin"])
    tx_hash = await contract.constructor().transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def interpreter_addr(fork_w3_module) -> str:
    """Return EVM interpreter address for this fork's Anvil instance.

    If the Analog-Labs interpreter is already deployed at its well-known
    CREATE2 address, that address is returned.  Otherwise a fresh copy of
    ``Interpreter.sol`` is compiled with py-solcx and deployed so that fork
    tests run on any network without needing a mainnet fork.
    """
    accounts = await fork_w3_module.eth.accounts
    deployer = accounts[0]
    return await _ensure_interpreter(fork_w3_module, deployer)


@pytest.fixture
async def eth_w3() -> AsyncWeb3:
    """Return an :class:`~web3.AsyncWeb3` instance backed by public RPC endpoints
    auto-discovered via chainlist.org, with automatic failover."""
    return await get_w3(ChainId.ETHEREUM)


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


@pytest.fixture
async def surfpool_rpc():
    """Start a surfpool Solana mainnet fork and yield the local RPC URL.

    surfpool is a drop-in replacement for ``solana-test-validator`` that mirrors
    live mainnet state without a full chain download.  It exposes the standard
    Solana JSON-RPC interface at the chosen port.

    The fixture is automatically skipped when the ``surfpool`` binary is not
    found on ``$PATH``.  Install it with::

        curl -sL https://run.surfpool.run/ | bash

    Set ``SOLANA_RPC_URL`` to override the upstream Solana mainnet RPC used for
    the fork (default: ``https://api.mainnet-beta.solana.com``).
    """
    import shutil

    if shutil.which("surfpool") is None:
        pytest.skip("surfpool not found on PATH — install surfpool to run Solana fork tests")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("SOLANA_RPC_URL", SOLANA_RPC_URL)

    proc = subprocess.Popen(
        ["surfpool", "start", "--rpc-port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until surfpool is ready (mainnet state sync can take up to 60 s).
    deadline = time.monotonic() + 60
    ready = False
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth", "params": []}
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json(content_type=None)
                    if "result" in data:
                        ready = True
                        break
        except Exception:  # noqa: BLE001 — expected during startup
            pass
        await asyncio.sleep(1)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        pytest.fail("surfpool did not start within 60 seconds")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

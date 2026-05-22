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

from pydefi.abi.codec import codec
from pydefi.rpc import get_w3
from pydefi.types import Address, ChainId
from tests.addrs import INTERPRETER_ADDR
from tests.live.sol_utils import compile_interpreter_sync

# ---------------------------------------------------------------------------
# Public RPC
# ---------------------------------------------------------------------------

ETH_RPC_URL = os.environ.get("ETH_RPC_URL") or "https://eth.drpc.org"

# ---------------------------------------------------------------------------
# Solana public RPC (used for simulation and as the surfpool upstream)
# ---------------------------------------------------------------------------

SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


async def _ensure_interpreter(w3: AsyncWeb3, deployer: str) -> Address:
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
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=0.1)
    return Address(receipt["contractAddress"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def interpreter_addr(fork_w3_module) -> Address:
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
    auto-discovered via chainlist.org, with automatic failover.

    Set the ``ETH_RPC_URL`` environment variable to use a specific endpoint
    instead of the auto-discovered ones (useful for authenticated providers
    such as Infura or Alchemy).
    """
    if "ETH_RPC_URL" in os.environ:
        return AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(ETH_RPC_URL))
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


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a local-node subprocess (Anvil or surfpool), escalating to
    SIGKILL if it does not exit on SIGTERM."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


async def _start_anvil_w3(extra_args: list[str]) -> tuple[subprocess.Popen, AsyncWeb3]:
    """Launch an Anvil node with *extra_args*, poll until it answers, and
    return its process and a connected :class:`~web3.AsyncWeb3`.

    Skips the test when ``anvil`` is not on ``$PATH``; fails it when the node
    does not come up within 30 s. The connection errors until the process is
    fully started, so every exception during the poll is expected.
    """
    import shutil

    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    proc = subprocess.Popen(
        ["anvil", "--port", str(port), "--silent", *extra_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(f"http://127.0.0.1:{port}"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            await w3.eth.chain_id
            break
        except Exception:  # noqa: BLE001 — expected during startup
            await asyncio.sleep(0.25)
    else:
        _terminate(proc)
        pytest.fail("Anvil did not start within 30 seconds")
    w3.codec = codec
    return proc, w3


@pytest.fixture
async def fork_w3():
    """Function-scoped Anvil fork of Ethereum mainnet (forks ``ETH_RPC_URL``)."""
    proc, w3 = await _start_anvil_w3(["--fork-url", ETH_RPC_URL])
    yield w3
    _terminate(proc)


@pytest.fixture
async def plain_anvil_w3():
    """A plain (non-forked) Anvil node — a fast, empty second chain for tests
    that need a source chain alongside ``fork_w3``."""
    proc, w3 = await _start_anvil_w3([])
    yield w3
    _terminate(proc)


@pytest.fixture(scope="module")
async def fork_w3_module():
    """Module-scoped ``fork_w3`` — one mainnet fork shared across a module."""
    proc, w3 = await _start_anvil_w3(["--fork-url", ETH_RPC_URL])
    yield w3
    _terminate(proc)


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
        _terminate(proc)
        pytest.fail("surfpool did not start within 60 seconds")

    yield url
    _terminate(proc)

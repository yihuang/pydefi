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
import contextlib
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import aiohttp
import pytest
from eth_utils import keccak
from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from pydefi.abi.codec import codec
from pydefi.rpc import get_w3
from pydefi.types import Address, ChainId
from tests.addrs import INTERPRETER_ADDR
from tests.live.sol_utils import compile_interpreter_sync

# ---------------------------------------------------------------------------
# Public RPC
# ---------------------------------------------------------------------------

ETH_RPC_URL = os.environ.get("ETH_RPC_URL") or "https://eth.drpc.org"
SEPOLIA_RPC_URL = os.environ.get("SEPOLIA_RPC_URL") or "https://sepolia.drpc.org"

# Polymarket's Conditional Tokens live on Polygon, so its fork tests fork
# Polygon mainnet rather than Ethereum.  Override with ``POLYGON_RPC_URL``.
POLYGON_RPC_URL = os.environ.get("POLYGON_RPC_URL") or "https://polygon.drpc.org"

# ---------------------------------------------------------------------------
# Solana public RPC (used for simulation and as the surfpool upstream)
# ---------------------------------------------------------------------------

SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def _expected_interpreter_codehash() -> bytes:
    """Parse ``PATCHED_INTERPRETER_CODEHASH`` from the auto-generated PatchedInterpreterConstants.sol."""
    constants_path = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "PatchedInterpreterConstants.sol"
    m = re.search(
        r"constant\s+PATCHED_INTERPRETER_CODEHASH\s*=\s*\n?\s*(0x[0-9a-fA-F]{64})",
        constants_path.read_text(),
    )
    if not m:
        raise RuntimeError(f"could not find PATCHED_INTERPRETER_CODEHASH in {constants_path}")
    return bytes.fromhex(m.group(1)[2:])


async def _ensure_interpreter(w3: AsyncWeb3, deployer: str) -> Address:
    """Return EVM interpreter address, compiling + deploying the pydefi-patched
    interpreter if needed.

    If the **pydefi-patched** interpreter is already deployed at the
    deterministic CREATE2 address :data:`INTERPRETER_ADDR` **and the deployed
    code's keccak256 matches** ``PATCHED_INTERPRETER_CODEHASH``, returns that
    address.  Otherwise compiles :file:`pydefi/vm/PatchedInterpreter.sol` and
    deploys a fresh copy at a new address so tests run against a known-good
    interpreter on any fork network.

    The codehash check matters because the deterministic address is
    pydefi-specific: on arbitrary forks there's a non-zero chance something
    unrelated has been deployed at that slot.  Without it, tests could silently
    run against an unintended contract that happens to live at the same salt +
    initcode-hash address.
    """
    code = await w3.eth.get_code(INTERPRETER_ADDR)
    if code and len(code) > 1 and keccak(code) == _expected_interpreter_codehash():
        return INTERPRETER_ADDR

    compiled = await asyncio.to_thread(compile_interpreter_sync)
    contract = w3.eth.contract(
        abi=compiled["<stdin>:Interpreter"]["abi"],
        bytecode=compiled["<stdin>:Interpreter"]["bin"],
    )
    tx_hash = await contract.constructor().transact({"from": deployer})
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=0.1)
    return Address(receipt["contractAddress"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def interpreter_addr(fork_w3_module) -> Address:
    """Return EVM interpreter address for this fork's Anvil instance.

    If the **pydefi-patched** interpreter is already deployed at the
    deterministic CREATE2 address :data:`INTERPRETER_ADDR` **and the
    deployed code's keccak256 matches** ``PATCHED_INTERPRETER_CODEHASH``,
    returns that address.

    Otherwise (no code, or wrong code at that slot) compiles
    :file:`pydefi/vm/PatchedInterpreter.sol` and deploys a fresh copy at a
    new address so tests run against a known-good interpreter.

    The codehash check matters because the deterministic address is
    pydefi-specific: on arbitrary forks there's a non-zero chance something
    unrelated has been deployed at that slot.  Without it, tests could
    silently run against an unintended contract that happens to live at
    the same salt + initcode-hash address.
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


@pytest.fixture
async def polygon_w3() -> AsyncWeb3:
    """Return an :class:`~web3.AsyncWeb3` backed by a public Polygon RPC.

    Chainlist auto-discovery often lands on gated Polygon endpoints, so this
    pins ``POLYGON_RPC_URL`` directly.  Polygon is a PoA chain, so the POA
    middleware is injected to keep block-fetching calls working."""
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(POLYGON_RPC_URL))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


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
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


@contextlib.asynccontextmanager
async def _anvil_node(extra_args: list[str]):
    """Spawn an Anvil node with *extra_args*, yield a connected AsyncWeb3, tear down.

    Finds a free port, launches ``anvil`` with *extra_args* (a ``--fork-url``
    for the mainnet forks, nothing for a plain second chain), polls the
    JSON-RPC endpoint until the node is ready (up to 30 s), then terminates the
    process on exit.  The connection actively fails until anvil is fully
    started, so startup exceptions are intentionally swallowed.

    Skips the test when the ``anvil`` binary is not on ``$PATH`` so the suite
    still runs where Foundry is not installed.
    """
    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["anvil", "--port", str(port), "--silent", *extra_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(url))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                await w3.eth.chain_id
                break
            except Exception:  # noqa: BLE001 — expected during startup
                await asyncio.sleep(0.25)
        else:
            pytest.fail("Anvil did not start within 30 seconds")
        w3.codec = codec
        yield w3
    finally:
        _terminate(proc)


@pytest.fixture
async def fork_w3():
    """Function-scoped Anvil fork of Ethereum mainnet (forks ``ETH_RPC_URL``)."""
    async with _anvil_node(["--fork-url", ETH_RPC_URL]) as w3:
        yield w3


@pytest.fixture
async def plain_anvil_w3():
    """A plain (non-forked) Anvil node — a fast, empty second chain for tests
    that need a source chain alongside ``fork_w3``."""
    async with _anvil_node([]) as w3:
        yield w3


@pytest.fixture(scope="module")
async def fork_w3_module():
    """Module-scoped Anvil fork of Ethereum mainnet, shared across a module to
    avoid per-test process startup costs."""
    async with _anvil_node(["--fork-url", ETH_RPC_URL]) as w3:
        yield w3


@pytest.fixture
async def polygon_fork_w3():
    """Function-scoped Anvil fork of Polygon mainnet (forks ``POLYGON_RPC_URL``).

    Polymarket's Conditional Tokens live on Polygon, so its fork tests need a
    Polygon fork rather than the Ethereum :func:`fork_w3`.  Polygon is a PoA
    chain with oversized block ``extraData``, so the POA middleware is injected
    to keep ``eth_sendTransaction`` (which fetches the latest block) working."""
    async with _anvil_node(["--fork-url", POLYGON_RPC_URL]) as w3:
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        yield w3


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

"""RUNCODE (EIP-7990) fork tests — the RuncodeRunner backend on live state.

Runs real pydefi programs natively in-context via the RUNCODE (0xF6) opcode on
an Anvil mainnet fork built against a RUNCODE-patched revm. A tiny raw-bytecode
executor copies the program from calldata and RUNCODEs it — the on-chain
analogue of ``RuncodeRunner`` (whose .sol can't be compiled by solc).

Build the patched Anvil and point ``PYDEFI_RUNCODE_ANVIL`` at it (skipped when
unset; ``ETH_RPC_URL`` comes from the repo-root .env)::

    cargo build -p anvil
    export PYDEFI_RUNCODE_ANVIL=/path/to/foundry/target/debug/anvil
    pytest -m fork tests/live/test_runcode_fork.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time

import pytest
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.vm import Program
from tests.live.conftest import ETH_RPC_URL

ANVIL_BIN = os.environ.get("PYDEFI_RUNCODE_ANVIL")

pytestmark = [
    pytest.mark.fork,
    pytest.mark.skipif(
        not ANVIL_BIN,
        reason="set PYDEFI_RUNCODE_ANVIL to a RUNCODE-enabled anvil binary",
    ),
]

WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
EXECUTOR_ADDR = Web3.to_checksum_address("0x000000000000000000000000000000000000faf6")

# Raw runtime for a RUNCODE executor (calldata IS the program): CALLDATACOPY the
# program into memory, RUNCODE it, then RETURN its returndata — or REVERT on
# failure (the JUMPI at 0x16 selects the path).
RUNCODE_EXECUTOR = HexBytes("0x365f5f375f5f5f5f365f5af63d5f5f3e6016573d5ffd5b3d5ff3")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
async def runcode_fork():
    """Launch the RUNCODE-enabled Anvil as a mainnet fork and deploy the
    raw-bytecode RUNCODE executor at a fixed address via anvil_setCode."""
    port = _free_port()
    proc = subprocess.Popen(
        # RUNCODE (0xF6) is gated to Amsterdam in the patched revm, so run the
        # fork under the Amsterdam spec to activate the opcode.
        [ANVIL_BIN, "--fork-url", ETH_RPC_URL, "--hardfork", "amsterdam", "--port", str(port), "--silent"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(f"http://127.0.0.1:{port}"))
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                await w3.eth.chain_id
                break
            except Exception:  # noqa: BLE001 — expected during startup
                await asyncio.sleep(0.25)
        else:
            pytest.fail("patched anvil did not start within 60s")

        await w3.provider.make_request("anvil_setCode", [EXECUTOR_ADDR, RUNCODE_EXECUTOR.to_0x_hex()])
        code = await w3.eth.get_code(EXECUTOR_ADDR)
        assert bytes(code) == bytes(RUNCODE_EXECUTOR), "executor code not set"
        yield w3
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _run_program(w3, program: bytes) -> bytes:
    """eth_call the RUNCODE executor with *program* as calldata; return output."""
    ret = await w3.eth.call({"to": EXECUTOR_ADDR, "data": HexBytes(program).to_0x_hex()})
    return bytes(ret)


async def test_runcode_executor_runs_pydefi_program(runcode_fork):
    """A real pydefi embed_and_load program runs via RUNCODE on the fork."""
    data = bytes(range(1, 33))
    prog = Program()
    buf = prog.embed_and_load(data)
    prog.return_(buf, 32)
    out = await _run_program(runcode_fork, prog.build())
    assert out == data


async def test_runcode_reads_live_forked_state(runcode_fork):
    """The RuncodeRunner path on live state: a pydefi program executed via
    RUNCODE calls the real mainnet WETH and returns its totalSupply."""
    w3 = runcode_fork

    prog = Program()
    ok = prog.call_contract(HexBytes(WETH), "function totalSupply() returns (uint256)")
    prog.assert_(ok)
    prog.return_word(prog.returndata_word(0))
    out = await _run_program(w3, prog.build())
    got = int.from_bytes(out, "big")

    # Compare against a direct read of WETH on the same fork.
    direct = await w3.eth.call({"to": WETH, "data": "0x18160ddd"})  # totalSupply()
    want = int.from_bytes(bytes(direct), "big")

    assert got == want, f"RUNCODE program read {got}, direct read {want}"
    assert got > 0, "WETH totalSupply should be non-zero on a mainnet fork"

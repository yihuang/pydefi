"""Local RUNCODE (EIP-7990) execution tests.

Runs pydefi-compiled programs natively in-context via the RUNCODE (0xf6) opcode
through a patched ``evm`` CLI — the local stand-in for the ``RuncodeRunner``
backend. Proves pydefi's emitted bytecode (notably ``embed_and_load``'s
data-section + ``CODECOPY``) runs correctly under RUNCODE.

Build the ``evm`` from mmsqe/go-ethereum @ a95baa03 and point ``PYDEFI_EVM_BIN``
at it (the module is skipped when unset)::

    git clone https://github.com/mmsqe/go-ethereum
    cd go-ethereum && git checkout a95baa03c0f4734e417341ea819b7b2a82f9f974
    go build -o /tmp/evm ./cmd/evm
    export PYDEFI_EVM_BIN=/tmp/evm
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest

from pydefi.vm import Program
from tests.runcode_evm import AMSTERDAM_CHAIN_CONFIG, EVM_OUTPUT_RE, runcode_wrap

# Pinned go-ethereum commit implementing RUNCODE (EIP-7990); build its cmd/evm
# and point PYDEFI_EVM_BIN at the result (see the module docstring).
GOETH_EVM_SOURCE = "https://github.com/mmsqe/go-ethereum/commit/a95baa03c0f4734e417341ea819b7b2a82f9f974"

EVM_BIN = os.environ.get("PYDEFI_EVM_BIN")

pytestmark = pytest.mark.skipif(
    not EVM_BIN,
    reason="set PYDEFI_EVM_BIN to a RUNCODE-enabled `evm` (build cmd/evm from mmsqe/go-ethereum@a95baa03; see module docstring)",
)


def run_via_evm(bytecode: bytes, *, amsterdam: bool = True) -> bytes:
    """Execute *bytecode* through the patched ``evm`` CLI and return its output.

    With ``amsterdam=True`` (default) RUNCODE is enabled. With ``amsterdam=False``
    the default fork (Osaka) is used, where 0xf6 is an invalid opcode. Raises
    ``AssertionError`` if execution errors (e.g. invalid opcode).
    """
    cmd = [EVM_BIN, "run"]
    genesis_path = None
    if amsterdam:
        genesis = {"config": AMSTERDAM_CHAIN_CONFIG, "gasLimit": "0x1000000", "difficulty": "0x0", "alloc": {}}
        fd, genesis_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(genesis, f)
        cmd += ["--prestate", genesis_path]
    cmd += ["--codefile", "-"]
    try:
        proc = subprocess.run(cmd, input=bytecode.hex(), capture_output=True, text=True, timeout=30)
    finally:
        if genesis_path:
            os.unlink(genesis_path)

    combined = proc.stdout + "\n" + proc.stderr
    assert "error:" not in combined, f"evm execution error:\n{combined}"
    matches = EVM_OUTPUT_RE.findall(proc.stdout)
    assert matches, f"no output line found in evm stdout:\n{proc.stdout}\n{proc.stderr}"
    return bytes.fromhex(matches[-1][2:])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_arithmetic_program_under_runcode():
    """A plain pydefi program runs correctly when executed via RUNCODE."""
    p = Program()
    p.return_word(p.add(p.const(7), p.const(5)))
    out = run_via_evm(runcode_wrap(p.build()))
    assert int.from_bytes(out, "big") == 12


def test_embed_and_load_codecopy_under_runcode():
    """The load-bearing case: embed_and_load (data section + CODECOPY) resolves
    against the in-memory program under RUNCODE, returning the embedded data."""
    data = bytes(range(1, 33))  # 0x01..0x20
    p = Program()
    buf = p.embed_and_load(data)
    p.return_(buf, 32)
    out = run_via_evm(runcode_wrap(p.build()))
    assert out == data


def test_runcode_disabled_without_amsterdam():
    """0xf6 is an invalid opcode unless the Amsterdam fork (EIP-7990) is active."""
    p = Program()
    p.return_word(p.const(1))
    wrapped = runcode_wrap(p.build())
    with pytest.raises(AssertionError, match="(?i)invalid opcode|error"):
        run_via_evm(wrapped, amsterdam=False)

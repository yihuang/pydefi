"""Benchmark: interpreter backend vs native RUNCODE (EIP-7990) gas.

Runs the *same* pydefi-compiled programs two ways and compares EVM gas:

  * ``interpreter`` — the program is passed as calldata to the pre-deployed
    Analog-Labs EVM interpreter via ``DELEGATECALL`` (today's ``DeFiVM`` /
    ``InterpreterRunner`` path). The interpreter re-executes each program
    opcode in-EVM.
  * ``runcode``     — the program runs natively in-context via the EIP-7990
    ``RUNCODE`` (0xf6) opcode (the future ``RuncodeRunner`` path).
  * ``native``      — the program run directly as account code, as a floor.

Gas is measured with a patched go-ethereum ``evm`` CLI (``--statdump``) at the
Amsterdam fork, where RUNCODE is enabled.

Setup::

    cd /path/to/go-ethereum            # branch: eip-7990-runcode
    go build -o /tmp/evm ./cmd/evm
    export PYDEFI_EVM_BIN=/tmp/evm     # defaults to /tmp/evm

    python3 tests/bench_runcode_backends.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydefi.vm import Program  # noqa: E402
from tests.addrs import INTERPRETER_ADDR  # noqa: E402
from tests.live.sol_utils import compile_interpreter_sync  # noqa: E402
from tests.runcode_evm import AMSTERDAM_CHAIN_CONFIG, EVM_OUTPUT_RE, push2, runcode_wrap  # noqa: E402

EVM_BIN = os.environ.get("PYDEFI_EVM_BIN", "/tmp/evm")

_GAS_RE = re.compile(r"EVM gas used:\s+(\d+)")


def interp_drive(program: bytes) -> bytes:
    """Prologue that CODECOPYs *program* to mem[0] and DELEGATECALLs the
    interpreter with it as calldata, returning the interpreter's RETURN data.
    Mirrors DeFiVM.execute."""
    ll = len(program)
    off = 0x2D  # fixed prologue length below
    pro = (
        push2(ll)
        + push2(off)
        + bytes([0x5F, 0x39])  # CODECOPY prog->mem[0]
        + bytes([0x5F, 0x5F])
        + push2(ll)
        + bytes([0x5F])  # retSize, retOff, argsSize, argsOff
        + bytes([0x73])
        + bytes(INTERPRETER_ADDR)  # PUSH20 interpreter
        + bytes([0x5A, 0xF4, 0x50])  # GAS, DELEGATECALL, POP
        + bytes([0x3D, 0x5F, 0x5F, 0x3E])  # RETURNDATACOPY(0, 0, rds)
        + bytes([0x3D, 0x5F, 0xF3])  # RETURN(0, rds)
    )
    assert len(pro) == off, (len(pro), off)
    return pro + program


# ---------------------------------------------------------------------------
# evm CLI harness
# ---------------------------------------------------------------------------


def _genesis(alloc: dict | None = None) -> str:
    g = {"config": AMSTERDAM_CHAIN_CONFIG, "gasLimit": "0x16345785d8a0000", "difficulty": "0x0", "alloc": alloc or {}}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(g, f)
    return path


def _run(code: bytes, alloc: dict | None = None) -> tuple[bytes, int]:
    gp = _genesis(alloc)
    try:
        proc = subprocess.run(
            [EVM_BIN, "run", "--statdump", "--prestate", gp, "--codefile", "-"],
            input=code.hex(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        os.unlink(gp)
    combined = proc.stdout + proc.stderr
    if "error:" in combined:
        raise RuntimeError(f"evm error:\n{combined}")
    outs = EVM_OUTPUT_RE.findall(proc.stdout)
    gas = _GAS_RE.search(proc.stderr)
    out = bytes.fromhex(outs[-1][2:]) if outs and outs[-1] != "0x" else b""
    return out, int(gas.group(1)) if gas else -1


def _get_interpreter_runtime() -> str:
    """Compile the interpreter and return its deployed runtime bytecode hex."""
    creation = compile_interpreter_sync()["<stdin>:Interpreter"]["bin"]
    proc = subprocess.run(
        [EVM_BIN, "run", "--create", "--codefile", "-"],
        input=creation,
        capture_output=True,
        text=True,
        timeout=120,
    )
    runtimes = EVM_OUTPUT_RE.findall(proc.stdout)
    if not runtimes:
        raise RuntimeError(f"could not extract interpreter runtime:\n{proc.stdout}\n{proc.stderr}")
    return runtimes[-1][2:]


# ---------------------------------------------------------------------------
# Benchmark programs
# ---------------------------------------------------------------------------


def _prog_arith_small() -> bytes:
    p = Program()
    p.return_word(p.add(p.const(7), p.const(5)))
    return p.build()


def _prog_arith_chain(n: int) -> bytes:
    # disable_constant_folding so the chain stays as N runtime ADDs — this is
    # what makes the interpreter's per-opcode overhead visible (an all-constant
    # chain would otherwise fold to a single PUSH).
    p = Program()
    v = p.const(1)
    for i in range(n):
        v = p.add(v, p.const(i + 1))
    p.return_word(v)
    return p.build(disable_constant_folding=True)


def _prog_embed(size: int) -> bytes:
    p = Program()
    buf = p.embed_and_load(bytes((i % 256) for i in range(size)))
    p.return_(buf, 32)
    return p.build()


PROGRAMS = {
    "arith_small": _prog_arith_small(),
    "arith_chain_20": _prog_arith_chain(20),
    "arith_chain_50": _prog_arith_chain(50),
    "embed_32": _prog_embed(32),
    "embed_256": _prog_embed(256),
}


def run() -> None:
    if not Path(EVM_BIN).exists():
        print(f"evm binary not found at {EVM_BIN}; set PYDEFI_EVM_BIN (build from go-ethereum eip-7990-runcode).")
        return

    print("compiling Analog-Labs interpreter ...")
    runtime = _get_interpreter_runtime()
    alloc = {"0x" + INTERPRETER_ADDR.hex(): {"code": "0x" + runtime, "balance": "0x0"}}

    print(f"\n{'program':>16}  {'bytes':>6}  {'native':>8}  {'interp':>8}  {'runcode':>8}  {'interp/runcode':>15}")
    print("-" * 76)
    for name, bc in PROGRAMS.items():
        native_out, native_gas = _run(bc)  # program as account code directly
        interp_out, interp_gas = _run(interp_drive(bc), alloc)
        runcode_out, runcode_gas = _run(runcode_wrap(bc))
        assert interp_out == runcode_out == native_out, (
            f"{name}: output mismatch native={native_out.hex()} interp={interp_out.hex()} runcode={runcode_out.hex()}"
        )
        ratio = interp_gas / runcode_gas if runcode_gas else float("nan")
        print(f"{name:>16}  {len(bc):>6}  {native_gas:>8}  {interp_gas:>8}  {runcode_gas:>8}  {ratio:>14.1f}x")
    print(
        "\nnative  = program run directly as account code (floor)\n"
        "interp  = DELEGATECALL to Analog-Labs interpreter (DeFiVM / InterpreterRunner today)\n"
        "runcode = native in-context execution via RUNCODE 0xf6 (RuncodeRunner, EIP-7990)"
    )


if __name__ == "__main__":
    run()

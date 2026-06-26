"""Benchmark: :class:`pydefi.vm.Program` call_contract gas/size.

Measures gas and bytecode size across a range of calldata sizes
(4 – 2048 bytes).

Run with::

    python3 tests/bench_program_call.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexbytes import HexBytes  # noqa: E402

from pydefi.vm import Program  # noqa: E402
from tests.conftest import mini_evm  # noqa: E402

_TARGET: HexBytes = HexBytes("0x" + "de" * 20)


def _build_call(calldata: bytes) -> bytes:
    """SSA program that calls _TARGET with *calldata* and returns the success flag."""
    p = Program()
    success = p.call_raw(_TARGET, calldata)
    p.return_word(success)
    return p.build()


SIZES = [4, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def run() -> None:
    print(f"{'size':>6}  {'gas':>8}  {'bytes':>10}  {'bytes/calldata':>16}")
    print("-" * 60)
    for size in SIZES:
        data = bytes(i % 256 for i in range(size))
        code = _build_call(data)
        result = mini_evm(code)
        assert not result.is_error, f"SSA call failed at size={size}: {result.output.hex()}"
        ratio = len(code) / size
        print(f"{size:>6}  {result.gas_used:>8}  {len(code):>10}  {ratio:>15.2f}x")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def test_ssa_call_correctness() -> None:
    """A typical-sized calldata must compile and execute without error."""
    data = bytes(range(68))  # 4-byte selector + 64-byte args
    result = mini_evm(_build_call(data))
    assert not result.is_error


def test_ssa_call_bytecode_growth_is_constant() -> None:
    """Bytecode growth per added calldata byte should be ~1 (data section), not many."""
    small = _build_call(bytes(64))
    large = _build_call(bytes(1024))
    extra_calldata = 1024 - 64
    extra_bytes = len(large) - len(small)
    # The CALL fragment overhead is constant; each extra calldata byte adds one
    # byte to the data section (after 32-byte padding rounding).  Allow for a
    # small constant overhead and 32-byte rounding slack.
    assert extra_bytes <= extra_calldata + 64, (
        f"unexpected bytecode growth: +{extra_bytes} bytes for +{extra_calldata} calldata bytes"
    )


def test_bench_table_runs(capsys) -> None:
    run()
    out = capsys.readouterr().out
    for size in SIZES:
        assert str(size) in out


if __name__ == "__main__":
    run()

from __future__ import annotations

import re

# Genesis ``config`` activating every fork through Amsterdam (where RUNCODE /
# EIP-7990 is enabled) at time 0. Callers wrap it in a full genesis with their
# own ``gasLimit`` and ``alloc``.
AMSTERDAM_CHAIN_CONFIG = {
    "chainId": 1,
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0,
    "byzantiumBlock": 0,
    "constantinopleBlock": 0,
    "petersburgBlock": 0,
    "istanbulBlock": 0,
    "berlinBlock": 0,
    "londonBlock": 0,
    "mergeNetsplitBlock": 0,
    "terminalTotalDifficulty": 0,
    "shanghaiTime": 0,
    "cancunTime": 0,
    "pragueTime": 0,
    "osakaTime": 0,
    "amsterdamTime": 0,
    "blobSchedule": {
        "cancun": {"target": 3, "max": 6, "baseFeeUpdateFraction": 3338477},
        "prague": {"target": 6, "max": 9, "baseFeeUpdateFraction": 5007716},
        "osaka": {"target": 6, "max": 9, "baseFeeUpdateFraction": 5007716},
        "amsterdam": {"target": 6, "max": 9, "baseFeeUpdateFraction": 5007716},
    },
}

# Matches an ``evm run`` hex output line; the last one is the return data.
EVM_OUTPUT_RE = re.compile(r"^0x[0-9a-fA-F]*$", re.MULTILINE)


def push2(value: int) -> bytes:
    """PUSH2 of a 16-bit constant (fixed width keeps prologue offsets stable)."""
    return bytes([0x61, (value >> 8) & 0xFF, value & 0xFF])


def runcode_wrap(program: bytes) -> bytes:
    """Wrap *program* in a RUNCODE prologue.

    The prologue CODECOPYs *program* to mem[0x40] and RUNCODEs it (no args, all
    remaining gas), returning the 32 bytes it left in mem[0:32]. PUSH2 lengths
    keep the prologue offset fixed for any program size.
    """
    off = 0x1E  # length of the prologue built below
    pro = (
        push2(len(program))
        + push2(off)
        + bytes([0x60, 0x40, 0x39])  # CODECOPY program -> mem[0x40]
        + bytes([0x60, 0x20, 0x60, 0x00])  # retLength, retOffset
        + bytes([0x60, 0x00, 0x60, 0x00])  # argsLength, argsOffset
        + push2(len(program))
        + bytes([0x60, 0x40])  # codeLength, codeOffset
        + bytes([0x5A, 0xF6, 0x50])  # GAS, RUNCODE, POP
        + bytes([0x60, 0x20, 0x60, 0x00, 0xF3])  # RETURN mem[0:32]
    )
    assert len(pro) == off, (len(pro), off)
    return pro + program

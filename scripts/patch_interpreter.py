"""Patch the Analog-Labs EVM interpreter to revert on storage-touching opcodes.

Applies a 5-slot patch (``SLOAD``/``SSTORE``/``CALLCODE``/``DELEGATECALL``/
``SELFDESTRUCT`` handler slots → existing disabled-handler template) to the
vendored upstream ``Constants.sol``.  With ``--emit`` regenerates
``pydefi/vm/PatchedInterpreterConstants.sol``.

Prints creation + runtime hex under explicit banners.  They are distinct
artifacts: creation code → ``UniversalFactory.create2``; runtime code is
what lives on-chain.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

from eth_utils import keccak

# Default location: the vendored snapshot of upstream `Constants.sol`.
# See ``vendor/evm-interpreter/README.md`` for the pinned commit + update
# procedure.  Override with a positional arg to point at a fresh upstream
# checkout (e.g. when bumping the vendor pin).
_DEFAULT_CONSTANTS = Path(__file__).resolve().parents[1] / "vendor" / "evm-interpreter" / "Constants.sol"

# 32-byte "disabled handler" template — exactly the form the interpreter
# already uses for reserved opcodes (0x1e, 0x1f, 0xfe, etc.).
#   5b   JUMPDEST   (landing pad — dispatch JUMPs here)
#   fe   INVALID    (revert, consumes remaining gas)
#   00 7b + 28 zeros  (unreachable padding to fill the 32-byte slot)
_DISABLED_HANDLER = bytes.fromhex("5bfe007b" + "00" * 28)
assert len(_DISABLED_HANDLER) == 32

# Forbidden opcodes per issue #138.  TSTORE (0x5d) intentionally NOT here —
# transient storage is tx-scoped and safe; composers rely on it.
_OPCODES_TO_DISABLE: dict[int, str] = {
    0x54: "SLOAD",
    0x55: "SSTORE",
    0xF2: "CALLCODE",
    0xF4: "DELEGATECALL",
    0xFF: "SELFDESTRUCT",
}

# Marker that identifies the start of the runtime body inside the creation code
# (the dispatch entry: ``JUMPDEST MSIZE ISZERO PUSH2 0x000e JUMPI ...``).
_RUNTIME_MARKER = bytes.fromhex("5b591561000e573610611fc057")

# CREATE2 deployer for deterministic-address deployments — same UniversalFactory
# Analog-Labs uses for the upstream interpreter.  Picking the same factory means
# any chain where Analog-Labs deployed the upstream interpreter has this factory
# present, so our patched interpreter can deploy on every chain via the same
# infrastructure.
_UNIVERSAL_FACTORY = bytes.fromhex("0000000000001C4Bf962dF86e38F0c10c7972C6E")

# Deterministic salt namespaced for pydefi.  Recompute on any breaking change
# (e.g. add/remove a disabled opcode) so the CREATE2 address shifts and stale
# deployments don't get confused with the new bytecode.
_SALT_PREIMAGE = b"pydefi.PatchedInterpreter.v1"


def compute_create2_address(deployer: bytes, salt: bytes, initcode: bytes) -> bytes:
    """EIP-1014 CREATE2 address: ``keccak256(0xff ++ deployer ++ salt ++ keccak(initcode))[12:]``."""
    initcode_hash = keccak(initcode)
    return keccak(b"\xff" + deployer + salt + initcode_hash)[12:]


def extract_creation_code(constants_sol: str) -> bytes:
    """Pull ``INTERPRETER_CREATION_CODE`` hex literal out of Constants.sol."""
    m = re.search(r'bytes constant INTERPRETER_CREATION_CODE = hex"([0-9a-fA-F]+)"', constants_sol)
    if not m:
        raise ValueError("INTERPRETER_CREATION_CODE constant not found in Constants.sol")
    return bytes.fromhex(m.group(1))


def patch(creation: bytes) -> tuple[bytes, dict[int, bytes]]:
    """Apply the 5-slot patch.  Returns (patched_creation, before_per_op).

    Raises if the runtime body cannot be located or if any target slot fails
    a sanity check (must begin with 0x5b JUMPDEST).
    """
    runtime_start = creation.find(_RUNTIME_MARKER)
    if runtime_start < 0:
        raise ValueError("Could not locate runtime body in creation code")

    out = bytearray(creation)
    before: dict[int, bytes] = {}
    for op in _OPCODES_TO_DISABLE:
        off = runtime_start + op * 32
        slot = bytes(out[off : off + 32])
        if not slot.startswith(b"\x5b"):
            raise ValueError(f"opcode 0x{op:02x} slot at 0x{off:x} doesn't start with 0x5b: {slot.hex()}")
        before[op] = slot
        out[off : off + 32] = _DISABLED_HANDLER
    return bytes(out), before


def _to_eip55_checksum(addr_bytes: bytes) -> str:
    """EIP-55 checksummed lowercase-hex address (no leading 0x)."""
    addr = addr_bytes.hex()
    h = keccak(addr.encode()).hex()
    return "".join(c.upper() if int(h[i], 16) >= 8 else c for i, c in enumerate(addr))


def _replace_once(source: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, source, count=1, flags=re.MULTILINE | re.DOTALL)
    if n != 1:
        raise ValueError(f"expected exactly one match for pattern: {pattern}")
    return out


def render_constants_sol(
    template: str,
    creation: bytes,
    salt: bytes,
    address_bytes: bytes,
    runtime_codehash: bytes,
    initcode_hash: bytes,
) -> str:
    """Render the Solidity constants file by updating an existing template file."""
    rendered = template

    rendered = _replace_once(
        rendered,
        r"bytes32\s+constant\s+PATCHED_INTERPRETER_CREATE2_SALT\s*=\s*\n\s*0x[0-9a-fA-F]{64};",
        f"bytes32 constant PATCHED_INTERPRETER_CREATE2_SALT =\n    0x{salt.hex()};",
    )
    rendered = _replace_once(
        rendered,
        r"address\s+constant\s+PATCHED_INTERPRETER_ADDRESS\s*=\s*0x[0-9a-fA-F]{40};",
        f"address constant PATCHED_INTERPRETER_ADDRESS = 0x{_to_eip55_checksum(address_bytes)};",
    )
    rendered = _replace_once(
        rendered,
        r"bytes32\s+constant\s+PATCHED_INTERPRETER_CODEHASH\s*=\s*\n\s*0x[0-9a-fA-F]{64};",
        f"bytes32 constant PATCHED_INTERPRETER_CODEHASH =\n    0x{runtime_codehash.hex()};",
    )
    rendered = _replace_once(
        rendered,
        r"bytes32\s+constant\s+PATCHED_INTERPRETER_INITCODE_HASH\s*=\s*\n\s*0x[0-9a-fA-F]{64};",
        f"bytes32 constant PATCHED_INTERPRETER_INITCODE_HASH =\n    0x{initcode_hash.hex()};",
    )
    rendered = _replace_once(
        rendered,
        r'bytes\s+constant\s+PATCHED_INTERPRETER_CREATION_CODE\s*=\s*hex"[0-9a-fA-F]+";',
        f'bytes constant PATCHED_INTERPRETER_CREATION_CODE = hex"{creation.hex()}";',
    )
    rendered = _replace_once(
        rendered,
        r"EIP-1014 CREATE2 salt = `keccak256\('[^']*'\)`\.",
        f"EIP-1014 CREATE2 salt = `keccak256('{_SALT_PREIMAGE.decode()}')`.",
    )

    # Regenerate the natspec offset table from `_OPCODES_TO_DISABLE` so the
    # comment stays in sync with the actual patch.  Without this, a change to
    # the disable list updates the bytecode silently while the doc rots.
    runtime_offset = creation.find(_RUNTIME_MARKER)
    offset_block = "\n".join(
        f" *        offset 0x{runtime_offset + op * 32:04x}  op 0x{op:02x}  {name}"
        for op, name in _OPCODES_TO_DISABLE.items()
    )
    rendered = _replace_once(
        rendered,
        r"(?: \*        offset 0x[0-9a-fA-F]+  op 0x[0-9a-fA-F]+  \w+\n)+",
        offset_block + "\n",
    )

    return rendered


def main() -> None:
    args = sys.argv[1:]
    emit = "--emit" in args
    args = [a for a in args if a != "--emit"]
    path = Path(args[0]) if args else _DEFAULT_CONSTANTS
    if not path.exists():
        sys.exit(f"Constants.sol not found at {path}. Pass an explicit path as first arg.")

    creation = extract_creation_code(path.read_text())
    patched, before = patch(creation)

    print(f"Source: {path}")
    print(f"Creation code length: {len(creation)} bytes (0x{len(creation):x})")
    print()
    print("Slots patched:")
    for op, name in _OPCODES_TO_DISABLE.items():
        runtime_start = creation.find(_RUNTIME_MARKER)
        off = runtime_start + op * 32
        print(f"  op=0x{op:02x} {name:12s}  creation off=0x{off:04x}")
        print(f"    before: {before[op].hex()}")
        print(f"    after : {_DISABLED_HANDLER.hex()}")

    diff_count = sum(1 for a, b in zip(creation, patched) if a != b)
    print()
    print(f"Bytes differing: {diff_count} of {len(creation)}")
    print(f"Original creation codehash (sha256): {hashlib.sha256(creation).hexdigest()}")
    print(f"Patched  creation codehash (sha256): {hashlib.sha256(patched).hexdigest()}")

    # Split runtime from creation wrapper.  The runtime body starts at the
    # dispatch entry marker; everything before is the constructor.
    runtime_offset = patched.find(_RUNTIME_MARKER)
    patched_runtime = patched[runtime_offset:]
    print(f"Patched runtime offset in creation: 0x{runtime_offset:x}")
    print(f"Patched runtime length:             {len(patched_runtime)} bytes")
    print(f"Patched  runtime  codehash (sha256): {hashlib.sha256(patched_runtime).hexdigest()}")

    # EIP-1014 / EVM-side constants for deterministic CREATE2 deployment.
    # NOTE: The "runtime codehash" Solidity sees via ``EXTCODEHASH`` is
    # ``keccak256(runtime_code)``, NOT ``keccak256(creation_code)``.  Both are
    # printed below; the distinction matters for cross-chain verification.
    salt = keccak(_SALT_PREIMAGE)
    initcode_hash = keccak(patched)  # for CREATE2 address derivation
    address = compute_create2_address(_UNIVERSAL_FACTORY, salt, patched)
    # The upstream interpreter's constructor mirrors `address(this)` into the
    # runtime at runtime offset 0x1fef (= 20-byte address slot kept for
    # parity with the now-dead address-check) before returning it.  Account
    # for this so ``EXTCODEHASH`` on a real CREATE2 deployment matches.
    _ADDR_FILL_OFFSET = 0x1FEF
    filled_runtime = bytearray(patched_runtime)
    filled_runtime[_ADDR_FILL_OFFSET : _ADDR_FILL_OFFSET + 20] = address
    runtime_codehash = keccak(bytes(filled_runtime))
    print()
    print("EIP-1014 / CREATE2 constants:")
    print(f"  CREATE2 deployer (UniversalFactory):  0x{_UNIVERSAL_FACTORY.hex()}")
    print(f"  CREATE2 salt preimage (keccak256 of): {_SALT_PREIMAGE.decode()!r}")
    print(f"  CREATE2 salt:                          0x{salt.hex()}")
    print(f"  Initcode keccak256 (for CREATE2):      0x{initcode_hash.hex()}")
    print(f"  Runtime  keccak256 (EXTCODEHASH):      0x{runtime_codehash.hex()}")
    print(f"  Deterministic deployed address:        0x{address.hex()}")

    if emit:
        out_path = Path(__file__).resolve().parents[1] / "pydefi" / "vm" / "PatchedInterpreterConstants.sol"
        out_path.write_text(
            render_constants_sol(
                template=out_path.read_text(),
                creation=patched,
                salt=salt,
                address_bytes=address,
                runtime_codehash=runtime_codehash,
                initcode_hash=initcode_hash,
            )
        )
        print()
        print(f"Wrote {out_path}")
        return

    print()
    print("=" * 78)
    print("PATCHED_INTERPRETER_CREATION_CODE  (use for CREATE2 / UniversalFactory deploy)")
    print("=" * 78)
    print(patched.hex())
    print()
    print("=" * 78)
    print("PATCHED_INTERPRETER_RUNTIME  (paste into pydefi/vm/PatchedInterpreter.sol)")
    print("=" * 78)
    print(patched_runtime.hex())
    print()
    print("Pass --emit to regenerate pydefi/vm/PatchedInterpreterConstants.sol from these values.")


if __name__ == "__main__":
    main()

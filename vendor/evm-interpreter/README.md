# vendor/evm-interpreter

Vendored snapshot of the upstream **Analog-Labs EVM interpreter** source,
plus the rationale for why pydefi ships a patched fork of it
(closes issue #138).

## Why pydefi ships a patched interpreter

`DeFiVM.execute(bytes)` runs caller-supplied bytecode via DELEGATECALL into
the Analog-Labs EVM interpreter. The program runs in DeFiVM's storage
context, so anyone can submit a program that does `SLOAD` / `SSTORE` /
`CALLCODE` / `DELEGATECALL` / `SELFDESTRUCT` against DeFiVM. Any subsequent
caller whose program reads persistent storage is reading attacker-
controlled bytes.

The fix is to make those five opcodes unreachable. Each opcode has a fixed
32-byte handler in the interpreter's dispatch table; pydefi ships a fork of
the interpreter with those five handlers overwritten by a revert stub. The
rejection is part of the EVM implementation itself — zero per-call gas, no
changes to `DeFiVM`, and any otherwise-valid program still runs.

## How the patch works

The Analog-Labs interpreter is a 256-slot dispatch table. Each opcode `op`
has a 32-byte handler at offset `op * 32` within the runtime body. The
interpreter already uses a "disabled handler" template
`5b fe 00 7b + 28 zeros` (JUMPDEST, INVALID, padding) for reserved opcodes
like `0x1e`, `0x1f`, `0xfe`.

We overwrite five additional slots with the same template:

| Op | Name | Runtime offset | Creation offset |
| --- | --- | --- | --- |
| 0x54 | SLOAD | 0x0a80 | 0x0ab8 |
| 0x55 | SSTORE | 0x0aa0 | 0x0ad8 |
| 0xF2 | CALLCODE | 0x1e40 | 0x1e78 |
| 0xF4 | DELEGATECALL | 0x1e80 | 0x1eb8 |
| 0xFF | SELFDESTRUCT | 0x1fe0 | 0x2018 |

Bytecode size is unchanged (5 × 32 = 160 bytes overwritten in place). All
other 251 handler offsets stay correct. The patch is mechanically generated
by `scripts/patch_interpreter.py` and applied to
`pydefi/vm/PatchedInterpreterConstants.sol` with `--emit`.

`TSTORE` / `TLOAD` intentionally **remain enabled**:

- Transient storage is tx-scoped (EIP-1153), so cross-user contamination is
  impossible by spec.
- Composers (CCTP / OFT / CCIP) use `tstore` in their Solidity entry
  points to pass bridged params to the program, which reads them via
  `tload`. Banning `TLOAD` would break that handoff.

## Deployment constants

Auto-generated into `pydefi/vm/PatchedInterpreterConstants.sol`:

| Constant | Value |
| --- | --- |
| `PATCHED_INTERPRETER_CREATE2_DEPLOYER` | `0x0000000000001C4Bf962dF86e38F0c10c7972C6E` (UniversalFactory — same as upstream) |
| `PATCHED_INTERPRETER_CREATE2_SALT` | `keccak256("pydefi.PatchedInterpreter.v1")` |
| `PATCHED_INTERPRETER_ADDRESS` | `0x64fE558B0F9a5dC18D4A36c85Ba99c3f222F7bde` |
| `PATCHED_INTERPRETER_CODEHASH` | `0xa02da76871a49f730d82341d2a5abd702d1bd95576d4f932402b50fd3773e860` |

Per-chain deploy:

```bash
python scripts/deploy_patched_interpreter.py --rpc <RPC> --pk $DEPLOYER_PK
# or --dry-run to simulate without sending a tx
```

The script verifies the resulting address matches the constant and refuses
to overwrite a colliding deployment. Idempotent — safe to re-run.

## Runtime cost

The patch adds no gas to the interpreter's hot path. The dispatch loop is
untouched and each of the five replaced handlers is a `JUMPDEST; INVALID`
stub — strictly cheaper than the handler it replaces. A forbidden opcode
reverts immediately; every other opcode runs exactly as upstream.

## Safety

`DeFiVM` and `InterpreterRunner` constructors revert with
`InterpreterNotDeployed(address)` if the resolved interpreter address has
no code. Catches the "deployed DeFiVM before deploying the interpreter"
foot-gun at deploy time; without this guard, DELEGATECALL to an empty
address would silently succeed (returning empty returndata) and `execute()`
would look like it ran when in fact nothing happened.

Test fork fixtures additionally validate that the deployed code's
`EXTCODEHASH` matches `PATCHED_INTERPRETER_CODEHASH` before reusing any
pre-deployed contract — so a chain that happens to have unrelated code at
the deterministic CREATE2 address doesn't silently get used as the
interpreter.

## What's in this directory

| File | Source |
| ---- | ------ |
| `Constants.sol` | https://github.com/Analog-Labs/evm-interpreter/blob/main/src/utils/Constants.sol |
| `README.md` | this file |

## Pinned upstream version

```
75df08b2e92510e2812c67fe6df79d4dd5e57806
2024-12-01 21:30:32 -0300
"Added Constants.sol to make new deployments easier"
```

The `INTERPRETER_CREATION_CODE` constant inside is the only thing pydefi
reads. Everything else (Analog-Labs's own salt, deployed address, codehash)
is here for cross-check.

## How to update on an upstream rev

1. Pull the latest `Constants.sol` from
   `https://raw.githubusercontent.com/Analog-Labs/evm-interpreter/main/src/utils/Constants.sol`
   and overwrite the local copy.
2. Update the "Pinned upstream version" block above with the new commit
   hash + date.
3. **Bump the salt preimage version** (`pydefi.PatchedInterpreter.v1` →
   `v2`) in `scripts/patch_interpreter.py` — different upstream code = a
   different patched contract = must deploy at a fresh CREATE2 address.
4. Run `python scripts/patch_interpreter.py --emit` to regenerate
   `pydefi/vm/PatchedInterpreterConstants.sol`. `--emit` rewrites that file
   in place via targeted regex replacements, so the file must exist and
   keep its declared-constant shape — do not delete it between revs. To
   bootstrap from scratch, run without `--emit` and paste the printed
   creation/runtime hex back into the file by hand.
5. Run the full fork suite:
   `pytest -m fork tests/live/test_patched_interpreter.py tests/live/test_defi_vm_fork.py tests/live/test_*composer_fork.py`.
6. Deploy the new patched interpreter on each target chain via
   `scripts/deploy_patched_interpreter.py` and commit
   `vendor/evm-interpreter/Constants.sol`,
   `pydefi/vm/PatchedInterpreterConstants.sol`, and the salt bump
   together as one atomic change.

## License

Upstream source is MIT-licensed; the SPDX header at the top of
`Constants.sol` is preserved. Vendoring under MIT terms requires no
additional notices.

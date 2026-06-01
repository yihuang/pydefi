// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./ProgramExecutor.sol";

/**
 * @title RuncodeRunner
 * @notice :class:`ProgramExecutor` backend that runs a program natively in
 *         this contract's own context via the EIP-7990 ``RUNCODE`` (0xF6)
 *         opcode — no interpreter deployment, no DELEGATECALL, no cold-account
 *         access, and no per-opcode interpretation overhead.
 *
 *         To switch any pydefi contract from the interpreter to native
 *         execution, change its inheritance from ``InterpreterRunner`` to
 *         ``RuncodeRunner`` and drop the ``_interpreter`` constructor argument.
 *         Nothing else changes: ``RUNCODE`` shares this contract's storage,
 *         transient storage, balance, address, ``msg.sender`` and ``msg.value``
 *         (only the EVM stack and the program's calldata are isolated), so the
 *         composers' TSTORE/TLOAD param-passing convention keeps working as-is.
 *
 *  ┌──────────────────────────────────────────────────────────────────────┐
 *  │ ACTIVATION GATE — does NOT compile under a 7990-unaware toolchain.     │
 *  │ EIP-7990 is Draft (opcode 0xF6, base 100 gas) as of 2026-04 and no     │
 *  │ released solc/Yul exposes the ``runcode`` builtin yet.  This file is   │
 *  │ the reference backend for the migration; keep it out of the build      │
 *  │ until a 7990-aware compiler + target chain exist.                      │
 *  └──────────────────────────────────────────────────────────────────────┘
 *
 * EIP-7990 RUNCODE stack interface (top-to-bottom):
 *   gas, codeOffset, codeLength, argsOffset, argsLength, retOffset, retLength
 *   → pushes 1 on success, 0 on failure.  RETURN data is copied to
 *     [retOffset, retOffset+retLength) and remains queryable via
 *     RETURNDATASIZE / RETURNDATACOPY.
 */
abstract contract RuncodeRunner is ProgramExecutor {
    /**
     * @dev Copy ``program`` into memory and execute it in-context with
     *      ``RUNCODE``.  We pass no args (the program reads runtime params via
     *      TLOAD, exactly as under the interpreter backend) and capture the
     *      RETURN data via the EVM return buffer rather than a fixed retOffset.
     *      On failure the program's revert data is bubbled up unchanged.
     */
    function _runProgram(bytes calldata program) internal override {
        assembly {
            let codePtr := mload(0x40)
            calldatacopy(codePtr, program.offset, program.length)

            // success := RUNCODE(gas, codeOffset, codeLength,
            //                    argsOffset, argsLength, retOffset, retLength)
            // No args (argsLength = 0); retLength = 0 → read RETURN via
            // returndatacopy/returndatasize below, mirroring the interpreter
            // backend's behaviour.
            //
            // `runcode` is the EIP-7990 builtin a 7990-aware solc/Yul will
            // expose, analogous to `delegatecall`.  Until then this line is
            // the activation point — see the header gate.
            let success := runcode(gas(), codePtr, program.length, 0, 0, 0, 0)

            if iszero(success) {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
    }

    /// @dev Memory variant: RUNCODE the in-memory ``program`` bytes directly.
    ///      No copy needed — the bytes are already in memory, so ``codeOffset``
    ///      points straight at them (cheaper than the calldata path).
    function _runProgramMemory(bytes memory program) internal override {
        assembly {
            let success := runcode(gas(), add(program, 32), mload(program), 0, 0, 0, 0)
            if iszero(success) {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ProgramExecutor
 * @notice The single seam through which every pydefi contract runs a compiled
 *         DeFiVM program.  ``DeFiVM``, the bridge composers (CCTP/CCIP/OFT/
 *         Eureka) and ``ApproveProxy`` all execute programs by calling
 *         ``_runProgram``; this abstract base lets the *how* be swapped out
 *         without touching any of them.
 *
 * Backends
 * --------
 *  * :class:`InterpreterRunner` — DELEGATECALLs a pre-deployed EVM interpreter
 *    (Analog-Labs by default) with the program as calldata.  Works on every
 *    EVM chain today, at the cost of interpreting each program opcode in EVM
 *    and one cold-account access per call.
 *  * :class:`RuncodeRunner` — runs the program natively in-context via the
 *    EIP-7990 ``RUNCODE`` (0xF6) opcode.  No interpreter deployment, no cold
 *    access, no interpretation overhead.  Activates once a 7990-aware chain +
 *    toolchain exists (the opcode is Draft as of 2026-04).
 *
 * Switching backend is one edit per contract: change the inherited mixin
 * (and drop/keep the interpreter constructor arg accordingly).  Because both
 * backends honour the same context guarantees, nothing else changes:
 *
 *   • ``address(this)`` inside the program == the running contract.
 *   • External CALLs originate from the running contract.
 *   • SLOAD/SSTORE and TLOAD/TSTORE hit the running contract's namespaces, so
 *     the composers' transient-slot param-passing convention is preserved.
 *
 * Both backends bubble the program's revert data unchanged on failure and
 * leave the program's RETURN data in the EVM return buffer on success
 * (readable by the caller via ``returndatacopy`` / ``returndatasize``).
 */
abstract contract ProgramExecutor {
    /**
     * @dev Execute ``program`` (raw, already-compiled EVM bytecode) in this
     *      contract's own execution context.  Reverts, bubbling the program's
     *      returndata, if the program reverts.
     *
     * @param program Raw EVM bytecode produced by ``pydefi.vm`` (Venom IR →
     *                EVM bytecode via ``ProgramContext.build()``).
     */
    function _runProgram(bytes calldata program) internal virtual;

    /**
     * @dev Memory-resident variant of :func:`_runProgram`, for callers whose
     *      program is already in memory (e.g. loaded from storage) rather than
     *      calldata — see ``EurekaComposer``. Same context guarantees and
     *      revert behaviour; kept as a separate name because Solidity cannot
     *      overload on data location alone.
     *
     * @param program Raw EVM bytecode in memory.
     */
    function _runProgramMemory(bytes memory program) internal virtual;
}
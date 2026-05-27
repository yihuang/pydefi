// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {PATCHED_INTERPRETER_ADDRESS} from "./PatchedInterpreterConstants.sol";

/**
 * @title InterpreterRunner
 * @notice Mixin that DELEGATECALLs an EVM interpreter to run a program in
 *         the inheriting contract's own context.  ``address(0)`` in the
 *         constructor resolves to the pydefi-patched Analog-Labs interpreter
 *         deployed at the deterministic CREATE2 address
 *         :data:`PATCHED_INTERPRETER_ADDRESS`
 *         (see ``PatchedInterpreterConstants.sol``).
 *         The program is copied into free memory (``mload(0x40)``) so
 *         Solidity's memory pointers survive for code that runs after
 *         ``_runProgram`` returns.
 *
 * @dev The patched interpreter rejects ``SLOAD`` / ``SSTORE`` / ``CALLCODE``
 *      / ``DELEGATECALL`` / ``SELFDESTRUCT`` at runtime (these opcodes' 32-byte
 *      handler slots are replaced with the interpreter's own ``5b fe 00 7b ...``
 *      disabled template).  This closes issue #138 with zero per-call gas
 *      overhead.  Pass an explicit upstream-Analog-Labs address
 *      to opt out and run against the unpatched interpreter (e.g. test
 *      environments).
 */
abstract contract InterpreterRunner {
    /// @notice EVM interpreter that executes programs via DELEGATECALL.
    address public immutable interpreter;

    /// @dev Thrown when the resolved interpreter address has no code at
    ///      construction time.  Without this guard, a DELEGATECALL to a
    ///      code-less address would silently succeed (returning empty
    ///      returndata), making ``_runProgram`` look like it executed
    ///      the program when in fact nothing happened.
    error InterpreterNotDeployed(address interpreter);

    /// @param _interpreter EVM interpreter to DELEGATECALL.  Pass
    ///                     ``address(0)`` to use the pydefi-patched
    ///                     interpreter at
    ///                     :data:`PATCHED_INTERPRETER_ADDRESS`.  Reverts
    ///                     with :data:`InterpreterNotDeployed` if the
    ///                     resolved address has no code on this chain
    ///                     (catches "deployed DeFiVM before deploying the
    ///                     interpreter" foot-guns at deploy time, not at
    ///                     execute() time).
    constructor(address _interpreter) {
        address resolved = _interpreter == address(0) ? PATCHED_INTERPRETER_ADDRESS : _interpreter;
        uint256 size;
        assembly { size := extcodesize(resolved) }
        if (size == 0) revert InterpreterNotDeployed(resolved);
        interpreter = resolved;
    }

    /**
     * @dev DELEGATECALL the interpreter with ``program`` as calldata.  The
     *      program runs in this contract's context.  On revert, returndata
     *      is bubbled up unchanged.
     */
    function _runProgram(bytes calldata program) internal {
        address _interpreter = interpreter;
        assembly {
            let ptr := mload(0x40)
            calldatacopy(ptr, program.offset, program.length)
            let success := delegatecall(gas(), _interpreter, ptr, program.length, 0, 0)
            if iszero(success) {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./ProgramExecutor.sol";

/**
 * @title InterpreterRunner
 * @notice :class:`ProgramExecutor` backend that DELEGATECALLs an EVM
 *         interpreter to run a program in the inheriting contract's own
 *         context.  ``address(0)`` in the constructor resolves to the
 *         well-known Analog-Labs deployment.  The program is copied into
 *         free memory (``mload(0x40)``) so Solidity's memory pointers
 *         survive for code that runs after ``_runProgram`` returns.
 *
 *         This is the default, works-everywhere-today backend.  See
 *         :class:`RuncodeRunner` for the native EIP-7990 alternative that
 *         removes the interpreter dependency entirely.
 */
abstract contract InterpreterRunner is ProgramExecutor {
    /// @dev Well-known Analog-Labs EVM interpreter
    /// (https://github.com/Analog-Labs/evm-interpreter).
    address private constant DEFAULT_INTERPRETER = 0x0000000000001e3F4F615cd5e20c681Cf7d85e8D;

    /// @notice EVM interpreter that executes programs via DELEGATECALL.
    address public immutable interpreter;

    /// @param _interpreter EVM interpreter to DELEGATECALL.  Pass
    ///                     ``address(0)`` to use the pre-deployed Analog-Labs
    ///                     interpreter.
    constructor(address _interpreter) {
        interpreter = _interpreter == address(0) ? DEFAULT_INTERPRETER : _interpreter;
    }

    /**
     * @dev DELEGATECALL the interpreter with ``program`` as calldata.  The
     *      program runs in this contract's context.  On revert, returndata
     *      is bubbled up unchanged.
     */
    function _runProgram(bytes calldata program) internal override {
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

    /// @dev Memory variant: DELEGATECALL the interpreter with the in-memory
    ///      ``program`` bytes directly (no copy needed — it is already in
    ///      memory). On revert, returndata is bubbled up unchanged.
    function _runProgramMemory(bytes memory program) internal override {
        address _interpreter = interpreter;
        assembly {
            let success := delegatecall(gas(), _interpreter, add(program, 32), mload(program), 0, 0)
            if iszero(success) {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
    }
}

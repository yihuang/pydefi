// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title InterpreterRunner
 * @notice Mixin that DELEGATECALLs an EVM interpreter to run a program in
 *         the inheriting contract's own context.  ``address(0)`` in the
 *         constructor resolves to the well-known Analog-Labs deployment.
 *         The program is copied into free memory (``mload(0x40)``) so
 *         Solidity's memory pointers survive for code that runs after
 *         ``_runProgram`` returns.
 */
abstract contract InterpreterRunner {
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

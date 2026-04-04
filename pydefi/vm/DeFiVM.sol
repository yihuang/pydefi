// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title DeFiVM
 * @notice A minimal, stateless executor for composable DeFi flows expressed as raw EVM bytecode.
 *
 * Design principles
 * -----------------
 *  * Atomic execution  - a "program" runs all-at-once; any revert undoes everything.
 *  * Native EVM stack  - programs ARE raw EVM bytecode; the EVM stack is the program stack.
 *  * Fully stateless   - no owner, no whitelist; execute() can run any bytecode.
 *
 * How it works
 * ------------
 *  ``execute()`` delegates execution to the pre-deployed Analog-Labs EVM interpreter
 *  (https://github.com/Analog-Labs/evm-interpreter) via ``DELEGATECALL``.  The interpreter
 *  accepts the program bytecode as calldata and executes it in DeFiVM's context:
 *  - ``address(this)`` inside the program is DeFiVM's address.
 *  - External ``CALL``s originate from DeFiVM (msg.sender to sub-calls is DeFiVM).
 *  - ETH held by DeFiVM is forwarded via ``callvalue()`` and available to ``CALL``.
 *  - No contract deployment (CREATE) required — no nonce increase.
 *
 * Memory conventions (inside programs)
 * -------------------------------------
 *  Programs execute in a fresh virtual memory context provided by the interpreter.
 *  Recommended layout (used by the Python DSL):
 *  Registers  : memory[0x80 + i*32] for i in 0..15  (16 x 32-byte slots)
 *  Free memory: memory[0x40] tracks the next free byte (initialised to 0x280 on first use)
 *  Dynamic buf: allocated starting from memory[0x280] upward
 *
 * Security assumptions
 * --------------------
 *  1. Never approve tokens directly to this contract.  Approvals can be drained by
 *     any caller because ``execute`` is permissionless.  Use permit signatures instead.
 *  2. Do not leave token or ETH balances in this contract between transactions.
 *  3. Verify every address in a program and simulate off-chain before broadcasting.
 *  4. Programs run via DELEGATECALL and have full access to DeFiVM's storage.
 */
contract DeFiVM {
    /// @dev Well-known Analog-Labs EVM interpreter.
    address private constant DEFAULT_INTERPRETER = 0x0000000000001e3F4F615cd5e20c681Cf7d85e8D;

    /// @dev Address of the EVM interpreter used for DELEGATECALL execution.
    address private immutable INTERPRETER;

    /// @param interpreter Address of the EVM interpreter to use.  Pass
    ///   ``address(0)`` to use the pre-deployed Analog-Labs interpreter at
    ///   ``0x0000000000001e3F4F615cd5e20c681Cf7d85e8D``.  Supply a custom
    ///   address for alternative chains or local test environments where the
    ///   interpreter may not be pre-deployed.
    constructor(address interpreter) {
        INTERPRETER = interpreter == address(0) ? DEFAULT_INTERPRETER : interpreter;
    }

    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically.
     * @param program  Raw EVM bytecode to execute.
     *
     * Delegates to a pre-deployed EVM interpreter via DELEGATECALL so the program
     * runs in DeFiVM's execution context.  Every opcode executes on the native EVM
     * stack — no emulation overhead.  Any revert undoes all side-effects.
     */
    function execute(bytes calldata program) external payable {
        address interpreter = INTERPRETER;
        assembly {
            // Copy the program bytecode to the start of memory so it can be
            // passed as calldata to the interpreter via delegatecall.
            // Inside the interpreted program, CALLDATACOPY reads from these
            // bytes, enabling PC-relative inline data loads.
            calldatacopy(0, program.offset, program.length)
            let ok := delegatecall(gas(), interpreter, 0, program.length, 0, 0)
            returndatacopy(0, 0, returndatasize())
            if iszero(ok) {
                revert(0, returndatasize())
            }
        }
    }
}

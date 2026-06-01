// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./DEXCallbackRouter.sol";
import "./InterpreterRunner.sol";

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
 *  ``execute()`` runs the program through its :class:`ProgramExecutor` backend
 *  (``InterpreterRunner`` by default — DELEGATECALL to the pre-deployed
 *  Analog-Labs EVM interpreter, https://github.com/Analog-Labs/evm-interpreter;
 *  swap in ``RuncodeRunner`` for native EIP-7990 execution).  The program runs
 *  in DeFiVM's context:
 *  - ``address(this)`` inside the program is DeFiVM's address.
 *  - External ``CALL``s originate from DeFiVM (msg.sender to sub-calls is DeFiVM).
 *  - ETH held by DeFiVM is forwarded via ``callvalue()`` and available to ``CALL``.
 *  - No contract deployment (CREATE) required — no nonce increase.
 *
 * Memory conventions (inside programs)
 * -------------------------------------
 *  Programs execute in a fresh virtual memory context provided by the interpreter.
 *  The Python DSL (``pydefi.vm``) compiles programs through Venom IR,
 *  which allocates and packs all memory buffers automatically via ``alloca``;
 *  the DSL does not hand-manage ``memory[0x40]`` or use fixed register slots.
 *
 * Security assumptions
 * --------------------
 *  1. Never approve tokens directly to this contract.  Approvals can be drained by
 *     any caller because ``execute`` is permissionless.  Use ``ApproveProxy``
 *     (see ``ApproveProxy.sol``) or permit signatures instead.
 *  2. Do not leave token or ETH balances in this contract between transactions.
 *  3. Verify every address in a program and simulate off-chain before broadcasting.
 *  4. Programs run via DELEGATECALL and have full access to DeFiVM's storage.
 *
 * Flash-swap callbacks
 * --------------------
 *  Inherits :class:`DEXCallbackRouter`, which provides a uniform ``fallback()``
 *  selector dispatch for V2/V3-style swap callbacks (Uniswap V2/V3, Algebra,
 *  PancakeSwap V3, Solidly V3, Aerodrome/Velodrome, Ramses V2).  See
 *  ``DEXCallbackRouter.sol`` for the encoding conventions of the ``data``
 *  parameter and the supported selectors.
 */
contract DeFiVM is DEXCallbackRouter, InterpreterRunner {
    /// @param interpreter Address of the EVM interpreter to use.  Pass
    ///   ``address(0)`` to use the pre-deployed Analog-Labs interpreter at
    ///   ``0x0000000000001e3F4F615cd5e20c681Cf7d85e8D``.  Supply a custom
    ///   address for alternative chains or local test environments where the
    ///   interpreter may not be pre-deployed.  To run programs natively via
    ///   EIP-7990 instead, inherit ``RuncodeRunner`` and drop this argument.
    constructor(address interpreter) InterpreterRunner(interpreter) {}

    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically through the configured
     *         :class:`ProgramExecutor` backend.  Any revert undoes all
     *         side-effects; the program's RETURN data is bubbled to the caller.
     * @param program  Raw EVM bytecode to execute.
     *
     * Programs that need runtime parameters read them via ``TLOAD(i)`` from
     * the caller's transient slots.  Composers (CCTP/OFT/ApproveProxy) avoid
     * this entry point and run the program in their own context directly — so
     * the composer can write its bridged params via ``TSTORE`` and the program
     * reads them via ``TLOAD`` from the same transient namespace.
     */
    function execute(bytes calldata program) external payable {
        _runProgram(program);
        // _runProgram bubbles revert data on failure; on success the program's
        // RETURN data is still in the EVM return buffer — surface it to the caller.
        assembly {
            returndatacopy(0, 0, returndatasize())
            return(0, returndatasize())
        }
    }
}

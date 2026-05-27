// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./DEXCallbackRouter.sol";
import {PATCHED_INTERPRETER_ADDRESS} from "./PatchedInterpreterConstants.sol";

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
 *  ``execute()`` delegates execution to the pydefi-patched Analog-Labs EVM
 *  interpreter via ``DELEGATECALL``.  The interpreter accepts the program
 *  bytecode as calldata and executes it in DeFiVM's context:
 *  - ``address(this)`` inside the program is DeFiVM's address.
 *  - External ``CALL``s originate from DeFiVM (msg.sender to sub-calls is DeFiVM).
 *  - ETH held by DeFiVM is forwarded via ``callvalue()`` and available to ``CALL``.
 *  - No contract deployment (CREATE) required — no nonce increase.
 *
 *  The patched interpreter (see ``PatchedInterpreterConstants.sol``)
 *  rejects ``SLOAD`` / ``SSTORE`` / ``CALLCODE`` / ``DELEGATECALL`` /
 *  ``SELFDESTRUCT`` at runtime — closes issue #138 with zero per-call gas
 *  overhead.  Pass an explicit upstream Analog-Labs address to the
 *  constructor to run against the unpatched interpreter (testing / fallback).
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
contract DeFiVM is DEXCallbackRouter {
    /// @dev Address of the EVM interpreter used for DELEGATECALL execution.
    address private immutable INTERPRETER;

    /// @dev Thrown when the resolved interpreter address has no code at
    ///      construction time.  Without this guard, a DELEGATECALL to a
    ///      code-less address would silently succeed (returning empty
    ///      returndata), making ``execute`` look like it ran the program
    ///      when nothing happened.
    error InterpreterNotDeployed(address interpreter);

    /// @param interpreter Address of the EVM interpreter to use.  Pass
    ///   ``address(0)`` to use the pydefi-patched interpreter at
    ///   :data:`PATCHED_INTERPRETER_ADDRESS` (storage-opcode-rejecting variant
    ///   from ``PatchedInterpreterConstants.sol``).  Supply a custom address
    ///   for local test environments or to fall back to the upstream
    ///   Analog-Labs interpreter explicitly.  Construction reverts if the
    ///   resolved address has no code.
    constructor(address interpreter) {
        address resolved = interpreter == address(0) ? PATCHED_INTERPRETER_ADDRESS : interpreter;
        uint256 size;
        assembly { size := extcodesize(resolved) }
        if (size == 0) revert InterpreterNotDeployed(resolved);
        INTERPRETER = resolved;
    }

    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically via DELEGATECALL to a
     *         pre-deployed EVM interpreter.  Any revert undoes all side-effects.
     * @param program  Raw EVM bytecode to execute.
     *
     * Programs that need runtime parameters read them via ``TLOAD(i)`` from
     * the caller's transient slots.  Composers (CCTP/OFT/ApproveProxy) avoid
     * this entry point and DELEGATECALL the interpreter directly so the
     * program runs in the composer's own context — that way the composer
     * can write its bridged params via ``TSTORE`` and the program reads
     * them via ``TLOAD`` from the same transient namespace.
     */
    function execute(bytes calldata program) external payable {
        address interpreter = INTERPRETER;
        assembly {
            calldatacopy(0, program.offset, program.length)
            let ok := delegatecall(gas(), interpreter, 0, program.length, 0, 0)
            returndatacopy(0, 0, returndatasize())
            if iszero(ok) {
                revert(0, returndatasize())
            }
            return(0, returndatasize())
        }
    }
}

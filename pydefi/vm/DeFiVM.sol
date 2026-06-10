// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./DEXCallbackRouter.sol";

/// @dev Minimal Permit2 SignatureTransfer surface used by ``executeWithPermit2``.
interface ISignatureTransfer {
    struct TokenPermissions {
        address token;
        uint256 amount;
    }

    struct PermitTransferFrom {
        TokenPermissions permitted;
        uint256 nonce;
        uint256 deadline;
    }

    struct SignatureTransferDetails {
        address to;
        uint256 requestedAmount;
    }

    function permitWitnessTransferFrom(
        PermitTransferFrom memory permit,
        SignatureTransferDetails calldata transferDetails,
        address owner,
        bytes32 witness,
        string calldata witnessTypeString,
        bytes calldata signature
    ) external;
}

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
 *  The Python DSL (``pydefi.vm``) compiles programs through Venom IR,
 *  which allocates and packs all memory buffers automatically via ``alloca``;
 *  the DSL does not hand-manage ``memory[0x40]`` or use fixed register slots.
 *
 * Security assumptions
 * --------------------
 *  1. Never approve tokens directly to this contract.  Approvals can be drained by
 *     any caller because ``execute`` is permissionless.  Use ``ApproveProxy``
 *     (see ``ApproveProxy.sol``) or ``executeWithPermit2`` instead.
 *  2. Do not leave token or ETH balances in this contract between transactions.
 *     Programs funded via ``executeWithPermit2`` must consume the pulled tokens
 *     within the same run.
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
    /// @dev Well-known Analog-Labs EVM interpreter.
    address private constant DEFAULT_INTERPRETER = 0x0000000000001e3F4F615cd5e20c681Cf7d85e8D;

    /// @dev Canonical Permit2 deployment (same address on every chain).
    ISignatureTransfer public constant PERMIT2 = ISignatureTransfer(0x000000000022D473030F116dDEE9F6B43aC78BA3);

    /// @dev Witness binding the Permit2 pull to the exact program that spends it.
    bytes32 private constant WITNESS_TYPEHASH = keccak256("Witness(bytes32 programHash)");
    string private constant WITNESS_TYPE_STRING =
        "Witness witness)TokenPermissions(address token,uint256 amount)Witness(bytes32 programHash)";

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
        _run(program);
    }

    /// @notice Pull tokens from ``owner`` via a Permit2 witness signature bound
    ///         to ``keccak256(program)``, then execute ``program`` atomically —
    ///         a relayer submits and pays gas but cannot alter what runs.
    ///         Permit2's unordered nonce gives replay protection.
    function executeWithPermit2(
        ISignatureTransfer.PermitTransferFrom calldata permit,
        address owner,
        bytes calldata signature,
        bytes calldata program
    ) external payable {
        PERMIT2.permitWitnessTransferFrom(
            permit,
            ISignatureTransfer.SignatureTransferDetails({to: address(this), requestedAmount: permit.permitted.amount}),
            owner,
            keccak256(abi.encode(WITNESS_TYPEHASH, keccak256(program))),
            WITNESS_TYPE_STRING,
            signature
        );
        _run(program);
    }

    /// @dev DELEGATECALL *program* to the interpreter, bubbling return/revert data.
    function _run(bytes calldata program) private {
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

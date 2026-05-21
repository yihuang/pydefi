// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./DEXCallbackRouter.sol";
import "./InterpreterRunner.sol";
import "./TransientReentrancyGuard.sol";

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

/**
 * @title ApproveProxy
 * @notice Proxy entry-point that lets users safely grant ERC-20 token allowances
 *         for DeFiVM-style programs without the security risks of approving the
 *         permissionless ``DeFiVM.execute`` directly.
 *
 * Problem
 * -------
 * ``DeFiVM.execute`` is permissionless — any caller can run any program.
 * Approving tokens directly to DeFiVM means *any* ``execute`` call can
 * drain those approvals.
 *
 * Solution
 * --------
 * Users approve ERC-20 tokens to *this* proxy and invoke programs through
 * ``ApproveProxy.execute()``.  The proxy:
 *
 * 1. Pulls the declared deposits from the caller into this proxy via
 *    ``token.transferFrom(msg.sender, address(this), amount)``.
 * 2. Stages caller-supplied params in this proxy's transient storage.
 * 3. DELEGATECALLs the EVM interpreter to execute the program in this
 *    proxy's context — programs operate on tokens already held here and
 *    read params via ``TLOAD(i)``.
 *
 * The ``execute`` entry is guarded by ``TransientReentrancyGuard`` — the
 * limitation documented there still applies: the program runs in this
 * proxy's context and can ``TSTORE`` the lock slot, so the guard is
 * defense-in-depth against external re-entry, not a sandbox against the
 * program itself.
 */
contract ApproveProxy is DEXCallbackRouter, TransientReentrancyGuard, InterpreterRunner {
    /// @dev Hard cap on transient-storage param slots.  ``execute`` reverts
    ///      if ``params.length`` exceeds this and zeros slots ``[0, MAX_PARAMS)``
    ///      after running the program so a same-tx ``execute`` with fewer
    ///      params cannot observe stale upper slots from an earlier call.
    uint256 private constant MAX_PARAMS = 32;

    /// @notice A single ERC-20 token deposit: token address + amount to pull
    ///         from the caller into this proxy before executing the program.
    struct Deposit {
        address token;
        uint256 amount;
    }

    /// @param _interpreter EVM interpreter to DELEGATECALL (see
    ///                     :class:`InterpreterRunner`); pass ``address(0)``
    ///                     for the well-known pre-deployed Analog-Labs
    ///                     interpreter.
    constructor(address _interpreter) InterpreterRunner(_interpreter) {}

    /// @notice Allow the proxy to receive ETH for value-bearing programs.
    receive() external payable {}

    /**
     * @notice Deposit ERC-20 tokens into this proxy and execute a program.
     *
     * For each entry in ``deposits``, this function calls
     * ``token.transferFrom(msg.sender, address(this), amount)``, moving the
     * tokens from the caller into this proxy.  ``params`` are written to
     * this proxy's transient slots ``[0, params.length)`` before the
     * program runs; the program reads them via ``TLOAD(i)``.  Slots are
     * cleared after execution.
     *
     * @param program  Raw EVM bytecode to execute.
     * @param deposits List of ERC-20 tokens and amounts to pull into this
     *                 proxy.  May be empty.
     * @param params   Runtime parameters staged in this proxy's transient
     *                 storage for the program.  May be empty.
     */
    function execute(
        bytes calldata program,
        Deposit[] calldata deposits,
        bytes32[] calldata params
    ) external payable nonReentrant {
        require(params.length <= MAX_PARAMS, "ApproveProxy: too many params");
        for (uint256 i = 0; i < deposits.length; i++) {
            _safeTransferFrom(deposits[i].token, msg.sender, address(this), deposits[i].amount);
        }

        // Stage caller params; cleanup spans [0, MAX_PARAMS) so a same-tx
        // ``execute`` with fewer params can't read stale upper slots from
        // an earlier call.
        assembly {
            let n := params.length
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                tstore(i, calldataload(add(params.offset, mul(i, 32))))
            }
        }
        _runProgram(program);
        assembly {
            for { let i := 0 } lt(i, MAX_PARAMS) { i := add(i, 1) } {
                tstore(i, 0)
            }
        }
    }

    /**
     * @dev Call ``token.transferFrom(from, to, amount)`` and revert if it fails.
     *      Supports both standard ERC-20s that return ``bool`` and widely-used
     *      non-compliant tokens that return no value.
     */
    function _safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        (bool success, bytes memory returndata) = token.call(
            abi.encodeWithSelector(IERC20.transferFrom.selector, from, to, amount)
        );
        require(success, "ApproveProxy: deposit failed");
        if (returndata.length > 0) {
            require(abi.decode(returndata, (bool)), "ApproveProxy: deposit failed");
        }
    }
}

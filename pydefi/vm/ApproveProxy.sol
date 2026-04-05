// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ApproveProxy
 * @notice A proxy entry-point that lets users safely grant ERC-20 token
 *         allowances for DeFiVM programs without the security risks of
 *         approving the permissionless DeFiVM executor directly.
 *
 * Problem
 * -------
 * ``DeFiVM.execute()`` is permissionless — any caller can run any program.
 * Approving tokens directly to DeFiVM means *any* ``execute()`` call can
 * drain those approvals.
 *
 * Solution
 * --------
 * Users approve ERC-20 tokens to *this* proxy and invoke DeFiVM programs
 * through ``ApproveProxy.execute()`` instead of ``DeFiVM.execute()``.  The
 * proxy records the current caller before forwarding to DeFiVM, and exposes
 * ``transferFrom()`` which programs can call to pull tokens.  Because
 * ``transferFrom()`` only succeeds when called by the paired DeFiVM *during*
 * an active ``ApproveProxy.execute()`` call, arbitrary ``vm.execute()``
 * callers cannot drain user approvals.
 *
 * Usage
 * -----
 * 1. User: ``token.approve(approveProxy, amount)``  — one-time or per-session.
 * 2. User: ``approveProxy.execute{value: v}(program)``  — run DeFiVM program.
 * 3. Program: calls ``approveProxy.transferFrom(token, recipient, amount)`` to
 *    pull tokens from the user who triggered step 2.
 */
contract ApproveProxy {
    /// @dev The DeFiVM instance this proxy is paired with.
    address public immutable vm;

    /// @dev Address of the user who initiated the current ``execute()`` call.
    ///      Zero when no execution is in progress; non-zero during execution.
    address private _currentSender;

    /// @param _vm Address of the paired DeFiVM contract.
    constructor(address _vm) {
        require(_vm != address(0), "ApproveProxy: zero vm address");
        vm = _vm;
    }

    /**
     * @notice Execute a DeFiVM program, making msg.sender available for token
     *         transfers inside the program via transferFrom().
     * @param program Raw EVM bytecode to execute (forwarded to DeFiVM).
     */
    function execute(bytes calldata program) external payable {
        require(_currentSender == address(0), "ApproveProxy: reentrant call");
        _currentSender = msg.sender;
        (bool ok, ) = vm.call{value: msg.value}(
            abi.encodeWithSignature("execute(bytes)", program)
        );
        _currentSender = address(0);
        if (!ok) {
            assembly {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
    }

    /**
     * @notice Transfer ERC-20 tokens from the current ``execute()`` caller to
     *         a recipient.  Must be called (indirectly) by the paired DeFiVM
     *         during an active ``ApproveProxy.execute()`` call.
     * @dev    Reverts if called outside of an active execution context or by
     *         any address other than the paired DeFiVM.
     * @param token     ERC-20 token contract address.
     * @param recipient Destination address for the tokens.
     * @param amount    Number of tokens to transfer.
     */
    function transferFrom(address token, address recipient, uint256 amount) external {
        require(msg.sender == vm, "ApproveProxy: caller is not the vm");
        address owner = _currentSender;
        require(owner != address(0), "ApproveProxy: no active sender");
        (bool ok, bytes memory ret) = token.call(
            abi.encodeWithSignature("transferFrom(address,address,uint256)", owner, recipient, amount)
        );
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "ApproveProxy: transfer failed");
    }

    /// @notice Allow the proxy to receive ETH (forwarded to DeFiVM).
    receive() external payable {}
}

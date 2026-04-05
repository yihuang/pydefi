// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IDeFiVM {
    function execute(bytes calldata program) external payable;
}

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
 * proxy pulls the declared token deposits from the caller into DeFiVM *before*
 * forwarding the program.  Programs then operate on tokens already held by
 * DeFiVM — no in-program ``transferFrom`` back-channel is required.
 *
 * Because the proxy only calls ``token.transferFrom(msg.sender, vm, amount)``
 * (user → VM, caller-controlled), there is no shared mutable state and no
 * reentrancy risk.
 *
 * Usage
 * -----
 * 1. User: ``token.approve(approveProxy, amount)``  — once per token/session.
 * 2. User: ``approveProxy.execute{value: v}(program, deposits)``
 *           where ``deposits`` lists the tokens and amounts to pull into DeFiVM.
 * 3. DeFiVM program operates on the deposited tokens (e.g. calls
 *    ``token.transfer(recipient, amount)`` from the VM's balance).
 */
contract ApproveProxy {
    /// @dev The DeFiVM instance this proxy is paired with.
    address public immutable vm;

    /// @notice A single ERC-20 token deposit: token address + amount to pull
    ///         from the caller into DeFiVM before executing the program.
    struct Deposit {
        address token;
        uint256 amount;
    }

    /// @param _vm Address of the paired DeFiVM contract.
    constructor(address _vm) {
        require(_vm != address(0), "ApproveProxy: zero vm address");
        vm = _vm;
    }

    /**
     * @notice Deposit ERC-20 tokens into DeFiVM and then execute a program.
     *
     * For each entry in ``deposits``, this function calls
     * ``token.transferFrom(msg.sender, vm, amount)``, moving the tokens from
     * the caller directly into the paired DeFiVM contract.  It then forwards
     * the program (and any ETH) to ``DeFiVM.execute()``.
     *
     * @param program  Raw EVM bytecode to execute (forwarded to DeFiVM).
     * @param deposits List of ERC-20 tokens and amounts to deposit into DeFiVM
     *                 before program execution.  May be empty.
     */
    function execute(bytes calldata program, Deposit[] calldata deposits) external payable {
        IDeFiVM _vm = IDeFiVM(vm);
        for (uint256 i = 0; i < deposits.length; i++) {
            _safeTransferFrom(deposits[i].token, msg.sender, address(_vm), deposits[i].amount);
        }
        _vm.execute{value: msg.value}(program);
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

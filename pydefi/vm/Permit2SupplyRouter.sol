// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title Permit2SupplyRouter
 * @notice Single-signature gasless deposits via Permit2.
 *
 * The owner signs one Permit2 witness transfer binding the action
 * (``witness = keccak256(protocol, keccak256(supplyData))``): the signature both
 * authorizes the pull and fixes the downstream call, so a front-runner can't
 * redirect it. Permit2's unordered nonce gives replay protection.
 *
 * Per deposit: Permit2 pull → ``protocol.call(supplyData)`` → sweep leftover to
 * owner. One-time setup: owner ``approve(PERMIT2)`` on the token, and
 * ``prime(token, protocol)`` to max-approve the protocol. Works for any
 * Permit2-supported token (incl. USDT).
 *
 * UNAUDITED — audit before mainnet.
 */

interface IERC20 {
    function approve(address spender, uint256 value) external returns (bool);
    function transfer(address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

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

contract Permit2SupplyRouter {
    ISignatureTransfer public constant PERMIT2 = ISignatureTransfer(0x000000000022D473030F116dDEE9F6B43aC78BA3);

    bytes32 private constant WITNESS_TYPEHASH = keccak256("Witness(address protocol,bytes32 supplyDataHash)");
    string private constant WITNESS_TYPE_STRING =
        "Witness witness)TokenPermissions(address token,uint256 amount)Witness(address protocol,bytes32 supplyDataHash)";

    error SupplyFailed();
    error ApproveFailed();
    error TransferFailed();

    /// @notice One-time max allowance for ``protocol`` to pull ``token`` (router holds no idle balance).
    function prime(address token, address protocol) external {
        _safeApprove(token, protocol, type(uint256).max);
    }

    /// @notice Pull via a witness sig bound to (``protocol``, ``keccak(supplyData)``),
    ///         call ``protocol`` with it, sweep any leftover back to ``owner``.
    function supply(
        ISignatureTransfer.PermitTransferFrom calldata permit,
        address owner,
        bytes calldata signature,
        address protocol,
        bytes calldata supplyData
    ) external {
        bytes32 witness = keccak256(abi.encode(WITNESS_TYPEHASH, protocol, keccak256(supplyData)));
        PERMIT2.permitWitnessTransferFrom(
            permit,
            ISignatureTransfer.SignatureTransferDetails({to: address(this), requestedAmount: permit.permitted.amount}),
            owner,
            witness,
            WITNESS_TYPE_STRING,
            signature
        );
        (bool ok,) = protocol.call(supplyData);
        if (!ok) revert SupplyFailed();
        uint256 leftover = IERC20(permit.permitted.token).balanceOf(address(this));
        if (leftover > 0) {
            _safeTransfer(permit.permitted.token, owner, leftover);
        }
    }

    /// @dev ERC-20 calls that tolerate non-standard tokens (e.g. USDT returns no bool).
    function _safeApprove(address token, address spender, uint256 value) private {
        (bool ok, bytes memory d) = token.call(abi.encodeCall(IERC20.approve, (spender, value)));
        if (!ok || (d.length != 0 && !abi.decode(d, (bool)))) revert ApproveFailed();
    }

    function _safeTransfer(address token, address to, uint256 value) private {
        (bool ok, bytes memory d) = token.call(abi.encodeCall(IERC20.transfer, (to, value)));
        if (!ok || (d.length != 0 && !abi.decode(d, (bool)))) revert TransferFailed();
    }
}

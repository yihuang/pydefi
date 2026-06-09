// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title EIP7702BatchExecutor
 * @notice Single-signature gasless deposits via EIP-7702.
 *
 * The owner delegates its EOA to this contract (one 7702 authorization, reusable
 * until revoked) and signs an EIP-712 ``Batch`` of calls. A sponsor submits the
 * type-4 transaction and pays gas; the batch runs in the EOA's own context, so
 * ``approve`` + ``supply`` credit the owner directly — no router intermediary,
 * no Permit2 pull, no ``onBehalfOf`` plumbing.
 *
 * Security: because the sponsor (not the EOA) sends the tx, ``execute`` is
 * callable by anyone, so it is signature-gated — ``ecrecover`` must return
 * ``address(this)`` (the delegated EOA signs for itself). ``batchNonce`` lives in
 * the EOA's own storage and gives replay protection.
 *
 * UNAUDITED — audit before mainnet.
 */
contract EIP7702BatchExecutor {
    struct Call {
        address to;
        uint256 value;
        bytes data;
    }

    bytes32 private constant DOMAIN_TYPEHASH =
        keccak256("EIP712Domain(string name,uint256 chainId,address verifyingContract)");
    bytes32 private constant CALL_TYPEHASH = keccak256("Call(address to,uint256 value,bytes data)");
    bytes32 private constant BATCH_TYPEHASH =
        keccak256("Batch(Call[] calls,uint256 nonce,uint256 deadline)Call(address to,uint256 value,bytes data)");
    bytes32 private constant NAME_HASH = keccak256("EIP7702BatchExecutor");

    /// @notice Per-account replay counter. Delegated code runs in the EOA's
    ///         context, so this slot lives in the EOA's own storage.
    uint256 public batchNonce;

    error Expired();
    error BadNonce();
    error BadSig();
    error CallFailed(uint256 index);

    /// @notice Run ``calls`` from the EOA's own context after verifying the EOA
    ///         signed exactly this batch. The signature binds (calls, nonce,
    ///         deadline), so a sponsor can submit but cannot alter it.
    function execute(Call[] calldata calls, uint256 nonce, uint256 deadline, bytes calldata signature) external {
        if (block.timestamp > deadline) revert Expired();
        if (nonce != batchNonce) revert BadNonce();
        if (_recover(_digest(calls, nonce, deadline), signature) != address(this)) revert BadSig();
        batchNonce = nonce + 1;
        for (uint256 i; i < calls.length; ++i) {
            (bool ok,) = calls[i].to.call{value: calls[i].value}(calls[i].data);
            if (!ok) revert CallFailed(i);
        }
    }

    function _digest(Call[] calldata calls, uint256 nonce, uint256 deadline) private view returns (bytes32) {
        bytes32[] memory hashes = new bytes32[](calls.length);
        for (uint256 i; i < calls.length; ++i) {
            hashes[i] =
                keccak256(abi.encode(CALL_TYPEHASH, calls[i].to, calls[i].value, keccak256(calls[i].data)));
        }
        bytes32 structHash =
            keccak256(abi.encode(BATCH_TYPEHASH, keccak256(abi.encodePacked(hashes)), nonce, deadline));
        bytes32 domainSeparator =
            keccak256(abi.encode(DOMAIN_TYPEHASH, NAME_HASH, block.chainid, address(this)));
        return keccak256(abi.encodePacked(hex"1901", domainSeparator, structHash));
    }

    /// @dev 65-byte ECDSA recover with EIP-2 low-``s`` enforcement. Accepts the
    ///      ``v`` recovery id as either 27/28 or 0/1 for broader wallet compatibility.
    function _recover(bytes32 digest, bytes calldata sig) private pure returns (address) {
        if (sig.length != 65) revert BadSig();
        bytes32 r = bytes32(sig[0:32]);
        bytes32 s = bytes32(sig[32:64]);
        uint8 v = uint8(sig[64]);
        if (v < 27) v += 27;
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) revert BadSig();
        address signer = ecrecover(digest, v, r, s);
        if (signer == address(0)) revert BadSig();
        return signer;
    }

    /// @notice Accept ETH so batched calls can forward native value.
    receive() external payable {}
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title OFTComposer
 * @notice LayerZero OFT compose receiver that executes a list of arbitrary
 *         on-chain calls after an OFT cross-chain token transfer arrives.
 *
 * How it works
 * ------------
 * 1. A sender on the source chain encodes a ``Call[]`` list as the
 *    ``composeMsg`` in their OFT ``send`` call.
 * 2. After the OFT tokens arrive on the destination chain, the LayerZero
 *    EndpointV2 contract calls ``lzCompose`` on this contract.
 * 3. ``lzCompose`` validates the caller (must be the authorised endpoint) and
 *    the originating OFT (must be in the approved list), then decodes the
 *    compose message and executes each ``Call`` sequentially.  Any revert in
 *    a sub-call reverts the entire compose execution.
 *
 * Security notes
 * --------------
 *  • Only the authorised LayerZero endpoint may call ``lzCompose``.
 *  • Only OFT contracts explicitly approved by the owner may trigger compose
 *    executions.  Approve with ``approveOFT``, revoke with ``revokeOFT``.
 *  • The compose payload encodes arbitrary calls; senders are responsible for
 *    constructing safe payloads.  Review each call's target and calldata
 *    off-chain before broadcasting.
 *  • Do not leave token or ETH balances in this contract between transactions;
 *    any residual balance is accessible to the next compose message.
 *
 * Compose-message encoding
 * ------------------------
 * The raw ``_message`` bytes that arrive in ``lzCompose`` use the standard
 * LayerZero ``OFTComposeMsgCodec`` layout::
 *
 *   | 8 bytes nonce | 4 bytes srcEid | 32 bytes amountLD | payload |
 *
 * The custom ``payload`` (everything after the first 44 bytes) must be
 * ABI-encoded as::
 *
 *   abi.encode(Call[] calls)
 *
 * where each ``Call`` is:
 *
 *   struct Call { address target; uint256 value; bytes data; }
 *
 * Python helper (eth_abi)::
 *
 *   import struct
 *   from eth_abi import encode
 *
 *   payload = encode(['(address,uint256,bytes)[]'], [calls])
 *   message = (
 *       struct.pack('>Q', nonce)        # 8 bytes  — uint64 nonce
 *       + struct.pack('>I', src_eid)    # 4 bytes  — uint32 srcEid
 *       + amount_ld.to_bytes(32, 'big') # 32 bytes — uint256 amountLD
 *       + payload                       # ABI-encoded Call[]
 *   )
 */

// ---------------------------------------------------------------------------
// Call struct
// ---------------------------------------------------------------------------

/// @notice A single external call to be executed inside ``lzCompose``.
struct Call {
    address target;
    uint256 value;
    bytes data;
}

// ---------------------------------------------------------------------------
// OFTComposer
// ---------------------------------------------------------------------------

contract OFTComposer {
    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// @notice Thrown when the caller is not the authorised LayerZero endpoint.
    error UnauthorizedEndpoint(address caller);

    /// @notice Thrown when the originating OFT is not in the approved list.
    error UnauthorizedOFT(address oft);

    /// @notice Thrown when a sub-call inside ``lzCompose`` reverts.
    error CallFailed(uint256 index, bytes reason);

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    /// @notice Emitted after a successful compose execution.
    event Composed(
        address indexed from,
        bytes32 indexed guid,
        uint256 amountLD,
        uint256 numCalls
    );

    /// @notice Emitted when an OFT contract is approved.
    event OFTApproved(address indexed oft);

    /// @notice Emitted when an OFT contract is revoked.
    event OFTRevoked(address indexed oft);

    /// @notice Emitted when ownership is transferred.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /// @notice The LayerZero v2 endpoint address authorised to call ``lzCompose``.
    address public immutable endpoint;

    /// @notice OFT contracts that are allowed to trigger compose executions.
    mapping(address => bool) public approvedOFTs;

    /// @notice Owner address — may approve or revoke OFT contracts.
    address public owner;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _endpoint  The LayerZero v2 EndpointV2 contract address.
     * @param _owner     Address that may call ``approveOFT`` / ``revokeOFT``.
     */
    constructor(address _endpoint, address _owner) {
        endpoint = _endpoint;
        owner = _owner;
    }

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "OFTComposer: not owner");
        _;
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    /// @notice Approve an OFT contract to trigger compose executions.
    function approveOFT(address _oft) external onlyOwner {
        approvedOFTs[_oft] = true;
        emit OFTApproved(_oft);
    }

    /// @notice Revoke an OFT contract from triggering compose executions.
    function revokeOFT(address _oft) external onlyOwner {
        approvedOFTs[_oft] = false;
        emit OFTRevoked(_oft);
    }

    /// @notice Transfer ownership to a new address.
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "OFTComposer: zero address");
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }

    // -----------------------------------------------------------------------
    // ILayerZeroComposer
    // -----------------------------------------------------------------------

    /**
     * @notice Receive and execute an OFT compose message from the LayerZero
     *         EndpointV2 contract.
     *
     * @param _from     The OFT contract on this chain that received the tokens.
     * @param _guid     Unique LayerZero message GUID.
     * @param _message  ``OFTComposeMsgCodec``-encoded message:
     *                  ``| 8B nonce | 4B srcEid | 32B amountLD | payload |``
     *                  where ``payload = abi.encode(Call[] calls)``.
     */
    function lzCompose(
        address _from,
        bytes32 _guid,
        bytes calldata _message,
        address /* _executor */,
        bytes calldata /* _extraData */
    ) external payable {
        // Only the authorised endpoint may call this function.
        if (msg.sender != endpoint) revert UnauthorizedEndpoint(msg.sender);

        // The originating OFT must be in the approved list.
        if (!approvedOFTs[_from]) revert UnauthorizedOFT(_from);

        // Validate minimum message length: 8B nonce + 4B srcEid + 32B amountLD = 44 bytes.
        require(_message.length >= 44, "OFTComposer: message too short");

        // Decode OFTComposeMsgCodec layout:
        //   bytes  0– 7 : uint64  nonce   (ignored here)
        //   bytes  8–11 : uint32  srcEid  (ignored here)
        //   bytes 12–43 : uint256 amountLD
        //   bytes 44+   : payload = abi.encode(Call[])
        uint256 amountLD = uint256(bytes32(_message[12:44]));
        Call[] memory calls = abi.decode(_message[44:], (Call[]));

        // Execute each call sequentially; any failure reverts the whole compose.
        for (uint256 i = 0; i < calls.length; i++) {
            (bool success, bytes memory reason) = calls[i].target.call{value: calls[i].value}(
                calls[i].data
            );
            if (!success) revert CallFailed(i, reason);
        }

        emit Composed(_from, _guid, amountLD, calls.length);
    }

    // -----------------------------------------------------------------------
    // ETH reception
    // -----------------------------------------------------------------------

    receive() external payable {}
}

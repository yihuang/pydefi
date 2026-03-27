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
 * 3. ``lzCompose`` validates the caller (must be the authorised endpoint),
 *    then decodes the compose message and executes each ``Call`` sequentially.
 *    Any revert in a sub-call reverts the entire compose execution.
 *
 * Security notes
 * --------------
 *  • Only the authorised LayerZero endpoint may call ``lzCompose``.
 *  • Any OFT contract forwarded by the endpoint can trigger compose.
 *  • The compose payload encodes arbitrary calls; senders are responsible for
 *    constructing safe payloads.  Review each call's target and calldata
 *    off-chain before broadcasting.
 *  • The owner can rescue any ETH or ERC-20 tokens stuck in this contract via
 *    ``rescueETH`` and ``rescueToken``, e.g. when a compose action keeps
 *    failing and the funds need to be recovered out-of-band.
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

    /// @notice Emitted when ownership is transferred.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /// @notice The LayerZero v2 endpoint address authorised to call ``lzCompose``.
    address public immutable endpoint;

    /// @notice Owner address — may rescue stuck funds and transfer ownership.
    address public owner;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _endpoint  The LayerZero v2 EndpointV2 contract address.
     * @param _owner     Address that may call rescue functions and transfer ownership.
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

    /// @notice Transfer ownership to a new address.
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "OFTComposer: zero address");
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }

    /**
     * @notice Rescue ETH stuck in this contract.
     *
     * Use this when a compose action fails permanently and the ETH sent along
     * with the compose message needs to be recovered out-of-band.
     *
     * @param _recipient Address to send the rescued ETH to.
     * @param _amount    Amount of ETH (in wei) to rescue.
     */
    function rescueETH(address payable _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "OFTComposer: zero address");
        (bool ok, ) = _recipient.call{value: _amount}("");
        require(ok, "OFTComposer: ETH transfer failed");
    }

    /**
     * @notice Rescue ERC-20 tokens stuck in this contract.
     *
     * Use this when OFT tokens or other ERC-20 tokens accumulate in the
     * contract and need to be recovered by the owner.
     *
     * @param _token     ERC-20 token contract address.
     * @param _recipient Address to send the rescued tokens to.
     * @param _amount    Token amount to rescue (in the token's native decimals).
     */
    function rescueToken(address _token, address _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "OFTComposer: zero address");
        // Inline low-level call to avoid importing IERC20.
        (bool ok, bytes memory ret) = _token.call(
            abi.encodeWithSignature("transfer(address,uint256)", _recipient, _amount)
        );
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "OFTComposer: token transfer failed");
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

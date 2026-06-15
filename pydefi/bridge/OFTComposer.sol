// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../vm/DEXCallbackRouter.sol";
import "../vm/InterpreterRunner.sol";
import "../vm/TransientReentrancyGuard.sol";

/**
 * @title OFTComposer
 * @notice LayerZero OFT compose receiver that executes a DeFiVM program after
 *         an OFT cross-chain token transfer arrives on the destination chain.
 *
 * How it works
 * ------------
 * 1. A sender on the source chain encodes a DeFiVM program as the
 *    ``composeMsg`` in their OFT ``send`` call.
 * 2. After the OFT tokens arrive, the LayerZero EndpointV2 calls ``lzCompose``.
 * 3. ``lzCompose`` validates the caller (must be the authorised endpoint), then
 *    transfers the OFT tokens from the composer to DeFiVM and forwards the
 *    program to DeFiVM, passing the OFT parameters via the transient-storage
 *    parameter channel.
 *
 * Security notes
 * --------------
 *  • Only the authorised LayerZero endpoint may call ``lzCompose``.
 *  • Any OFT contract forwarded by the endpoint can trigger compose.
 *  • The compose payload is raw DeFiVM bytecode; senders are responsible for
 *    constructing safe programs.  Simulate the full execution off-chain before
 *    broadcasting.
 *  • The owner can rescue any ETH or ERC-20 tokens stuck in this contract via
 *    ``rescueETH`` and ``rescueToken``, e.g. when a compose program keeps
 *    failing and the funds need to be recovered out-of-band.
 *
 * Compose-message encoding
 * ------------------------
 * The raw ``_message`` bytes use the standard ``OFTComposeMsgCodec`` layout::
 *
 *   | 8 bytes nonce | 4 bytes srcEid | 32 bytes amountLD | DeFiVM program |
 *
 * The custom payload (bytes 44+) is raw DeFiVM bytecode.
 *
 * Execution model
 * ---------------
 * The composer DELEGATECALLs the EVM interpreter directly with the program
 * as calldata, so the program runs in this composer's context.  OFT params
 * are staged in this composer's transient slots before the DELEGATECALL::
 *
 *   slot 0 = amountLD  (uint256 amount delivered by the OFT, in local decimals)
 *   slot 1 = _from     (address of the OFT app contract, right-aligned in 32B)
 *
 * A Python program reads them via ``prog.builder.tload(IRLiteral(0))`` /
 * ``tload(IRLiteral(1))``.  Build the LZ compose message as::
 *
 *   message = (
 *       struct.pack('>Q', nonce)        # 8 bytes  — uint64 nonce
 *       + struct.pack('>I', src_eid)    # 4 bytes  — uint32 srcEid
 *       + amount_ld.to_bytes(32, 'big') # 32 bytes — uint256 amountLD
 *       + prog.build(...)               # DeFiVM bytecode
 *   )
 *
 * Reentrancy
 * ----------
 * ``lzCompose`` is guarded by ``TransientReentrancyGuard`` — the limitation
 * documented there still applies: the program runs in this composer's
 * context and can ``TSTORE`` the lock slot, so the guard is defense-in-depth
 * against external re-entry, not a sandbox against the program itself.
 */

// ---------------------------------------------------------------------------
// OFTComposer
// ---------------------------------------------------------------------------

contract OFTComposer is DEXCallbackRouter, TransientReentrancyGuard, InterpreterRunner {
    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// @notice Thrown when the caller is not the authorised LayerZero endpoint.
    error UnauthorizedEndpoint(address caller);

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    /// @notice Emitted after a successful compose execution.
    event Composed(address indexed from, bytes32 indexed guid, uint256 amountLD);

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
     * @param _endpoint     The LayerZero v2 EndpointV2 contract address.
     * @param _interpreter  EVM interpreter to DELEGATECALL (see
     *                      :class:`InterpreterRunner`); pass ``address(0)``
     *                      for the well-known pre-deployed Analog-Labs
     *                      interpreter.
     * @param _owner        Address that may call rescue functions and transfer ownership.
     */
    constructor(address _endpoint, address _interpreter, address _owner)
        InterpreterRunner(_interpreter)
    {
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
     * Use this when a compose program fails permanently and the ETH sent along
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
     *                  ``| 8B nonce | 4B srcEid | 32B amountLD | DeFiVM program |``
     */
    function lzCompose(
        address _from,
        bytes32 _guid,
        bytes calldata _message,
        address /* _executor */,
        bytes calldata /* _extraData */
    ) external payable nonReentrant {
        // Only the authorised endpoint may call this function.
        if (msg.sender != endpoint) revert UnauthorizedEndpoint(msg.sender);

        // Validate minimum message length: 8B nonce + 4B srcEid + 32B amountLD = 44 bytes.
        require(_message.length >= 44, "OFTComposer: message too short");

        // Decode OFTComposeMsgCodec layout:
        //   bytes  0– 7 : uint64  nonce    (ignored)
        //   bytes  8–11 : uint32  srcEid   (ignored)
        //   bytes 12–43 : uint256 amountLD
        //   bytes 44+   : DeFiVM program bytecode
        uint256 amountLD = uint256(bytes32(_message[12:44]));
        bytes calldata program = _message[44:];

        // Tokens delivered by the OFT are already on this composer; the
        // program (running in composer context via DELEGATECALL) operates
        // on the held balance directly.
        assembly {
            tstore(0, amountLD)
            tstore(1, _from)
        }
        _runProgram(program);
        assembly {
            tstore(0, 0)
            tstore(1, 0)
        }

        emit Composed(_from, _guid, amountLD);
    }

    // -----------------------------------------------------------------------
    // ETH reception
    // -----------------------------------------------------------------------

    receive() external payable {}
}

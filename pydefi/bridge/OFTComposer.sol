// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

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
 * The OFT parameters are exposed to the program via DeFiVM's EIP-1153
 * transient-storage channel; the program reads them with ``TLOAD(slot)``::
 *
 *   slot 0 = amountLD  (uint256 amount delivered by the OFT, in local decimals)
 *   slot 1 = _from     (address of the OFT app contract, right-aligned in 32B)
 *
 * A Python program reads them via the Venom builder::
 *
 *   from pydefi.vm import Program
 *   from vyper.venom.basicblock import IRLiteral
 *
 *   prog = Program()
 *   amount_ld = prog.builder.tload(IRLiteral(0))
 *   from_addr = prog.builder.tload(IRLiteral(1))
 *   ; ... use amount_ld / from_addr anywhere later ...
 *
 *   message = (
 *       struct.pack('>Q', nonce)        # 8 bytes  — uint64 nonce
 *       + struct.pack('>I', src_eid)    # 4 bytes  — uint32 srcEid
 *       + amount_ld.to_bytes(32, 'big') # 32 bytes — uint256 amountLD
 *       + prog.build(...)               # DeFiVM bytecode
 *   )
 */

// ---------------------------------------------------------------------------
// IOFT
// ---------------------------------------------------------------------------

/// @notice Minimal interface for querying the underlying ERC-20 token of an OFT.
///
/// For a native OFT (the OFT contract *is* the ERC-20), ``token()`` returns
/// ``address(this)``.  For an OFT Adapter that wraps a pre-existing ERC-20,
/// ``token()`` returns the address of that underlying ERC-20 contract.
interface IOFT {
    function token() external view returns (address);
}

// ---------------------------------------------------------------------------
// IDeFiVM
// ---------------------------------------------------------------------------

/// @notice Minimal interface for staging params and executing a DeFiVM program.
interface IDeFiVM {
    function setParams(bytes32[] calldata params) external;
    function execute(bytes calldata program) external payable;
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

    /// @notice The DeFiVM contract used to execute compose programs.
    IDeFiVM public immutable vm;

    /// @notice Owner address — may rescue stuck funds and transfer ownership.
    address public owner;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _endpoint  The LayerZero v2 EndpointV2 contract address.
     * @param _vm        The DeFiVM contract address.
     * @param _owner     Address that may call rescue functions and transfer ownership.
     */
    constructor(address _endpoint, address _vm, address _owner) {
        endpoint = _endpoint;
        vm = IDeFiVM(_vm);
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
    ) external payable {
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

        // Transfer the received OFT tokens from this composer to DeFiVM FIRST.
        // _from is the OFT *app* contract; call token() to get the underlying
        // ERC-20 address (for a native OFT token() returns address(this),
        // for an OFT Adapter it returns the wrapped ERC-20).  Doing the
        // external calls before setParams closes the reentrancy window where
        // a malicious _from could re-enter ``vm.setParams`` between our
        // staging and ``execute``.
        if (amountLD > 0) {
            address token = IOFT(_from).token();
            (bool ok, bytes memory ret) = token.call(
                abi.encodeWithSignature("transfer(address,uint256)", address(vm), amountLD)
            );
            require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "OFTComposer: token transfer failed");
        }

        // Stage OFT parameters in DeFiVM's transient store, then execute
        // immediately — no external call between these two.
        //   slot 0 = amountLD  → program reads via TLOAD(0)
        //   slot 1 = _from     → program reads via TLOAD(1)
        bytes32[] memory params = new bytes32[](2);
        params[0] = bytes32(amountLD);
        params[1] = bytes32(uint256(uint160(_from)));
        vm.setParams(params);

        // Execute via DeFiVM, forwarding any ETH received with this compose call.
        vm.execute{value: msg.value}(program);

        emit Composed(_from, _guid, amountLD);
    }

    // -----------------------------------------------------------------------
    // ETH reception
    // -----------------------------------------------------------------------

    receive() external payable {}
}

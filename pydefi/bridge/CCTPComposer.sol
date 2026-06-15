// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../vm/DEXCallbackRouter.sol";
import "../vm/InterpreterRunner.sol";
import "../vm/TransientReentrancyGuard.sol";

/**
 * @title CCTPComposer
 * @notice Circle CCTP v2 compose receiver that mints USDC via CCTP v2 and then
 *         executes a DeFiVM program on the destination chain.
 *
 * How it works
 * ------------
 * 1. A sender on the source chain calls
 *    ``TokenMessengerV2.depositForBurnWithHook`` with the DeFiVM program
 *    passed as ``hookData`` and ``mintRecipient`` set to this contract's address.
 * 2. After Circle's attestation service signs the burn proof, a relayer (or
 *    the original sender) calls ``receiveAndExecute`` on this contract with
 *    the raw CCTP v2 ``message`` and the Circle ``attestation``.
 * 3. ``receiveAndExecute`` calls the CCTP v2 ``MessageTransmitterV2.receiveMessage``
 *    which mints USDC directly to this contract.  The DeFiVM program is then
 *    read from the ``hookData`` field in the message body, a PUSH prologue is
 *    prepended, and the combined program is forwarded to DeFiVM for execution.
 *
 * Security notes
 * --------------
 *  • Only valid Circle attestations can trigger ``receiveMessage``; the
 *    ``MessageTransmitterV2`` enforces this and tracks spent nonces so the
 *    same attestation can never be replayed.
 *  • Set ``destinationCaller`` to this contract's address when calling
 *    ``depositForBurnWithHook`` on the source chain.  The
 *    ``MessageTransmitterV2`` will then only accept ``receiveMessage`` calls
 *    originating from this contract, preventing external front-running.
 *  • The compose payload (DeFiVM bytecode) is committed on-chain at burn time
 *    and cannot be changed afterwards — the program is part of the attested
 *    CCTP message.
 *  • The owner can rescue any ETH or ERC-20 tokens stuck in this contract via
 *    ``rescueETH`` and ``rescueToken``.
 *
 * CCTP v2 message layout
 * -----------------------
 * The raw CCTP v2 ``message`` bytes (MessageV2) have the following layout::
 *
 *   MessageV2 header (148 bytes):
 *   | 4B version | 4B sourceDomain | 4B destinationDomain | 32B nonce |
 *   | 32B sender | 32B recipient   | 32B destinationCaller |
 *   | 4B minFinalityThreshold | 4B finalityThresholdExecuted |
 *
 *   BurnMessageV2 body (starts at byte 148):
 *   | 4B version  | 32B burnToken | 32B mintRecipient | 32B amount |
 *   | 32B messageSender | 32B maxFee | 32B feeExecuted |
 *   | 32B expirationBlock | dynamic hookData |
 *
 * Relevant absolute offsets (from message start):
 *   sourceDomain  : bytes[4:8]     (uint32)
 *   nonce         : bytes[12:44]   (bytes32)  ← v2 uses bytes32, not uint64
 *   amount        : bytes[216:248] (uint256)  — 148 header + 68 BurnMessageV2 offset
 *   feeExecuted   : bytes[312:344] (uint256)  — 148 header + 164 BurnMessageV2 offset
 *   hookData      : bytes[376:]    (bytes)    — 148 header + 228 BurnMessageV2 offset
 *
 * Minimum message length: 376 bytes (header=148 + fixed BurnMessageV2=228).
 *
 * Execution model
 * ---------------
 * The composer DELEGATECALLs a pre-deployed EVM interpreter (Analog-Labs by
 * default) directly with the program as calldata.  The program runs in this
 * composer's context, so:
 *
 *   • ``address(this)`` inside the program == this composer's address.
 *   • External CALLs originate from this composer.
 *   • Transient storage operations (TSTORE / TLOAD) read/write this
 *     composer's transient namespace.
 *
 * Bridged parameters are staged in this composer's transient slots before
 * the DELEGATECALL; the program reads them with ``TLOAD(slot)``::
 *
 *   slot 0 = amountReceived  (= amount - feeExecuted, USDC minted to this contract)
 *   slot 1 = sourceDomain    (CCTP domain ID of the source chain)
 *
 * A Python program reads them via the Venom builder::
 *
 *   from pydefi.vm import Program
 *   from vyper.venom.basicblock import IRLiteral
 *
 *   prog = Program()
 *   amount_received = prog.builder.tload(IRLiteral(0))
 *   source_domain   = prog.builder.tload(IRLiteral(1))
 *   ; ... use amount_received / source_domain anywhere later ...
 *
 * Reentrancy
 * ----------
 * ``receiveAndExecute`` is guarded by ``TransientReentrancyGuard`` — the
 * limitation documented there still applies: the program runs in this
 * composer's context and can ``TSTORE`` the lock slot, so the guard is
 * defense-in-depth against external re-entry, not a sandbox against the
 * program itself.
 */

// ---------------------------------------------------------------------------
// CCTPComposer
// ---------------------------------------------------------------------------

contract CCTPComposer is DEXCallbackRouter, TransientReentrancyGuard, InterpreterRunner {
    // -----------------------------------------------------------------------
    // CCTP v2 message offsets
    // -----------------------------------------------------------------------

    // MessageV2 header layout:
    //   [0:4]    version                   (uint32)
    //   [4:8]    sourceDomain              (uint32)
    //   [8:12]   destinationDomain         (uint32)
    //   [12:44]  nonce                     (bytes32)   ← 32 bytes in v2
    //   [44:76]  sender                    (bytes32)
    //   [76:108] recipient                 (bytes32)
    //   [108:140] destinationCaller        (bytes32)
    //   [140:144] minFinalityThreshold     (uint32)
    //   [144:148] finalityThresholdExecuted (uint32)
    //   Total header: 148 bytes

    uint256 private constant SOURCE_DOMAIN_OFFSET = 4;
    uint256 private constant NONCE_OFFSET = 12;
    uint256 private constant MSG_BODY_OFFSET = 148;

    // BurnMessageV2 body layout (relative to MSG_BODY_OFFSET = 148):
    //   [0:4]    burnMessageVersion (uint32)
    //   [4:36]   burnToken          (bytes32)
    //   [36:68]  mintRecipient      (bytes32)
    //   [68:100] amount             (uint256)
    //   [100:132] messageSender     (bytes32)
    //   [132:164] maxFee            (uint256)
    //   [164:196] feeExecuted       (uint256)   ← new in v2
    //   [196:228] expirationBlock   (uint256)   ← new in v2
    //   [228:]   hookData           (bytes)     ← new in v2 (DeFiVM program)

    uint256 private constant BURN_MSG_AMOUNT_OFFSET = 68;
    uint256 private constant BURN_MSG_FEE_EXECUTED_OFFSET = 164;
    uint256 private constant BURN_MSG_HOOK_DATA_OFFSET = 228;

    // Absolute offsets in the full message:
    //   amount        = 148 + 68  = 216
    //   feeExecuted   = 148 + 164 = 312
    //   hookData      = 148 + 228 = 376
    uint256 private constant AMOUNT_OFFSET = MSG_BODY_OFFSET + BURN_MSG_AMOUNT_OFFSET;
    uint256 private constant FEE_EXECUTED_OFFSET = MSG_BODY_OFFSET + BURN_MSG_FEE_EXECUTED_OFFSET;
    uint256 private constant HOOK_DATA_OFFSET = MSG_BODY_OFFSET + BURN_MSG_HOOK_DATA_OFFSET;

    // Minimum message length: 148 header + 228 fixed BurnMessageV2 = 376 bytes
    uint256 private constant MIN_MESSAGE_LENGTH = HOOK_DATA_OFFSET;

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// @notice Thrown when the CCTP ``receiveMessage`` call fails.
    error ReceiveMessageFailed();

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    /// @notice Emitted after a successful compose execution.
    event Composed(uint32 indexed sourceDomain, bytes32 indexed nonce, uint256 amountReceived);

    /// @notice Emitted when ownership is transferred.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /// @notice The Circle CCTP v2 ``MessageTransmitterV2`` contract address.
    address public immutable messageTransmitter;

    /// @notice The USDC token contract address on this chain.
    address public immutable usdc;

    /// @notice Owner address — may rescue stuck funds and transfer ownership.
    address public owner;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _messageTransmitter  The Circle CCTP v2 ``MessageTransmitterV2`` address.
     * @param _usdc                USDC token address on this chain.
     * @param _interpreter         EVM interpreter to DELEGATECALL (see
     *                             :class:`InterpreterRunner`); pass
     *                             ``address(0)`` for the well-known
     *                             pre-deployed Analog-Labs interpreter.
     * @param _owner               Address that may call rescue functions and transfer ownership.
     */
    constructor(address _messageTransmitter, address _usdc, address _interpreter, address _owner)
        InterpreterRunner(_interpreter)
    {
        messageTransmitter = _messageTransmitter;
        usdc = _usdc;
        owner = _owner;
    }

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "CCTPComposer: not owner");
        _;
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    /// @notice Transfer ownership to a new address.
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "CCTPComposer: zero address");
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }

    /**
     * @notice Rescue ETH stuck in this contract.
     *
     * @param _recipient Address to send the rescued ETH to.
     * @param _amount    Amount of ETH (in wei) to rescue.
     */
    function rescueETH(address payable _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "CCTPComposer: zero address");
        (bool ok, ) = _recipient.call{value: _amount}("");
        require(ok, "CCTPComposer: ETH transfer failed");
    }

    /**
     * @notice Rescue ERC-20 tokens stuck in this contract.
     *
     * @param _token     ERC-20 token contract address.
     * @param _recipient Address to send the rescued tokens to.
     * @param _amount    Token amount to rescue.
     */
    function rescueToken(address _token, address _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "CCTPComposer: zero address");
        (bool ok, bytes memory ret) = _token.call(
            abi.encodeWithSignature("transfer(address,uint256)", _recipient, _amount)
        );
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "CCTPComposer: token transfer failed");
    }

    // -----------------------------------------------------------------------
    // Core: receive CCTP v2 message and execute compose program
    // -----------------------------------------------------------------------

    /**
     * @notice Mint USDC via CCTP v2 and execute the DeFiVM program embedded
     *         as ``hookData`` in the CCTP message.
     *
     * Flow:
     * 1. Validate the message is at least ``MIN_MESSAGE_LENGTH`` bytes.
     * 2. Decode ``sourceDomain``, ``nonce``, ``amount``, ``feeExecuted``, and
     *    ``hookData`` (= the DeFiVM program) from the CCTP v2 message.
     * 3. Call ``MessageTransmitterV2.receiveMessage(message, attestation)`` to
     *    mint ``amount - feeExecuted`` USDC to this contract.
     * 4. Stage bridged parameters in this composer's transient-storage slots
     *    (slot 0 = amountReceived, slot 1 = sourceDomain).
     * 5. DELEGATECALL the EVM interpreter with the program as calldata.  The
     *    program runs in this composer's context — USDC stays here and the
     *    program reads params via ``TLOAD`` from this contract's transient
     *    namespace.
     * 6. Clear the staged param slots so a subsequent compose call in the
     *    same tx starts clean.
     *
     * Transient-storage layout exposed to the program:
     *   slot 0 = amountReceived  (USDC received = amount - feeExecuted, 6 dec.)
     *   slot 1 = sourceDomain    (CCTP domain ID of the source chain)
     *
     * @param message      Raw CCTP v2 message bytes (``MessageSent`` event data).
     * @param attestation  Circle attestation bytes for the message.
     */
    function receiveAndExecute(bytes calldata message, bytes calldata attestation)
        external
        payable
        nonReentrant
    {
        require(message.length >= MIN_MESSAGE_LENGTH, "CCTPComposer: message too short");

        // Decode bridged parameters from the CCTP v2 message.
        uint32 sourceDomain = uint32(bytes4(message[SOURCE_DOMAIN_OFFSET:SOURCE_DOMAIN_OFFSET + 4]));
        bytes32 nonce = bytes32(message[NONCE_OFFSET:NONCE_OFFSET + 32]);
        uint256 amount = uint256(bytes32(message[AMOUNT_OFFSET:AMOUNT_OFFSET + 32]));
        uint256 feeExecuted = uint256(bytes32(message[FEE_EXECUTED_OFFSET:FEE_EXECUTED_OFFSET + 32]));

        // The DeFiVM program is embedded as hookData in the BurnMessageV2 body.
        bytes calldata program = message[HOOK_DATA_OFFSET:];

        // Mint USDC to this contract by processing the CCTP v2 message.
        // MessageTransmitterV2 enforces that mintRecipient == address(this)
        // and that each nonce can only be used once.
        (bool ok, bytes memory result) = messageTransmitter.call(
            abi.encodeWithSignature("receiveMessage(bytes,bytes)", message, attestation)
        );
        if (!ok || !abi.decode(result, (bool))) revert ReceiveMessageFailed();

        // Actual USDC minted = amount - feeExecuted (relayer fee deducted).
        uint256 amountReceived = amount - feeExecuted;

        // Stage bridged params in our own transient storage, then DELEGATECALL
        // the interpreter so the program runs in this composer's context and
        // reads them via TLOAD.
        assembly {
            tstore(0, amountReceived)
            tstore(1, sourceDomain)
        }
        _runProgram(program);
        assembly {
            tstore(0, 0)
            tstore(1, 0)
        }

        emit Composed(sourceDomain, nonce, amountReceived);
    }

    // -----------------------------------------------------------------------
    // ETH reception
    // -----------------------------------------------------------------------

    receive() external payable {}
}

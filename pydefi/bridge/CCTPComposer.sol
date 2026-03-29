// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title CCTPComposer
 * @notice Circle CCTP compose receiver that mints USDC via CCTP and then
 *         executes a DeFiVM program on the destination chain.
 *
 * How it works
 * ------------
 * 1. A sender on the source chain calls ``TokenMessenger.depositForBurn`` (or
 *    ``depositForBurnWithCaller``) setting ``mintRecipient`` to this contract's
 *    address.  Optionally they emit a ``BridgeCompose`` event with the intended
 *    DeFiVM program so that an off-chain relayer can retrieve it.
 * 2. After Circle's attestation service signs the burn proof, a relayer (or
 *    the original sender) calls ``receiveAndExecute`` on this contract with
 *    the raw CCTP ``message``, the Circle ``attestation``, and the DeFiVM
 *    ``program`` to execute.
 * 3. ``receiveAndExecute`` calls the CCTP ``MessageTransmitter.receiveMessage``
 *    which mints USDC directly to this contract.  It then prepends a PUSH
 *    prologue for the bridged parameters and forwards the combined program to
 *    DeFiVM for execution.
 *
 * Security notes
 * --------------
 *  • Only valid Circle attestations can trigger ``receiveMessage``; the
 *    ``MessageTransmitter`` enforces this and tracks spent nonces so the same
 *    attestation can never be replayed.
 *  • For ``depositForBurnWithCaller`` flows, set ``destinationCaller`` to this
 *    contract's address.  The ``MessageTransmitter`` will then only accept
 *    ``receiveMessage`` calls that originate from this contract (i.e. via
 *    ``receiveAndExecute``), preventing external front-running of the mint.
 *  • The compose payload is raw DeFiVM bytecode; callers are responsible for
 *    constructing safe programs.  Simulate the full execution off-chain before
 *    broadcasting.
 *  • The owner can rescue any ETH or ERC-20 tokens stuck in this contract via
 *    ``rescueETH`` and ``rescueToken``.
 *
 * CCTP message layout (v1)
 * ------------------------
 * The raw CCTP ``message`` bytes have the following layout::
 *
 *   Header (116 bytes):
 *   | 4B version | 4B sourceDomain | 4B destinationDomain | 8B nonce |
 *   | 32B sender | 32B recipient | 32B destinationCaller |
 *
 *   BurnMessage body (starts at byte 116):
 *   | 4B version | 32B burnToken | 32B mintRecipient | 32B amount |
 *   | 32B messageSender |
 *
 * Relevant offsets (from message start):
 *   sourceDomain : bytes[4:8]   (uint32)
 *   nonce        : bytes[12:20] (uint64)
 *   amount       : bytes[184:216] (uint256)  — 116 header + 68 BurnMessage prefix
 *
 * DeFiVM stack layout after prologue
 * -----------------------------------
 * Before executing the user program, CCTPComposer prepends two PUSH
 * instructions so the bridged parameters are already on the stack::
 *
 *   PUSH_U256 <amount>        ; pushed first  → stack[0] (bottom)
 *   PUSH_U256 <sourceDomain>  ; pushed second → stack[1] (top)
 *
 * A typical program begins by saving these into registers::
 *
 *   STORE_REG 0   ; R0 = sourceDomain
 *   STORE_REG 1   ; R1 = amount (USDC bridged, 6 decimals)
 *   ; ... use R0 and R1 anywhere later with LOAD_REG ...
 *
 * Python helper (``pydefi.vm.program``)::
 *
 *   from pydefi.vm.program import store_reg, ...
 *
 *   program = store_reg(0) + store_reg(1) + ...
 */

// ---------------------------------------------------------------------------
// IDeFiVM
// ---------------------------------------------------------------------------

/// @notice Minimal interface for calling DeFiVM.execute.
interface IDeFiVM {
    function execute(bytes calldata program) external payable;
}

// ---------------------------------------------------------------------------
// IMessageTransmitter
// ---------------------------------------------------------------------------

/// @notice Minimal interface for Circle CCTP MessageTransmitter.
interface IMessageTransmitter {
    function receiveMessage(bytes calldata message, bytes calldata attestation) external returns (bool success);
}

// ---------------------------------------------------------------------------
// CCTPComposer
// ---------------------------------------------------------------------------

contract CCTPComposer {
    // DeFiVM PUSH opcode (mirrors DeFiVM.sol)
    uint8 private constant OP_PUSH_U256 = 0x01;

    // -----------------------------------------------------------------------
    // CCTP message offsets (v1)
    // -----------------------------------------------------------------------

    // Header layout:
    //   [0:4]    version          (uint32)
    //   [4:8]    sourceDomain     (uint32)
    //   [8:12]   destinationDomain (uint32)
    //   [12:20]  nonce            (uint64)
    //   [20:52]  sender           (bytes32)
    //   [52:84]  recipient        (bytes32)
    //   [84:116] destinationCaller (bytes32)
    //   Total header: 116 bytes

    uint256 private constant SOURCE_DOMAIN_OFFSET = 4;
    uint256 private constant NONCE_OFFSET = 12;
    uint256 private constant MSG_BODY_OFFSET = 116;

    // BurnMessage body layout (relative to MSG_BODY_OFFSET):
    //   [0:4]    burnMessageVersion (uint32)
    //   [4:36]   burnToken          (bytes32)
    //   [36:68]  mintRecipient      (bytes32)
    //   [68:100] amount             (uint256)
    //   [100:132] messageSender     (bytes32)

    uint256 private constant BURN_MSG_AMOUNT_OFFSET = 68;

    // Absolute offset of amount in the full message: 116 + 68 = 184
    uint256 private constant AMOUNT_OFFSET = MSG_BODY_OFFSET + BURN_MSG_AMOUNT_OFFSET;

    // Minimum message length: 116 header + 100 burn body = 216 bytes
    uint256 private constant MIN_MESSAGE_LENGTH = AMOUNT_OFFSET + 32;

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// @notice Thrown when the CCTP ``receiveMessage`` call fails.
    error ReceiveMessageFailed();

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    /// @notice Emitted after a successful compose execution.
    event Composed(uint32 indexed sourceDomain, uint64 indexed nonce, uint256 amount);

    /// @notice Emitted when ownership is transferred.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /// @notice The Circle CCTP ``MessageTransmitter`` contract address.
    address public immutable messageTransmitter;

    /// @notice The USDC token contract address on this chain.
    address public immutable usdc;

    /// @notice The DeFiVM contract used to execute compose programs.
    IDeFiVM public immutable vm;

    /// @notice Owner address — may rescue stuck funds and transfer ownership.
    address public owner;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _messageTransmitter  The Circle CCTP ``MessageTransmitter`` address.
     * @param _usdc                USDC token address on this chain.
     * @param _vm                  The DeFiVM contract address.
     * @param _owner               Address that may call rescue functions and transfer ownership.
     */
    constructor(address _messageTransmitter, address _usdc, address _vm, address _owner) {
        messageTransmitter = _messageTransmitter;
        usdc = _usdc;
        vm = IDeFiVM(_vm);
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
    // Core: receive CCTP message and execute compose program
    // -----------------------------------------------------------------------

    /**
     * @notice Mint USDC via CCTP and execute a DeFiVM compose program.
     *
     * Flow:
     * 1. Call ``MessageTransmitter.receiveMessage(message, attestation)`` to
     *    mint USDC to this contract (``mintRecipient`` in the CCTP message must
     *    be ``address(this)``).
     * 2. Decode ``amount`` and ``sourceDomain`` from the CCTP message.
     * 3. Build a DeFiVM prologue that pushes the bridged parameters onto the
     *    stack before the user program runs.
     * 4. Transfer the minted USDC to the DeFiVM contract.
     * 5. Execute the combined program via DeFiVM, forwarding any ETH supplied
     *    with this call.
     *
     * Stack layout after prologue (bottom to top):
     *   stack[0] = amount (USDC amount received, 6 decimals)
     *   stack[1] = sourceDomain (CCTP domain ID of the source chain)
     *
     * Tip: use ``depositForBurnWithCaller`` on the source chain with
     * ``destinationCaller = address(this)`` to ensure that only this contract
     * can mint the USDC, preventing front-running of the ``receiveMessage`` call.
     *
     * @param message      Raw CCTP message bytes (as emitted in the ``MessageSent`` event).
     * @param attestation  Circle attestation bytes for the message.
     * @param program      DeFiVM bytecode to execute after USDC is minted.
     */
    function receiveAndExecute(
        bytes calldata message,
        bytes calldata attestation,
        bytes calldata program
    ) external payable {
        // Validate minimum message length.
        require(message.length >= MIN_MESSAGE_LENGTH, "CCTPComposer: message too short");

        // Decode bridged parameters from the CCTP message.
        uint32 sourceDomain = uint32(bytes4(message[SOURCE_DOMAIN_OFFSET:SOURCE_DOMAIN_OFFSET + 4]));
        uint64 nonce = uint64(bytes8(message[NONCE_OFFSET:NONCE_OFFSET + 8]));
        uint256 amount = uint256(bytes32(message[AMOUNT_OFFSET:AMOUNT_OFFSET + 32]));

        // Mint USDC to this contract by processing the CCTP message.
        // The MessageTransmitter enforces that mintRecipient == address(this)
        // and that each (sourceDomain, nonce) pair can only be used once.
        (bool ok, bytes memory result) = messageTransmitter.call(
            abi.encodeWithSignature("receiveMessage(bytes,bytes)", message, attestation)
        );
        if (!ok || !abi.decode(result, (bool))) revert ReceiveMessageFailed();

        // Build a prologue that pushes the CCTP transfer parameters onto the
        // DeFiVM stack before the user program runs:
        //
        //   PUSH_U256 <amount>        (1B opcode + 32B value = 33B)
        //   PUSH_U256 <sourceDomain>  (1B opcode + 32B value = 33B)
        //
        // After the prologue the initial stack layout is:
        //   stack[0] = amount       (pushed first, bottom)
        //   stack[1] = sourceDomain (pushed second, top)
        bytes memory fullProgram = bytes.concat(
            abi.encodePacked(OP_PUSH_U256, bytes32(amount), OP_PUSH_U256, bytes32(uint256(sourceDomain))),
            program
        );

        // Transfer the minted USDC from this composer to DeFiVM.
        if (amount > 0) {
            (bool tok, bytes memory ret) = usdc.call(
                abi.encodeWithSignature("transfer(address,uint256)", address(vm), amount)
            );
            require(tok && (ret.length == 0 || abi.decode(ret, (bool))), "CCTPComposer: usdc transfer failed");
        }

        // Execute via DeFiVM, forwarding any ETH received with this call.
        vm.execute{value: msg.value}(fullProgram);

        emit Composed(sourceDomain, nonce, amount);
    }

    // -----------------------------------------------------------------------
    // ETH reception
    // -----------------------------------------------------------------------

    receive() external payable {}
}

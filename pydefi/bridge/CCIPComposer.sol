// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title CCIPComposer
 * @notice Destination-chain CCIP receiver that executes a DeFiVM program
 *         attached to the inbound message after the Router delivers the
 *         bridged tokens.
 *
 * Flow: the Router calls ccipReceive(Any2EVMMessage) → this contract prepends
 * two PUSH_U256 instructions for (amountReceived, sourceChainSelector),
 * transfers the received token to DeFiVM, and forwards the combined program
 * with msg.value attached.  Compose flows expect exactly one entry in
 * destTokenAmounts; multi-token programs should subclass this receiver.
 *
 * Source-side authorisation is opt-in via the (sourceChainSelector,
 * senderHash) allowlist; when allowlistEnabled is false any Router-delivered
 * sender may trigger compose (matches OFTComposer).  Set it true and register
 * trusted senders before deploying any contract that forwards real value.
 * The compose payload is raw DeFiVM bytecode — simulate before broadcasting.
 * The owner can recover stuck funds via rescueETH / rescueToken.
 */

// CCIP types delivered by the Router.
struct EVMTokenAmount {
    address token;
    uint256 amount;
}

struct Any2EVMMessage {
    bytes32 messageId;
    uint64 sourceChainSelector;
    bytes sender;
    bytes data;                          // DeFiVM bytecode
    EVMTokenAmount[] destTokenAmounts;   // minted to address(this)
}

interface IDeFiVM {
    function execute(bytes calldata program) external payable;
}

contract CCIPComposer {
    // EVM PUSH32 opcode: 1-byte op + 32-byte immediate.
    uint8 private constant OP_PUSH_U256 = 0x7F;

    error UnauthorizedRouter(address caller);
    error UnexpectedTokenCount(uint256 count);
    error UnauthorizedSender(uint64 sourceChainSelector, bytes32 senderHash);
    error ZeroAddress();

    event Composed(
        uint64 indexed sourceChainSelector,
        bytes32 indexed messageId,
        address token,
        uint256 amountReceived
    );
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event AllowlistEnabledChanged(bool enabled);
    event AllowedSenderChanged(uint64 indexed sourceChainSelector, bytes32 indexed senderHash, bool allowed);

    /// @notice CCIP Router authorised to call ccipReceive.
    address public immutable router;
    /// @notice DeFiVM contract that executes compose programs.
    IDeFiVM public immutable vm;
    /// @notice Owner — may rescue funds, toggle the allowlist, and transfer ownership.
    address public owner;
    /// @notice Whether ccipReceive consults `allowedSender`.
    bool public allowlistEnabled;
    /// @notice Allowlist keyed by `keccak256(message.sender)` because CCIP
    ///         delivers the source address as variable-length `bytes`.
    mapping(uint64 sourceChainSelector => mapping(bytes32 senderHash => bool)) public allowedSender;

    /**
     * @param _allowlistEnabled  Pass `true` for production deployments that
     *        forward real value — they start fail-closed and require the
     *        owner to register trusted (selector, sender) pairs via
     *        `setAllowed`.  Pass `false` for tests / dev.
     * @dev   All address arguments must be non-zero.
     */
    constructor(address _router, address _vm, address _owner, bool _allowlistEnabled) {
        if (_router == address(0)) revert ZeroAddress();
        if (_vm == address(0)) revert ZeroAddress();
        if (_owner == address(0)) revert ZeroAddress();
        router = _router;
        vm = IDeFiVM(_vm);
        owner = _owner;
        allowlistEnabled = _allowlistEnabled;
        emit AllowlistEnabledChanged(_allowlistEnabled);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "CCIPComposer: not owner");
        _;
    }

    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "CCIPComposer: zero address");
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }

    function rescueETH(address payable _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "CCIPComposer: zero address");
        (bool ok, ) = _recipient.call{value: _amount}("");
        require(ok, "CCIPComposer: ETH transfer failed");
    }

    function rescueToken(address _token, address _recipient, uint256 _amount) external onlyOwner {
        require(_recipient != address(0), "CCIPComposer: zero address");
        (bool ok, bytes memory ret) = _token.call(
            abi.encodeWithSignature("transfer(address,uint256)", _recipient, _amount)
        );
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "CCIPComposer: token transfer failed");
    }

    function setAllowlistEnabled(bool _enabled) external onlyOwner {
        allowlistEnabled = _enabled;
        emit AllowlistEnabledChanged(_enabled);
    }

    /// @param _sender Raw `message.sender` bytes; EVM senders are `abi.encode(address)`.
    function setAllowed(
        uint64 _sourceChainSelector,
        bytes calldata _sender,
        bool _allowed
    ) external onlyOwner {
        bytes32 senderHash = keccak256(_sender);
        allowedSender[_sourceChainSelector][senderHash] = _allowed;
        emit AllowedSenderChanged(_sourceChainSelector, senderHash, _allowed);
    }

    /// @notice Authenticate the Router, prepend the DeFiVM prologue, hand the
    ///         token to DeFiVM, and execute the program.
    function ccipReceive(Any2EVMMessage calldata message) external payable {
        if (msg.sender != router) revert UnauthorizedRouter(msg.sender);

        if (allowlistEnabled) {
            bytes32 senderHash = keccak256(message.sender);
            if (!allowedSender[message.sourceChainSelector][senderHash]) {
                revert UnauthorizedSender(message.sourceChainSelector, senderHash);
            }
        }

        if (message.destTokenAmounts.length != 1) {
            revert UnexpectedTokenCount(message.destTokenAmounts.length);
        }

        EVMTokenAmount calldata tokenAmount = message.destTokenAmounts[0];
        uint256 amountReceived = tokenAmount.amount;
        uint64 sourceSelector = message.sourceChainSelector;

        // Initial DeFiVM stack: [amountReceived, sourceChainSelector] (top).
        bytes memory program = bytes.concat(
            abi.encodePacked(
                OP_PUSH_U256, bytes32(amountReceived),
                OP_PUSH_U256, bytes32(uint256(sourceSelector))
            ),
            message.data
        );

        if (amountReceived > 0) {
            (bool ok, bytes memory ret) = tokenAmount.token.call(
                abi.encodeWithSignature("transfer(address,uint256)", address(vm), amountReceived)
            );
            require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "CCIPComposer: token transfer failed");
        }

        vm.execute{value: msg.value}(program);
        emit Composed(sourceSelector, message.messageId, tokenAmount.token, amountReceived);
    }

    receive() external payable {}
}

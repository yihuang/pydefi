// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../vm/DEXCallbackRouter.sol";
import "../vm/InterpreterRunner.sol";
import "../vm/TransientReentrancyGuard.sol";

/**
 * @title CCIPComposer
 * @notice Destination-chain CCIP receiver that executes a DeFiVM program
 *         attached to the inbound message after the Router delivers the
 *         bridged tokens.
 *
 * Flow: the Router calls ccipReceive(Any2EVMMessage) → this contract stages
 * the bridged parameters in transient storage and DELEGATECALLs the EVM
 * interpreter with the program as calldata so the program runs in this
 * composer's own context (any tokens delivered by the Router are already on
 * this contract).  Compose flows expect exactly one entry in
 * destTokenAmounts; multi-token programs should subclass this receiver.
 *
 * Transient storage layout (cleared on tx end by EIP-1153, and reset to 0
 * after the program runs):
 *     slot 0 = amountReceived       (tokens delivered to the composer)
 *     slot 1 = sourceChainSelector  (CCIP selector of the source chain)
 *
 * A typical program reads them via TLOAD:
 *     amount_received = prog.builder.tload(IRLiteral(0))
 *     source_selector = prog.builder.tload(IRLiteral(1))
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

contract CCIPComposer is DEXCallbackRouter, TransientReentrancyGuard, InterpreterRunner {
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
    /// @notice Owner — may rescue funds, toggle the allowlist, and transfer ownership.
    address public owner;
    /// @notice Whether ccipReceive consults `allowedSender`.
    bool public allowlistEnabled;
    /// @notice Allowlist keyed by `keccak256(message.sender)` because CCIP
    ///         delivers the source address as variable-length `bytes`.
    mapping(uint64 sourceChainSelector => mapping(bytes32 senderHash => bool)) public allowedSender;

    /**
     * @param _router            The CCIP Router authorised to call ccipReceive.
     * @param _interpreter       EVM interpreter to DELEGATECALL (see
     *                           :class:`InterpreterRunner`); pass
     *                           ``address(0)`` for the well-known pre-deployed
     *                           Analog-Labs interpreter.
     * @param _owner             Address that may rescue funds, toggle the
     *                           allowlist, and transfer ownership.
     * @param _allowlistEnabled  Pass `true` for production deployments that
     *                           forward real value — they start fail-closed
     *                           and require the owner to register trusted
     *                           (selector, sender) pairs via `setAllowed`.
     *                           Pass `false` for tests / dev.
     */
    constructor(address _router, address _interpreter, address _owner, bool _allowlistEnabled)
        InterpreterRunner(_interpreter)
    {
        if (_router == address(0)) revert ZeroAddress();
        if (_owner == address(0)) revert ZeroAddress();
        router = _router;
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

    /// @notice Authenticate the Router, stage the bridged parameters in
    ///         transient storage, and DELEGATECALL the interpreter to run the
    ///         compose program in this composer's context.
    function ccipReceive(Any2EVMMessage calldata message) external payable nonReentrant {
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

        // Stage bridged parameters in transient storage; the program reads
        // them via TLOAD.  Cleared after the program runs (transient storage
        // is tx-scoped per EIP-1153, but we zero explicitly so a single tx
        // delivering multiple compose messages sees a clean slate each time).
        assembly {
            tstore(0, amountReceived)
            tstore(1, sourceSelector)
        }
        _runProgram(message.data);
        assembly {
            tstore(0, 0)
            tstore(1, 0)
        }

        emit Composed(sourceSelector, message.messageId, tokenAmount.token, amountReceived);
    }

    receive() external payable {}
}

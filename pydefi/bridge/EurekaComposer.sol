// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../vm/DEXCallbackRouter.sol";
import "../vm/InterpreterRunner.sol";
import "../vm/TransientReentrancyGuard.sol";

// Inline the IERC165 interface — kept self-contained to match the other
// pydefi composers, which avoid OpenZeppelin imports.
interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

// IIBCSenderCallbacks interface id == onAckPacket(...) ^ onTimeoutPacket(...).
// Hard-coded so this file can compile without pulling in upstream
// solidity-ibc-eureka headers. To re-verify, see
// `pydefi.vm.eureka.IIBC_SENDER_CALLBACKS_INTERFACE_ID` — same computation.
//
//   bytes4(keccak256("onAckPacket(bool,(string,string,uint64,(string,string,string,string,bytes),bytes,address))"))
//   ^
//   bytes4(keccak256("onTimeoutPacket((string,string,uint64,(string,string,string,string,bytes),address))"))
bytes4 constant IIBC_SENDER_CALLBACKS_INTERFACE_ID = 0xd3ce6f1b;

// Minimal interfaces — defined inline to avoid pulling in the full
// solidity-ibc-eureka build for what's a thin sender contract.

interface IICS20Transfer_minimal {
    struct SendTransferMsg {
        address denom;
        uint256 amount;
        string receiver;
        string sourceClient;
        string destPort;
        uint64 timeoutTimestamp;
        string memo;
    }
    function sendTransfer(SendTransferMsg calldata msg_) external returns (uint64);
}

interface IIBCAppCallbacks_minimal {
    struct Payload {
        string sourcePort;
        string destPort;
        string version;
        string encoding;
        bytes value;
    }
    struct OnAcknowledgementPacketCallback {
        string sourceClient;
        string destinationClient;
        uint64 sequence;
        Payload payload;
        bytes acknowledgement;
        address relayer;
    }
    struct OnTimeoutPacketCallback {
        string sourceClient;
        string destinationClient;
        uint64 sequence;
        Payload payload;
        address relayer;
    }
}

interface IERC20_minimal {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

/**
 * @title EurekaComposer
 * @notice IBC v2 (Eureka) sender that registers a DeFiVM follow-up program at
 *         send time and executes it on ack/timeout callbacks from the router.
 *
 * Flow
 * ----
 * 1. Caller approves this contract for ``amount`` of ``denom``.
 * 2. Caller invokes :func:`sendTransferAndCompose` with the
 *    ``SendTransferMsg`` they want submitted and the follow-up DeFiVM
 *    program bytecode.
 * 3. This contract pulls ``amount`` from the caller, approves the
 *    ``ICS20Transfer`` proxy, and submits the packet. The returned
 *    ``sequence`` is used to key the stored program: ``programs[clientId][sequence]``.
 * 4. ``ICS20Transfer.sendTransfer`` records *this contract* as the packet
 *    sender (the immediate ``_msgSender()``), so the eventual
 *    ``onAckPacket`` / ``onTimeoutPacket`` callback (routed through
 *    ``IBCSenderCallbacksLib.ackPacketCallback``) fires here.
 * 5. On callback, we DELEGATECALL the EVM interpreter with the stored
 *    program. Transient-storage layout exposed to the program::
 *
 *        slot 0 = success    (1 on ack-success, 0 on ack-error / timeout)
 *        slot 1 = sequence   (the original packet sequence)
 *
 *    A Python program reads them via the Venom builder::
 *
 *        from pydefi.vm import Program
 *        from vyper.venom.basicblock import IRLiteral
 *
 *        prog = Program()
 *        success  = prog.builder.tload(IRLiteral(0))
 *        sequence = prog.builder.tload(IRLiteral(1))
 *
 * ERC-165
 * -------
 * Implements ``supportsInterface`` for the ``IIBCSenderCallbacks``
 * interface id (``IBCSenderCallbacksLib.ackPacketCallback`` ERC165-probes
 * before calling, and silently skips contracts that don't advertise
 * support).
 *
 * Reentrancy
 * ----------
 * ``onAckPacket`` and ``onTimeoutPacket`` are guarded by
 * :class:`TransientReentrancyGuard`. The program still runs in this
 * composer's context and could ``TSTORE`` the guard slot, so the guard is
 * defense-in-depth against external re-entry, not a sandbox against the
 * program itself.
 */
contract EurekaComposer is DEXCallbackRouter, TransientReentrancyGuard, InterpreterRunner, IERC165 {
    /// @notice Deployed ICS20Transfer proxy on this chain.
    address public immutable ics20Transfer;

    /// @notice Mapping (sourceClient -> sequence -> program bytecode).
    /// @dev    Cleared after the ack/timeout callback fires.
    mapping(string => mapping(uint64 => bytes)) public programs;

    /// @notice Emitted when a packet was sent and its follow-up registered.
    event Composed(string indexed sourceClient, uint64 indexed sequence, uint256 programLen);

    /// @notice Emitted after the ack callback ran the registered program.
    event AckExecuted(string indexed sourceClient, uint64 indexed sequence, bool success);

    /// @notice Emitted after the timeout callback ran the registered program.
    event TimeoutExecuted(string indexed sourceClient, uint64 indexed sequence);

    error AlreadyComposed();
    error EmptyProgram();
    error TransferFromFailed();
    error ApproveFailed();
    error UnauthorizedCallback();

    /// @dev OpenZeppelin SafeERC20-style call: tolerates tokens that return
    /// nothing (e.g. classic USDT). Reverts on call failure or on explicit
    /// `false` return. Empty returndata is treated as success only when the
    /// target actually has code — a bare `call` to an EOA/no-code address
    /// succeeds with empty returndata, so without this guard a non-token
    /// `denom` would let transferFrom/approve "succeed" having moved nothing.
    function _erc20Call(address token, bytes memory data) internal returns (bool) {
        if (token.code.length == 0) return false;
        (bool ok, bytes memory ret) = token.call(data);
        if (!ok) return false;
        if (ret.length == 0) return true;
        return abi.decode(ret, (bool));
    }

    /**
     * @param _ics20Transfer  ICS20Transfer proxy on this chain.
     * @param _interpreter    EVM interpreter to DELEGATECALL (see
     *                        :class:`InterpreterRunner`). Pass
     *                        ``address(0)`` to use the well-known
     *                        pre-deployed Analog-Labs interpreter.
     */
    constructor(address _ics20Transfer, address _interpreter) InterpreterRunner(_interpreter) {
        ics20Transfer = _ics20Transfer;
    }

    /// @dev ICS20Transfer is the only contract that legitimately invokes our
    /// IIBCSenderCallbacks methods (via IBCSenderCallbacksLib, in its
    /// onAcknowledgementPacket / onTimeoutPacket handlers). Anyone else
    /// calling onAckPacket / onTimeoutPacket directly could either prematurely
    /// run a registered program with attacker-chosen success/sequence or wipe
    /// a registered program before the real settlement arrives.
    modifier onlyTransfer() {
        if (msg.sender != ics20Transfer) revert UnauthorizedCallback();
        _;
    }

    /// @notice ERC-165 interface detection — IIBCSenderCallbacks + IERC165 itself.
    function supportsInterface(bytes4 interfaceId) external pure override returns (bool) {
        return interfaceId == IIBC_SENDER_CALLBACKS_INTERFACE_ID
            || interfaceId == type(IERC165).interfaceId;
    }

    /**
     * @notice Pull tokens from the caller, submit an ICS-20 transfer, and
     *         register ``program`` as the follow-up to execute on ack /
     *         timeout.
     *
     * @param transferMsg ICS20 send message; ``denom`` and ``amount`` are
     *                    used to pull from the caller.
     * @param program     Raw EVM bytecode (a built DeFiVM program) to
     *                    execute via DELEGATECALL once the packet settles.
     *
     * @return sequence   The sequence the router assigned to this packet.
     */
    function sendTransferAndCompose(
        IICS20Transfer_minimal.SendTransferMsg calldata transferMsg,
        bytes calldata program
    ) external nonReentrant returns (uint64 sequence) {
        // Reject zero-length programs: registering one would store empty bytes
        // and _runRegistered would silently no-op on the callback, leaving the
        // caller no signal that the follow-up never ran.
        if (program.length == 0) revert EmptyProgram();

        // 1) Pull funds from caller into this contract. Use a SafeERC20-style
        //    low-level call so non-standard tokens (e.g. classic USDT, which
        //    doesn't return bool) work alongside spec-compliant ones.
        bool ok = _erc20Call(
            transferMsg.denom,
            abi.encodeWithSelector(
                IERC20_minimal.transferFrom.selector,
                msg.sender, address(this), transferMsg.amount
            )
        );
        if (!ok) revert TransferFromFailed();

        // 2) Approve the transfer app to pull from us.
        ok = _erc20Call(
            transferMsg.denom,
            abi.encodeWithSelector(
                IERC20_minimal.approve.selector, ics20Transfer, transferMsg.amount
            )
        );
        if (!ok) revert ApproveFailed();

        // 3) Submit the packet — we are the packet's sender, so the eventual
        //    callback fires here.
        sequence = IICS20Transfer_minimal(ics20Transfer).sendTransfer(transferMsg);

        // 4) Register the follow-up. Reject duplicates within the same
        //    (sourceClient, sequence) slot defensively, although the router
        //    won't reuse sequences.
        if (programs[transferMsg.sourceClient][sequence].length != 0) revert AlreadyComposed();
        programs[transferMsg.sourceClient][sequence] = program;

        emit Composed(transferMsg.sourceClient, sequence, program.length);
    }

    /// @notice ICS20Transfer-invoked ack callback.
    function onAckPacket(
        bool success,
        IIBCAppCallbacks_minimal.OnAcknowledgementPacketCallback calldata msg_
    ) external onlyTransfer nonReentrant {
        _runRegistered(msg_.sourceClient, msg_.sequence, success);
        emit AckExecuted(msg_.sourceClient, msg_.sequence, success);
    }

    /// @notice ICS20Transfer-invoked timeout callback.
    function onTimeoutPacket(
        IIBCAppCallbacks_minimal.OnTimeoutPacketCallback calldata msg_
    ) external onlyTransfer nonReentrant {
        _runRegistered(msg_.sourceClient, msg_.sequence, false);
        emit TimeoutExecuted(msg_.sourceClient, msg_.sequence);
    }

    /// @dev Look up the registered program, stage params in transient
    ///      storage, DELEGATECALL the interpreter, then clear both storage
    ///      and transient slots.
    function _runRegistered(string calldata sourceClient, uint64 sequence, bool success) internal {
        bytes memory program = programs[sourceClient][sequence];
        if (program.length == 0) return;
        delete programs[sourceClient][sequence];

        uint256 successUint = success ? 1 : 0;
        assembly {
            tstore(0, successUint)
            tstore(1, sequence)
        }

        // Inline the InterpreterRunner._runProgram body so we can pass memory
        // bytes (we already loaded it from storage).
        address _interpreter = interpreter;
        assembly {
            let ptr := add(program, 32)
            let len := mload(program)
            let ok := delegatecall(gas(), _interpreter, ptr, len, 0, 0)
            if iszero(ok) {
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
            tstore(0, 0)
            tstore(1, 0)
        }
    }

}

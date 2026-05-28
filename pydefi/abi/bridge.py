"""
Bridge contract ABI definitions.

All human-readable ABI fragments and pre-built :class:`~eth_contract.Contract`
objects for cross-chain bridge protocols are defined here so that they can be
imported from a single location.  Bind a contract to a specific on-chain
address at the call site::

    from pydefi.abi.bridge import CCTP_TOKEN_MESSENGER_V2

    messenger = CCTP_TOKEN_MESSENGER_V2(to="0xMessenger...")
    await messenger.fns.depositForBurn(...).transact(w3, account)
"""

from __future__ import annotations

from typing import Annotated

from eth_contract import ABIStruct, Contract

# ---------------------------------------------------------------------------
# Circle CCTP v2
# ---------------------------------------------------------------------------

CCTP_TOKEN_MESSENGER_V2 = Contract.from_abi(
    [
        # depositForBurn — standard transfer (no compose hook)
        "function depositForBurn(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold) external",
        # depositForBurnWithHook — compose transfer; DeFiVM program passed as hookData
        "function depositForBurnWithHook(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold, bytes calldata hookData) external",
    ]
)

# ---------------------------------------------------------------------------
# GasZip
# ---------------------------------------------------------------------------

GASZIP = Contract.from_abi(
    [
        "function deposit(uint256 to, uint256[] calldata chains) external payable",
    ]
)

# ---------------------------------------------------------------------------
# LayerZero OFT v2 — ABI struct definitions
# ---------------------------------------------------------------------------


class OFTSendParam(ABIStruct):
    """SendParam struct for LayerZero OFT ``quoteSend`` and ``send``."""

    dstEid: Annotated[int, "uint32"]
    to: Annotated[bytes, "bytes32"]
    amountLD: Annotated[int, "uint256"]
    minAmountLD: Annotated[int, "uint256"]
    extraOptions: Annotated[bytes, "bytes"]
    composeMsg: Annotated[bytes, "bytes"]
    oftCmd: Annotated[bytes, "bytes"]


class MessagingFee(ABIStruct):
    """MessagingFee struct for LayerZero OFT ``send``."""

    nativeFee: Annotated[int, "uint256"]
    lzTokenFee: Annotated[int, "uint256"]


# ---------------------------------------------------------------------------
# LayerZero OFT v2 — Contract object
# ---------------------------------------------------------------------------

LAYERZERO_OFT = Contract.from_abi(
    OFTSendParam.human_readable_abi()
    + MessagingFee.human_readable_abi()
    + [
        # quoteSend(SendParam, payInLzToken) -> MessagingFee
        "function quoteSend(OFTSendParam _sendParam, bool _payInLzToken) external view returns (uint256 nativeFee, uint256 lzTokenFee)",
        # send(SendParam, MessagingFee, refundAddress) -> (MessagingReceipt, OFTReceipt)
        "function send(OFTSendParam _sendParam, MessagingFee _fee, address _refundAddress) external payable",
    ]
)

# ---------------------------------------------------------------------------
# Stargate Finance
# ---------------------------------------------------------------------------

STARGATE_ROUTER = Contract.from_abi(
    [
        "function swap(uint16 _dstChainId, uint256 _srcPoolId, uint256 _dstPoolId, address payable _refundAddress, uint256 _amountLD, uint256 _minAmountLD, (uint256 dstGasForCall, uint256 dstNativeAmount, bytes dstNativeAddr) _lzTxParams, bytes calldata _to, bytes calldata _payload) external payable",
        "function quoteLayerZeroFee(uint16 _dstChainId, uint8 _functionType, bytes calldata _toAddress, bytes calldata _transferAndCallPayload, (uint256 dstGasForCall, uint256 dstNativeAmount, bytes dstNativeAddr) _lzTxParams) external view returns (uint256, uint256)",
    ]
)

STARGATE_POOL = Contract.from_abi(
    [
        "function amountLPtoLD(uint256 _amountLP) external view returns (uint256)",
        "function totalLiquidity() external view returns (uint256)",
        "function totalSupply() external view returns (uint256)",
        "function deltaCredit() external view returns (uint256)",
    ]
)

STARGATE_FACTORY = Contract.from_abi(
    [
        "function getPool(uint256 _poolId) external view returns (address)",
    ]
)

# ---------------------------------------------------------------------------
# Mayan Finance — ABI struct definitions
# ---------------------------------------------------------------------------


class MayanSwiftOrderParams(ABIStruct):
    """OrderParams struct for ``MayanSwift.createOrderWithToken`` (V2)."""

    payloadType: Annotated[int, "uint8"]
    trader: Annotated[bytes, "bytes32"]
    destAddr: Annotated[bytes, "bytes32"]
    destChainId: Annotated[int, "uint16"]
    referrerAddr: Annotated[bytes, "bytes32"]
    tokenOut: Annotated[bytes, "bytes32"]
    minAmountOut: Annotated[int, "uint64"]
    gasDrop: Annotated[int, "uint64"]
    cancelFee: Annotated[int, "uint64"]
    refundFee: Annotated[int, "uint64"]
    deadline: Annotated[int, "uint64"]
    referrerBps: Annotated[int, "uint8"]
    auctionMode: Annotated[int, "uint8"]
    random: Annotated[bytes, "bytes32"]


# ---------------------------------------------------------------------------
# Mayan Finance — Contract objects
# ---------------------------------------------------------------------------

MAYAN_FORWARDER = Contract.from_abi(
    [
        "function forwardEth(address mayanProtocol, bytes protocolData) external payable",
        "function swapAndForwardEth("
        "  uint256 amountIn,"
        "  address swapProtocol,"
        "  bytes swapData,"
        "  address middleToken,"
        "  uint256 minMiddleAmount,"
        "  address mayanProtocol,"
        "  bytes mayanData"
        ") external payable",
    ]
)

MAYAN_SWIFT_V2 = Contract.from_abi(
    MayanSwiftOrderParams.human_readable_abi()
    + [
        "function createOrderWithToken("
        "  address tokenIn,"
        "  uint256 amountIn,"
        "  MayanSwiftOrderParams params,"
        "  bytes customPayload"
        ") external returns (bytes32 orderHash)",
    ]
)

# ---------------------------------------------------------------------------
# Across Protocol
# ---------------------------------------------------------------------------

ACROSS_SPOKE_POOL = Contract.from_abi(
    [
        "function depositV3(address depositor, address recipient, address inputToken, address outputToken, uint256 inputAmount, uint256 outputAmount, uint256 destinationChainId, address exclusiveRelayer, uint32 quoteTimestamp, uint32 fillDeadline, uint32 exclusivityDeadline, bytes calldata message) external payable",
        "function getCurrentTime() external view returns (uint256)",
    ]
)

# ---------------------------------------------------------------------------
# Chainlink CCIP — ABI struct definitions
# ---------------------------------------------------------------------------


class CCIPEVMTokenAmount(ABIStruct):
    """``Client.EVMTokenAmount`` — one (token, amount) entry in ``EVM2AnyMessage``."""

    token: Annotated[str, "address"]
    amount: Annotated[int, "uint256"]


class CCIPEVM2AnyMessage(ABIStruct):
    """``Client.EVM2AnyMessage`` — payload to ``IRouterClient.ccipSend`` / ``getFee``.

    Fields:
        receiver: ``abi.encode(<destination address>)`` — for EVM destinations
            this is a 32-byte word containing the address right-aligned.
        data: Arbitrary destination payload (empty for plain token transfers;
            DeFiVM bytecode for compose flows).
        tokenAmounts: List of tokens and amounts to transfer.  Each entry's
            token must be approved to the Router by the sender.
        feeToken: ERC-20 fee token, or ``address(0)`` to pay the message fee
            in native gas via ``msg.value``.
        extraArgs: Tag-prefixed extra arguments.  For v2 lanes this is
            ``EVM_EXTRA_ARGS_V2_TAG (0x181dcf10)`` followed by
            ``abi.encode(uint256 gasLimit, bool allowOutOfOrderExecution)``.
            Empty falls back to the lane default.
    """

    receiver: Annotated[bytes, "bytes"]
    data: Annotated[bytes, "bytes"]
    tokenAmounts: list[CCIPEVMTokenAmount]
    feeToken: Annotated[str, "address"]
    extraArgs: Annotated[bytes, "bytes"]


# ---------------------------------------------------------------------------
# Chainlink CCIP — Router contract
# ---------------------------------------------------------------------------

CCIP_ROUTER = Contract.from_abi(
    CCIPEVMTokenAmount.human_readable_abi()
    + CCIPEVM2AnyMessage.human_readable_abi()
    + [
        # getFee(destChainSelector, message) -> uint256 fee
        "function getFee(uint64 destinationChainSelector, CCIPEVM2AnyMessage message) external view returns (uint256 fee)",
        # ccipSend(destChainSelector, message) payable -> bytes32 messageId
        "function ccipSend(uint64 destinationChainSelector, CCIPEVM2AnyMessage message) external payable returns (bytes32)",
    ]
)


# Lucid Labs AssetController. Tokens move 1:1; native fee is msg.value
# forwarded to the adapter via IBaseAdapter.relayMessage{value: msg.value}.
LUCID_ASSET_CONTROLLER = Contract.from_abi(
    [
        "function transferTo(address recipient, uint256 amount, bool unwrap, uint256 destChainId, address bridgeAdapter, bytes bridgeOptions) external payable",
        "function token() external view returns (address)",
        "function getControllerForChain(uint256 destChainId) external view returns (address)",
        "function transfersPausedTo(uint256 destChainId) external view returns (bool)",
        "function paused() external view returns (bool)",
        "function minBridges() external view returns (uint256)",
        "function allowTokenUnwrapping() external view returns (bool)",
    ]
)

# Lucid bridge adapters. Only gasLimit width differs (LZ: uint128, HL: uint256);
# the same difference shows up in the bridgeOptions encoding in lucid.py.
LUCID_HYPERLANE_ADAPTER = Contract.from_abi(
    [
        "function quoteMessage(address destination, uint256 chainId, uint256 gasLimit, bytes message, bool includeFee) external view returns (uint256)",
        "function paused() external view returns (bool)",
    ]
)

LUCID_LAYERZERO_ADAPTER = Contract.from_abi(
    [
        "function quoteMessage(address destination, uint256 chainId, uint128 gasLimit, bytes message, bool includeFee) external view returns (uint256)",
        "function isChainIdSupported(uint256 chainId) external view returns (bool)",
        "function paused() external view returns (bool)",
    ]
)

# ---------------------------------------------------------------------------
# IBC v2 (Eureka) — ICS20Transfer + EurekaComposer
# ---------------------------------------------------------------------------
# Field order matches solidity-ibc-eureka/contracts/msgs/IICS20TransferMsgs.sol.

ICS20_DEFAULT_PORT = "transfer"


class ICS20SendTransferMsg(ABIStruct):
    """``IICS20TransferMsgs.SendTransferMsg`` — input to ``sendTransfer`` /
    ``sendTransferAndCompose``."""

    denom: Annotated[str, "address"]
    amount: Annotated[int, "uint256"]
    receiver: Annotated[str, "string"]
    sourceClient: Annotated[str, "string"]
    destPort: Annotated[str, "string"]
    timeoutTimestamp: Annotated[int, "uint64"]
    memo: Annotated[str, "string"]


ICS20_TRANSFER = Contract.from_abi(
    ICS20SendTransferMsg.human_readable_abi()
    + [
        "function sendTransfer(ICS20SendTransferMsg msg_) external returns (uint64)",
    ]
)

EUREKA_COMPOSER = Contract.from_abi(
    ICS20SendTransferMsg.human_readable_abi()
    + [
        "function sendTransferAndCompose(ICS20SendTransferMsg msg_, bytes program) external returns (uint64)",
    ]
)

# ERC-165 ``IIBCSenderCallbacks`` id (onAckPacket ^ onTimeoutPacket selectors).
IIBC_SENDER_CALLBACKS_INTERFACE_ID: bytes = bytes.fromhex("d3ce6f1b")

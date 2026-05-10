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

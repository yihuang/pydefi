"""Minimal OKX DexRouter ABI fragments for the gas benchmark.

Pool word for ``uniswapV3SwapTo`` (``uint256`` per pool):
  bits 0..159 pool, bit 255 ONE_FOR_ZERO, bit 253 WETH_UNWRAP.
Pool word for ``unxswapByOrderId`` (``bytes32`` per pool):
  bits 0..159 pool, bit 255 REVERSE, bit 254 WETH, bit 253 WETH_UNWRAP.
``receiver`` is also packed: low 160 bits = recipient, upper 96 = orderId.
"""

from __future__ import annotations

from typing import Annotated

from eth_contract import ABIStruct, Contract

# Encoding masks (mirrored from CommonUtils.sol).
ADDRESS_MASK: int = (1 << 160) - 1
ONE_FOR_ZERO_MASK: int = 1 << 255
REVERSE_MASK: int = 1 << 255

# RouterPath.rawData: bits 0..159 pool | 160..175 weight (bps) | bit 255 reverse.
WEIGHT_SHIFT: int = 160

# DagRouter.rawData extends RouterPath.rawData with bits 176..183 outputIndex
# (which DAG node this edge feeds into) and bits 184..191 inputIndex (which
# node this edge consumes from). See DagRouter.sol::_exeNode.
OUTPUT_INDEX_SHIFT: int = 176
INPUT_INDEX_SHIFT: int = 184


class BaseRequest(ABIStruct):
    """``fromToken`` is uint256 (upper bits = commission flags); pass plain
    address as uint160."""

    fromToken: Annotated[int, "uint256"]
    toToken: Annotated[str, "address"]
    fromTokenAmount: Annotated[int, "uint256"]
    minReturnAmount: Annotated[int, "uint256"]
    deadLine: Annotated[int, "uint256"]


class RouterPath(ABIStruct):
    mixAdapters: Annotated[list[str], "address[]"]
    assetTo: Annotated[list[str], "address[]"]
    rawData: Annotated[list[int], "uint256[]"]
    extraData: Annotated[list[bytes], "bytes[]"]
    fromToken: Annotated[int, "uint256"]


class PMMSwapRequest(ABIStruct):
    """Unused — encoders pass an empty array."""

    pathIndex: Annotated[int, "uint256"]
    payer: Annotated[str, "address"]
    fromToken: Annotated[str, "address"]
    toToken: Annotated[str, "address"]
    fromTokenAmountMax: Annotated[int, "uint256"]
    toTokenAmountMax: Annotated[int, "uint256"]
    salt: Annotated[int, "uint256"]
    deadLine: Annotated[int, "uint256"]
    isPushOrder: Annotated[bool, "bool"]
    extension: Annotated[bytes, "bytes"]


OKX_DEX_ROUTER = Contract.from_abi(
    BaseRequest.human_readable_abi()
    + RouterPath.human_readable_abi()
    + PMMSwapRequest.human_readable_abi()
    + [
        "function uniswapV3SwapTo(uint256 receiver, uint256 amount, uint256 minReturn, uint256[] pools) external payable returns (uint256 returnAmount)",
        "function unxswapByOrderId(uint256 srcToken, uint256 amount, uint256 minReturn, bytes32[] pools) external payable returns (uint256 returnAmount)",
        "function smartSwapByOrderId(uint256 orderId, BaseRequest baseRequest, uint256[] batchesAmount, RouterPath[][] batches, PMMSwapRequest[] extraData) external payable returns (uint256 returnAmount)",
        "function smartSwapTo(uint256 orderId, address receiver, BaseRequest baseRequest, uint256[] batchesAmount, RouterPath[][] batches, PMMSwapRequest[] extraData) external payable returns (uint256 returnAmount)",
        "function dagSwapByOrderId(uint256 orderId, BaseRequest baseRequest, RouterPath[] paths) external payable returns (uint256 returnAmount)",
        "function dagSwapTo(uint256 orderId, address receiver, BaseRequest baseRequest, RouterPath[] paths) external payable returns (uint256 returnAmount)",
    ]
)

OKX_TOKEN_APPROVE_PROXY = Contract.from_abi(
    [
        "function claimTokens(address token, address who, address dest, uint256 amount) external",
    ]
)


# Mainnet (DEPLOYMENT.md). TOKEN_APPROVE is the address users approve —
# the underlying TokenApprove, NOT the TokenApproveProxy column. Queried via
# TokenApproveProxy.tokenApprove() and pinned (immutable on mainnet).
OKX_DEX_ROUTER_ETHEREUM: str = "0x5E1f62Dac767b0491e3CE72469C217365D5B48cC"
OKX_TOKEN_APPROVE_ETHEREUM: str = "0x40aA958dd87FC8305b97f2BA922CDdCa374bcD7f"

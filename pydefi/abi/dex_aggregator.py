"""
OKX DexRouter contract ABI definitions for DAG swap aggregation.

The OKX DexRouter (``DexRouter.sol``, also known as DEX Router V1) is a
multi-protocol DEX aggregation contract used by the OKX DEX platform across
many EVM chains.  Its ``dagSwapTo`` entry point accepts a :class:`BaseRequest`
and a list of :class:`RouterPath` structs that encode a Directed Acyclic Graph
(DAG) of swap edges, each calling an adapter contract for execution.

Usage::

    from pydefi.abi.dex_aggregator import OKX_DEX_ROUTER, BaseRequest, RouterPath
    from pydefi._utils import decode_address

    router = OKX_DEX_ROUTER(to=dex_router_address)
    calldata = router.fns.dagSwapTo(
        order_id,
        receiver,
        (from_token_packed, to_token_addr, amount_in, min_out, deadline),
        [(adapters, asset_tos, raw_datas, extra_datas, from_token_packed)],
    ).data
"""

from __future__ import annotations

from typing import Annotated

from eth_contract import ABIStruct, Contract

# ---------------------------------------------------------------------------
# Bit-mask constants (from CommonUtils.sol / DagRouter.sol)
# ---------------------------------------------------------------------------

ADDRESS_MASK: int = (1 << 160) - 1
ONE_FOR_ZERO_MASK: int = 1 << 255
REVERSE_MASK: int = 1 << 255
WEIGHT_SHIFT: int = 160
WEIGHT_MASK: int = 0xFFFF << 160
OUTPUT_INDEX_SHIFT: int = 176
OUTPUT_INDEX_MASK: int = 0xFF << 176
INPUT_INDEX_SHIFT: int = 184
INPUT_INDEX_MASK: int = 0xFF << 184

# ---------------------------------------------------------------------------
# Struct definitions
# ---------------------------------------------------------------------------


class BaseRequest(ABIStruct):
    """Parameters for a swap via the OKX DexRouter.

    ``fromToken`` is a ``uint256`` packing the token address in the low 160
    bits and optional transfer-mode flags in the high bits
    (``_MODE_NO_TRANSFER``, ``_MODE_BY_INVEST``, ``_MODE_PERMIT2``).
    """

    fromToken: Annotated[int, "uint256"]
    toToken: Annotated[str, "address"]
    fromTokenAmount: Annotated[int, "uint256"]
    minReturnAmount: Annotated[int, "uint256"]
    deadLine: Annotated[int, "uint256"]


class RouterPath(ABIStruct):
    """A DAG node in the OKX DexRouter route encoding.

    Each ``RouterPath`` represents one *node* in the DAG.  The
    parallel arrays ``mixAdapters`` / ``rawData`` / ``extraData`` /
    ``assetTo`` define the node's *edges*.  The router splits the
    node's input balance among edges according to the weight encoded
    in ``rawData``, transfers each split to ``assetTo[i]``, then
    calls ``mixAdapters[i]`` with the encoded parameters.

    ``fromToken`` packs the source token address in the low 160 bits
    and optional transfer-mode flags above.
    """

    mixAdapters: Annotated[list[str], "address[]"]
    """Adapter contract addresses, one per edge."""

    assetTo: Annotated[list[str], "address[]"]
    """Destination for each edge's pre-swap token transfer."""

    rawData: Annotated[list[int], "uint256[]"]
    """Encoded edge parameters (pool address, direction, weight, indices)."""

    extraData: Annotated[list[bytes], "bytes[]"]
    """Extra data passed to each adapter's ``sellBase`` / ``sellQuote`` call."""

    fromToken: Annotated[int, "uint256"]
    """Packed source token for this node."""


class PMMSwapRequest(ABIStruct):
    """PMM order struct — pass an empty array when unused."""

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


# ---------------------------------------------------------------------------
# Contract objects
# ---------------------------------------------------------------------------

OKX_DEX_ROUTER = Contract.from_abi(
    BaseRequest.human_readable_abi()
    + RouterPath.human_readable_abi()
    + PMMSwapRequest.human_readable_abi()
    + [
        "function uniswapV3SwapTo(uint256 receiver, uint256 amount, uint256 minReturn, uint256[] pools) external payable returns (uint256 returnAmount)",
        "function smartSwapTo(uint256 orderId, address receiver, BaseRequest baseRequest, uint256[] batchesAmount, RouterPath[][] batches, PMMSwapRequest[] extraData) external payable returns (uint256 returnAmount)",
        "function dagSwapTo(uint256 orderId, address receiver, BaseRequest baseRequest, RouterPath[] paths) external payable returns (uint256 returnAmount)",
        "function dagSwapByOrderId(uint256 orderId, BaseRequest baseRequest, RouterPath[] paths) external payable returns (uint256 returnAmount)",
    ]
)

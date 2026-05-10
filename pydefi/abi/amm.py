"""
AMM contract ABI definitions.

All human-readable ABI fragments and pre-built :class:`~eth_contract.Contract`
objects for AMM protocols are defined here so that they can be imported from a
single location.  Bind a contract to a specific on-chain address at the call
site::

    from pydefi.abi.amm import UNISWAP_V2_ROUTER

    router = UNISWAP_V2_ROUTER(to="0xRouter...")
    amounts = await router.fns.getAmountsOut(amount_in, path).call(w3)
"""

from __future__ import annotations

from typing import Annotated

from eth_contract import ABIStruct, Contract

# ---------------------------------------------------------------------------
# Uniswap V2
# ---------------------------------------------------------------------------

UNISWAP_V2_ROUTER = Contract.from_abi(
    [
        "function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts)",
        "function getAmountsIn(uint amountOut, address[] calldata path) external view returns (uint[] memory amounts)",
        "function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
        "function swapTokensForExactTokens(uint amountOut, uint amountInMax, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
        "function swapExactETHForTokens(uint amountOutMin, address[] calldata path, address to, uint deadline) external payable returns (uint[] memory amounts)",
        "function swapExactTokensForETH(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
        "function factory() external pure returns (address)",
        "function WETH() external pure returns (address)",
    ]
)

UNISWAP_V2_FACTORY = Contract.from_abi(
    [
        "function getPair(address tokenA, address tokenB) external view returns (address pair)",
        "function allPairs(uint) external view returns (address pair)",
        "function allPairsLength() external view returns (uint)",
        "event PairCreated(address indexed token0, address indexed token1, address pair, uint256)",
    ]
)

UNISWAP_V2_PAIR = Contract.from_abi(
    [
        "function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)",
        "function token0() external view returns (address)",
        "function token1() external view returns (address)",
        "event Sync(uint112 reserve0, uint112 reserve1)",
    ]
)

# ---------------------------------------------------------------------------
# Uniswap V3 — ABI struct definitions
# ---------------------------------------------------------------------------


class QuoteExactInputSingleParams(ABIStruct):
    """Params struct for ``QuoterV2.quoteExactInputSingle``."""

    tokenIn: Annotated[str, "address"]
    tokenOut: Annotated[str, "address"]
    amountIn: Annotated[int, "uint256"]
    fee: Annotated[int, "uint24"]
    sqrtPriceLimitX96: Annotated[int, "uint160"]


class QuoteExactOutputSingleParams(ABIStruct):
    """Params struct for ``QuoterV2.quoteExactOutputSingle``."""

    tokenIn: Annotated[str, "address"]
    tokenOut: Annotated[str, "address"]
    amount: Annotated[int, "uint256"]
    fee: Annotated[int, "uint24"]
    sqrtPriceLimitX96: Annotated[int, "uint160"]


class ExactInputSingleParams(ABIStruct):
    """Params struct for ``SwapRouter.exactInputSingle``."""

    tokenIn: Annotated[str, "address"]
    tokenOut: Annotated[str, "address"]
    fee: Annotated[int, "uint24"]
    recipient: Annotated[str, "address"]
    deadline: Annotated[int, "uint256"]
    amountIn: Annotated[int, "uint256"]
    amountOutMinimum: Annotated[int, "uint256"]
    sqrtPriceLimitX96: Annotated[int, "uint160"]


class ExactInputParams(ABIStruct):
    """Params struct for ``SwapRouter.exactInput``."""

    path: Annotated[bytes, "bytes"]
    recipient: Annotated[str, "address"]
    deadline: Annotated[int, "uint256"]
    amountIn: Annotated[int, "uint256"]
    amountOutMinimum: Annotated[int, "uint256"]


class ExactOutputSingleParams(ABIStruct):
    """Params struct for ``SwapRouter.exactOutputSingle``."""

    tokenIn: Annotated[str, "address"]
    tokenOut: Annotated[str, "address"]
    fee: Annotated[int, "uint24"]
    recipient: Annotated[str, "address"]
    deadline: Annotated[int, "uint256"]
    amountOut: Annotated[int, "uint256"]
    amountInMaximum: Annotated[int, "uint256"]
    sqrtPriceLimitX96: Annotated[int, "uint160"]


# ---------------------------------------------------------------------------
# Uniswap V3 — Contract objects
# ---------------------------------------------------------------------------

UNISWAP_V3_QUOTER_V2 = Contract.from_abi(
    QuoteExactInputSingleParams.human_readable_abi()
    + QuoteExactOutputSingleParams.human_readable_abi()
    + [
        "function quoteExactInputSingle(QuoteExactInputSingleParams params) external returns (uint256 amountOut, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate)",
        "function quoteExactOutputSingle(QuoteExactOutputSingleParams params) external returns (uint256 amountIn, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate)",
        "function quoteExactInput(bytes path, uint256 amountIn) external returns (uint256 amountOut, uint160[] sqrtPriceX96AfterList, uint32[] initializedTicksCrossedList, uint256 gasEstimate)",
    ]
)

UNISWAP_V3_ROUTER = Contract.from_abi(
    ExactInputSingleParams.human_readable_abi()
    + ExactInputParams.human_readable_abi()
    + ExactOutputSingleParams.human_readable_abi()
    + [
        "function exactInputSingle(ExactInputSingleParams params) external payable returns (uint256 amountOut)",
        "function exactInput(ExactInputParams params) external payable returns (uint256 amountOut)",
        "function exactOutputSingle(ExactOutputSingleParams params) external payable returns (uint256 amountIn)",
    ]
)

UNISWAP_V3_FACTORY = Contract.from_abi(
    [
        "function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)",
        "event PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool)",
    ]
)

UNISWAP_V3_POOL = Contract.from_abi(
    [
        "function slot0() external view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)",
        "function liquidity() external view returns (uint128)",
        "function fee() external view returns (uint24)",
        "function token0() external view returns (address)",
        "function token1() external view returns (address)",
        "event Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)",
    ]
)

# ---------------------------------------------------------------------------
# Curve Finance
# ---------------------------------------------------------------------------

CURVE_POOL = Contract.from_abi(
    [
        "function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256)",
        "function get_dy_underlying(int128 i, int128 j, uint256 dx) external view returns (uint256)",
        "function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
        "function exchange_underlying(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
        "function coins(uint256 i) external view returns (address)",
        "function balances(uint256 i) external view returns (uint256)",
        "function A() external view returns (uint256)",
        "function fee() external view returns (uint256)",
    ]
)

CURVE_REGISTRY = Contract.from_abi(
    [
        "function find_pool_for_coins(address from, address to) external view returns (address)",
        "function find_pool_for_coins(address from, address to, uint256 i) external view returns (address)",
        "function get_coin_indices(address pool, address from, address to) external view returns (int128, int128, bool)",
        "function get_best_rate(address from, address to, uint256 amount) external view returns (address, uint256)",
    ]
)

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
        "function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes data)",
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
        "function swap(address recipient, bool zeroForOne, int256 amountSpecified, uint160 sqrtPriceLimitX96, bytes data) returns (int256 amount0, int256 amount1)",
        "event Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)",
    ]
)

# ---------------------------------------------------------------------------
# Uniswap V4 — Contract objects (singleton PoolManager architecture)
# ---------------------------------------------------------------------------
# V4 keeps all pool state in one ``PoolManager``.  Pools are identified by a
# ``PoolKey = (currency0, currency1, fee, tickSpacing, hooks)`` whose
# keccak256 is the 32-byte ``poolId``.  ``StateView`` is the read-only view
# over PoolManager storage; ``V4Quoter`` is the on-chain swap simulator.
# Quoter params are encoded as raw tuples (nested-struct human-readable ABI is
# ambiguous in eth_contract):
#   QuoteExactSingleParams = (PoolKey, bool zeroForOne, uint128 exactAmount, bytes hookData)

UNISWAP_V4_STATE_VIEW = Contract.from_abi(
    [
        "function getSlot0(bytes32 poolId) external view returns (uint160 sqrtPriceX96, int24 tick, uint24 protocolFee, uint24 lpFee)",
        "function getLiquidity(bytes32 poolId) external view returns (uint128 liquidity)",
    ]
)

UNISWAP_V4_QUOTER = Contract.from_abi(
    [
        "function quoteExactInputSingle(((address,address,uint24,int24,address),bool,uint128,bytes) params) external returns (uint256 amountOut, uint256 gasEstimate)",
        "function quoteExactOutputSingle(((address,address,uint24,int24,address),bool,uint128,bytes) params) external returns (uint256 amountIn, uint256 gasEstimate)",
    ]
)

# ---------------------------------------------------------------------------
# Curve Finance
# ---------------------------------------------------------------------------

CURVE_POOL = Contract.from_abi(
    [
        "function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256)",
        "function get_dx(int128 i, int128 j, uint256 dy) external view returns (uint256)",
        "function get_dy_underlying(int128 i, int128 j, uint256 dx) external view returns (uint256)",
        "function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
        "function exchange_underlying(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
        "function coins(uint256 i) external view returns (address)",
        "function balances(uint256 i) external view returns (uint256)",
        "function A() external view returns (uint256)",
        "function A_precise() external view returns (uint256)",
        "function fee() external view returns (uint256)",
        "function get_virtual_price() external view returns (uint256)",
        "function totalSupply() external view returns (uint256)",
        # Stable-NG / rate-stabilised pools
        "function N_COINS() external view returns (uint256)",
        "function stored_rates() external view returns (uint256[])",
        "function offpeg_fee_multiplier() external view returns (uint256)",
    ]
)

# Curve V2 (cryptoswap: Twocrypto / Tricrypto) view interface used to snapshot
# pool state for local pricing via :mod:`pydefi.amm.curve_math`.
CURVE_V2_POOL = Contract.from_abi(
    [
        "function get_dy(uint256 i, uint256 j, uint256 dx) external view returns (uint256)",
        "function exchange(uint256 i, uint256 j, uint256 dx, uint256 min_dy) external returns (uint256)",
        "function coins(uint256 i) external view returns (address)",
        "function balances(uint256 i) external view returns (uint256)",
        "function A() external view returns (uint256)",
        "function gamma() external view returns (uint256)",
        "function D() external view returns (uint256)",
        "function fee() external view returns (uint256)",
        "function mid_fee() external view returns (uint256)",
        "function out_fee() external view returns (uint256)",
        "function fee_gamma() external view returns (uint256)",
        "function price_scale() external view returns (uint256)",
        "function price_scale(uint256 k) external view returns (uint256)",
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

# Curve MetaRegistry aggregates every Curve registry/factory on a chain and is
# used for pool discovery.  Mainnet: 0xF98B45FA17DE75FB1aD0e7aFD971b0ca00e379fC
CURVE_METAREGISTRY = Contract.from_abi(
    [
        "function find_pools_for_coins(address from, address to) external view returns (address[])",
        "function get_n_coins(address pool) external view returns (uint256)",
        "function is_meta(address pool) external view returns (bool)",
    ]
)

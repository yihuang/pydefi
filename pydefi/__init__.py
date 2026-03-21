"""
pydefi — Modern Python library for DeFi.

Provides integrations with:
- AMM DEXes (Uniswap V2/V3, Curve)
- DEX aggregator APIs (1inch, ParaSwap, 0x)
- Cross-chain bridges (Stargate, Across)
- DEX pathfinding algorithm

Built on top of ``eth-contract`` for modern interaction with on-chain smart
contracts.

Quick-start example::

    from pydefi.amm import UniswapV2
    from pydefi.types import Token, TokenAmount, ChainId

    # Define tokens
    ETH = Token(ChainId.ETHEREUM, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH")
    USDC = Token(ChainId.ETHEREUM, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", decimals=6)

    # Create AMM client
    uniswap = UniswapV2(w3, router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")

    # Build a swap route (requires live node)
    route = await uniswap.build_swap_route(
        amount_in=TokenAmount.from_human(ETH, "1.0"),
        token_out=USDC,
    )
"""

from pydefi.exceptions import (
    AggregatorError,
    BridgeError,
    InsufficientLiquidityError,
    NoRouteFoundError,
    PydefiError,
    SlippageExceededError,
)
from pydefi.types import (
    BridgeQuote,
    ChainId,
    SwapRoute,
    SwapStep,
    SwapTransaction,
    Token,
    TokenAmount,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "ChainId",
    "Token",
    "TokenAmount",
    "SwapStep",
    "SwapRoute",
    "SwapTransaction",
    "BridgeQuote",
    # Exceptions
    "PydefiError",
    "InsufficientLiquidityError",
    "NoRouteFoundError",
    "AggregatorError",
    "BridgeError",
    "SlippageExceededError",
]

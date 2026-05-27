"""Solidity sources compiled inline by the bench harness.

OKX's canonical UniV3Adapter address isn't published, so the bench deploys
this minimal equivalent on the fork. Same external behavior as
``Web3-DEX-Router-EVM-V1/contracts/8/adapter/UniV3Adapter.sol`` for the
single-payer exact-input case: DexRouter pre-funds ``assetTo`` (this
adapter) via ``_transferInternal``, the callback pays the pool from the
adapter's own balance (no transferFrom).
"""

from __future__ import annotations

UNI_V3_ADAPTER_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

interface IUniV3Pool {
    function swap(
        address recipient,
        bool zeroForOne,
        int256 amountSpecified,
        uint160 sqrtPriceLimitX96,
        bytes calldata data
    ) external returns (int256 amount0, int256 amount1);
}

interface IERC20Min {
    function balanceOf(address) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
}

/// Minimal Uniswap V3 adapter compatible with OKX DexRouter.smartSwap.
/// extraData layout: abi.encode(uint160 sqrtX96, bytes innerData) where
/// innerData = abi.encode(address fromToken, address toToken, uint24 fee).
contract MinimalUniV3Adapter {
    uint160 internal constant MIN_SQRT_RATIO_PLUS_ONE = 4295128740;
    uint160 internal constant MAX_SQRT_RATIO_MINUS_ONE =
        1461446703485210103287273052203988822378723970341;

    function sellBase(address to, address pool, bytes calldata moreInfo) external {
        _swap(to, pool, moreInfo);
    }

    function sellQuote(address to, address pool, bytes calldata moreInfo) external {
        _swap(to, pool, moreInfo);
    }

    function _swap(address to, address pool, bytes calldata moreInfo) internal {
        (uint160 sqrtX96, bytes memory data) = abi.decode(moreInfo, (uint160, bytes));
        (address fromToken, address toToken, ) = abi.decode(data, (address, address, uint24));
        uint256 sellAmount = IERC20Min(fromToken).balanceOf(address(this));
        bool zeroForOne = fromToken < toToken;
        uint160 limit = sqrtX96 == 0
            ? (zeroForOne ? MIN_SQRT_RATIO_PLUS_ONE : MAX_SQRT_RATIO_MINUS_ONE)
            : sqrtX96;
        IUniV3Pool(pool).swap(to, zeroForOne, int256(sellAmount), limit, data);
    }

    function uniswapV3SwapCallback(
        int256 amount0Delta,
        int256 amount1Delta,
        bytes calldata data
    ) external {
        (address tokenIn, address tokenOut, ) = abi.decode(data, (address, address, uint24));
        bool payIn = amount0Delta > 0 ? (tokenIn < tokenOut) : (tokenOut < tokenIn);
        uint256 owed = amount0Delta > 0 ? uint256(amount0Delta) : uint256(amount1Delta);
        address payToken = payIn ? tokenIn : tokenOut;
        // Safe-transfer: low-level call so non-standard tokens (USDT returns
        // nothing from transfer) don't revert the decoder.
        (bool ok, bytes memory ret) = payToken.call(
            abi.encodeWithSelector(IERC20Min.transfer.selector, msg.sender, owed)
        );
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "transfer failed");
    }
}
"""

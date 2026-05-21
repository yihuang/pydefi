// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title DEXCallbackRouter
 * @notice Mixin that answers V2/V3 DEX swap callbacks for the inheriting
 *         contract.  Programs run in the caller's context (DELEGATECALL),
 *         so any pool callback comes back to whichever contract initiated
 *         the swap (DeFiVM, CCTPComposer, OFTComposer, ApproveProxy);
 *         inheriting this keeps the routing logic in one place.
 *
 * ``data`` encoding (set by the program when calling pool.swap):
 *  • V3 (uniswapV3 / algebra / pancakeV3 / solidlyV3):
 *      ``abi.encode(address tokenIn)`` — positive delta is paid to the pool.
 *  • V2 (uniswapV2Call / Aerodrome hook / ramsesV2FlashCallback):
 *      ``abi.encode(address tokenIn, uint256 amountOwed)`` — ``amountOwed``
 *      is paid to the pool.
 *
 * No caller whitelist; safety relies on the program being atomic and
 * leaving no balance or allowance behind.  Simulate before broadcasting.
 */
abstract contract DEXCallbackRouter {
    // -------------------------------------------------------------------------
    // DEX callback selectors (keccak256 of the function signature, first 4 bytes)
    // -------------------------------------------------------------------------

    /// @dev uniswapV3SwapCallback(int256,int256,bytes)
    bytes4 private constant SEL_V3_CALLBACK      = 0xfa461e33;
    /// @dev algebraSwapCallback(int256,int256,bytes)  — QuickSwap / Algebra CLMM
    bytes4 private constant SEL_ALGEBRA_CALLBACK = 0x2c8958f6;
    /// @dev pancakeV3SwapCallback(int256,int256,bytes) — PancakeSwap V3
    bytes4 private constant SEL_PANCAKE_V3       = 0x23a69e75;
    /// @dev solidlyV3SwapCallback(int256,int256,bytes) — Solidly V3
    bytes4 private constant SEL_SOLIDLY_V3       = 0x3a1c453c;
    /// @dev uniswapV2Call(address,uint256,uint256,bytes)
    bytes4 private constant SEL_V2_CALLBACK      = 0x10d1e85c;
    /// @dev hook(address,uint256,uint256,bytes) — Aerodrome / Velodrome
    bytes4 private constant SEL_AERODROME_HOOK   = 0x9a7bff79;
    /// @dev ramsesV2FlashCallback(uint256,uint256,bytes) — Ramses V2
    bytes4 private constant SEL_RAMSES_V2        = 0xde5f4ecc;

    /// @dev ERC-20 transfer(address,uint256) selector
    bytes4 private constant TRANSFER_SEL = 0xa9059cbb;

    /**
     * @notice Universal DEX swap callback dispatcher.  Routes incoming
     *         callbacks from DEX pools to the appropriate payment handler
     *         based on the 4-byte function selector.  Unknown selectors
     *         revert to avoid silently accepting unexpected calls.
     */
    fallback() external virtual {
        bytes4 sel;
        assembly {
            sel := calldataload(0)
        }

        if (
            sel == SEL_V3_CALLBACK ||
            sel == SEL_ALGEBRA_CALLBACK ||
            sel == SEL_PANCAKE_V3 ||
            sel == SEL_SOLIDLY_V3
        ) {
            // V3-style: uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes data)
            // Calldata layout (bytes, 0-indexed):
            //   [0..4)   selector
            //   [4..36)  int256 amount0Delta
            //   [36..68) int256 amount1Delta
            //   [68..100) uint256 data_offset  (relative to byte 4, typically 0x60)
            //   [4+data_offset..+32) uint256 data_length
            //   [4+data_offset+32..) bytes  data_content  = abi.encode(tokenIn)
            int256 amount0Delta;
            int256 amount1Delta;
            address tokenIn;
            assembly {
                amount0Delta := calldataload(4)
                amount1Delta := calldataload(36)
                let dataRelOff := calldataload(68)
                tokenIn := calldataload(add(add(4, dataRelOff), 32))
            }
            int256 amount = amount0Delta > 0 ? amount0Delta : amount1Delta;
            if (amount > 0) {
                _callTransfer(tokenIn, msg.sender, uint256(amount));
            }
        } else if (sel == SEL_V2_CALLBACK || sel == SEL_AERODROME_HOOK) {
            // V2-style: uniswapV2Call(address sender, uint256 amount0, uint256 amount1, bytes data)
            //           hook(address sender, uint256 amount0, uint256 amount1, bytes data)
            address tokenIn;
            uint256 amountOwed;
            assembly {
                let dataRelOff := calldataload(100)
                let dataStart  := add(add(4, dataRelOff), 32)
                tokenIn    := calldataload(dataStart)
                amountOwed := calldataload(add(dataStart, 32))
            }
            if (amountOwed > 0) {
                _callTransfer(tokenIn, msg.sender, amountOwed);
            }
        } else if (sel == SEL_RAMSES_V2) {
            // Ramses V2: ramsesV2FlashCallback(uint256 amount0, uint256 amount1, bytes data)
            address tokenIn;
            uint256 amountOwed;
            assembly {
                let dataRelOff := calldataload(68)
                let dataStart  := add(add(4, dataRelOff), 32)
                tokenIn    := calldataload(dataStart)
                amountOwed := calldataload(add(dataStart, 32))
            }
            if (amountOwed > 0) {
                _callTransfer(tokenIn, msg.sender, amountOwed);
            }
        } else {
            revert("DEXCallbackRouter: unknown callback selector");
        }
    }

    /**
     * @dev Perform ``token.transfer(to, amount)`` and revert on failure.
     *
     * Supports both standard ERC-20s (that return ``bool``) and non-standard
     * tokens that return no value (e.g. USDT on mainnet).
     */
    function _callTransfer(address token, address to, uint256 amount) internal {
        (bool ok, bytes memory ret) = token.call(abi.encodeWithSelector(TRANSFER_SEL, to, amount));
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "DEXCallbackRouter: transfer failed");
    }
}

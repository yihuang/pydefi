// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title DeFiVM
 * @notice A minimal, stateless executor for composable DeFi flows expressed as raw EVM bytecode.
 *
 * Design principles
 * -----------------
 *  * Atomic execution  - a "program" runs all-at-once; any revert undoes everything.
 *  * Native EVM stack  - programs ARE raw EVM bytecode; the EVM stack is the program stack.
 *  * Fully stateless   - no owner, no whitelist; execute() can run any bytecode.
 *
 * How it works
 * ------------
 *  ``execute()`` delegates execution to the pre-deployed Analog-Labs EVM interpreter
 *  (https://github.com/Analog-Labs/evm-interpreter) via ``DELEGATECALL``.  The interpreter
 *  accepts the program bytecode as calldata and executes it in DeFiVM's context:
 *  - ``address(this)`` inside the program is DeFiVM's address.
 *  - External ``CALL``s originate from DeFiVM (msg.sender to sub-calls is DeFiVM).
 *  - ETH held by DeFiVM is forwarded via ``callvalue()`` and available to ``CALL``.
 *  - No contract deployment (CREATE) required — no nonce increase.
 *
 * Memory conventions (inside programs)
 * -------------------------------------
 *  Programs execute in a fresh virtual memory context provided by the interpreter.
 *  Recommended layout (used by the Python DSL):
 *  Registers  : memory[0x80 + i*32] for i in 0..15  (16 x 32-byte slots)
 *  Free memory: memory[0x40] tracks the next free byte (initialised to 0x280 on first use)
 *  Dynamic buf: allocated starting from memory[0x280] upward
 *
 * Security assumptions
 * --------------------
 *  1. Never approve tokens directly to this contract.  Approvals can be drained by
 *     any caller because ``execute`` is permissionless.  Use ``ApproveProxy``
 *     (see ``ApproveProxy.sol``) or permit signatures instead.
 *  2. Do not leave token or ETH balances in this contract between transactions.
 *  3. Verify every address in a program and simulate off-chain before broadcasting.
 *  4. Programs run via DELEGATECALL and have full access to DeFiVM's storage.
 *
 * Flash-swap callbacks
 * --------------------
 *  The ``fallback()`` function handles DEX swap callbacks so that DeFiVM programs
 *  can initiate flash swaps on Uniswap V2/V3 and compatible forks.  When a pool
 *  calls back into this contract during a flash swap, the fallback reads the
 *  ``data`` field, decodes the token to repay, and transfers the owed amount
 *  back to the pool (``msg.sender``).
 *
 *  Encoding conventions for the ``data`` parameter:
 *
 *  • V3-style callbacks (uniswapV3SwapCallback / algebraSwapCallback /
 *    pancakeV3SwapCallback / solidlyV3SwapCallback):
 *      ``data = abi.encode(address tokenIn)``
 *    The positive delta (amount0Delta or amount1Delta) is forwarded to the pool.
 *
 *  • V2-style callbacks (uniswapV2Call / Aerodrome hook / ramsesV2FlashCallback):
 *      ``data = abi.encode(address tokenIn, uint256 amountOwed)``
 *    Exactly ``amountOwed`` tokens are forwarded to the pool.
 *
 *  There is no caller whitelist.  Safety relies on the program being atomic: it
 *  must leave no token balance or allowance on this contract after execution.
 *  Always simulate and verify a program before broadcasting.
 */
contract DeFiVM {
    /// @dev Well-known Analog-Labs EVM interpreter.
    address private constant DEFAULT_INTERPRETER = 0x0000000000001e3F4F615cd5e20c681Cf7d85e8D;

    /// @dev Address of the EVM interpreter used for DELEGATECALL execution.
    address private immutable INTERPRETER;

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

    /// @param interpreter Address of the EVM interpreter to use.  Pass
    ///   ``address(0)`` to use the pre-deployed Analog-Labs interpreter at
    ///   ``0x0000000000001e3F4F615cd5e20c681Cf7d85e8D``.  Supply a custom
    ///   address for alternative chains or local test environments where the
    ///   interpreter may not be pre-deployed.
    constructor(address interpreter) {
        INTERPRETER = interpreter == address(0) ? DEFAULT_INTERPRETER : interpreter;
    }

    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically.
     * @param program  Raw EVM bytecode to execute.
     *
     * Delegates to a pre-deployed EVM interpreter via DELEGATECALL so the program
     * runs in DeFiVM's execution context.  Every opcode executes on the native EVM
     * stack — no emulation overhead.  Any revert undoes all side-effects.
     */
    function execute(bytes calldata program) external payable {
        address interpreter = INTERPRETER;
        assembly {
            calldatacopy(0, program.offset, program.length)
            let ok := delegatecall(gas(), interpreter, 0, program.length, 0, 0)
            returndatacopy(0, 0, returndatasize())
            if iszero(ok) {
                revert(0, returndatasize())
            }
            return(0, returndatasize())
        }
    }

    // -------------------------------------------------------------------------
    // Universal DEX swap callback handler
    // -------------------------------------------------------------------------

    /**
     * @notice Universal DEX swap callback handler.
     *
     * Routes incoming callbacks from DEX pools to the appropriate payment
     * handler based on the 4-byte function selector.  The callback ``data``
     * must be encoded as documented in the contract header.
     *
     * Supported protocols
     * -------------------
     *  V3-style (data = abi.encode(address tokenIn)):
     *   • Uniswap V3        — uniswapV3SwapCallback (0xfa461e33)
     *   • QuickSwap/Algebra — algebraSwapCallback   (0x2c8958f6)
     *   • PancakeSwap V3    — pancakeV3SwapCallback  (0x23a69e75)
     *   • Solidly V3        — solidlyV3SwapCallback  (0x3a1c453c)
     *
     *  V2-style (data = abi.encode(address tokenIn, uint256 amountOwed)):
     *   • Uniswap V2 forks  — uniswapV2Call          (0x10d1e85c)
     *   • Aerodrome/Velodrome — hook                 (0x9a7bff79)
     *   • Ramses V2         — ramsesV2FlashCallback  (0xde5f4ecc)
     *
     * Unknown selectors revert to avoid silently accepting unexpected calls.
     */
    fallback() external {
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
                // data_offset is relative to the first parameter (byte 4)
                let dataRelOff := calldataload(68)
                // content starts 32 bytes (length word) after the offset anchor
                tokenIn := calldataload(add(add(4, dataRelOff), 32))
            }
            int256 amount = amount0Delta > 0 ? amount0Delta : amount1Delta;
            if (amount > 0) {
                _callTransfer(tokenIn, msg.sender, uint256(amount));
            }
        } else if (sel == SEL_V2_CALLBACK || sel == SEL_AERODROME_HOOK) {
            // V2-style: uniswapV2Call(address sender, uint256 amount0, uint256 amount1, bytes data)
            //           hook(address sender, uint256 amount0, uint256 amount1, bytes data)
            // Calldata layout (bytes, 0-indexed):
            //   [0..4)    selector
            //   [4..36)   address sender (padded)
            //   [36..68)  uint256 amount0
            //   [68..100) uint256 amount1
            //   [100..132) uint256 data_offset (relative to byte 4, typically 0x80)
            //   [4+data_offset..+32) uint256 data_length
            //   [4+data_offset+32..) bytes  data_content = abi.encode(tokenIn, amountOwed)
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
            // Calldata layout:
            //   [0..4)    selector
            //   [4..36)   uint256 amount0
            //   [36..68)  uint256 amount1
            //   [68..100) uint256 data_offset (relative to byte 4, typically 0x60)
            //   [4+data_offset..+32) uint256 data_length
            //   [4+data_offset+32..) bytes  data_content = abi.encode(tokenIn, amountOwed)
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
            revert("DeFiVM: unknown callback selector");
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * @dev Perform ``token.transfer(to, amount)`` and revert on failure.
     *
     * Supports both standard ERC-20s (that return ``bool``) and non-standard
     * tokens that return no value (e.g. USDT on mainnet).
     */
    function _callTransfer(address token, address to, uint256 amount) internal {
        (bool ok, bytes memory ret) = token.call(abi.encodeWithSelector(TRANSFER_SEL, to, amount));
        require(ok && (ret.length == 0 || abi.decode(ret, (bool))), "DeFiVM: cb transfer failed");
    }
}

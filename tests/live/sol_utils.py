"""Shared Solidity compile and deploy helpers for DeFiVM fork tests.

These utilities are used by multiple fork test files to avoid duplication.
"""

from __future__ import annotations

from pathlib import Path

import solcx
from web3 import AsyncWeb3

from pydefi.types import Address


def ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc once (no-op if already installed)."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def compile_sol_file(path: Path, contract_name: str, *, evm_version: str = "cancun") -> dict:
    """Compile a Solidity file and return ABI + bytecode for *contract_name*.

    Defaults to ``cancun`` so DeFiVM's TSTORE/TLOAD parameter channel compiles.
    """
    ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(path)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
        evm_version=evm_version,
    )
    key = next(k for k in result if k.endswith(f":{contract_name}"))
    return result[key]


def compile_sol_source(source: str, contract_name: str, *, evm_version: str = "cancun") -> dict:
    """Compile an inline Solidity source string and return ABI + bytecode.

    Defaults to ``cancun`` so DeFiVM's TSTORE/TLOAD parameter channel compiles.
    """
    ensure_solc("0.8.24")
    result = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        evm_version=evm_version,
    )
    return result[f"<stdin>:{contract_name}"]


async def deploy(w3: AsyncWeb3, compiled: dict, deployer: Address, *args) -> Address:
    """Deploy a contract and return its address."""
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=0.1)
    return Address(receipt["contractAddress"])


# ---------------------------------------------------------------------------
# MockV3Pool helpers — used by V3 callback regression tests across files
# ---------------------------------------------------------------------------

_compiled_mock_v3_pool: dict | None = None


async def deploy_mock_v3_pool(
    w3: AsyncWeb3,
    deployer: Address,
    token0: Address,
    token1: Address,
) -> tuple[Address, list]:
    """Deploy a 1:1 :class:`MockV3Pool`; return ``(address, abi)``.  The
    pool source is compiled once and cached for the process lifetime."""
    global _compiled_mock_v3_pool
    compiled = _compiled_mock_v3_pool
    if compiled is None:
        ensure_solc("0.8.24")
        result = solcx.compile_source(
            MOCK_POOLS_SOL,
            output_values=["abi", "bin"],
            solc_version="0.8.24",
            evm_version="cancun",
        )
        compiled = result["<stdin>:MockV3Pool"]
        _compiled_mock_v3_pool = compiled
    addr = await deploy(w3, compiled, deployer, token0, token1, 1, 1)
    return addr, compiled["abi"]


# ---------------------------------------------------------------------------
# Shared MockToken Solidity source
# ---------------------------------------------------------------------------

MOCK_TOKEN_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mintable ERC-20 token used in tests.
contract MockToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "MockToken: insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "MockToken: insufficient balance");
        require(allowance[from][msg.sender] >= amount, "MockToken: insufficient allowance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }
}
"""

# ---------------------------------------------------------------------------
# Shared mock DEX pools (V3 + V2) — used by both swap-router fork tests and
# composer/proxy callback regression tests.  The V3 pool's ``swap`` triggers
# a real ``uniswapV3SwapCallback`` on ``msg.sender`` so each entrypoint
# inheriting ``DEXCallbackRouter`` can be exercised end-to-end.
# ---------------------------------------------------------------------------

MOCK_POOLS_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IMockToken {
    function balanceOf(address) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function mint(address to, uint256 amount) external;
}

/// @notice Simulates a Uniswap V3 pool that fires uniswapV3SwapCallback.
///   Used both for DEX callback handler tests and multi-hop swap tests.
contract MockV3Pool {
    address public immutable token0;
    address public immutable token1;
    uint256 public rateNumerator;
    uint256 public rateDenominator;

    constructor(address _token0, address _token1, uint256 _rateNum, uint256 _rateDen) {
        token0 = _token0;
        token1 = _token1;
        rateNumerator = _rateNum;
        rateDenominator = _rateDen;
    }

    /// @notice Simulates pool.swap() — the V3 pool interface called by DeFiVM programs.
    ///   amountSpecified > 0 = exact input.
    ///   Returns (amount0, amount1): positive = owed to pool, negative = sent to recipient.
    function swap(
        address recipient,
        bool zeroForOne,
        int256 amountSpecified,
        uint160 /*sqrtPriceLimitX96*/,
        bytes calldata data
    ) external returns (int256 amount0, int256 amount1) {
        require(amountSpecified > 0, "MockV3Pool: exact input only");
        uint256 amountIn = uint256(amountSpecified);
        uint256 amountOut = amountIn * rateNumerator / rateDenominator;

        address tokenOut = zeroForOne ? token1 : token0;

        // Send output to recipient
        IMockToken(tokenOut).mint(recipient, amountOut);

        // Call back into msg.sender (= caller) with uniswapV3SwapCallback selector
        (bool ok,) = msg.sender.call(
            abi.encodeWithSelector(
                bytes4(0xfa461e33),
                zeroForOne ? int256(amountIn) : -int256(amountOut),
                zeroForOne ? -int256(amountOut) : int256(amountIn),
                data
            )
        );
        require(ok, "MockV3Pool: callback failed");

        amount0 = zeroForOne ? int256(amountIn) : -int256(amountOut);
        amount1 = zeroForOne ? -int256(amountOut) : int256(amountIn);
    }

    /// @notice Used by DeFiVM callback tests: give tokens to callee then fire callback.
    function simulateFlashSwap(
        address callee,
        uint256 amountOut,
        int256  amount0Delta,
        int256  amount1Delta,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        address tokenToGive = token1;  // simplified: always give token1
        IMockToken(tokenToGive).mint(callee, amountOut);

        uint256 balBefore = IMockToken(token0).balanceOf(address(this));

        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0xfa461e33),
                amount0Delta,
                amount1Delta,
                data
            )
        );
        require(ok, "MockV3Pool: callback reverted");

        uint256 received = IMockToken(token0).balanceOf(address(this)) - balBefore;
        require(received >= repayAmount, "MockV3Pool: insufficient repayment");
    }

    /// @notice Used by DeFiVM callback tests with algebraSwapCallback selector.
    function simulateAlgebraFlashSwap(
        address callee,
        uint256 amountOut,
        int256  amount0Delta,
        int256  amount1Delta,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        IMockToken(token1).mint(callee, amountOut);
        uint256 balBefore = IMockToken(token0).balanceOf(address(this));

        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x2c8958f6),
                amount0Delta,
                amount1Delta,
                data
            )
        );
        require(ok, "MockV3Pool: algebra callback reverted");

        uint256 received = IMockToken(token0).balanceOf(address(this)) - balBefore;
        require(received >= repayAmount, "MockV3Pool: insufficient repayment");
    }
}

/// @notice Simulates a Uniswap V2 pair (pre-transfer + pair.swap model).
///   DeFiVM programs first transfer tokenIn to this contract, then call swap().
contract MockV2Pair {
    address public immutable token0;
    address public immutable token1;
    uint112 public reserve0;
    uint112 public reserve1;

    constructor(address _token0, address _token1, uint112 _reserve0, uint112 _reserve1) {
        token0 = _token0;
        token1 = _token1;
        reserve0 = _reserve0;
        reserve1 = _reserve1;
    }

    function getReserves() external view returns (uint112, uint112, uint32) {
        return (reserve0, reserve1, uint32(block.timestamp));
    }

    function swap(
        uint amount0Out,
        uint amount1Out,
        address to,
        bytes calldata data
    ) external {
        require(amount0Out > 0 || amount1Out > 0, "MockV2Pair: zero output");

        // Transfer output tokens
        if (amount0Out > 0) IMockToken(token0).transfer(to, amount0Out);
        if (amount1Out > 0) IMockToken(token1).transfer(to, amount1Out);

        // V2 flash-swap callback if data is provided
        if (data.length > 0) {
            (bool ok,) = to.call(
                abi.encodeWithSelector(
                    bytes4(0x10d1e85c),
                    msg.sender, amount0Out, amount1Out, data
                )
            );
            require(ok, "MockV2Pair: callback failed");
        }

        // Update reserves after
        reserve0 = uint112(IMockToken(token0).balanceOf(address(this)));
        reserve1 = uint112(IMockToken(token1).balanceOf(address(this)));
    }

    /// @notice Used by DeFiVM callback tests: fire uniswapV2Call directly.
    function simulateFlashSwap(
        address callee,
        uint256 amountOut,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        IMockToken(token0).mint(callee, amountOut);
        uint256 balBefore = IMockToken(token0).balanceOf(address(this));

        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x10d1e85c),
                msg.sender, amountOut, 0, data
            )
        );
        require(ok, "MockV2Pair: callback reverted");

        uint256 received = IMockToken(token0).balanceOf(address(this)) - balBefore;
        require(received >= repayAmount, "MockV2Pair: insufficient repayment");
    }

    /// @notice Aerodrome hook callback test.
    function simulateAerodromeHook(
        address callee,
        uint256 amountOut,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        IMockToken(token0).mint(callee, amountOut);
        uint256 balBefore = IMockToken(token0).balanceOf(address(this));

        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x9a7bff79),
                msg.sender, amountOut, 0, data
            )
        );
        require(ok, "MockV2Pair: aerodrome hook reverted");

        uint256 received = IMockToken(token0).balanceOf(address(this)) - balBefore;
        require(received >= repayAmount, "MockV2Pair: insufficient repayment");
    }
}
"""


# ---------------------------------------------------------------------------
# Patched EVM interpreter
# ---------------------------------------------------------------------------


_PATCHED_INTERPRETER_SOL_PATH = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "PatchedInterpreter.sol"


def compile_interpreter_sync() -> dict:
    """Compile ``PatchedInterpreter.sol`` and return the solcx output dict.

    Key normalized to ``"<stdin>:Interpreter"`` for backwards compatibility
    with existing callers.
    """
    ensure_solc("0.8.24")
    out = solcx.compile_files(
        [str(_PATCHED_INTERPRETER_SOL_PATH)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in out if k.endswith(":PatchedInterpreter"))
    return {"<stdin>:Interpreter": out[key]}

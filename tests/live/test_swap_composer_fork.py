"""Fork tests for DeFiVM multi-hop swap composer and DEX callback handler.

These tests compile DeFiVM.sol with py-solc-x, deploy it on a local Anvil fork
of Ethereum mainnet, and exercise:

 - DeFiVM.fallback() callback routing for V3-style protocols
   (uniswapV3SwapCallback, algebraSwapCallback)
 - DeFiVM.fallback() callback routing for V2-style protocols
   (uniswapV2Call, Aerodrome hook)
 - Callback data encoding helpers (encode_v3_callback_data, encode_v2_callback_data)
 - Calldata builder helpers (v3_pool_swap_calldata, encode_v3_path)
 - Multi-hop program composition calling pool contracts directly
 - Two-hop swap on local Anvil fork with mock V3 pool → mock V2 pair

Run with::

    pytest -m fork tests/live/test_swap_composer_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from eth_contract.erc20 import ERC20
from hexbytes import HexBytes
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.abi.codec import codec
from pydefi.pathfinder.graph import PoolEdge, V3PoolEdge
from pydefi.types import Address, ChainId, RouteDAG, Token
from pydefi.vm import build_execution_program_for_dag
from pydefi.vm.swap import (
    encode_v2_callback_data,
    encode_v3_callback_data,
    encode_v3_path,
    v3_pool_swap_calldata,
)
from tests.addrs import (
    DAI,
    USDC,
    WETH,
)
from tests.live.sol_utils import MOCK_TOKEN_SOL, compile_sol_file, compile_sol_source, deploy, ensure_solc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Well-known mainnet addresses (for ABI/selector tests only; no live calls)
# ---------------------------------------------------------------------------

WETH_ADDR: Address = WETH.address
USDC_ADDR: Address = USDC.address
DAI_ADDR: Address = DAI.address

# ---------------------------------------------------------------------------
# Mock pool contracts (inline Solidity)
# ---------------------------------------------------------------------------

_MOCK_POOLS_SOL = """\
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

        // Call back into msg.sender (= DeFiVM) with uniswapV3SwapCallback selector
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

MOCK_V2_PAIR = Contract.from_abi(
    [
        "function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)",
        "function token0() external view returns (address)",
        "function token1() external view returns (address)",
        "function swap(uint amount0Out, uint amount1Out, address to, bytes calldata data) external",
        "function simulateFlashSwap(address callee, uint256 amountOut, bytes calldata data, uint256 repayAmount) external",
        "function simulateAerodromeHook(address callee, uint256 amountOut, bytes calldata data, uint256 repayAmount) external",
    ]
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_vm():
    return compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")


@pytest.fixture(scope="module")
def compiled_pools():
    ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_POOLS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return result


@pytest.fixture(scope="module")
async def ctx(fork_w3_module, compiled_vm, compiled_pools, interpreter_addr):
    """Deploy DeFiVM + mock pool contracts; return shared context dict."""
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy DeFiVM
    vm_address = await deploy(w3, compiled_vm, deployer, interpreter_addr)
    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])

    # Deploy two shared mock tokens (token0 and token1)
    token_compiled = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    token0_address = await deploy(w3, token_compiled, deployer)
    token1_address = await deploy(w3, token_compiled, deployer)
    token0 = w3.eth.contract(address=token0_address, abi=token_compiled["abi"])
    token1 = w3.eth.contract(address=token1_address, abi=token_compiled["abi"])

    # Deploy MockV3Pool (token0→token1 at 2:1 rate, i.e. token1/token0 = 2)
    v3pool_compiled = compiled_pools["<stdin>:MockV3Pool"]
    v3pool_address = await deploy(w3, v3pool_compiled, deployer, token0_address, token1_address, 2, 1)
    v3pool = w3.eth.contract(address=v3pool_address, abi=v3pool_compiled["abi"])

    # Deploy MockV2Pair with initial reserves (token0: 1M, token1: 3M → 3:1 rate token1/token0)
    RESERVE0 = 10**12
    RESERVE1 = 3 * 10**12
    v2pair_compiled = compiled_pools["<stdin>:MockV2Pair"]
    # Deploy pair with token1_address as its token0 and token0_address as its token1
    # (reversed order gives us a hop2 of token1→token0).
    # Actual token balances are minted to the pair after deployment.
    v2pair_address = await deploy(
        w3,
        v2pair_compiled,
        deployer,
        token1_address,
        token0_address,  # pair.token0 = token1_address, pair.token1 = token0_address
        RESERVE0,
        RESERVE1,
    )
    v2pair = w3.eth.contract(address=v2pair_address, abi=v2pair_compiled["abi"])

    # Mint reserves to v2pair so it can give tokens during swaps
    await token0.functions.mint(v3pool_address, 10**24).transact({"from": deployer})
    await token1.functions.mint(v3pool_address, 10**24).transact({"from": deployer})
    await token0.functions.mint(v2pair_address, RESERVE0).transact({"from": deployer})
    await token1.functions.mint(v2pair_address, RESERVE1).transact({"from": deployer})

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "token0": token0,
        "token0_address": token0_address,
        "token1": token1,
        "token1_address": token1_address,
        "v3pool": v3pool,
        "v3pool_address": v3pool_address,
        "v2pair": v2pair,
        "v2pair_address": v2pair_address,
        "deployer": deployer,
        "accounts": accounts,
        "_compiled_pools": compiled_pools,
        "_token_compiled": token_compiled,
    }


# ---------------------------------------------------------------------------
# Unit tests (no fork required)
# ---------------------------------------------------------------------------


class TestCallbackDataEncoding:
    """Unit tests for callback data encoding helpers (no network access needed)."""

    def test_encode_v3_callback_data_length(self):
        """encode_v3_callback_data returns 32 bytes (ABI-encoded address)."""
        data = encode_v3_callback_data(WETH_ADDR)
        assert len(data) == 32

    def test_encode_v3_callback_data_roundtrip(self):
        """The encoded address can be decoded back."""
        data = encode_v3_callback_data(WETH_ADDR)
        (decoded,) = codec.decode(["address"], data)
        assert decoded == WETH_ADDR

    def test_encode_v2_callback_data_length(self):
        """encode_v2_callback_data returns 64 bytes (address + uint256)."""
        data = encode_v2_callback_data(WETH_ADDR, 1_000_000)
        assert len(data) == 64

    def test_encode_v2_callback_data_roundtrip(self):
        """The encoded (tokenIn, amountOwed) can be decoded back."""
        amount_owed = 3_003_000
        data = encode_v2_callback_data(WETH_ADDR, amount_owed)
        decoded_token, decoded_amount = codec.decode(["address", "uint256"], data)
        assert decoded_token == WETH_ADDR
        assert decoded_amount == amount_owed


class TestCalldataBuilders:
    """Unit tests for swap calldata builders (no network access needed)."""

    def test_v3_pool_swap_calldata_selector(self):
        """v3_pool_swap_calldata produces calldata with the correct 4-byte selector."""
        from eth_utils import keccak

        calldata = v3_pool_swap_calldata(
            recipient=Address("0x1234567890123456789012345678901234567890"),
            zero_for_one=True,
            amount_in=10**18,
            sqrt_price_limit_x96=0,
            token_in=WETH_ADDR,
        )
        expected_selector = keccak(b"swap(address,bool,int256,uint160,bytes)")[:4]
        assert calldata[:4] == expected_selector
        assert len(calldata) > 4

    def test_encode_v3_path_two_hops(self):
        """encode_v3_path encodes a two-hop path correctly."""
        path = encode_v3_path([WETH_ADDR, USDC_ADDR, DAI_ADDR], fees=[500, 3000])
        assert len(path) == 20 + 3 + 20 + 3 + 20

    def test_encode_v3_path_wrong_fees_raises(self):
        """encode_v3_path raises ValueError when fees length is wrong."""
        with pytest.raises(ValueError, match="encode_v3_path"):
            encode_v3_path([WETH_ADDR, USDC_ADDR], fees=[500, 3000])


# ---------------------------------------------------------------------------
# Fork tests: DEX callback handler
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestDeFiVMCallbacks:
    """Fork tests for DeFiVM.fallback() DEX callback routing."""

    async def test_v3_callback_repays_pool(self, ctx):
        """uniswapV3SwapCallback: DeFiVM receives tokens and repays the pool.

        Flow:
        1. MockV3Pool mints token and transfers to DeFiVM.
        2. MockV3Pool calls DeFiVM.uniswapV3SwapCallback(amount0Delta, ...).
        3. DeFiVM.fallback() decodes tokenIn from data, transfers tokens to pool.
        4. MockV3Pool verifies it received the expected repayment.
        """
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v3pool = ctx["v3pool"]
        token0_address = ctx["token0_address"]
        token0 = ctx["token0"]

        repay_amount = 1_000_000
        # Mint repay_amount of token0 to vm (simulates vm having token0)
        await token0.functions.mint(vm_address, repay_amount).transact({"from": deployer})

        data = encode_v3_callback_data(token0_address)

        tx = await v3pool.functions.simulateFlashSwap(
            vm_address,
            repay_amount,  # amountOut minted to vm
            repay_amount,  # amount0Delta (positive = owed to pool)
            0,
            data,
            repay_amount,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "V3 callback repayment failed"

    async def test_v3_callback_uses_amount1delta(self, ctx):
        """When amount0Delta <= 0, the callback uses amount1Delta."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v3pool = ctx["v3pool"]
        token0_address = ctx["token0_address"]
        token0 = ctx["token0"]

        repay_amount = 500_000
        await token0.functions.mint(vm_address, repay_amount).transact({"from": deployer})
        data = encode_v3_callback_data(token0_address)

        tx = await v3pool.functions.simulateFlashSwap(
            vm_address,
            repay_amount,
            0,  # amount0Delta = 0 → use amount1Delta
            repay_amount,
            data,
            repay_amount,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_algebra_callback_repays_pool(self, ctx):
        """algebraSwapCallback (QuickSwap V3 style) is handled correctly."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v3pool = ctx["v3pool"]
        token0_address = ctx["token0_address"]
        token0 = ctx["token0"]

        repay_amount = 750_000
        await token0.functions.mint(vm_address, repay_amount).transact({"from": deployer})
        data = encode_v3_callback_data(token0_address)

        tx = await v3pool.functions.simulateAlgebraFlashSwap(
            vm_address,
            repay_amount,
            repay_amount,
            0,
            data,
            repay_amount,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

    async def test_v2_callback_repays_pool(self, ctx):
        """uniswapV2Call: DeFiVM repays the pool using the data-encoded amountOwed."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v2pair_address = ctx["v2pair_address"]
        # v2pair was deployed with pair.token0 = token1_address; simulateFlashSwap
        # mints and checks pair.token0, so we must repay with token1.
        token1_address = ctx["token1_address"]

        flash_amount = 2_000_000
        amount_owed = 2_006_000

        await ERC20.fns.mint(vm_address, amount_owed).transact(w3, deployer, to=token1_address)
        data = encode_v2_callback_data(token1_address, amount_owed)

        receipt = await MOCK_V2_PAIR.fns.simulateFlashSwap(
            vm_address,
            flash_amount,
            data,
            amount_owed,
        ).transact(w3, deployer, to=v2pair_address)
        assert receipt["status"] == 1, "V2 callback repayment failed"

    async def test_aerodrome_hook_repays_pool(self, ctx):
        """Aerodrome hook callback is handled identically to uniswapV2Call."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        # v2pair was deployed with pair.token0 = token1_address; simulateAerodromeHook
        # mints and checks pair.token0, so we must repay with token1.
        token1_address = ctx["token1_address"]

        flash_amount = 1_500_000
        amount_owed = 1_504_500

        await ERC20.fns.mint(vm_address, amount_owed).transact(w3, deployer, to=token1_address)
        data = encode_v2_callback_data(token1_address, amount_owed)

        receipt = await MOCK_V2_PAIR.fns.simulateAerodromeHook(
            vm_address,
            flash_amount,
            data,
            amount_owed,
        ).transact(w3, deployer, to=ctx["v2pair_address"])
        assert receipt["status"] == 1

    async def test_unknown_selector_reverts(self, ctx):
        """An unknown callback selector should revert with 'DeFiVM: unknown callback selector'."""
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]

        calldata = b"\xde\xad\xbe\xef" + b"\x00" * 32
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await ctx["w3"].eth.send_transaction(
                {
                    "from": deployer,
                    "to": vm_address,
                    "data": "0x" + calldata.hex(),
                }
            )


# ---------------------------------------------------------------------------
# Module-scoped fixture for split trading tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def split_ctx(ctx, compiled_pools):
    """Extend *ctx* with extra pools and a third token for split trading tests.

    Additions:
    - ``v3pool_b``, ``v3pool_b_address``: second V3 pool (token0 → token1, 2:1)
    - ``v3pool_c``, ``v3pool_c_address``: third V3 pool  (token0 → token1, 2:1)
    - ``token2``, ``token2_address``: a third ERC-20 token (T3 in the issue)
    - ``v3pool_1to2_a``, ``v3pool_1to2_a_address``: pool (token1 → token2, 2:1)
    - ``v3pool_1to2_b``, ``v3pool_1to2_b_address``: pool (token1 → token2, 2:1)
    """
    w3 = ctx["w3"]
    deployer = ctx["deployer"]
    token0_address = ctx["token0_address"]
    token1_address = ctx["token1_address"]
    token_compiled = ctx["_token_compiled"]
    v3pool_compiled = compiled_pools["<stdin>:MockV3Pool"]

    # Two more V3 pools routing token0 → token1 (same 2:1 rate)
    v3pool_b_address = await deploy(w3, v3pool_compiled, deployer, token0_address, token1_address, 2, 1)
    v3pool_b = w3.eth.contract(address=v3pool_b_address, abi=v3pool_compiled["abi"])
    v3pool_c_address = await deploy(w3, v3pool_compiled, deployer, token0_address, token1_address, 2, 1)
    v3pool_c = w3.eth.contract(address=v3pool_c_address, abi=v3pool_compiled["abi"])

    # Third token (T3 / token2)
    token2_address = await deploy(w3, token_compiled, deployer)
    token2 = w3.eth.contract(address=token2_address, abi=token_compiled["abi"])

    # Two V3 pools routing token1 → token2 (2:1 rate)
    v3pool_1to2_a_address = await deploy(w3, v3pool_compiled, deployer, token1_address, token2_address, 2, 1)
    v3pool_1to2_a = w3.eth.contract(address=v3pool_1to2_a_address, abi=v3pool_compiled["abi"])
    v3pool_1to2_b_address = await deploy(w3, v3pool_compiled, deployer, token1_address, token2_address, 2, 1)
    v3pool_1to2_b = w3.eth.contract(address=v3pool_1to2_b_address, abi=v3pool_compiled["abi"])

    return {
        **ctx,
        "v3pool_b": v3pool_b,
        "v3pool_b_address": v3pool_b_address,
        "v3pool_c": v3pool_c,
        "v3pool_c_address": v3pool_c_address,
        "token2": token2,
        "token2_address": token2_address,
        "v3pool_1to2_a": v3pool_1to2_a,
        "v3pool_1to2_a_address": v3pool_1to2_a_address,
        "v3pool_1to2_b": v3pool_1to2_b,
        "v3pool_1to2_b_address": v3pool_1to2_b_address,
    }


@pytest.mark.fork
class TestDAGProgramFork:
    async def test_execution_program_for_dag_runs_on_vm(self, ctx):
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        vm = ctx["vm"]
        token0 = ctx["token0"]
        token0_address = ctx["token0_address"]
        token1_address = ctx["token1_address"]
        v3pool_address = ctx["v3pool_address"]
        v2pair = ctx["v2pair"]
        v2pair_address = ctx["v2pair_address"]

        amount_in = 1000
        await token0.functions.mint(vm_address, amount_in).transact({"from": deployer})
        bal_before = await token0.functions.balanceOf(deployer).call()

        token0_meta = Token(chain_id=ChainId.ETHEREUM, address=token0_address, symbol="T0")
        token1_meta = Token(chain_id=ChainId.ETHEREUM, address=token1_address, symbol="T1")

        v3_edge = V3PoolEdge(
            token_in=token0_meta,
            token_out=token1_meta,
            pool_address=v3pool_address,
            protocol="UniswapV3",
            fee_bps=30,
            sqrt_price_x96=2**96,
            liquidity=10**12,
            is_token0_in=True,
        )

        pair_token0 = await v2pair.functions.token0().call()
        v2_edge = PoolEdge(
            token_in=token1_meta,
            token_out=token0_meta,
            pool_address=v2pair_address,
            protocol="UniswapV2",
            reserve_in=10**12,
            reserve_out=10**12,
            fee_bps=30,
            extra={"is_token0_in": HexBytes(pair_token0) == token1_address},
        )

        dag = (
            RouteDAG()
            .from_token(token0_meta)
            .split()
            .leg(5000)
            .swap(token1_meta, v3_edge)
            .leg(5000)
            .swap(token1_meta, v3_edge)
            .merge()
            .swap(token0_meta, v2_edge)
        )
        bytecode = build_execution_program_for_dag(
            dag,
            amount_in=amount_in,
            vm_address=vm_address,
            recipient=deployer,
            min_final_out=1,
        ).build()

        tx = await vm.functions.execute(bytecode).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        bal_after = await token0.functions.balanceOf(deployer).call()
        assert bal_after > bal_before

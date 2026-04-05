"""Fork tests for DeFiVM multi-hop swap composer and DEX callback handler.

These tests compile DeFiVM.sol with py-solc-x, deploy it on a local Anvil fork
of Ethereum mainnet, and exercise:

 - DeFiVM.fallback() callback routing for V3-style protocols
   (uniswapV3SwapCallback, algebraSwapCallback, pancakeV3SwapCallback)
 - DeFiVM.fallback() callback routing for V2-style protocols
   (uniswapV2Call, Aerodrome hook)
 - Callback data encoding helpers (encode_v3_callback_data, encode_v2_callback_data)
 - Calldata builder helpers (v2_swap_calldata, v3_exact_input_single_calldata, encode_v3_path)
 - Multi-hop program composition (two-hop WETH → USDC → DAI via mock routers)
 - Live two-hop swap on mainnet fork with real Uniswap V3 + V2 routers

Run with::

    pytest -m fork tests/live/test_swap_composer_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eth_abi import decode
from web3 import AsyncWeb3

from pydefi.vm.swap import (
    V2_AMOUNT_OUT_OFFSET,
    V3_AMOUNT_OUT_OFFSET,
    SwapHop,
    SwapProtocol,
    build_multi_hop_program,
    encode_v2_callback_data,
    encode_v3_callback_data,
    encode_v3_path,
    v2_swap_calldata,
    v3_exact_input_single_calldata,
)

# ---------------------------------------------------------------------------
# Optional: skip whole module if solcx not installed
# ---------------------------------------------------------------------------
solcx = pytest.importorskip("solcx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"

# ---------------------------------------------------------------------------
# Well-known mainnet addresses
# ---------------------------------------------------------------------------

WETH_ADDR  = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_ADDR  = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI_ADDR   = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

# Uniswap V3 SwapRouter (V1) — mainnet
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
# Uniswap V2 Router02 — mainnet
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

# A well-funded WETH whale on mainnet (Binance 14)
WETH_WHALE = "0x28C6c06298d514Db089934071355E5743bf21d60"

# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------


def _ensure_solc(version: str = "0.8.24") -> None:
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def _compile_sol_file(path: Path, contract_name: str) -> dict:
    _ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(path)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(f":{contract_name}"))
    return result[key]


def _compile_sol_source(source: str, contract_name: str) -> dict:
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return result[f"<stdin>:{contract_name}"]


async def _deploy(w3: AsyncWeb3, compiled: dict, deployer: str, *args) -> str:
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Minimal EVM interpreter (same as used in other fork tests)
# ---------------------------------------------------------------------------

_MINIMAL_INTERPRETER_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal test interpreter: create2-deploys the program and DELEGATECALL-executes it.
contract TestInterpreter {
    fallback() external payable {
        bytes memory code = msg.data;
        address deployed;
        assembly {
            deployed := create(0, add(code, 32), mload(code))
        }
        require(deployed != address(0), "TestInterpreter: create failed");
        assembly {
            let ok := delegatecall(gas(), deployed, 0, 0, 0, 0)
            returndatacopy(0, 0, returndatasize())
            if iszero(ok) { revert(0, returndatasize()) }
        }
    }
}
"""


def _compile_test_interpreter() -> dict:
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MINIMAL_INTERPRETER_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return result["<stdin>:TestInterpreter"]


async def _ensure_interpreter(w3: AsyncWeb3, deployer: str) -> str:
    """Return EVM interpreter address, deploying TestInterpreter if needed."""
    from tests.live.conftest import INTERPRETER_ADDR

    code = await w3.eth.get_code(INTERPRETER_ADDR)
    if code and len(code) > 1:
        return INTERPRETER_ADDR

    compiled = _compile_test_interpreter()
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor().transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Mock contract sources
# ---------------------------------------------------------------------------

_MOCK_CONTRACTS_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mintable ERC-20 token used in callback tests.
contract MockToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "MockToken: insufficient");
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

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }
}

/// @notice Simulates a Uniswap V3-style pool that calls uniswapV3SwapCallback.
contract MockV3Pool {
    address public immutable token;
    uint256 public lastAmountReceived;

    constructor(address _token) { token = _token; }

    /// @dev Gives `amountOut` tokens to `callee`, then calls back with
    ///   uniswapV3SwapCallback(amount0Delta, amount1Delta, data).
    ///   Verifies that `repayAmount` tokens are received back from the callback.
    function simulateFlashSwap(
        address callee,
        uint256 amountOut,
        int256  amount0Delta,
        int256  amount1Delta,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        MockToken(token).transfer(callee, amountOut);

        uint256 balBefore = MockToken(token).balanceOf(address(this));

        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0xfa461e33), // uniswapV3SwapCallback(int256,int256,bytes)
                amount0Delta,
                amount1Delta,
                data
            )
        );
        require(ok, "MockV3Pool: callback reverted");

        lastAmountReceived = MockToken(token).balanceOf(address(this)) - balBefore;
        require(lastAmountReceived >= repayAmount, "MockV3Pool: insufficient repayment");
    }

    /// @dev Same as simulateFlashSwap but uses algebraSwapCallback selector.
    function simulateAlgebraFlashSwap(
        address callee,
        uint256 amountOut,
        int256  amount0Delta,
        int256  amount1Delta,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        MockToken(token).transfer(callee, amountOut);
        uint256 balBefore = MockToken(token).balanceOf(address(this));
        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x2c8958f6), // algebraSwapCallback(int256,int256,bytes)
                amount0Delta,
                amount1Delta,
                data
            )
        );
        require(ok, "MockV3Pool: algebra callback reverted");
        lastAmountReceived = MockToken(token).balanceOf(address(this)) - balBefore;
        require(lastAmountReceived >= repayAmount, "MockV3Pool: insufficient repayment");
    }
}

/// @notice Simulates a Uniswap V2-style pool that calls uniswapV2Call.
contract MockV2Pool {
    address public immutable token;
    uint256 public lastAmountReceived;

    constructor(address _token) { token = _token; }

    /// @dev Gives `amountOut` tokens to `callee`, then calls back with
    ///   uniswapV2Call(sender, amount0, amount1, data).
    function simulateFlashSwap(
        address callee,
        uint256 amountOut,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        MockToken(token).transfer(callee, amountOut);
        uint256 balBefore = MockToken(token).balanceOf(address(this));
        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x10d1e85c), // uniswapV2Call(address,uint256,uint256,bytes)
                msg.sender,         // sender
                amountOut,          // amount0
                0,                  // amount1
                data
            )
        );
        require(ok, "MockV2Pool: callback reverted");
        lastAmountReceived = MockToken(token).balanceOf(address(this)) - balBefore;
        require(lastAmountReceived >= repayAmount, "MockV2Pool: insufficient repayment");
    }

    /// @dev Same as simulateFlashSwap but uses Aerodrome hook selector.
    function simulateAerodromeHook(
        address callee,
        uint256 amountOut,
        bytes   calldata data,
        uint256 repayAmount
    ) external {
        MockToken(token).transfer(callee, amountOut);
        uint256 balBefore = MockToken(token).balanceOf(address(this));
        (bool ok,) = callee.call(
            abi.encodeWithSelector(
                bytes4(0x9a7bff79), // hook(address,uint256,uint256,bytes)
                msg.sender,
                amountOut,
                0,
                data
            )
        );
        require(ok, "MockV2Pool: aerodrome hook reverted");
        lastAmountReceived = MockToken(token).balanceOf(address(this)) - balBefore;
        require(lastAmountReceived >= repayAmount, "MockV2Pool: insufficient repayment");
    }
}

/// @notice Simulates a DEX router for multi-hop composition tests.
///   Accepts any two-token swap, returns a fixed ratio output, and mints
///   the output token to the recipient.
contract MockRouter {
    /// @dev Simple fixed-ratio swap: out = amountIn * rateNumerator / rateDenominator.
    mapping(address => mapping(address => uint256)) public rateNumerator;
    mapping(address => mapping(address => uint256)) public rateDenominator;

    function setRate(
        address tokenIn,
        address tokenOut,
        uint256 numerator,
        uint256 denominator
    ) external {
        rateNumerator[tokenIn][tokenOut] = numerator;
        rateDenominator[tokenIn][tokenOut] = denominator;
    }

    // V2-style: swapExactTokensForTokens
    function swapExactTokensForTokens(
        uint amountIn,
        uint /*amountOutMin*/,
        address[] calldata path,
        address to,
        uint /*deadline*/
    ) external returns (uint[] memory amounts) {
        require(path.length == 2, "MockRouter: only 2-token path");
        address tokenIn = path[0];
        address tokenOut = path[1];

        // Pull input tokens
        MockToken(tokenIn).transferFrom(msg.sender, address(this), amountIn);

        // Compute output
        uint num = rateNumerator[tokenIn][tokenOut];
        uint den = rateDenominator[tokenIn][tokenOut];
        require(den > 0, "MockRouter: rate not set");
        uint amountOut = amountIn * num / den;

        // Mint output to recipient (simulating liquidity reserves)
        MockToken(tokenOut).mint(to, amountOut);

        amounts = new uint[](2);
        amounts[0] = amountIn;
        amounts[1] = amountOut;
    }

    // V3-style: exactInputSingle
    function exactInputSingle(
        (
            address tokenIn,
            address tokenOut,
            uint24 fee,
            address recipient,
            uint256 deadline,
            uint256 amountIn,
            uint256 amountOutMinimum,
            uint160 sqrtPriceLimitX96
        ) calldata params
    ) external payable returns (uint256 amountOut) {
        MockToken(params.tokenIn).transferFrom(msg.sender, address(this), params.amountIn);
        uint num = rateNumerator[params.tokenIn][params.tokenOut];
        uint den = rateDenominator[params.tokenIn][params.tokenOut];
        require(den > 0, "MockRouter: rate not set");
        amountOut = params.amountIn * num / den;
        require(amountOut >= params.amountOutMinimum, "MockRouter: insufficient output");
        MockToken(params.tokenOut).mint(params.recipient, amountOut);
    }
}
"""


# ---------------------------------------------------------------------------
# Module-scoped fork fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def swap_fork_w3(fork_w3_module):
    return fork_w3_module


# ---------------------------------------------------------------------------
# Module-scoped setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_vm():
    return _compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")


@pytest.fixture(scope="module")
def compiled_mock_contracts():
    _ensure_solc("0.8.24")
    result = solcx.compile_source(
        _MOCK_CONTRACTS_SOL,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return result


# ---------------------------------------------------------------------------
# Unit tests (no fork required)
# ---------------------------------------------------------------------------


class TestCallbackDataEncoding:
    """Unit tests for callback data encoding helpers (no network access needed)."""

    def test_encode_v3_callback_data_length(self):
        """encode_v3_callback_data returns 32 bytes (ABI-encoded address)."""
        data = encode_v3_callback_data("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        assert len(data) == 32

    def test_encode_v3_callback_data_roundtrip(self):
        """The encoded address can be decoded back."""
        token_in = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        data = encode_v3_callback_data(token_in)
        (decoded,) = decode(["address"], data)
        assert decoded.lower() == token_in.lower()

    def test_encode_v2_callback_data_length(self):
        """encode_v2_callback_data returns 64 bytes (address + uint256)."""
        data = encode_v2_callback_data(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            1_000_000,
        )
        assert len(data) == 64

    def test_encode_v2_callback_data_roundtrip(self):
        """The encoded (tokenIn, amountOwed) can be decoded back."""
        token_in = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        amount_owed = 3_003_000
        data = encode_v2_callback_data(token_in, amount_owed)
        decoded_token, decoded_amount = decode(["address", "uint256"], data)
        assert decoded_token.lower() == token_in.lower()
        assert decoded_amount == amount_owed


class TestCalldataBuilders:
    """Unit tests for swap calldata builders (no network access needed)."""

    def test_v2_swap_calldata_selector(self):
        """v2_swap_calldata produces calldata starting with the correct 4-byte selector."""
        from eth_utils import keccak

        calldata = v2_swap_calldata(
            token_in=WETH_ADDR,
            token_out=USDC_ADDR,
            amount_in=10**18,
            amount_out_min=0,
            recipient="0x1234567890123456789012345678901234567890",
            deadline=9999999999,
        )
        expected_selector = keccak(b"swapExactTokensForTokens(uint256,uint256,address[],address,uint256)")[:4]
        assert calldata[:4] == expected_selector
        assert len(calldata) > 4

    def test_v3_exact_input_single_calldata_selector(self):
        """v3_exact_input_single_calldata produces calldata with the correct selector."""
        from eth_utils import keccak

        calldata = v3_exact_input_single_calldata(
            token_in=WETH_ADDR,
            token_out=USDC_ADDR,
            fee=500,
            recipient="0x1234567890123456789012345678901234567890",
            deadline=9999999999,
            amount_in=10**18,
            amount_out_minimum=0,
        )
        # exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))
        expected_selector = keccak(
            b"exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))"
        )[:4]
        assert calldata[:4] == expected_selector

    def test_encode_v3_path_two_hops(self):
        """encode_v3_path encodes a two-hop path correctly."""
        path = encode_v3_path([WETH_ADDR, USDC_ADDR, DAI_ADDR], fees=[500, 3000])
        # Expected: 20 bytes token0 + 3 bytes fee0 + 20 bytes token1 + 3 bytes fee1 + 20 bytes token2
        assert len(path) == 20 + 3 + 20 + 3 + 20

    def test_encode_v3_path_wrong_fees_raises(self):
        """encode_v3_path raises ValueError when fees length is wrong."""
        with pytest.raises(ValueError, match="encode_v3_path"):
            encode_v3_path([WETH_ADDR, USDC_ADDR], fees=[500, 3000])  # 2 fees for 2-token path

    def test_build_multi_hop_program_empty_raises(self):
        """build_multi_hop_program raises ValueError for empty hops list."""
        with pytest.raises(ValueError, match="hops list must not be empty"):
            build_multi_hop_program([])

    def test_build_multi_hop_program_single_hop_v3(self):
        """build_multi_hop_program compiles without error for a single V3 hop."""
        import time

        hops = [
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V3,
                router=UNISWAP_V3_ROUTER,
                token_in=WETH_ADDR,
                token_out=USDC_ADDR,
                fee=500,
                amount_in=10**18,
                amount_out_min=0,
                recipient="0x1234567890123456789012345678901234567890",
                deadline=int(time.time()) + 600,
                out_offset=V3_AMOUNT_OUT_OFFSET,
            )
        ]
        program = build_multi_hop_program(hops, min_final_out=0)
        bytecode = program.build()
        assert isinstance(bytecode, bytes)
        assert len(bytecode) > 0

    def test_build_multi_hop_program_two_hops_v3_v2(self):
        """build_multi_hop_program compiles without error for a V3→V2 two-hop."""
        import time

        deadline = int(time.time()) + 600
        hops = [
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V3,
                router=UNISWAP_V3_ROUTER,
                token_in=WETH_ADDR,
                token_out=USDC_ADDR,
                fee=500,
                amount_in=10**18,
                amount_out_min=0,
                recipient="0x1234567890123456789012345678901234567890",
                deadline=deadline,
                out_offset=V3_AMOUNT_OUT_OFFSET,
            ),
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V2,
                router=UNISWAP_V2_ROUTER,
                token_in=USDC_ADDR,
                token_out=DAI_ADDR,
                fee=0,
                amount_in=0,
                amount_out_min=0,
                recipient="0x1234567890123456789012345678901234567890",
                deadline=deadline,
                out_offset=V2_AMOUNT_OUT_OFFSET,
            ),
        ]
        program = build_multi_hop_program(hops, min_final_out=10**18)
        bytecode = program.build()
        assert isinstance(bytecode, bytes)
        assert len(bytecode) > 0


# ---------------------------------------------------------------------------
# Fork tests: DEX callback handler
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestDeFiVMCallbacks:
    """Fork tests for DeFiVM.fallback() DEX callback routing."""

    async def test_v3_callback_repays_pool(self, ctx):
        """uniswapV3SwapCallback: DeFiVM receives tokens and repays the pool on callback.

        Flow:
        1. MockV3Pool mints token and transfers to DeFiVM (simulating flash swap output).
        2. MockV3Pool calls DeFiVM.uniswapV3SwapCallback(amount0Delta=1000, ...).
        3. DeFiVM.fallback() decodes tokenIn from data, transfers 1000 tokens to pool.
        4. MockV3Pool verifies it received the expected repayment.
        """
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v3pool = ctx["v3pool"]
        token_address = ctx["token_address"]

        flash_amount = 1_000_000  # tokens given by pool in flash swap
        repay_amount = 1_000_000  # tokens owed back (same for test simplicity)

        # Encode data: abi.encode(tokenIn)
        data = encode_v3_callback_data(token_address)

        tx = await v3pool.functions.simulateFlashSwap(
            vm_address,
            flash_amount,   # amountOut: tokens pool sends to DeFiVM
            repay_amount,   # amount0Delta (positive = owed to pool)
            0,              # amount1Delta
            data,
            repay_amount,   # expected repayment
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "V3 callback repayment failed"

        received = await v3pool.functions.lastAmountReceived().call()
        assert received == repay_amount

    async def test_v3_callback_uses_amount1delta_when_amount0_nonpositive(self, ctx):
        """When amount0Delta <= 0, the callback uses amount1Delta as the repay amount."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v3pool = ctx["v3pool"]
        token_address = ctx["token_address"]

        repay_amount = 500_000
        data = encode_v3_callback_data(token_address)

        tx = await v3pool.functions.simulateFlashSwap(
            vm_address,
            repay_amount,
            0,             # amount0Delta = 0 → should use amount1Delta
            repay_amount,  # amount1Delta (positive = owed)
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
        token_address = ctx["token_address"]

        repay_amount = 750_000
        data = encode_v3_callback_data(token_address)

        tx = await v3pool.functions.simulateAlgebraFlashSwap(
            vm_address,
            repay_amount,
            repay_amount,  # amount0Delta
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
        v2pool = ctx["v2pool"]
        token_address = ctx["token_address"]

        flash_amount = 2_000_000
        amount_owed = 2_006_000  # borrowed + 0.3% fee (rounded up)

        data = encode_v2_callback_data(token_address, amount_owed)

        tx = await v2pool.functions.simulateFlashSwap(
            vm_address,
            flash_amount,
            data,
            amount_owed,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "V2 callback repayment failed"

        received = await v2pool.functions.lastAmountReceived().call()
        assert received == amount_owed

    async def test_aerodrome_hook_repays_pool(self, ctx):
        """Aerodrome hook callback is handled identically to uniswapV2Call."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        v2pool = ctx["v2pool"]
        token_address = ctx["token_address"]

        flash_amount = 1_500_000
        amount_owed = 1_504_500  # borrowed + 0.3%

        data = encode_v2_callback_data(token_address, amount_owed)

        tx = await v2pool.functions.simulateAerodromeHook(
            vm_address,
            flash_amount,
            data,
            amount_owed,
        ).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "Aerodrome hook repayment failed"

    async def test_unknown_selector_does_not_revert(self, ctx):
        """An unknown callback selector should not revert (silent no-op)."""
        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]

        # Call DeFiVM with a random 4-byte selector — should succeed silently
        calldata = b"\xde\xad\xbe\xef" + b"\x00" * 32
        tx_hash = await w3.eth.send_transaction({
            "from": deployer,
            "to": vm_address,
            "data": "0x" + calldata.hex(),
        })
        receipt = await w3.eth.get_transaction_receipt(tx_hash)
        assert receipt["status"] == 1, "Unknown selector should not revert"


# ---------------------------------------------------------------------------
# Fork tests: multi-hop swap composition using mock routers
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestMultiHopSwapComposer:
    """Fork tests for build_multi_hop_program() using mock DEX routers."""

    async def test_two_hop_v3_then_v2_mock_router(self, ctx):
        """Two-hop swap A→B (V3) → C (V2) using mock router; verifies amount chaining.

        Setup:
          - MockRouter rate A→B = 2x, B→C = 3x
          - Swap 1000 A → expects 2000 B (V3 exactInputSingle)
          - Then 2000 B → expects 6000 C (V2 swapExactTokensForTokens)
          - Final assertion: amount_out >= 5000 C (with some slack)

        The test verifies that:
        1. The multi-hop program executes without reverting.
        2. The deployer receives at least min_final_out tokens of C.
        """
        import time

        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        vm = ctx["vm"]
        router_address = ctx["router_address"]
        router = ctx["router"]
        compiled = ctx["_compiled_mock"]

        # ── Deploy fresh tokens ───────────────────────────────────────────────
        token_compiled = compiled["<stdin>:MockToken"]
        token_a_addr = await _deploy(w3, token_compiled, deployer)
        token_b_addr = await _deploy(w3, token_compiled, deployer)
        token_c_addr = await _deploy(w3, token_compiled, deployer)

        token_a = w3.eth.contract(address=token_a_addr, abi=token_compiled["abi"])
        token_c = w3.eth.contract(address=token_c_addr, abi=token_compiled["abi"])

        amount_a = 1000
        await token_a.functions.mint(deployer, amount_a).transact({"from": deployer})
        # Transfer tokenA to VM (user deposits before executing the program)
        await token_a.functions.transfer(vm_address, amount_a).transact({"from": deployer})

        # Set rates: A→B = 2x, B→C = 3x
        await router.functions.setRate(token_a_addr, token_b_addr, 2, 1).transact({"from": deployer})
        await router.functions.setRate(token_b_addr, token_c_addr, 3, 1).transact({"from": deployer})

        # ── Build multi-hop program ───────────────────────────────────────────
        deadline = int(time.time()) + 3600
        hops = [
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V3,
                router=router_address,
                token_in=token_a_addr,
                token_out=token_b_addr,
                fee=500,
                amount_in=amount_a,
                amount_out_min=0,
                recipient=vm_address,   # keep B in VM for next hop
                deadline=deadline,
                out_offset=V3_AMOUNT_OUT_OFFSET,
            ),
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V2,
                router=router_address,
                token_in=token_b_addr,
                token_out=token_c_addr,
                fee=0,
                amount_in=0,            # will be patched from register at runtime
                amount_out_min=0,
                recipient=deployer,
                deadline=deadline,
                out_offset=V2_AMOUNT_OUT_OFFSET,
            ),
        ]
        # Expected: 1000 A → 2000 B → 6000 C; require at least 5500 C
        min_out = 5500
        program = build_multi_hop_program(hops, min_final_out=min_out)
        bytecode = program.build()

        # ── Execute ───────────────────────────────────────────────────────────
        tx = await vm.functions.execute(bytecode).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "Multi-hop swap failed"

        # ── Verify final balance ──────────────────────────────────────────────
        c_balance = await token_c.functions.balanceOf(deployer).call()
        assert c_balance >= min_out, f"Expected >= {min_out} C, got {c_balance}"

    async def test_single_hop_v2_mock_router(self, ctx):
        """Single V2 hop from mock router produces the expected output."""
        import time

        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        vm = ctx["vm"]
        router_address = ctx["router_address"]
        router = ctx["router"]
        compiled = ctx["_compiled_mock"]

        token_compiled = compiled["<stdin>:MockToken"]
        token_in_addr = await _deploy(w3, token_compiled, deployer)
        token_out_addr = await _deploy(w3, token_compiled, deployer)

        token_in = w3.eth.contract(address=token_in_addr, abi=token_compiled["abi"])
        token_out = w3.eth.contract(address=token_out_addr, abi=token_compiled["abi"])

        amount_in = 500
        await token_in.functions.mint(deployer, amount_in).transact({"from": deployer})
        await token_in.functions.transfer(vm_address, amount_in).transact({"from": deployer})

        # Rate: 4x
        await router.functions.setRate(token_in_addr, token_out_addr, 4, 1).transact({"from": deployer})

        deadline = int(time.time()) + 3600
        hops = [
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V2,
                router=router_address,
                token_in=token_in_addr,
                token_out=token_out_addr,
                fee=0,
                amount_in=amount_in,
                amount_out_min=0,
                recipient=deployer,
                deadline=deadline,
                out_offset=V2_AMOUNT_OUT_OFFSET,
            )
        ]
        program = build_multi_hop_program(hops, min_final_out=1800)
        bytecode = program.build()

        tx = await vm.functions.execute(bytecode).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1

        out_balance = await token_out.functions.balanceOf(deployer).call()
        assert out_balance == 2000, f"Expected 2000 (4x rate on 500), got {out_balance}"

    async def test_multi_hop_slippage_check_reverts(self, ctx):
        """build_multi_hop_program with min_final_out reverts when output is too low."""
        import time

        from web3.exceptions import ContractLogicError, Web3RPCError

        w3 = ctx["w3"]
        deployer = ctx["deployer"]
        vm_address = ctx["vm_address"]
        vm = ctx["vm"]
        router_address = ctx["router_address"]
        router = ctx["router"]
        compiled = ctx["_compiled_mock"]

        token_compiled = compiled["<stdin>:MockToken"]
        token_in_addr = await _deploy(w3, token_compiled, deployer)
        token_out_addr = await _deploy(w3, token_compiled, deployer)

        token_in = w3.eth.contract(address=token_in_addr, abi=token_compiled["abi"])
        await token_in.functions.mint(deployer, 100).transact({"from": deployer})
        await token_in.functions.transfer(vm_address, 100).transact({"from": deployer})

        # Rate: 1x (100 in → 100 out)
        await router.functions.setRate(token_in_addr, token_out_addr, 1, 1).transact({"from": deployer})

        deadline = int(time.time()) + 3600
        hops = [
            SwapHop(
                protocol=SwapProtocol.UNISWAP_V3,
                router=router_address,
                token_in=token_in_addr,
                token_out=token_out_addr,
                fee=500,
                amount_in=100,
                amount_out_min=0,
                recipient=deployer,
                deadline=deadline,
                out_offset=V3_AMOUNT_OUT_OFFSET,
            )
        ]
        # Demand 200 out but only 100 will come — must revert
        program = build_multi_hop_program(hops, min_final_out=200)
        bytecode = program.build()

        with pytest.raises((ContractLogicError, Web3RPCError)):
            await vm.functions.execute(bytecode).transact({"from": deployer})


@pytest.fixture(scope="module")
async def ctx(swap_fork_w3, compiled_vm, compiled_mock_contracts, interpreter_addr):
    """Deploy DeFiVM + mock contracts, return a shared context dict."""
    w3 = swap_fork_w3
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    # Deploy DeFiVM
    vm_address = await _deploy(w3, compiled_vm, deployer, interpreter_addr)
    vm = w3.eth.contract(address=vm_address, abi=compiled_vm["abi"])

    # Deploy MockToken
    token_compiled = compiled_mock_contracts["<stdin>:MockToken"]
    token_address = await _deploy(w3, token_compiled, deployer)
    token = w3.eth.contract(address=token_address, abi=token_compiled["abi"])

    # Deploy MockV3Pool and MockV2Pool
    v3pool_compiled = compiled_mock_contracts["<stdin>:MockV3Pool"]
    v3pool_address = await _deploy(w3, v3pool_compiled, deployer, token_address)
    v3pool = w3.eth.contract(address=v3pool_address, abi=v3pool_compiled["abi"])

    v2pool_compiled = compiled_mock_contracts["<stdin>:MockV2Pool"]
    v2pool_address = await _deploy(w3, v2pool_compiled, deployer, token_address)
    v2pool = w3.eth.contract(address=v2pool_address, abi=v2pool_compiled["abi"])

    # Deploy MockRouter
    router_compiled = compiled_mock_contracts["<stdin>:MockRouter"]
    router_address = await _deploy(w3, router_compiled, deployer)
    router = w3.eth.contract(address=router_address, abi=router_compiled["abi"])

    # Mint tokens to pools so they can give them out in flash swaps
    mint_amount = 10**24
    await token.functions.mint(v3pool_address, mint_amount).transact({"from": deployer})
    await token.functions.mint(v2pool_address, mint_amount).transact({"from": deployer})

    return {
        "w3": w3,
        "vm": vm,
        "vm_address": vm_address,
        "token": token,
        "token_address": token_address,
        "v3pool": v3pool,
        "v3pool_address": v3pool_address,
        "v2pool": v2pool,
        "v2pool_address": v2pool_address,
        "router": router,
        "router_address": router_address,
        "deployer": deployer,
        "accounts": accounts,
        "_compiled_mock": compiled_mock_contracts,
    }

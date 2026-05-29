"""Gas benchmark — DEX aggregation encoders compared on a pinned mainnet fork.

Runs each path through SSA composer / OKX router / Uniswap SwapRouter02 and
prints a gas + calldata-size table. Every encoder pulls input via
transferFrom inside the swap tx, so numbers are directly comparable.

Run with ``pytest -m bench tests/live/test_bench_aggregation_gas.py -s``
(``-s`` is required, pytest swallows the table without it).
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
from eth_abi.abi import encode as abi_encode
from eth_contract.erc20 import ERC20
from eth_utils.abi import function_signature_to_4byte_selector
from hexbytes import HexBytes
from web3 import AsyncWeb3

from pydefi.abi.amm import UNISWAP_V2_PAIR, UNISWAP_V3_POOL
from pydefi.pathfinder.graph import PoolEdge, V3PoolEdge
from pydefi.types import Address, RouteDAG, Token
from tests.addrs import (
    DAI,
    ETH_WHALE,
    PAIR_USDC_DAI,
    POOL_DAI_USDC_100,
    POOL_WETH_USDC_500,
    POOL_WETH_USDC_3000,
    UNISWAP_V3_ROUTER,
    UNIVERSAL_ROUTER,
    USDC,
    USDT,
    WETH,
)
from tests.bench.encoders import (
    EncodedTx,
    encode_okx,
    encode_okx_swap_then_aave_supply,
    encode_ssa,
    encode_ssa_swap_then_aave_supply,
    encode_uniswap,
    encode_universal_router,
)
from tests.bench.okx_router_abi import (
    OKX_DEX_ROUTER_ETHEREUM,
    OKX_TOKEN_APPROVE_ETHEREUM,
)
from tests.bench.sol_sources import AAVE_V3_ADAPTER_SOL, UNI_V2_ADAPTER_SOL, UNI_V3_ADAPTER_SOL
from tests.live.anvil_helpers import (
    erc20_approve,
    fund_usdc,
    impersonate,
    send_ok,
    set_balance,
    wrap_eth,
)
from tests.live.conftest import ETH_RPC_URL, _ensure_interpreter
from tests.live.sol_utils import compile_sol_file, compile_sol_source

DEFI_VM_SOL = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "DeFiVM.sol"

_AMOUNT_IN: int = 10**16  # 0.01 WETH — stays inside POOL_WETH_USDC_500's tick range
_DEADLINE: int = 2**63 - 1
_TEST_USER: Address = ETH_WHALE
_MAX_UINT: int = 2**256 - 1

_FORK_BLOCK: int = 25_000_000

POOL_WETH_USDC_10000: Address = Address("0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387")
POOL_WETH_DAI_3000: Address = Address("0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8")
POOL_WETH_USDT_500: Address = Address("0x11b815efB8f581194ae79006d24E0d814B7697F6")
POOL_USDC_USDT_100: Address = Address("0x3416cF6C708Da44DB2624D63ea0AAef7113527C6")
AAVE_V3_POOL_ETHEREUM: Address = Address("0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2")
# Permit2 — same address on every chain.
PERMIT2_ETHEREUM: str = "0x000000000022D473030F116dDEE9F6B43aC78BA3"


@dataclass(frozen=True)
class BenchRow:
    path: str
    encoder: str
    gas_used: int
    calldata_bytes: int
    amount_out: int


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _deploy_adapter(w3: AsyncWeb3, deployer: str, source: str, name: str, *args) -> Address:
    """Compile a Solidity contract and deploy from ``deployer`` with ``args``."""
    compiled = compile_sol_source(source, name)
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
    return Address(receipt["contractAddress"])


@pytest.fixture(scope="module")
async def bench_fork_w3():
    """Module-scoped anvil fork pinned to ``_FORK_BLOCK`` for reproducibility."""
    if shutil.which("anvil") is None:
        pytest.skip("anvil not found on PATH — install Foundry to run fork tests")

    port = _free_port()
    proc = subprocess.Popen(
        [
            "anvil",
            "--fork-url",
            ETH_RPC_URL,
            "--fork-block-number",
            str(_FORK_BLOCK),
            "--port",
            str(port),
            "--silent",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(f"http://127.0.0.1:{port}"))
    # Pinned-block forks fetch archive state first request; allow extra time.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            await w3.eth.chain_id
            break
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.25)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.fail(f"Anvil did not start within 120s (block {_FORK_BLOCK})")

    yield w3

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture(scope="module")
async def bench_ctx(bench_fork_w3: AsyncWeb3) -> dict:
    """Deploy DeFiVM + adapter, fund + approve once per test module."""
    w3 = bench_fork_w3

    accounts = await w3.eth.accounts
    deployer = accounts[0]

    interpreter_addr = await _ensure_interpreter(w3, deployer)

    compiled = compile_sol_file(DEFI_VM_SOL, "DeFiVM")
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx = await contract.constructor(interpreter_addr).transact({"from": deployer})
    receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
    vm_address = Address(receipt["contractAddress"])

    # MinimalUniV3 / V2 / AaveV3 adapters used by OKX smart-swap path.
    uni_v3_adapter_address = await _deploy_adapter(w3, deployer, UNI_V3_ADAPTER_SOL, "MinimalUniV3Adapter")
    uni_v2_adapter_address = await _deploy_adapter(w3, deployer, UNI_V2_ADAPTER_SOL, "MinimalUniV2Adapter")
    aave_v3_adapter_address = await _deploy_adapter(
        w3,
        deployer,
        AAVE_V3_ADAPTER_SOL,
        "MinimalAaveV3Adapter",
        AAVE_V3_POOL_ETHEREUM,
    )

    await impersonate(w3, _TEST_USER)
    await set_balance(w3, _TEST_USER, 100 * 10**21)
    await wrap_eth(w3, _TEST_USER, WETH.address, _AMOUNT_IN * 100)

    # Pre-approve everything so approval gas isn't counted in swap rows.
    await erc20_approve(w3, WETH.address, _TEST_USER, vm_address, _MAX_UINT)
    await erc20_approve(w3, WETH.address, _TEST_USER, OKX_TOKEN_APPROVE_ETHEREUM, _MAX_UINT)
    await erc20_approve(w3, WETH.address, _TEST_USER, UNISWAP_V3_ROUTER, _MAX_UINT)
    # UR pulls via Permit2: ERC20.approve(Permit2), then Permit2.approve(token, UR).
    permit2_addr = Address(PERMIT2_ETHEREUM)
    await erc20_approve(w3, WETH.address, _TEST_USER, permit2_addr, _MAX_UINT)
    permit2_selector = function_signature_to_4byte_selector("approve(address,address,uint160,uint48)")
    permit2_args = abi_encode(
        ["address", "address", "uint160", "uint48"],
        [bytes(WETH.address), bytes(UNIVERSAL_ROUTER), (1 << 160) - 1, (1 << 48) - 1],
    )
    await send_ok(
        w3,
        _TEST_USER,
        {"to": permit2_addr, "data": "0x" + permit2_selector.hex() + permit2_args.hex(), "value": 0},
        "Permit2.approve(WETH, UR, max, max)",
    )

    # USDC for cross-protocol V3+V2 split (USDC→DAI). 6 decimals.
    await fund_usdc(w3, USDC.address, _TEST_USER, 10_000 * 10**6)
    await erc20_approve(w3, USDC.address, _TEST_USER, vm_address, _MAX_UINT)
    await erc20_approve(w3, USDC.address, _TEST_USER, OKX_TOKEN_APPROVE_ETHEREUM, _MAX_UINT)

    return {
        "w3": w3,
        "deployer": deployer,
        "vm_address": vm_address,
        "uni_v3_adapter": uni_v3_adapter_address,
        "uni_v2_adapter": uni_v2_adapter_address,
        "aave_v3_adapter": aave_v3_adapter_address,
        "user": _TEST_USER,
    }


async def _fetch_v3_pool_state(w3: AsyncWeb3, pool: Address) -> tuple:
    return await asyncio.gather(
        UNISWAP_V3_POOL.fns.slot0().call(w3, to=pool),
        UNISWAP_V3_POOL.fns.liquidity().call(w3, to=pool),
        UNISWAP_V3_POOL.fns.token0().call(w3, to=pool),
    )


async def _build_v3_edge(w3: AsyncWeb3, pool: Address, token_in: Token, token_out: Token, fee_bps: int) -> V3PoolEdge:
    slot0, liquidity, pool_token0 = await _fetch_v3_pool_state(w3, pool)
    is_token0_in = HexBytes(pool_token0) == HexBytes(token_in.address)
    return V3PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=pool,
        protocol="UniswapV3",
        fee_bps=fee_bps,
        sqrt_price_x96=slot0[0],
        liquidity=liquidity,
        is_token0_in=is_token0_in,
    )


async def _build_single_v3_dag(w3: AsyncWeb3) -> RouteDAG:
    """WETH → USDC via fee=500."""
    edge = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    return RouteDAG().from_token(WETH).swap(USDC, edge)


async def _build_two_hop_v3_dag(w3: AsyncWeb3) -> RouteDAG:
    """WETH → USDC fee=500 → DAI fee=100."""
    weth_usdc = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    usdc_dai = await _build_v3_edge(w3, POOL_DAI_USDC_100, USDC, DAI, fee_bps=1)
    return RouteDAG().from_token(WETH).swap(USDC, weth_usdc).swap(DAI, usdc_dai)


async def _build_split_v3_two_pools_dag(w3: AsyncWeb3) -> RouteDAG:
    """WETH → USDC, 50/50 across fee=500 + fee=3000."""
    edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, fee_bps=30)
    return RouteDAG().from_token(WETH).split().leg(5000).swap(USDC, edge_500).leg(5000).swap(USDC, edge_3000).merge()


async def _build_v2_edge(
    w3: AsyncWeb3, pair: Address, token_in: Token, token_out: Token, fee_bps: int = 30
) -> PoolEdge:
    reserves, pair_token0 = await asyncio.gather(
        UNISWAP_V2_PAIR.fns.getReserves().call(w3, to=pair),
        UNISWAP_V2_PAIR.fns.token0().call(w3, to=pair),
    )
    is_token0_in = HexBytes(pair_token0) == HexBytes(token_in.address)
    reserve_in = reserves[0] if is_token0_in else reserves[1]
    reserve_out = reserves[1] if is_token0_in else reserves[0]
    return PoolEdge(
        token_in=token_in,
        token_out=token_out,
        pool_address=pair,
        protocol="UniswapV2",
        reserve_in=reserve_in,
        reserve_out=reserve_out,
        fee_bps=fee_bps,
        extra={"is_token0_in": is_token0_in},
    )


async def _build_cross_protocol_split_dag(w3: AsyncWeb3) -> RouteDAG:
    """USDC → DAI, 50/50 V3 (fee=100) + V2. The V3-only routers (SR02 / UR /
    OKX uniswapV3SwapTo) can't express this without falling back to multi-tx."""
    v3_edge = await _build_v3_edge(w3, POOL_DAI_USDC_100, USDC, DAI, fee_bps=1)
    v2_edge = await _build_v2_edge(w3, PAIR_USDC_DAI, USDC, DAI, fee_bps=30)
    return RouteDAG().from_token(USDC).split().leg(5000).swap(DAI, v3_edge).leg(5000).swap(DAI, v2_edge).merge()


async def _build_fan_in_dag(w3: AsyncWeb3) -> RouteDAG:
    """WETH split across 2 V3 fee tiers → both end in USDC → merged USDC → DAI.
    Composer-only: multicall can't pass sum-of-prior-legs to a downstream call."""
    edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, fee_bps=30)
    edge_usdc_dai = await _build_v3_edge(w3, POOL_DAI_USDC_100, USDC, DAI, fee_bps=1)
    return (
        RouteDAG()
        .from_token(WETH)
        .split()
        .leg(5000)
        .swap(USDC, edge_500)
        .leg(5000)
        .swap(USDC, edge_3000)
        .merge()
        .swap(DAI, edge_usdc_dai)
    )


async def _build_split_v3_five_pools_dag(w3: AsyncWeb3) -> RouteDAG:
    """5-leg split: 3 single-hop + 2 multi-hop (via DAI and via USDT)."""
    edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, fee_bps=30)
    edge_10000 = await _build_v3_edge(w3, POOL_WETH_USDC_10000, WETH, USDC, fee_bps=100)
    edge_weth_dai = await _build_v3_edge(w3, POOL_WETH_DAI_3000, WETH, DAI, fee_bps=30)
    edge_dai_usdc = await _build_v3_edge(w3, POOL_DAI_USDC_100, DAI, USDC, fee_bps=1)
    edge_weth_usdt = await _build_v3_edge(w3, POOL_WETH_USDT_500, WETH, USDT, fee_bps=5)
    edge_usdt_usdc = await _build_v3_edge(w3, POOL_USDC_USDT_100, USDT, USDC, fee_bps=1)
    return (
        RouteDAG()
        .from_token(WETH)
        .split()
        .leg(3000)
        .swap(USDC, edge_500)
        .leg(2500)
        .swap(USDC, edge_3000)
        .leg(2000)
        .swap(USDC, edge_10000)
        .leg(1500)
        .swap(DAI, edge_weth_dai)
        .swap(USDC, edge_dai_usdc)
        .leg(1000)
        .swap(USDT, edge_weth_usdt)
        .swap(USDC, edge_usdt_usdc)
        .merge()
    )


async def _build_split_v3_four_pools_dag(w3: AsyncWeb3) -> RouteDAG:
    """4-leg split (40/30/20/10), last leg multi-hop via DAI. Forces
    encode_okx_smart's N-batch layout."""
    edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, fee_bps=30)
    edge_10000 = await _build_v3_edge(w3, POOL_WETH_USDC_10000, WETH, USDC, fee_bps=100)
    edge_weth_dai = await _build_v3_edge(w3, POOL_WETH_DAI_3000, WETH, DAI, fee_bps=30)
    edge_dai_usdc = await _build_v3_edge(w3, POOL_DAI_USDC_100, DAI, USDC, fee_bps=1)
    return (
        RouteDAG()
        .from_token(WETH)
        .split()
        .leg(4000)
        .swap(USDC, edge_500)
        .leg(3000)
        .swap(USDC, edge_3000)
        .leg(2000)
        .swap(USDC, edge_10000)
        .leg(1000)
        .swap(DAI, edge_weth_dai)
        .swap(USDC, edge_dai_usdc)
        .merge()
    )


async def _build_split_v3_three_pools_dag(w3: AsyncWeb3) -> RouteDAG:
    """3-leg split across all three V3 fee tiers (3333+3333+3334 bps)."""
    edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, fee_bps=5)
    edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, fee_bps=30)
    edge_10000 = await _build_v3_edge(w3, POOL_WETH_USDC_10000, WETH, USDC, fee_bps=100)
    return (
        RouteDAG()
        .from_token(WETH)
        .split()
        .leg(3333)
        .swap(USDC, edge_500)
        .leg(3333)
        .swap(USDC, edge_3000)
        .leg(3334)
        .swap(USDC, edge_10000)
        .merge()
    )


async def _run_encoder(
    bench_ctx: dict,
    *,
    path_name: str,
    encoder_name: str,
    token_out: Token | Address,
    encode: Callable[[], EncodedTx],
) -> BenchRow:
    """Execute one (path, encoder) cell of the bench matrix. ``token_out``
    can be a ``Token`` (V3 paths) or a raw ``Address`` (e.g. aUSDC)."""
    w3 = bench_ctx["w3"]
    user = bench_ctx["user"]
    out_addr = token_out.address if isinstance(token_out, Token) else token_out

    bal_before = await _balance(w3, out_addr, user)
    tx = encode()
    receipt = await send_ok(
        w3,
        user,
        {"to": tx.to, "data": "0x" + tx.data.hex(), "value": tx.value},
        f"{encoder_name} swap on {path_name}",
    )
    bal_after = await _balance(w3, out_addr, user)
    return BenchRow(
        path=path_name,
        encoder=encoder_name,
        gas_used=int(receipt["gasUsed"]),
        calldata_bytes=len(tx.data),
        amount_out=int(bal_after - bal_before),
    )


@asynccontextmanager
async def _snapshot_revert(w3: AsyncWeb3):
    """Take an ``evm_snapshot``, yield, ``evm_revert`` even if the body raises.
    Some web3 versions wrap the snapshot id in ``{"result": ...}``."""
    snap = await w3.provider.make_request("evm_snapshot", [])
    snap_id = snap["result"] if isinstance(snap, dict) else snap
    try:
        yield
    finally:
        await w3.provider.make_request("evm_revert", [snap_id])


async def _balance(w3: AsyncWeb3, token: Address | str, holder: Address) -> int:
    return int(await ERC20.fns.balanceOf(holder).call(w3, to=token))


async def _run_v3_matrix(
    bench_ctx: dict,
    *,
    dag: RouteDAG,
    token_out: Token,
    path_name: str,
    amount_in: int = _AMOUNT_IN,
) -> list[BenchRow]:
    """Snapshot-fair run of a pure-V3 DAG through all three encoders. Each row
    starts from identical pool state, so gas and amount_out are directly comparable."""
    w3 = bench_ctx["w3"]
    vm_address = bench_ctx["vm_address"]
    user = bench_ctx["user"]

    async def run(encoder_name: str, encode: Callable[[], EncodedTx]) -> BenchRow:
        async with _snapshot_revert(w3):
            return await _run_encoder(
                bench_ctx,
                path_name=path_name,
                encoder_name=encoder_name,
                token_out=token_out,
                encode=encode,
            )

    rows: list[BenchRow] = []
    rows.append(
        await run(
            "ssa",
            lambda: encode_ssa(
                dag,
                amount_in=amount_in,
                min_amount_out=0,
                recipient=user,
                vm_address=vm_address,
            ),
        )
    )
    rows.append(
        await run(
            "okx",
            lambda: encode_okx(
                dag,
                amount_in=amount_in,
                min_amount_out=0,
                recipient=user,
                router_address=OKX_DEX_ROUTER_ETHEREUM,
                v3_adapter_address=bench_ctx["uni_v3_adapter"],
                deadline=_DEADLINE,
            ),
        )
    )
    rows.append(
        await run(
            "uniswap",
            lambda: encode_uniswap(
                dag,
                amount_in=amount_in,
                min_amount_out=0,
                recipient=user,
                router_address=UNISWAP_V3_ROUTER,
                deadline=_DEADLINE,
            ),
        )
    )
    return rows


def _print_table(rows: list[BenchRow]) -> None:
    print()
    print(f"{'path':<22}  {'encoder':<12}  {'gas':>10}  {'calldata':>10}  {'amount_out':>16}")
    print("-" * 80)
    for r in rows:
        print(f"{r.path:<22}  {r.encoder:<12}  {r.gas_used:>10}  {r.calldata_bytes:>10}  {r.amount_out:>16}")


def _print_dag_shape(dag: RouteDAG, *, label: str) -> None:
    from pydefi.types import RouteSplit

    if len(dag.actions) == 1 and isinstance(dag.actions[0], RouteSplit):
        bps = [leg.fraction_bps for leg in dag.actions[0].legs]
        print(f"\n{label} DAG: 1-action split, {len(bps)} legs, bps={bps}")
    else:
        types = [type(a).__name__ for a in dag.actions]
        print(f"\n{label} DAG: {len(dag.actions)} actions = {types}")


@pytest.mark.bench
@pytest.mark.fork
class TestAggregationGas:
    """Encoder gas matrix — printed as a table when run with ``-s``."""

    async def test_single_v3(self, bench_ctx: dict) -> None:
        dag = await _build_single_v3_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=USDC, path_name="single_v3")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_two_hop_v3(self, bench_ctx: dict) -> None:
        dag = await _build_two_hop_v3_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=DAI, path_name="two_hop_v3")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_split_v3_two_pools(self, bench_ctx: dict) -> None:
        dag = await _build_split_v3_two_pools_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=USDC, path_name="split_v3_two_pools")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_split_v3_three_pools(self, bench_ctx: dict) -> None:
        dag = await _build_split_v3_three_pools_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=USDC, path_name="split_v3_three_pools")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_split_v3_four_pools(self, bench_ctx: dict) -> None:
        """4-leg split: SSA's per-leg lead over multicall should net out
        ahead here (per-leg ~70-77K vs ~75-81K)."""
        dag = await _build_split_v3_four_pools_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=USDC, path_name="split_v3_four_pools")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_split_v3_five_pools(self, bench_ctx: dict) -> None:
        """5-leg split with two multi-hop legs — confirms SSA's lead keeps growing."""
        dag = await _build_split_v3_five_pools_dag(bench_ctx["w3"])
        rows = await _run_v3_matrix(bench_ctx, dag=dag, token_out=USDC, path_name="split_v3_five_pools")
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_swap_then_aave_supply(self, bench_ctx: dict) -> None:
        """Heterogeneous bundle: WETH → USDC → Aave supply, one atomic tx.
        SSA via composer; OKX via smartSwapTo with [V3 adapter, MinimalAaveV3Adapter]
        sequential hops (V3 pool output routed directly into the Aave adapter,
        which calls pool.supply(USDC, amount, recipient, 0) → aUSDC to user)."""
        w3 = bench_ctx["w3"]
        user = bench_ctx["user"]
        vm_address = bench_ctx["vm_address"]
        dag = await _build_single_v3_dag(w3)
        au_usdc = Address("0x98C23E9d8f34FEFb1B7BD6a91B7FF122F4e16F5c")

        async def run(encoder_name: str, encode: Callable[[], EncodedTx]) -> BenchRow:
            async with _snapshot_revert(w3):
                return await _run_encoder(
                    bench_ctx,
                    path_name="swap_aave_supply",
                    encoder_name=encoder_name,
                    token_out=au_usdc,
                    encode=encode,
                )

        rows = [
            await run(
                "ssa",
                lambda: encode_ssa_swap_then_aave_supply(
                    dag,
                    amount_in=_AMOUNT_IN,
                    min_amount_out=0,
                    recipient=user,
                    vm_address=vm_address,
                    aave_pool=AAVE_V3_POOL_ETHEREUM,
                ),
            ),
            await run(
                "okx",
                lambda: encode_okx_swap_then_aave_supply(
                    dag,
                    amount_in=_AMOUNT_IN,
                    min_amount_out=0,
                    recipient=user,
                    router_address=OKX_DEX_ROUTER_ETHEREUM,
                    v3_adapter_address=bench_ctx["uni_v3_adapter"],
                    aave_adapter_address=bench_ctx["aave_v3_adapter"],
                    aave_pool=AAVE_V3_POOL_ETHEREUM,
                    aave_atoken=au_usdc,
                    deadline=_DEADLINE,
                ),
            ),
        ]
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_min_out_overhead(self, bench_ctx: dict) -> None:
        """Whole-route min-out gas + semantic comparison on split_v3_two_pools.
        The finding isn't gas (deltas <3K) — it's that SR02.multicall has
        no whole-route check, so encode_uniswap raises rather than silently
        weakening slippage to per-leg 0."""
        w3 = bench_ctx["w3"]
        user = bench_ctx["user"]
        vm_address = bench_ctx["vm_address"]
        dag = await _build_split_v3_two_pools_dag(w3)

        def ssa_encode(min_out: int) -> Callable[[], EncodedTx]:
            return lambda: encode_ssa(
                dag,
                amount_in=_AMOUNT_IN,
                min_amount_out=min_out,
                recipient=user,
                vm_address=vm_address,
            )

        def okx_encode(min_out: int) -> Callable[[], EncodedTx]:
            return lambda: encode_okx(
                dag,
                amount_in=_AMOUNT_IN,
                min_amount_out=min_out,
                recipient=user,
                router_address=OKX_DEX_ROUTER_ETHEREUM,
                v3_adapter_address=bench_ctx["uni_v3_adapter"],
                deadline=_DEADLINE,
            )

        async def run(encoder_name: str, encode_fn: Callable[[], EncodedTx]) -> BenchRow:
            async with _snapshot_revert(w3):
                bal_before = await _balance(w3, USDC.address, user)
                tx = encode_fn()
                receipt = await send_ok(
                    w3,
                    user,
                    {"to": tx.to, "data": "0x" + tx.data.hex(), "value": tx.value},
                    f"{encoder_name} with min_out",
                )
                bal_after = await _balance(w3, USDC.address, user)
                return BenchRow(
                    path="min_out_overhead",
                    encoder=encoder_name,
                    gas_used=int(receipt["gasUsed"]),
                    calldata_bytes=len(tx.data),
                    amount_out=int(bal_after - bal_before),
                )

        # Live zero-min-out baselines (gas + amount_out) from the same
        # snapshot-fair state — replaces hardcoded per-block constants.
        zero_ssa = await run("ssa", ssa_encode(0))
        zero_okx = await run("okx", okx_encode(0))

        # Whole-route min-out at 99.9% of the measured (snapshot-fair) output,
        # not a constant hand-copied from a prior run at this pinned block.
        min_total = min(zero_ssa.amount_out, zero_okx.amount_out) * 999 // 1000

        rows: list[BenchRow] = [
            await run("ssa(whole)", ssa_encode(min_total)),
            await run("okx(whole)", okx_encode(min_total)),
        ]
        # Encoder must refuse: silently weakening to per-leg 0 would mislead callers.
        uni_refused: str | None = None
        try:
            encode_uniswap(
                dag,
                amount_in=_AMOUNT_IN,
                min_amount_out=min_total,
                recipient=user,
                router_address=UNISWAP_V3_ROUTER,
                deadline=_DEADLINE,
            )
        except ValueError as e:
            uni_refused = str(e)
        assert uni_refused is not None, "encode_uniswap should raise on split + min_out>0"

        _print_table(rows)
        # Baselines measured live above (zero-min-out), not pinned to a block.
        baselines = {"ssa(whole)": zero_ssa.gas_used, "okx(whole)": zero_okx.gas_used}
        for r in rows:
            delta = r.gas_used - baselines[r.encoder]
            print(f"  {r.encoder:<14}  Δgas vs zero-min-out = {delta:+d}")
        print(f"  uni(multicall)  raised: {uni_refused[:80]}...")

    async def test_cross_protocol_v3_v2_split(self, bench_ctx: dict) -> None:
        """USDC → DAI, 50/50 V3 (fee=100) + V2. SSA via composer; OKX via
        smartSwapTo with mixed [V3 adapter, V2 adapter] (V2 funds go to
        pool, V3 funds go to adapter); Uniswap V3-only routers can't
        express V2."""
        w3 = bench_ctx["w3"]
        user = bench_ctx["user"]
        vm_address = bench_ctx["vm_address"]
        dag = await _build_cross_protocol_split_dag(w3)
        _print_dag_shape(dag, label="cross-protocol")

        usdc_in = 100 * 10**6  # 100 USDC, 6 decimals

        async def run(encoder_name: str, encode: Callable[[], EncodedTx]) -> BenchRow:
            async with _snapshot_revert(w3):
                return await _run_encoder(
                    bench_ctx,
                    path_name="cross_protocol_v3_v2",
                    encoder_name=encoder_name,
                    token_out=DAI,
                    encode=encode,
                )

        rows = [
            await run(
                "ssa",
                lambda: encode_ssa(
                    dag,
                    amount_in=usdc_in,
                    min_amount_out=0,
                    recipient=user,
                    vm_address=vm_address,
                ),
            ),
            await run(
                "okx",
                lambda: encode_okx(
                    dag,
                    amount_in=usdc_in,
                    min_amount_out=0,
                    recipient=user,
                    router_address=OKX_DEX_ROUTER_ETHEREUM,
                    v3_adapter_address=bench_ctx["uni_v3_adapter"],
                    v2_adapter_address=bench_ctx["uni_v2_adapter"],
                    deadline=_DEADLINE,
                ),
            ),
        ]
        _print_table(rows)
        _assert_amount_out_consistent(rows)

    async def test_universal_router_v3(self, bench_ctx: dict) -> None:
        """Modern Uniswap baseline (SR02 is now legacy). UR uses command-byte
        dispatch + Permit2 pulls. Compares against the existing SR02 rows."""
        w3 = bench_ctx["w3"]
        user = bench_ctx["user"]
        rows: list[BenchRow] = []
        for name, dag_builder in [
            ("ur_single_v3", _build_single_v3_dag),
            ("ur_two_hop_v3", _build_two_hop_v3_dag),
        ]:
            dag = await dag_builder(w3)
            token_out = USDC if name == "ur_single_v3" else DAI
            async with _snapshot_revert(w3):
                tx = encode_universal_router(
                    dag,
                    amount_in=_AMOUNT_IN,
                    min_amount_out=0,
                    recipient=user,
                    router_address=UNIVERSAL_ROUTER,
                    deadline=_DEADLINE,
                )
                bal_before = await _balance(w3, token_out.address, user)
                receipt = await send_ok(
                    w3,
                    user,
                    {"to": tx.to, "data": "0x" + tx.data.hex(), "value": tx.value},
                    f"UR {name}",
                )
                bal_after = await _balance(w3, token_out.address, user)
                rows.append(
                    BenchRow(
                        path=name,
                        encoder="universal_router",
                        gas_used=int(receipt["gasUsed"]),
                        calldata_bytes=len(tx.data),
                        amount_out=int(bal_after - bal_before),
                    )
                )
        _print_table(rows)
        for r in rows:
            assert r.amount_out > 0, f"{r.path}: expected positive output, got {r.amount_out}"

    async def test_fan_in_dag(self, bench_ctx: dict) -> None:
        """Fan-in DAG: 2-leg split → both end in USDC → merged USDC → DAI.
        SSA via composer; OKX via DagRouter.dagSwapTo (true DAG executor
        with per-edge input/output indices); Uniswap multicall can't feed
        sum-of-prior-legs into a downstream call."""
        w3 = bench_ctx["w3"]
        user = bench_ctx["user"]
        vm_address = bench_ctx["vm_address"]
        dag = await _build_fan_in_dag(w3)
        _print_dag_shape(dag, label="fan-in")

        async def run(encoder_name: str, encode: Callable[[], EncodedTx]) -> BenchRow:
            async with _snapshot_revert(w3):
                return await _run_encoder(
                    bench_ctx,
                    path_name="fan_in_dag",
                    encoder_name=encoder_name,
                    token_out=DAI,
                    encode=encode,
                )

        rows = [
            await run(
                "ssa",
                lambda: encode_ssa(
                    dag,
                    amount_in=_AMOUNT_IN,
                    min_amount_out=0,
                    recipient=user,
                    vm_address=vm_address,
                ),
            ),
            await run(
                "okx",
                lambda: encode_okx(
                    dag,
                    amount_in=_AMOUNT_IN,
                    min_amount_out=0,
                    recipient=user,
                    router_address=OKX_DEX_ROUTER_ETHEREUM,
                    v3_adapter_address=bench_ctx["uni_v3_adapter"],
                    deadline=_DEADLINE,
                ),
            ),
        ]
        _print_table(rows)
        _assert_amount_out_consistent(rows)


def _assert_amount_out_consistent(rows: list[BenchRow], *, threshold: float = 0.001) -> None:
    """Snapshot-fair harness makes amount_out wei-exact; threshold is a safety
    margin for callers that bypass the snapshot wrapper."""
    outs = [r.amount_out for r in rows]
    assert min(outs) > 0, f"some encoder returned 0 amount_out: {outs}"
    spread = (max(outs) - min(outs)) / max(outs)
    assert spread < threshold, f"amount_out spread {spread:.4%} > {threshold:.2%}: {outs}"

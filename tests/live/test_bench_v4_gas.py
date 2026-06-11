"""Gas benchmark — Uniswap V4 vs V3 swap encodings on a pinned mainnet fork.

Every row executes through the same Universal Router on the same fork block, so
the deltas isolate protocol mechanics (V4 flash accounting in the singleton
PoolManager vs V3 per-pool transfers), not router dispatch. DeFiVM/SSA rows are
absent by design: a V4 swap needs the PoolManager unlock callback, which a
composed CALL program cannot provide (pydefi/vm/swap.py raises).

V4 pool keys are pinned to no-hooks pools verified (via StateView) to hold
liquidity at the fork block: WETH/USDC 500/10, WETH/USDC 3000/60, USDC/USDT 10/1.

Run with ``pytest -m bench tests/live/test_bench_v4_gas.py -s``
(``-s`` is required, pytest swallows the table without it).
"""

from __future__ import annotations

from typing import Callable

import pytest
from web3 import AsyncWeb3

from pydefi.amm.uniswap_v4 import UniswapV4
from pydefi.pathfinder.graph import V4PoolEdge
from pydefi.types import Address, RouteDAG, Token
from tests.addrs import (
    UNISWAP_V4_POOL_MANAGER,
    UNISWAP_V4_QUOTER,
    UNISWAP_V4_STATE_VIEW,
    UNIVERSAL_ROUTER,
    USDC,
    USDT,
    WETH,
)
from tests.bench.encoders import (
    EncodedTx,
    encode_universal_router,
    encode_universal_router_v3_split,
    encode_universal_router_v4_split,
)
from tests.live.anvil_helpers import impersonate, permit2_approve, set_balance, wrap_eth
from tests.live.test_bench_aggregation_gas import (
    _AMOUNT_IN,
    _DEADLINE,
    _TEST_USER,
    POOL_USDC_USDT_100,
    POOL_WETH_USDC_500,
    POOL_WETH_USDC_3000,
    BenchRow,
    _assert_amount_out_consistent,
    _build_v3_edge,
    _print_table,
    _run_encoder,
    _snapshot_revert,
    bench_fork_w3,  # noqa: F401 — module-scoped anvil fork fixture
)

# V4 PoolKey (fee pips, tick spacing) constants — see module docstring.
V4_WETH_USDC_500 = (500, 10)
V4_WETH_USDC_3000 = (3000, 60)
V4_USDC_USDT_10 = (10, 1)


@pytest.fixture(scope="module")
async def v4_client(bench_fork_w3: AsyncWeb3) -> UniswapV4:  # noqa: F811
    """UniswapV4 client wired to the mainnet singleton on the fork."""
    return UniswapV4(
        w3=bench_fork_w3,
        pool_manager_address=UNISWAP_V4_POOL_MANAGER,
        state_view_address=UNISWAP_V4_STATE_VIEW,
        quoter_address=UNISWAP_V4_QUOTER,
    )


@pytest.fixture(scope="module")
async def v4_bench_ctx(bench_fork_w3: AsyncWeb3) -> dict:  # noqa: F811
    """Fund the test user and grant the Permit2 → Universal Router allowance,
    so approval gas is identical across rows and excluded from measurements."""
    w3 = bench_fork_w3
    await impersonate(w3, _TEST_USER)
    await set_balance(w3, _TEST_USER, 100 * 10**21)
    await wrap_eth(w3, _TEST_USER, WETH.address, _AMOUNT_IN * 100)
    await permit2_approve(w3, _TEST_USER, WETH.address, UNIVERSAL_ROUTER)
    return {"w3": w3, "user": _TEST_USER}


async def _build_v4_edge(v4_client: UniswapV4, token_in: Token, token_out: Token, key: tuple[int, int]) -> V4PoolEdge:
    fee, tick_spacing = key
    return await v4_client.get_pool_edge(token_in, token_out, fee=fee, tick_spacing=tick_spacing)


async def _run_rows(
    v4_bench_ctx: dict,
    *,
    path_name: str,
    token_out: Token,
    encoders: list[tuple[str, Callable[[], EncodedTx]]],
) -> list[BenchRow]:
    """Snapshot-fair run: every row starts from identical fork state."""
    rows: list[BenchRow] = []
    for encoder_name, encode in encoders:
        async with _snapshot_revert(v4_bench_ctx["w3"]):
            rows.append(
                await _run_encoder(
                    v4_bench_ctx,
                    path_name=path_name,
                    encoder_name=encoder_name,
                    token_out=token_out,
                    encode=encode,
                )
            )
    return rows


def _ur_encode(encoder, dag: RouteDAG, user: Address) -> Callable[[], EncodedTx]:
    return lambda: encoder(
        dag,
        amount_in=_AMOUNT_IN,
        min_amount_out=0,
        recipient=user,
        router_address=UNIVERSAL_ROUTER,
        deadline=_DEADLINE,
    )


@pytest.mark.bench
@pytest.mark.fork
class TestV4Gas:
    """V4 vs V3 gas matrix via the Universal Router — printed with ``-s``."""

    async def test_single_hop(self, v4_bench_ctx: dict, v4_client: UniswapV4) -> None:
        """WETH → USDC on the fee=500 tier of each protocol: the gas delta is
        V3 pool mechanics vs V4 unlock + swap + settle/take on the singleton."""
        w3 = v4_bench_ctx["w3"]
        user = v4_bench_ctx["user"]
        v3_dag = RouteDAG().from_token(WETH).swap(USDC, await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, 5))
        v4_dag = RouteDAG().from_token(WETH).swap(USDC, await _build_v4_edge(v4_client, WETH, USDC, V4_WETH_USDC_500))

        rows = await _run_rows(
            v4_bench_ctx,
            path_name="ur_single",
            token_out=USDC,
            encoders=[
                ("v3", _ur_encode(encode_universal_router, v3_dag, user)),
                ("v4", _ur_encode(encode_universal_router, v4_dag, user)),
            ],
        )
        _print_table(rows)
        # Both fee=500 pools are deep and arbed at the pinned block.
        _assert_amount_out_consistent(rows, threshold=0.02)

    async def test_two_hop(self, v4_bench_ctx: dict, v4_client: UniswapV4) -> None:
        """WETH → USDC → USDT, two hops per protocol — where flash accounting
        pays off: the V4 intermediate USDC never leaves the PoolManager."""
        w3 = v4_bench_ctx["w3"]
        user = v4_bench_ctx["user"]
        v3_dag = (
            RouteDAG()
            .from_token(WETH)
            .swap(USDC, await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, 5))
            .swap(USDT, await _build_v3_edge(w3, POOL_USDC_USDT_100, USDC, USDT, 1))
        )
        v4_dag = (
            RouteDAG()
            .from_token(WETH)
            .swap(USDC, await _build_v4_edge(v4_client, WETH, USDC, V4_WETH_USDC_500))
            .swap(USDT, await _build_v4_edge(v4_client, USDC, USDT, V4_USDC_USDT_10))
        )

        rows = await _run_rows(
            v4_bench_ctx,
            path_name="ur_two_hop",
            token_out=USDT,
            encoders=[
                ("v3", _ur_encode(encode_universal_router, v3_dag, user)),
                ("v4", _ur_encode(encode_universal_router, v4_dag, user)),
            ],
        )
        _print_table(rows)
        _assert_amount_out_consistent(rows, threshold=0.02)

    async def test_split_two_pools(self, v4_bench_ctx: dict, v4_client: UniswapV4) -> None:
        """WETH → USDC, 50/50 across the 500 + 3000 tiers of each protocol. V4
        runs both legs in one V4_SWAP / PoolManager lock with a single settle
        and take — the route shape the EdgeKey diversity fix (#164) enables —
        while the V3 baseline pays one command + Permit2 pull per leg."""
        w3 = v4_bench_ctx["w3"]
        user = v4_bench_ctx["user"]
        v3_edge_500 = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, 5)
        v3_edge_3000 = await _build_v3_edge(w3, POOL_WETH_USDC_3000, WETH, USDC, 30)
        v3_dag = (
            RouteDAG()
            .from_token(WETH)
            .split()
            .leg(5000)
            .swap(USDC, v3_edge_500)
            .leg(5000)
            .swap(USDC, v3_edge_3000)
            .merge()
        )
        v4_edge_500 = await _build_v4_edge(v4_client, WETH, USDC, V4_WETH_USDC_500)
        v4_edge_3000 = await _build_v4_edge(v4_client, WETH, USDC, V4_WETH_USDC_3000)
        v4_dag = (
            RouteDAG()
            .from_token(WETH)
            .split()
            .leg(5000)
            .swap(USDC, v4_edge_500)
            .leg(5000)
            .swap(USDC, v4_edge_3000)
            .merge()
        )

        rows = await _run_rows(
            v4_bench_ctx,
            path_name="ur_split_two_pools",
            token_out=USDC,
            encoders=[
                ("v3", _ur_encode(encode_universal_router_v3_split, v3_dag, user)),
                ("v4", _ur_encode(encode_universal_router_v4_split, v4_dag, user)),
            ],
        )
        _print_table(rows)
        # The V4 fee=3000 pool is thinner, so its leg slips a bit more.
        _assert_amount_out_consistent(rows, threshold=0.05)

    async def test_mixed_v3_v4_two_hop(self, v4_bench_ctx: dict, v4_client: UniswapV4) -> None:
        """WETH → USDC → USDT crossing protocols in both directions — the
        segment transitions the pathfinder can now emit (mid-route V4 settles
        the router's ERC-20 balance, mid-route V3 spends CONTRACT_BALANCE).
        Compare with the pure two-hop rows to price the transition overhead."""
        w3 = v4_bench_ctx["w3"]
        user = v4_bench_ctx["user"]
        v3_weth_usdc = await _build_v3_edge(w3, POOL_WETH_USDC_500, WETH, USDC, 5)
        v3_usdc_usdt = await _build_v3_edge(w3, POOL_USDC_USDT_100, USDC, USDT, 1)
        v4_weth_usdc = await _build_v4_edge(v4_client, WETH, USDC, V4_WETH_USDC_500)
        v4_usdc_usdt = await _build_v4_edge(v4_client, USDC, USDT, V4_USDC_USDT_10)

        v3_to_v4 = RouteDAG().from_token(WETH).swap(USDC, v3_weth_usdc).swap(USDT, v4_usdc_usdt)
        v4_to_v3 = RouteDAG().from_token(WETH).swap(USDC, v4_weth_usdc).swap(USDT, v3_usdc_usdt)

        rows = await _run_rows(
            v4_bench_ctx,
            path_name="ur_mixed_two_hop",
            token_out=USDT,
            encoders=[
                ("v3->v4", _ur_encode(encode_universal_router, v3_to_v4, user)),
                ("v4->v3", _ur_encode(encode_universal_router, v4_to_v3, user)),
            ],
        )
        _print_table(rows)
        _assert_amount_out_consistent(rows, threshold=0.02)

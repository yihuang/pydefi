"""Calldata encoders for the DEX aggregation gas benchmark.

Every encoder prepends a ``transferFrom(token_in, caller, executor, amount)``
so all three caller models (DeFiVM / OKX TokenApprove / SwapRouter02) are
self-contained and their gas numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from eth_abi import encode as abi_encode
from eth_contract import Contract

from pydefi.abi.amm import UNISWAP_V3_ROUTER, ExactInputParams, ExactInputSingleParams
from pydefi.abi.dex_aggregator import (
    ADDRESS_MASK,
    INPUT_INDEX_SHIFT,
    OKX_DEX_ROUTER,
    ONE_FOR_ZERO_MASK,
    OUTPUT_INDEX_SHIFT,
    REVERSE_MASK,
    WEIGHT_SHIFT,
    BaseRequest,
    RouterPath,
)
from pydefi.abi.vm import DeFiVM
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.amm.universal_router import (
    PoolHop,
    RouterCommand,
    UniversalRouter,
    V2Hop,
    V3Hop,
    V4Action,
    V4Hop,
)
from pydefi.amm.v4_pool_key import sort_currencies
from pydefi.pathfinder.graph import PoolEdge, V3PoolEdge, V4PoolEdge
from pydefi.types import (
    Address,
    RouteDAG,
    RouteSplit,
    RouteSwap,
    SwapProtocol,
    TokenAmount,
)
from pydefi.vm import Program
from pydefi.vm.dag import _build_dag_actions
from pydefi.vm.swap import _pool_to_swap_protocol

# SwapRouter02.multicall — not in pydefi.abi.amm.UNISWAP_V3_ROUTER.
_UNISWAP_V3_MULTICALL = Contract.from_abi(
    ["function multicall(bytes[] data) external payable returns (bytes[] results)"]
)


# ---------------------------------------------------------------------------
# EncodedTx — uniform return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedTx:
    to: Address
    data: bytes
    value: int


def _edge_to_hop(edge) -> PoolHop:
    """Map a pathfinder pool edge to the matching Universal Router hop. V4 hops
    rebuild the PoolKey from the edge, so dynamic-fee pools (no static key fee
    on the edge) are refused."""
    if isinstance(edge, V4PoolEdge):
        if edge.lp_fee_pips & 0x800000:
            raise ValueError("_edge_to_hop: dynamic-fee V4 pools carry no static key fee")
        return V4Hop(
            token_in=edge.token_in,
            token_out=edge.token_out,
            fee=edge.lp_fee_pips,
            tick_spacing=edge.tick_spacing,
            hooks=edge.hooks,
        )
    if isinstance(edge, V3PoolEdge):
        return V3Hop(token_in=edge.token_in, token_out=edge.token_out, fee=edge.fee_bps * 100)
    return V2Hop(token_in=edge.token_in, token_out=edge.token_out)


def encode_universal_router(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    deadline: int,
) -> EncodedTx:
    """Linear V2/V3/V4 (or mixed) path via the Universal Router. Single hops use
    the dedicated single-swap encodings; multi-hop paths go through
    ``build_multihop_exact_in_transaction``, which merges consecutive
    same-protocol hops into one command per segment. Permit2 setup happens in
    the bench fixture; not counted in gas."""
    if dag.token_in is None:
        raise ValueError("encode_universal_router: DAG missing token_in")
    if not all(isinstance(a, RouteSwap) for a in dag.actions):
        raise ValueError("encode_universal_router: linear paths only (no splits)")

    ur = UniversalRouter(router_address)
    total_in = TokenAmount(token=dag.token_in, amount=amount_in)
    hops = [_edge_to_hop(a.pool) for a in dag.actions]
    if len(hops) == 1 and isinstance(hops[0], V3Hop):
        tx = ur.build_v3_exact_in_transaction(
            amount_in=total_in,
            token_out=hops[0].token_out,
            recipient=recipient,
            amount_out_minimum=min_amount_out,
            fee=hops[0].fee,
            deadline=deadline,
        )
    elif len(hops) == 1 and isinstance(hops[0], V4Hop):
        tx = ur.build_v4_exact_in_single_transaction(
            amount_in=total_in,
            token_out=hops[0].token_out,
            fee=hops[0].fee,
            tick_spacing=hops[0].tick_spacing,
            recipient=recipient,
            amount_out_minimum=min_amount_out,
            hooks=hops[0].hooks,
            deadline=deadline,
        )
    else:
        tx = ur.build_multihop_exact_in_transaction(
            amount_in=total_in,
            hops=hops,
            recipient=recipient,
            amount_out_minimum=min_amount_out,
            deadline=deadline,
        )
    return EncodedTx(to=tx.to, data=bytes(tx.data), value=0)


def _split_leg_edges(dag: RouteDAG, encoder_name: str) -> list:
    """Validate a one-action single-hop split DAG and return one edge per leg."""
    if dag.token_in is None:
        raise ValueError(f"{encoder_name}: DAG missing token_in")
    if len(dag.actions) != 1 or not isinstance(dag.actions[0], RouteSplit):
        raise ValueError(f"{encoder_name}: DAG must be a single RouteSplit action")
    split = dag.actions[0]
    edges = []
    for leg in split.legs:
        if len(leg.actions) != 1 or not isinstance(leg.actions[0], RouteSwap):
            raise ValueError(f"{encoder_name}: each split leg must be a single swap")
        edges.append(leg.actions[0].pool)
    return edges


def _split_amounts(amount_in: int, legs) -> list[int]:
    """Per-leg input amounts: floor by fraction_bps, remainder to the last leg."""
    amounts = [amount_in * leg.fraction_bps // 10_000 for leg in legs]
    amounts[-1] += amount_in - sum(amounts)
    return amounts


def encode_universal_router_v4_split(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    deadline: int,
) -> EncodedTx:
    """Single-hop split across V4 pools as ONE ``V4_SWAP`` command: all legs run
    in one PoolManager lock, then one ``SETTLE_ALL`` pulls the total input via
    Permit2 and one take pays out — no per-leg transfers. A non-zero
    ``min_amount_out`` is enforced whole-route via ``TAKE_ALL``, which pays the
    transaction sender (no recipient-carrying take-with-minimum action exists);
    with ``min_amount_out == 0`` the output goes to ``recipient`` via ``TAKE``."""
    edges = _split_leg_edges(dag, "encode_universal_router_v4_split")
    split = dag.actions[0]
    assert isinstance(split, RouteSplit)
    token_in = dag.token_in
    assert token_in is not None
    token_out = split.token_out
    currency0, currency1 = sort_currencies(token_in.address, token_out.address)

    ur = UniversalRouter(router_address)
    actions: list[V4Action | int] = []
    params: list[bytes] = []
    for edge, leg_amount in zip(edges, _split_amounts(amount_in, split.legs)):
        if not isinstance(edge, V4PoolEdge):
            raise ValueError("encode_universal_router_v4_split: V4 pools only")
        actions.append(V4Action.SWAP_EXACT_IN_SINGLE)
        params.append(
            ur.encode_v4_exact_in_single_params(
                currency0=currency0,
                currency1=currency1,
                fee=edge.lp_fee_pips,
                tick_spacing=edge.tick_spacing,
                hooks=edge.hooks,
                zero_for_one=token_in.address == currency0,
                amount_in=leg_amount,
                amount_out_minimum=0,
            )
        )

    actions.append(V4Action.SETTLE_ALL)
    params.append(ur.encode_v4_settle_all_params(token_in.address, amount_in))
    if min_amount_out > 0:
        actions.append(V4Action.TAKE_ALL)
        params.append(ur.encode_v4_take_all_params(token_out.address, min_amount_out))
    else:
        actions.append(V4Action.TAKE)
        params.append(ur.encode_v4_take_params(token_out.address, recipient, 0))

    v4_input = ur.encode_v4_swap_actions(actions, params)
    calldata = ur.build_execute_calldata([RouterCommand.V4_SWAP], [v4_input], deadline)
    return EncodedTx(to=router_address, data=bytes(calldata), value=0)


def encode_universal_router_v3_split(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    deadline: int,
) -> EncodedTx:
    """Single-hop split across V3 pools as one ``V3_SWAP_EXACT_IN`` command per
    leg — the UR split baseline. Like SR02.multicall there is no whole-route
    output check, so a non-zero ``min_amount_out`` is refused rather than
    silently weakened to per-leg zero (same semantics as :func:`encode_uniswap`)."""
    if min_amount_out > 0:
        raise ValueError(
            "encode_universal_router_v3_split: per-command V3 splits cannot enforce a whole-route min_amount_out"
        )
    edges = _split_leg_edges(dag, "encode_universal_router_v3_split")
    split = dag.actions[0]
    assert isinstance(split, RouteSplit)
    token_in = dag.token_in
    assert token_in is not None

    ur = UniversalRouter(router_address)
    commands: list[RouterCommand | int] = []
    inputs: list[bytes] = []
    for edge, leg_amount in zip(edges, _split_amounts(amount_in, split.legs)):
        if not isinstance(edge, V3PoolEdge) or isinstance(edge, V4PoolEdge):
            raise ValueError("encode_universal_router_v3_split: V3 pools only")
        path = UniswapV3._encode_path([token_in, split.token_out], [edge.fee_bps * 100])
        commands.append(RouterCommand.V3_SWAP_EXACT_IN)
        inputs.append(ur.encode_v3_swap_exact_in(recipient, leg_amount, 0, path, payer_is_user=True))

    calldata = ur.build_execute_calldata(commands, inputs, deadline)
    return EncodedTx(to=router_address, data=bytes(calldata), value=0)


# ---------------------------------------------------------------------------
# SSA composer
# ---------------------------------------------------------------------------


def encode_ssa(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    vm_address: Address,
) -> EncodedTx:
    """Compile the RouteDAG to DeFiVM bytecode with a ``transferFrom`` prelude
    so the swap tx is self-contained (matches OKX/Uniswap caller model)."""
    if dag.token_in is None:
        raise ValueError("encode_ssa: DAG missing token_in")
    if not dag.actions:
        raise ValueError("encode_ssa: DAG must contain at least one action")

    prog = Program()
    success = prog.call_contract(
        bytes(dag.token_in.address),
        "transferFrom(address,address,uint256)",
        prog.builder.caller(),
        prog.builder.address(),
        amount_in,
    )
    prog.assert_(success, "ssa: pull failed")

    final_out = _build_dag_actions(
        prog,
        prog.const(amount_in),
        dag.actions,
        vm_address=vm_address,
        terminal_recipient=recipient,
    )
    if min_amount_out > 0:
        prog.assert_ge(final_out, min_amount_out, "slippage")
    prog.builder.stop()

    bytecode = prog.build()
    data = DeFiVM.fns.execute(bytecode).data
    return EncodedTx(to=vm_address, data=bytes(data), value=0)


# ---------------------------------------------------------------------------
# Heterogeneous bundle — swap + Aave V3 supply in one tx
# ---------------------------------------------------------------------------


def encode_ssa_swap_then_aave_supply(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    vm_address: Address,
    aave_pool: Address,
) -> EncodedTx:
    """Pull token_in, swap, then supply the output to Aave V3 on the user's
    behalf — one tx, one approval, one min-out. Swap terminus is the VM so
    the output is held locally for approve+supply; Aave's ``onBehalfOf``
    routes aToken back to the user."""
    if dag.token_in is None:
        raise ValueError("encode_ssa_swap_then_aave_supply: DAG missing token_in")
    if not dag.actions:
        raise ValueError("encode_ssa_swap_then_aave_supply: DAG must contain at least one action")

    last = dag.actions[-1]
    assert isinstance(last, (RouteSplit, RouteSwap))
    out_token_addr = bytes(last.token_out.address)

    prog = Program()
    success = prog.call_contract(
        bytes(dag.token_in.address),
        "transferFrom(address,address,uint256)",
        prog.builder.caller(),
        prog.builder.address(),
        amount_in,
    )
    prog.assert_(success, "ssa-supply: pull failed")

    swap_out = _build_dag_actions(
        prog,
        prog.const(amount_in),
        dag.actions,
        vm_address=vm_address,
        terminal_recipient=vm_address,
    )
    if min_amount_out > 0:
        prog.assert_ge(swap_out, min_amount_out, "slippage")

    # snapshot/revert harness resets allowance, so each run pays cold SSTORE ~22.1K.
    approve_ok = prog.call_contract(
        out_token_addr,
        "approve(address,uint256)",
        bytes(aave_pool),
        2**256 - 1,
    )
    prog.assert_(approve_ok, "ssa-supply: approve failed")

    supply_ok = prog.call_contract(
        bytes(aave_pool),
        "supply(address,uint256,address,uint16)",
        out_token_addr,
        swap_out,
        bytes(recipient),
        0,
    )
    prog.assert_(supply_ok, "ssa-supply: supply failed")
    prog.builder.stop()

    bytecode = prog.build()
    data = DeFiVM.fns.execute(bytecode).data
    return EncodedTx(to=vm_address, data=bytes(data), value=0)


# ---------------------------------------------------------------------------
# OKX router
# ---------------------------------------------------------------------------


def _is_v3_swap(action) -> bool:
    if not isinstance(action, RouteSwap):
        return False
    try:
        return _pool_to_swap_protocol(action.pool.protocol) == SwapProtocol.UNISWAP_V3
    except ValueError:
        return False


def _is_v2_swap(action) -> bool:
    if not isinstance(action, RouteSwap):
        return False
    try:
        return _pool_to_swap_protocol(action.pool.protocol) == SwapProtocol.UNISWAP_V2
    except ValueError:
        return False


def _is_pure_v3_linear(dag: RouteDAG) -> bool:
    """True iff every action is a V3 swap (no splits)."""
    return all(_is_v3_swap(a) for a in dag.actions)


def _addr_to_uint160(addr: Address) -> int:
    return int.from_bytes(addr, "big") & ADDRESS_MASK


def _v3_pool_word(edge: V3PoolEdge) -> int:
    """Pack V3 pool into uniswapV3SwapTo word: low 160 bits = pool, bit 255 = one-for-zero."""
    word = _addr_to_uint160(edge.pool_address)
    if not edge.is_token0_in:
        word |= ONE_FOR_ZERO_MASK
    return word


def encode_okx_v3(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    order_id: int = 0,
) -> EncodedTx:
    """Pure-V3 linear via ``DexRouter.uniswapV3SwapTo`` (fast path)."""
    if not _is_pure_v3_linear(dag):
        raise ValueError("encode_okx_v3: DAG must be a linear sequence of V3 swaps")
    if dag.token_in is None:
        raise ValueError("encode_okx_v3: DAG missing token_in")

    pools: list[int] = []
    for action in dag.actions:
        assert isinstance(action, RouteSwap)
        assert isinstance(action.pool, V3PoolEdge)
        pools.append(_v3_pool_word(action.pool))

    # receiver: low 160 bits address, upper 96 bits orderId
    receiver_word = _addr_to_uint160(recipient) | (order_id << 160)
    data = OKX_DEX_ROUTER.fns.uniswapV3SwapTo(
        receiver_word,
        amount_in,
        min_amount_out,
        pools,
    ).data
    return EncodedTx(to=router_address, data=bytes(data), value=0)


def _v3_extra_data(token_in: Address, token_out: Address, fee_bps: int) -> bytes:
    """``abi.encode(uint160 sqrtX96=0, abi.encode(fromToken, toToken, fee))``."""
    inner = abi_encode(
        ["address", "address", "uint24"],
        [bytes(token_in), bytes(token_out), fee_bps * 100],
    )
    return abi_encode(["uint160", "bytes"], [0, inner])


def _build_base_request(
    *, token_in: Address, token_out: Address, amount_in: int, min_amount_out: int, deadline: int
) -> BaseRequest:
    return BaseRequest(
        fromToken=_addr_to_uint160(token_in),
        toToken=bytes(token_out),
        fromTokenAmount=amount_in,
        minReturnAmount=min_amount_out,
        deadLine=deadline,
    )


def _v3_hop_router_path(*, swap: RouteSwap, v3_adapter: bytes) -> RouterPath:
    """One V3 hop, single adapter @ 100% weight. Used in N-batch layout."""
    assert isinstance(swap.pool, V3PoolEdge)
    return RouterPath(
        mixAdapters=[v3_adapter],
        assetTo=[v3_adapter],
        rawData=[_addr_to_uint160(swap.pool.pool_address) | (10_000 << WEIGHT_SHIFT)],
        extraData=[_v3_extra_data(swap.pool.token_in.address, swap.pool.token_out.address, swap.pool.fee_bps)],
        fromToken=_addr_to_uint160(swap.pool.token_in.address),
    )


class _SmartLeg(NamedTuple):
    """One row of a smart-swap ``RouterPath``: which adapter, where DexRouter
    pre-funds, packed pool/weight word, and adapter-specific extraData."""

    adapter: bytes
    asset_to: bytes
    raw_data: int
    extra_data: bytes


def _encode_smart_leg(
    *,
    swap: RouteSwap,
    weight_bps: int,
    token_in: Address,
    token_out: Address,
    v3_adapter: bytes,
    v2_adapter: bytes | None,
) -> _SmartLeg:
    """One smart-swap leg.

    - V3: DexRouter pre-funds the adapter; extraData carries (from, to, fee).
    - V2: DexRouter pre-funds the pool (assetTo = pool); reverse bit
      dispatches sellBase vs sellQuote; extraData is empty.
    """
    pool_word = _addr_to_uint160(swap.pool.pool_address) | (weight_bps << WEIGHT_SHIFT)
    if _is_v3_swap(swap):
        extra = _v3_extra_data(token_in, token_out, swap.pool.fee_bps)
        return _SmartLeg(v3_adapter, v3_adapter, pool_word, extra)
    if _is_v2_swap(swap):
        if v2_adapter is None:
            raise ValueError("encode_okx: V2 leg requires v2_adapter_address")
        assert isinstance(swap.pool, PoolEdge)
        is_token0_in = swap.pool.extra.get("is_token0_in")
        if is_token0_in is None:
            raise ValueError("encode_okx: V2 pool edge missing extra['is_token0_in']")
        if not is_token0_in:
            pool_word |= REVERSE_MASK  # sellQuote dispatch
        return _SmartLeg(v2_adapter, bytes(swap.pool.pool_address), pool_word, b"")
    raise ValueError(f"encode_okx: unsupported leg protocol {swap.pool.protocol}")


def _encode_okx_smart_parallel(
    *,
    dag: RouteDAG,
    split: RouteSplit,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    v3_adapter_address: Address,
    v2_adapter_address: Address | None,
    deadline: int,
    order_id: int,
) -> EncodedTx:
    """1-batch parallel-adapters layout (single-hop legs only) — cheapest
    OKX smart encoding for that shape. Mixed V2 + V3 legs supported."""
    token_in_addr = dag.token_in.address
    token_out_addr = split.token_out.address
    v3_adapter = bytes(v3_adapter_address)
    v2_adapter = bytes(v2_adapter_address) if v2_adapter_address is not None else None

    legs: list[_SmartLeg] = []
    for leg in split.legs:
        swap = leg.actions[0]
        assert isinstance(swap, RouteSwap)
        legs.append(
            _encode_smart_leg(
                swap=swap,
                weight_bps=leg.fraction_bps,
                token_in=token_in_addr,
                token_out=token_out_addr,
                v3_adapter=v3_adapter,
                v2_adapter=v2_adapter,
            )
        )
    router_path = RouterPath(
        mixAdapters=[L.adapter for L in legs],
        assetTo=[L.asset_to for L in legs],
        rawData=[L.raw_data for L in legs],
        extraData=[L.extra_data for L in legs],
        fromToken=_addr_to_uint160(token_in_addr),
    )
    data = OKX_DEX_ROUTER.fns.smartSwapTo(
        order_id,
        bytes(recipient),
        _build_base_request(
            token_in=token_in_addr,
            token_out=token_out_addr,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            deadline=deadline,
        ),
        [amount_in],
        [[router_path]],
        [],
    ).data
    return EncodedTx(to=router_address, data=bytes(data), value=0)


def _encode_okx_smart_batched(
    *,
    dag: RouteDAG,
    split: RouteSplit,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    v3_adapter_address: Address,
    deadline: int,
    order_id: int,
) -> EncodedTx:
    """N-batch layout — one batch per leg, hops as the batch's RouterPath[].
    Required when any leg is multi-hop."""
    v3_adapter = bytes(v3_adapter_address)
    batches_amount = [amount_in * leg.fraction_bps // 10_000 for leg in split.legs]
    batches = [
        [_v3_hop_router_path(swap=action, v3_adapter=v3_adapter) for action in leg.actions] for leg in split.legs
    ]
    data = OKX_DEX_ROUTER.fns.smartSwapTo(
        order_id,
        bytes(recipient),
        _build_base_request(
            token_in=dag.token_in.address,
            token_out=split.token_out.address,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            deadline=deadline,
        ),
        batches_amount,
        batches,
        [],
    ).data
    return EncodedTx(to=router_address, data=bytes(data), value=0)


def _dag_raw_data(*, pool: Address, weight_bps: int, input_idx: int, output_idx: int) -> int:
    """``rawData`` for one DagRouter edge: pool | weight | output_idx | input_idx."""
    return (
        _addr_to_uint160(pool)
        | (weight_bps << WEIGHT_SHIFT)
        | ((output_idx & 0xFF) << OUTPUT_INDEX_SHIFT)
        | ((input_idx & 0xFF) << INPUT_INDEX_SHIFT)
    )


def _v3_dag_node(
    *,
    swaps: list[tuple[V3PoolEdge, int]],
    input_idx: int,
    output_idx: int,
    from_token: Address,
    to_token: Address,
    v3_adapter: bytes,
) -> RouterPath:
    """Build one DagRouter node from V3 ``(pool, weight_bps)`` edges that all
    consume from ``input_idx`` and produce into ``output_idx``."""
    n = len(swaps)
    return RouterPath(
        mixAdapters=[v3_adapter] * n,
        assetTo=[v3_adapter] * n,
        rawData=[
            _dag_raw_data(pool=p.pool_address, weight_bps=w, input_idx=input_idx, output_idx=output_idx)
            for p, w in swaps
        ],
        extraData=[_v3_extra_data(from_token, to_token, p.fee_bps) for p, _ in swaps],
        fromToken=_addr_to_uint160(from_token),
    )


def _encode_okx_dag_fan_in(
    *,
    dag: RouteDAG,
    split: RouteSplit,
    downstream: RouteSwap,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    v3_adapter_address: Address,
    deadline: int,
    order_id: int,
) -> EncodedTx:
    """Fan-in via ``dagSwapTo`` — node 0 parallels the split into the merge
    token (multiple edges → output_idx=1), node 1 swaps merge→final."""
    assert dag.token_in is not None  # checked by encode_okx dispatcher
    assert isinstance(downstream.pool, V3PoolEdge)  # checked by _is_v3_fan_in
    token_in = dag.token_in.address
    merge_token = split.token_out.address
    final_token = downstream.token_out.address
    v3_adapter = bytes(v3_adapter_address)

    leg_swaps: list[tuple[V3PoolEdge, int]] = []
    for leg in split.legs:
        action = leg.actions[0]
        assert isinstance(action, RouteSwap) and isinstance(action.pool, V3PoolEdge)
        leg_swaps.append((action.pool, leg.fraction_bps))

    node0 = _v3_dag_node(
        swaps=leg_swaps,
        input_idx=0,
        output_idx=1,
        from_token=token_in,
        to_token=merge_token,
        v3_adapter=v3_adapter,
    )
    node1 = _v3_dag_node(
        swaps=[(downstream.pool, 10_000)],
        input_idx=1,
        output_idx=2,  # == nodeNum → routes to receiver
        from_token=merge_token,
        to_token=final_token,
        v3_adapter=v3_adapter,
    )

    data = OKX_DEX_ROUTER.fns.dagSwapTo(
        order_id,
        bytes(recipient),
        _build_base_request(
            token_in=token_in,
            token_out=final_token,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            deadline=deadline,
        ),
        [node0, node1],
    ).data
    return EncodedTx(to=router_address, data=bytes(data), value=0)


def _is_v3_fan_in(dag: RouteDAG) -> tuple[RouteSplit, RouteSwap] | None:
    """``[RouteSplit(all V3 single-hop), RouteSwap(V3)]`` — DagRouter 2-node fan-in."""
    if len(dag.actions) != 2:
        return None
    split, downstream = dag.actions
    if (
        isinstance(split, RouteSplit)
        and isinstance(downstream, RouteSwap)
        and _is_v3_swap(downstream)
        and all(len(leg.actions) == 1 and _is_v3_swap(leg.actions[0]) for leg in split.legs)
    ):
        return split, downstream
    return None


def encode_okx(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    v3_adapter_address: Address | None = None,
    v2_adapter_address: Address | None = None,
    deadline: int,
    order_id: int = 0,
) -> EncodedTx:
    """Pick the best OKX entry for the DAG: uniswapV3SwapTo for pure-V3
    linear; smartSwapTo (1-batch parallel, V2 + V3 mixed; or N-batch for
    multi-hop legs); dagSwapTo for ``[split, downstream swap]`` fan-in."""
    if _is_pure_v3_linear(dag):
        return encode_okx_v3(
            dag,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            recipient=recipient,
            router_address=router_address,
            order_id=order_id,
        )

    if dag.token_in is None:
        raise ValueError("encode_okx: DAG missing token_in")

    fan_in = _is_v3_fan_in(dag)
    if fan_in is not None:
        if v3_adapter_address is None:
            raise ValueError("encode_okx: dag path requires v3_adapter_address")
        split, downstream = fan_in
        return _encode_okx_dag_fan_in(
            dag=dag,
            split=split,
            downstream=downstream,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            recipient=recipient,
            router_address=router_address,
            v3_adapter_address=v3_adapter_address,
            deadline=deadline,
            order_id=order_id,
        )

    if len(dag.actions) != 1 or not isinstance(dag.actions[0], RouteSplit):
        raise ValueError("encode_okx: smart path expects one top-level RouteSplit")
    split = dag.actions[0]
    for leg in split.legs:
        if not leg.actions:
            raise ValueError("encode_okx: leg must contain at least one action")
        if not all(_is_v3_swap(a) or _is_v2_swap(a) for a in leg.actions):
            raise ValueError("encode_okx: every leg's action must be a Uniswap V2 or V3 swap")
    if v3_adapter_address is None:
        raise ValueError("encode_okx: smart-swap path requires v3_adapter_address")

    all_single_hop = all(len(leg.actions) == 1 for leg in split.legs)
    if not all_single_hop:
        if any(_is_v2_swap(a) for leg in split.legs for a in leg.actions):
            raise ValueError("encode_okx: V2 multi-hop legs not supported yet")
        return _encode_okx_smart_batched(
            dag=dag,
            split=split,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            recipient=recipient,
            router_address=router_address,
            v3_adapter_address=v3_adapter_address,
            deadline=deadline,
            order_id=order_id,
        )
    return _encode_okx_smart_parallel(
        dag=dag,
        split=split,
        amount_in=amount_in,
        min_amount_out=min_amount_out,
        recipient=recipient,
        router_address=router_address,
        v3_adapter_address=v3_adapter_address,
        v2_adapter_address=v2_adapter_address,
        deadline=deadline,
        order_id=order_id,
    )


# Back-compat alias.
encode_okx_smart = encode_okx


def encode_okx_swap_then_aave_supply(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    v3_adapter_address: Address,
    aave_adapter_address: Address,
    aave_pool: Address,
    aave_atoken: Address,
    deadline: int,
    order_id: int = 0,
) -> EncodedTx:
    """OKX smart-swap V3 swap → Aave supply (2 sequential hops in 1 batch).

    DexRouter routes the V3 pool output straight into the Aave adapter
    (next-hop-assetTo optimization); the Aave adapter then calls
    ``pool.supply(token, amount, recipient, 0)`` which mints aTokens to
    the user. ``aave_atoken`` is the ``BaseRequest.toToken`` so DexRouter's
    end-of-swap balance check measures the aToken delta.
    """
    if len(dag.actions) != 1 or not _is_v3_swap(dag.actions[0]):
        raise ValueError("encode_okx_swap_then_aave_supply: expect a single V3 swap")
    if dag.token_in is None:
        raise ValueError("encode_okx_swap_then_aave_supply: DAG missing token_in")
    swap = dag.actions[0]
    assert isinstance(swap, RouteSwap) and isinstance(swap.pool, V3PoolEdge)
    middle = swap.token_out.address
    aave_adapter = bytes(aave_adapter_address)

    hop_v3 = _v3_hop_router_path(swap=swap, v3_adapter=bytes(v3_adapter_address))
    # Adapter ignores poolAddress (uses immutable AAVEV3_POOL); the Aave pool
    # itself goes in the low 160 bits for semantic clarity.
    hop_aave = RouterPath(
        mixAdapters=[aave_adapter],
        assetTo=[aave_adapter],
        rawData=[_addr_to_uint160(aave_pool) | (10_000 << WEIGHT_SHIFT)],
        extraData=[abi_encode(["address", "address", "bool"], [bytes(middle), bytes(middle), True])],
        fromToken=_addr_to_uint160(middle),
    )

    data = OKX_DEX_ROUTER.fns.smartSwapTo(
        order_id,
        bytes(recipient),
        _build_base_request(
            token_in=dag.token_in.address,
            token_out=aave_atoken,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
            deadline=deadline,
        ),
        [amount_in],
        [[hop_v3, hop_aave]],
        [],
    ).data
    return EncodedTx(to=router_address, data=bytes(data), value=0)


# ---------------------------------------------------------------------------
# Uniswap baseline
# ---------------------------------------------------------------------------


def _v3_path_bytes(token_in: Address, actions) -> bytes:
    """``tokenIn || fee || tokenOut[1] || fee || tokenOut[2] || ...``"""
    chunks: list[bytes] = [bytes(token_in)]
    for action in actions:
        assert isinstance(action, RouteSwap)
        assert isinstance(action.pool, V3PoolEdge)
        chunks.append((action.pool.fee_bps * 100).to_bytes(3, "big"))
        chunks.append(bytes(action.token_out.address))
    return b"".join(chunks)


def _exact_input_calldata(
    *,
    path: bytes,
    recipient: Address,
    deadline: int,
    amount_in: int,
    min_amount_out: int,
) -> bytes:
    params = ExactInputParams(
        path=path,
        recipient=recipient,
        deadline=deadline,
        amountIn=amount_in,
        amountOutMinimum=min_amount_out,
    )
    return bytes(UNISWAP_V3_ROUTER.fns.exactInput(params).data)


def _exact_input_single_calldata(
    *,
    token_in: Address,
    token_out: Address,
    fee_bps: int,
    recipient: Address,
    deadline: int,
    amount_in: int,
    min_amount_out: int,
) -> bytes:
    params = ExactInputSingleParams(
        tokenIn=token_in,
        tokenOut=token_out,
        fee=fee_bps * 100,
        recipient=recipient,
        deadline=deadline,
        amountIn=amount_in,
        amountOutMinimum=min_amount_out,
        sqrtPriceLimitX96=0,
    )
    return bytes(UNISWAP_V3_ROUTER.fns.exactInputSingle(params).data)


def encode_uniswap(
    dag: RouteDAG,
    *,
    amount_in: int,
    min_amount_out: int,
    recipient: Address,
    router_address: Address,
    deadline: int,
) -> EncodedTx:
    """Linear → exactInput[Single]; split → multicall of per-leg exactInput[Single]."""
    if dag.token_in is None:
        raise ValueError("encode_uniswap: DAG missing token_in")

    if len(dag.actions) == 1 and isinstance(dag.actions[0], RouteSplit):
        split = dag.actions[0]
        for leg in split.legs:
            if not leg.actions or not all(_is_v3_swap(a) for a in leg.actions):
                raise ValueError("encode_uniswap split: every leg's actions must be V3 swaps")
        # SwapRouter02.multicall has no whole-route min-out — per-leg
        # amountOutMinimum doesn't compose into "sum ≥ X". Refuse rather than
        # silently weaken to per-leg 0; callers should use encode_ssa.
        if min_amount_out > 0:
            raise ValueError(
                "encode_uniswap split: SwapRouter02.multicall has no whole-route "
                "min-out. Pass min_amount_out=0 or use encode_ssa."
            )
        leg_calls: list[bytes] = []
        for leg in split.legs:
            leg_amount = amount_in * leg.fraction_bps // 10_000
            if len(leg.actions) == 1:
                swap = leg.actions[0]
                assert isinstance(swap, RouteSwap)
                assert isinstance(swap.pool, V3PoolEdge)
                leg_calls.append(
                    _exact_input_single_calldata(
                        token_in=dag.token_in.address,
                        token_out=swap.token_out.address,
                        fee_bps=swap.pool.fee_bps,
                        recipient=recipient,
                        deadline=deadline,
                        amount_in=leg_amount,
                        min_amount_out=0,
                    )
                )
            else:
                path = _v3_path_bytes(dag.token_in.address, leg.actions)
                leg_calls.append(
                    _exact_input_calldata(
                        path=path,
                        recipient=recipient,
                        deadline=deadline,
                        amount_in=leg_amount,
                        min_amount_out=0,
                    )
                )
        data = _UNISWAP_V3_MULTICALL.fns.multicall(leg_calls).data
        return EncodedTx(to=router_address, data=bytes(data), value=0)

    if not _is_pure_v3_linear(dag):
        raise ValueError("encode_uniswap: baseline only handles V3 (linear or split)")

    if len(dag.actions) == 1:
        swap = dag.actions[0]
        assert isinstance(swap, RouteSwap)
        assert isinstance(swap.pool, V3PoolEdge)
        data = _exact_input_single_calldata(
            token_in=dag.token_in.address,
            token_out=swap.token_out.address,
            fee_bps=swap.pool.fee_bps,
            recipient=recipient,
            deadline=deadline,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
        )
    else:
        data = _exact_input_calldata(
            path=_v3_path_bytes(dag.token_in.address, dag.actions),
            recipient=recipient,
            deadline=deadline,
            amount_in=amount_in,
            min_amount_out=min_amount_out,
        )
    return EncodedTx(to=router_address, data=bytes(data), value=0)

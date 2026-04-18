"""RouteDAG → DeFiVM bytecode generation.

This module compiles :class:`pydefi.types.RouteDAG` action trees into DeFiVM
program fragments for execution and quote simulation.

Stack conventions
-----------------
All DAG actions communicate amounts through the EVM stack:

* action input: ``amount_in`` is at TOS before the action
* action output: ``amount_out`` is at TOS after the action

Split handling uses stack frames, so nested multi-way splits do not clobber one
another:

1. enter split with ``[... parent, total_in]``
2. initialise frame as ``[... parent, total_in, accum=0]``
3. each leg:
   - ``DUP2`` total_in
   - compute ``leg_amount = total_in * bps / 10_000``
   - execute leg actions recursively (returns ``leg_out`` at TOS)
   - accumulate via ``ADD`` (``accum += leg_out``)
4. exit with ``SWAP1`` + ``POP`` to drop ``total_in`` and keep merged output
"""

from __future__ import annotations

from collections.abc import Sequence

from pydefi.pathfinder.graph import V3PoolEdge
from pydefi.types import RouteAction, RouteDAG, RouteSplit, RouteSwap
from pydefi.vm.builder import Program
from pydefi.vm.program import add, assert_ge, assert_le, div, dup, dup2, mul, pop, push_u256, swap
from pydefi.vm.swap import (
    SwapHop,
    SwapProtocol,
    _build_v2_direct_swap_segment_on_stack,
    _build_v3_pool_swap_segment_on_stack,
)

_BPS_DENOMINATOR = 10_000
# Guard for `total_in * fraction_bps` before dividing by 10_000 in multi-leg splits.
_MAX_TOTAL_IN_FOR_SPLIT = ((1 << 256) - 1) // _BPS_DENOMINATOR


def build_execution_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    recipient: str,
    min_final_out: int = 0,
) -> Program:
    """Build an execution program from a :class:`RouteDAG`."""
    return _build_program_for_dag(
        dag,
        amount_in=amount_in,
        vm_address=vm_address,
        terminal_recipient=recipient,
        min_final_out=min_final_out,
    )


def build_quote_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    min_final_out: int = 0,
) -> Program:
    """Build a quote/simulation program from a :class:`RouteDAG`."""
    return _build_program_for_dag(
        dag,
        amount_in=amount_in,
        vm_address=vm_address,
        terminal_recipient=vm_address,
        min_final_out=min_final_out,
    )


def _build_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    terminal_recipient: str,
    min_final_out: int,
) -> Program:
    payload = dag.to_dict()
    actions = payload["actions"]
    if not actions:
        raise ValueError("build_program_for_dag: route DAG must contain at least one action")

    segments: list[Program] = [Program()._emit(push_u256(amount_in))]
    segments.extend(
        _build_dag_actions(
            actions,
            vm_address=vm_address,
            terminal_recipient=terminal_recipient,
        )
    )

    if min_final_out > 0:
        segments.append(
            Program()
            ._emit(dup())
            ._emit(push_u256(min_final_out))
            ._emit(swap())
            ._emit(assert_ge("slippage: out too low"))
        )

    return Program.compose(segments)


def _build_dag_actions(
    actions: Sequence[RouteAction],
    *,
    vm_address: str,
    terminal_recipient: str,
) -> list[Program]:
    segments: list[Program] = []
    for i, action in enumerate(actions):
        action_recipient = terminal_recipient if i == len(actions) - 1 else vm_address
        if isinstance(action, RouteSwap):
            segments.append(
                _build_route_swap_segment_on_stack(
                    action,
                    recipient=action_recipient,
                )
            )
            continue

        if isinstance(action, RouteSplit):
            segments.extend(
                _build_route_split_segment(
                    action,
                    vm_address=vm_address,
                    terminal_recipient=action_recipient,
                )
            )
            continue

        raise ValueError(f"build_program_for_dag: unsupported route action {type(action)!r}")

    return segments


def _build_route_split_segment(
    split: RouteSplit,
    *,
    vm_address: str,
    terminal_recipient: str,
) -> list[Program]:
    if len(split.legs) == 1:
        # Fast path: full-allocation single leg does not need split-frame
        # bookkeeping; emit leg actions directly.
        return _build_dag_actions(
            split.legs[0].actions,
            vm_address=vm_address,
            terminal_recipient=terminal_recipient,
        )

    # Runtime guard: for each leg we compute total_in * fraction_bps / 10_000.
    # Enforce `total_in <= floor((2**256-1)/10_000)` before any multiplication.
    segments: list[Program] = [
        Program()
        ._emit(dup())
        ._emit(push_u256(_MAX_TOTAL_IN_FOR_SPLIT))
        ._emit(swap())
        ._emit(assert_le("split: total_in overflow guard"))
        ._emit(push_u256(0))
    ]
    for leg in split.legs:
        segments.append(
            Program()
            ._emit(dup2())  # duplicate split frame total_in (2nd item in [.., total_in, accum])
            ._emit(push_u256(leg.fraction_bps))
            ._emit(mul())
            ._emit(push_u256(_BPS_DENOMINATOR))
            ._emit(swap())
            ._emit(div())
        )
        segments.extend(
            _build_dag_actions(
                leg.actions,
                vm_address=vm_address,
                terminal_recipient=terminal_recipient,
            )
        )
        segments.append(Program()._emit(add()))

    segments.append(Program()._emit(swap())._emit(pop()))
    return segments


def _build_route_swap_segment_on_stack(
    swap_action: RouteSwap,
    *,
    recipient: str,
) -> Program:
    hop = _swap_hop_from_route_swap(swap_action, recipient=recipient)
    if hop.protocol == SwapProtocol.UNISWAP_V3:
        return _build_v3_pool_swap_segment_on_stack(hop)
    return _build_v2_direct_swap_segment_on_stack(hop)


def _swap_hop_from_route_swap(swap_action: RouteSwap, *, recipient: str) -> SwapHop:
    pool = swap_action.pool
    protocol = _pool_to_swap_protocol(pool.protocol)
    if isinstance(pool, V3PoolEdge):
        zero_for_one = pool.is_token0_in
    else:
        is_token0_in = getattr(pool, "extra", {}).get("is_token0_in")
        if is_token0_in is None:
            raise ValueError("build_program_for_dag: non-V3 pool is missing extra['is_token0_in'] metadata")
        zero_for_one = bool(is_token0_in)
    return SwapHop(
        protocol=protocol,
        pool=pool.pool_address,
        token_in=pool.token_in.address,
        token_out=pool.token_out.address,
        fee=pool.fee_bps,
        amount_in=0,
        amount_out_min=0,
        recipient=recipient,
        zero_for_one=zero_for_one,
    )


def _pool_to_swap_protocol(protocol_name: str) -> SwapProtocol:
    name = protocol_name.lower()
    if "v3" in name:
        return SwapProtocol.UNISWAP_V3
    if "v2" in name:
        return SwapProtocol.UNISWAP_V2
    raise ValueError(f"build_program_for_dag: unsupported pool protocol {protocol_name!r}")

"""RouteDAG → DeFiVM bytecode generation.

This module compiles :class:`pydefi.types.RouteDAG` action trees into DeFiVM
program fragments for execution and quote simulation.

Stack/register conventions
--------------------------
The DAG compiler reuses the existing swap segment builders from
``pydefi.vm.swap`` (V2/V3 direct pool calls). Those segment builders use a
register-based amount convention, so DAG compilation follows the same ABI:

* ``amount_reg``: current amount entering/leaving each action.
* ``amount_out_reg``: scratch register for V2 output patching.
* ``total_in_reg``: split-local immutable total used for per-leg pro-rata math.
* ``accum_reg``: split-local accumulator for merged leg outputs.

Nested splits are handled recursively with the same register contract:

1. parent action leaves the active amount in ``amount_reg``
2. split snapshots this value into ``total_in_reg`` and zeroes ``accum_reg``
3. each leg computes ``leg_amount = total_in * fraction_bps / 10000`` into
   ``amount_reg``, executes leg actions recursively, and adds the leg output
   into ``accum_reg``
4. merged amount is moved back to ``amount_reg`` for downstream actions

This keeps split semantics deterministic for arbitrary multi-way nested DAGs.
"""

from __future__ import annotations

from pydefi.pathfinder.graph import V3PoolEdge
from pydefi.types import RouteAction, RouteDAG, RouteSplit, RouteSwap
from pydefi.vm.builder import Program
from pydefi.vm.program import add, assert_ge, div, load_reg, mul, push_u256, store_reg, swap
from pydefi.vm.swap import (
    _ACCUM_REG,
    _AMOUNT_OUT_REG,
    _AMOUNT_REG,
    _TOTAL_IN_REG,
    SwapHop,
    SwapProtocol,
    _build_v2_direct_swap_segment,
    _build_v3_pool_swap_segment,
)

_BPS_DENOMINATOR = 10_000


def build_execution_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    recipient: str,
    min_final_out: int = 0,
    amount_reg: int = _AMOUNT_REG,
    amount_out_reg: int = _AMOUNT_OUT_REG,
    accum_reg: int = _ACCUM_REG,
    total_in_reg: int = _TOTAL_IN_REG,
) -> Program:
    """Build an execution program from a :class:`RouteDAG`."""
    return _build_program_for_dag(
        dag,
        amount_in=amount_in,
        vm_address=vm_address,
        terminal_recipient=recipient,
        min_final_out=min_final_out,
        amount_reg=amount_reg,
        amount_out_reg=amount_out_reg,
        accum_reg=accum_reg,
        total_in_reg=total_in_reg,
    )


def build_quote_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    min_final_out: int = 0,
    amount_reg: int = _AMOUNT_REG,
    amount_out_reg: int = _AMOUNT_OUT_REG,
    accum_reg: int = _ACCUM_REG,
    total_in_reg: int = _TOTAL_IN_REG,
) -> Program:
    """Build a quote/simulation program from a :class:`RouteDAG`."""
    return _build_program_for_dag(
        dag,
        amount_in=amount_in,
        vm_address=vm_address,
        terminal_recipient=vm_address,
        min_final_out=min_final_out,
        amount_reg=amount_reg,
        amount_out_reg=amount_out_reg,
        accum_reg=accum_reg,
        total_in_reg=total_in_reg,
    )


def _build_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    vm_address: str,
    terminal_recipient: str,
    min_final_out: int,
    amount_reg: int,
    amount_out_reg: int,
    accum_reg: int,
    total_in_reg: int,
) -> Program:
    payload = dag.to_dict()
    actions = payload["actions"]
    if not actions:
        raise ValueError("build_program_for_dag: route DAG must contain at least one action")

    segments: list[Program] = [Program()._emit(push_u256(amount_in))._emit(store_reg(amount_reg))]
    segments.extend(
        _build_dag_actions(
            actions,
            vm_address=vm_address,
            terminal_recipient=terminal_recipient,
            amount_reg=amount_reg,
            amount_out_reg=amount_out_reg,
            accum_reg=accum_reg,
            total_in_reg=total_in_reg,
        )
    )

    if min_final_out > 0:
        segments.append(
            Program()
            ._emit(push_u256(min_final_out))
            ._emit(load_reg(amount_reg))
            ._emit(assert_ge("slippage: out too low"))
        )

    return Program.compose(segments)


def _build_dag_actions(
    actions: list[RouteAction],
    *,
    vm_address: str,
    terminal_recipient: str,
    amount_reg: int,
    amount_out_reg: int,
    accum_reg: int,
    total_in_reg: int,
) -> list[Program]:
    segments: list[Program] = []
    for i, action in enumerate(actions):
        action_recipient = terminal_recipient if i == len(actions) - 1 else vm_address
        if isinstance(action, RouteSwap):
            hop = _swap_hop_from_route_swap(action, recipient=action_recipient)
            if hop.protocol == SwapProtocol.UNISWAP_V3:
                segments.append(_build_v3_pool_swap_segment(hop, amount_reg=amount_reg))
            else:
                segments.append(
                    _build_v2_direct_swap_segment(hop, amount_reg=amount_reg, amount_out_reg=amount_out_reg)
                )
            continue

        if isinstance(action, RouteSplit):
            segments.extend(
                _build_route_split_segment(
                    action,
                    vm_address=vm_address,
                    terminal_recipient=action_recipient,
                    amount_reg=amount_reg,
                    amount_out_reg=amount_out_reg,
                    accum_reg=accum_reg,
                    total_in_reg=total_in_reg,
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
    amount_reg: int,
    amount_out_reg: int,
    accum_reg: int,
    total_in_reg: int,
) -> list[Program]:
    if len(split.legs) == 1 and split.legs[0].fraction_bps == _BPS_DENOMINATOR:
        # Fast path: full-allocation single leg does not need split accounting
        # registers; emit the leg actions directly.
        return _build_dag_actions(
            split.legs[0].actions,
            vm_address=vm_address,
            terminal_recipient=terminal_recipient,
            amount_reg=amount_reg,
            amount_out_reg=amount_out_reg,
            accum_reg=accum_reg,
            total_in_reg=total_in_reg,
        )

    segments: list[Program] = []
    segments.append(
        Program()
        ._emit(load_reg(amount_reg))
        ._emit(store_reg(total_in_reg))
        ._emit(push_u256(0))
        ._emit(store_reg(accum_reg))
    )

    for leg in split.legs:
        segments.append(
            Program()
            ._emit(load_reg(total_in_reg))
            ._emit(push_u256(leg.fraction_bps))
            ._emit(mul())
            ._emit(push_u256(_BPS_DENOMINATOR))
            ._emit(swap())
            ._emit(div())
            ._emit(store_reg(amount_reg))
        )
        segments.extend(
            _build_dag_actions(
                leg.actions,
                vm_address=vm_address,
                terminal_recipient=terminal_recipient,
                amount_reg=amount_reg,
                amount_out_reg=amount_out_reg,
                accum_reg=accum_reg,
                total_in_reg=total_in_reg,
            )
        )
        segments.append(
            Program()._emit(load_reg(amount_reg))._emit(load_reg(accum_reg))._emit(add())._emit(store_reg(accum_reg))
        )

    segments.append(Program()._emit(load_reg(accum_reg))._emit(store_reg(amount_reg)))
    return segments


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

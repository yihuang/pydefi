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

from collections.abc import Callable, Sequence

from pydefi.types import ZERO_ADDRESS, Address, RouteAction, RouteDAG, RouteSplit, RouteSwap, SwapProtocol
from pydefi.vm.builder import Program
from pydefi.vm.program import add, assert_ge, assert_le, div, dup, dup2, mul, pop, push_u256, return_tos, swap
from pydefi.vm.swap import (
    _build_route_swap_segment,
    _build_v2_quote_segment,
    _build_v3_quote_segment,
    _swap_hop_from_route_swap,
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
    min_final_out: int = 0,
    quoter_address: str | None = None,
) -> Program:
    """Build a quote/simulation program from a :class:`RouteDAG`.

    Produces a view-only DeFiVM program safe for ``eth_call`` — no tokens are
    transferred.  V2 hops compute ``amountOut`` from ``pair.getReserves()``
    on-stack; V3 hops call ``quoter.quoteExactInput``.  The program ends with
    ``return_tos`` so the caller reads the final ``amountOut`` from returndata.

    Args:
        dag: Route DAG to simulate.
        amount_in: Input amount for the first hop.
        min_final_out: If > 0, revert if ``amountOut < min_final_out``.
        quoter_address: Uniswap V3-compatible quoter address.  Required when
            the DAG contains V3 hops; ignored for V2-only routes.
    """
    payload = dag.to_dict()
    actions = payload["actions"]
    if not actions:
        raise ValueError("build_quote_program_for_dag: route DAG must contain at least one action")

    segments: list[Program] = [Program()._emit(push_u256(amount_in))]
    segments.extend(_build_dag_quote_actions(actions, quoter_address=quoter_address))

    if min_final_out > 0:
        segments.append(
            Program()
            ._emit(dup())
            ._emit(push_u256(min_final_out))
            ._emit(swap())
            ._emit(assert_ge("slippage: out too low"))
        )

    segments.append(Program()._emit(return_tos()))
    return Program.compose(segments)


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
                _build_route_swap_segment(
                    action,
                    recipient=Address(action_recipient),
                )
            )
            continue

        if isinstance(action, RouteSplit):
            segments.extend(
                _build_split_frame(
                    action,
                    lambda acts, r=action_recipient: _build_dag_actions(
                        acts, vm_address=vm_address, terminal_recipient=r
                    ),
                )
            )
            continue

        raise ValueError(f"build_program_for_dag: unsupported route action {type(action)!r}")

    return segments


def _build_split_frame(
    split: RouteSplit,
    build_leg_actions: Callable[[Sequence[RouteAction]], list[Program]],
) -> list[Program]:
    """Emit split-frame bytecode, delegating leg action building to a callback.

    Stack convention on entry/exit:
    - entry: ``[... , total_in]``
    - exit:  ``[... , total_out]``
    """
    if len(split.legs) == 1:
        # Fast path: full-allocation single leg does not need split-frame
        # bookkeeping; emit leg actions directly.
        return build_leg_actions(split.legs[0].actions)

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
        segments.extend(build_leg_actions(leg.actions))
        segments.append(Program()._emit(add()))

    segments.append(Program()._emit(swap())._emit(pop()))
    return segments


def _build_dag_quote_actions(
    actions: Sequence[RouteAction],
    *,
    quoter_address: str | None,
) -> list[Program]:
    segments: list[Program] = []
    for action in actions:
        if isinstance(action, RouteSwap):
            hop = _swap_hop_from_route_swap(action, recipient=ZERO_ADDRESS)
            if hop.protocol == SwapProtocol.UNISWAP_V3:
                if quoter_address is None:
                    raise ValueError("build_quote_program_for_dag: quoter_address required for V3 hops")
                segments.append(_build_v3_quote_segment(hop, quoter_address))
            else:
                segments.append(_build_v2_quote_segment(hop))
            continue

        if isinstance(action, RouteSplit):
            segments.extend(
                _build_split_frame(
                    action,
                    lambda acts, q=quoter_address: _build_dag_quote_actions(acts, quoter_address=q),
                )
            )
            continue

        raise ValueError(f"build_quote_program_for_dag: unsupported route action {type(action)!r}")
    return segments

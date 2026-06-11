"""RouteDAG → DeFiVM bytecode generation.

Walks a :class:`pydefi.types.RouteDAG` action tree and emits SSA IR into
:class:`pydefi.vm.Program` for execution and quote programs.

Each ``_build_*`` helper takes a ``Program`` plus ``amount_in: Operand`` and
returns ``amount_out: Operand`` — values flow through Python rather than the
implicit EVM stack, so multi-leg splits become plain Python ``for`` loops.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydefi.pathfinder.dag import RouteDAG
from pydefi.types import (
    ZERO_ADDRESS,
    Address,
    RouteAction,
    RouteBridge,
    RouteSplit,
    RouteSwap,
    SwapProtocol,
)
from pydefi.vm.context import Operand, Program
from pydefi.vm.eureka import approve_then_send_transfer
from pydefi.vm.swap import (
    _build_route_swap,
    _build_v2_quote,
    _build_v3_quote,
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
    if not dag.actions:
        raise ValueError("build_execution_program_for_dag: route DAG must contain at least one action")

    prog = Program()
    final_out = _build_dag_actions(
        prog,
        prog.const(amount_in),
        dag.actions,
        vm_address=vm_address,
        terminal_recipient=recipient,
    )
    if min_final_out > 0:
        prog.assert_ge(final_out, min_final_out, "slippage: out too low")
    prog.builder.stop()
    return prog


def build_quote_program_for_dag(
    dag: RouteDAG,
    *,
    amount_in: int,
    quoter_address: str | None = None,
) -> Program:
    """Build a quote/simulation program from a :class:`RouteDAG`.

    Returns ``amountOut`` via ``RETURN(0, 32)`` so callers reading from
    ``eth_call`` returndata see the final output amount.
    """
    if not dag.actions:
        raise ValueError("build_quote_program_for_dag: route DAG must contain at least one action")

    prog = Program()
    final_out = _build_dag_quote_actions(
        prog,
        prog.const(amount_in),
        dag.actions,
        quoter_address=quoter_address,
    )
    prog.return_word(final_out)
    return prog


# ---------------------------------------------------------------------------
# Internal SSA action walkers
# ---------------------------------------------------------------------------


def _build_dag_actions(
    prog: Program,
    amount_in: Operand,
    actions: Sequence[RouteAction],
    *,
    vm_address: str,
    terminal_recipient: str,
) -> Operand:
    """Walk *actions* sequentially, threading the running amount through each."""
    current = amount_in
    last_index = len(actions) - 1
    for i, action in enumerate(actions):
        action_recipient = terminal_recipient if i == last_index else vm_address
        if isinstance(action, RouteSwap):
            current = _build_route_swap(prog, current, action, Address(action_recipient))
        elif isinstance(action, RouteSplit):
            current = _build_split(
                prog,
                current,
                action,
                lambda p, amt, acts, r=action_recipient: _build_dag_actions(
                    p, amt, acts, vm_address=vm_address, terminal_recipient=r
                ),
            )
        elif isinstance(action, RouteBridge):
            current = _build_route_bridge(prog, current, action)
        else:
            raise ValueError(f"build_program_for_dag: unsupported route action {type(action)!r}")
    return current


def _build_route_bridge(
    prog: Program,
    amount_in: Operand,
    action: RouteBridge,
) -> Operand:
    """Lower ``RouteBridge`` to ``approve`` + ``ICS20Transfer.sendTransfer``.

    ICS-20 preserves amount on the source side (settlement is asynchronous
    via the relayer), so ``amount_in`` passes through unchanged as the
    ``"out"`` operand — downstream ``min_final_out`` checks then apply to
    the bridged amount.
    """
    success = approve_then_send_transfer(
        prog,
        transfer_addr=action.transfer_addr,
        denom=action.denom,
        amount=amount_in,
        receiver=action.receiver,
        source_client=action.source_client,
        dest_port=action.dest_port,
        memo=action.memo,
        timeout_seconds=action.timeout_seconds,
        timeout_timestamp=action.timeout_timestamp,
    )
    prog.assert_(success, "bridge failed")
    return amount_in


def _build_dag_quote_actions(
    prog: Program,
    amount_in: Operand,
    actions: Sequence[RouteAction],
    *,
    quoter_address: str | None,
) -> Operand:
    current = amount_in
    for action in actions:
        if isinstance(action, RouteSwap):
            hop = _swap_hop_from_route_swap(action, recipient=ZERO_ADDRESS)
            if hop.protocol == SwapProtocol.UNISWAP_V3:
                if quoter_address is None:
                    raise ValueError("build_quote_program_for_dag: quoter_address required for V3 hops")
                current = _build_v3_quote(prog, current, hop, quoter_address)
            else:
                current = _build_v2_quote(prog, current, hop)
        elif isinstance(action, RouteSplit):
            current = _build_split(
                prog,
                current,
                action,
                lambda p, amt, acts: _build_dag_quote_actions(p, amt, acts, quoter_address=quoter_address),
            )
        elif isinstance(action, RouteBridge):
            # ICS-20 is amount-preserving on the source side; the quote pass
            # never touches the chain, so just thread the running amount.
            pass
        else:
            raise ValueError(f"build_quote_program_for_dag: unsupported route action {type(action)!r}")
    return current


def _build_split(
    prog: Program,
    total_in: Operand,
    split: RouteSplit,
    build_leg: Callable[[Program, Operand, Sequence[RouteAction]], Operand],
) -> Operand:
    """Compute a split: route fractions of *total_in* through each leg, sum the outputs.

    *build_leg* is a callback ``(prog, amount_in, actions) -> amount_out`` that
    emits IR for one leg's action list into *prog* and returns the leg's
    output amount as a :class:`Operand`.
    """
    if len(split.legs) == 1:
        # Fast path: single leg gets the full input.
        return build_leg(prog, total_in, split.legs[0].actions)

    # Runtime guard: each leg computes total_in * fraction_bps, which must fit in uint256.
    prog.assert_le(total_in, _MAX_TOTAL_IN_FOR_SPLIT, "split: total_in overflow guard")

    accum: Operand = prog.const(0)
    for leg in split.legs:
        leg_amount = prog.div(prog.mul(total_in, leg.fraction_bps), _BPS_DENOMINATOR)
        leg_out = build_leg(prog, leg_amount, leg.actions)
        accum = prog.add(accum, leg_out)
    return accum

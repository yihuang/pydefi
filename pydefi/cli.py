"""
pydefi CLI – Unified DeFi command-line interface.

Usage::

    pydefi plan  --src-chain <chain> --src-token <token> \\
                 --dst-chain <chain> --dst-token <token> \\
                 --amount <amount>

    pydefi swap  --chain <chain> --token-in <token> --token-out <token> \\
                 --amount <amount>

    pydefi bridge --src-chain <chain> --dst-chain <chain> \\
                  --token <token> --amount <amount>

    pydefi execute --plan <plan.json> [--dry-run]

Chain names (case-insensitive) or numeric IDs are accepted wherever a chain is
required.  See ``pydefi plan --help`` for a list of recognised chain names.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from pydefi.plan import BridgeAction, ExecutionPlan, SwapAction
from pydefi.planner import _CHAIN_NAMES, build_plan
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# Chain name → ID resolution
# ---------------------------------------------------------------------------

_NAME_TO_CHAIN_ID: dict[str, int] = {
    name.lower(): cid for cid, name in _CHAIN_NAMES.items()
}
# Extra aliases
_NAME_TO_CHAIN_ID.update(
    {
        "eth": ChainId.ETHEREUM,
        "mainnet": ChainId.ETHEREUM,
        "op": ChainId.OPTIMISM,
        "bnb": ChainId.BSC,
        "poly": ChainId.POLYGON,
        "arb": ChainId.ARBITRUM,
        "avax": ChainId.AVALANCHE,
        "zk": ChainId.ZKSYNC,
    }
)


def _resolve_chain(value: str) -> int:
    """Return a chain ID from a name (case-insensitive) or numeric string."""
    try:
        return int(value)
    except ValueError:
        key = value.lower()
        if key in _NAME_TO_CHAIN_ID:
            return _NAME_TO_CHAIN_ID[key]
        known = ", ".join(sorted(_NAME_TO_CHAIN_ID))
        raise argparse.ArgumentTypeError(
            f"Unknown chain {value!r}.  Known names: {known}"
        ) from None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_plan(plan: ExecutionPlan) -> None:
    print(plan.describe())
    print()
    print("JSON representation:")
    print(plan.to_json())


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(
        src_chain_id=args.src_chain,
        src_token=args.src_token,
        dst_chain_id=args.dst_chain,
        dst_token=args.dst_token,
        amount=args.amount,
        bridge_protocol=args.bridge_protocol,
        swap_protocol=args.swap_protocol,
    )
    _print_plan(plan)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(plan.to_json())
        print(f"\nPlan saved to {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: swap
# ---------------------------------------------------------------------------


def _cmd_swap(args: argparse.Namespace) -> int:
    action = SwapAction(
        chain_id=args.chain,
        token_in=args.token_in.upper(),
        token_out=args.token_out.upper(),
        amount_in=args.amount,
        protocol=args.protocol,
    )
    plan = ExecutionPlan(
        description=f"Swap {args.amount} {action.token_in} → {action.token_out}",
        actions=[action],
    )
    _print_plan(plan)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(plan.to_json())
        print(f"\nPlan saved to {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: bridge
# ---------------------------------------------------------------------------


def _cmd_bridge(args: argparse.Namespace) -> int:
    action = BridgeAction(
        src_chain_id=args.src_chain,
        dst_chain_id=args.dst_chain,
        token=args.token.upper(),
        amount_in=args.amount,
        protocol=args.protocol,
    )
    plan = ExecutionPlan(
        description=(
            f"Bridge {args.amount} {action.token} "
            f"from chain {args.src_chain} to chain {args.dst_chain}"
        ),
        actions=[action],
    )
    _print_plan(plan)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(plan.to_json())
        print(f"\nPlan saved to {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: execute
# ---------------------------------------------------------------------------


def _cmd_execute(args: argparse.Namespace) -> int:
    with open(args.plan) as fh:
        plan = ExecutionPlan.from_json(fh.read())

    print(plan.describe())
    print()

    if args.dry_run:
        print("[dry-run] Execution skipped.  Steps that would be executed:")
        for i, action in enumerate(plan.actions, 1):
            print(f"  {i}. {action.describe()}")
        return 0

    print("Executing plan...")
    for i, action in enumerate(plan.actions, 1):
        print(f"  Step {i}/{len(plan.actions)}: {action.describe()}")
        _execute_action(action, args)
        print(f"  Step {i} done.")

    print("\nAll steps completed successfully.")
    return 0


def _execute_action(action: SwapAction | BridgeAction, args: argparse.Namespace) -> None:
    """Execute a single action.

    In this prototype the execution layer prints the action details and
    available context.  Actual on-chain submission requires a wallet / signer
    (``--wallet`` and ``--rpc``) and resolved token addresses.
    """
    if isinstance(action, SwapAction):
        _execute_swap(action, args)
    elif isinstance(action, BridgeAction):
        _execute_bridge(action, args)


def _execute_swap(action: SwapAction, args: argparse.Namespace) -> None:
    """Build and optionally submit a swap transaction."""
    rpc = getattr(args, "rpc", None)

    if not rpc:
        print("    [info] No --rpc provided; skipping on-chain swap simulation.")
        print(f"    [info] Action: {action.describe()}")
        return

    # Token addresses must be resolved before execution; the CLI action stores
    # only symbols.  Execution with real addresses requires a token registry or
    # explicit --token-address flags which are outside the scope of this prototype.
    print("    [info] On-chain swap execution requires resolved token addresses.")
    print(f"    [info] Action: {action.describe()}")
    print(f"    [info] RPC endpoint: {rpc}")


def _execute_bridge(action: BridgeAction, args: argparse.Namespace) -> None:
    """Build and optionally submit a bridge transaction."""
    rpc = getattr(args, "rpc", None)
    if not rpc:
        print("    [info] No --rpc provided; skipping on-chain bridge simulation.")
        print(f"    [info] Action: {action.describe()}")
        return

    print(f"    [info] Bridge execution via {action.protocol} — provide --rpc to enable.")


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pydefi",
        description=(
            "pydefi – Unified DeFi CLI\n\n"
            "Chain actions: swap, bridge, and compose them into cross-chain\n"
            "execution plans that AI agents (or humans) can review and execute."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ------------------------------------------------------------------ plan
    plan_p = subparsers.add_parser(
        "plan",
        help="Generate a cross-chain execution plan",
        description=(
            "Generate a step-by-step execution plan to convert a token on one\n"
            "chain to a token on another chain (or on the same chain).\n\n"
            "Recognised chain names: "
            + ", ".join(sorted({v for v in _NAME_TO_CHAIN_ID}))
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan_p.add_argument("--src-chain", required=True, type=_resolve_chain, metavar="CHAIN",
                        help="Source chain name or ID (e.g. ethereum, base, 1)")
    plan_p.add_argument("--src-token", required=True, metavar="TOKEN",
                        help="Source token symbol or address (e.g. USDC)")
    plan_p.add_argument("--dst-chain", required=True, type=_resolve_chain, metavar="CHAIN",
                        help="Destination chain name or ID")
    plan_p.add_argument("--dst-token", required=True, metavar="TOKEN",
                        help="Destination token symbol or address")
    plan_p.add_argument("--amount", required=True, metavar="AMOUNT",
                        help="Input amount in human-readable form (e.g. 100 or 0.5)")
    plan_p.add_argument("--bridge-protocol", default="auto", metavar="PROTOCOL",
                        help="Bridge protocol to use (default: auto)")
    plan_p.add_argument("--swap-protocol", default="auto", metavar="PROTOCOL",
                        help="Swap protocol/aggregator to use (default: auto)")
    plan_p.add_argument("-o", "--output", metavar="FILE",
                        help="Save the plan as JSON to FILE")
    plan_p.set_defaults(func=_cmd_plan)

    # ------------------------------------------------------------------ swap
    swap_p = subparsers.add_parser(
        "swap",
        help="Create a single-chain swap plan",
        description="Create a plan for a single-chain token swap.",
    )
    swap_p.add_argument("--chain", required=True, type=_resolve_chain, metavar="CHAIN",
                        help="Chain name or ID")
    swap_p.add_argument("--token-in", required=True, metavar="TOKEN",
                        help="Input token symbol or address")
    swap_p.add_argument("--token-out", required=True, metavar="TOKEN",
                        help="Output token symbol or address")
    swap_p.add_argument("--amount", required=True, metavar="AMOUNT",
                        help="Input amount in human-readable form")
    swap_p.add_argument("--protocol", default="auto", metavar="PROTOCOL",
                        help="DEX/aggregator protocol (default: auto)")
    swap_p.add_argument("-o", "--output", metavar="FILE",
                        help="Save the plan as JSON to FILE")
    swap_p.set_defaults(func=_cmd_swap)

    # --------------------------------------------------------------- bridge
    bridge_p = subparsers.add_parser(
        "bridge",
        help="Create a cross-chain bridge plan",
        description="Create a plan for bridging tokens across chains.",
    )
    bridge_p.add_argument("--src-chain", required=True, type=_resolve_chain, metavar="CHAIN",
                          help="Source chain name or ID")
    bridge_p.add_argument("--dst-chain", required=True, type=_resolve_chain, metavar="CHAIN",
                          help="Destination chain name or ID")
    bridge_p.add_argument("--token", required=True, metavar="TOKEN",
                          help="Token symbol or address")
    bridge_p.add_argument("--amount", required=True, metavar="AMOUNT",
                          help="Amount in human-readable form")
    bridge_p.add_argument("--protocol", default="auto", metavar="PROTOCOL",
                          help="Bridge protocol (default: auto)")
    bridge_p.add_argument("-o", "--output", metavar="FILE",
                          help="Save the plan as JSON to FILE")
    bridge_p.set_defaults(func=_cmd_bridge)

    # --------------------------------------------------------------- execute
    execute_p = subparsers.add_parser(
        "execute",
        help="Execute a previously generated plan",
        description=(
            "Execute every step in a plan JSON file produced by the 'plan',\n"
            "'swap', or 'bridge' subcommands.\n\n"
            "Pass --dry-run to print each step without submitting transactions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    execute_p.add_argument("--plan", required=True, metavar="FILE",
                           help="Path to the plan JSON file")
    execute_p.add_argument("--rpc", metavar="URL",
                           help="JSON-RPC endpoint URL for on-chain calls")
    execute_p.add_argument("--wallet", metavar="ADDRESS",
                           help="Wallet address that will sign transactions")
    execute_p.add_argument("--dry-run", action="store_true",
                           help="Print steps without submitting transactions")
    execute_p.set_defaults(func=_cmd_execute)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

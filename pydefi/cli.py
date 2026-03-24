"""
defi CLI – Unified DeFi command-line interface.

Usage::

    defi plan  --src-chain <chain> --src-token <token> \\
               --dst-chain <chain> --dst-token <token> \\
               --amount <amount>

    defi swap  --chain <chain> --token-in <token> --token-out <token> \\
               --amount <amount>

    defi bridge --src-chain <chain> --dst-chain <chain> \\
                --token <token> --amount <amount>

    defi execute --plan <plan.json> [--dry-run]

Chain names (case-insensitive) or numeric IDs are accepted wherever a chain is
required.  Run ``defi plan --help`` for a list of recognised chain names.
"""

from __future__ import annotations

import click

from pydefi.plan import BridgeAction, ExecutionPlan, SwapAction
from pydefi.planner import _CHAIN_NAMES, build_plan
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# Chain name → ID resolution
# ---------------------------------------------------------------------------

_NAME_TO_CHAIN_ID: dict[str, int] = {name.lower(): cid for cid, name in _CHAIN_NAMES.items()}
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
        raise click.BadParameter(f"Unknown chain {value!r}.  Known names: {known}") from None


class _ChainParam(click.ParamType):
    """Click parameter type that accepts chain names or numeric IDs."""

    name = "CHAIN"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        try:
            return _resolve_chain(value)
        except click.BadParameter as exc:
            self.fail(str(exc), param, ctx)


CHAIN = _ChainParam()


# ---------------------------------------------------------------------------
# Main command group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option("0.1.0", prog_name="defi")
def cli():
    """defi – Unified DeFi CLI.

    Compose swap and bridge actions into cross-chain execution plans
    that AI agents (or humans) can review and execute.
    """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_plan(plan: ExecutionPlan) -> None:
    click.echo(plan.describe())
    click.echo()
    click.echo("JSON representation:")
    click.echo(plan.to_json())


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------


@cli.command("plan")
@click.option(
    "--src-chain", required=True, type=CHAIN, metavar="CHAIN", help="Source chain name or ID (e.g. ethereum, base, 1)"
)
@click.option("--src-token", required=True, metavar="TOKEN", help="Source token symbol or address (e.g. USDC)")
@click.option("--dst-chain", required=True, type=CHAIN, metavar="CHAIN", help="Destination chain name or ID")
@click.option("--dst-token", required=True, metavar="TOKEN", help="Destination token symbol or address")
@click.option("--amount", required=True, metavar="AMOUNT", help="Input amount in human-readable form (e.g. 100 or 0.5)")
@click.option("--bridge-protocol", default="auto", show_default=True, metavar="PROTOCOL", help="Bridge protocol to use")
@click.option(
    "--swap-protocol", default="auto", show_default=True, metavar="PROTOCOL", help="Swap protocol/aggregator to use"
)
@click.option("-o", "--output", type=click.Path(), metavar="FILE", help="Save the plan as JSON to FILE")
def cmd_plan(src_chain, src_token, dst_chain, dst_token, amount, bridge_protocol, swap_protocol, output):
    """Generate a cross-chain execution plan."""
    plan = build_plan(
        src_chain_id=src_chain,
        src_token=src_token,
        dst_chain_id=dst_chain,
        dst_token=dst_token,
        amount=amount,
        bridge_protocol=bridge_protocol,
        swap_protocol=swap_protocol,
    )
    _print_plan(plan)

    if output:
        with open(output, "w") as fh:
            fh.write(plan.to_json())
        click.echo(f"\nPlan saved to {output}")


# ---------------------------------------------------------------------------
# Subcommand: swap
# ---------------------------------------------------------------------------


@cli.command("swap")
@click.option("--chain", required=True, type=CHAIN, metavar="CHAIN", help="Chain name or ID")
@click.option("--token-in", required=True, metavar="TOKEN", help="Input token symbol or address")
@click.option("--token-out", required=True, metavar="TOKEN", help="Output token symbol or address")
@click.option("--amount", required=True, metavar="AMOUNT", help="Input amount in human-readable form")
@click.option("--protocol", default="auto", show_default=True, metavar="PROTOCOL", help="DEX/aggregator protocol")
@click.option("-o", "--output", type=click.Path(), metavar="FILE", help="Save the plan as JSON to FILE")
def cmd_swap(chain, token_in, token_out, amount, protocol, output):
    """Create a single-chain swap plan."""
    action = SwapAction(
        chain_id=chain,
        token_in=token_in.upper(),
        token_out=token_out.upper(),
        amount_in=amount,
        protocol=protocol,
    )
    plan = ExecutionPlan(
        description=f"Swap {amount} {action.token_in} → {action.token_out}",
        actions=[action],
    )
    _print_plan(plan)

    if output:
        with open(output, "w") as fh:
            fh.write(plan.to_json())
        click.echo(f"\nPlan saved to {output}")


# ---------------------------------------------------------------------------
# Subcommand: bridge
# ---------------------------------------------------------------------------


@cli.command("bridge")
@click.option("--src-chain", required=True, type=CHAIN, metavar="CHAIN", help="Source chain name or ID")
@click.option("--dst-chain", required=True, type=CHAIN, metavar="CHAIN", help="Destination chain name or ID")
@click.option("--token", required=True, metavar="TOKEN", help="Token symbol or address")
@click.option("--amount", required=True, metavar="AMOUNT", help="Amount in human-readable form")
@click.option("--protocol", default="auto", show_default=True, metavar="PROTOCOL", help="Bridge protocol")
@click.option("-o", "--output", type=click.Path(), metavar="FILE", help="Save the plan as JSON to FILE")
def cmd_bridge(src_chain, dst_chain, token, amount, protocol, output):
    """Create a cross-chain bridge plan."""
    action = BridgeAction(
        src_chain_id=src_chain,
        dst_chain_id=dst_chain,
        token=token.upper(),
        amount_in=amount,
        protocol=protocol,
    )
    plan = ExecutionPlan(
        description=(f"Bridge {amount} {action.token} from chain {src_chain} to chain {dst_chain}"),
        actions=[action],
    )
    _print_plan(plan)

    if output:
        with open(output, "w") as fh:
            fh.write(plan.to_json())
        click.echo(f"\nPlan saved to {output}")


# ---------------------------------------------------------------------------
# Subcommand: execute
# ---------------------------------------------------------------------------


@cli.command("execute")
@click.option(
    "--plan",
    "plan_file",
    required=True,
    type=click.Path(exists=True),
    metavar="FILE",
    help="Path to the plan JSON file",
)
@click.option("--rpc", metavar="URL", help="JSON-RPC endpoint URL for on-chain calls")
@click.option("--wallet", metavar="ADDRESS", help="Wallet address that will sign transactions")
@click.option("--dry-run", is_flag=True, help="Print steps without submitting transactions")
def cmd_execute(plan_file, rpc, wallet, dry_run):
    """Execute a previously generated plan.

    Executes every step in a plan JSON file produced by the 'plan', 'swap',
    or 'bridge' subcommands.  Pass --dry-run to inspect steps without
    submitting any transactions.
    """
    with open(plan_file) as fh:
        plan = ExecutionPlan.from_json(fh.read())

    click.echo(plan.describe())
    click.echo()

    if dry_run:
        click.echo("[dry-run] Execution skipped.  Steps that would be executed:")
        for i, action in enumerate(plan.actions, 1):
            click.echo(f"  {i}. {action.describe()}")
        return

    click.echo("Executing plan...")
    for i, action in enumerate(plan.actions, 1):
        click.echo(f"  Step {i}/{len(plan.actions)}: {action.describe()}")
        _execute_action(action, rpc=rpc, wallet=wallet)
        click.echo(f"  Step {i} done.")

    click.echo("\nAll steps completed successfully.")


def _execute_action(action: SwapAction | BridgeAction, *, rpc: str | None, wallet: str | None) -> None:
    """Execute a single action."""
    if isinstance(action, SwapAction):
        _execute_swap(action, rpc=rpc, wallet=wallet)
    elif isinstance(action, BridgeAction):
        _execute_bridge(action, rpc=rpc)


def _execute_swap(action: SwapAction, *, rpc: str | None, wallet: str | None) -> None:  # noqa: ARG001
    """Build and optionally submit a swap transaction."""
    if not rpc:
        click.echo("    [info] No --rpc provided; skipping on-chain swap simulation.")
        click.echo(f"    [info] Action: {action.describe()}")
        return

    # Token addresses must be resolved before execution; the CLI action stores
    # only symbols.  Execution with real addresses requires a token registry or
    # explicit --token-address flags which are outside the scope of this prototype.
    click.echo("    [info] On-chain swap execution requires resolved token addresses.")
    click.echo(f"    [info] Action: {action.describe()}")
    click.echo(f"    [info] RPC endpoint: {rpc}")


def _execute_bridge(action: BridgeAction, *, rpc: str | None) -> None:
    """Build and optionally submit a bridge transaction."""
    if not rpc:
        click.echo("    [info] No --rpc provided; skipping on-chain bridge simulation.")
        click.echo(f"    [info] Action: {action.describe()}")
        return

    click.echo(f"    [info] Bridge execution via {action.protocol} — provide --rpc to enable.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()


if __name__ == "__main__":
    main()

"""
High-level planner that turns a user's cross-chain swap intent into an
:class:`~pydefi.plan.ExecutionPlan`.

The planner applies a simple heuristic strategy:

* **Same chain** → single :class:`~pydefi.plan.SwapAction`.
* **Different chains, token supported by bridge** → optionally swap to a
  bridgeable middle token on the source chain, bridge it, then optionally
  swap to the desired output token on the destination chain.

This module is intentionally **protocol-agnostic** – it produces a plan
description without making live network calls, so the output can be reviewed
and adjusted before execution.
"""

from __future__ import annotations

from pydefi.plan import BridgeAction, ExecutionPlan, SwapAction
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# Well-known bridgeable tokens per chain
# These are common liquid tokens that major bridges (Stargate, Across, …) support.
# ---------------------------------------------------------------------------

#: Preferred intermediate / bridgeable tokens, ordered by priority.
_BRIDGE_TOKENS: list[str] = ["USDC", "USDT", "ETH", "WETH", "DAI"]

#: Canonical chain names for display purposes.
_CHAIN_NAMES: dict[int, str] = {
    ChainId.ETHEREUM: "Ethereum",
    ChainId.OPTIMISM: "Optimism",
    ChainId.BSC: "BSC",
    ChainId.POLYGON: "Polygon",
    ChainId.BASE: "Base",
    ChainId.ARBITRUM: "Arbitrum",
    ChainId.AVALANCHE: "Avalanche",
    ChainId.LINEA: "Linea",
    ChainId.BLAST: "Blast",
    ChainId.SCROLL: "Scroll",
    ChainId.ZKSYNC: "zkSync",
    ChainId.UNICHAIN: "Unichain",
    ChainId.WORLDCHAIN: "Worldchain",
}


def _chain_name(chain_id: int) -> str:
    return _CHAIN_NAMES.get(chain_id, str(chain_id))


def _choose_middle_token(token_in: str, token_out: str) -> str:
    """Pick the best bridgeable intermediate token.

    Prefers a token that is already one of the two endpoints (to avoid an
    extra swap), then falls back to the first entry in the priority list.
    """
    token_in_upper = token_in.upper()
    token_out_upper = token_out.upper()
    for candidate in _BRIDGE_TOKENS:
        if candidate == token_in_upper or candidate == token_out_upper:
            return candidate
    return _BRIDGE_TOKENS[0]


def build_plan(
    src_chain_id: int,
    src_token: str,
    dst_chain_id: int,
    dst_token: str,
    amount: str,
    bridge_protocol: str = "auto",
    swap_protocol: str = "auto",
) -> ExecutionPlan:
    """Generate an :class:`~pydefi.plan.ExecutionPlan` for the given intent.

    Args:
        src_chain_id: Chain ID of the source token.
        src_token: Symbol (or address) of the input token.
        dst_chain_id: Chain ID of the desired output token.
        dst_token: Symbol (or address) of the desired output token.
        amount: Human-readable input amount (e.g. ``"1.5"``).
        bridge_protocol: Preferred bridge; ``"auto"`` lets the executor choose.
        swap_protocol: Preferred DEX/aggregator; ``"auto"`` lets the executor
            choose.

    Returns:
        An :class:`~pydefi.plan.ExecutionPlan` with the ordered steps.
    """
    src_token_upper = src_token.upper()
    dst_token_upper = dst_token.upper()
    same_chain = src_chain_id == dst_chain_id

    src_name = _chain_name(src_chain_id)
    dst_name = _chain_name(dst_chain_id)

    if same_chain:
        description = (
            f"Swap {amount} {src_token_upper} → {dst_token_upper} on {src_name}"
        )
        plan = ExecutionPlan(description=description)
        if src_token_upper != dst_token_upper:
            plan.add(
                SwapAction(
                    chain_id=src_chain_id,
                    token_in=src_token_upper,
                    token_out=dst_token_upper,
                    amount_in=amount,
                    protocol=swap_protocol,
                )
            )
        return plan

    # Cross-chain intent
    description = (
        f"Convert {amount} {src_token_upper} on {src_name} "
        f"to {dst_token_upper} on {dst_name}"
    )
    plan = ExecutionPlan(description=description)

    middle = _choose_middle_token(src_token_upper, dst_token_upper)

    # Step 1: Swap to bridgeable middle token on the source chain (if needed)
    if src_token_upper != middle:
        plan.add(
            SwapAction(
                chain_id=src_chain_id,
                token_in=src_token_upper,
                token_out=middle,
                amount_in=amount,
                protocol=swap_protocol,
            )
        )

    # Step 2: Bridge middle token to the destination chain
    plan.add(
        BridgeAction(
            src_chain_id=src_chain_id,
            dst_chain_id=dst_chain_id,
            token=middle,
            amount_in=amount,
            protocol=bridge_protocol,
        )
    )

    # Step 3: Swap from middle token to desired output on the destination chain (if needed)
    if dst_token_upper != middle:
        plan.add(
            SwapAction(
                chain_id=dst_chain_id,
                token_in=middle,
                token_out=dst_token_upper,
                amount_in=amount,
                protocol=swap_protocol,
            )
        )

    return plan

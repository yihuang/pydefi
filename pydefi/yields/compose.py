"""One-signature cross-chain yield deposits via bridge compose hooks.

A normal cross-chain yield route is two-phase: bridge on the source chain,
then :func:`~pydefi.yields.build_followup_route` to supply on the destination
once the bridge settles. Compose-capable bridges collapse that — the bridge
message carries a DeFiVM program, and a composer contract on the destination
chain runs it the instant the funds arrive, approving and supplying into the
target market in the same transaction. The user signs once, on the source
chain; there is no follow-up leg.

Both CCTP v2 (``depositForBurnWithHook`` → ``CCTPComposer``) and Chainlink
CCIP (``ccipSend`` with ``data`` → ``CCIPComposer``) expose this pattern
behind a duck-typed ``build_bridge_compose_tx`` method (see
:class:`~pydefi.bridge.Bridge`), so :func:`build_compose_supply_route` works
with either bridge unchanged.
"""

from __future__ import annotations

from eth_contract.erc20 import ERC20
from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3

from pydefi.abi.lending import AAVE_V3_POOL, COMPOUND_V3_COMET
from pydefi.bridge import Bridge
from pydefi.types import Address, Token, TokenAmount
from pydefi.vm import ProgramContext
from pydefi.yields.router import Protocol, YieldMarket, YieldRoute, YieldStep, build_approve_tx, supply_contract

#: The composer stages the bridged amount in this transient-storage slot
#: before delegatecalling the program (slot 1 holds the source domain/selector).
_AMOUNT_RECEIVED_SLOT = 0


def build_compose_supply_program(protocol: Protocol, supply_target: Address, token: Token, user: Address) -> bytes:
    """Build the DeFiVM program a composer runs after the bridged tokens arrive.

    The composer stages the received amount in transient slot 0; the program
    reads it, approves *supply_target* for it, and supplies it into the
    destination lending market for *user*. Returns raw DeFiVM bytecode to
    embed in the bridge's compose message.

    The program runs in the composer's context, so the approve and the pulled
    tokens are the composer's — but the supplied position is credited to
    *user* (Aave V3 ``onBehalfOf`` / Comet ``supplyTo`` recipient).
    """
    prog = ProgramContext()
    amount = prog.builder.tload(IRLiteral(_AMOUNT_RECEIVED_SLOT))
    prog.assert_(prog.call_contract(token.address, ERC20.fns.approve, supply_target, amount))
    if protocol == "aave_v3":
        supplied = prog.call_contract(supply_target, AAVE_V3_POOL.fns.supply, token.address, amount, user, 0)
    elif protocol == "compound_v3":
        supplied = prog.call_contract(supply_target, COMPOUND_V3_COMET.fns.supplyTo, user, token.address, amount)
    else:
        raise ValueError(f"build_compose_supply_program: unsupported protocol {protocol!r}")
    prog.assert_(supplied)
    prog.builder.stop()
    return prog.build()


async def build_compose_supply_route(
    user: Address,
    amount_in: TokenAmount,
    target_market: YieldMarket,
    composer_address: Address,
    bridge: Bridge,
    w3s: dict[int, AsyncWeb3],
) -> YieldRoute:
    """Build a one-signature cross-chain yield deposit via bridge compose hooks.

    Returns a :class:`YieldRoute` of ``[approve, bridge]``, both on the source
    chain: approve the bridge entrypoint to pull *amount_in*, then a compose
    transaction carrying a DeFiVM program that supplies into *target_market*
    on the destination. There is no follow-up — the destination supply runs
    inside the bridge's compose hook, so ``route.pending`` is ``None``.

    Args:
        user: The account credited with the destination supply position.
        amount_in: Tokens on the source chain to bridge and supply.
        target_market: The destination ``aave_v3`` / ``compound_v3`` market.
        composer_address: The composer contract deployed on the destination.
        bridge: A compose-capable bridge whose src / dst chains match
            *amount_in*'s chain and *target_market*'s chain.
        w3s: Per-chain ``AsyncWeb3`` map — the destination chain must be
            present (its supply target is resolved on-chain).
    """
    if amount_in.token.chain_id != bridge.src_chain_id:
        raise ValueError(
            f"build_compose_supply_route: amount_in.token is on chain {amount_in.token.chain_id}, "
            f"bridge source is {bridge.src_chain_id}"
        )
    if target_market.chain_id != bridge.dst_chain_id:
        raise ValueError(
            f"build_compose_supply_route: target_market is on chain {target_market.chain_id}, "
            f"bridge destination is {bridge.dst_chain_id}"
        )
    spender = getattr(bridge, "spender", None)
    if spender is None:
        raise ValueError(f"{bridge.protocol_name} bridge exposes no ERC-20 spender — it cannot carry a compose route")
    w3 = w3s.get(target_market.chain_id)
    if w3 is None:
        raise ValueError(f"build_compose_supply_route: w3s has no entry for destination chain {target_market.chain_id}")
    supply_target = await supply_contract(target_market, w3)
    program = build_compose_supply_program(target_market.protocol, supply_target, target_market.token, user)
    build_compose_tx = getattr(bridge, "build_bridge_compose_tx", None)
    if build_compose_tx is None:
        raise NotImplementedError(f"{bridge.protocol_name} bridge has no compose path")
    bridge_tx = await build_compose_tx(amount_in, composer_address, program)
    approve_tx = build_approve_tx(amount_in.token, spender, amount_in.amount)
    return YieldRoute(
        strategy="compose_supply",
        source_chain=bridge.src_chain_id,
        steps=(
            YieldStep("approve", bridge.src_chain_id, approve_tx),
            YieldStep("bridge", bridge.src_chain_id, bridge_tx),
        ),
        route_id=f"compose_supply:chain:{bridge.src_chain_id}->{target_market.market_id}",
        target_market=target_market,
        target_chain=target_market.chain_id,
    )


__all__ = ["build_compose_supply_route", "build_compose_supply_program"]

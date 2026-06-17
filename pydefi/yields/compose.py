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

from pydefi._utils import erc20_approve_tx
from pydefi.abi.lending import AAVE_V3_POOL, COMPOUND_V3_COMET
from pydefi.bridge import Bridge
from pydefi.types import Address, Token, TokenAmount
from pydefi.vm import ProgramContext
from pydefi.yields.router import (
    Protocol,
    YieldMarket,
    YieldRoute,
    YieldStep,
    _sign_request_7702,
    _sign_request_permit2,
    supply_contract,
)

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
    defivm: Address | None = None,
    delegate: Address | None = None,
) -> YieldRoute:
    """Build a one-signature cross-chain yield deposit via bridge compose hooks.

    Returns a ``[approve, bridge]`` :class:`YieldRoute` on the source chain: the
    bridge message carries a DeFiVM program that supplies *amount_in* into
    *target_market* the instant funds land, so there is no follow-up
    (``route.pending is None``). *bridge* must match *amount_in*'s and
    *target_market*'s chains; *w3s* must hold the destination chain (and the
    source chain when going gasless).

    Pass *delegate* (EIP-7702) or *defivm* (Permit2) — the same knobs as
    :func:`build_yield_route` — to collapse the source leg to one sponsored step
    signed via :func:`sign_route`. *delegate* batches approve + bridge in the
    EOA's own context (``bridge_with_7702``), paying any native bridge fee from
    the EOA; *defivm* runs them inside the VM (``bridge_with_permit2``), which
    holds no native funds and so needs a zero-value lane (CCTP). *delegate* wins
    if both are set.
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
    approve_tx = erc20_approve_tx(amount_in.token.address, spender, amount_in.amount)
    if delegate is not None or defivm is not None:
        w3_src = w3s.get(bridge.src_chain_id)
        if w3_src is None:
            raise ValueError(f"build_compose_supply_route: w3s has no entry for source chain {bridge.src_chain_id}")
    if delegate is not None:
        sign_req = await _sign_request_7702(w3_src, user, delegate, bridge.src_chain_id, [approve_tx, bridge_tx])
        steps: tuple[YieldStep, ...] = (
            YieldStep("bridge_with_7702", bridge.src_chain_id, None, sign_request=sign_req),
        )
    elif defivm is not None:
        if int(bridge_tx.get("value") or 0) != 0:
            raise ValueError(
                "build_compose_supply_route: defivm cannot pay the bridge's native fee "
                f"(value={bridge_tx['value']}) — use a zero-value lane (CCTP) or delegate= instead"
            )
        bridge_data = bytes.fromhex(bridge_tx["data"][2:])
        sign_req = await _sign_request_permit2(w3_src, user, defivm, amount_in, spender, bridge_data)
        steps = (YieldStep("bridge_with_permit2", bridge.src_chain_id, None, sign_request=sign_req),)
    else:
        steps = (
            YieldStep("approve", bridge.src_chain_id, approve_tx),
            YieldStep("bridge", bridge.src_chain_id, bridge_tx),
        )
    return YieldRoute(
        strategy="compose_supply",
        source_chain=bridge.src_chain_id,
        steps=steps,
        route_id=f"compose_supply:chain:{bridge.src_chain_id}->{target_market.market_id}",
        target_market=target_market,
        target_chain=target_market.chain_id,
    )


__all__ = ["build_compose_supply_route", "build_compose_supply_program"]

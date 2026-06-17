"""Market-aware counterfactual deposit addresses — give out an address, the
deposit lands in a yield market.

The :class:`~pydefi.yields.YieldMarket` layer over
:mod:`pydefi.vm.yield_deposit`: pick a market, get a deterministic address
committed to "supply whatever arrives here into this market for this owner".
Fund it by plain transfer (CEX withdrawal, salary, a friend) — no signature,
no prior deployment — then anyone broadcasts the sweep tx, earning the
committed ``execution_fee``. Each address is single-shot (the Safe's ``setup``
runs once); rotate ``salt_nonce`` for recurring inflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web3 import AsyncWeb3

from pydefi.lending import AaveV3, CompoundV3
from pydefi.types import Address, TokenAmount
from pydefi.vm.yield_deposit import (
    AMOUNT_SENTINEL,
    build_deposit_program,
    build_sweep_tx,
    deposit_address,
)
from pydefi.yields.router import YieldMarket


@dataclass(frozen=True)
class YieldDeposit:
    """A counterfactual deposit address bound to one market and one owner.

    ``address`` receives funds by plain transfer; ``sweep_tx`` is the
    ``createProxyWithNonce`` deployment any relayer broadcasts to pay the fee to
    ``tx.origin`` and supply the rest into ``market`` for ``owner``. ``program``
    is the committed sweep bytecode."""

    address: Address
    sweep_tx: dict[str, Any]
    owner: Address
    market: YieldMarket
    salt_nonce: int
    program: bytes


async def _supply_template(market: YieldMarket, owner: Address, w3: AsyncWeb3) -> tuple[Address, bytes]:
    """The market's supply contract and its supply calldata crediting *owner*,
    with :data:`AMOUNT_SENTINEL` where the swept balance is patched in. The
    Safe is ``msg.sender`` at sweep time, so only ``onBehalfOf`` /
    ``supplyTo``-style protocols qualify."""
    sentinel = TokenAmount(market.token, AMOUNT_SENTINEL)
    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        return aave.pool_address, bytes.fromhex(aave.build_supply_tx(owner, sentinel)["data"][2:])
    if market.protocol == "compound_v3":
        comet = CompoundV3.from_chain(w3, market.chain_id, market.token.symbol)
        return comet.comet_address, bytes.fromhex(comet.build_supply_tx(sentinel, dst=owner)["data"][2:])
    raise ValueError(f"build_yield_deposit: unsupported protocol {market.protocol!r}")


async def build_yield_deposit(
    owner: Address,
    market: YieldMarket,
    defivm: Address,
    w3: AsyncWeb3,
    execution_fee: int,
    salt_nonce: int = 0,
    *,
    approve_reset: bool = False,
) -> YieldDeposit:
    """Derive the counterfactual deposit address for *market* and its sweep tx.

    *owner* is credited the supplied position and owns the deployed Safe (so
    leftovers stay under its control); the address only accepts *market.token*
    (``aave_v3`` / ``compound_v3`` only — both credit *owner* via
    ``onBehalfOf`` / ``supplyTo``). *execution_fee* is *market.token* units paid
    to whoever broadcasts the sweep (0 if *owner* sweeps); *salt_nonce* rotates
    the address for recurring inflows; *approve_reset* prepends a USDT-style
    ``approve(0)``. *defivm* is the deployed :sol:`DeFiVM` the Safe setup
    delegatecalls, and *w3* targets the market's chain.
    """
    target, template = await _supply_template(market, owner, w3)
    program = build_deposit_program(market.token.address, target, template, execution_fee, approve_reset=approve_reset)
    address = await deposit_address(w3, owner, defivm, program, salt_nonce)
    return YieldDeposit(
        address=address,
        sweep_tx=build_sweep_tx(owner, defivm, program, salt_nonce),
        owner=owner,
        market=market,
        salt_nonce=salt_nonce,
        program=program,
    )


__all__ = ["YieldDeposit", "build_yield_deposit"]

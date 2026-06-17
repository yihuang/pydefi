"""Cross-chain compose primitives: the DeFiVM programs a bridge runs, and the
supply calldata they embed.

* :func:`build_supply_template` — per-protocol ``supply`` calldata (Aave V3 /
  Compound III / Morpho), with :data:`~pydefi.vm.yield_deposit.AMOUNT_SENTINEL`
  marking the runtime-amount patch site, crediting the user via ``onBehalfOf`` /
  ``dst``. Aave V4 rejected (position-manager gated).
* :func:`build_compose_deposit_program` — destination program: read the bridged
  amount from transient slot 0 (``TLOAD``), optionally swap, then approve +
  supply. Runs in the composer's DELEGATECALL context (so ``msg.sender`` is the
  composer; V3/V4 callbacks resolve via the inherited ``DEXCallbackRouter``).
  Committed at bridge time, so ``min_out`` is the only adverse-price guard.
* :func:`build_source_swap_bridge_program` — source mirror: swap the input token
  into USDC, then ``depositForBurnWithHook`` with the swapped amount and the
  destination program as ``hookData``. Run via ``DeFiVM.execute``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vyper.venom.basicblock import IRLiteral
from web3 import AsyncWeb3

from pydefi.lending.aave_v3 import AaveV3
from pydefi.lending.compound_v3 import CompoundV3
from pydefi.lending.morpho import MorphoBlue
from pydefi.pathfinder.dag import RouteDAG
from pydefi.types import Address, TokenAmount
from pydefi.vm.context import Program
from pydefi.vm.dag import _build_dag_actions
from pydefi.vm.erc20 import emit_approve
from pydefi.vm.yield_deposit import AMOUNT_SENTINEL, find_amount_offset
from pydefi.yields.router import YieldMarket, morpho_params

#: Transient-storage slot the bridge composer stages the received amount in.
#: ``CCTPComposer.receiveAndExecute`` writes ``amountReceived`` to slot 0 (and
#: ``sourceDomain`` to slot 1) before DELEGATECALLing the program; the program
#: reads it back with ``TLOAD(0)``.
AMOUNT_RECEIVED_SLOT = 0

#: Calldata offset of the burn ``amount`` in ``depositForBurnWithHook`` — the
#: first ABI argument, so the 32-byte word right after the 4-byte selector.  The
#: shared :data:`~pydefi.vm.yield_deposit.AMOUNT_SENTINEL` can't locate it here:
#: the destination program (which *is* the burn ``hookData``) embeds a supply
#: template that already carries the sentinel, so it would occur more than once.
CCTP_BURN_AMOUNT_OFFSET = 4


# ---------------------------------------------------------------------------
# Supply-calldata templates (per protocol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupplyTemplate:
    """A protocol ``supply`` calldata template plus the contract to approve.

    Pass *template* / *spender* / *token* straight into
    :func:`build_compose_deposit_program` as ``supply_template`` / ``protocol`` /
    ``target_token``.
    """

    template: bytes
    spender: Address
    token: Address


def _calldata(tx: dict) -> bytes:
    """Extract raw calldata bytes from a pydefi tx-dict, asserting the amount
    sentinel is present exactly once (so the runtime patch lands)."""
    data = tx["data"]
    raw = bytes.fromhex(data[2:] if data.startswith("0x") else data)
    find_amount_offset(raw)  # raises unless the sentinel occurs exactly once
    return raw


async def build_supply_template(
    market: YieldMarket,
    w3: AsyncWeb3,
    on_behalf_of: Address,
) -> SupplyTemplate:
    """Build the supply-calldata template for *market*, crediting *on_behalf_of*.

    The amount is :data:`~pydefi.vm.yield_deposit.AMOUNT_SENTINEL`; the compose
    program overwrites it with the runtime deposit amount.

    Raises:
        ValueError: For ``aave_v4`` (position-manager gated) or an unknown
            protocol.
    """
    amount = TokenAmount(market.token, AMOUNT_SENTINEL)

    if market.protocol == "aave_v3":
        aave = await AaveV3.from_chain(w3, market.chain_id)
        tx = aave.build_supply_tx(on_behalf_of, amount, on_behalf_of=on_behalf_of)
        return SupplyTemplate(_calldata(tx), aave.pool_address, market.token.address)

    if market.protocol == "compound_v3":
        comet = CompoundV3.from_chain(w3, market.chain_id, market.token.symbol)
        tx = comet.build_supply_tx(amount, dst=on_behalf_of)
        return SupplyTemplate(_calldata(tx), comet.comet_address, market.token.address)

    if market.protocol == "morpho":
        morpho = MorphoBlue.from_chain(w3, market.chain_id)
        params = await morpho_params(morpho, market)
        tx = morpho.build_supply_tx(params, assets=amount, on_behalf_of=on_behalf_of)
        return SupplyTemplate(_calldata(tx), morpho.morpho_address, market.token.address)

    if market.protocol == "aave_v4":
        raise ValueError(
            "aave_v4 is unsupported for compose: Spoke.supply(onBehalfOf) requires the caller "
            "to be a user-approved position manager, which the composer is not."
        )

    raise ValueError(f"unknown protocol: {market.protocol!r}")


# ---------------------------------------------------------------------------
# Destination program: (swap ->) supply, run by the bridge composer
# ---------------------------------------------------------------------------


def build_compose_deposit_program(
    *,
    supply_template: bytes,
    protocol: Address,
    target_token: Address,
    composer: Address,
    swap_dag: RouteDAG | None = None,
    min_out: int = 0,
    approve_reset: bool = False,
) -> bytes:
    """Destination program: read the delivered amount from transient slot 0,
    optionally lower *swap_dag* into *target_token* (crediting *composer*), then
    approve *protocol* and run *supply_template* with the amount patched over its
    ``AMOUNT_SENTINEL`` word. Returns DeFiVM bytecode to embed as the bridge
    ``hookData``.

    *supply_template* must credit the user (the composer is ``msg.sender``).
    *min_out* guards the deposit amount — the only adverse-price guard, since the
    program is committed at bridge time. *approve_reset* prepends ``approve(0)``
    for USDT-style tokens.
    """
    amount_offset = find_amount_offset(supply_template)  # validates: exactly one sentinel

    prog = Program()
    received = prog.builder.tload(IRLiteral(AMOUNT_RECEIVED_SLOT))

    if swap_dag is not None:
        if not swap_dag.actions:
            raise ValueError("swap_dag has no actions; pass swap_dag=None for a no-swap deposit")
        self_hex = composer.to_0x_hex()
        deposit_amount = _build_dag_actions(
            prog,
            received,
            swap_dag.actions,
            vm_address=self_hex,
            terminal_recipient=self_hex,
        )
    else:
        deposit_amount = received

    if min_out > 0:
        prog.assert_ge(deposit_amount, min_out, "compose: out too low")

    emit_approve(prog, target_token, protocol, deposit_amount, reset=approve_reset)
    prog.assert_(
        prog.call_raw(bytes(protocol), supply_template, patches={amount_offset: deposit_amount}),
        "supply failed",
    )
    prog.builder.stop()
    return prog.build()


# ---------------------------------------------------------------------------
# Source program: swap -> burn (CCTP depositForBurnWithHook)
# ---------------------------------------------------------------------------


def build_source_swap_bridge_program(
    *,
    swap_dag: RouteDAG,
    amount_in: int,
    usdc: Address,
    token_messenger: Address,
    burn_template: bytes,
    executor: Address,
    burn_amount_offset: int = CCTP_BURN_AMOUNT_OFFSET,
    min_usdc_out: int = 0,
    approve_reset: bool = False,
) -> bytes:
    """Source program: swap *amount_in* of the input token into USDC (crediting
    *executor*), approve the CCTP ``TokenMessengerV2``, then run *burn_template*
    (``depositForBurnWithHook``) with the swapped amount patched at
    *burn_amount_offset*. Returns DeFiVM bytecode to run via ``DeFiVM.execute``
    (the VM must hold *amount_in*).

    *swap_dag*'s last token must be USDC. *min_usdc_out* guards the swap output;
    *approve_reset* prepends ``approve(0)`` for USDT-style tokens.
    """
    if not swap_dag.actions:
        raise ValueError("swap_dag has no actions; the source swap must produce USDC to bridge")

    prog = Program()
    self_hex = executor.to_0x_hex()
    usdc_out = _build_dag_actions(
        prog,
        prog.const(amount_in),
        swap_dag.actions,
        vm_address=self_hex,
        terminal_recipient=self_hex,
    )
    if min_usdc_out > 0:
        prog.assert_ge(usdc_out, min_usdc_out, "source swap: usdc out too low")

    emit_approve(prog, usdc, token_messenger, usdc_out, reset=approve_reset)
    prog.assert_(
        prog.call_raw(bytes(token_messenger), burn_template, patches={burn_amount_offset: usdc_out}),
        "burn failed",
    )
    prog.builder.stop()
    return prog.build()

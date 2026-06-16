"""Source-chain route builders for the CCTP / CCIP atomic-compose path.

``build_cctp_deposit_route`` / ``build_ccip_deposit_route`` compile the
destination ``swap → supply`` program and return the source legs (approve +
bridge), committing "bridge → swap → deposit, crediting the user" in one source
transaction. The destination leg isn't a transaction here — it is the committed
``dest_program``, run when a relayer triggers the composer after settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hexbytes import HexBytes
from web3 import AsyncWeb3

from pydefi._utils import to_tx
from pydefi.abi.vm import DeFiVM
from pydefi.bridge.ccip import CCIP
from pydefi.bridge.cctp import CCTP, FINALITY_THRESHOLD_CONFIRMED
from pydefi.crosschain.compose import (
    build_compose_deposit_program,
    build_source_swap_bridge_program,
    build_supply_template,
)
from pydefi.pathfinder.dag import RouteDAG
from pydefi.types import ZERO_ADDRESS, Address, Token, TokenAmount
from pydefi.yields.router import YieldMarket, build_approve_tx

LegKind = Literal["approve", "bridge_compose", "swap_bridge"]


@dataclass(frozen=True)
class CrossChainLeg:
    """One source-chain transaction in a :class:`CrossChainRoute`."""

    kind: LegKind
    chain_id: int
    tx: dict[str, Any]


@dataclass(frozen=True)
class CrossChainRoute:
    """Source-chain legs for an atomic-compose cross-chain deposit.

    The destination ``swap → supply`` is committed in *dest_program* (embedded as
    the bridge ``hookData``) and executes when a relayer triggers
    ``Composer.receiveAndExecute`` after the bridge settles — there is no
    destination transaction to broadcast here.
    """

    strategy: str
    source_chain: int
    dest_chain: int
    composer: Address
    dest_program: bytes
    steps: tuple[CrossChainLeg, ...]
    route_id: str
    dest_market: YieldMarket | None = None

    def build_transactions(self) -> list[dict[str, Any]]:
        """The source-chain txs to broadcast in order (approve, then burn)."""
        return [leg.tx for leg in self.steps]


async def _build_dest_program(
    *,
    w3_dst: AsyncWeb3,
    composer: Address,
    user: Address,
    dest_market: YieldMarket,
    dest_swap: RouteDAG | None,
    min_out: int,
    approve_reset: bool,
) -> bytes:
    """Compile the destination ``swap → supply`` program for *dest_market*,
    crediting *user* (shared by the USDC-in and swap-in route builders)."""
    supply = await build_supply_template(dest_market, w3_dst, user)
    return build_compose_deposit_program(
        supply_template=supply.template,
        protocol=supply.spender,
        target_token=supply.token,
        composer=composer,
        swap_dag=dest_swap,
        min_out=min_out,
        approve_reset=approve_reset,
    )


def _src_usdc_token(cctp: CCTP) -> Token:
    return Token(chain_id=cctp.src_chain_id, address=cctp.src_usdc_address, symbol="USDC", decimals=6)


async def build_cctp_compose_route(
    *,
    cctp: CCTP,
    amount_in: TokenAmount,
    composer: Address,
    dest_program: bytes,
    max_fee: int = 0,
    min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
    dst_domain: int | None = None,
    dest_market: YieldMarket | None = None,
) -> CrossChainRoute:
    """Source legs for a CCTP atomic compose: ``approve(TokenMessengerV2)`` then
    ``depositForBurnWithHook`` with *dest_program* as ``hookData`` (USDC minted
    to, and callable only by, *composer*). *dst_domain* overrides the destination
    domain for chains absent from the bundled mainnet-only map (e.g. testnets)."""
    approve_tx = build_approve_tx(amount_in.token, cctp.token_messenger_address, amount_in.amount)
    burn_tx = await cctp.build_bridge_compose_tx(
        amount_in,
        composer,
        dest_program,
        dst_domain=dst_domain,
        max_fee=max_fee,
        min_finality_threshold=min_finality_threshold,
    )
    return CrossChainRoute(
        strategy="cctp_compose",
        source_chain=cctp.src_chain_id,
        dest_chain=cctp.dst_chain_id,
        composer=composer,
        dest_program=dest_program,
        steps=(
            CrossChainLeg("approve", cctp.src_chain_id, approve_tx),
            CrossChainLeg("bridge_compose", cctp.src_chain_id, burn_tx),
        ),
        route_id=f"cctp_compose:{cctp.src_chain_id}->{cctp.dst_chain_id}",
        dest_market=dest_market,
    )


async def build_cctp_deposit_route(
    *,
    cctp: CCTP,
    w3_dst: AsyncWeb3,
    amount_in: TokenAmount,
    composer: Address,
    user: Address,
    dest_market: YieldMarket,
    dest_swap: RouteDAG | None = None,
    min_out: int = 0,
    approve_reset: bool = False,
    max_fee: int = 0,
    min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
    dst_domain: int | None = None,
) -> CrossChainRoute:
    """Full source route for "bridge USDC → (swap) → supply into *dest_market*,
    crediting *user*". *dest_swap* is the destination USDC→``dest_market.token``
    route (``None`` if the market token is USDC); set *min_out* from an off-chain
    quote. *dst_domain* overrides the CCTP destination domain for testnet chains."""
    dest_program = await _build_dest_program(
        w3_dst=w3_dst,
        composer=composer,
        user=user,
        dest_market=dest_market,
        dest_swap=dest_swap,
        min_out=min_out,
        approve_reset=approve_reset,
    )
    return await build_cctp_compose_route(
        cctp=cctp,
        amount_in=amount_in,
        composer=composer,
        dest_program=dest_program,
        max_fee=max_fee,
        min_finality_threshold=min_finality_threshold,
        dst_domain=dst_domain,
        dest_market=dest_market,
    )


async def build_cctp_swap_deposit_route(
    *,
    cctp: CCTP,
    w3_dst: AsyncWeb3,
    source_swap: RouteDAG,
    amount_in: int,
    vm_address: Address,
    composer: Address,
    user: Address,
    dest_market: YieldMarket,
    dest_swap: RouteDAG | None = None,
    min_usdc_out: int = 0,
    min_dst_out: int = 0,
    source_approve_reset: bool = False,
    dest_approve_reset: bool = False,
    max_fee: int = 0,
    min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
) -> CrossChainRoute:
    """Full source route for "swap *source_swap* into USDC → bridge → (swap) →
    supply into *dest_market*, crediting *user*" when the input is not USDC. One
    DeFiVM ``execute`` leg (``swap_bridge``): swap *amount_in* into USDC and burn
    it with the destination program as ``hookData``. *vm_address* must hold
    *amount_in* (funded via Permit2/transfer). *source_swap* ends at USDC;
    *dest_swap* is the destination USDC→``dest_market.token`` route. Set
    *min_usdc_out* / *min_dst_out* from off-chain quotes."""
    dest_program = await _build_dest_program(
        w3_dst=w3_dst,
        composer=composer,
        user=user,
        dest_market=dest_market,
        dest_swap=dest_swap,
        min_out=min_dst_out,
        approve_reset=dest_approve_reset,
    )
    # Placeholder burn amount (0); the source program patches the swapped USDC
    # amount over CCTP_BURN_AMOUNT_OFFSET at runtime.
    burn_tx = await cctp.build_bridge_compose_tx(
        TokenAmount(_src_usdc_token(cctp), 0),
        composer,
        dest_program,
        max_fee=max_fee,
        min_finality_threshold=min_finality_threshold,
    )
    burn_template = bytes.fromhex(burn_tx["data"].removeprefix("0x"))
    source_program = build_source_swap_bridge_program(
        swap_dag=source_swap,
        amount_in=amount_in,
        usdc=cctp.src_usdc_address,
        token_messenger=cctp.token_messenger_address,
        burn_template=burn_template,
        executor=vm_address,
        min_usdc_out=min_usdc_out,
        approve_reset=source_approve_reset,
    )
    execute_tx = to_tx(vm_address, DeFiVM.fns.execute(source_program).data)
    return CrossChainRoute(
        strategy="cctp_swap_compose",
        source_chain=cctp.src_chain_id,
        dest_chain=cctp.dst_chain_id,
        composer=composer,
        dest_program=dest_program,
        steps=(CrossChainLeg("swap_bridge", cctp.src_chain_id, execute_tx),),
        route_id=f"cctp_swap_compose:{cctp.src_chain_id}->{cctp.dst_chain_id}",
        dest_market=dest_market,
    )


async def estimate_cctp_delivered(
    cctp: CCTP,
    amount_in: TokenAmount,
    min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
) -> int:
    """USDC delivered to the composer on the destination chain (``amount_in``
    minus the CCTP relayer fee), via the Iris fee API.  Use it to set ``min_out``
    — the destination program runs on the delivered amount, not ``amount_in``."""
    dst_usdc = Token(
        chain_id=cctp.dst_chain_id,
        address=CCTP.usdc_address(cctp.dst_chain_id),
        symbol="USDC",
        decimals=6,
    )
    quote = await cctp.get_quote(amount_in.token, dst_usdc, amount_in, min_finality_threshold=min_finality_threshold)
    return quote.amount_out.amount


# ---------------------------------------------------------------------------
# CCIP — Chainlink CCIP atomic compose (any pooled token, not just USDC)
# ---------------------------------------------------------------------------


async def build_ccip_compose_route(
    *,
    ccip: CCIP,
    amount_in: TokenAmount,
    composer: Address,
    dest_program: bytes,
    gas_limit: int = 800_000,
    dest_market: YieldMarket | None = None,
) -> CrossChainRoute:
    """Source legs for a CCIP atomic compose: approve the bridged token (and the
    fee token if ERC-20) to the CCIP Router, then ``ccipSend`` with *dest_program*
    as the message ``data`` and *composer* as receiver. ``CCIPComposer`` stages
    the delivered amount in slot 0 and runs *dest_program* — the same
    :func:`~pydefi.crosschain.compose.build_compose_deposit_program` output as for
    CCTP."""
    router = Address(HexBytes(ccip.router_address))
    send_tx = await ccip.build_bridge_compose_tx(
        amount_in.token, amount_in, composer, dest_program, gas_limit=gas_limit
    )

    steps: list[CrossChainLeg] = [
        CrossChainLeg("approve", ccip.src_chain_id, build_approve_tx(amount_in.token, router, amount_in.amount)),
    ]
    if ccip.fee_token != ZERO_ADDRESS:
        fee = await ccip.quote_fee(amount_in.token, amount_in, composer, data=dest_program, gas_limit=gas_limit)
        fee_token = Token(
            chain_id=ccip.src_chain_id, address=ccip.fee_token, symbol="FEE", decimals=ccip.fee_token_decimals
        )
        steps.append(CrossChainLeg("approve", ccip.src_chain_id, build_approve_tx(fee_token, router, fee)))
    steps.append(CrossChainLeg("bridge_compose", ccip.src_chain_id, send_tx))

    return CrossChainRoute(
        strategy="ccip_compose",
        source_chain=ccip.src_chain_id,
        dest_chain=ccip.dst_chain_id,
        composer=composer,
        dest_program=dest_program,
        steps=tuple(steps),
        route_id=f"ccip_compose:{ccip.src_chain_id}->{ccip.dst_chain_id}",
        dest_market=dest_market,
    )


async def build_ccip_deposit_route(
    *,
    ccip: CCIP,
    w3_dst: AsyncWeb3,
    amount_in: TokenAmount,
    composer: Address,
    user: Address,
    dest_market: YieldMarket,
    dest_swap: RouteDAG | None = None,
    min_out: int = 0,
    dest_approve_reset: bool = False,
    gas_limit: int = 800_000,
) -> CrossChainRoute:
    """Full source route for "bridge *amount_in* via CCIP → (swap) → supply into
    *dest_market*, crediting *user*". *amount_in* is the CCIP-pooled token bridged
    1:1; *dest_swap* is the destination route into ``dest_market.token`` (``None``
    if they match). Set *min_out* from an off-chain quote."""
    dest_program = await _build_dest_program(
        w3_dst=w3_dst,
        composer=composer,
        user=user,
        dest_market=dest_market,
        dest_swap=dest_swap,
        min_out=min_out,
        approve_reset=dest_approve_reset,
    )
    return await build_ccip_compose_route(
        ccip=ccip,
        amount_in=amount_in,
        composer=composer,
        dest_program=dest_program,
        gas_limit=gas_limit,
        dest_market=dest_market,
    )

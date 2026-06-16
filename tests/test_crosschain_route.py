from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.crosschain.compose import build_compose_deposit_program
from pydefi.crosschain.route import (
    build_cctp_swap_deposit_route,
    build_compose_route,
    build_deposit_route,
    estimate_cctp_delivered,
)
from pydefi.types import BridgeQuote, ChainId, TokenAmount
from tests.crosschain_helpers import (
    BRIDGE,
    COMPOSER,
    LINK,
    SENTINEL_TEMPLATE,
    SPENDER,
    USDC_BASE,
    USDC_ETH,
    USER,
    VM,
    WETH_ETH,
    ccip,
    cctp,
    market,
    supply_template,
    swap_dag,
)

_AMOUNT = TokenAmount(USDC_ETH, 1_000_000_000)  # 1000 USDC
_PATCH_SUPPLY = "pydefi.crosschain.route.build_supply_template"
_SEND_TX = {"to": BRIDGE, "data": "0xfeed", "value": "1000", "gas": "800000"}


# ---------------------------------------------------------------------------
# CCTP routes (build_compose_route / build_deposit_route over a CCTP bridge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_route_has_approve_then_burn():
    program = build_compose_deposit_program(
        supply_template=SENTINEL_TEMPLATE, protocol=SPENDER, target_token=USDC_BASE.address, composer=COMPOSER
    )
    route = await build_compose_route(bridge=cctp(), amount_in=_AMOUNT, composer=COMPOSER, dest_program=program)

    assert route.source_chain == ChainId.ETHEREUM
    assert route.dest_chain == ChainId.BASE
    assert route.composer == COMPOSER
    assert [leg.kind for leg in route.steps] == ["approve", "bridge_compose"]

    approve, burn = route.steps
    # Approve targets the USDC token; the messenger is the spender in calldata.
    assert approve.tx["to"] == USDC_ETH.address
    assert BRIDGE.hex() in approve.tx["data"].lower()
    # Burn targets the TokenMessengerV2 and carries the program as hookData.
    assert burn.tx["to"].lower() == BRIDGE.to_0x_hex().lower()
    assert program.hex() in burn.tx["data"].lower()
    assert route.build_transactions() == [approve.tx, burn.tx]


@pytest.mark.asyncio
async def test_compose_route_dst_domain_override():
    program = build_compose_deposit_program(
        supply_template=SENTINEL_TEMPLATE, protocol=SPENDER, target_token=USDC_BASE.address, composer=COMPOSER
    )
    route = await build_compose_route(
        bridge=cctp(), amount_in=_AMOUNT, composer=COMPOSER, dest_program=program, dst_domain=7
    )
    # depositForBurnWithHook calldata: selector(4) + amount(32) + destinationDomain(32, uint32 right-aligned).
    burn_data = bytes.fromhex(route.steps[1].tx["data"].removeprefix("0x"))
    assert int.from_bytes(burn_data[36:68], "big") == 7


@pytest.mark.asyncio
async def test_deposit_route_compiles_program_and_embeds_supply():
    with patch(_PATCH_SUPPLY, new=AsyncMock(return_value=supply_template())):
        route = await build_deposit_route(
            bridge=cctp(), w3_dst=MagicMock(), amount_in=_AMOUNT, composer=COMPOSER, user=USER, dest_market=market()
        )

    assert route.strategy == "cctp_compose"
    assert SENTINEL_TEMPLATE in route.dest_program
    assert route.dest_program.hex() in route.steps[1].tx["data"].lower()
    assert route.dest_market is not None and route.dest_market.protocol == "aave_v3"
    assert [leg.kind for leg in route.steps] == ["approve", "bridge_compose"]


@pytest.mark.asyncio
async def test_swap_deposit_route_is_single_execute_leg():
    with patch(_PATCH_SUPPLY, new=AsyncMock(return_value=supply_template())):
        route = await build_cctp_swap_deposit_route(
            cctp=cctp(),
            w3_dst=MagicMock(),
            source_swap=swap_dag(WETH_ETH, USDC_ETH),
            amount_in=10**18,
            vm_address=VM,
            composer=COMPOSER,
            user=USER,
            dest_market=market(),
        )

    assert route.strategy == "cctp_swap_compose"
    assert [leg.kind for leg in route.steps] == ["swap_bridge"]
    leg = route.steps[0]
    assert leg.tx["to"].lower() == VM.to_0x_hex().lower()
    # dest program -> burn template hookData -> source program -> execute calldata.
    assert SENTINEL_TEMPLATE in route.dest_program
    assert route.dest_program.hex() in leg.tx["data"].lower()


# ---------------------------------------------------------------------------
# CCIP routes (same generic builders over a CCIP bridge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ccip_compose_route_native_fee_is_approve_then_send():
    bridge = ccip()
    with patch.object(bridge, "build_bridge_compose_tx", new=AsyncMock(return_value=_SEND_TX)):
        route = await build_compose_route(bridge=bridge, amount_in=_AMOUNT, composer=COMPOSER, dest_program=b"\x00\x01")

    assert route.strategy == "ccip_compose"
    assert [leg.kind for leg in route.steps] == ["approve", "bridge_compose"]
    approve, send = route.steps
    assert approve.tx["to"] == USDC_ETH.address  # approve the bridged token, spender = Router
    assert BRIDGE.hex() in approve.tx["data"].lower()
    assert send.tx == _SEND_TX


@pytest.mark.asyncio
async def test_ccip_compose_route_erc20_fee_adds_fee_approve():
    bridge = ccip(fee_token=LINK)
    with (
        patch.object(bridge, "build_bridge_compose_tx", new=AsyncMock(return_value=_SEND_TX)),
        patch.object(bridge, "quote_fee", new=AsyncMock(return_value=5 * 10**17)),
    ):
        route = await build_compose_route(bridge=bridge, amount_in=_AMOUNT, composer=COMPOSER, dest_program=b"\x00\x01")

    # token approve, then LINK fee approve, then ccipSend.
    assert [leg.kind for leg in route.steps] == ["approve", "approve", "bridge_compose"]
    fee_approve = route.steps[1]
    assert fee_approve.tx["to"] == LINK
    assert BRIDGE.hex() in fee_approve.tx["data"].lower()


@pytest.mark.asyncio
async def test_ccip_deposit_route_compiles_program_and_embeds_supply():
    bridge = ccip()
    with (
        patch(_PATCH_SUPPLY, new=AsyncMock(return_value=supply_template())),
        patch.object(bridge, "build_bridge_compose_tx", new=AsyncMock(return_value=_SEND_TX)),
    ):
        route = await build_deposit_route(
            bridge=bridge, w3_dst=MagicMock(), amount_in=_AMOUNT, composer=COMPOSER, user=USER, dest_market=market()
        )

    assert route.strategy == "ccip_compose"
    assert SENTINEL_TEMPLATE in route.dest_program
    assert route.dest_market is not None and route.dest_market.protocol == "aave_v3"
    assert [leg.kind for leg in route.steps] == ["approve", "bridge_compose"]


# ---------------------------------------------------------------------------
# estimate_cctp_delivered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_cctp_delivered_returns_amount_out():
    bridge = cctp()
    quote = BridgeQuote(
        token_in=USDC_ETH,
        token_out=USDC_ETH,
        amount_in=_AMOUNT,
        amount_out=TokenAmount(USDC_ETH, 999_500_000),  # 1000 USDC minus 0.5 fee
        bridge_fee=TokenAmount(USDC_ETH, 500_000),
        estimated_time_seconds=20,
        protocol="CCTP",
    )
    with patch.object(bridge, "get_quote", new=AsyncMock(return_value=quote)) as mock:
        delivered = await estimate_cctp_delivered(bridge, _AMOUNT)

    assert delivered == 999_500_000
    assert mock.call_args.args[1].chain_id == ChainId.BASE  # priced against destination USDC

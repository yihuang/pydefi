from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.crosschain.compose import (
    AMOUNT_RECEIVED_SLOT,
    CCTP_BURN_AMOUNT_OFFSET,
    _calldata,
    build_compose_deposit_program,
    build_source_swap_bridge_program,
    build_supply_template,
)
from pydefi.pathfinder.dag import RouteDAG
from pydefi.vm.yield_deposit import AMOUNT_SENTINEL
from tests.crosschain_helpers import (
    BRIDGE,
    COMPOSER,
    SENTINEL_TEMPLATE,
    SPENDER,
    USDC_BASE,
    USDC_ETH,
    USER,
    VM,
    WETH_ETH,
    market,
    swap_dag,
)

# ---------------------------------------------------------------------------
# Destination program — build_compose_deposit_program
# ---------------------------------------------------------------------------


def _build_compose(**overrides):
    kwargs = dict(supply_template=SENTINEL_TEMPLATE, protocol=SPENDER, target_token=USDC_ETH.address, composer=COMPOSER)
    kwargs.update(overrides)
    return build_compose_deposit_program(**kwargs)


def test_slot_is_zero():
    # The composer (CCTPComposer.sol) stages amountReceived in transient slot 0.
    assert AMOUNT_RECEIVED_SLOT == 0


def test_no_swap_program_builds_and_embeds_template():
    program = _build_compose()
    assert isinstance(program, bytes) and len(program) > 0
    # call_raw embeds the template verbatim; the amount is patched at runtime.
    assert SENTINEL_TEMPLATE in program


def test_no_swap_program_is_deterministic():
    assert _build_compose() == _build_compose()


def test_min_out_guard_changes_bytecode():
    assert _build_compose(min_out=0) != _build_compose(min_out=10**6)


def test_approve_reset_changes_bytecode():
    assert _build_compose() != _build_compose(approve_reset=True)


def test_with_swap_program_builds_and_embeds_template():
    program = _build_compose(swap_dag=swap_dag(USDC_ETH, WETH_ETH))
    assert SENTINEL_TEMPLATE in program
    assert program != _build_compose()  # the swap leg makes it differ from the no-swap deposit


def test_with_swap_program_is_deterministic():
    assert _build_compose(swap_dag=swap_dag(USDC_ETH, WETH_ETH)) == _build_compose(
        swap_dag=swap_dag(USDC_ETH, WETH_ETH)
    )


def test_empty_swap_dag_rejected():
    with pytest.raises(ValueError, match="no actions"):
        _build_compose(swap_dag=RouteDAG().from_token(USDC_ETH))


def test_template_without_sentinel_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        _build_compose(supply_template=b"\xde\xad\xbe\xef" + b"\x00" * 32)


# ---------------------------------------------------------------------------
# Source program — build_source_swap_bridge_program
# ---------------------------------------------------------------------------

# depositForBurnWithHook calldata: selector + amount word (patched at offset 4) + filler.
_BURN_TEMPLATE = b"\xaa\xbb\xcc\xdd" + b"\x00" * 32 + b"\x11" * 64


def _build_source(**overrides):
    kwargs = dict(
        swap_dag=swap_dag(WETH_ETH, USDC_ETH),
        amount_in=10**18,
        usdc=USDC_ETH.address,
        token_messenger=BRIDGE,
        burn_template=_BURN_TEMPLATE,
        executor=VM,
    )
    kwargs.update(overrides)
    return build_source_swap_bridge_program(**kwargs)


def test_source_program_builds_and_embeds_burn_template():
    program = _build_source()
    assert isinstance(program, bytes) and len(program) > 0
    # call_raw embeds the burn template; the swapped amount is patched at runtime.
    assert _BURN_TEMPLATE in program


def test_source_program_is_deterministic():
    assert _build_source() == _build_source()


def test_min_usdc_out_guard_changes_bytecode():
    assert _build_source(min_usdc_out=0) != _build_source(min_usdc_out=1_000_000)


def test_source_approve_reset_changes_bytecode():
    assert _build_source() != _build_source(approve_reset=True)


def test_source_empty_swap_dag_rejected():
    with pytest.raises(ValueError, match="no actions"):
        _build_source(swap_dag=RouteDAG().from_token(WETH_ETH))


def test_default_burn_amount_offset_is_first_arg():
    assert CCTP_BURN_AMOUNT_OFFSET == 4


def test_out_of_range_burn_offset_rejected():
    with pytest.raises(ValueError, match="out of bounds"):
        _build_source(burn_amount_offset=len(_BURN_TEMPLATE))


# ---------------------------------------------------------------------------
# Supply templates — build_supply_template (per protocol)
# ---------------------------------------------------------------------------

# A supply tx-dict whose calldata carries the amount sentinel exactly once.
_CANNED_TX = {"to": SPENDER, "data": "0x" + SENTINEL_TEMPLATE.hex(), "value": "0"}


def test_calldata_extracts_bytes():
    assert _calldata(_CANNED_TX) == SENTINEL_TEMPLATE


def test_calldata_requires_sentinel():
    with pytest.raises(ValueError, match="exactly once"):
        _calldata({"data": "0x" + (b"\x11\x22\x33\x44" + b"\x00" * 32).hex()})


def _client(**attrs) -> MagicMock:
    client = MagicMock()
    client.build_supply_tx = MagicMock(return_value=_CANNED_TX)
    for name, value in attrs.items():
        setattr(client, name, value)
    return client


@pytest.mark.asyncio
async def test_aave_v3_dispatch_credits_user():
    aave = _client(pool_address=SPENDER)
    with patch("pydefi.crosschain.compose.AaveV3.from_chain", new=AsyncMock(return_value=aave)):
        result = await build_supply_template(market("aave_v3"), MagicMock(), USER)

    assert result.template == SENTINEL_TEMPLATE
    assert result.spender == SPENDER
    assert result.token == USDC_BASE.address
    # Credits the user via on_behalf_of and uses the sentinel amount.
    assert aave.build_supply_tx.call_args.kwargs["on_behalf_of"] == USER
    assert aave.build_supply_tx.call_args.args[1].amount == AMOUNT_SENTINEL


@pytest.mark.asyncio
async def test_compound_v3_dispatch_uses_supply_to():
    comet = _client(comet_address=SPENDER)
    with patch("pydefi.crosschain.compose.CompoundV3.from_chain", new=MagicMock(return_value=comet)):
        result = await build_supply_template(market("compound_v3"), MagicMock(), USER)
    assert result.spender == SPENDER
    assert comet.build_supply_tx.call_args.kwargs["dst"] == USER


@pytest.mark.asyncio
async def test_morpho_dispatch_credits_user():
    morpho = _client(morpho_address=SPENDER)
    with (
        patch("pydefi.crosschain.compose.MorphoBlue.from_chain", new=MagicMock(return_value=morpho)),
        patch("pydefi.crosschain.compose.morpho_params", new=AsyncMock(return_value=MagicMock())),
    ):
        result = await build_supply_template(market("morpho"), MagicMock(), USER)
    assert result.spender == SPENDER
    assert morpho.build_supply_tx.call_args.kwargs["on_behalf_of"] == USER


@pytest.mark.asyncio
async def test_aave_v4_rejected():
    with pytest.raises(ValueError, match="position manager"):
        await build_supply_template(market("aave_v4"), MagicMock(), USER)


@pytest.mark.asyncio
async def test_unknown_protocol_rejected():
    with pytest.raises(ValueError, match="unknown protocol"):
        await build_supply_template(market("bogus"), MagicMock(), USER)

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydefi.types import Address, ChainId, TokenAmount
from pydefi.vm.yield_deposit import (
    AMOUNT_SENTINEL,
    SAFE_PROXY_FACTORY,
    build_deposit_program,
    build_initializer,
    find_amount_offset,
)
from pydefi.yields import YieldMarket, build_yield_deposit
from tests.addrs import COMET_USDC, USDC

_OWNER = Address("0x" + "AA" * 20)
_TOKEN = Address("0x" + "a0" * 20)
_PROTOCOL = Address("0x" + "CE" * 20)
_DEFIVM = Address("0x" + "DD" * 20)
_WORD = AMOUNT_SENTINEL.to_bytes(32, "big")
# Stand-in supply calldata: selector + one arg word + the sentinel amount word.
_TEMPLATE = b"\xde\xad\xbe\xef" + b"\x00" * 28 + _WORD


def test_find_amount_offset():
    assert find_amount_offset(_TEMPLATE) == 32


@pytest.mark.parametrize("template", [b"\x12\x34\x56", 2 * _WORD], ids=["missing", "duplicate"])
def test_find_amount_offset_requires_exactly_one(template):
    with pytest.raises(ValueError, match="exactly once"):
        find_amount_offset(template)


def test_deposit_program_embeds_template_and_is_deterministic():
    program = build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6)
    assert _TEMPLATE in program  # call_raw embeds the template; the amount is patched at runtime
    assert program == build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6)
    # the fee and reset legs change the program (and therefore the deposit address)
    assert program != build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 0)
    assert program != build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6, approve_reset=True)


def test_initializer_commits_owner_and_program():
    """The Safe setup calldata — and therefore the CREATE2 address — binds the
    owner and the exact program."""
    program = build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 0)
    init = build_initializer(_OWNER, _DEFIVM, program)
    assert program in init  # embedded via DeFiVM.execute(program)
    assert bytes(_OWNER) in init
    assert init != build_initializer(Address("0x" + "BB" * 20), _DEFIVM, program)
    assert init != build_initializer(_OWNER, _DEFIVM, program + b"\x00")


# ---------------------------------------------------------------------------
# build_yield_deposit — the YieldMarket-aware wrapper
# ---------------------------------------------------------------------------

_DEPOSIT_ADDR = Address("0x" + "5A" * 20)


def _market(protocol: str) -> YieldMarket:
    return YieldMarket(
        protocol=protocol,
        chain_id=ChainId.ETHEREUM,
        token=USDC,
        supply_apy=Decimal("0.05"),
        utilization=Decimal("0.7"),
        available_liquidity=TokenAmount(USDC, 10**18),
        market_id=f"{protocol}:1:USDC",
    )


def _assert_committed(dep, target: Address) -> None:
    """The deposit commits to the derived address and a program that credits
    the owner, hits *target* (approve + supply), and carries one patch sentinel."""
    assert dep.address == _DEPOSIT_ADDR
    assert bytes(_OWNER) in dep.program
    assert bytes(target) in dep.program
    assert dep.program.count(_WORD) == 1


@pytest.mark.asyncio
async def test_build_yield_deposit_compound():
    """compound_v3 builds genuine Comet.supplyTo(owner) calldata; the sweep
    deploys through the canonical factory carrying the committed program."""
    with patch("pydefi.yields.deposit.deposit_address", new=AsyncMock(return_value=_DEPOSIT_ADDR)) as derive:
        dep = await build_yield_deposit(_OWNER, _market("compound_v3"), _DEFIVM, w3=object(), execution_fee=10**6)

    _assert_committed(dep, COMET_USDC)
    assert Address(dep.sweep_tx["to"]) == SAFE_PROXY_FACTORY
    assert dep.program in bytes.fromhex(dep.sweep_tx["data"][2:])  # committed into the initializer
    # the derived address commits to exactly this (owner, defivm, program, nonce)
    assert derive.await_args.args[1:] == (_OWNER, _DEFIVM, dep.program, 0)


@pytest.mark.asyncio
async def test_build_yield_deposit_aave():
    """aave_v3 resolves the Pool (also the approve target) and builds
    Pool.supply(onBehalfOf=owner) calldata."""
    pool = Address("0x" + "A1" * 20)
    client = MagicMock(pool_address=pool)
    # Stand-in supply calldata: owner word + the sentinel amount find_amount_offset needs.
    client.build_supply_tx = lambda user, amount: {
        "data": "0x" + (bytes(user).rjust(32, b"\x00") + amount.amount.to_bytes(32, "big")).hex()
    }
    with (
        patch("pydefi.yields.deposit.AaveV3") as Aave,
        patch("pydefi.yields.deposit.deposit_address", new=AsyncMock(return_value=_DEPOSIT_ADDR)),
    ):
        Aave.from_chain = AsyncMock(return_value=client)
        dep = await build_yield_deposit(_OWNER, _market("aave_v3"), _DEFIVM, w3=object(), execution_fee=0)

    _assert_committed(dep, pool)


@pytest.mark.asyncio
async def test_build_yield_deposit_rejects_unsupported_protocol():
    """Only onBehalfOf/supplyTo-style protocols can credit the owner from the
    Safe's context."""
    with pytest.raises(ValueError, match="unsupported protocol"):
        await build_yield_deposit(_OWNER, _market("morpho"), _DEFIVM, w3=object(), execution_fee=0)

"""Shared bits for the gasless-deposit fork tests (Permit2 and EIP-7702).

Each path keeps its own ``_setup`` / ``_gasless_deposit``; the market builder,
the sponsor broadcast helper, and the "did the supply credit the owner?"
assertions live here, run by both.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Awaitable, Callable

from web3 import Web3

from pydefi.lending import CompoundV3, MorphoBlue
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.yields import YieldMarket
from tests.addrs import COMET_USDC, MORPHO_CBBTC_USDC, USDC

AMT = 1_000 * 10**6
FUT = 9_999_999_999

# A deposit step: broadcast the (already-signed) route and wait for it to land.
Deposit = Callable[[], Awaitable[object]]


async def send_sponsored(fork_w3, sponsor_acct, tx: dict) -> dict:
    """Locally sign *tx* with the sponsor key and broadcast it — the sponsor pays
    gas. Handles both the type-4 (authorization attached) and plain shapes."""
    full = {
        "to": Web3.to_checksum_address(tx["to"]),
        "data": tx["data"],
        "value": int(tx["value"]),
        "chainId": await fork_w3.eth.chain_id,
        "nonce": await fork_w3.eth.get_transaction_count(sponsor_acct.address),
        "gas": 3_000_000,
        "maxFeePerGas": 200 * 10**9,
        "maxPriorityFeePerGas": 10**9,
    }
    if tx.get("type") == 4:
        full["type"] = 4
        full["authorizationList"] = tx["authorizationList"]
    signed = sponsor_acct.sign_transaction(full)
    tx_hash = await fork_w3.eth.send_raw_transaction(signed.raw_transaction)
    return await fork_w3.eth.wait_for_transaction_receipt(tx_hash)


def market(protocol: str, market_id: str) -> YieldMarket:
    """A minimal USDC :class:`YieldMarket` on Ethereum — APY/utilization are
    placeholders; only ``protocol`` and ``market_id`` drive route building."""
    return YieldMarket(
        protocol=protocol,
        chain_id=ChainId.ETHEREUM,
        token=USDC,
        supply_apy=Decimal("0.05"),
        utilization=Decimal("0.7"),
        available_liquidity=TokenAmount(USDC, 10**18),
        market_id=market_id,
    )


async def assert_compound_credited(fork_w3, owner: Address, deposit: Deposit) -> None:
    """Run *deposit* and assert the owner's Compound V3 USDC supply grew by ~``AMT``."""
    comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
    before = (await comet.get_user_position(owner)).base_supply.amount
    await deposit()
    after = (await comet.get_user_position(owner)).base_supply.amount
    assert after - before >= AMT - 100


async def assert_morpho_credited(fork_w3, owner: Address, deposit: Deposit) -> None:
    """Run *deposit* and assert the owner's Morpho Blue USDC supply grew by ~``AMT``."""
    morpho = MorphoBlue.from_chain(fork_w3, ChainId.ETHEREUM)
    params = await morpho.get_market_params(MORPHO_CBBTC_USDC)
    before = (await morpho.get_position(owner, params)).supply_assets.amount
    await deposit()
    after = (await morpho.get_position(owner, params)).supply_assets.amount
    assert after - before >= AMT - 100

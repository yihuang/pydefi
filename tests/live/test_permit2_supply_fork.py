"""Fork tests for :sol:`Permit2SupplyRouter` — the gasless deposit path.

End-to-end: ``build_yield_route(gasless_router=…)`` → ``sign_route`` → one tx
pulls USDC via a Permit2 witness signature and supplies it on the owner's behalf
(Compound V3, Morpho Blue); a tampered ``supplyData`` reverts.

A fresh keypair signs — Anvil/vitalik accounts carry EIP-7702 delegations that
break Permit2's ECDSA path. Run with::

    pytest -m fork tests/live/test_permit2_supply_fork.py
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError

from pydefi.deployments import get_address
from pydefi.lending import CompoundV3, MorphoBlue
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.vm.permit2_supply import (
    PERMIT2,
    build_prime_tx,
    build_supply_tx,
    build_witness_typed_data,
    random_nonce,
    sign_witness,
)
from pydefi.yields import YieldMarket, build_yield_route, sign_route
from tests.addrs import USDC
from tests.live.anvil_helpers import erc20_approve, fund_usdc, impersonate, send_tx, set_balance
from tests.live.sol_utils import compile_sol_file, deploy

ROUTER_SOL = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "Permit2SupplyRouter.sol"
COMET_USDC = get_address("COMPOUND_V3_USDC", ChainId.ETHEREUM)
MORPHO_CBBTC_USDC = bytes.fromhex("64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64")
AMT = 1_000 * 10**6
FUT = 9_999_999_999


def _market(protocol: str, market_id: str) -> YieldMarket:
    return YieldMarket(
        protocol=protocol,
        chain_id=ChainId.ETHEREUM,
        token=USDC,
        supply_apy=Decimal("0.05"),
        utilization=Decimal("0.7"),
        available_liquidity=TokenAmount(USDC, 10**18),
        market_id=market_id,
    )


async def _setup(fork_w3, protocol_addr: Address):
    """Deploy + prime the router for (USDC, protocol), mint a fresh code-less owner,
    seed USDC, and do the one-time owner ``approve(Permit2)``."""
    deployer = (await fork_w3.eth.accounts)[0]
    router = await deploy(fork_w3, compile_sol_file(ROUTER_SOL, "Permit2SupplyRouter"), deployer)
    await send_tx(fork_w3, deployer, build_prime_tx(router, USDC.address, protocol_addr))

    owner_acct = Account.create()
    owner = Address(owner_acct.address)
    assert len(bytes(await fork_w3.eth.get_code(owner_acct.address))) == 0
    await impersonate(fork_w3, owner)
    await set_balance(fork_w3, owner, 100 * 10**18)
    await fund_usdc(fork_w3, USDC.address, owner, AMT)
    await erc20_approve(fork_w3, USDC.address, owner, Address(PERMIT2), 2**256 - 1)
    return deployer, router, owner_acct, owner


async def _gasless_deposit(fork_w3, deployer, router, owner_acct, owner, market: YieldMarket):
    """build_yield_route → sign_route → broadcast (relayer submits; owner pays no gas)."""
    route = await build_yield_route(
        "supply_then_bridge",
        user=owner,
        amount_in=TokenAmount(USDC, AMT),
        w3s={ChainId.ETHEREUM: fork_w3},
        target_market=market,
        target_chain=ChainId.KITE,
        gasless_router=router,
    )
    assert [s.kind for s in route.steps] == ["supply_with_permit2"]
    signed = sign_route(route, owner_acct.key.hex())
    rc = await send_tx(fork_w3, deployer, signed.steps[0].tx)
    assert rc["status"] == 1, "supply_with_permit2 reverted"


@pytest.mark.fork
class TestPermit2SupplyRouterFork:
    async def test_compound_via_build_yield_route(self, fork_w3):
        ctx = await _setup(fork_w3, COMET_USDC)
        owner = ctx[3]
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        before = (await comet.get_user_position(owner)).base_supply.amount
        await _gasless_deposit(fork_w3, *ctx, _market("compound_v3", "compound_v3:1:USDC"))
        after = (await comet.get_user_position(owner)).base_supply.amount
        assert after - before >= AMT - 100

    async def test_morpho_blue_via_build_yield_route(self, fork_w3):
        morpho = MorphoBlue.from_chain(fork_w3, ChainId.ETHEREUM)
        ctx = await _setup(fork_w3, morpho.morpho_address)
        owner = ctx[3]
        params = await morpho.get_market_params(MORPHO_CBBTC_USDC)
        before = (await morpho.get_position(owner, params)).supply_assets.amount
        await _gasless_deposit(fork_w3, *ctx, _market("morpho", "morpho:1:0x" + MORPHO_CBBTC_USDC.hex()))
        after = (await morpho.get_position(owner, params)).supply_assets.amount
        assert after - before >= AMT - 100

    async def test_tampered_supply_data_reverts(self, fork_w3):
        """Signing for one supplyData but submitting another → witness mismatch →
        Permit2 reverts (eth_call, no state change)."""
        deployer, router, owner_acct, owner = await _setup(fork_w3, COMET_USDC)
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        good = bytes.fromhex(comet.build_supply_tx(TokenAmount(USDC, AMT), dst=owner)["data"][2:])
        evil = bytes.fromhex(comet.build_supply_tx(TokenAmount(USDC, AMT), dst=Address("0x" + "99" * 20))["data"][2:])

        nonce = random_nonce()
        td = build_witness_typed_data(USDC.address, AMT, router, nonce, FUT, COMET_USDC, good, ChainId.ETHEREUM)
        sig = sign_witness(td, owner_acct.key.hex())
        tx = build_supply_tx(router, USDC.address, AMT, nonce, FUT, owner, sig, COMET_USDC, evil)
        with pytest.raises(ContractLogicError):
            await fork_w3.eth.call(
                {
                    "from": Web3.to_checksum_address(deployer),
                    "to": Web3.to_checksum_address(tx["to"]),
                    "data": tx["data"],
                }
            )

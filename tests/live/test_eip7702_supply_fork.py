"""Fork tests for the EIP-7702 gasless deposit path via the Calibur delegate.

End-to-end: ``build_yield_route(delegate=CALIBUR)`` → ``sign_route`` → one type-4
tx, submitted and paid for by a *sponsor*, delegates the owner's EOA to the
already-deployed Calibur singleton and runs a signed ``[approve, supply]`` batch
in the owner's own context (Compound V3, Morpho Blue). The owner spends no gas;
a tampered batch reverts ``InvalidSignature``.

A fresh code-less keypair is the owner; a second funded keypair is the sponsor.
Run with::

    pytest -m fork tests/live/test_eip7702_supply_fork.py
"""

from __future__ import annotations

import pytest
from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError

from pydefi._utils import erc20_approve_tx
from pydefi.lending import CompoundV3
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.vm.eip712 import sign_typed_data
from pydefi.vm.eip7702_supply import (
    CALIBUR,
    build_batch_typed_data,
    build_execute_tx,
    build_revoke_authorization,
    delegation_status,
    is_delegated_to,
)
from pydefi.yields import YieldMarket, build_yield_route, sign_route
from tests.addrs import USDC
from tests.live.anvil_helpers import fund_usdc, impersonate, set_balance
from tests.live.gasless_common import (
    AMT,
    COMET_USDC,
    FUT,
    MORPHO_CBBTC_USDC,
    assert_compound_credited,
    assert_morpho_credited,
    market,
    send_sponsored,
)


async def _setup(fork_w3):
    """Mint a fresh code-less, gas-less owner seeded with USDC and a
    separately-funded sponsor. No contract deployment: the delegate is the
    Calibur singleton already on the mainnet fork. The owner does NO approve —
    that is batched."""
    code = bytes(await fork_w3.eth.get_code(Web3.to_checksum_address(bytes(CALIBUR))))
    assert len(code) > 2, "Calibur singleton not present on this fork"

    owner_acct = Account.create()
    owner = Address(owner_acct.address)
    assert len(bytes(await fork_w3.eth.get_code(owner_acct.address))) == 0
    await impersonate(fork_w3, owner)
    await fund_usdc(fork_w3, USDC.address, owner, AMT)

    sponsor_acct = Account.create()
    await set_balance(fork_w3, Address(sponsor_acct.address), 100 * 10**18)
    return owner_acct, owner, sponsor_acct


async def _gasless_deposit(fork_w3, owner_acct, owner, sponsor_acct, target: YieldMarket):
    """build_yield_route(delegate=CALIBUR) → sign_route → sponsor broadcasts the type-4 tx."""
    route = await build_yield_route(
        "supply_then_bridge",
        user=owner,
        amount_in=TokenAmount(USDC, AMT),
        w3s={ChainId.ETHEREUM: fork_w3},
        target_market=target,
        target_chain=ChainId.KITE,
        delegate=CALIBUR,
    )
    assert [s.kind for s in route.steps] == ["supply_with_7702"]
    signed = sign_route(route, owner_acct.key.hex())
    tx = signed.steps[0].tx
    assert tx["type"] == 4 and len(tx["authorizationList"]) == 1  # first deposit sets the code
    rc = await send_sponsored(fork_w3, sponsor_acct, tx)
    assert rc["status"] == 1, "supply_with_7702 reverted"
    assert await is_delegated_to(fork_w3, owner, CALIBUR)
    assert await fork_w3.eth.get_balance(owner_acct.address) == 0  # owner paid no gas


@pytest.mark.fork
class TestEIP7702SupplyFork:
    async def test_compound_via_build_yield_route(self, fork_w3):
        ctx = await _setup(fork_w3)
        await assert_compound_credited(
            fork_w3, ctx[1], lambda: _gasless_deposit(fork_w3, *ctx, market("compound_v3", "compound_v3:1:USDC"))
        )

    async def test_morpho_blue_via_build_yield_route(self, fork_w3):
        ctx = await _setup(fork_w3)
        await assert_morpho_credited(
            fork_w3,
            ctx[1],
            lambda: _gasless_deposit(fork_w3, *ctx, market("morpho", "morpho:1:0x" + MORPHO_CBBTC_USDC.hex())),
        )

    async def test_revoke_clears_delegation(self, fork_w3):
        """After a deposit delegates the owner, a sponsor-submitted revoke tx
        (authorization to 0x0) leaves the EOA code-less again."""
        owner_acct, owner, sponsor_acct = await _setup(fork_w3)
        await _gasless_deposit(fork_w3, owner_acct, owner, sponsor_acct, market("compound_v3", "compound_v3:1:USDC"))
        assert await is_delegated_to(fork_w3, owner, CALIBUR)

        nonce = await fork_w3.eth.get_transaction_count(owner_acct.address)
        revoke = build_revoke_authorization(owner_acct.key.hex(), await fork_w3.eth.chain_id, nonce)
        rc = await send_sponsored(fork_w3, sponsor_acct, revoke)
        assert rc["status"] == 1, "revoke reverted"
        assert len(bytes(await fork_w3.eth.get_code(owner_acct.address))) == 0
        assert not await is_delegated_to(fork_w3, owner, CALIBUR)

    async def test_tampered_batch_reverts(self, fork_w3):
        """Sign one batch but submit another → Calibur recovers the wrong signer
        → execute reverts InvalidSignature. Run after a real deposit so the EOA
        is delegated (an undelegated EOA has no code to call)."""
        owner_acct, owner, sponsor_acct = await _setup(fork_w3)
        await _gasless_deposit(fork_w3, owner_acct, owner, sponsor_acct, market("compound_v3", "compound_v3:1:USDC"))

        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        approve = erc20_approve_tx(USDC.address, Address(COMET_USDC), AMT)
        good = [approve, comet.build_supply_tx(TokenAmount(USDC, AMT))]
        evil = [approve, comet.build_supply_tx(TokenAmount(USDC, AMT), dst=Address("0x" + "99" * 20))]

        nonce, needs_auth = await delegation_status(fork_w3, owner)
        assert not needs_auth  # delegated by the first deposit
        td = build_batch_typed_data(good, nonce, FUT, owner, ChainId.ETHEREUM)
        sig = sign_typed_data(td, owner_acct.key.hex())
        # Already delegated → no authorization needed; submit the mismatched batch.
        tx = build_execute_tx(owner, evil, nonce, FUT, sig)
        with pytest.raises(ContractLogicError):
            await fork_w3.eth.call(
                {
                    "from": Web3.to_checksum_address(sponsor_acct.address),
                    "to": Web3.to_checksum_address(tx["to"]),
                    "data": tx["data"],
                }
            )

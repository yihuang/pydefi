"""Fork tests for :sol:`DeFiVM.executeWithPermit2` — the Permit2 gasless deposit path.

End-to-end: ``build_yield_route(defivm=…)`` → ``sign_route`` → one tx pulls USDC
via a Permit2 witness signature bound to the compiled ``[approve, supply]``
program and runs it in the VM, crediting the owner (Compound V3, Morpho Blue);
a tampered program reverts.

A fresh keypair signs — Anvil/vitalik accounts carry EIP-7702 delegations that
break Permit2's ECDSA path. Run with::

    pytest -m fork tests/live/test_permit2_supply_fork.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError

from pydefi.lending import CompoundV3
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.vm.eip712 import sign_typed_data
from pydefi.vm.permit2_supply import (
    PERMIT2,
    build_supply_program,
    build_supply_tx,
    build_witness_typed_data,
    random_nonce,
)
from pydefi.yields import YieldMarket, build_yield_route, sign_route
from tests.addrs import USDC
from tests.live.anvil_helpers import erc20_approve, fund_usdc, impersonate, send_tx, set_balance
from tests.live.conftest import _ensure_interpreter
from tests.live.gasless_common import (
    AMT,
    COMET_USDC,
    FUT,
    MORPHO_CBBTC_USDC,
    assert_compound_credited,
    assert_morpho_credited,
    market,
)
from tests.live.sol_utils import compile_sol_file, deploy

DEFI_VM_SOL = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "DeFiVM.sol"


async def _setup(fork_w3):
    """Deploy DeFiVM, mint a fresh code-less owner, seed USDC, and do the
    one-time owner ``approve(Permit2)``. No prime step: the program approves
    the protocol inline."""
    deployer = (await fork_w3.eth.accounts)[0]
    interpreter = await _ensure_interpreter(fork_w3, deployer)
    defivm = await deploy(fork_w3, compile_sol_file(DEFI_VM_SOL, "DeFiVM"), deployer, interpreter)

    owner_acct = Account.create()
    owner = Address(owner_acct.address)
    assert len(bytes(await fork_w3.eth.get_code(owner_acct.address))) == 0
    await impersonate(fork_w3, owner)
    await set_balance(fork_w3, owner, 100 * 10**18)
    await fund_usdc(fork_w3, USDC.address, owner, AMT)
    await erc20_approve(fork_w3, USDC.address, owner, Address(PERMIT2), 2**256 - 1)
    return deployer, defivm, owner_acct, owner


async def _gasless_deposit(fork_w3, deployer, defivm, owner_acct, owner, target: YieldMarket):
    """build_yield_route → sign_route → broadcast (relayer submits; owner pays no gas)."""
    route = await build_yield_route(
        "supply_then_bridge",
        user=owner,
        amount_in=TokenAmount(USDC, AMT),
        w3s={ChainId.ETHEREUM: fork_w3},
        target_market=target,
        target_chain=ChainId.KITE,
        defivm=defivm,
    )
    assert [s.kind for s in route.steps] == ["supply_with_permit2"]
    signed = sign_route(route, owner_acct.key.hex())
    rc = await send_tx(fork_w3, deployer, signed.steps[0].tx)
    assert rc["status"] == 1, "supply_with_permit2 reverted"


@pytest.mark.fork
class TestPermit2SupplyFork:
    async def test_compound_via_build_yield_route(self, fork_w3):
        ctx = await _setup(fork_w3)
        await assert_compound_credited(
            fork_w3, ctx[3], lambda: _gasless_deposit(fork_w3, *ctx, market("compound_v3", "compound_v3:1:USDC"))
        )

    async def test_morpho_blue_via_build_yield_route(self, fork_w3):
        ctx = await _setup(fork_w3)
        await assert_morpho_credited(
            fork_w3,
            ctx[3],
            lambda: _gasless_deposit(fork_w3, *ctx, market("morpho", "morpho:1:0x" + MORPHO_CBBTC_USDC.hex())),
        )

    async def test_tampered_program_reverts(self, fork_w3):
        """Signing the witness for one program but submitting another → witness
        mismatch → Permit2 reverts (eth_call, no state change)."""
        deployer, defivm, owner_acct, owner = await _setup(fork_w3)
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        good_data = bytes.fromhex(comet.build_supply_tx(TokenAmount(USDC, AMT), dst=owner)["data"][2:])
        evil_data = bytes.fromhex(
            comet.build_supply_tx(TokenAmount(USDC, AMT), dst=Address("0x" + "99" * 20))["data"][2:]
        )
        good = build_supply_program(USDC.address, Address(COMET_USDC), good_data)
        evil = build_supply_program(USDC.address, Address(COMET_USDC), evil_data)

        nonce = random_nonce()
        td = build_witness_typed_data([(USDC.address, AMT)], defivm, nonce, FUT, good, ChainId.ETHEREUM)
        sig = sign_typed_data(td, owner_acct.key.hex())
        tx = build_supply_tx(defivm, [(USDC.address, AMT)], nonce, FUT, owner, sig, evil)
        with pytest.raises(ContractLogicError):
            await fork_w3.eth.call(
                {
                    "from": Web3.to_checksum_address(deployer),
                    "to": Web3.to_checksum_address(tx["to"]),
                    "data": tx["data"],
                }
            )

    async def test_multi_token_batch(self, fork_w3):
        """One signature pulls several TokenPermissions entries (here the deposit
        split in two) and funds a single program run."""
        deployer, defivm, owner_acct, owner = await _setup(fork_w3)
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        supply_data = bytes.fromhex(comet.build_supply_tx(TokenAmount(USDC, AMT), dst=owner)["data"][2:])
        program = build_supply_program(USDC.address, Address(COMET_USDC), supply_data)

        permitted = [(USDC.address, AMT // 2), (USDC.address, AMT - AMT // 2)]
        nonce = random_nonce()
        td = build_witness_typed_data(permitted, defivm, nonce, FUT, program, ChainId.ETHEREUM)
        sig = sign_typed_data(td, owner_acct.key.hex())
        tx = build_supply_tx(defivm, permitted, nonce, FUT, owner, sig, program)

        async def deposit():
            rc = await send_tx(fork_w3, deployer, tx)
            assert rc["status"] == 1, "batch executeWithPermit2 reverted"

        await assert_compound_credited(fork_w3, owner, deposit)

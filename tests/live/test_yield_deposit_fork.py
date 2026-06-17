from __future__ import annotations

from pathlib import Path

import pytest
from eth_account import Account
from eth_contract.erc20 import ERC20
from web3 import Web3
from web3.exceptions import ContractLogicError

from pydefi.abi.safe import SAFE
from pydefi.lending import CompoundV3
from pydefi.types import Address, ChainId, TokenAmount
from pydefi.vm.yield_deposit import (
    AMOUNT_SENTINEL,
    build_deposit_program,
    build_sweep_tx,
    deposit_address,
)
from pydefi.yields import build_yield_deposit
from tests.addrs import USDC
from tests.live.anvil_helpers import fund_usdc, send_tx
from tests.live.conftest import _ensure_interpreter
from tests.live.gasless_common import AMT, COMET_USDC, assert_compound_credited, market
from tests.live.sol_utils import compile_sol_file, deploy

DEFI_VM_SOL = Path(__file__).resolve().parents[2] / "pydefi" / "vm" / "DeFiVM.sol"
FEE = 10**6  # 1 USDC to the sweeping relayer


def _supply_template(fork_w3, owner: Address) -> bytes:
    """Compound supplyTo(owner) calldata with the amount sentinel to patch."""
    comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
    return bytes.fromhex(comet.build_supply_tx(TokenAmount(USDC, AMOUNT_SENTINEL), dst=owner)["data"][2:])


async def _usdc_balance(fork_w3, addr) -> int:
    usdc = Web3.to_checksum_address(bytes(USDC.address))
    return await ERC20.fns.balanceOf(bytes(Address(addr))).call(fork_w3, to=usdc)


async def _setup(fork_w3):
    """Deploy DeFiVM; the owner is a fresh keypair with no ETH and no code —
    it only ever receives tokens. The Safe factory/singleton are the canonical
    mainnet deployments already on the fork."""
    relayer = (await fork_w3.eth.accounts)[0]
    interpreter = await _ensure_interpreter(fork_w3, relayer)
    defivm = await deploy(fork_w3, compile_sol_file(DEFI_VM_SOL, "DeFiVM"), relayer, interpreter)
    owner = Address(Account.create().address)
    program = build_deposit_program(USDC.address, Address(COMET_USDC), _supply_template(fork_w3, owner), FEE)
    deposit = await deposit_address(fork_w3, owner, defivm, program)
    assert not bytes(await fork_w3.eth.get_code(Web3.to_checksum_address(bytes(deposit))))
    return relayer, defivm, owner, program, deposit


@pytest.mark.fork
class TestYieldDepositFork:
    async def test_transfer_then_sweep_credits_owner(self, fork_w3):
        relayer, defivm, owner, program, deposit = await _setup(fork_w3)

        async def sweep(salt_nonce=0):
            rc = await send_tx(fork_w3, relayer, build_sweep_tx(owner, defivm, program, salt_nonce))
            assert rc["status"] == 1, "sweep reverted"

        # "CEX withdrawal": a plain transfer to the counterfactual address.
        await fund_usdc(fork_w3, USDC.address, deposit, AMT + FEE)
        fee_before = await _usdc_balance(fork_w3, relayer)
        await assert_compound_credited(fork_w3, owner, sweep)
        assert await _usdc_balance(fork_w3, relayer) - fee_before == FEE  # tx.origin earned the fee

        # The owner ends up owning the deployed Safe (structural escape hatch).
        owners = await SAFE.fns.getOwners().call(fork_w3, to=Web3.to_checksum_address(bytes(deposit)))
        assert [Address(o) for o in owners] == [owner]

        # Recurring inflows rotate salt_nonce: a fresh single-shot address.
        deposit2 = await deposit_address(fork_w3, owner, defivm, program, salt_nonce=1)
        assert deposit2 != deposit
        await fund_usdc(fork_w3, USDC.address, deposit2, AMT + FEE)
        await assert_compound_credited(fork_w3, owner, lambda: sweep(salt_nonce=1))

    async def test_unfunded_or_tampered_program_reverts(self, fork_w3):
        """A tampered program (different beneficiary) derives a different —
        unfunded — address, so its deploy reverts in the balance check; the
        funded address can only ever run the committed program."""
        relayer, defivm, owner, program, deposit = await _setup(fork_w3)
        await fund_usdc(fork_w3, USDC.address, deposit, AMT + FEE)

        evil_owner = Address("0x" + "99" * 20)
        evil = build_deposit_program(USDC.address, Address(COMET_USDC), _supply_template(fork_w3, evil_owner), FEE)
        assert await deposit_address(fork_w3, evil_owner, defivm, evil) != deposit
        tx = build_sweep_tx(evil_owner, defivm, evil)  # deploys the OTHER, unfunded address
        with pytest.raises(ContractLogicError):
            await fork_w3.eth.call(
                {
                    "from": Web3.to_checksum_address(relayer),
                    "to": Web3.to_checksum_address(tx["to"]),
                    "data": tx["data"],
                }
            )

    async def test_build_yield_deposit_market_wrapper(self, fork_w3):
        """The YieldMarket-aware wrapper end to end: build_yield_deposit gives
        out an address; a plain transfer + the returned sweep_tx credits the
        owner's Compound position and pays the relayer the committed fee."""
        relayer = (await fork_w3.eth.accounts)[0]
        interpreter = await _ensure_interpreter(fork_w3, relayer)
        defivm = await deploy(fork_w3, compile_sol_file(DEFI_VM_SOL, "DeFiVM"), relayer, interpreter)
        owner = Address(Account.create().address)

        dep = await build_yield_deposit(owner, market("compound_v3", "compound_v3:1:USDC"), defivm, fork_w3, FEE)
        assert not bytes(await fork_w3.eth.get_code(Web3.to_checksum_address(bytes(dep.address))))

        await fund_usdc(fork_w3, USDC.address, dep.address, AMT + FEE)
        fee_before = await _usdc_balance(fork_w3, relayer)

        async def sweep():
            rc = await send_tx(fork_w3, relayer, dep.sweep_tx)
            assert rc["status"] == 1, "sweep reverted"

        await assert_compound_credited(fork_w3, owner, sweep)
        assert await _usdc_balance(fork_w3, relayer) - fee_before == FEE

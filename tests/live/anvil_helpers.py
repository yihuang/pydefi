"""Shared helpers for Anvil mainnet-fork tests.

Every lending fork test repeats the same boilerplate — impersonate the
whale, top up its ETH balance, wrap some ETH, approve a spender, send
a tx, wait for the receipt — so it lives here in one place. Import the
helpers you need from this module instead of copy-pasting them into
each test file.
"""

from __future__ import annotations

from eth_contract import Contract
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3, Web3

from pydefi._utils import erc20_approve_tx
from pydefi.types import Address
from pydefi.vm.permit2_supply import PERMIT2
from tests.addrs import USDC_WHALE

# WETH9 ``deposit()`` is not on the ERC-20 ABI; declare it once here.
_WETH9 = Contract.from_abi(["function deposit() external payable"])

_PERMIT2 = Contract.from_abi(["function approve(address token, address spender, uint160 amount, uint48 expiration)"])


async def impersonate(w3: AsyncWeb3, address: Address) -> None:
    """``anvil_impersonateAccount`` wrapper."""
    await w3.provider.make_request("anvil_impersonateAccount", [Web3.to_checksum_address(address)])


async def set_balance(w3: AsyncWeb3, address: Address, amount_wei: int) -> None:
    """``anvil_setBalance`` wrapper — top whale up so gas isn't a problem."""
    await w3.provider.make_request("anvil_setBalance", [Web3.to_checksum_address(address), hex(amount_wei)])


async def wrap_eth(w3: AsyncWeb3, sender: Address, weth: Address, amount: int) -> None:
    """Wrap ``amount`` of native ETH via ``WETH9.deposit{value: amount}()``."""
    tx_hash = await w3.eth.send_transaction(
        {
            "to": Web3.to_checksum_address(weth),
            "from": Web3.to_checksum_address(sender),
            "value": amount,
            "data": "0x" + _WETH9.fns.deposit().data.hex(),
        }
    )
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt["status"] == 1, "WETH deposit reverted"


async def erc20_approve(w3: AsyncWeb3, token: Address, owner: Address, spender: Address, amount: int) -> None:
    """``IERC20(token).approve(spender, amount)`` sent from *owner*."""
    tx_hash = await w3.eth.send_transaction(
        {
            "to": Web3.to_checksum_address(token),
            "from": Web3.to_checksum_address(owner),
            "data": erc20_approve_tx(token, spender, amount)["data"],
        }
    )
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt["status"] == 1, "ERC20 approve reverted"


async def permit2_approve(w3: AsyncWeb3, owner: Address, token: Address, spender: Address) -> None:
    """Max-approve *token* to Permit2, then grant *spender* a max Permit2 allowance for it."""
    permit2 = Address(PERMIT2)
    await erc20_approve(w3, token, owner, permit2, (1 << 256) - 1)
    call = _PERMIT2.fns.approve(token, spender, (1 << 160) - 1, (1 << 48) - 1).data
    await send_ok(w3, owner, {"to": permit2, "data": "0x" + call.hex(), "value": 0}, "Permit2.approve")


async def send_tx(w3: AsyncWeb3, sender: Address, tx: dict) -> dict:
    """Submit a pydefi-shaped tx dict ``{to, data, value}`` and return its receipt."""
    tx_hash = await w3.eth.send_transaction(
        {
            "to": Web3.to_checksum_address(tx["to"]),
            "from": Web3.to_checksum_address(sender),
            "data": tx["data"],
            "value": int(tx["value"]),
        }
    )
    return await w3.eth.wait_for_transaction_receipt(tx_hash)


async def send_ok(w3: AsyncWeb3, sender: Address, tx: dict, label: str) -> dict:
    """Submit *tx* via :func:`send_tx` and assert it did not revert.

    *label* names the action in the ``"<label> reverted"`` failure message.
    """
    receipt = await send_tx(w3, sender, tx)
    assert receipt["status"] == 1, f"{label} reverted"
    return receipt


async def fund_usdc(w3: AsyncWeb3, usdc: Address, recipient: Address, amount: int) -> None:
    """Send ``amount`` of USDC to *recipient* by impersonating :data:`USDC_WHALE`."""
    await impersonate(w3, USDC_WHALE)
    await set_balance(w3, USDC_WHALE, 10**18)  # gas
    call = ERC20.fns.transfer(recipient, amount).data
    tx_hash = await w3.eth.send_transaction(
        {
            "to": Web3.to_checksum_address(usdc),
            "from": Web3.to_checksum_address(USDC_WHALE),
            "data": "0x" + call.hex(),
        }
    )
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    assert receipt["status"] == 1, "USDC fund transfer reverted"

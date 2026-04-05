"""ApproveProxy — Python helpers for the ApproveProxy contract.

The :class:`ApproveProxy` class provides:

* Solidity source and ABI for the ``ApproveProxy`` contract.
* A helper for building the ``Deposit`` list passed to
  ``ApproveProxy.execute(program, deposits)``.

Typical usage::

    from pydefi.vm.approve_proxy import ApproveProxy
    from pydefi.vm import Program
    from eth_contract.erc20 import ERC20

    # --- Deploy (once) ---
    proxy_address = await ApproveProxy.deploy(w3, deployer, vm_address)

    # --- Off-chain: user approves ERC-20 to the proxy ---
    # await token.functions.approve(proxy_address, amount).transact({"from": user})

    # --- Build a program that works with tokens already held by DeFiVM ---
    program = (
        Program()
        .call_contract(token_address, ERC20.fns.transfer(recipient, amount).data)
        .pop()
        .build()
    )

    # --- Execute through the proxy, depositing tokens into DeFiVM first ---
    deposits = [{"token": token_address, "amount": amount}]
    await proxy.functions.execute(program, deposits).transact({"from": user})
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Solidity source and ABI
# ---------------------------------------------------------------------------

#: Absolute path to the ApproveProxy Solidity source file.
SOL_FILE = Path(__file__).with_name("ApproveProxy.sol")

#: Minimal ABI for the deployed ``ApproveProxy`` contract.
ABI = [
    {
        "type": "constructor",
        "inputs": [{"name": "_vm", "type": "address"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "execute",
        "inputs": [
            {"name": "program", "type": "bytes"},
            {
                "name": "deposits",
                "type": "tuple[]",
                "components": [
                    {"name": "token", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
            },
        ],
        "outputs": [],
        "stateMutability": "payable",
    },
    {
        "type": "function",
        "name": "vm",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
]


def _compile(solc_version: str = "0.8.24") -> dict:
    """Compile ``ApproveProxy.sol`` and return ``{"abi": …, "bin": …}``."""
    try:
        import solcx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("py-solc-x is required to compile ApproveProxy. Install the 'dev' extras.") from exc

    if solc_version not in solcx.get_installed_solc_versions():
        solcx.install_solc(solc_version, show_progress=False)

    result = solcx.compile_files(
        [str(SOL_FILE)],
        output_values=["abi", "bin"],
        solc_version=solc_version,
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(":ApproveProxy"))
    return result[key]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ApproveProxy:
    """Helper for the ``ApproveProxy`` Solidity contract.

    This class provides class-level utilities for deploying and interacting
    with ``ApproveProxy``.  It is not meant to be instantiated directly;
    use the class methods instead.

    Security model
    --------------
    ``ApproveProxy`` is paired 1-to-1 with a ``DeFiVM`` instance.  Users:

    1. Call ``token.approve(proxy_address, amount)`` — grant the proxy ERC-20
       allowance.
    2. Call ``proxy.execute(program, deposits)`` where ``deposits`` is a list of
       ``{"token": address, "amount": int}`` dicts.  For each deposit, the proxy
       calls ``token.transferFrom(user, vm, amount)``, moving tokens directly
       into DeFiVM before running the program.
    3. The DeFiVM program operates on tokens already held by DeFiVM (e.g. calls
       ``token.transfer(recipient, amount)`` from the VM's balance).

    Because the proxy only performs ``transferFrom(msg.sender → vm)`` — with the
    amounts explicitly declared by the caller — there is no shared mutable state
    and no reentrancy risk.
    """

    @classmethod
    def compile(cls, solc_version: str = "0.8.24") -> dict:
        """Compile ``ApproveProxy.sol`` and return the ABI + bytecode dict."""
        return _compile(solc_version)

    @classmethod
    async def deploy(cls, w3, deployer: str, vm_address: str, solc_version: str = "0.8.24") -> str:
        """Compile and deploy ``ApproveProxy`` on *w3*, returning the contract address.

        Args:
            w3:          An :class:`web3.AsyncWeb3` (or ``Web3``) instance.
            deployer:    Address used to send the deployment transaction.
            vm_address:  Address of the paired ``DeFiVM`` contract.
            solc_version: Solidity compiler version to use (default ``"0.8.24"``).

        Returns:
            Checksummed address of the newly-deployed ``ApproveProxy``.
        """
        compiled = cls.compile(solc_version)
        contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
        tx_hash = await contract.constructor(vm_address).transact({"from": deployer})
        receipt = await w3.eth.get_transaction_receipt(tx_hash)
        return receipt["contractAddress"]

    @classmethod
    def contract(cls, w3, address: str):
        """Return a web3 ``Contract`` instance for an already-deployed proxy.

        Args:
            w3:      A :class:`web3.AsyncWeb3` (or ``Web3``) instance.
            address: Address of the deployed ``ApproveProxy``.

        Returns:
            A :class:`web3.contract.Contract` bound to *address*.
        """
        return w3.eth.contract(address=address, abi=ABI)

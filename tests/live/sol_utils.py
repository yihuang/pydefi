"""Shared Solidity compile and deploy helpers for DeFiVM fork tests.

These utilities are used by multiple fork test files to avoid duplication.
"""

from __future__ import annotations

from pathlib import Path

import solcx
from web3 import AsyncWeb3


def ensure_solc(version: str = "0.8.24") -> None:
    """Install *version* of solc once (no-op if already installed)."""
    if version not in solcx.get_installed_solc_versions():
        solcx.install_solc(version, show_progress=False)


def compile_sol_file(path: Path, contract_name: str) -> dict:
    """Compile a Solidity file and return ABI + bytecode for *contract_name*."""
    ensure_solc("0.8.24")
    result = solcx.compile_files(
        [str(path)],
        output_values=["abi", "bin"],
        solc_version="0.8.24",
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(f":{contract_name}"))
    return result[key]


def compile_sol_source(source: str, contract_name: str) -> dict:
    """Compile an inline Solidity source string and return ABI + bytecode."""
    ensure_solc("0.8.24")
    result = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version="0.8.24",
    )
    return result[f"<stdin>:{contract_name}"]


async def deploy(w3: AsyncWeb3, compiled: dict, deployer: str, *args) -> str:
    """Deploy a contract and return its address."""
    contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
    tx_hash = await contract.constructor(*args).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# Shared MockToken Solidity source
# ---------------------------------------------------------------------------

MOCK_TOKEN_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal mintable ERC-20 token used in tests.
contract MockToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "MockToken: insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "MockToken: insufficient balance");
        require(allowance[from][msg.sender] >= amount, "MockToken: insufficient allowance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }
}
"""

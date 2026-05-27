"""Deploy ``PatchedInterpreter`` at its deterministic CREATE2 address.

Reads constants from ``pydefi/vm/PatchedInterpreterConstants.sol`` and calls
``UniversalFactory.create2(SALT, CREATION_CODE)``.  Idempotent — if the
address already has matching codehash, exits without sending a tx.  Pass
``--dry-run`` to simulate.

Budget ~4M gas (interpreter runtime is ~8 KB).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from eth_account import Account
from eth_typing import ChecksumAddress
from eth_utils import keccak
from web3 import Web3

_CONSTANTS_SOL = Path(__file__).resolve().parents[1] / "pydefi" / "vm" / "PatchedInterpreterConstants.sol"

_PATTERNS = {
    "address": r"constant\s+{name}\s*=\s*(0x[0-9a-fA-F]{{40}})",
    "bytes32": r"constant\s+{name}\s*=\s*\n?\s*(0x[0-9a-fA-F]{{64}})",
    "bytes": r'constant\s+{name}\s*=\s*hex"([0-9a-fA-F]+)"',
}

_FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "salt", "type": "bytes32"},
            {"internalType": "bytes", "name": "creationCode", "type": "bytes"},
        ],
        "name": "create2",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "payable",
        "type": "function",
    }
]


def _read(src: str, name: str, kind: str) -> str:
    m = re.search(_PATTERNS[kind].format(name=re.escape(name)), src)
    if not m:
        raise ValueError(f"could not find {name} ({kind}) in {_CONSTANTS_SOL}")
    return m.group(1)


def _codehash(w3: Web3, addr: ChecksumAddress) -> str:
    """Return ``0x``-prefixed hex codehash, or empty string if no code."""
    code = w3.eth.get_code(addr)
    return "0x" + keccak(code).hex() if code else ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rpc", default=os.environ.get("ETH_RPC_URL"), help="RPC URL (or env ETH_RPC_URL)")
    p.add_argument(
        "--pk",
        default=os.environ.get("DEPLOYER_PRIVATE_KEY"),
        help="Deployer private key (or env DEPLOYER_PRIVATE_KEY)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan without sending tx")
    p.add_argument("--gas-limit", type=int, default=5_000_000, help="Gas limit for the CREATE2 tx (default: 5,000,000)")
    args = p.parse_args()

    if not args.rpc:
        sys.exit("Missing --rpc (or ETH_RPC_URL)")
    if not args.dry_run and not args.pk:
        sys.exit("Missing --pk (or DEPLOYER_PRIVATE_KEY); pass --dry-run to skip")

    src = _CONSTANTS_SOL.read_text()
    factory_addr = Web3.to_checksum_address(_read(src, "PATCHED_INTERPRETER_CREATE2_DEPLOYER", "address"))
    expected_addr = Web3.to_checksum_address(_read(src, "PATCHED_INTERPRETER_ADDRESS", "address"))
    expected_codehash = _read(src, "PATCHED_INTERPRETER_CODEHASH", "bytes32")
    salt = bytes.fromhex(_read(src, "PATCHED_INTERPRETER_CREATE2_SALT", "bytes32")[2:])
    creation = bytes.fromhex(_read(src, "PATCHED_INTERPRETER_CREATION_CODE", "bytes"))

    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        sys.exit(f"Could not connect to {args.rpc}")
    chain_id = w3.eth.chain_id

    print(f"Chain ID:                {chain_id}")
    print(f"UniversalFactory:        {factory_addr}")
    print(f"Expected interpreter:    {expected_addr}")
    print(f"Expected codehash:       {expected_codehash}")
    print(f"Creation code length:    {len(creation)} bytes")
    print()

    # Idempotency: if the deterministic address already has matching codehash,
    # the deployment is byte-for-byte identical and we're done.
    actual = _codehash(w3, expected_addr)
    if actual.lower() == expected_codehash.lower():
        print(f"Already deployed at {expected_addr} with matching codehash. Nothing to do.")
        return 0
    if actual:
        sys.exit(
            f"FATAL: {expected_addr} already has code but codehash mismatches.\n"
            f"  expected: {expected_codehash}\n"
            f"  actual  : {actual}\n"
            f"Salt collided with a different deployment on this chain. "
            f"Bump the salt preimage version in patch_interpreter.py."
        )

    if not w3.eth.get_code(factory_addr):
        sys.exit(
            f"FATAL: UniversalFactory has no code at {factory_addr} on chain {chain_id}. "
            f"Deploy that first — see https://github.com/Analog-Labs/universal-factory"
        )

    call = w3.eth.contract(address=factory_addr, abi=_FACTORY_ABI).functions.create2(salt, creation)

    if args.dry_run:
        try:
            simulated = call.call({"gas": args.gas_limit})
        except Exception as e:
            sys.exit(f"Dry-run reverted: {e}")
        print(f"Dry-run simulation returned address: {simulated}")
        if Web3.to_checksum_address(simulated) != expected_addr:
            sys.exit(
                f"FATAL: simulated {simulated} != expected {expected_addr}. "
                f"Constants.sol drifted — regenerate with "
                f"`python scripts/patch_interpreter.py --emit`."
            )
        print("Address matches expected. Dry-run OK.")
        return 0

    acct = Account.from_key(args.pk)
    print(f"Deployer:                {acct.address}")
    print(f"Deployer balance:        {w3.from_wei(w3.eth.get_balance(acct.address), 'ether')} ETH")
    print()

    tx = call.build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": args.gas_limit,
            "gasPrice": w3.eth.gas_price,
            "chainId": chain_id,
        }
    )
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Sent tx: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt.status != 1:
        sys.exit(f"FATAL: deployment tx reverted (status={receipt.status})")

    actual = _codehash(w3, expected_addr)
    if not actual:
        sys.exit(f"FATAL: no code at {expected_addr} after tx {tx_hash.hex()}")
    if actual.lower() != expected_codehash.lower():
        sys.exit(
            f"FATAL: codehash mismatch after deploy.\n"
            f"  expected: {expected_codehash}\n"
            f"  actual  : {actual}\n"
            f"Check the constants are up-to-date with the script."
        )

    print()
    print(f"Deployed to:             {expected_addr}")
    print(f"Codehash verified:       {expected_codehash}")
    print(f"Gas used:                {receipt.gasUsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

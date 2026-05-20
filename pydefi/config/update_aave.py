#!/usr/bin/env python3
"""Generate config/aave.json — the Aave V3 ``PoolAddressesProvider`` per chain.

It is the only Aave V3 address pydefi pins; ``Pool``, ``DataProvider`` and
``Oracle`` are resolved from it on-chain (see ``AaveV3.from_addresses_provider``).

Run via update.sh. Stdlib only.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com/aave-dao/aave-address-book/main/src"
_AAVE_JSON = Path(__file__).resolve().parent / "aave.json"

# pydefi registry name -> Solidity constant in the AaveV3<Chain> library.
_NAME = "AAVE_V3_ADDRESSES_PROVIDER"
_CONST = "POOL_ADDRESSES_PROVIDER"

# pydefi chain id -> aave-address-book AaveV3<Chain>.sol file stem.
_FILE_BY_CHAIN: dict[int, str] = {
    1: "AaveV3Ethereum",
    10: "AaveV3Optimism",
    56: "AaveV3BNB",
    100: "AaveV3Gnosis",
    137: "AaveV3Polygon",
    324: "AaveV3ZkSync",
    8453: "AaveV3Base",
    42161: "AaveV3Arbitrum",
    43114: "AaveV3Avalanche",
    59144: "AaveV3Linea",
    534352: "AaveV3Scroll",
    11155111: "AaveV3Sepolia",
}


def _fetch(stem: str) -> str:
    with urllib.request.urlopen(f"{_RAW}/{stem}.sol", timeout=30) as resp:
        return resp.read().decode()


def _extract(sol: str, const: str) -> str:
    """Pull the address a Solidity ``constant`` is assigned. Declarations may
    span lines (``NAME =\\n    IPool(0x...)``); the search stops at the first
    address before the statement-ending ``;``."""
    match = re.search(const + r"\b[^;]*?(0x[0-9a-fA-F]{40})", sol)
    if match is None:
        raise SystemExit(f"update_aave: constant {const!r} not found upstream")
    return match.group(1)


def main() -> None:
    provider: dict[str, str] = {}
    for chain_id, stem in sorted(_FILE_BY_CHAIN.items()):
        provider[str(chain_id)] = _extract(_fetch(stem), _CONST)

    _AAVE_JSON.write_text(json.dumps({_NAME: provider}, indent=2, sort_keys=True) + "\n")
    print(f"update_aave: wrote {_AAVE_JSON.name} — {_NAME} across {len(_FILE_BY_CHAIN)} chains")


if __name__ == "__main__":
    main()

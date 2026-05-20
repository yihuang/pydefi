#!/usr/bin/env python3
"""Generate config/aave.json — Aave V3 core addresses from aave-address-book.

Run via update.sh. Stdlib only.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com/aave-dao/aave-address-book/main/src"
_AAVE_JSON = Path(__file__).resolve().parent / "aave.json"

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
}

# pydefi registry name -> Solidity constant in the AaveV3<Chain> library.
_CONST_BY_NAME: dict[str, str] = {
    "AAVE_V3_POOL": "POOL",
    "AAVE_V3_DATA_PROVIDER": "AAVE_PROTOCOL_DATA_PROVIDER",
    "AAVE_V3_ADDRESSES_PROVIDER": "POOL_ADDRESSES_PROVIDER",
    "AAVE_V3_ORACLE": "ORACLE",
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
    aave: dict[str, dict[str, str]] = {name: {} for name in _CONST_BY_NAME}
    for chain_id, stem in sorted(_FILE_BY_CHAIN.items()):
        sol = _fetch(stem)
        for name, const in _CONST_BY_NAME.items():
            aave[name][str(chain_id)] = _extract(sol, const)

    _AAVE_JSON.write_text(json.dumps(aave, indent=2, sort_keys=True) + "\n")
    print(f"update_aave: wrote {_AAVE_JSON.name} — {len(_CONST_BY_NAME)} contracts across {len(_FILE_BY_CHAIN)} chains")


if __name__ == "__main__":
    main()

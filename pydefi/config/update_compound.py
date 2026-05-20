#!/usr/bin/env python3
"""Generate config/compound.json — Compound III (Comet) market addresses.

Reads each market's ``deployments/<network>/<asset>/roots.json`` from
compound-finance/comet and takes the ``comet`` proxy address. Run via
update.sh. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com/compound-finance/comet/main/deployments"
_COMPOUND_JSON = Path(__file__).resolve().parent / "compound.json"

# pydefi chain id -> comet repo network directory.
_NETWORK_BY_CHAIN: dict[int, str] = {
    1: "mainnet",
    10: "optimism",
    137: "polygon",
    8453: "base",
    42161: "arbitrum",
}

# registry name -> (comet asset directory, chain ids the market is deployed on).
_MARKETS: dict[str, tuple[str, list[int]]] = {
    "COMPOUND_V3_USDC": ("usdc", [1, 10, 137, 8453, 42161]),
    "COMPOUND_V3_WETH": ("weth", [1, 10, 8453, 42161]),
    "COMPOUND_V3_USDT": ("usdt", [1, 42161]),
}


def _comet_address(network: str, asset: str) -> str:
    url = f"{_RAW}/{network}/{asset}/roots.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        roots = json.loads(resp.read().decode())
    comet = roots.get("comet")
    if not comet:
        raise SystemExit(f"update_compound: no 'comet' key in {url}")
    return comet


def main() -> None:
    compound: dict[str, dict[str, str]] = {}
    for name, (asset, chain_ids) in _MARKETS.items():
        compound[name] = {
            str(chain_id): _comet_address(_NETWORK_BY_CHAIN[chain_id], asset) for chain_id in sorted(chain_ids)
        }

    _COMPOUND_JSON.write_text(json.dumps(compound, indent=2, sort_keys=True) + "\n")
    deployments = sum(len(v) for v in compound.values())
    print(f"update_compound: wrote {_COMPOUND_JSON.name} — {len(compound)} markets, {deployments} deployments")


if __name__ == "__main__":
    main()

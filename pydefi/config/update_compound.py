#!/usr/bin/env python3
"""Generate config/compound.json — Compound III (Comet) market addresses.

For each (chain, base asset) it reads ``deployments/<network>/<asset>/roots.json``
from compound-finance/comet and takes the ``comet`` proxy address. Markets the
repo does not have for a given chain are skipped. Run via update.sh. Stdlib only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com/compound-finance/comet/main/deployments"
_COMPOUND_JSON = Path(__file__).resolve().parent / "compound.json"

# pydefi chain id -> comet repo network directory. Chains where pydefi and
# Compound III overlap (testnets included); the repo carries more.
_NETWORK_BY_CHAIN: dict[int, str] = {
    1: "mainnet",
    10: "optimism",
    137: "polygon",
    8453: "base",
    42161: "arbitrum",
    59144: "linea",
    534352: "scroll",
    11155111: "sepolia",
}

# comet asset directory -> pydefi registry name.
_NAME_BY_ASSET: dict[str, str] = {
    "usdc": "COMPOUND_V3_USDC",
    "weth": "COMPOUND_V3_WETH",
    "usdt": "COMPOUND_V3_USDT",
}


def _comet_address(network: str, asset: str) -> str | None:
    """Comet proxy address for a market, or None when the repo has no such
    (network, asset) deployment."""
    url = f"{_RAW}/{network}/{asset}/roots.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            roots = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    comet = roots.get("comet")
    if not comet:
        raise SystemExit(f"update_compound: no 'comet' key in {url}")
    return comet


def main() -> None:
    compound: dict[str, dict[str, str]] = {name: {} for name in _NAME_BY_ASSET.values()}
    for chain_id, network in sorted(_NETWORK_BY_CHAIN.items()):
        for asset, name in _NAME_BY_ASSET.items():
            address = _comet_address(network, asset)
            if address is not None:
                compound[name][str(chain_id)] = address

    compound = {name: chains for name, chains in compound.items() if chains}
    _COMPOUND_JSON.write_text(json.dumps(compound, indent=2, sort_keys=True) + "\n")
    deployments = sum(len(v) for v in compound.values())
    print(f"update_compound: wrote {_COMPOUND_JSON.name} — {len(compound)} markets, {deployments} deployments")


if __name__ == "__main__":
    main()

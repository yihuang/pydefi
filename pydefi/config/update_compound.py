#!/usr/bin/env python3
"""Generate config/compound.json — Compound III (Comet) market addresses.

Walks every market under comet/deployments and writes
``COMPOUND_V3_<SYMBOL>`` -> {chain id: comet proxy address}.

Run via update.sh. Stdlib only.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com/compound-finance/comet/main"
_API_DEPLOYMENTS = "https://api.github.com/repos/compound-finance/comet/contents/deployments"
_COMPOUND_JSON = Path(__file__).resolve().parent / "compound.json"


def _fetch_text(url: str) -> str:
    try:
        proc = subprocess.run(["curl", "-fsSL", url], check=True, capture_output=True, text=True)
        return proc.stdout
    except Exception:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode()


def _fetch_json(url: str) -> dict | list:
    return json.loads(_fetch_text(url))


def _fetch_object(url: str) -> dict | None:
    try:
        data = _fetch_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not isinstance(data, dict):
        raise SystemExit(f"update_compound: expected a JSON object at {url}")
    return data


def _list_subdirs(api_url: str) -> list[str]:
    """Sorted subdirectory names at a GitHub contents *api_url*."""
    items = _fetch_json(api_url)
    if not isinstance(items, list):
        raise SystemExit(f"update_compound: unexpected GitHub API response from {api_url}: {items}")
    return sorted(item["name"] for item in items if item["type"] == "dir")


def _network_chain_ids() -> dict[str, int]:
    """Map comet network name -> chain id, parsed from hardhat.config.ts."""
    text = _fetch_text(f"{_RAW}/hardhat.config.ts")
    pairs = re.findall(r"network\s*:\s*'([^']+)'\s*,\s*chainId\s*:\s*(\d+)", text)
    if not pairs:
        raise SystemExit("update_compound: failed to parse networkConfigs from hardhat.config.ts")
    return {network: int(chain_id) for network, chain_id in pairs}


def _contract_name(base_token: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9]+", "_", base_token.upper()).strip("_")
    return f"COMPOUND_V3_{symbol}"


def main() -> None:
    network_chain = _network_chain_ids()

    compound: dict[str, dict[str, str]] = {}
    for network in _list_subdirs(_API_DEPLOYMENTS):
        chain_id = network_chain.get(network)
        if chain_id is None:
            continue
        chain_key = str(chain_id)

        for asset in _list_subdirs(f"{_API_DEPLOYMENTS}/{network}"):
            market = f"{_RAW}/deployments/{network}/{asset}"
            cfg = _fetch_object(f"{market}/configuration.json")
            roots = _fetch_object(f"{market}/roots.json")
            if cfg is None or roots is None:
                continue

            base_token = cfg.get("baseToken")
            comet = roots.get("comet")
            if not base_token:
                raise SystemExit(f"update_compound: missing baseToken in {market}/configuration.json")
            if not comet:
                raise SystemExit(f"update_compound: no 'comet' key in {market}/roots.json")

            name = _contract_name(base_token)
            chains = compound.setdefault(name, {})
            existing = chains.get(chain_key)
            if existing and existing.lower() != comet.lower():
                raise SystemExit(
                    f"update_compound: conflicting addresses for {name} on chain {chain_id}: "
                    f"{existing} vs {comet} ({network}/{asset})"
                )
            chains[chain_key] = comet

    _COMPOUND_JSON.write_text(json.dumps(compound, indent=2, sort_keys=True) + "\n")
    deployments = sum(len(v) for v in compound.values())
    print(f"update_compound: wrote {_COMPOUND_JSON.name} — {len(compound)} markets, {deployments} deployments")


if __name__ == "__main__":
    main()

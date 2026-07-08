#!/usr/bin/env python3
"""Generate config/babylon.json from Babylon's TBV contract-addresses docs.

Discovers the per-network ``*-info/contract-addresses.mdx`` docs in
``babylonlabs-io/babylonlabs.github.io``, parses each one's markdown address
tables, and maps the documented contract names to pydefi registry keys, merged
per chain id (read from the doc) into ``{NAME: {chain_id: addr}}``. Sepolia-only
today; a mainnet doc is picked up automatically once added. Tokens and the
keeper / council keys are excluded — tokens live in ``deployments._TOKENS``.

Usage::

    python3 update_babylon.py             # discover + merge all published docs
    python3 update_babylon.py URL_OR_PATH  # parse just one doc (e.g. a local checkout)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

_REPO = "babylonlabs-io/babylonlabs.github.io"
_DOC_DIR = "docs/trustless-bitcoin-vault"
_API_URL = f"https://api.github.com/repos/{_REPO}/contents/{_DOC_DIR}"
_RAW = f"https://raw.githubusercontent.com/{_REPO}/main/{_DOC_DIR}/{{dir}}/contract-addresses.mdx"
#: Used only if the discovery API is unavailable (rate-limited / offline).
_FALLBACK_DOCS = [_RAW.format(dir="testnet-info")]

# Each doc declares its chain inline, e.g. "… release on Sepolia (chain ID `11155111`)".
_CHAIN_ID = re.compile(r"chain ID\s*`(\d+)`", re.IGNORECASE)

# Documented contract name -> pydefi registry key.
_KEYS: dict[str, str] = {
    "BabylonCoreSpoke": "AAVE_V4_BABYLON_SPOKE",
    "AaveHub": "AAVE_V4_BABYLON_HUB",
    "AaveAdapter": "BABYLON_AAVE_ADAPTER",
    "AaveAdapterConfig": "BABYLON_AAVE_ADAPTER_CONFIG",
    "AaveAdapterLens": "BABYLON_AAVE_ADAPTER_LENS",
    "BTCVaultSwap": "BABYLON_BTC_VAULT_SWAP",
    "BTCVaultRegistry": "BABYLON_BTC_VAULT_REGISTRY",
    "BTCVaultsMetadataRegistry": "BABYLON_BTC_VAULTS_METADATA_REGISTRY",
    "ProtocolParams": "BABYLON_PROTOCOL_PARAMS",
    "ApplicationRegistry": "BABYLON_APPLICATION_REGISTRY",
    "CapPolicy": "BABYLON_CAP_POLICY",
    "FeeEscrow": "BABYLON_FEE_ESCROW",
}

# A table row whose first cell is a single backticked name: ``| `Name` | …``
_ROW = re.compile(r"^\s*\|\s*`([A-Za-z0-9]+)`\s*\|")
_ADDR = re.compile(r"0x[a-fA-F0-9]{40}")


def _load(src: str) -> str | None:
    """Return the doc text, or ``None`` if it isn't published yet (404/unreachable)."""
    if not src.startswith("http"):
        with open(src) as f:
            return f.read()
    req = urllib.request.Request(src, headers={"User-Agent": "pydefi"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed https host
            return resp.read().decode()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"skip {src}: not published (404)")
            return None
        raise
    except urllib.error.URLError as exc:
        print(f"skip {src}: unreachable ({exc.reason})")
        return None


def _discover_docs() -> list[str]:
    """Return a ``contract-addresses.mdx`` raw URL for each ``*-info`` network dir
    in the docs repo. Falls back to the known testnet doc if the GitHub contents
    API is unavailable (rate-limited / offline)."""
    req = urllib.request.Request(_API_URL, headers={"User-Agent": "pydefi"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed https host
            entries = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"discovery failed ({exc}); using fallback doc list")
        return _FALLBACK_DOCS
    dirs = sorted(e["name"] for e in entries if e.get("type") == "dir" and e["name"].endswith("-info"))
    return [_RAW.format(dir=d) for d in dirs] or _FALLBACK_DOCS


def _parse(text: str, out: dict[str, dict[str, str]]) -> None:
    """Parse one doc into *out* (``{NAME: {chain_id: addr}}``), merging in place."""
    chain = _CHAIN_ID.search(text)
    if not chain:
        raise SystemExit("babylon doc: could not read the chain id")
    chain_id = chain.group(1)

    n = 0
    for line in text.splitlines():
        m = _ROW.match(line)
        addr = _ADDR.search(line) if m else None
        key = _KEYS.get(m.group(1)) if m else None
        if key and addr:
            out.setdefault(key, {})[chain_id] = addr.group(0)
            n += 1
    print(f"chain {chain_id}: parsed {n} contracts")


def main() -> None:
    sources = [sys.argv[1]] if len(sys.argv) > 1 else _discover_docs()
    out: dict[str, dict[str, str]] = {}
    for src in sources:
        text = _load(src)
        if text is not None:
            _parse(text, out)

    missing = sorted(set(_KEYS.values()) - set(out))
    if missing:
        raise SystemExit(f"babylon.json: no published doc provided {missing}")

    out = {key: dict(sorted(out[key].items())) for key in sorted(out)}
    with open("babylon.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    networks = sorted({cid for chains in out.values() for cid in chains})
    print(f"babylon.json: wrote {len(out)} contracts across chain(s) {networks}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate cctp.json / cctp-testnet.json from Circle's official docs.

Uses ``url-to-md`` to fetch clean markdown from both pages, then
extracts TokenMessengerV2 + MessageTransmitterV2 + USDC per chain.
Domain→chainId mapping is hardcoded (stable).

It's not added into update.sh because of unstable.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_CCTP_JSON = _CONFIG_DIR / "cctp.json"
_CCTP_TESTNET_JSON = _CONFIG_DIR / "cctp-testnet.json"

_CCTP_URL = "https://developers.circle.com/cctp/references/contract-addresses"
_USDC_URL = "https://developers.circle.com/stablecoins/usdc-contract-addresses"

# Domain → EVM chain ID (mainnet)
_MAINNET_DOMAIN: dict[int, int] = {
    0: 1,        # Ethereum
    1: 43114,    # Avalanche C-Chain
    2: 10,       # OP Mainnet
    3: 42161,    # Arbitrum One
    6: 8453,     # Base
    7: 137,      # Polygon PoS
    10: 1301,    # Unichain
    11: 59144,   # Linea
    12: 5115,    # Codex
    13: 146,     # Sonic
    14: 480,     # World Chain
    16: 1329,    # Sei
    18: 50,      # XDC Network
    19: 999,     # HyperEVM
    21: 57073,   # Ink
    22: 98865,   # Plume
    28: 3343,    # EDGE
    29: 2525,    # Injective
    30: 2818,    # Morph
    31: 10182,   # Pharos
}

# Domain → EVM chain ID (testnet)
_TESTNET_DOMAIN: dict[int, int] = {
    0: 11155111,   # Ethereum Sepolia
    1: 43113,      # Avalanche Fuji
    2: 11155420,   # OP Sepolia
    3: 421614,     # Arbitrum Sepolia
    6: 84532,      # Base Sepolia
    7: 80002,      # Polygon PoS Amoy
    10: 1301,      # Unichain Sepolia
    11: 59141,     # Linea Sepolia
    12: 5115,      # Codex Testnet
    13: 57054,     # Sonic Testnet
    14: 4801,      # World Chain Sepolia
    16: 1328,      # Sei Testnet
    18: 51,        # XDC Apothem
    19: 998,       # HyperEVM Testnet
    21: 763373,    # Ink Testnet
    22: 161221135, # Plume Testnet
    26: 5042002,   # Arc Testnet
    28: 33431,     # EDGE Testnet
    29: 2526,      # Injective Testnet
    30: 2810,      # Morph Hoodi Testnet
    31: 10182002,  # Pharos Testnet
}

# Normalize CCTP page names → USDC page names
_NAME_MAP: dict[str, str] = {
    "Avalanche": "Avalanche C-Chain",
}


def _url_to_md(url: str) -> str:
    cp = subprocess.run(
        ["url-to-md", url, "--clean-content"],
        capture_output=True, text=True, timeout=60,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"url-to-md failed: {cp.stderr}")
    return cp.stdout


def _clean_addr(raw: str) -> str:
    """Remove whitespace from an address string."""
    return "".join(raw.split())


def _parse_cctp(md: str) -> tuple[dict[int, dict], dict[int, dict]]:
    """Parse CCTP markdown → (mainnet, testnet) data per domain."""
    mainnet: dict[int, dict] = {}
    testnet: dict[int, dict] = {}

    env = "mainnet"
    current_heading = ""

    lines = md.splitlines()
    for i, line in enumerate(lines):
        # Detect section environment
        if "Mainnet contract addresses" in line:
            env = "mainnet"
        elif "Testnet contract addresses" in line:
            env = "testnet"

        # h3 headings: "### [\u200b" / empty / "](" / empty / "HeadingText"
        if line.strip().startswith("### [") and "\u200b" in line and i + 4 < len(lines):
            heading = lines[i + 4].strip()
            if heading in ("TokenMessengerV2", "MessageTransmitterV2",
                           "TokenMinterV2", "MessageV2"):
                current_heading = heading

        # Table data rows: | **Name** | Domain | Address |
        m = re.match(r"^\| \*\*(.+?)\*\* \| (\d+) \| .+ \|\s*$", line)
        if not m:
            continue
        name = m.group(1).strip()
        domain = int(m.group(2).strip())
        # Extract address from the inline-code or link
        addr_raw = line.split("|")[3].strip()
        # Strip markdown link: [`0x...`](...) → 0x...
        addr_raw = re.sub(r"\[`([^`]+)`\].*", r"\1", addr_raw)
        addr_raw = addr_raw.removeprefix("`").removesuffix("`")
        addr = _clean_addr(addr_raw)

        if not addr.startswith("0x"):
            continue

        target = mainnet if env == "mainnet" else testnet
        entry = target.setdefault(domain, {"Name": name})

        if current_heading == "TokenMessengerV2":
            entry["TokenMessengerV2"] = addr
        elif current_heading == "MessageTransmitterV2":
            entry["MessageTransmitterV2"] = addr

    return mainnet, testnet


def _parse_usdc(md: str) -> dict[str, dict[str, str]]:
    """Parse USDC markdown → {mainnet|testnet: {name: USDC_addr}}."""
    usdc: dict[str, dict[str, str]] = {"mainnet": {}, "testnet": {}}
    env = "mainnet"

    lines = md.splitlines()
    for i, line in enumerate(lines):
        # Detect mainnet/testnet boundary.
        # Headings are split: "## [\u200b" / "](" / "Testnet".
        stripped = line.strip()
        if stripped.startswith("## [") and i + 2 < len(lines):
            url_line = lines[i + 2].lower()
            if "testnet" in url_line:
                env = "testnet"
            elif "mainnet" in url_line:
                env = "mainnet"

        # | Name | [`addr`](...) |  or  | Name | `addr` |
        m = re.match(
            r"^\| \*{0,2}(.+?)\*{0,2} \| "
            r"(?:\[`(.+?)`\].*|`(.+?)`)\s*\|",
            line,
        )
        if not m:
            continue
        name = m.group(1).strip()
        addr_raw = m.group(2) or m.group(3) or ""
        if addr_raw:
            addr = _clean_addr(addr_raw)
            usdc[env][name] = addr

    return usdc


def main() -> None:
    cctp_md = _url_to_md(_CCTP_URL)
    usdc_md = _url_to_md(_USDC_URL)

    mainnet_cctp, testnet_cctp = _parse_cctp(cctp_md)
    all_usdc = _parse_usdc(usdc_md)

    def build(env: str) -> dict[str, dict]:
        dom_map = _MAINNET_DOMAIN if env == "mainnet" else _TESTNET_DOMAIN
        cctp_data = mainnet_cctp if env == "mainnet" else testnet_cctp
        usdc_data = all_usdc[env]
        result: dict[str, dict] = {}

        for domain, info in sorted(cctp_data.items()):
            chain_id = dom_map.get(domain)
            if chain_id is None:
                continue
            name = info.get("Name", "")

            usdc_name = _NAME_MAP.get(name, name)
            usdc_addr = usdc_data.get(usdc_name, "") or usdc_data.get(name, "")

            entry = {
                "Name": name,
                "TokenMessengerV2": info.get("TokenMessengerV2", ""),
                "MessageTransmitterV2": info.get("MessageTransmitterV2", ""),
            }
            if usdc_addr:
                entry["USDC"] = usdc_addr
            result[str(chain_id)] = entry
        return result

    mainnet_out = build("mainnet")
    testnet_out = build("testnet")

    _CCTP_JSON.write_text(json.dumps(mainnet_out, indent=2) + "\n")
    _CCTP_TESTNET_JSON.write_text(json.dumps(testnet_out, indent=2) + "\n")

    print(
        f"update_cctp: wrote {_CCTP_JSON.name} ({len(mainnet_out)} chains) "
        f"and {_CCTP_TESTNET_JSON.name} ({len(testnet_out)} chains)"
    )


if __name__ == "__main__":
    main()

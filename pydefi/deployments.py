"""Deployment registry: (name, chain_id) -> address / Token.

The canonical address data lives in ``pydefi/data/deployments.jsonnet``.
Edit that file to add or update addresses, then the changes are picked up
automatically at import time.

Usage::

    from pydefi.deployments import get_address, get_token

    factory = get_address("UNISWAP_V3_FACTORY", chain_id=1)
    weth    = get_token("WETH", chain_id=1)
    chains  = chains_for("UNISWAP_V3_FACTORY")  # [1, 11155111, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import _jsonnet

from pydefi.types import Token

_JSONNET_FILE = Path(__file__).parent / "data" / "deployments.jsonnet"
_DATA: dict = json.loads(_jsonnet.evaluate_file(str(_JSONNET_FILE)))


def get_address(name: str, chain_id: int) -> str:
    """Return the deployed address of *name* on *chain_id*.

    Searches ``contracts`` first, then the ``addresses`` sub-key of ``tokens``.

    Raises :exc:`KeyError` when the name is unknown or has no deployment on
    the requested chain.
    """
    contracts = _DATA.get("contracts", {})
    if name in contracts:
        addr = contracts[name].get(str(chain_id))
        if addr is None:
            raise KeyError(f"{name!r} has no deployment on chain {chain_id}")
        return addr

    tokens = _DATA.get("tokens", {})
    if name in tokens:
        addr = tokens[name].get("addresses", {}).get(str(chain_id))
        if addr is None:
            raise KeyError(f"Token {name!r} has no deployment on chain {chain_id}")
        return addr

    raise KeyError(f"Unknown deployment name {name!r}")


def get_token(name: str, chain_id: int) -> Token:
    """Return a :class:`~pydefi.types.Token` for *name* on *chain_id*.

    Raises :exc:`KeyError` when the token is unknown or has no address on the
    requested chain.
    """
    tokens = _DATA.get("tokens", {})
    entry = tokens.get(name)
    if entry is None:
        raise KeyError(f"Unknown token {name!r}")
    addr = entry.get("addresses", {}).get(str(chain_id))
    if addr is None:
        raise KeyError(f"Token {name!r} has no deployment on chain {chain_id}")
    return Token(
        chain_id=chain_id,
        address=addr,
        symbol=entry["symbol"],
        decimals=entry["decimals"],
    )


def chains_for(name: str) -> list[int]:
    """Return the chain IDs that have a deployment of *name*."""
    contracts = _DATA.get("contracts", {})
    if name in contracts:
        return [int(c) for c in contracts[name]]
    tokens = _DATA.get("tokens", {})
    if name in tokens:
        return [int(c) for c in tokens[name].get("addresses", {})]
    raise KeyError(f"Unknown deployment name {name!r}")

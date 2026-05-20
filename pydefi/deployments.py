"""Deployment registry: (name, chain_id) -> address / Token.

Usage::

    from pydefi.deployments import get_address, get_token

    factory = get_address("UNISWAP_V3_FACTORY", chain_id=1)
    weth    = get_token("WETH", chain_id=1)
    chains  = chains_for("UNISWAP_V3_FACTORY")  # [1, 11155111, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

from pydefi._utils import decode_address
from pydefi.types import Address, ChainId, Token

_ETH = ChainId.ETHEREUM
_SEP = ChainId.SEPOLIA

# ── Tokens ────────────────────────────────────────────────────────────────────
# { name: { "symbol": str, "decimals": int, "addresses": { chain_id: address } } }

_TOKENS: dict[str, dict] = {
    "WETH": {
        "symbol": "WETH",
        "decimals": 18,
        "addresses": {
            _ETH: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            _SEP: "0xfff9976782d46cc05630d1f6ebab18b2324d6b14",
            ChainId.MANTRA: "0xEc09D14459e18fa15C1F916a8d8a2575eA5F7Ac4",
            ChainId.KITE: "0x3D66d6c3201190952e8EA973F59c4428b32D5F9b",
        },
    },
    "USDC": {
        "symbol": "USDC",
        "decimals": 6,
        "addresses": {
            _ETH: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            _SEP: "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
            ChainId.MANTRA: "0x8f726A72e3a28d21f153F1698e60d6a97eFd1a00",
        },
    },
    "USDC.e": {
        "symbol": "USDC.e",
        "decimals": 6,
        "addresses": {
            ChainId.KITE: "0x7aB6f3ed87C42eF0aDb67Ed95090f8bF5240149e",
        },
    },
    "DAI": {
        "symbol": "DAI",
        "decimals": 18,
        "addresses": {
            _ETH: "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        },
    },
    "USDT": {
        "symbol": "USDT",
        "decimals": 6,
        "addresses": {
            _ETH: "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            ChainId.MANTRA: "0x3806640578b710d8480910bF51510bc538d2F51A",
            ChainId.KITE: "0x3Fdd283C4c43A60398bf93CA01a8a8BD773a755b",
        },
    },
    "UNI": {
        "symbol": "UNI",
        "decimals": 18,
        "addresses": {
            _SEP: "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        },
    },
}

# ── Contracts ─────────────────────────────────────────────────────────────────
# Source: https://docs.uniswap.org/contracts/v2/reference/smart-contracts/v2-deployments
#         https://docs.uniswap.org/contracts/v3/reference/deployments/ethereum-deployments
# { name: { chain_id: address } }

_CONTRACTS: dict[str, dict[int, str]] = {
    # Uniswap V2
    "UNISWAP_V2_ROUTER": {
        _ETH: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        _SEP: "0xeE567Fe1712Faf6149d80dA1E6934E354124CfE3",
    },
    "UNISWAP_V2_FACTORY": {_SEP: "0xF62c03E08ada871A0bEb309762E260a7a6a880E6"},
    # Uniswap V3
    "UNISWAP_V3_ROUTER": {
        _ETH: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        _SEP: "0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E",
    },  # SwapRouter02
    "UNISWAP_V3_QUOTER": {
        _ETH: "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        _SEP: "0xEd1f6473345F45b75F8179591dd5bA1888cf2FB3",
    },  # QuoterV2
    "UNISWAP_V3_FACTORY": {
        _ETH: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        _SEP: "0x0227628f3F023bb0B980b67D528571c95c6DaC1c",
    },
    # Uniswap V4 / Universal Router
    "UNISWAP_V4_POOL_MANAGER": {_ETH: "0x000000000004444c5dc75cB358380D2e3dE08A90"},
    "UNIVERSAL_ROUTER": {_ETH: "0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af"},
    # Well-known Uniswap V3 pools
    "POOL_WETH_USDC_500": {_ETH: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"},  # 0.05%
    "POOL_WETH_USDC_3000": {_ETH: "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"},  # 0.30%
    "POOL_DAI_USDC_100": {_ETH: "0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168"},  # 0.01%
    # Well-known Uniswap V2 pairs
    "PAIR_WETH_USDC": {_ETH: "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"},
    "PAIR_WETH_DAI": {_ETH: "0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11"},
    "PAIR_USDC_DAI": {_ETH: "0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5"},
    "PAIR_USDC_USDT": {_ETH: "0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f"},
    # Lucid AssetControllers — https://docs.lucidlabs.fi/developer-reference/deployed-contracts/controller-contracts
    # MANTRA lanes: USDC/WETH -> Base,      USDT -> Optimism (Hyperlane adapter)
    # Kite lanes:   USDC/WETH -> Avalanche, USDT -> Celo (LayerZero adapter)
    "LUCID_USDC_CONTROLLER": {
        ChainId.MANTRA: "0xFf54162d7061290A966119784a0F34663d00E190",
        ChainId.KITE: "0x7144ADe61f4BEcD513658db3441a0CE286aE90fd",
    },
    "LUCID_WETH_CONTROLLER": {
        ChainId.MANTRA: "0x47263861BD8E964425910FAA0870e9d0fb630876",
        ChainId.KITE: "0x1324Be9339e1618387d2405C6f4CE660447cB582",
    },
    "LUCID_USDT_CONTROLLER": {
        ChainId.MANTRA: "0x0569725992ee675f862bf0d25CA40fCfB2B69d6E",
        ChainId.KITE: "0xc620C68Cb63EA35e9930e240051A6A7C02D568A1",
    },
    # Adapter addresses are not published by Lucid; discover via TransferRelayed
    # event scan against any controller, then pass to LucidBridge(adapter_address=…).
    "LUCID_LAYERZERO_ADAPTER": {ChainId.KITE: "0x5ef37628D45C80740Fb6Db7Ed9C0A753B4f85263"},
}


def _merge_generated(contracts: dict[str, dict[int, str]], filename: str) -> None:
    """Merge a generated address file (config/<filename>) into *contracts*.

    aave.json and compound.json are produced from upstream registries by
    config/update.sh. Regenerate with ``bash config/update.sh`` if a file
    is missing/empty/corrupt."""
    path = Path(__file__).resolve().parent / "config" / filename
    hint = "run config/update.sh to (re)generate it"
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found: {path} \u2014 {hint}")
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"{filename} is empty: {path} \u2014 {hint}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{filename} is corrupt: {path} ({exc}) \u2014 {hint}") from exc
    for name, chains in data.items():
        contracts[name] = {int(cid): addr for cid, addr in chains.items()}


_merge_generated(_CONTRACTS, "aave.json")
_merge_generated(_CONTRACTS, "compound.json")


def get_address(name: str, chain_id: int) -> str:
    """Return the deployed address of *name* on *chain_id*.

    Searches contracts first, then token addresses.

    Raises :exc:`KeyError` when the name is unknown or has no deployment on
    the requested chain.
    """
    if name in _CONTRACTS:
        addr = _CONTRACTS[name].get(chain_id)
        if addr is None:
            raise KeyError(f"{name!r} has no deployment on chain {chain_id}")
        return addr

    if name in _TOKENS:
        addr = _TOKENS[name]["addresses"].get(chain_id)
        if addr is None:
            raise KeyError(f"Token {name!r} has no deployment on chain {chain_id}")
        return addr

    raise KeyError(f"Unknown deployment name {name!r}")


def get_token(name: str, chain_id: int) -> Token:
    """Return a :class:`~pydefi.types.Token` for *name* on *chain_id*.

    Raises :exc:`KeyError` when the token is unknown or has no address on the
    requested chain.
    """
    entry = _TOKENS.get(name)
    if entry is None:
        raise KeyError(f"Unknown token {name!r}")
    addr = entry["addresses"].get(chain_id)
    if addr is None:
        raise KeyError(f"Token {name!r} has no deployment on chain {chain_id}")
    return Token(
        chain_id=chain_id, address=decode_address(addr, chain_id), symbol=entry["symbol"], decimals=entry["decimals"]
    )


def chains_for(name: str) -> list[int]:
    """Return the chain IDs that have a deployment of *name*."""
    if name in _CONTRACTS:
        return list(_CONTRACTS[name])
    if name in _TOKENS:
        return list(_TOKENS[name]["addresses"])
    raise KeyError(f"Unknown deployment name {name!r}")


def address_for(name: str, chain_id: int) -> Address:
    """Like :func:`get_address` but returns a chain-aware :class:`Address`
    instead of a raw string — saves callers from wrapping in
    ``Address(decode_address(...))`` at every call site."""
    return Address(decode_address(get_address(name, chain_id), chain_id))


# Compound III deployments are keyed by their base asset.
COMET_CONTRACT_BY_SYMBOL: dict[str, str] = {
    "USDC": "COMPOUND_V3_USDC",
    "WETH": "COMPOUND_V3_WETH",
    "USDT": "COMPOUND_V3_USDT",
}


def comet_contract_for(token_symbol: str) -> str:
    """Comet deployment name for *token_symbol*, or :class:`ValueError`
    listing supported symbols."""
    name = COMET_CONTRACT_BY_SYMBOL.get(token_symbol)
    if name is None:
        raise ValueError(
            f"Compound V3 has no market for token {token_symbol!r}; supported: {sorted(COMET_CONTRACT_BY_SYMBOL)}"
        )
    return name

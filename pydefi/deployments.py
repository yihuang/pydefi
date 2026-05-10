"""Deployment registry: (name, chain_id) -> address / Token.

Usage::

    from pydefi.deployments import get_address, get_token

    factory = get_address("UNISWAP_V3_FACTORY", chain_id=1)
    weth    = get_token("WETH", chain_id=1)
    chains  = chains_for("UNISWAP_V3_FACTORY")  # [1, 11155111, ...]
"""

from __future__ import annotations

from pydefi._utils import decode_address
from pydefi.types import ChainId, Token

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
        },
    },
    "USDC": {
        "symbol": "USDC",
        "decimals": 6,
        "addresses": {
            _ETH: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            _SEP: "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
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
}


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

"""Deployment registry: (name, chain_id) -> address / Token.

Usage::

    from pydefi.deployments import get_address, get_token

    factory = get_address("UNISWAP_V3_FACTORY", chain_id=1)
    weth    = get_token("WETH", chain_id=1)
    chains  = chains_for("UNISWAP_V3_FACTORY")  # [1, 11155111, ...]
"""

from __future__ import annotations

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
    # Aave V3
    # Canonical source: https://github.com/aave-dao/aave-address-book
    # PoolAddressesProvider is the only address Aave guarantees is immutable.
    # Pool, AaveProtocolDataProvider and AaveOracle are upgraded via
    # governance; the DataProvider in particular is re-deployed on every
    # protocol revision. Verify with
    # `PoolAddressesProvider.getPoolDataProvider() / .getPool() / .getPriceOracle()`
    # at runtime before relying on the pinned values below.
    "AAVE_V3_POOL": {
        _ETH: "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
        ChainId.ARBITRUM: "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        ChainId.BASE: "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
        ChainId.OPTIMISM: "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        ChainId.POLYGON: "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        ChainId.AVALANCHE: "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        ChainId.BSC: "0x6807dc923806fE8Fd134338EABCA509979a7e0cB",
        ChainId.SCROLL: "0x11fCfe756c05AD438e312a7fd934381537D3cFfe",
        ChainId.LINEA: "0xc47b8C00b0f69a36fa203Ffeac0334874574a8Ac",
        ChainId.ZKSYNC: "0x78e30497a3c7527d953c6B1E3541b021A98Ac43c",
        ChainId.GNOSIS: "0xb50201558B00496A145fE76f7424749556E326D8",
    },
    "AAVE_V3_DATA_PROVIDER": {
        _ETH: "0x0a16f2FCC0D44FaE41cc54e079281D84A363bECD",
        ChainId.ARBITRUM: "0x243Aa95cac2a25651Eda86E80BEe66114413c43B",
        ChainId.BASE: "0x0F43731Eb8D45A581F4a36DD74F5f358Bc90c73A",
        ChainId.OPTIMISM: "0x243Aa95cac2a25651Eda86E80BEe66114413c43B",
        ChainId.POLYGON: "0x243Aa95cac2a25651Eda86E80BEe66114413c43B",
        ChainId.AVALANCHE: "0x243Aa95cAC2a25651eda86e80bEe66114413c43b",
        ChainId.BSC: "0xc90Df74A7c16245c5F5C5870327Ceb38Fe5d5328",
        ChainId.SCROLL: "0xBEa2B648f05887eCbF1d115da8b7E4A317975A51",
        ChainId.LINEA: "0x47cd4b507B81cB831669c71c7077f4daF6762FF4",
        ChainId.ZKSYNC: "0x9057ac7b2D35606F8AD5aE2FCBafcD94E58D9927",
        ChainId.GNOSIS: "0xF1F5acB596568895393cB5E4D0452D6592A2fA70",
    },
    "AAVE_V3_ADDRESSES_PROVIDER": {
        _ETH: "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e",
        ChainId.ARBITRUM: "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
        ChainId.BASE: "0xe20fCBdBfFC4Dd138cE8b2E6FBb6CB49777ad64D",
        ChainId.OPTIMISM: "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
        ChainId.POLYGON: "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
        ChainId.AVALANCHE: "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
        ChainId.BSC: "0xff75B6da14FfbbfD355Daf7a2731456b3562Ba6D",
        ChainId.SCROLL: "0x69850D0B276776781C063771b161bd8894BCdD04",
        ChainId.LINEA: "0x89502c3731F69DDC95B65753708A07F8Cd0373F4",
        ChainId.ZKSYNC: "0x2A3948BB219D6B2Fa83D64100006391a96bE6cb7",
        ChainId.GNOSIS: "0x36616cf17557639614c1cdDb356b1B83fc0B2132",
    },
    "AAVE_V3_ORACLE": {
        _ETH: "0x54586bE62E3c3580375aE3723C145253060Ca0C2",
        ChainId.ARBITRUM: "0xb56c2F0B653B2e0b10C9b928C8580Ac5Df02C7C7",
        ChainId.BASE: "0x2Cc0Fc26eD4563A5ce5e8bdcfe1A2878676Ae156",
        ChainId.OPTIMISM: "0xD81eb3728a631871a7eBBaD631b5f424909f0c77",
        ChainId.POLYGON: "0xb023e699F5a33916Ea823A16485e259257cA8Bd1",
        ChainId.AVALANCHE: "0xEBd36016B3eD09D4693Ed4251c67Bd858c3c7C9C",
        ChainId.BSC: "0x39bc1bfDa2130d6Bb6DBEfd366939b4c7aa7C697",
        ChainId.SCROLL: "0x04421D8C506E2fA2371a08EfAaBf791F624054F3",
        ChainId.LINEA: "0xCFDAdA7DCd2e785cF706BaDBC2B8Af5084d595e9",
        ChainId.ZKSYNC: "0xC7F58Fca663a8d377B6D0c9703C697f56dC40088",
        ChainId.GNOSIS: "0xeb0a051be10228213BAEb449db63719d6742F7c4",
    },
    # Compound III (Comet)
    # Canonical source: https://github.com/compound-finance/comet/tree/main/deployments
    # One proxy per (chain, base asset) market. Naming pattern:
    # COMPOUND_V3_<BASE><CHAIN-SUFFIX-IF-MULTIPLE>.
    "COMPOUND_V3_USDC": {
        _ETH: "0xc3d688B66703497DAA19211EEdff47f25384cdc3",
        ChainId.ARBITRUM: "0x9c4ec768c28520B50860ea7a15bd7213a9fF58bf",
        ChainId.BASE: "0xb125E6687d4313864e53df431d5425969c15Eb2F",
        ChainId.POLYGON: "0xF25212E676D1F7F89Cd72fFEe66158f541246445",
        ChainId.OPTIMISM: "0x2e44e174f7D53F0212823acC11C01A11d58c5bCB",
    },
    "COMPOUND_V3_WETH": {
        _ETH: "0xA17581A9E3356d9A858b789D68B4d866e593aE94",
        ChainId.ARBITRUM: "0x6f7D514bbD4aFf3BcD1140B7344b32f063dEe486",
        ChainId.BASE: "0x46e6b214b524310239732D51387075E0e70970bf",
        ChainId.OPTIMISM: "0xE36A30D249f7761327fd973001A32010b521b6Fd",
    },
    "COMPOUND_V3_USDT": {
        _ETH: "0x3Afdc9BCA9213A35503b077a6072F3D0d5AB0840",
        ChainId.ARBITRUM: "0xd98Be00b5D27fc98112BdE293e487f8D4cA57d07",
    },
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

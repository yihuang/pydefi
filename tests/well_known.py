"""Canonical on-chain addresses and token objects for pydefi tests.

Everything is grouped by network in plain namespace classes so any test file
can import exactly what it needs without duplicating address strings:

    from tests.well_known import mainnet, sepolia

    edge = V3PoolEdge(
        token_in=mainnet.WETH,
        pool_address=mainnet.POOL_WETH_USDC_500,
        ...
    )

All addresses are stored as plain mixed-case hex strings.  Use
``Web3.to_checksum_address(addr)`` only when the target API requires it.

No pytest, no network I/O — safe to import from unit tests.
"""

from __future__ import annotations

from pydefi.types import ChainId, Token

# ---------------------------------------------------------------------------
# Ethereum mainnet (chain_id = 1)
# ---------------------------------------------------------------------------


class mainnet:
    """Well-known Ethereum mainnet addresses."""

    # ── Tokens ────────────────────────────────────────────────────────────
    WETH = Token(ChainId.ETHEREUM, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", 18)
    USDC = Token(ChainId.ETHEREUM, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", 6)
    DAI = Token(ChainId.ETHEREUM, "0x6B175474E89094C44Da98b954EedeAC495271d0F", "DAI", 18)
    USDT = Token(ChainId.ETHEREUM, "0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT", 6)

    # ── Uniswap contracts ─────────────────────────────────────────────────
    UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    UNISWAP_V3_QUOTER = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
    UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    UNISWAP_V4_POOL_MANAGER = "0x000000000004444c5dc75cB358380D2e3dE08A90"
    UNIVERSAL_ROUTER = "0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af"

    # ── Well-known V3 pools ───────────────────────────────────────────────
    POOL_WETH_USDC_500 = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # 0.05 %
    POOL_WETH_USDC_3000 = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"  # 0.3 %
    POOL_DAI_USDC_100 = "0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168"  # 0.01 %

    # ── Well-known V2 pairs ───────────────────────────────────────────────
    PAIR_WETH_USDC = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"
    PAIR_WETH_DAI = "0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11"
    PAIR_USDC_DAI = "0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5"
    PAIR_USDC_USDT = "0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f"

    # ── Utility ───────────────────────────────────────────────────────────
    #: vitalik.eth — well-funded address used for impersonation in fork tests.
    ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    ZERO_ADDR = "0x0000000000000000000000000000000000000000"
    #: Analog-Labs EVM interpreter — pre-deployed via CREATE2 on mainnet.
    INTERPRETER_ADDR = "0x0000000000001e3F4F615cd5e20c681Cf7d85e8D"


# ---------------------------------------------------------------------------
# Ethereum Sepolia testnet (chain_id = 11155111)
# ---------------------------------------------------------------------------


class sepolia:
    """Well-known Sepolia testnet addresses."""

    # ── Tokens ────────────────────────────────────────────────────────────
    WETH = Token(ChainId.SEPOLIA, "0xfff9976782d46cc05630d1f6ebab18b2324d6b14", "WETH", 18)
    USDC = Token(ChainId.SEPOLIA, "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", "USDC", 6)
    UNI = Token(ChainId.SEPOLIA, "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "UNI", 18)

    # ── Uniswap V2 ────────────────────────────────────────────────────────
    #: https://docs.uniswap.org/contracts/v2/reference/smart-contracts/v2-deployments
    V2_ROUTER = "0xeE567Fe1712Faf6149d80dA1E6934E354124CfE3"
    V2_FACTORY = "0xF62c03E08ada871A0bEb309762E260a7a6a880E6"
    #: Alternate V2 factory (older Uniswap deployment on Sepolia)
    V2_FACTORY_ALT = "0x7E0987E5b3a30e3f2828572Bb659A548460a3003"

    # ── Uniswap V3 ────────────────────────────────────────────────────────
    #: https://docs.uniswap.org/contracts/v3/reference/deployments/ethereum-deployments
    V3_ROUTER = "0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E"  # SwapRouter02
    V3_QUOTER = "0xEd1f6473345F45b75F8179591dd5bA1888cf2FB3"  # QuoterV2
    V3_FACTORY = "0x0227628f3F023bb0B980b67D528571c95c6DaC1c"

    # ── CCTP (Circle Cross-Chain Transfer Protocol) ───────────────────────
    CCTP_TOKEN_MESSENGER = "0x8fe6b999dc680ccfdd5bf7eb0974218be2542daa"
    CCTP_MESSAGE_TRANSMITTER_V2 = "0xE737e5cEBEEBa77EFE34D4aa090756590b1CE275"
    CCTP_DOMAIN = 0

    # ── Cross-chain ───────────────────────────────────────────────────────
    ZERO_ADDR = "0x0000000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Arbitrum Sepolia testnet (chain_id = 421614)
# ---------------------------------------------------------------------------


class arbitrum_sepolia:
    """Well-known Arbitrum Sepolia addresses."""

    USDC = "0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d"

    # ── CCTP ──────────────────────────────────────────────────────────────
    CCTP_MESSAGE_TRANSMITTER_V2 = "0xE737e5cEBEEBa77EFE34D4aa090756590b1CE275"
    CCTP_DOMAIN = 3

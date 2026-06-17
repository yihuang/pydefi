"""Well-known on-chain addresses for pydefi tests.

Single source for both unit tests and live/fork tests.
Data lives in pydefi/data/deployments.jsonnet; add new entries there.

No pytest, no network I/O — safe to import from unit tests.
"""

from __future__ import annotations

from pydefi.deployments import get_address, get_token
from pydefi.types import ZERO_ADDRESS, Address, ChainId, Token

# ── Chain-agnostic ────────────────────────────────────────────────────────────

ZERO_ADDR = ZERO_ADDRESS

# ── Test-only constants (not in the deployment registry) ──────────────────────

#: vitalik.eth — well-funded address used for impersonation in fork tests.
ETH_WHALE: Address = Address("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
#: Polygon PoS bridge escrow — $1 B+ in mainnet USDC, stable across forks
#: and never moves spontaneously, so impersonation transfers are reliable.
USDC_WHALE: Address = Address("0x40ec5B33f54e0E8A33A975908C5BA1c14e5BbbDf")
#: pydefi-patched EVM interpreter — deterministic CREATE2 address from
#: ``pydefi/vm/PatchedInterpreterConstants.sol``.  The patched build rejects
#: SLOAD / SSTORE / CALLCODE / DELEGATECALL / SELFDESTRUCT at the opcode-
#: dispatch level (issue #138).  Tests must run against this, not the
#: upstream Analog-Labs interpreter, so production code paths are exercised
#: faithfully — e.g. a swap program that accidentally emits SLOAD should
#: fail in tests too, not silently succeed against an unpatched runtime.
INTERPRETER_ADDR: Address = Address("0x64fE558B0F9a5dC18D4A36c85Ba99c3f222F7bde")

# ── Ethereum mainnet tokens ───────────────────────────────────────────────────

WETH = get_token("WETH", ChainId.ETHEREUM)
USDC = get_token("USDC", ChainId.ETHEREUM)
DAI = get_token("DAI", ChainId.ETHEREUM)
USDT = get_token("USDT", ChainId.ETHEREUM)

# ── Uniswap mainnet contracts ─────────────────────────────────────────────────

UNISWAP_V2_ROUTER: Address = get_address("UNISWAP_V2_ROUTER", ChainId.ETHEREUM)
UNISWAP_V3_ROUTER: Address = get_address("UNISWAP_V3_ROUTER", ChainId.ETHEREUM)
UNISWAP_V3_QUOTER: Address = get_address("UNISWAP_V3_QUOTER", ChainId.ETHEREUM)
UNISWAP_V3_FACTORY: Address = get_address("UNISWAP_V3_FACTORY", ChainId.ETHEREUM)
UNISWAP_V4_POOL_MANAGER: Address = get_address("UNISWAP_V4_POOL_MANAGER", ChainId.ETHEREUM)
UNISWAP_V4_STATE_VIEW: Address = get_address("UNISWAP_V4_STATE_VIEW", ChainId.ETHEREUM)
UNISWAP_V4_QUOTER: Address = get_address("UNISWAP_V4_QUOTER", ChainId.ETHEREUM)
UNIVERSAL_ROUTER: Address = get_address("UNIVERSAL_ROUTER", ChainId.ETHEREUM)

# ── Well-known Uniswap V3 pools ───────────────────────────────────────────────

POOL_WETH_USDC_500: Address = get_address("POOL_WETH_USDC_500", ChainId.ETHEREUM)
POOL_WETH_USDC_3000: Address = get_address("POOL_WETH_USDC_3000", ChainId.ETHEREUM)
POOL_DAI_USDC_100: Address = get_address("POOL_DAI_USDC_100", ChainId.ETHEREUM)

# ── Well-known Uniswap V2 pairs ───────────────────────────────────────────────

PAIR_WETH_USDC: Address = get_address("PAIR_WETH_USDC", ChainId.ETHEREUM)
PAIR_WETH_DAI: Address = get_address("PAIR_WETH_DAI", ChainId.ETHEREUM)
PAIR_USDC_DAI: Address = get_address("PAIR_USDC_DAI", ChainId.ETHEREUM)
PAIR_USDC_USDT: Address = get_address("PAIR_USDC_USDT", ChainId.ETHEREUM)

# ── Aave V3 mainnet ───────────────────────────────────────────────────────────
# Only the PoolAddressesProvider is pinned — Pool / DataProvider / Oracle are
# resolved from it on-chain (AaveV3.from_chain), so live tests construct via
# from_chain rather than pinning the rotating addresses.

AAVE_ADDRESSES_PROVIDER: Address = get_address("AAVE_V3_ADDRESSES_PROVIDER", ChainId.ETHEREUM)

# ── Compound III (Comet) mainnet ──────────────────────────────────────────────

COMET_USDC: Address = get_address("COMPOUND_V3_USDC", ChainId.ETHEREUM)

# ── Morpho Blue mainnet ───────────────────────────────────────────────────────
# Morpho Blue is an immutable singleton; the AdaptiveCurveIRM is the standard
# interest-rate model used by virtually every market.

MORPHO_BLUE: Address = get_address("MORPHO_BLUE", ChainId.ETHEREUM)
MORPHO_IRM: Address = get_address("MORPHO_ADAPTIVE_CURVE_IRM", ChainId.ETHEREUM)
#: cbBTC/USDC market unique key — a deep, stable market used by lending fork
#: tests (USDC is the loan asset).
MORPHO_CBBTC_USDC: bytes = bytes.fromhex("64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64")

# ── Lucid Labs ────────────────────────────────────────────────────────────────

LUCID_USDC_CTRL_MANTRA: Address = get_address("LUCID_USDC_CONTROLLER", ChainId.MANTRA)
# Synthetic — Lucid's docs don't publish the Hyperlane adapter; tests don't need a real one.
LUCID_HL_ADAPTER: Address = Address("0x" + "AB" * 20)
USDC_MANTRA = get_token("USDC", ChainId.MANTRA)
USDC_BASE = Token(chain_id=ChainId.BASE, address=Address("0x" + "C2" * 20), symbol="USDC", decimals=6)

LUCID_USDC_CTRL_KITE: Address = get_address("LUCID_USDC_CONTROLLER", ChainId.KITE)
LUCID_LZ_ADAPTER: Address = get_address("LUCID_LAYERZERO_ADAPTER", ChainId.KITE)
USDC_E_KITE = get_token("USDC.e", ChainId.KITE)
# Synthetic destination token — only address matters since LucidBridge doesn't validate it.
USDC_E_AVAX = Token(chain_id=ChainId.AVALANCHE, address=Address("0x" + "C1" * 20), symbol="USDC.e", decimals=6)

"""Well-known on-chain addresses for pydefi tests.

Single source for both unit tests and live/fork tests.
Data lives in pydefi/data/deployments.jsonnet; add new entries there.

No pytest, no network I/O — safe to import from unit tests.
"""

from __future__ import annotations

from pydefi.deployments import get_address, get_token
from pydefi.types import ChainId

# ── Chain-agnostic ────────────────────────────────────────────────────────────

ZERO_ADDR = "0x0000000000000000000000000000000000000000"

# ── Test-only constants (not in the deployment registry) ──────────────────────

#: vitalik.eth — well-funded address used for impersonation in fork tests.
ETH_WHALE = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
#: Analog-Labs EVM interpreter — pre-deployed via CREATE2 on mainnet.
INTERPRETER_ADDR = "0x0000000000001e3F4F615cd5e20c681Cf7d85e8D"

# ── Ethereum mainnet tokens ───────────────────────────────────────────────────

WETH = get_token("WETH", ChainId.ETHEREUM)
USDC = get_token("USDC", ChainId.ETHEREUM)
DAI = get_token("DAI", ChainId.ETHEREUM)
USDT = get_token("USDT", ChainId.ETHEREUM)

# ── Uniswap mainnet contracts ─────────────────────────────────────────────────

UNISWAP_V2_ROUTER = get_address("UNISWAP_V2_ROUTER", ChainId.ETHEREUM)
UNISWAP_V3_ROUTER = get_address("UNISWAP_V3_ROUTER", ChainId.ETHEREUM)
UNISWAP_V3_QUOTER = get_address("UNISWAP_V3_QUOTER", ChainId.ETHEREUM)
UNISWAP_V3_FACTORY = get_address("UNISWAP_V3_FACTORY", ChainId.ETHEREUM)
UNISWAP_V4_POOL_MANAGER = get_address("UNISWAP_V4_POOL_MANAGER", ChainId.ETHEREUM)
UNIVERSAL_ROUTER = get_address("UNIVERSAL_ROUTER", ChainId.ETHEREUM)

# ── Well-known Uniswap V3 pools ───────────────────────────────────────────────

POOL_WETH_USDC_500 = get_address("POOL_WETH_USDC_500", ChainId.ETHEREUM)
POOL_WETH_USDC_3000 = get_address("POOL_WETH_USDC_3000", ChainId.ETHEREUM)
POOL_DAI_USDC_100 = get_address("POOL_DAI_USDC_100", ChainId.ETHEREUM)

# ── Well-known Uniswap V2 pairs ───────────────────────────────────────────────

PAIR_WETH_USDC = get_address("PAIR_WETH_USDC", ChainId.ETHEREUM)
PAIR_WETH_DAI = get_address("PAIR_WETH_DAI", ChainId.ETHEREUM)
PAIR_USDC_DAI = get_address("PAIR_USDC_DAI", ChainId.ETHEREUM)
PAIR_USDC_USDT = get_address("PAIR_USDC_USDT", ChainId.ETHEREUM)

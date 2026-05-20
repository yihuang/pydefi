#!/usr/bin/env bash
# Regenerate config/*.json from canonical upstream sources.
set -euo pipefail

cd "$(dirname "$0")"

# CCIP chain directory (ccip-chains.json).
# Source: https://docs.chain.link/ccip/directory
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' \
  | jq -S '.data.evm' > ccip-chains.json

# Aave V3 PoolAddressesProvider per chain (aave.json) — the immutable root;
# Pool / DataProvider / Oracle are resolved from it on-chain at runtime.
# Source: the @aave-dao/aave-address-book npm package; jq picks the canonical
# V3 markets and reduces each to its PoolAddressesProvider, keyed by chain id.
deno eval 'import * as book from "npm:@aave-dao/aave-address-book"; console.log(JSON.stringify(book, null, " "))' \
  | jq -S '{AAVE_V3_ADDRESSES_PROVIDER: (
      [.AaveV3Ethereum, .AaveV3Optimism, .AaveV3BNB, .AaveV3Gnosis, .AaveV3Polygon,
       .AaveV3ZkSync, .AaveV3Base, .AaveV3Arbitrum, .AaveV3Avalanche, .AaveV3Linea,
       .AaveV3Scroll, .AaveV3Sepolia]
      | map({(.CHAIN_ID | tostring): .POOL_ADDRESSES_PROVIDER}) | add)}' \
  > aave.json

# Compound III (Comet) market addresses (compound.json).
# Source: https://github.com/compound-finance/comet
python3 update_compound.py

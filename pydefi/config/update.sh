#!/usr/bin/env bash
# Regenerate config/*.json from canonical upstream sources.
set -euo pipefail

cd "$(dirname "$0")"

# CCIP chain directory (ccip-chains.json).
# Source: https://docs.chain.link/ccip/directory
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' \
  | jq -S '.data.evm' > ccip-chains.json

# Aave V3 PoolAddressesProvider per chain (aave.json).
# Source: the @aave-dao/aave-address-book npm package — it ships several
# markets per chain id, so jq keeps the canonical one (shortest export name).
deno eval 'import * as book from "npm:@aave-dao/aave-address-book"; console.log(JSON.stringify(book, null, " "))' \
  | jq -S '{AAVE_V3_ADDRESSES_PROVIDER: (
      to_entries
      | map(select((.key | startswith("AaveV3"))
                   and .value.CHAIN_ID != null
                   and .value.POOL_ADDRESSES_PROVIDER != null))
      | group_by(.value.CHAIN_ID)
      | map(min_by(.key | length))
      | map({(.value.CHAIN_ID | tostring): .value.POOL_ADDRESSES_PROVIDER})
      | add)}' \
  > aave.json

# Compound III (Comet) market addresses (compound.json).
# Source: https://github.com/compound-finance/comet
python3 update_compound.py

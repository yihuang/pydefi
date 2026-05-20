#!/usr/bin/env bash
# Regenerate config/*.json from canonical upstream sources.
set -euo pipefail

cd "$(dirname "$0")"

# CCIP chain directory (ccip-chains.json).
# Source: https://docs.chain.link/ccip/directory
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' \
  | jq -S '.data.evm' > ccip-chains.json

# Aave V3 core addresses (the AAVE_V3_* block of deployments.json).
# Source: https://github.com/aave-dao/aave-address-book
python3 update_aave.py

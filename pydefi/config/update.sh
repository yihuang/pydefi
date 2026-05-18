#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' \
  | jq -S '.data.evm' > ccip-chains.json

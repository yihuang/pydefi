#!/usr/bin/env bash
# Regenerate config/*.json from canonical upstream sources.
set -euo pipefail

cd "$(dirname "$0")"

# CCIP chain directory (ccip-chains.json).
# Source: https://docs.chain.link/ccip/directory
echo "Fetching CCIP chain directory"
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' \
  | jq -S '.data.evm' > ccip-chains.json

# Aave V3 PoolAddressesProvider per chain (aave.json).
# Source: the @aave-dao/aave-address-book npm package — it ships several
# markets per chain id, so jq keeps the canonical one (shortest export name).
echo "Fetching Aave V3 addresses"
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

# Aave V4 Hub & Spoke addresses per chain (aave_v4.json).
# Source: the @aave-dao/aave-address-book npm package — V4 ships one
# `AaveV4<Chain>` export per chain, each with named HUBS / SPOKES /
# TOKENIZATION_SPOKES dicts. jq flattens those into AAVE_V4_<NAME>; the
# `_ORACLE` keys mixed into SPOKES are dropped (only `_HUB` / `_SPOKE` /
# `_ESPOKE` are kept — `_TOKENIZATION_SPOKE` ends with `_SPOKE`).
echo "Fetching Aave V4 addresses"
deno eval 'import * as book from "npm:@aave-dao/aave-address-book"; console.log(JSON.stringify(book))' \
  | jq -S '
      [ to_entries[]
        | select(.key | startswith("AaveV4"))
        | select(.value.CHAIN_ID != null)
        | .value as $v
        | ($v.CHAIN_ID | tostring) as $cid
        | ((($v.HUBS // {}) + ($v.SPOKES // {}) + ($v.TOKENIZATION_SPOKES // {})) | to_entries[])
        | select(.key | (endswith("_HUB") or endswith("_SPOKE") or endswith("_ESPOKE")))
        | {name: ("AAVE_V4_" + .key), cid: $cid, addr: .value} ]
      | reduce .[] as $e ({}; .[$e.name][$e.cid] = $e.addr)' \
  > aave_v4.json

# Compound III (Comet) market addresses (compound.json).
# Source: https://github.com/compound-finance/comet
echo "Fetching Compound III addresses"
python3 update_compound.py

# Morpho Blue core + AdaptiveCurveIRM per chain (morpho.json).
# Source: the @morpho-org/blue-sdk npm package's addressesRegistry. Morpho
# Blue is immutable, but only its Ethereum/Base deployment got the vanity
# 0xBBBB… address — later chains were deployed at distinct addresses — so the
# registry is the canonical per-chain source. blue-sdk needs viem as a peer
# dependency, hence the extra import.
echo "Fetching Morpho Blue addresses"
deno eval 'import { addressesRegistry } from "npm:@morpho-org/blue-sdk"; import "npm:viem@2"; console.log(JSON.stringify(addressesRegistry))' \
  | jq -S '{
      MORPHO_BLUE: (to_entries | map(select(.value.morpho != null))
                    | map({(.key): .value.morpho}) | add),
      MORPHO_ADAPTIVE_CURVE_IRM: (to_entries | map(select(.value.adaptiveCurveIrm != null))
                                  | map({(.key): .value.adaptiveCurveIrm}) | add)
    }' \
  > morpho.json

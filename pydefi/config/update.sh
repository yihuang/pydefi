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

# Babylon TBV × Aave V4 testnet addresses (babylon.json).
# Source: the published contract-addresses doc in babylonlabs.github.io. TBV is
# testnet-only (Sepolia); the parser maps documented contract names to registry
# keys. Tokens (vaultBTC etc.) stay in deployments._TOKENS.
echo "Fetching Babylon TBV addresses"
python3 update_babylon.py

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

# Uniswap V3 factory + V4 PoolManager / StateView / Quoter + UniversalRouter, per chain (uniswap.json).
# Source: @uniswap/sdk-core's CHAIN_TO_ADDRESSES_MAP, plus
# @uniswap/universal-router-sdk for the UniversalRouter (not in sdk-core).
# Only routers >= 2.1.1 are emitted (newest per chain) — older V2_0 routers
# use the pre-minHopPriceX36 V4 structs and would misdecode pydefi's calldata.
echo "Fetching Uniswap V3/V4 addresses"
deno eval '
import { CHAIN_TO_ADDRESSES_MAP as M } from "https://esm.sh/@uniswap/sdk-core@7";
import { UNIVERSAL_ROUTER_ADDRESS, UniversalRouterVersion } from "https://esm.sh/@uniswap/universal-router-sdk@5";
const FIELDS = {
  UNISWAP_V3_FACTORY: "v3CoreFactoryAddress",
  UNISWAP_V4_POOL_MANAGER: "v4PoolManagerAddress",
  UNISWAP_V4_STATE_VIEW: "v4StateView",
  UNISWAP_V4_QUOTER: "v4QuoterAddress",
};
const out = {};
for (const [name, field] of Object.entries(FIELDS)) {
  const m = {};
  for (const [cid, a] of Object.entries(M)) if (a[field]) m[cid] = a[field];
  out[name] = m;
}
out.UNIVERSAL_ROUTER = {};
for (const cid of Object.keys(M)) {
  for (const v of [UniversalRouterVersion.V2_2_0, UniversalRouterVersion.V2_1_1]) {
    try {
      out.UNIVERSAL_ROUTER[cid] = UNIVERSAL_ROUTER_ADDRESS(v, Number(cid));
      break;
    } catch {
      // chain has no UniversalRouter of this generation
    }
  }
}
console.log(JSON.stringify(out));
' \
  | jq -S '.' \
  > uniswap.json

# Polymarket neg-risk + CTF addresses (polymarket.json).
# Source: https://github.com/Polymarket/neg-risk-ctf-adapter — addresses.json.
# Transpose to {CONTRACT: {chain: addr}}, Polygon mainnet only; the appended
# ConditionalTokens Amoy (80002) address isn't in the manifest.
echo "Fetching Polymarket addresses"
curl -s 'https://raw.githubusercontent.com/Polymarket/neg-risk-ctf-adapter/main/addresses.json' \
  | jq -S --arg amoy "0x69308FB512518e39F9b16112fA8d994F4e2Bf8bB" '
      {
        ctf: "POLYMARKET_CONDITIONAL_TOKENS",
        negRiskAdapter: "POLYMARKET_NEG_RISK_ADAPTER",
        negRiskCtfExchange: "POLYMARKET_NEG_RISK_CTF_EXCHANGE",
        negRiskFeeModule: "POLYMARKET_NEG_RISK_FEE_MODULE",
        negRiskOperator: "POLYMARKET_NEG_RISK_OPERATOR",
        negRiskVault: "POLYMARKET_NEG_RISK_VAULT",
        negRiskUmaCtfAdapter: "POLYMARKET_NEG_RISK_UMA_CTF_ADAPTER",
        negRiskWrappedCollateral: "POLYMARKET_NEG_RISK_WRAPPED_COLLATERAL"
      } as $names
      | reduce ((."137" // {}) | to_entries[]) as $e ({};
          if $names[$e.key] then .[$names[$e.key]]["137"] = $e.value else . end)
      | .POLYMARKET_CONDITIONAL_TOKENS["80002"] = $amoy' \
  > polymarket.json

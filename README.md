# pydefi
Modern Python library and CLI for DeFi – integrates AMM DEXes, DEX aggregators, cross-chain bridges, and pathfinding into a single tool.

## Installation

```bash
pip install pydefi
```

---

## CLI

After installation a `defi` command is available.  It lets you compose DeFi actions and chain them into **execution plans** that can be reviewed, saved as JSON, and executed step-by-step.

### Commands

| Command | Description |
|---------|-------------|
| `defi plan` | Generate a cross-chain execution plan |
| `defi swap` | Create a single-chain swap plan |
| `defi bridge` | Create a cross-chain bridge plan |
| `defi execute` | Execute a previously generated plan |

### `defi plan` – Cross-chain execution plan

Generate a step-by-step plan to convert a token on one chain to a token on another chain.  The planner selects a bridgeable intermediate token (e.g. USDC or ETH) and produces the minimal sequence of swaps and bridge steps required.

```bash
defi plan \
  --src-chain ethereum \
  --src-token DAI \
  --dst-chain base \
  --dst-token WETH \
  --amount 1000
```

Example output:
```
Convert 1000 DAI on Ethereum to WETH on Base

  Step 1: Swap 1000 DAI → USDC on chain 1 via auto
  Step 2: Bridge 1000 USDC from chain 1 to chain 8453 via auto
  Step 3: Swap 1000 USDC → WETH on chain 8453 via auto

JSON representation:
{
  "description": "Convert 1000 DAI on Ethereum to WETH on Base",
  ...
}
```

Save the plan to a file for later execution:
```bash
defi plan \
  --src-chain ethereum --src-token DAI \
  --dst-chain base --dst-token WETH \
  --amount 1000 \
  --output plan.json
```

### `defi swap` – Single-chain swap plan

```bash
defi swap \
  --chain ethereum \
  --token-in USDC \
  --token-out WETH \
  --amount 500 \
  --output swap_plan.json
```

### `defi bridge` – Cross-chain bridge plan

```bash
defi bridge \
  --src-chain ethereum \
  --dst-chain arbitrum \
  --token USDC \
  --amount 1000 \
  --output bridge_plan.json
```

### `defi execute` – Execute a plan

Review and execute a plan that was saved to JSON.  Use `--dry-run` to print the steps without submitting any transactions.

```bash
# Dry-run: inspect steps without executing
defi execute --plan plan.json --dry-run

# Live execution (requires an RPC endpoint)
defi execute --plan plan.json --rpc https://eth.drpc.org --wallet 0xYourAddress
```

### Recognised chain names

The CLI accepts both numeric chain IDs and human-friendly names:

| Name | Chain ID |
|------|----------|
| `ethereum` / `eth` / `mainnet` | 1 |
| `optimism` / `op` | 10 |
| `bsc` / `bnb` | 56 |
| `polygon` / `poly` | 137 |
| `base` | 8453 |
| `arbitrum` / `arb` | 42161 |
| `avalanche` / `avax` | 43114 |
| `linea` | 59144 |
| `blast` | 81457 |
| `scroll` | 534352 |
| `zksync` / `zk` | 324 |
| `unichain` | 130 |
| `worldchain` | 480 |

---

## Python library

The library provides building blocks for integrating DeFi protocols in Python applications.

### Quick-start example

```python
from pydefi.amm import UniswapV2
from pydefi.types import Token, TokenAmount, ChainId

# Define tokens
ETH  = Token(ChainId.ETHEREUM, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH")
USDC = Token(ChainId.ETHEREUM, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", decimals=6)

# Create AMM client
uniswap = UniswapV2(w3, router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")

# Build a swap route (requires live node)
route = await uniswap.build_swap_route(
    amount_in=TokenAmount.from_human(ETH, "1.0"),
    token_out=USDC,
)
```

### Planner API

The planner module lets you build execution plans programmatically:

```python
from pydefi.planner import build_plan
from pydefi.types import ChainId

plan = build_plan(
    src_chain_id=ChainId.ETHEREUM,
    src_token="DAI",
    dst_chain_id=ChainId.BASE,
    dst_token="WETH",
    amount="1000",
)

print(plan.describe())
print(plan.to_json())
```

---

## Architecture

```
pydefi/
├── amm/          # AMM DEX integrations (Uniswap V2/V3, Curve, Universal Router)
├── aggregator/   # DEX aggregator APIs (1inch, ParaSwap, 0x)
├── bridge/       # Cross-chain bridges (Stargate, Across)
├── pathfinder/   # Graph-based optimal route finder
├── plan.py       # Execution plan types (SwapAction, BridgeAction, ExecutionPlan)
├── planner.py    # Intent → ExecutionPlan builder
└── cli.py        # Command-line interface entry point
```


# ABI Definitions and Usage Patterns

This document describes the conventions for defining contract ABIs and using
them throughout the `pydefi` codebase.  Follow these patterns whenever you add
support for a new protocol or modify an existing one.

---

## 1. Use eth-contract

Never use json abi and `w3.eth.contract` directly, always use eth-contract library with human readable abi,
the details are described in following sections.

## 2. Where ABIs Live

All contract ABI definitions are centralised in `pydefi/abi/`:

| File | Contents |
|------|----------|
| `pydefi/abi/amm.py` | AMM protocols: Uniswap V2/V3, Curve |
| `pydefi/abi/bridge.py` | Bridge protocols: CCTP, GasZip, LayerZero OFT, Stargate, Mayan, Across |
| `pydefi/abi/__init__.py` | Re-exports everything for convenience |

**Never** define ABI strings or `ABIStruct` classes inside protocol-specific
implementation files (`amm/`, `bridge/`, etc.).  Always put them in the
appropriate `pydefi/abi/` module and import from there.

---

## 3. Defining ABIs

### Human-Readable ABI Strings

Use Solidity-style signature strings.  Group them into a single
`Contract.from_abi(...)` call and assign the result to an `ALL_CAPS` module
constant:

```python
# pydefi/abi/amm.py
from eth_contract import Contract

UNISWAP_V2_ROUTER = Contract.from_abi(
    [
        "function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts)",
        "function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
    ]
)
```

### Struct-Typed ABIs (`ABIStruct`)

For contracts whose functions take or return Solidity structs, define the
struct as an `ABIStruct` subclass first, then reference it in the ABI:

```python
# pydefi/abi/amm.py
from typing import Annotated
from eth_contract import ABIStruct, Contract

class ExactInputSingleParams(ABIStruct):
    """Params struct for SwapRouter.exactInputSingle."""

    tokenIn: Annotated[str, "address"]
    tokenOut: Annotated[str, "address"]
    fee: Annotated[int, "uint24"]
    recipient: Annotated[str, "address"]
    amountIn: Annotated[int, "uint256"]
    amountOutMinimum: Annotated[int, "uint256"]

UNISWAP_V3_ROUTER = Contract.from_abi(
    ExactInputSingleParams.human_readable_abi()
    + [
        "function exactInputSingle(ExactInputSingleParams params) external payable returns (uint256 amountOut)",
    ]
)
```

Rules:
- Field types use `Annotated[PythonType, 'solidity_type']`.
- For nested structs, use the inner `ABIStruct` subclass directly as the
  field type (no `Annotated` wrapper needed).
- Name struct classes in `PascalCase` matching the Solidity struct name.
- Name `Contract` constants in `UPPER_SNAKE_CASE`.

---

## 4. Using ABIs in Implementation Files

### The Preferred Pattern: Pass `to=` at Call Time

The ABI definition and the contract address are **intentionally decoupled**.
Import the unbound contract constant and pass the on-chain address directly in
the `.call()` or `.transact()` call via the `to=` keyword argument.  There is
no need to store a bound contract instance:

```python
# pydefi/amm/uniswap_v2.py
from pydefi.abi.amm import UNISWAP_V2_ROUTER

class UniswapV2(BaseAMM):
    def __init__(self, w3, router_address):
        self.w3 = w3
        self.router_address = router_address  # plain string, no binding

    async def get_amounts_out(self, amount_in, path):
        # Pass the address at call time via to=
        return await UNISWAP_V2_ROUTER.fns.getAmountsOut(
            amount_in, path
        ).call(self.w3, to=self.router_address)
```

The same principle applies to `transact()`:

```python
receipt = await UNISWAP_V2_ROUTER.fns.swapExactTokensForTokens(
    amount_in, min_out, path, recipient, deadline
).transact(w3, account, to=router_address)
```

For encoding calldata (no network call), the `to=` address is not part of the
calldata itself and should **not** be passed — just use `.data` directly:

```python
calldata = STARGATE_ROUTER.fns.swap(
    dst_chain, src_pool, dst_pool, ...
).data
# Then include the address separately in the tx dict:
return {"to": router_address, "data": "0x" + calldata.hex(), ...}
```

### Passing Struct Arguments

Instantiate the `ABIStruct` subclass and pass it directly to the function:

```python
from pydefi.abi.amm import QuoteExactInputSingleParams, UNISWAP_V3_QUOTER_V2

params = QuoteExactInputSingleParams(
    tokenIn=token_in.address,
    tokenOut=token_out.address,
    amountIn=amount_in,
    fee=3000,
    sqrtPriceLimitX96=0,
)
result = await UNISWAP_V3_QUOTER_V2.fns.quoteExactInputSingle(params).call(
    w3, to=quoter_address
)
```

---

## 5. Adding a New Protocol

1. Decide which `pydefi/abi/` file the new ABIs belong in (`amm.py` for DEX
   protocols, `bridge.py` for cross-chain protocols, or create a new file for
   a new category such as `lending.py`).

2. Add any `ABIStruct` subclasses first (if the contract uses Solidity structs).

3. Add a module-level `Contract.from_abi(...)` constant for each logical
   contract interface (router, factory, pool, etc.).  **Do not** pass `to=`
   here — the constant is unbound by design.

4. Export the new names from `pydefi/abi/__init__.py`.

5. Import the constant(s) in the implementation file and pass the on-chain
   address via `to=` at call/transact time.

---

## 6. Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| `ABIStruct` class | `PascalCase`, Solidity name | `ExactInputSingleParams` |
| Unbound `Contract` constant | `UPPER_SNAKE_CASE`, protocol + role | `UNISWAP_V3_ROUTER` |
| Address attribute | plain `str` attribute on the class | `self.router_address` |

---

## 7. Example: Adding a New Lending Protocol

```python
# pydefi/abi/lending.py
from typing import Annotated
from eth_contract import ABIStruct, Contract

class BorrowParams(ABIStruct):
    asset: Annotated[str, "address"]
    amount: Annotated[int, "uint256"]
    interestRateMode: Annotated[int, "uint256"]
    referralCode: Annotated[int, "uint16"]
    onBehalfOf: Annotated[str, "address"]

AAVE_V3_POOL = Contract.from_abi(
    BorrowParams.human_readable_abi()
    + [
        "function supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external",
        "function borrow(BorrowParams params) external",
        "function repay(address asset, uint256 amount, uint256 interestRateMode, address onBehalfOf) external returns (uint256)",
    ]
)
```

```python
# pydefi/lending/aave_v3.py
from pydefi.abi.lending import AAVE_V3_POOL

class AaveV3:
    def __init__(self, w3, pool_address):
        self.w3 = w3
        self.pool_address = pool_address

    async def supply(self, asset, amount, recipient, referral=0):
        return await AAVE_V3_POOL.fns.supply(
            asset, amount, recipient, referral
        ).transact(self.w3, account, to=self.pool_address)
```

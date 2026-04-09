# ABI Definitions and Usage Patterns

This document describes the conventions for defining contract ABIs and using
them throughout the `pydefi` codebase.  Follow these patterns whenever you add
support for a new protocol or modify an existing one.

---

## 1. Where ABIs Live

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

## 2. Defining ABIs

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

## 3. Using ABIs in Implementation Files

### Binding a Contract to an Address

`Contract.from_abi(...)` returns an unbound contract class.  Bind it to a
specific on-chain address with the `to=` keyword argument:

```python
# pydefi/amm/uniswap_v2.py
from pydefi.abi.amm import UNISWAP_V2_ROUTER

class UniswapV2(BaseAMM):
    def __init__(self, w3, router_address):
        self._router = UNISWAP_V2_ROUTER(to=router_address)
```

Do **not** call `Contract.from_abi(abi_list, to=address)` directly in
implementation files.  The ABI list belongs in `pydefi/abi/`, and the address
binding happens at instantiation time.

### Calling Contract Functions

```python
# Read call (no gas, returns decoded value)
amounts = await self._router.fns.getAmountsOut(amount_in, path).call(w3)

# State-changing transaction
receipt = await self._router.fns.swapExactTokensForTokens(
    amount_in, min_out, path, recipient, deadline
).transact(w3, account)

# Build calldata without a provider
calldata = self._router.fns.swapExactTokensForTokens(
    amount_in, min_out, path, recipient, deadline
).data
```

### Passing Struct Arguments

Instantiate the `ABIStruct` subclass and pass it directly to the function:

```python
from pydefi.abi.amm import ExactInputSingleParams, UNISWAP_V3_ROUTER

params = ExactInputSingleParams(
    tokenIn=token_in.address,
    tokenOut=token_out.address,
    fee=3000,
    recipient=recipient,
    amountIn=amount_in,
    amountOutMinimum=min_out,
)
result = await self._router.fns.exactInputSingle(params).call(w3)
```

---

## 4. Adding a New Protocol

1. Decide which `pydefi/abi/` file the new ABIs belong in (`amm.py` for DEX
   protocols, `bridge.py` for cross-chain protocols, or create a new file for
   a new category such as `lending.py`).

2. Add any `ABIStruct` subclasses first (if the contract uses Solidity structs).

3. Add a module-level `Contract.from_abi(...)` constant for each logical
   contract interface (router, factory, pool, etc.).

4. Export the new names from `pydefi/abi/__init__.py`.

5. Import the constant(s) in the implementation file and use them with
   `CONTRACT_NAME(to=address)` to bind addresses.

---

## 5. Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| `ABIStruct` class | `PascalCase`, Solidity name | `ExactInputSingleParams` |
| Unbound `Contract` constant | `UPPER_SNAKE_CASE`, protocol + role | `UNISWAP_V3_ROUTER` |
| Bound contract instance | `_snake_case` (private attr) | `self._router` |

---

## 6. Example: Adding a New Lending Protocol

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
        self._pool = AAVE_V3_POOL(to=pool_address)
```

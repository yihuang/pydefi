# ABI Definitions and Usage Patterns

This document describes the conventions for defining contract ABIs and using
them throughout the `pydefi` codebase.  Follow these patterns whenever you add
support for a new protocol or modify an existing one.

---

## 1. Where ABIs Live

All contract ABI definitions are centralised in `pydefi/abi/`,
with different files for different protocol categories.

Always define new ABIs here instead of scattering around.

---

## 2. Defining ABIs

### Human-Readable ABI Strings

Use human readable ABI format. Group them into a single
`Contract.from_abi(...)` call and assign the result to an `ALL_CAPS` module
constant:

```python
# pydefi/abi/amm.py
from eth_contract import Contract

UNISWAP_V2_ROUTER = Contract.from_abi(
    [
        "function getAmountsOut(uint amountIn, address[] path) returns (uint[] amounts)",
        "function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] path, address to, uint deadline) returns (uint[] amounts)",
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
    amountIn: int
    amountOutMinimum: int

UNISWAP_V3_ROUTER = Contract.from_abi(
    ExactInputSingleParams.human_readable_abi()
    + [
        "function exactInputSingle(ExactInputSingleParams params) returns (uint256 amountOut)",
    ]
)
```

**Supported annotation forms:**

| Python annotation | Solidity ABI type |
|---|---|
| `Annotated[T, 'solidity_type']` | explicit type (always works) |
| `bool` | `bool` |
| `int` | `uint256` |
| `str` | `string` |
| `bytes` | `bytes` |
| `list[bool\|int\|str\|bytes]` | `bool[]` / `uint256[]` / … |
| `SomeStruct` (ABIStruct subclass) | `tuple` (nested struct) |
| `list[SomeStruct]` | `tuple[]` (dynamic array of structs) |
| `Annotated[list[SomeStruct], 'SomeStruct[N]']` | `tuple[N]` (fixed-size array of structs) |

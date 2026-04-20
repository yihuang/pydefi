# Smart Contract Interaction Patterns

This document describes the conventions for smart contract interactions.

Always use eth-contract library for smart contract interactions.
Build calldata first, bind to w3 provider and contract address later.

```python
# wrong
from web3 import Web3
contract = w3.eth.contract(address=contract_address, abi=contract_abi)
ret = await contract.functions.someFunction(...).call()
```

```python
# correct
from eth_contract import Contract

# ABI definitions
CONTRACT_ABI = Contract.from_abi([
    "function someFunction(...) returns (...)",
])

# Contract interaction
ret = await CONTRACT_ABI.fns.someFunction(...).call(w3, to=contract_address)
receipt = await CONTRACT_ABI.fns.someFunction(...).transact(w3, acct, to=contract_address)

# Build transaction body manually if needed
tx = {
    "to": contract_address,
    "data": CONTRACT_ABI.fns.someFunction(...).data
}

# Composed in multicall
from eth_contract.multicall3 import multicall
calls = [
    (contract_address, CONTRACT_ABI.fns.someFunction(...)),
    (contract_address, CONTRACT_ABI.fns.anotherFunction(...)),
]
results = await multicall(w3, calls)
```

The advantage of this pattern is that calldata building is pure function,
and it's easy to compose in higher level abstractions.

## Reuse existing ABIs

There are existing ABIs defined in eth-contract library and `pydefi.abi` that you can reuse.
For example, ERC20 ABI is already defined in eth-contract, and Uniswap V2 Router ABI is defined in `pydefi.abi.amm`.
Always check if there's an existing ABI you can reuse before defining a new one.

```python
# Reuse ERC20 ABI from eth-contract
from eth_contract.erc20 import ERC20

# Reuse Uniswap V2 Router ABI from pydefi.abi.amm
from pydefi.abi.amm import UNISWAP_V2_ROUTER
```

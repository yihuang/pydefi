from eth_contract import Contract

DeFiVM = Contract.from_abi(
    [
        "function setParams(bytes32[] params)",
        "function execute(bytes program)",
    ]
)

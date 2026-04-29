from eth_contract import Contract

DeFiVM = Contract.from_abi(["function execute(bytes program, bytes32[] params)"])

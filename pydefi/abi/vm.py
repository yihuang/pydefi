from eth_contract import Contract

DeFiVM = Contract.from_abi(
    [
        "function execute(bytes program)",
    ]
)

# Permit2SupplyRouter (pydefi/vm/Permit2SupplyRouter.sol) — native gasless deposits.
# ``permit`` is Permit2's PermitTransferFrom: ((token, amount) permitted, nonce, deadline).
PERMIT2_SUPPLY_ROUTER = Contract.from_abi(
    [
        "function supply(((address token, uint256 amount) permitted, uint256 nonce, uint256 deadline) permit, address owner, bytes signature, address protocol, bytes supplyData) external",
        "function prime(address token, address protocol) external",
    ]
)

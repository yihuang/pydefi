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

# EIP7702BatchExecutor (pydefi/vm/EIP7702BatchExecutor.sol) — EIP-7702 delegate.
# The owner delegates its EOA here, signs a ``Batch`` of calls, and a sponsor
# submits the type-4 tx; ``execute`` runs the batch in the EOA's own context.
EIP7702_EXECUTOR = Contract.from_abi(
    [
        "function execute((address to, uint256 value, bytes data)[] calls, uint256 nonce, uint256 deadline, bytes signature) external",
        "function batchNonce() view returns (uint256)",
    ]
)

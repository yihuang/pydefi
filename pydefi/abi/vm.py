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

# Uniswap Calibur (https://github.com/Uniswap/calibur) — audited EIP-7702
# delegate singleton: the owner signs an EIP-712 ``SignedBatchedCall`` and a
# sponsor submits ``execute`` as a plain tx targeting the EOA.
CALIBUR = Contract.from_abi(
    [
        "function execute((((address to, uint256 value, bytes data)[] calls, bool revertOnFailure) batchedCall, uint256 nonce, bytes32 keyHash, address executor, uint256 deadline) signedBatchedCall, bytes wrappedSignature) payable",
        "function getSeq(uint256 key) view returns (uint256)",
        "function invalidateNonce(uint256 newNonce)",
    ]
)

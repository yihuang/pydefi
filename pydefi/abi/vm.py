from eth_contract import Contract

# executeWithPermit2: a Permit2 batch witness signature bound to
# keccak256(program) pulls the permitted tokens, then the program runs.
DeFiVM = Contract.from_abi(
    [
        "function execute(bytes program)",
        "function executeWithPermit2(((address token, uint256 amount)[] permitted, uint256 nonce, uint256 deadline) permit, address owner, bytes signature, bytes program) payable",
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

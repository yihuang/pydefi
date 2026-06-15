from eth_contract import Contract

# Safe v1.4.1 (github.com/safe-global/safe-deployments) — the minimal surface
# the counterfactual deposit-address flow uses. Canonical addresses live in
# pydefi.deployments ("SAFE_PROXY_FACTORY" / "SAFE_SINGLETON").
SAFE_PROXY_FACTORY = Contract.from_abi(
    [
        "function createProxyWithNonce(address singleton, bytes initializer, uint256 saltNonce) returns (address)",
        "function proxyCreationCode() pure returns (bytes)",
    ]
)

SAFE = Contract.from_abi(
    [
        "function setup(address[] owners, uint256 threshold, address to, bytes data, address fallbackHandler,"
        " address paymentToken, uint256 payment, address paymentReceiver)",
        "function getOwners() view returns (address[])",
    ]
)

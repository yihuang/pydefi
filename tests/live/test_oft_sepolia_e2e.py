"""End-to-end tests for OFT bridging with DeFiVM compose on Sepolia testnets.

**NOT run in CI** — execute manually::

    pytest -m e2e tests/live/test_oft_sepolia_e2e.py -v -s

Setup (one-time)
----------------
1. The test derives a deterministic deployer account from the well-known test
   mnemonic (Hardhat/Foundry account #0).  Its address is printed at start-up.
   Fund it with at least 0.05 ETH on each of the two chains:

   - Sepolia ETH:    https://cloud.google.com/application/web3/faucet/ethereum/sepolia
   - OP Sepolia ETH: https://app.optimism.io/faucet

2. Override the RPC endpoints via environment variables if the defaults are slow::

       export SEPOLIA_RPC_URL=https://...
       export OP_SEPOLIA_RPC_URL=https://...

3. Run the test once to deploy contracts to deterministic CREATE2 addresses.
   On subsequent runs the existing contracts are reused automatically.

Architecture
------------
Source chain:   Sepolia    (chain ID 11155111, LZ EID 40161)
Dest chain:     OP Sepolia (chain ID 11155420, LZ EID 40232)

Three contracts are deployed at **identical addresses on both chains** using the
Aave CREATE2 factory (same CREATE2 salt + same bytecode/constructor args):

- **DeFiVM**      — stateless opcode interpreter (no constructor args)
- **OFTComposer** — executes DeFiVM programs triggered by lzCompose
- **TestOFT**     — minimal mintable OFT v2 for testing (anyone can mint)

Bridge flow
-----------
1. Mint TestOFT tokens on Sepolia.
2. Call ``TestOFT.send()`` with a DeFiVM bytecode ``composeMsg`` and ``to`` set
   to the OFTComposer address.
3. LayerZero delivers the message to OP Sepolia.
4. ``TestOFT.lzReceive()`` mints tokens to OFTComposer and calls
   ``endpoint.sendCompose()`` to queue the compose.
5. The LayerZero executor calls ``OFTComposer.lzCompose()`` which runs the
   DeFiVM program (saves ``_from`` → R0 and ``amountLD`` → R1).
6. The test polls for the ``Composed`` event on OP Sepolia to confirm delivery.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from eth_abi import encode as abi_encode
from eth_account import Account
from web3 import AsyncWeb3, Web3
from web3.middleware import SignAndSendRawMiddlewareBuilder

from pydefi.vm.program import store_reg

# ---------------------------------------------------------------------------
# Account — deterministic key from the well-known Hardhat/Foundry test mnemonic
# ---------------------------------------------------------------------------

Account.enable_unaudited_hdwallet_features()

#: Standard BIP-39 test mnemonic (Hardhat / Foundry account #0).
#: Never use with real funds.  Fund the derived address on Sepolia/OP Sepolia
#: via a testnet faucet before running these tests.
MNEMONIC = "test test test test test test test test test test test junk"
_ACCOUNT = Account.from_mnemonic(MNEMONIC)

# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

SEPOLIA_CHAIN_ID = 11155111
OP_SEPOLIA_CHAIN_ID = 11155420
SEPOLIA_LZ_EID = 40161
OP_SEPOLIA_LZ_EID = 40232

SEPOLIA_RPC_URL = os.environ.get("SEPOLIA_RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
OP_SEPOLIA_RPC_URL = os.environ.get("OP_SEPOLIA_RPC_URL", "https://sepolia.optimism.io")

#: LayerZero EndpointV2 — deployed at this address on every EVM chain.
LZ_ENDPOINT = "0x6EDCE65403992e310A62460808c4b910D972f10f"

#: Aave CREATE2 factory — available on all standard EVM chains including testnets.
CREATE2_FACTORY = "0x4e59b44847b379578588920cA78FbF26c0B4956C"

_CREATE2_FACTORY_ABI = [
    {
        "inputs": [
            {"name": "salt", "type": "bytes32"},
            {"name": "initcode", "type": "bytes"},
        ],
        "name": "deploy",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "payable",
        "type": "function",
    }
]

# ---------------------------------------------------------------------------
# Contract source paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFI_VM_SOL = _REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"
_OFT_COMPOSER_SOL = _REPO_ROOT / "pydefi" / "bridge" / "OFTComposer.sol"

# ---------------------------------------------------------------------------
# TestOFT — inline Solidity for a minimal mintable OFT v2
# ---------------------------------------------------------------------------

#: Minimal OFT v2 for e2e testing.  Implements the LayerZero OFT protocol
#: without depending on the LayerZero library so it can be compiled by solcx.
#: NOT for production — anyone can mint tokens.
_TEST_OFT_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface ILZEndpointV2 {
    struct MessagingParams {
        uint32 dstEid;
        bytes32 receiver;
        bytes message;
        bytes options;
        bool payInLzToken;
    }
    struct MessagingFee   { uint256 nativeFee; uint256 lzTokenFee; }
    struct MessagingReceipt { bytes32 guid; uint64 nonce; MessagingFee fee; }
    struct Origin         { uint32 srcEid; bytes32 sender; uint64 nonce; }

    function quote(MessagingParams calldata params, address sender)
        external view returns (MessagingFee memory);
    function send(MessagingParams calldata params, address refundAddress)
        external payable returns (MessagingReceipt memory);
    function setDelegate(address delegate) external;
    function sendCompose(address to, bytes32 guid, uint16 index, bytes calldata message) external;
}

/// @title TestOFT
/// @notice Minimal OFT v2 for end-to-end testing on LayerZero testnets.
///         Anyone can mint tokens.  NOT for production use.
contract TestOFT {
    // --- ERC-20 state ---
    string  public name;
    string  public symbol;
    uint8   public constant decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // --- LayerZero state ---
    ILZEndpointV2 public immutable lzEndpoint;
    /// @notice Registered peer OFT on each remote chain (dstEid => bytes32 address).
    mapping(uint32 => bytes32) public peers;
    address public owner;

    // --- OFT constants ---
    // sharedDecimals = 6; conversion: LD (18 dec) <-> SD (6 dec) = factor 10^12
    uint256 private constant SD_FACTOR    = 1e12;
    // Bit 48 of the packed amountSD signals a compose message is present.
    uint64  private constant COMPOSE_FLAG = 0x0001000000000000;

    // --- OFT structs (mirrors IOFT) ---
    struct SendParam {
        uint32 dstEid;
        bytes32 to;
        uint256 amountLD;
        uint256 minAmountLD;
        bytes extraOptions;
        bytes composeMsg;
        bytes oftCmd;
    }
    struct MessagingFee { uint256 nativeFee; uint256 lzTokenFee; }
    struct OFTReceipt   { uint256 amountSentLD; uint256 amountReceivedLD; }

    // --- Events ---
    event Transfer(address indexed from, address indexed to, uint256 value);
    event OFTSent(
        bytes32 indexed guid,
        uint32  dstEid,
        address indexed from,
        uint256 amountLD
    );
    event OFTReceived(
        bytes32 indexed guid,
        uint32  srcEid,
        address indexed to,
        uint256 amountLD
    );

    constructor(
        string memory _name,
        string memory _symbol,
        address _endpoint,
        address _owner
    ) {
        name     = _name;
        symbol   = _symbol;
        lzEndpoint = ILZEndpointV2(_endpoint);
        owner    = _owner;
        // Register owner as the delegate who can configure this OApp.
        ILZEndpointV2(_endpoint).setDelegate(_owner);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "TestOFT: not owner");
        _;
    }

    // --- ERC-20 ---

    /// @notice Anyone can mint test tokens.
    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply   += amount;
        emit Transfer(address(0), to, amount);
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "ERC20: insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to]         += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(allowance[from][msg.sender] >= amount, "ERC20: insufficient allowance");
        require(balanceOf[from] >= amount,             "ERC20: insufficient balance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to]   += amount;
        emit Transfer(from, to, amount);
        return true;
    }

    // --- Admin ---

    function setPeer(uint32 eid, bytes32 peer) external onlyOwner {
        peers[eid] = peer;
    }

    // --- OFT helpers ---

    function _toSD(uint256 amountLD) internal pure returns (uint64) {
        return uint64(amountLD / SD_FACTOR);
    }

    function _toLD(uint64 amountSD) internal pure returns (uint256) {
        return uint256(amountSD) * SD_FACTOR;
    }

    /// @dev Truncate amountLD to the sharedDecimals precision (remove dust).
    function _removeDust(uint256 a) internal pure returns (uint256) {
        return (a / SD_FACTOR) * SD_FACTOR;
    }

    /// @dev Encode: bytes32 to | uint64 amountSD [| composeMsg].
    ///      Sets the COMPOSE_FLAG bit in amountSD when composeMsg is non-empty.
    function _encodeMsg(
        bytes32 to,
        uint64  amountSD,
        bytes memory composeMsg
    ) internal pure returns (bytes memory) {
        if (composeMsg.length > 0) {
            return abi.encodePacked(to, amountSD | COMPOSE_FLAG, composeMsg);
        }
        return abi.encodePacked(to, amountSD);
    }

    // --- IOFT (source side) ---

    function quoteSend(
        SendParam calldata sp,
        bool payInLzToken
    ) external view returns (MessagingFee memory fee) {
        bytes32 peer = peers[sp.dstEid];
        require(peer != bytes32(0), "TestOFT: no peer for dstEid");
        ILZEndpointV2.MessagingFee memory f = lzEndpoint.quote(
            ILZEndpointV2.MessagingParams({
                dstEid:      sp.dstEid,
                receiver:    peer,
                message:     _encodeMsg(sp.to, _toSD(sp.amountLD), sp.composeMsg),
                options:     sp.extraOptions,
                payInLzToken: payInLzToken
            }),
            address(this)
        );
        fee.nativeFee  = f.nativeFee;
        fee.lzTokenFee = f.lzTokenFee;
    }

    function send(
        SendParam calldata sp,
        MessagingFee calldata,
        address refundTo
    ) external payable returns (bytes32 guid, OFTReceipt memory r) {
        uint256 amountLD = _removeDust(sp.amountLD);
        require(amountLD >= sp.minAmountLD,              "TestOFT: slippage");
        require(balanceOf[msg.sender] >= amountLD,       "TestOFT: insufficient balance");

        // Burn source tokens.
        balanceOf[msg.sender] -= amountLD;
        totalSupply           -= amountLD;
        emit Transfer(msg.sender, address(0), amountLD);

        bytes32 peer = peers[sp.dstEid];
        require(peer != bytes32(0), "TestOFT: no peer for dstEid");

        ILZEndpointV2.MessagingReceipt memory mr = lzEndpoint.send{value: msg.value}(
            ILZEndpointV2.MessagingParams({
                dstEid:       sp.dstEid,
                receiver:     peer,
                message:      _encodeMsg(sp.to, _toSD(amountLD), sp.composeMsg),
                options:      sp.extraOptions,
                payInLzToken: false
            }),
            refundTo
        );

        guid               = mr.guid;
        r.amountSentLD     = amountLD;
        r.amountReceivedLD = amountLD;
        emit OFTSent(guid, sp.dstEid, msg.sender, amountLD);
    }

    // --- ILayerZeroReceiver (destination side, called by the LZ endpoint) ---

    function lzReceive(
        ILZEndpointV2.Origin calldata origin,
        bytes32 guid,
        bytes calldata message,
        address,
        bytes calldata
    ) external payable {
        require(msg.sender == address(lzEndpoint), "TestOFT: caller not endpoint");
        require(peers[origin.srcEid] == origin.sender, "TestOFT: invalid peer");
        require(message.length >= 40, "TestOFT: message too short");

        // Layout: bytes32 to | uint64 amountSD [| composeMsg bytes]
        bytes32 toAddr  = bytes32(message[0:32]);
        uint64  rawSD   = uint64(bytes8(message[32:40]));
        bool    hasComp = (rawSD & COMPOSE_FLAG) != 0;
        uint256 amountLD = _toLD(rawSD & ~COMPOSE_FLAG);

        address to = address(uint160(uint256(toAddr)));

        // Mint tokens on destination.
        balanceOf[to] += amountLD;
        totalSupply   += amountLD;
        emit Transfer(address(0), to, amountLD);
        emit OFTReceived(guid, origin.srcEid, to, amountLD);

        if (hasComp) {
            // Build the OFTComposeMsgCodec message expected by OFTComposer:
            //   | 8B nonce | 4B srcEid | 32B amountLD | DeFiVM bytecode |
            lzEndpoint.sendCompose(
                to,
                guid,
                0,
                abi.encodePacked(
                    origin.nonce,       // uint64  → 8 bytes
                    origin.srcEid,      // uint32  → 4 bytes
                    bytes32(amountLD),  // uint256 → 32 bytes
                    message[40:]        // DeFiVM bytecode
                )
            );
        }
    }

    receive() external payable {}
}
"""

# ---------------------------------------------------------------------------
# Solcx compilation helpers
# ---------------------------------------------------------------------------

solcx = pytest.importorskip("solcx", reason="py-solc-x not installed")

_SOLC_VERSION = "0.8.24"

# Module-level cache: compile once across all tests.
_compiled_defi_vm: dict | None = None
_compiled_oft_composer: dict | None = None
_compiled_test_oft: dict | None = None


def _ensure_solc() -> None:
    if _SOLC_VERSION not in solcx.get_installed_solc_versions():
        solcx.install_solc(_SOLC_VERSION, show_progress=False)


def _compile_file(path: Path, contract_name: str) -> dict:
    _ensure_solc()
    result = solcx.compile_files(
        [str(path)],
        output_values=["abi", "bin"],
        solc_version=_SOLC_VERSION,
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(f":{contract_name}"))
    return result[key]


def _compile_source(source: str, contract_name: str) -> dict:
    _ensure_solc()
    result = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=_SOLC_VERSION,
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in result if k.endswith(f":{contract_name}"))
    return result[key]


def _get_compiled() -> tuple[dict, dict, dict]:
    global _compiled_defi_vm, _compiled_oft_composer, _compiled_test_oft
    if _compiled_defi_vm is None:
        print("\n  Compiling contracts (one-time)…")
        _compiled_defi_vm = _compile_file(_DEFI_VM_SOL, "DeFiVM")
        _compiled_oft_composer = _compile_file(_OFT_COMPOSER_SOL, "OFTComposer")
        _compiled_test_oft = _compile_source(_TEST_OFT_SOL, "TestOFT")
    return _compiled_defi_vm, _compiled_oft_composer, _compiled_test_oft


# ---------------------------------------------------------------------------
# CREATE2 helpers
# ---------------------------------------------------------------------------


def _create2_addr(salt: bytes, init_code: bytes) -> str:
    """Compute the CREATE2 deployment address."""
    factory_bytes = bytes.fromhex(CREATE2_FACTORY[2:])
    init_hash = bytes(Web3.keccak(init_code))
    pre_image = b"\xff" + factory_bytes + salt + init_hash
    return Web3.to_checksum_address("0x" + bytes(Web3.keccak(pre_image))[12:].hex())


async def _deploy_create2(w3: AsyncWeb3, deployer: str, salt: bytes, init_code: bytes) -> str:
    """Deploy via the CREATE2 factory. Returns the deployed address."""
    factory = w3.eth.contract(address=CREATE2_FACTORY, abi=_CREATE2_FACTORY_ABI)
    tx_hash = await factory.functions.deploy(salt, init_code).transact({"from": deployer})
    receipt = await w3.eth.get_transaction_receipt(tx_hash)
    assert receipt["status"] == 1, "CREATE2 deployment reverted"
    return _create2_addr(salt, init_code)


async def _ensure_deployed(
    w3: AsyncWeb3,
    deployer: str,
    salt: bytes,
    init_code: bytes,
    label: str = "",
) -> str:
    """Deploy the contract if it doesn't already exist; return its address."""
    addr = _create2_addr(salt, init_code)
    code = await w3.eth.get_code(addr)
    if not code or code == b"":
        print(f"\n    Deploying {label} → {addr} …")
        await _deploy_create2(w3, deployer, salt, init_code)
        print(f"    {label} deployed.")
    else:
        print(f"\n    {label} already at {addr}")
    return addr


# ---------------------------------------------------------------------------
# Deterministic salts
# ---------------------------------------------------------------------------

_SALT_DEFI_VM = bytes(Web3.keccak(text="pydefi:DeFiVM:v1"))
_SALT_OFT_COMPOSER = bytes(Web3.keccak(text="pydefi:OFTComposer:v1"))
_SALT_TEST_OFT = bytes(Web3.keccak(text="pydefi:TestOFT:v1"))


# ---------------------------------------------------------------------------
# LayerZero options encoding
# ---------------------------------------------------------------------------


def _lz_options(lz_receive_gas: int = 200_000, compose_gas: int | None = None) -> bytes:
    """Build LayerZero Type-3 executor options bytes.

    Each option has the format::

        worker_id (1 B) | size (2 B) | option_type (1 B) | data …

    where *size* counts the bytes starting from *option_type*.

    Args:
        lz_receive_gas: Gas budget for ``lzReceive`` on the destination chain.
        compose_gas:    Gas budget for ``lzCompose``; omit for non-compose sends.
    """
    opts = b"\x00\x03"  # Type-3 header

    # lzReceive option  (size = 1 type + 16 gas = 17)
    opts += b"\x01" + (17).to_bytes(2, "big") + b"\x01" + lz_receive_gas.to_bytes(16, "big")

    if compose_gas is not None:
        # lzCompose option  (size = 1 type + 2 index + 16 gas = 19)
        opts += b"\x01" + (19).to_bytes(2, "big") + b"\x03" + (0).to_bytes(2, "big") + compose_gas.to_bytes(16, "big")

    return opts


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------


def _addr_to_bytes32(addr: str) -> bytes:
    """Zero-pad an EVM address to 32 bytes."""
    return bytes.fromhex(addr[2:]).rjust(32, b"\x00")


# ---------------------------------------------------------------------------
# Per-chain setup: deploy + configure
# ---------------------------------------------------------------------------


async def _setup_chain(w3: AsyncWeb3, deployer: str, peer_eid: int) -> dict:
    """Deploy (or reuse) DeFiVM, OFTComposer, and TestOFT on *w3*.

    Also calls ``setPeer`` if the TestOFT doesn't yet know the remote chain.
    Because we use identical CREATE2 salts and the same constructor arguments on
    both chains, all three contracts land at the **same address** on each chain.

    Returns a dict with ``defi_vm_addr``, ``composer_addr``, ``test_oft_addr``,
    ``test_oft`` (web3 contract object), and ``c_oft`` (compiled ABI/bin).
    """
    c_vm, c_composer, c_oft = _get_compiled()

    # 1. DeFiVM — no constructor args
    defi_vm_init = bytes.fromhex(c_vm["bin"])
    defi_vm_addr = await _ensure_deployed(w3, deployer, _SALT_DEFI_VM, defi_vm_init, "DeFiVM")

    # 2. OFTComposer (endpoint, vm, owner)
    composer_init = bytes.fromhex(c_composer["bin"]) + abi_encode(
        ["address", "address", "address"],
        [LZ_ENDPOINT, defi_vm_addr, deployer],
    )
    composer_addr = await _ensure_deployed(w3, deployer, _SALT_OFT_COMPOSER, composer_init, "OFTComposer")

    # 3. TestOFT (name, symbol, endpoint, owner)
    test_oft_init = bytes.fromhex(c_oft["bin"]) + abi_encode(
        ["string", "string", "address", "address"],
        ["TestOFT", "TOFT", LZ_ENDPOINT, deployer],
    )
    test_oft_addr = await _ensure_deployed(w3, deployer, _SALT_TEST_OFT, test_oft_init, "TestOFT")

    # 4. Configure peer (the remote TestOFT — same address thanks to CREATE2)
    test_oft = w3.eth.contract(address=test_oft_addr, abi=c_oft["abi"])
    expected_peer = _addr_to_bytes32(test_oft_addr)
    current_peer = await test_oft.functions.peers(peer_eid).call()
    if current_peer != expected_peer:
        print(f"\n    Setting peer eid={peer_eid} on TestOFT @ {test_oft_addr} …")
        tx = await test_oft.functions.setPeer(peer_eid, expected_peer).transact({"from": deployer})
        await w3.eth.get_transaction_receipt(tx)
        print("    Peer set.")
    else:
        print(f"\n    Peer eid={peer_eid} already configured on TestOFT @ {test_oft_addr}")

    return {
        "defi_vm_addr": defi_vm_addr,
        "composer_addr": composer_addr,
        "test_oft_addr": test_oft_addr,
        "test_oft": test_oft,
        "c_oft": c_oft,
        "c_composer": c_composer,
    }


# ---------------------------------------------------------------------------
# Module-scoped fixture: set up both chains once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def e2e_ctx():
    """Deploy / reuse contracts on Sepolia and OP Sepolia and return context.

    Skips automatically when the test account has no ETH on either chain.
    """
    deployer = _ACCOUNT.address
    sep_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(SEPOLIA_RPC_URL))
    op_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(OP_SEPOLIA_RPC_URL))

    # Inject signing middleware so `transact()` signs and broadcasts automatically.
    sep_w3.middleware_onion.inject(SignAndSendRawMiddlewareBuilder.build(_ACCOUNT.key), layer=0)
    op_w3.middleware_onion.inject(SignAndSendRawMiddlewareBuilder.build(_ACCOUNT.key), layer=0)

    sep_bal = await sep_w3.eth.get_balance(deployer)
    op_bal = await op_w3.eth.get_balance(deployer)

    sep_eth = sep_bal / 1e18
    op_eth = op_bal / 1e18

    print(f"\n{'=' * 60}")
    print(f"  Test account     : {deployer}")
    print(f"  Mnemonic         : {MNEMONIC}")
    print(f"  Sepolia balance  : {sep_eth:.6f} ETH  ({SEPOLIA_RPC_URL})")
    print(f"  OP Sepolia bal.  : {op_eth:.6f} ETH  ({OP_SEPOLIA_RPC_URL})")
    print(f"{'=' * 60}")

    if sep_bal == 0:
        pytest.skip(
            f"No Sepolia ETH — fund {deployer} at https://cloud.google.com/application/web3/faucet/ethereum/sepolia"
        )
    if op_bal == 0:
        pytest.skip(f"No OP Sepolia ETH — fund {deployer} at https://app.optimism.io/faucet")

    # Check that the CREATE2 factory is available on both chains.
    sep_factory_code = await sep_w3.eth.get_code(CREATE2_FACTORY)
    op_factory_code = await op_w3.eth.get_code(CREATE2_FACTORY)
    if not sep_factory_code or sep_factory_code == b"":
        pytest.skip(f"Aave CREATE2 factory not deployed on Sepolia at {CREATE2_FACTORY}")
    if not op_factory_code or op_factory_code == b"":
        pytest.skip(f"Aave CREATE2 factory not deployed on OP Sepolia at {CREATE2_FACTORY}")

    print("\n--- Sepolia setup ---")
    sep = await _setup_chain(sep_w3, deployer, OP_SEPOLIA_LZ_EID)
    print("\n--- OP Sepolia setup ---")
    op = await _setup_chain(op_w3, deployer, SEPOLIA_LZ_EID)

    # Sanity: CREATE2 guarantees identical addresses on both chains.
    assert sep["defi_vm_addr"] == op["defi_vm_addr"], "DeFiVM address mismatch between chains"
    assert sep["composer_addr"] == op["composer_addr"], "OFTComposer address mismatch between chains"
    assert sep["test_oft_addr"] == op["test_oft_addr"], "TestOFT address mismatch between chains"

    print(f"\n  DeFiVM      : {sep['defi_vm_addr']}")
    print(f"  OFTComposer : {sep['composer_addr']}")
    print(f"  TestOFT     : {sep['test_oft_addr']}")

    return {
        "deployer": deployer,
        "sep_w3": sep_w3,
        "op_w3": op_w3,
        "sep": sep,
        "op": op,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestOFTSepoliaE2E:
    """End-to-end tests for OFT bridge with DeFiVM compose on Sepolia testnets.

    All tests in this class share the ``e2e_ctx`` fixture which deploys contracts
    once and is reused across the session.
    """

    # ------------------------------------------------------------------
    # Infrastructure sanity checks
    # ------------------------------------------------------------------

    async def test_contract_addresses_identical_on_both_chains(self, e2e_ctx):
        """CREATE2 guarantees all three contracts share addresses across chains."""
        sep = e2e_ctx["sep"]
        op = e2e_ctx["op"]
        assert sep["defi_vm_addr"] == op["defi_vm_addr"]
        assert sep["composer_addr"] == op["composer_addr"]
        assert sep["test_oft_addr"] == op["test_oft_addr"]

    async def test_lz_endpoint_has_code_on_both_chains(self, e2e_ctx):
        """LayerZero EndpointV2 is deployed at the expected address on both chains."""
        sep_code = await e2e_ctx["sep_w3"].eth.get_code(LZ_ENDPOINT)
        op_code = await e2e_ctx["op_w3"].eth.get_code(LZ_ENDPOINT)
        assert sep_code and sep_code != b"", f"LZ endpoint missing on Sepolia ({LZ_ENDPOINT})"
        assert op_code and op_code != b"", f"LZ endpoint missing on OP Sepolia ({LZ_ENDPOINT})"

    async def test_oft_composer_endpoint_matches_lz(self, e2e_ctx):
        """OFTComposer.endpoint matches the LayerZero EndpointV2 address on each chain."""
        c_composer = e2e_ctx["sep"]["c_composer"]
        composer_addr = e2e_ctx["sep"]["composer_addr"]
        for w3 in (e2e_ctx["sep_w3"], e2e_ctx["op_w3"]):
            composer = w3.eth.contract(address=composer_addr, abi=c_composer["abi"])
            ep = await composer.functions.endpoint().call()
            assert ep.lower() == LZ_ENDPOINT.lower(), f"Endpoint mismatch: {ep} vs {LZ_ENDPOINT}"

    # ------------------------------------------------------------------
    # Mint tokens
    # ------------------------------------------------------------------

    async def test_mint_tokens_on_sepolia(self, e2e_ctx):
        """Anyone can mint TestOFT tokens — verify balance increases."""
        deployer = e2e_ctx["deployer"]
        sep_w3 = e2e_ctx["sep_w3"]
        test_oft = e2e_ctx["sep"]["test_oft"]

        before = await test_oft.functions.balanceOf(deployer).call()
        mint_amount = 10 * 10**18  # 10 TestOFT

        tx = await test_oft.functions.mint(deployer, mint_amount).transact({"from": deployer})
        receipt = await sep_w3.eth.get_transaction_receipt(tx)
        assert receipt["status"] == 1, "mint() reverted"

        after = await test_oft.functions.balanceOf(deployer).call()
        assert after == before + mint_amount
        print(f"\n    Minted {mint_amount / 1e18:.0f} TestOFT; new balance: {after / 1e18:.0f}")

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    async def test_quote_send_returns_nonzero_fee(self, e2e_ctx):
        """TestOFT.quoteSend returns a positive native fee from the LZ endpoint."""
        deployer = e2e_ctx["deployer"]
        test_oft = e2e_ctx["sep"]["test_oft"]
        composer_addr = e2e_ctx["sep"]["composer_addr"]

        amount = 1 * 10**18  # 1 TestOFT
        options = _lz_options(lz_receive_gas=200_000, compose_gas=400_000)
        send_param = (
            OP_SEPOLIA_LZ_EID,
            _addr_to_bytes32(composer_addr),  # to = OFTComposer on OP Sepolia
            amount,
            amount * 95 // 100,  # 5 % slippage tolerance
            options,
            b"",  # composeMsg (placeholder — real value used in bridge test)
            b"",  # oftCmd
        )

        fee_result = await test_oft.functions.quoteSend(send_param, False).call({"from": deployer})
        native_fee = fee_result[0]

        print(f"\n    LZ native fee: {native_fee / 1e18:.8f} ETH ({native_fee} wei)")
        assert isinstance(native_fee, int), "nativeFee must be int"
        assert native_fee > 0, "Expected a positive LayerZero messaging fee"

    # ------------------------------------------------------------------
    # Bridge with compose
    # ------------------------------------------------------------------

    async def test_bridge_send_with_compose(self, e2e_ctx):
        """Send 1 TestOFT from Sepolia to OP Sepolia with a DeFiVM compose program.

        The compose program is minimal: ``store_reg(0) + store_reg(1)`` which
        saves ``_from`` (top of stack) into R0 and ``amountLD`` into R1.

        Asserts that the ``send()`` transaction is accepted on Sepolia (status=1)
        and prints the LayerZero scan URL for the delivery confirmation.
        """
        deployer = e2e_ctx["deployer"]
        sep_w3 = e2e_ctx["sep_w3"]
        test_oft = e2e_ctx["sep"]["test_oft"]
        composer_addr = e2e_ctx["sep"]["composer_addr"]

        # Ensure the deployer has tokens to bridge.
        balance = await test_oft.functions.balanceOf(deployer).call()
        bridge_amount = 1 * 10**18  # 1 TestOFT
        if balance < bridge_amount:
            tx = await test_oft.functions.mint(deployer, bridge_amount).transact({"from": deployer})
            await sep_w3.eth.get_transaction_receipt(tx)

        # DeFiVM compose program:
        #   OFTComposer prologue pushes amountLD (bottom) then _from (top).
        #   store_reg(0) pops _from → R0; store_reg(1) pops amountLD → R1.
        compose_program = store_reg(0) + store_reg(1)

        options = _lz_options(lz_receive_gas=200_000, compose_gas=400_000)
        send_param = (
            OP_SEPOLIA_LZ_EID,
            _addr_to_bytes32(composer_addr),  # to = OFTComposer on OP Sepolia
            bridge_amount,
            bridge_amount * 95 // 100,  # 5 % slippage tolerance
            options,
            compose_program,  # DeFiVM bytecode embedded as composeMsg
            b"",  # oftCmd
        )

        # Get the exact native fee to cover the composed message.
        fee_result = await test_oft.functions.quoteSend(send_param, False).call({"from": deployer})
        native_fee = fee_result[0]
        messaging_fee = (native_fee, 0)

        print(f"\n    Bridging {bridge_amount / 1e18:.0f} TestOFT: Sepolia → OP Sepolia")
        print(f"    Destination: OFTComposer @ {composer_addr}")
        print(f"    LZ fee: {native_fee / 1e18:.8f} ETH")
        print("    Compose: store_reg(0) + store_reg(1)")

        tx_hash = await test_oft.functions.send(send_param, messaging_fee, deployer).transact(
            {"from": deployer, "value": native_fee}
        )
        receipt = await sep_w3.eth.get_transaction_receipt(tx_hash)
        assert receipt["status"] == 1, f"send() reverted — tx: {tx_hash.hex()}"

        tx_hex = tx_hash.hex()
        print(f"\n    ✓ send() succeeded — tx: {tx_hex}")
        print(f"    LayerZero scan: https://layerzeroscan.com/tx/{tx_hex}")

        # Store guid for the delivery-confirmation test.
        e2e_ctx["bridge_tx"] = tx_hex

    # ------------------------------------------------------------------
    # Optional: poll for compose delivery on OP Sepolia
    # ------------------------------------------------------------------

    async def test_compose_delivered_on_op_sepolia(self, e2e_ctx):
        """Poll for the ``Composed`` event on OP Sepolia (timeout = 5 min).

        This test is marked ``xfail`` if the bridge send has not been executed
        yet, or skipped if we time out waiting for LayerZero delivery.
        LayerZero v2 on testnets typically delivers within 1–3 minutes.
        """
        if "bridge_tx" not in e2e_ctx:
            pytest.skip("bridge send test did not run — skipping delivery check")

        op_w3 = e2e_ctx["op_w3"]
        composer_addr = e2e_ctx["sep"]["composer_addr"]
        c_composer = e2e_ctx["sep"]["c_composer"]
        composer = op_w3.eth.contract(address=composer_addr, abi=c_composer["abi"])

        # Build the Composed event topic for log filtering.
        composed_event = composer.events.Composed
        timeout = 300  # 5 minutes
        poll_interval = 15  # seconds

        start = asyncio.get_running_loop().time()
        print(f"\n    Polling for Composed event on OP Sepolia (timeout {timeout}s) …")

        while True:
            elapsed = asyncio.get_running_loop().time() - start
            if elapsed > timeout:
                pytest.skip(f"Composed event not seen within {timeout}s — delivery may still be in-flight")

            latest = await op_w3.eth.block_number
            # Scan the last 200 blocks (~7 min on OP) for the Composed event.
            from_block = max(0, latest - 200)
            logs = await composed_event.get_logs(from_block=from_block, to_block="latest")
            if logs:
                log = logs[-1]
                print("\n    ✓ Composed event received on OP Sepolia!")
                print(f"      from  : {log['args']['from']}")
                print(f"      guid  : {log['args']['guid'].hex()}")
                print(f"      amount: {log['args']['amountLD'] / 1e18:.0f} TestOFT")
                return  # Test passes

            remaining = timeout - elapsed
            print(f"    … waiting (elapsed {int(elapsed)}s, {int(remaining)}s left)")
            await asyncio.sleep(poll_interval)

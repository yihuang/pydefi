"""Anvil fork tests for the cross-chain compose-deposit feature.

Three independent harnesses (each fixture deploys only its own contracts, so a
selective run touches one chain setup):

* CCTP destination — ``CCTPComposer.receiveAndExecute`` runs the dest program,
  crediting the user; covers no-swap deposit, fee-net amount, and ``min_out``.
* Source swap → burn — ``DeFiVM.execute`` swaps WETH→USDC then
  ``depositForBurnWithHook``; the swapped USDC is burned with the dest program
  as hookData.
* CCIP destination — ``CCIPComposer.ccipReceive`` runs the same dest program.

Run with ``pytest -m fork tests/live/test_crosschain_fork.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import solcx
from eth_contract import Contract
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3
from web3.exceptions import ContractLogicError, Web3RPCError

from pydefi.bridge.cctp import CCTP
from pydefi.crosschain.compose import build_compose_deposit_program, build_source_swap_bridge_program
from pydefi.pathfinder.dag import RouteDAG
from pydefi.pathfinder.graph import PoolEdge
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.vm.yield_deposit import AMOUNT_SENTINEL
from tests.live.sol_utils import (
    MOCK_POOLS_SOL,
    MOCK_TOKEN_SOL,
    compile_sol_file,
    compile_sol_source,
    deploy,
    ensure_solc,
)
from tests.live.test_cctp_composer_fork import make_cctp_v2_message

REPO_ROOT = Path(__file__).resolve().parents[2]
CCTP_COMPOSER_SOL = REPO_ROOT / "pydefi" / "bridge" / "CCTPComposer.sol"
CCIP_COMPOSER_SOL = REPO_ROOT / "pydefi" / "bridge" / "CCIPComposer.sol"
DEFI_VM_SOL_FILE = REPO_ROOT / "pydefi" / "vm" / "DeFiVM.sol"


# ---------------------------------------------------------------------------
# Shared mocks/helpers (CCTP + CCIP both supply into a mock Aave pool)
# ---------------------------------------------------------------------------

# Aave-like pool: supply() pulls `amount` via transferFrom and credits onBehalfOf.
MOCK_AAVE_POOL_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MockAavePool {
    mapping(address => uint256) public supplied;
    address public lastOnBehalf;

    function supply(address asset, uint256 amount, address onBehalfOf, uint16) external {
        (bool ok, ) = asset.call(
            abi.encodeWithSignature("transferFrom(address,address,uint256)", msg.sender, address(this), amount)
        );
        require(ok, "pull failed");
        supplied[onBehalfOf] += amount;
        lastOnBehalf = onBehalfOf;
    }
}
"""

_SUPPLY_FN = Contract.from_abi(["function supply(address,uint256,address,uint16)"]).fns.supply


def aave_supply_template(token: Address, user: Address) -> bytes:
    """``supply(token, AMOUNT_SENTINEL, user, 0)`` — runtime amount patched by the compose program."""
    return bytes(_SUPPLY_FN(Web3.to_checksum_address(token), AMOUNT_SENTINEL, Web3.to_checksum_address(user), 0).data)


def contract_at(compiled: dict, address: str) -> Contract:
    return Contract(abi=compiled["abi"], tx={"to": Web3.to_checksum_address(address)})


async def deploy_usdc_and_pool(w3: AsyncWeb3, deployer: str):
    """Deploy a mintable USDC (``MOCK_TOKEN_SOL``) + ``MockAavePool``; return
    ``(usdc, usdc_addr, pool, pool_addr)``."""
    token = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    pool = compile_sol_source(MOCK_AAVE_POOL_SOL, "MockAavePool")
    usdc_addr = await deploy(w3, token, deployer)
    pool_addr = await deploy(w3, pool, deployer)
    return (
        contract_at(token, usdc_addr),
        Address(HexBytes(usdc_addr)),
        contract_at(pool, pool_addr),
        Address(HexBytes(pool_addr)),
    )


# ===========================================================================
# CCTP destination — receiveAndExecute runs the dest program
# ===========================================================================

_ETHEREUM_DOMAIN = 0

# CCTP v2 MessageTransmitterV2 mock: mints (amount - feeExecuted) USDC to the
# mintRecipient (the composer).
_TRANSMITTER_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MockMessageTransmitterV2 {
    uint256 private constant MSG_BODY_OFFSET       = 148;
    uint256 private constant MINT_RECIPIENT_OFFSET = MSG_BODY_OFFSET + 36;
    uint256 private constant AMOUNT_OFFSET         = MSG_BODY_OFFSET + 68;
    uint256 private constant FEE_EXECUTED_OFFSET   = MSG_BODY_OFFSET + 164;
    uint256 private constant MIN_MESSAGE_LENGTH    = MSG_BODY_OFFSET + 228;

    address public immutable usdc;
    constructor(address _usdc) { usdc = _usdc; }

    function receiveMessage(bytes calldata message, bytes calldata) external returns (bool) {
        require(message.length >= MIN_MESSAGE_LENGTH, "short");
        address mintRecipient = address(
            uint160(uint256(bytes32(message[MINT_RECIPIENT_OFFSET:MINT_RECIPIENT_OFFSET + 32])))
        );
        uint256 amount = uint256(bytes32(message[AMOUNT_OFFSET:AMOUNT_OFFSET + 32]));
        uint256 feeExecuted = uint256(bytes32(message[FEE_EXECUTED_OFFSET:FEE_EXECUTED_OFFSET + 32]));
        uint256 amountReceived = amount - feeExecuted;
        if (amountReceived > 0) {
            (bool ok, ) = usdc.call(abi.encodeWithSignature("mint(address,uint256)", mintRecipient, amountReceived));
            require(ok, "mint failed");
        }
        return true;
    }
}
"""


@pytest.fixture(scope="module")
async def cctp_ctx(fork_w3_module, interpreter_addr):
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer, user = accounts[0], accounts[1]

    usdc, usdc_addr, pool, pool_addr = await deploy_usdc_and_pool(w3, deployer)
    transmitter = compile_sol_source(_TRANSMITTER_SOL, "MockMessageTransmitterV2")
    transmitter_addr = await deploy(w3, transmitter, deployer, usdc_addr)
    composer = compile_sol_file(CCTP_COMPOSER_SOL, "CCTPComposer")
    composer_addr = await deploy(w3, composer, deployer, transmitter_addr, usdc_addr, interpreter_addr, deployer)

    return {
        "w3": w3,
        "deployer": deployer,
        "user": Address(HexBytes(user)),
        "usdc": usdc,
        "usdc_addr": usdc_addr,
        "pool": pool,
        "pool_addr": pool_addr,
        "composer": contract_at(composer, composer_addr),
        "composer_addr": Address(HexBytes(composer_addr)),
    }


def _cctp_message(composer: Address, amount: int, program: bytes, nonce: int, fee: int = 0):
    message = make_cctp_v2_message(
        source_domain=_ETHEREUM_DOMAIN,
        nonce=nonce.to_bytes(32, "big"),
        amount=amount,
        mint_recipient=composer,
        hook_data=program,
        fee_executed=fee,
    )
    return message, b"\x00" * 65


def _cctp_program(ctx, *, min_out: int = 0) -> bytes:
    return build_compose_deposit_program(
        supply_template=aave_supply_template(ctx["usdc_addr"], ctx["user"]),
        protocol=ctx["pool_addr"],
        target_token=ctx["usdc_addr"],
        composer=ctx["composer_addr"],
        min_out=min_out,
    )


@pytest.mark.fork
class TestCctpComposeDeposit:
    async def test_no_swap_deposit_credits_user(self, cctp_ctx):
        w3, composer, pool, usdc = cctp_ctx["w3"], cctp_ctx["composer"], cctp_ctx["pool"], cctp_ctx["usdc"]
        amount = 1000 * 10**6
        message, attestation = _cctp_message(cctp_ctx["composer_addr"], amount, _cctp_program(cctp_ctx), nonce=1)

        pre = await pool.fns.supplied(cctp_ctx["user"]).call(w3)
        receipt = await composer.fns.receiveAndExecute(message, attestation).transact(w3, cctp_ctx["deployer"])
        assert receipt["status"] == 1

        assert await pool.fns.supplied(cctp_ctx["user"]).call(w3) == pre + amount
        assert (await pool.fns.lastOnBehalf().call(w3)).lower() == cctp_ctx["user"].to_0x_hex().lower()
        assert await usdc.fns.balanceOf(cctp_ctx["pool_addr"]).call(w3) == amount
        assert await usdc.fns.balanceOf(cctp_ctx["composer_addr"]).call(w3) == 0

    async def test_fee_is_credited_net(self, cctp_ctx):
        w3, composer, pool = cctp_ctx["w3"], cctp_ctx["composer"], cctp_ctx["pool"]
        amount, fee = 500 * 10**6, 2 * 10**6
        message, attestation = _cctp_message(
            cctp_ctx["composer_addr"], amount, _cctp_program(cctp_ctx), nonce=2, fee=fee
        )

        pre = await pool.fns.supplied(cctp_ctx["user"]).call(w3)
        receipt = await composer.fns.receiveAndExecute(message, attestation).transact(w3, cctp_ctx["deployer"])
        assert receipt["status"] == 1
        assert await pool.fns.supplied(cctp_ctx["user"]).call(w3) == pre + (amount - fee)

    async def test_min_out_guard_reverts(self, cctp_ctx):
        amount = 100 * 10**6
        message, attestation = _cctp_message(
            cctp_ctx["composer_addr"], amount, _cctp_program(cctp_ctx, min_out=amount + 1), nonce=3
        )
        with pytest.raises((ContractLogicError, Web3RPCError)):
            await (
                cctp_ctx["composer"]
                .fns.receiveAndExecute(message, attestation)
                .transact(cctp_ctx["w3"], cctp_ctx["deployer"])
            )


# ===========================================================================
# Source swap -> burn — DeFiVM.execute swaps then depositForBurnWithHook
# ===========================================================================

_SRC_COMPOSER = Address(HexBytes("0x" + "C0" * 20))
_DUMMY = Address(HexBytes("0x" + "a0" * 20))

# Records the depositForBurnWithHook call and pulls USDC like the real one.
_MESSENGER_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MockTokenMessenger {
    address public immutable usdc;
    uint256 public lastAmount;
    bytes32 public lastMintRecipient;
    bytes public lastHookData;

    constructor(address _usdc) { usdc = _usdc; }

    function depositForBurnWithHook(
        uint256 amount,
        uint32,
        bytes32 mintRecipient,
        address,
        bytes32,
        uint256,
        uint32,
        bytes calldata hookData
    ) external {
        (bool ok, ) = usdc.call(
            abi.encodeWithSignature("transferFrom(address,address,uint256)", msg.sender, address(this), amount)
        );
        require(ok, "pull failed");
        lastAmount = amount;
        lastMintRecipient = mintRecipient;
        lastHookData = hookData;
    }
}
"""


def _dest_program() -> bytes:
    # A placeholder destination program — only its bytes are checked (carried as
    # hookData); it is never executed on a destination chain in this test.
    return build_compose_deposit_program(
        supply_template=aave_supply_template(_DUMMY, _SRC_COMPOSER),
        protocol=Address(HexBytes("0x" + "CE" * 20)),
        target_token=_DUMMY,
        composer=_SRC_COMPOSER,
    )


@pytest.fixture(scope="module")
async def source_ctx(fork_w3_module, interpreter_addr):
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer = accounts[0]

    vm_addr = await deploy(w3, compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM"), deployer, interpreter_addr)
    vm = w3.eth.contract(address=vm_addr, abi=compile_sol_file(DEFI_VM_SOL_FILE, "DeFiVM")["abi"])

    token_compiled = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
    weth_addr = await deploy(w3, token_compiled, deployer)
    usdc_addr = await deploy(w3, token_compiled, deployer)
    weth = w3.eth.contract(address=weth_addr, abi=token_compiled["abi"])
    usdc = w3.eth.contract(address=usdc_addr, abi=token_compiled["abi"])

    ensure_solc("0.8.24")
    pools = solcx.compile_source(MOCK_POOLS_SOL, output_values=["abi", "bin"], solc_version="0.8.24")
    v2pair_compiled = pools["<stdin>:MockV2Pair"]
    reserve = 10**24
    pair_addr = await deploy(w3, v2pair_compiled, deployer, weth_addr, usdc_addr, reserve, reserve)
    await weth.functions.mint(pair_addr, reserve).transact({"from": deployer})
    await usdc.functions.mint(pair_addr, reserve).transact({"from": deployer})

    messenger_compiled = compile_sol_source(_MESSENGER_SOL, "MockTokenMessenger")
    messenger_addr = await deploy(w3, messenger_compiled, deployer, usdc_addr)
    messenger = Contract(abi=messenger_compiled["abi"], tx={"to": Web3.to_checksum_address(messenger_addr)})

    return {
        "w3": w3,
        "deployer": deployer,
        "vm": vm,
        "vm_addr": Address(HexBytes(vm_addr)),
        "weth": weth,
        "weth_addr": Address(HexBytes(weth_addr)),
        "usdc": usdc,
        "usdc_addr": Address(HexBytes(usdc_addr)),
        "pair_addr": pair_addr,
        "messenger": messenger,
        "messenger_addr": Address(HexBytes(messenger_addr)),
    }


def _source_swap(ctx) -> RouteDAG:
    weth_meta = Token(chain_id=ChainId.ETHEREUM, address=ctx["weth_addr"], symbol="WETH", decimals=18)
    usdc_meta = Token(chain_id=ChainId.ETHEREUM, address=ctx["usdc_addr"], symbol="USDC", decimals=6)
    edge = PoolEdge(
        token_in=weth_meta,
        token_out=usdc_meta,
        pool_address=ctx["pair_addr"],
        protocol="UniswapV2",
        reserve_in=10**24,
        reserve_out=10**24,
        fee_bps=30,
        extra={"is_token0_in": True},  # pair token0 == WETH
    )
    return RouteDAG().from_token(weth_meta).swap(usdc_meta, edge)


async def _burn_template(ctx, dest_program: bytes) -> bytes:
    cctp = CCTP(
        w3=ctx["w3"],
        src_chain_id=ChainId.ETHEREUM,
        dst_chain_id=ChainId.BASE,
        token_messenger_address=ctx["messenger_addr"],
        src_usdc_address=ctx["usdc_addr"],
    )
    src_usdc = Token(chain_id=ChainId.ETHEREUM, address=ctx["usdc_addr"], symbol="USDC", decimals=6)
    burn_tx = await cctp.build_bridge_compose_tx(TokenAmount(src_usdc, 0), _SRC_COMPOSER, dest_program)
    return bytes.fromhex(burn_tx["data"].removeprefix("0x"))


@pytest.mark.fork
class TestSourceSwapBridge:
    async def test_swap_then_burn_sends_swapped_usdc(self, source_ctx):
        w3, vm, messenger = source_ctx["w3"], source_ctx["vm"], source_ctx["messenger"]
        amount_in = 10**18
        await (
            source_ctx["weth"]
            .functions.mint(source_ctx["vm_addr"], amount_in)
            .transact({"from": source_ctx["deployer"]})
        )

        dest_program = _dest_program()
        program = build_source_swap_bridge_program(
            swap_dag=_source_swap(source_ctx),
            amount_in=amount_in,
            usdc=source_ctx["usdc_addr"],
            token_messenger=source_ctx["messenger_addr"],
            burn_template=await _burn_template(source_ctx, dest_program),
            executor=source_ctx["vm_addr"],
        )

        tx = await vm.functions.execute(program).transact({"from": source_ctx["deployer"]})
        receipt = await w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
        assert receipt["status"] == 1

        burned = await messenger.fns.lastAmount().call(w3)
        # Swapped USDC was burned (≈ amount_in minus the 0.3% pool fee), not the WETH input.
        assert 0 < burned < amount_in
        # The messenger pulled exactly that USDC and carries the dest program as hookData.
        assert await source_ctx["usdc"].functions.balanceOf(source_ctx["messenger_addr"]).call() == burned
        assert bytes(await messenger.fns.lastHookData().call(w3)) == dest_program
        mint_recipient = await messenger.fns.lastMintRecipient().call(w3)
        assert mint_recipient[-20:] == bytes(_SRC_COMPOSER)

    async def test_min_usdc_out_guard_reverts(self, source_ctx):
        amount_in = 10**18
        await (
            source_ctx["weth"]
            .functions.mint(source_ctx["vm_addr"], amount_in)
            .transact({"from": source_ctx["deployer"]})
        )

        program = build_source_swap_bridge_program(
            swap_dag=_source_swap(source_ctx),
            amount_in=amount_in,
            usdc=source_ctx["usdc_addr"],
            token_messenger=source_ctx["messenger_addr"],
            burn_template=await _burn_template(source_ctx, _dest_program()),
            executor=source_ctx["vm_addr"],
            min_usdc_out=amount_in,  # swap output is below the input after fees -> revert
        )
        with pytest.raises((ContractLogicError, Web3RPCError)):
            tx = await source_ctx["vm"].functions.execute(program).transact({"from": source_ctx["deployer"]})
            await source_ctx["w3"].eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)


# ===========================================================================
# CCIP destination — ccipReceive runs the same dest program
# ===========================================================================

_SELECTOR = 5009297550715157269  # arbitrary CCIP source-chain selector

# Mock CCIP Router: assembles an Any2EVMMessage and invokes ccipReceive.  Real
# CCIP also transfers the bridged token to the receiver — the test mints it to
# the composer directly.
_ROUTER_SOL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

struct EVMTokenAmount { address token; uint256 amount; }
struct Any2EVMMessage {
    bytes32 messageId;
    uint64 sourceChainSelector;
    bytes sender;
    bytes data;
    EVMTokenAmount[] destTokenAmounts;
}
interface ICCIPComposer { function ccipReceive(Any2EVMMessage calldata message) external payable; }

contract MockRouter {
    function deliverCompose(
        address composer,
        bytes32 messageId,
        uint64 sourceChainSelector,
        bytes calldata sender,
        bytes calldata data,
        address token,
        uint256 amount
    ) external payable {
        EVMTokenAmount[] memory tokens = new EVMTokenAmount[](1);
        tokens[0] = EVMTokenAmount({token: token, amount: amount});
        Any2EVMMessage memory message = Any2EVMMessage({
            messageId: messageId,
            sourceChainSelector: sourceChainSelector,
            sender: sender,
            data: data,
            destTokenAmounts: tokens
        });
        ICCIPComposer(composer).ccipReceive{value: msg.value}(message);
    }
}
"""


@pytest.fixture(scope="module")
async def ccip_ctx(fork_w3_module, interpreter_addr):
    w3 = fork_w3_module
    accounts = await w3.eth.accounts
    deployer, user = accounts[0], accounts[1]

    usdc, usdc_addr, pool, pool_addr = await deploy_usdc_and_pool(w3, deployer)
    router = compile_sol_source(_ROUTER_SOL, "MockRouter")
    router_addr = await deploy(w3, router, deployer)
    composer = compile_sol_file(CCIP_COMPOSER_SOL, "CCIPComposer")
    composer_addr = await deploy(w3, composer, deployer, router_addr, interpreter_addr, deployer, False)

    return {
        "w3": w3,
        "deployer": deployer,
        "user": Address(HexBytes(user)),
        "router": contract_at(router, router_addr),
        "usdc": usdc,
        "usdc_addr": usdc_addr,
        "pool": pool,
        "pool_addr": pool_addr,
        "composer_addr": Address(HexBytes(composer_addr)),
    }


@pytest.mark.fork
class TestCcipComposeDeposit:
    async def test_delivered_token_is_supplied_to_user(self, ccip_ctx):
        w3, router, pool, usdc = ccip_ctx["w3"], ccip_ctx["router"], ccip_ctx["pool"], ccip_ctx["usdc"]
        user, composer_addr = ccip_ctx["user"], ccip_ctx["composer_addr"]
        pool_addr, usdc_addr = ccip_ctx["pool_addr"], ccip_ctx["usdc_addr"]

        amount = 1000 * 10**6
        program = build_compose_deposit_program(
            supply_template=aave_supply_template(usdc_addr, user),
            protocol=pool_addr,
            target_token=usdc_addr,
            composer=composer_addr,
        )

        # CCIP delivers the bridged token to the composer; the mock router only
        # invokes ccipReceive, so mint the token to the composer first.
        await usdc.fns.mint(composer_addr, amount).transact(w3, ccip_ctx["deployer"])

        pre = await pool.fns.supplied(user).call(w3)
        receipt = await router.fns.deliverCompose(
            composer_addr, b"\x01" * 32, _SELECTOR, b"", program, usdc_addr, amount
        ).transact(w3, ccip_ctx["deployer"])
        assert receipt["status"] == 1

        assert await pool.fns.supplied(user).call(w3) == pre + amount
        assert (await pool.fns.lastOnBehalf().call(w3)).lower() == user.to_0x_hex().lower()
        assert await usdc.fns.balanceOf(pool_addr).call(w3) == amount
        assert await usdc.fns.balanceOf(composer_addr).call(w3) == 0

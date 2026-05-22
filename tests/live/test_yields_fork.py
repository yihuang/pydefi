"""Fork tests for :mod:`pydefi.yields` against an Anvil mainnet fork.

Covers the route flows whose on-chain steps are fully built today:

* ``supply_then_bridge`` — entry leg (approve + supply). The bridge tail
  is deferred and not exercised here.
* ``withdraw_then_supply`` — same-chain rebalance from Aave V3 to
  Compound V3 USDC.
* :func:`build_followup_route` — the deferred destination leg of a
  cross-chain route, built from a ``PendingLeg`` and broadcast.
* A full cross-chain route — ``build_yield_route`` → relay → ``build_followup_route``
  — executed end to end across two Anvil nodes with a ``MockBridge`` double.
* :func:`build_compose_supply_route` — its ``[approve, depositForBurnWithHook]``
  source leg broadcast against the live CCTP v2 TokenMessenger.

All reuse :data:`tests.addrs.ETH_WHALE` (vitalik) as the user, top its
ETH balance for gas, and seed USDC from :data:`tests.addrs.USDC_WHALE`.

Run with::

    pytest -m fork tests/live/test_yields_fork.py
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from eth_contract import Contract
from eth_contract.erc20 import ERC20

from pydefi.bridge import CCIP, CCTP, BaseBridge
from pydefi.deployments import get_address
from pydefi.exceptions import BridgeError
from pydefi.lending import AaveV3, CompoundV3
from pydefi.lending.utils import to_tx
from pydefi.types import Address, ChainId, Token, TokenAmount
from pydefi.yields import (
    PendingLeg,
    YieldMarket,
    YieldRoute,
    build_compose_supply_route,
    build_followup_route,
    build_supply_program,
    build_yield_route,
)
from pydefi.yields.router import Protocol
from tests.addrs import ETH_WHALE, USDC
from tests.live.anvil_helpers import fund_usdc, impersonate, send_tx, set_balance
from tests.live.sol_utils import MOCK_TOKEN_SOL, compile_sol_source, deploy

# ---------------------------------------------------------------------------
# Pinned addresses + per-test constants
# ---------------------------------------------------------------------------

COMET_USDC = Address(get_address("COMPOUND_V3_USDC", ChainId.ETHEREUM))

USDC_TEST_AMOUNT = 1_000 * 10**6  # 1000 USDC

# Aave's rayDiv/rayMul both round half-up, so an aToken balance read straight
# after a supply can sit a couple of wei below the principal — the exact
# shortfall depends on the fork block's liquidity index. A small slack keeps
# the "~1:1" assertions stable across fork blocks.
_ATOKEN_SUPPLY_SLACK = 5


# ---------------------------------------------------------------------------
# MockBridge — a BaseBridge test double for the cross-chain end-to-end test
# ---------------------------------------------------------------------------

#: ``bridge`` pulls the token on the source chain; ``deliver`` releases it on
#: the destination — the relay leg the test drives itself.
MOCK_BRIDGE_SOL = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract MockBridge {
    event Bridged(address token, uint256 amount, address recipient);

    function bridge(address token, uint256 amount, address recipient) external {
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        emit Bridged(token, amount, recipient);
    }

    function deliver(address token, uint256 amount, address recipient) external {
        IERC20(token).transfer(recipient, amount);
    }
}
"""

_MOCK_TOKEN = Contract.from_abi(["function mint(address to, uint256 amount) external"])
_MOCK_BRIDGE = Contract.from_abi(
    [
        "function bridge(address token, uint256 amount, address recipient) external",
        "function deliver(address token, uint256 amount, address recipient) external",
    ]
)


class _MockBridge(BaseBridge):
    """A :class:`BaseBridge` double backed by a ``MockBridge.sol`` on the
    source chain — the test drives the relay to the destination."""

    def __init__(self, src_chain_id: int, dst_chain_id: int, contract: Address) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self._contract = contract

    @property
    def protocol_name(self) -> str:
        return "MockBridge"

    @property
    def spender(self) -> Address:
        return self._contract

    async def get_quote(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError  # build_yield_route never calls get_quote

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = _MOCK_BRIDGE.fns.bridge(token_in.address, amount_in.amount, recipient).data
        return to_tx(self._contract, data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usdc_market(
    protocol: Protocol,
    apy: str = "0.04",
    *,
    chain_id: int = ChainId.ETHEREUM,
    token: Token = USDC,
) -> YieldMarket:
    """Synthesized YieldMarket — APY/util/liquidity are placeholders since
    build_yield_route doesn't read them; live versions live in get_yield_markets."""
    return YieldMarket(
        protocol=protocol,
        chain_id=chain_id,
        token=token,
        supply_apy=Decimal(apy),
        utilization=Decimal("0.7"),
        available_liquidity=TokenAmount(token, 10**18),
        market_id=f"{protocol}:{chain_id}:{token.symbol}",
    )


async def _seed_whale_with_usdc(fork_w3, amount: int) -> None:
    await impersonate(fork_w3, ETH_WHALE)
    await set_balance(fork_w3, ETH_WHALE, 100 * 10**18)  # gas
    await fund_usdc(fork_w3, USDC.address, ETH_WHALE, amount)


async def _seed_aave_position(fork_w3, amount: int) -> None:
    """Set up a pre-existing aUSDC position on Aave for rebalance scenarios."""
    aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
    approve_tx = {
        "to": USDC.address,
        "data": "0x" + ERC20.fns.approve(bytes(aave.pool_address), amount).data.hex(),
        "value": "0",
        "gas": "100000",
    }
    r = await send_tx(fork_w3, ETH_WHALE, approve_tx)
    assert r["status"] == 1, "USDC approve to Aave reverted"
    r = await send_tx(fork_w3, ETH_WHALE, aave.build_supply_tx(ETH_WHALE, TokenAmount(USDC, amount)))
    assert r["status"] == 1, "Aave supply reverted"


async def _send_ok(w3, sender: Address, tx: dict[str, Any], label: str) -> dict:
    """Broadcast *tx* via :func:`send_tx` and assert it did not revert."""
    receipt = await send_tx(w3, sender, tx)
    assert receipt["status"] == 1, f"{label} reverted"
    return receipt


async def _broadcast(fork_w3, route: YieldRoute) -> None:
    for step in route.steps:
        await _send_ok(fork_w3, ETH_WHALE, step.tx, f"{step.kind} step")


async def _a_usdc_balance(fork_w3) -> int:
    """Whale's aUSDC balance (Aave V3 receipt token on USDC)."""
    aave = await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)
    a_usdc = (await aave.get_reserve_data(USDC)).a_token_address
    return int(await ERC20.fns.balanceOf(ETH_WHALE).call(fork_w3, to=a_usdc))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fork
class TestYieldRouterFork:
    """End-to-end: build_yield_route + broadcast all steps against a real fork."""

    async def test_supply_then_bridge_entry_leg_mints_atoken(self, fork_w3):
        """[approve, supply] from a supply_then_bridge route must leave the whale holding aUSDC."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)
        before = await _a_usdc_balance(fork_w3)

        route = await build_yield_route(
            "supply_then_bridge",
            user=ETH_WHALE,
            amount_in=TokenAmount(USDC, USDC_TEST_AMOUNT),
            w3s={ChainId.ETHEREUM: fork_w3},
            target_market=_usdc_market("aave_v3"),
            target_chain=ChainId.KITE,
        )
        assert [s.kind for s in route.steps] == ["approve", "supply"]
        await _broadcast(fork_w3, route)

        assert (await _a_usdc_balance(fork_w3)) - before >= USDC_TEST_AMOUNT - _ATOKEN_SUPPLY_SLACK

    async def test_withdraw_then_supply_rebalances_aave_to_compound(self, fork_w3):
        """A withdraw_then_supply route must move the whale's USDC from Aave V3
        into Compound V3 in one [withdraw, approve, supply] sequence."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)
        await _seed_aave_position(fork_w3, USDC_TEST_AMOUNT)

        # Aave's rayDiv/rayMul rounding can shave one wei off the supplied
        # principal, so withdrawing USDC_TEST_AMOUNT exactly trips
        # NotEnoughAvailableUserBalance(). Pull a notch under that — also
        # what a real rebalancer would do after calling get_positions.
        rebalance_amount = USDC_TEST_AMOUNT - 1_000  # 0.001 USDC slack
        a_before = await _a_usdc_balance(fork_w3)
        comet = CompoundV3(w3=fork_w3, chain_id=ChainId.ETHEREUM, comet_address=COMET_USDC)
        comet_before = (await comet.get_user_position(ETH_WHALE)).base_supply.amount

        route = await build_yield_route(
            "withdraw_then_supply",
            user=ETH_WHALE,
            amount_in=TokenAmount(USDC, rebalance_amount),
            w3s={ChainId.ETHEREUM: fork_w3},
            source_market=_usdc_market("aave_v3"),
            target_market=_usdc_market("compound_v3", apy="0.05"),
        )
        assert [s.kind for s in route.steps] == ["withdraw", "approve", "supply"]
        await _broadcast(fork_w3, route)

        a_after = await _a_usdc_balance(fork_w3)
        comet_after = (await comet.get_user_position(ETH_WHALE)).base_supply.amount

        # aUSDC drained by ~rebalance_amount; Compound base supply grew by it.
        # Aave's rayMul/rayDiv adds a few wei of slack each direction, so the
        # tolerance has to absorb both the supply-side and withdraw-side rounding.
        _ROUNDING_SLACK = 100  # 0.0001 USDC
        assert a_before - a_after >= rebalance_amount - _ROUNDING_SLACK
        assert comet_after - comet_before >= rebalance_amount - _ROUNDING_SLACK

    async def test_build_followup_route_supplies_destination(self, fork_w3):
        """build_followup_route turns a cross-chain route's PendingLeg into an
        [approve, supply] leg that, broadcast, mints the destination's aToken.

        A cross-chain route needs a bridge — absent on a single-chain fork —
        so the post-settlement state is built directly: a YieldRoute carrying
        a PendingLeg, exactly what build_followup_route expects once the bridge
        has confirmed and the received amount is known."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)
        before = await _a_usdc_balance(fork_w3)

        market = _usdc_market("aave_v3")
        cross_chain_route = YieldRoute(
            strategy="bridge_then_supply",
            source_chain=ChainId.ETHEREUM,
            steps=(),
            route_id="bridge_then_supply:fork",
            target_market=market,
            target_chain=ChainId.ETHEREUM,
            pending=PendingLeg(ChainId.ETHEREUM, market),
        )
        received = TokenAmount(USDC, USDC_TEST_AMOUNT)
        followup = await build_followup_route(cross_chain_route, ETH_WHALE, received, w3s={ChainId.ETHEREUM: fork_w3})

        assert followup.pending is None
        assert followup.route_id == "followup:bridge_then_supply:fork"
        assert [s.kind for s in followup.steps] == ["approve", "supply"]
        await _broadcast(fork_w3, followup)
        assert (await _a_usdc_balance(fork_w3)) - before >= USDC_TEST_AMOUNT - _ATOKEN_SUPPLY_SLACK

    async def test_cross_chain_route_executes_end_to_end(self, fork_w3, plain_anvil_w3):
        """A full cross-chain route across two Anvil nodes: build_yield_route →
        broadcast source leg → relay → build_followup_route → broadcast.

        ``fork_w3`` is the destination (Ethereum fork, real Aave V3); a
        MockBridge double carries the value and the test drives the relay."""
        dest_w3, src_w3 = fork_w3, plain_anvil_w3
        src_accounts = await src_w3.eth.accounts
        deployer, user = Address(src_accounts[0]), Address(src_accounts[1])

        # Source chain: deploy a mock token + the MockBridge, mint to the user.
        token_artifact = compile_sol_source(MOCK_TOKEN_SOL, "MockToken")
        bridge_artifact = compile_sol_source(MOCK_BRIDGE_SOL, "MockBridge")
        src_token_addr = await deploy(src_w3, token_artifact, deployer)
        src_bridge_addr = await deploy(src_w3, bridge_artifact, deployer)
        mint = _MOCK_TOKEN.fns.mint(user, USDC_TEST_AMOUNT).data
        await _send_ok(src_w3, deployer, to_tx(src_token_addr, mint), "mint source token")

        # Destination chain: deploy the MockBridge, pre-fund it with the USDC
        # the relay releases once the bridge "settles".
        dst_bridge_addr = await deploy(dest_w3, bridge_artifact, deployer)
        await fund_usdc(dest_w3, USDC.address, dst_bridge_addr, USDC_TEST_AMOUNT)

        # Source leg: build the cross-chain route and broadcast it.
        src_token = Token(chain_id=ChainId.BASE, address=src_token_addr, symbol="USDC", decimals=6)
        dst_market = _usdc_market("aave_v3")
        bridge = _MockBridge(ChainId.BASE, ChainId.ETHEREUM, src_bridge_addr)
        route = await build_yield_route(
            "bridge_then_supply",
            user=user,
            amount_in=TokenAmount(src_token, USDC_TEST_AMOUNT),
            w3s={ChainId.BASE: src_w3},
            target_market=dst_market,
            bridge=bridge,
        )
        assert [s.kind for s in route.steps] == ["approve", "bridge"]
        assert route.pending == PendingLeg(ChainId.ETHEREUM, dst_market)
        await _send_ok(src_w3, user, route.steps[0].tx, "approve (source)")
        bridge_receipt = await _send_ok(src_w3, user, route.steps[1].tx, "bridge (source)")
        assert bridge_receipt["logs"], "MockBridge.bridge emitted no Bridged event"

        # Relay: deliver the bridged value on the destination chain.
        deliver = _MOCK_BRIDGE.fns.deliver(USDC.address, USDC_TEST_AMOUNT, user).data
        await _send_ok(dest_w3, deployer, to_tx(dst_bridge_addr, deliver), "relay deliver")

        # Destination leg: build_followup_route against what actually arrived.
        received = TokenAmount(USDC, int(await ERC20.fns.balanceOf(user).call(dest_w3, to=USDC.address)))
        assert received.amount == USDC_TEST_AMOUNT, "relay did not deliver the bridged amount"
        followup = await build_followup_route(route, user, received, w3s={ChainId.ETHEREUM: dest_w3})
        assert [s.kind for s in followup.steps] == ["approve", "supply"]

        aave = await AaveV3.from_chain(dest_w3, ChainId.ETHEREUM)
        a_usdc = (await aave.get_reserve_data(USDC)).a_token_address
        a_before = int(await ERC20.fns.balanceOf(user).call(dest_w3, to=a_usdc))
        for step in followup.steps:
            await _send_ok(dest_w3, user, step.tx, f"{step.kind} (destination)")
        a_after = int(await ERC20.fns.balanceOf(user).call(dest_w3, to=a_usdc))
        assert a_after - a_before >= USDC_TEST_AMOUNT - _ATOKEN_SUPPLY_SLACK

    async def test_bridge_then_supply_broadcasts_a_real_ccip_send(self, fork_w3):
        """A bridge_then_supply route built with a real CCIP bridge: its
        [approve, ccipSend] source leg must be accepted by the live CCIP
        Router on the mainnet fork — CCIP's off-chain delivery is out of scope.

        Skipped if the Ethereum->Arbitrum lane rejects the USDC transfer
        (not pool-allowlisted / rate-limited) — a CCIP fact, not a pydefi bug."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)

        usdc_arb = Token(chain_id=ChainId.ARBITRUM, address=USDC.address, symbol="USDC", decimals=6)
        bridge = CCIP(fork_w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)
        try:
            route = await build_yield_route(
                "bridge_then_supply",
                user=ETH_WHALE,
                amount_in=TokenAmount(USDC, USDC_TEST_AMOUNT),
                w3s={ChainId.ETHEREUM: fork_w3},
                target_market=_usdc_market("aave_v3", chain_id=ChainId.ARBITRUM, token=usdc_arb),
                bridge=bridge,
            )
        except BridgeError as exc:
            pytest.skip(f"CCIP Ethereum->Arbitrum lane rejected the USDC transfer: {exc}")

        assert [s.kind for s in route.steps] == ["approve", "bridge"]
        approve_step, bridge_step = route.steps
        assert Address(bridge_step.tx["to"]) == bridge.spender  # the real CCIP Router

        await _send_ok(fork_w3, ETH_WHALE, approve_step.tx, "approve (CCIP)")
        receipt = await _send_ok(fork_w3, ETH_WHALE, bridge_step.tx, "ccipSend")
        assert receipt["logs"], "ccipSend accepted but emitted no CCIPMessageSent"

    async def test_compose_supply_broadcasts_a_real_cctp_burn(self, fork_w3):
        """A compose_supply route built with a real CCTP bridge: its
        [approve, depositForBurnWithHook] source leg — the burn carrying a real
        build_supply_program DeFiVM program as hookData — must be accepted by
        the live CCTP v2 TokenMessenger on the mainnet fork. CCTP's off-chain
        attestation and the destination compose are out of scope; the
        CCTPComposer side is covered by test_cctp_composer_fork.py."""
        await _seed_whale_with_usdc(fork_w3, USDC_TEST_AMOUNT)

        # The destination Aave Pool is resolved on the dst chain, which a
        # single-chain Ethereum fork can't reach — pin it to the live mainnet
        # Aave V3 Pool so build_supply_program emits genuine DeFiVM bytecode.
        # The program rides as opaque CCTP hookData; this source leg never runs it.
        pool = (await AaveV3.from_chain(fork_w3, ChainId.ETHEREUM)).pool_address
        usdc_base = Token(
            chain_id=ChainId.BASE, address=Address(CCTP.usdc_address(ChainId.BASE)), symbol="USDC", decimals=6
        )
        bridge = CCTP(fork_w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.BASE)

        with patch("pydefi.yields.compose._supply_target", new=AsyncMock(return_value=pool)):
            route = await build_compose_supply_route(
                user=ETH_WHALE,
                amount_in=TokenAmount(USDC, USDC_TEST_AMOUNT),
                target_market=_usdc_market("aave_v3", chain_id=ChainId.BASE, token=usdc_base),
                composer_address=Address("0x" + "C0" * 20),
                bridge=bridge,
                w3s={ChainId.BASE: fork_w3},
            )

        assert route.strategy == "compose_supply"
        assert route.pending is None  # the destination supply rides in the hook
        assert [s.kind for s in route.steps] == ["approve", "bridge"]
        approve_step, bridge_step = route.steps
        assert Address(bridge_step.tx["to"]) == Address(bridge.token_messenger_address)
        program = build_supply_program("aave_v3", pool, usdc_base, ETH_WHALE)
        assert program.hex() in bridge_step.tx["data"].lower(), "hookData does not carry the supply program"

        await _send_ok(fork_w3, ETH_WHALE, approve_step.tx, "approve (CCTP)")
        receipt = await _send_ok(fork_w3, ETH_WHALE, bridge_step.tx, "depositForBurnWithHook")
        assert receipt["logs"], "depositForBurnWithHook accepted but emitted no MessageSent"

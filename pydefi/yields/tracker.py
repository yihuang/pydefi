"""Read a user's existing supply positions and wait for cross-chain
bridge settlements. Pairs balances with their :class:`YieldMarket` so the
result feeds directly into :func:`build_yield_route('withdraw_then_supply')`."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3

from pydefi.abi.lending import COMPOUND_V3_COMET
from pydefi.deployments import comet_contract_for, get_address
from pydefi.exceptions import BridgeError
from pydefi.lending.aave_v3 import AaveV3
from pydefi.lending.aave_v4 import AaveV4
from pydefi.lending.morpho import MorphoBlue
from pydefi.types import Address, TokenAmount
from pydefi.yields.router import Protocol, YieldMarket, aave_v4_ref, get_yield_markets, morpho_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Position:
    """``balance`` is denominated in the underlying token (not aTokens /
    Comet base supply units) so a rebalance route can be built directly."""

    market: YieldMarket
    balance: TokenAmount


async def _aave_balance(market: YieldMarket, user: Address, w3: AsyncWeb3) -> TokenAmount:
    aave = await AaveV3.from_chain(w3, market.chain_id)
    user_reserve = await aave.get_user_reserve_data(user, market.token)
    return user_reserve.a_token_balance


async def _compound_balance(market: YieldMarket, user: Address, w3: AsyncWeb3) -> TokenAmount:
    # Direct balanceOf — get_user_position would also iterate every collateral
    # via collateralBalanceOf, wasted RPC for a pure supply read.
    comet_address = get_address(comet_contract_for(market.token.symbol), market.chain_id)
    balance = await COMPOUND_V3_COMET.fns.balanceOf(user).call(w3, to=comet_address)
    return TokenAmount(token=market.token, amount=int(balance))


async def _morpho_balance(market: YieldMarket, user: Address, w3: AsyncWeb3) -> TokenAmount:
    morpho = MorphoBlue.from_chain(w3, market.chain_id)
    params = await morpho_params(morpho, market)
    position = await morpho.get_position(user, params)
    return position.supply_assets


async def _aave_v4_balance(market: YieldMarket, user: Address, w3: AsyncWeb3) -> TokenAmount:
    spoke_name, reserve_id = aave_v4_ref(market)
    v4 = AaveV4.from_chain(w3, market.chain_id, spoke_name)
    return (await v4.get_user_reserve(reserve_id, user, market.token)).supplied


async def get_positions(
    user: Address,
    token_symbol: str,
    w3s: dict[int, AsyncWeb3],
    chains: list[int] | None = None,
    protocols: list[Protocol] | None = None,
) -> list[Position]:
    """*user*'s non-zero supply positions in *token_symbol*, sorted by
    balance descending."""
    markets = await get_yield_markets(token_symbol, w3s, chains, protocols)
    logger.debug("get_positions: scanning %d markets for %s on %s", len(markets), token_symbol, user.hex())
    out: list[Position] = []
    for market in markets:
        w3 = w3s[market.chain_id]
        if market.protocol == "aave_v3":
            balance = await _aave_balance(market, user, w3)
        elif market.protocol == "compound_v3":
            balance = await _compound_balance(market, user, w3)
        elif market.protocol == "aave_v4":
            balance = await _aave_v4_balance(market, user, w3)
        elif market.protocol == "morpho":
            balance = await _morpho_balance(market, user, w3)
        else:
            raise ValueError(f"unknown protocol: {market.protocol!r}")
        if balance.amount > 0:
            logger.debug("get_positions: %s balance=%s", market.market_id, balance.amount)
            out.append(Position(market=market, balance=balance))
    out.sort(key=lambda p: p.balance.amount, reverse=True)
    logger.info("get_positions: %d non-zero positions for %s on %s", len(out), token_symbol, user.hex())
    return out


async def wait_for_bridge_settlement(
    w3: AsyncWeb3,
    token_address: Address,
    recipient: Address,
    *,
    starting_balance: int,
    min_delta: int,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 5.0,
) -> int:
    """Poll until *recipient*'s ERC-20 balance has grown by at least
    *min_delta*. Lucid's ``transferTo`` has no destination-side callback,
    so the caller has to watch the wrapped token balance directly.
    Capture ``starting_balance`` *before* broadcasting the source bridge tx.
    Raises :class:`BridgeError` if ``timeout_seconds`` elapses first."""
    logger.info(
        "wait_for_bridge_settlement: 0x%s on token 0x%s, starting_balance=%d, min_delta=%d, timeout=%.0fs",
        recipient.hex(),
        token_address.hex(),
        starting_balance,
        min_delta,
        timeout_seconds,
    )
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    polls = 0
    while True:
        balance = int(await ERC20.fns.balanceOf(recipient).call(w3, to=token_address))
        delta = balance - starting_balance
        polls += 1
        if delta >= min_delta:
            logger.info("wait_for_bridge_settlement: settled after %d poll(s), delta=%d", polls, delta)
            return balance
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            logger.warning(
                "wait_for_bridge_settlement: timeout after %d poll(s), delta=%d < %d",
                polls,
                delta,
                min_delta,
            )
            raise BridgeError(
                f"bridge settlement timeout: balance of {recipient.hex()} grew by "
                f"{delta}, expected ≥ {min_delta} within {timeout_seconds}s"
            )
        logger.debug("wait_for_bridge_settlement: poll %d delta=%d (need ≥ %d)", polls, delta, min_delta)
        await asyncio.sleep(min(poll_interval_seconds, remaining))

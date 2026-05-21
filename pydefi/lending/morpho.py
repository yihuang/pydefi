"""
Morpho Blue lending protocol integration.

Morpho Blue is a single immutable contract hosting an unbounded number of
fully isolated lending markets. Each market is identified by an immutable
:class:`MarketParams` tuple — ``(loanToken, collateralToken, oracle, irm,
lltv)`` — whose ``keccak256(abi.encode(...))`` is its ``bytes32`` Id, so
every operation is scoped to one market and carries its full params.

There are no aTokens or cTokens: accounting is ERC-4626-style shares with
virtual shares/assets (:data:`VIRTUAL_SHARES` / :data:`VIRTUAL_ASSETS`) that
block share-price manipulation on an empty market.

Docs: https://docs.morpho.org/
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from eth_contract.erc20 import ERC20
from eth_utils import keccak
from web3 import AsyncWeb3

from pydefi.abi.lending import MORPHO_BLUE, MORPHO_IRM, MORPHO_ORACLE, MorphoMarketParams, MorphoMarketStruct
from pydefi.lending.utils import WAD, per_second_rate_to_apy, to_tx
from pydefi.types import ZERO_ADDRESS, Address, Token, TokenAmount

#: Morpho scales every oracle price by 1e36 (``ORACLE_PRICE_SCALE``). The
#: loan-token value of a collateral balance is ``collateral * price / 1e36``.
ORACLE_PRICE_SCALE: int = 10**36

#: Virtual shares/assets added to every market's accounting. They make the
#: initial share price well-defined and immune to the classic ERC-4626
#: empty-vault donation attack.
VIRTUAL_SHARES: int = 10**6
VIRTUAL_ASSETS: int = 1


# ---------------------------------------------------------------------------
# Interest accrual math — pure, mirrors Morpho's on-chain libs
# ---------------------------------------------------------------------------
#
# Morpho's Solidity uses MathLib.mulDivDown/Up and SharesMathLib helpers
# because EVM integer math can overflow; Python ints can't, so those map to
# plain ``a * b // c`` (floor) and ``(a * b + d - 1) // d`` (ceil) inline at
# each use site. Shares ↔ assets always carry the VIRTUAL_SHARES /
# VIRTUAL_ASSETS offsets.


def _w_taylor_compounded(rate_per_second: int, elapsed: int) -> int:
    """Morpho's 3-term Taylor approximation of ``e^(rate*elapsed) - 1``.

    Both *rate_per_second* and the result are WAD (1e18)-scaled. Matches
    ``MathLib.wTaylorCompounded`` used inside on-chain interest accrual.
    """
    first = rate_per_second * elapsed
    second = first * first // (2 * WAD)
    third = second * first // (3 * WAD)
    return first + second + third


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketParams:
    """The immutable definition of one Morpho Blue market.

    A market is created once via ``createMarket`` and its parameters can
    never change. The market's ``Id`` is :attr:`id`.

    Attributes:
        loan_token: The asset suppliers lend and borrowers borrow.
        collateral_token: The asset borrowers post as collateral. It earns
            no interest and cannot be borrowed.
        oracle: ``IOracle`` contract pricing collateral in loan-token terms
            (scaled by 1e36). May be the zero address for collateral-free
            "idle" markets used by MetaMorpho.
        irm: Interest Rate Model contract (usually the AdaptiveCurveIRM).
        lltv: Liquidation loan-to-value, WAD-scaled (``0.86e18`` = 86%).
    """

    loan_token: Token
    collateral_token: Token
    oracle: Address
    irm: Address
    lltv: int

    @property
    def id(self) -> bytes:
        """The market's ``bytes32`` Id — ``keccak256(abi.encode(self))``.

        Field order (loanToken, collateralToken, oracle, irm, lltv) is part
        of the hash, so swapping any two fields yields a different market.
        """
        return keccak(_params_struct(self).encode())


@dataclass
class MarketState:
    """A market's raw on-chain accounting state (the ``market()`` getter).

    Morpho accrues interest lazily, so these totals reflect the last
    ``lastUpdate`` checkpoint. :func:`accrue_interest` advances them to a
    later timestamp when an exact figure is needed.

    Attributes:
        total_supply_assets: Total loan tokens supplied (principal + interest).
        total_supply_shares: Total supply shares outstanding.
        total_borrow_assets: Total loan tokens borrowed (principal + interest).
        total_borrow_shares: Total borrow shares outstanding.
        last_update: Unix timestamp of the last interest accrual.
        fee: Protocol fee on borrow interest, WAD-scaled.
    """

    total_supply_assets: int
    total_supply_shares: int
    total_borrow_assets: int
    total_borrow_shares: int
    last_update: int
    fee: int

    @property
    def utilization(self) -> Decimal:
        """``total_borrow_assets / total_supply_assets`` in ``[0, 1]``."""
        if self.total_supply_assets == 0:
            return Decimal(0)
        return Decimal(self.total_borrow_assets) / Decimal(self.total_supply_assets)


@dataclass
class MorphoMarket:
    """A snapshot of one market — state plus derived rates (cf. Aave's
    ``ReserveData``).

    Attributes:
        params: The market's :class:`MarketParams`.
        state: Raw :class:`MarketState` read on-chain.
        supply_apy: Annualised supply rate (``0.045`` = 4.5%).
        borrow_apy: Annualised borrow rate.
        utilization: ``total_borrow_assets / total_supply_assets``.
        liquidity: Loan tokens idle in the market and immediately borrowable.
    """

    params: MarketParams
    state: MarketState
    supply_apy: Decimal
    borrow_apy: Decimal
    utilization: Decimal
    liquidity: TokenAmount


@dataclass
class MorphoPosition:
    """A user's position in one market.

    Attributes:
        market: The market's :class:`MarketParams`.
        supply_shares: Raw supply shares held.
        borrow_shares: Raw borrow shares held.
        supply_assets: Supply balance in loan tokens (shares → assets, down).
        borrow_assets: Debt in loan tokens (shares → assets, up).
        collateral: Collateral balance in collateral tokens.
        health_factor: ``maxBorrow / borrowed``; the position is liquidatable
            below ``1``. :attr:`decimal.Decimal('Infinity')` when there is
            no debt.
    """

    market: MarketParams
    supply_shares: int
    borrow_shares: int
    supply_assets: TokenAmount
    borrow_assets: TokenAmount
    collateral: TokenAmount
    health_factor: Decimal


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _params_struct(params: MarketParams) -> MorphoMarketParams:
    """Build the ABI :class:`MorphoMarketParams` struct from *params*."""
    return MorphoMarketParams(
        loanToken=params.loan_token.address,
        collateralToken=params.collateral_token.address,
        oracle=params.oracle,
        irm=params.irm,
        lltv=params.lltv,
    )


def _market_struct(state: MarketState) -> MorphoMarketStruct:
    """Build the ABI :class:`MorphoMarketStruct` from a :class:`MarketState`."""
    return MorphoMarketStruct(
        totalSupplyAssets=state.total_supply_assets,
        totalSupplyShares=state.total_supply_shares,
        totalBorrowAssets=state.total_borrow_assets,
        totalBorrowShares=state.total_borrow_shares,
        lastUpdate=state.last_update,
        fee=state.fee,
    )


def accrue_interest(state: MarketState, borrow_rate_per_second: int, elapsed: int) -> MarketState:
    """Return *state* advanced by *elapsed* seconds of interest accrual.

    Mirrors Morpho's on-chain ``_accrueInterest``: borrow interest accrues to
    both total borrow and total supply assets, and a non-zero market fee mints
    fee shares into total supply shares. *borrow_rate_per_second* is the
    WAD-scaled IRM rate (see :meth:`MorphoBlue.get_borrow_rate`). A zero
    *elapsed*, or a market with no debt, returns *state* unchanged.
    """
    if elapsed <= 0 or state.total_borrow_assets == 0:
        return state
    interest = state.total_borrow_assets * _w_taylor_compounded(borrow_rate_per_second, elapsed) // WAD
    total_supply_assets = state.total_supply_assets + interest
    total_borrow_assets = state.total_borrow_assets + interest
    total_supply_shares = state.total_supply_shares
    if state.fee != 0:
        fee_amount = interest * state.fee // WAD
        # The fee recipient is credited fee_amount worth of supply shares
        # (toSharesDown). totalSupplyAssets already includes the fee, hence
        # the subtraction in the denominator.
        fee_shares = (
            fee_amount * (total_supply_shares + VIRTUAL_SHARES) // (total_supply_assets - fee_amount + VIRTUAL_ASSETS)
        )
        total_supply_shares += fee_shares
    return MarketState(
        total_supply_assets=total_supply_assets,
        total_supply_shares=total_supply_shares,
        total_borrow_assets=total_borrow_assets,
        total_borrow_shares=state.total_borrow_shares,
        last_update=state.last_update + elapsed,
        fee=state.fee,
    )


def compute_health_factor(collateral: int, borrowed: int, collateral_price: int, lltv: int) -> Decimal:
    """Return a position's health factor — ``maxBorrow / borrowed``.

    ``maxBorrow = collateral * collateral_price / 1e36 * lltv / 1e18``. The
    position becomes liquidatable once the health factor drops below ``1``.
    A position with no debt is infinitely healthy.

    Args:
        collateral: Collateral balance in collateral-token units.
        borrowed: Debt in loan-token units.
        collateral_price: Oracle price, 1e36-scaled (see :func:`MORPHO_ORACLE`).
        lltv: Liquidation LTV, WAD-scaled.
    """
    if borrowed == 0:
        return Decimal("Infinity")
    collateral_value = collateral * collateral_price // ORACLE_PRICE_SCALE
    max_borrow = collateral_value * lltv // WAD
    return Decimal(max_borrow) / Decimal(borrowed)


def supply_apy_from_borrow(borrow_rate_per_second: int, state: MarketState) -> Decimal:
    """Derive the supply APY from the borrow rate and a market's state.

    Suppliers earn the borrow rate scaled by utilization and net of the
    protocol fee: ``supplyRate = borrowRate * utilization * (1 - fee)``.
    """
    if state.total_supply_assets == 0:
        return Decimal(0)
    util_wad = state.total_borrow_assets * WAD // state.total_supply_assets
    supply_rate = borrow_rate_per_second * util_wad // WAD * (WAD - state.fee) // WAD
    return per_second_rate_to_apy(supply_rate)


def _resolve_assets_shares(assets: TokenAmount | None, shares: int | None) -> tuple[int, int]:
    """Validate the Morpho ``(assets, shares)`` pair — exactly one non-zero.

    Morpho's supply / withdraw / borrow / repay take both an ``assets`` and
    a ``shares`` argument and require exactly one to be zero. pydefi exposes
    that as two optional keyword arguments; this enforces the XOR.
    """
    if (assets is None) == (shares is None):
        raise ValueError("pass exactly one of assets= or shares=")
    if assets is not None:
        return assets.amount, 0
    assert shares is not None  # narrowed by the XOR above
    return 0, shares


# ---------------------------------------------------------------------------
# MorphoBlue client
# ---------------------------------------------------------------------------


class MorphoBlue:
    """Client for the Morpho Blue singleton on one chain.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the target chain.
        chain_id: EVM chain ID.
        morpho_address: Address of the Morpho Blue core contract.
        protocol_name: Override the default ``"MorphoBlue"`` name.
    """

    protocol_name: str = "MorphoBlue"

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        morpho_address: Address,
        protocol_name: str = "MorphoBlue",
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.morpho_address = morpho_address
        self.protocol_name = protocol_name

    @classmethod
    def from_chain(cls, w3: AsyncWeb3, chain_id: int, protocol_name: str = "MorphoBlue") -> MorphoBlue:
        """Construct from the pydefi deployment registry.

        Synchronous — the Morpho Blue core is an immutable singleton, so
        nothing has to be resolved on-chain (unlike ``AaveV3.from_chain``).
        """
        from pydefi.deployments import address_for

        return cls(
            w3=w3,
            chain_id=chain_id,
            morpho_address=address_for("MORPHO_BLUE", chain_id),
            protocol_name=protocol_name,
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_market_params(
        self,
        market_id: bytes,
        tokens: dict[Address, Token] | None = None,
    ) -> MarketParams:
        """Resolve a market's :class:`MarketParams` from its ``bytes32`` Id.

        Args:
            market_id: The 32-byte market Id.
            tokens: Optional address → :class:`Token` map for human-readable
                loan / collateral metadata. Addresses absent from the map
                have their ``decimals`` fetched on-chain and ``symbol`` set
                to ``"?"``.
        """
        tokens = tokens or {}
        loan_raw, collateral_raw, oracle_raw, irm_raw, lltv = await MORPHO_BLUE.fns.idToMarketParams(market_id).call(
            self.w3, to=self.morpho_address
        )
        loan_addr, collateral_addr = Address(loan_raw), Address(collateral_raw)
        loan, collateral = await asyncio.gather(
            self._resolve_token(loan_addr, tokens),
            self._resolve_token(collateral_addr, tokens),
        )
        return MarketParams(
            loan_token=loan,
            collateral_token=collateral,
            oracle=Address(oracle_raw),
            irm=Address(irm_raw),
            lltv=lltv,
        )

    async def _resolve_token(self, address: Address, tokens: dict[Address, Token]) -> Token:
        cached = tokens.get(address)
        if cached is not None:
            return cached
        decimals = await ERC20.fns.decimals().call(self.w3, to=address)
        return Token(chain_id=self.chain_id, address=address, symbol="?", decimals=decimals)

    async def get_market_state(self, market: MarketParams | bytes) -> MarketState:
        """Return a market's raw on-chain :class:`MarketState`."""
        market_id = market.id if isinstance(market, MarketParams) else market
        raw = await MORPHO_BLUE.fns.market(market_id).call(self.w3, to=self.morpho_address)
        return MarketState(*raw)

    async def get_borrow_rate(self, params: MarketParams, state: MarketState | None = None) -> int:
        """Return the per-second borrow rate (WAD-scaled) from the market's IRM.

        Returns ``0`` for a market whose ``irm`` is the zero address. When
        *state* is omitted it is fetched first — pass it to save a call.
        """
        if params.irm == ZERO_ADDRESS:
            return 0
        if state is None:
            state = await self.get_market_state(params)
        return await MORPHO_IRM.fns.borrowRateView(_params_struct(params), _market_struct(state)).call(
            self.w3, to=params.irm
        )

    async def get_market(self, params: MarketParams) -> MorphoMarket:
        """Return a :class:`MorphoMarket` snapshot — state plus derived APYs.

        Uses the instantaneous on-chain ``market()`` state; pending interest
        is not simulated and does not change the current rate.
        """
        state = await self.get_market_state(params)
        borrow_rate = await self.get_borrow_rate(params, state)
        liquidity = max(0, state.total_supply_assets - state.total_borrow_assets)
        return MorphoMarket(
            params=params,
            state=state,
            supply_apy=supply_apy_from_borrow(borrow_rate, state),
            borrow_apy=per_second_rate_to_apy(borrow_rate),
            utilization=state.utilization,
            liquidity=TokenAmount(token=params.loan_token, amount=liquidity),
        )

    async def get_position(self, user: Address, params: MarketParams) -> MorphoPosition:
        """Return *user*'s :class:`MorphoPosition` in the market.

        Supply / borrow shares are converted to assets against the raw
        on-chain market state; the health factor uses the market oracle's
        current price. For a market untouched for a long time, accrue first
        with :func:`accrue_interest` for an exact figure.
        """
        market_id = params.id
        has_oracle = params.oracle != ZERO_ADDRESS
        tasks: list[Any] = [
            MORPHO_BLUE.fns.position(market_id, user).call(self.w3, to=self.morpho_address),
            MORPHO_BLUE.fns.market(market_id).call(self.w3, to=self.morpho_address),
        ]
        if has_oracle:
            tasks.append(MORPHO_ORACLE.fns.price().call(self.w3, to=params.oracle))
        results = await asyncio.gather(*tasks)

        supply_shares, borrow_shares, collateral = results[0]
        state = MarketState(*results[1])
        price = results[2] if has_oracle else 0

        # Shares → assets with the virtual shares/assets offsets (Morpho
        # SharesMathLib): supply rounds down, debt rounds up so a borrower
        # can never shed dust for free.
        supply_assets = (
            supply_shares * (state.total_supply_assets + VIRTUAL_ASSETS) // (state.total_supply_shares + VIRTUAL_SHARES)
        )
        borrow_den = state.total_borrow_shares + VIRTUAL_SHARES
        borrow_assets = (borrow_shares * (state.total_borrow_assets + VIRTUAL_ASSETS) + borrow_den - 1) // borrow_den
        return MorphoPosition(
            market=params,
            supply_shares=supply_shares,
            borrow_shares=borrow_shares,
            supply_assets=TokenAmount(token=params.loan_token, amount=supply_assets),
            borrow_assets=TokenAmount(token=params.loan_token, amount=borrow_assets),
            collateral=TokenAmount(token=params.collateral_token, amount=collateral),
            health_factor=compute_health_factor(collateral, borrow_assets, price, params.lltv),
        )

    # ------------------------------------------------------------------
    # Writes — return tx dicts {to, data, value}
    # ------------------------------------------------------------------

    def build_supply_tx(
        self,
        params: MarketParams,
        *,
        assets: TokenAmount | None = None,
        shares: int | None = None,
        on_behalf_of: Address,
        data: bytes = b"",
    ) -> dict[str, Any]:
        """Supply loan tokens. Pass exactly one of *assets* / *shares*."""
        amount, share_amount = _resolve_assets_shares(assets, shares)
        call_data = MORPHO_BLUE.fns.supply(_params_struct(params), amount, share_amount, on_behalf_of, data).data
        return to_tx(self.morpho_address, call_data)

    def build_withdraw_tx(
        self,
        params: MarketParams,
        *,
        assets: TokenAmount | None = None,
        shares: int | None = None,
        on_behalf_of: Address,
        receiver: Address,
    ) -> dict[str, Any]:
        """Withdraw supplied loan tokens. Pass exactly one of *assets* /
        *shares* — a full withdrawal uses ``shares=`` (the position's
        ``supply_shares``), since Morpho has no max sentinel."""
        amount, share_amount = _resolve_assets_shares(assets, shares)
        call_data = MORPHO_BLUE.fns.withdraw(_params_struct(params), amount, share_amount, on_behalf_of, receiver).data
        return to_tx(self.morpho_address, call_data)

    def build_supply_collateral_tx(
        self,
        params: MarketParams,
        assets: TokenAmount,
        on_behalf_of: Address,
        data: bytes = b"",
    ) -> dict[str, Any]:
        """Supply collateral tokens (collateral earns no interest)."""
        call_data = MORPHO_BLUE.fns.supplyCollateral(_params_struct(params), assets.amount, on_behalf_of, data).data
        return to_tx(self.morpho_address, call_data)

    def build_withdraw_collateral_tx(
        self,
        params: MarketParams,
        assets: TokenAmount,
        on_behalf_of: Address,
        receiver: Address,
    ) -> dict[str, Any]:
        """Withdraw collateral tokens (reverts if it would leave the position
        unhealthy)."""
        call_data = MORPHO_BLUE.fns.withdrawCollateral(
            _params_struct(params), assets.amount, on_behalf_of, receiver
        ).data
        return to_tx(self.morpho_address, call_data)

    def build_borrow_tx(
        self,
        params: MarketParams,
        *,
        assets: TokenAmount | None = None,
        shares: int | None = None,
        on_behalf_of: Address,
        receiver: Address,
    ) -> dict[str, Any]:
        """Borrow loan tokens against posted collateral. Pass exactly one of
        *assets* / *shares*."""
        amount, share_amount = _resolve_assets_shares(assets, shares)
        call_data = MORPHO_BLUE.fns.borrow(_params_struct(params), amount, share_amount, on_behalf_of, receiver).data
        return to_tx(self.morpho_address, call_data)

    def build_repay_tx(
        self,
        params: MarketParams,
        *,
        assets: TokenAmount | None = None,
        shares: int | None = None,
        on_behalf_of: Address,
        data: bytes = b"",
    ) -> dict[str, Any]:
        """Repay borrowed loan tokens. Pass exactly one of *assets* /
        *shares* — a full repayment uses ``shares=`` (the position's
        ``borrow_shares``)."""
        amount, share_amount = _resolve_assets_shares(assets, shares)
        call_data = MORPHO_BLUE.fns.repay(_params_struct(params), amount, share_amount, on_behalf_of, data).data
        return to_tx(self.morpho_address, call_data)

    def build_set_authorization_tx(self, authorized: Address, is_authorized: bool) -> dict[str, Any]:
        """Grant or revoke *authorized*'s right to manage the sender's positions."""
        call_data = MORPHO_BLUE.fns.setAuthorization(authorized, is_authorized).data
        return to_tx(self.morpho_address, call_data)

    def build_create_market_tx(self, params: MarketParams) -> dict[str, Any]:
        """Create a new market. Permissionless, but *params.irm* and
        *params.lltv* must already be enabled by Morpho governance."""
        call_data = MORPHO_BLUE.fns.createMarket(_params_struct(params)).data
        return to_tx(self.morpho_address, call_data)

    def build_flashloan_tx(
        self,
        token: Token,
        amount: TokenAmount,
        data: bytes = b"",
    ) -> dict[str, Any]:
        """Build a fee-free ``flashLoan``.

        The caller must implement ``onMorphoFlashLoan(uint256 assets, bytes
        data)`` and approve Morpho to pull ``amount`` back within that
        callback. Morpho flash loans charge no premium.
        """
        call_data = MORPHO_BLUE.fns.flashLoan(token.address, amount.amount, data).data
        return to_tx(self.morpho_address, call_data)


__all__ = [
    "ORACLE_PRICE_SCALE",
    "VIRTUAL_ASSETS",
    "VIRTUAL_SHARES",
    "MarketParams",
    "MarketState",
    "MorphoBlue",
    "MorphoMarket",
    "MorphoPosition",
    "accrue_interest",
    "compute_health_factor",
    "supply_apy_from_borrow",
]

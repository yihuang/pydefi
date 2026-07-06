"""Liquidation discovery and execution across Aave V3, Aave V4, Compound III
and Morpho Blue.

pydefi is local-first: this module does **not** crawl an indexer for every
underwater borrower. The caller supplies the candidate positions to inspect —
from a subgraph query, an event scan, or a watchlist — and
:class:`LiquidationRouter` checks each one's health on the live node,
returning a :class:`LiquidationOpportunity` with a prebuilt transaction for
those that are liquidatable.

The four protocols liquidate differently, so each has its own candidate type:
:class:`MorphoCandidate`, :class:`AaveV3Candidate`, :class:`AaveV4Candidate`
and :class:`CompoundV3Candidate` — see each for the inputs it needs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from web3 import AsyncWeb3

from pydefi.lending.aave_v3 import AaveV3
from pydefi.lending.aave_v4 import AaveV4
from pydefi.lending.compound_v3 import CompoundV3
from pydefi.lending.morpho import MarketParams, MorphoBlue, max_liquidation
from pydefi.types import Address, Token, TokenAmount

logger = logging.getLogger(__name__)

Protocol = Literal["aave_v3", "compound_v3", "morpho", "aave_v4"]


# ---------------------------------------------------------------------------
# Candidate inputs — one per protocol, supplied by the caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AaveV3Candidate:
    """An Aave V3 position to inspect for liquidatability.

    Attributes:
        chain_id: Chain the Aave V3 deployment lives on.
        user: The borrower.
        collateral: Collateral asset to seize. Aave's ``liquidationCall``
            acts on a single collateral / debt pair, so the caller picks it.
        debt: Debt asset to repay.
        receive_atoken: Receive the seized collateral as the reserve's aToken
            instead of the underlying.
    """

    chain_id: int
    user: Address
    collateral: Token
    debt: Token
    receive_atoken: bool = False


@dataclass(frozen=True)
class AaveV4Candidate:
    """An Aave V4 position to inspect for liquidatability.

    Attributes:
        chain_id: Chain the Spoke lives on.
        spoke: Registry suffix of the Spoke (``"MAIN_SPOKE"``, …).
        user: The borrower.
        collateral_reserve_id: Reserve index of the collateral to seize.
        debt_reserve_id: Reserve index of the debt to repay.
        receive_shares: Receive the seized collateral as Spoke supply shares
            instead of the underlying token.
    """

    chain_id: int
    spoke: str
    user: Address
    collateral_reserve_id: int
    debt_reserve_id: int
    receive_shares: bool = False


@dataclass(frozen=True)
class CompoundV3Candidate:
    """A Compound III position to inspect for liquidatability.

    Attributes:
        chain_id: Chain the Comet market lives on.
        base_symbol: Base-asset symbol identifying the Comet market (``"USDC"``).
        account: The borrower.
    """

    chain_id: int
    base_symbol: str
    account: Address


@dataclass(frozen=True)
class MorphoCandidate:
    """A Morpho Blue position to inspect for liquidatability.

    Attributes:
        chain_id: Chain the Morpho Blue singleton lives on.
        market: The isolated market's immutable :class:`MarketParams`.
        borrower: The borrower.
    """

    chain_id: int
    market: MarketParams
    borrower: Address


#: Any of the four protocol-specific candidate inputs.
LiquidationCandidate = AaveV3Candidate | AaveV4Candidate | CompoundV3Candidate | MorphoCandidate


# ---------------------------------------------------------------------------
# Opportunity output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidationOpportunity:
    """A position confirmed liquidatable on-chain, ready to execute.

    Attributes:
        protocol: Which protocol the position belongs to.
        chain_id: Chain it lives on.
        borrower: The account being liquidated.
        health_factor: The position's health factor — liquidatable below
            ``1``. ``None`` for Compound III, which exposes only a boolean
            ``isLiquidatable``.
        debt_to_repay: Debt the liquidation clears, for sizing the debt-asset
            approval — the chosen-reserve debt for Aave (close-factor-capped
            on-chain), the full borrow balance for Morpho (an upper bound in the
            bad-debt seize-all case), Compound's total base debt.
        collateral_to_seize: The collateral asset received. ``None`` for
            Compound III — ``absorb`` seizes every collateral, not one asset.
        tx: The liquidation transaction (``{to, data, value}``), prebuilt for
            the router's liquidator. Send it from that liquidator, which must
            first approve the protocol to pull the debt asset (not needed for
            Compound's ``absorb``, which moves no funds).
    """

    protocol: Protocol
    chain_id: int
    borrower: Address
    health_factor: Decimal | None
    debt_to_repay: TokenAmount
    collateral_to_seize: Token | None
    tx: dict[str, Any]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class LiquidationRouter:
    """Checks caller-supplied positions for liquidatability across protocols.

    Args:
        w3s: Map of chain id → :class:`~web3.AsyncWeb3`. A candidate on a
            chain absent from this map is skipped by
            :meth:`find_liquidatable_positions` and raises in the single
            ``check_*`` methods.
        liquidator: The account that sends the liquidation transactions — the
            Compound III ``absorb`` ``absorber`` and the implicit collateral
            recipient elsewhere.
    """

    def __init__(self, w3s: dict[int, AsyncWeb3], liquidator: Address) -> None:
        self.w3s = w3s
        self.liquidator = liquidator
        # AaveV3.from_chain reads Pool / DataProvider / Oracle off the chain;
        # memoize the in-flight resolve per chain so a batch of candidates
        # shares one resolve instead of three RPCs apiece. Caching the Task
        # (not the resolved client) collapses concurrent same-chain first-misses
        # without a lock — the get/set in _aave_v3_client has no await between
        # them, so it is atomic across coroutines, and distinct chains still
        # resolve concurrently. The other clients' from_chain is a registry
        # lookup, so they need no cache.
        self._aave_v3: dict[int, asyncio.Task[AaveV3]] = {}

    def _w3(self, chain_id: int) -> AsyncWeb3:
        w3 = self.w3s.get(chain_id)
        if w3 is None:
            raise ValueError(f"no AsyncWeb3 configured for chain {chain_id}")
        return w3

    async def _aave_v3_client(self, chain_id: int) -> AaveV3:
        """Return a cached :class:`AaveV3` for *chain_id*, resolving it once.

        Concurrent same-chain first-misses share one in-flight resolve; a failed
        resolve is evicted so the next candidate retries rather than inheriting a
        cached failure.
        """
        task = self._aave_v3.get(chain_id)
        if task is None:
            task = asyncio.ensure_future(AaveV3.from_chain(self._w3(chain_id), chain_id))
            self._aave_v3[chain_id] = task
        try:
            return await task
        except Exception:  # noqa: BLE001 — don't cache a failed resolve
            # Evict only our own task: a concurrent awaiter of the same failure
            # may already have evicted it and a retry cached a fresh resolve.
            if self._aave_v3.get(chain_id) is task:
                del self._aave_v3[chain_id]
            raise

    # ------------------------------------------------------------------
    # Per-protocol checks — return None when the position is healthy
    # ------------------------------------------------------------------

    async def check_morpho(self, candidate: MorphoCandidate) -> LiquidationOpportunity | None:
        """Return an opportunity if the Morpho position is liquidatable, else ``None``.

        Repays the whole debt by shares when solvent; on a bad-debt position
        (full repayment would over-seize and revert) seizes all collateral
        instead — see :func:`~pydefi.lending.morpho.max_liquidation`.
        """
        morpho = MorphoBlue.from_chain(self._w3(candidate.chain_id), candidate.chain_id)
        position = await morpho.get_position(candidate.borrower, candidate.market)
        if position.health_factor >= 1:
            return None
        seized_assets, repaid_shares = max_liquidation(position)
        tx = morpho.build_liquidate_tx(
            candidate.market, candidate.borrower, seized_assets=seized_assets, repaid_shares=repaid_shares
        )
        return LiquidationOpportunity(
            protocol="morpho",
            chain_id=candidate.chain_id,
            borrower=candidate.borrower,
            health_factor=position.health_factor,
            debt_to_repay=position.borrow_assets,
            collateral_to_seize=candidate.market.collateral_token,
            tx=tx,
        )

    async def check_aave_v3(self, candidate: AaveV3Candidate) -> LiquidationOpportunity | None:
        """Return an opportunity if the Aave V3 position is liquidatable, else
        ``None`` — repaying the close-factor maximum of the chosen debt reserve.

        Raises :class:`ValueError` if the borrower has no *candidate.debt* debt
        or no *candidate.collateral* enabled — ``liquidationCall`` would revert.
        """
        aave = await self._aave_v3_client(candidate.chain_id)
        account = await aave.get_user_account_data(candidate.user)
        if account.health_factor >= 1:
            return None
        debt_reserve, collateral_reserve = await asyncio.gather(
            aave.get_user_reserve_data(candidate.user, candidate.debt),
            aave.get_user_reserve_data(candidate.user, candidate.collateral),
        )
        if debt_reserve.variable_debt.amount == 0:
            raise ValueError(
                f"Aave V3 candidate {candidate.user} carries no {candidate.debt.symbol} debt — "
                "liquidationCall would revert"
            )
        if collateral_reserve.a_token_balance.amount == 0 or not collateral_reserve.usage_as_collateral_enabled:
            raise ValueError(
                f"Aave V3 candidate {candidate.user} has no {candidate.collateral.symbol} collateral "
                "enabled — liquidationCall would seize nothing"
            )
        tx = aave.build_liquidation_call_tx(
            candidate.collateral,
            (candidate.debt, "max"),
            candidate.user,
            receive_atoken=candidate.receive_atoken,
        )
        return LiquidationOpportunity(
            protocol="aave_v3",
            chain_id=candidate.chain_id,
            borrower=candidate.user,
            health_factor=account.health_factor,
            debt_to_repay=debt_reserve.variable_debt,
            collateral_to_seize=candidate.collateral,
            tx=tx,
        )

    async def check_aave_v4(self, candidate: AaveV4Candidate) -> LiquidationOpportunity | None:
        """Return an opportunity if the Aave V4 position is liquidatable, else
        ``None`` — covering the user's full debt on the chosen reserve. V4 caps
        what it liquidates at restoring the target health factor and pays a
        dynamic (Dutch-auction) bonus that grows the more underwater the
        position is — not V3's fixed close factor / bonus.

        Raises :class:`ValueError` if the borrower has no debt in
        *candidate.debt_reserve_id* or no collateral in
        *candidate.collateral_reserve_id* — ``liquidationCall`` would revert.
        """
        v4 = AaveV4.from_chain(self._w3(candidate.chain_id), candidate.chain_id, candidate.spoke)
        account = await v4.get_user_account_data(candidate.user)
        if account.health_factor >= 1:
            return None
        debt_reserve, collateral_reserve = await asyncio.gather(
            v4.get_user_reserve(candidate.debt_reserve_id, candidate.user),
            v4.get_user_reserve(candidate.collateral_reserve_id, candidate.user),
        )
        if debt_reserve.debt.amount == 0:
            raise ValueError(
                f"Aave V4 candidate {candidate.user} carries no debt in reserve "
                f"{candidate.debt_reserve_id} — liquidationCall would revert"
            )
        if collateral_reserve.supplied.amount == 0 or not collateral_reserve.using_as_collateral:
            raise ValueError(
                f"Aave V4 candidate {candidate.user} has no collateral enabled in reserve "
                f"{candidate.collateral_reserve_id} — liquidationCall would seize nothing"
            )
        tx = v4.build_liquidation_call_tx(
            candidate.collateral_reserve_id,
            candidate.debt_reserve_id,
            candidate.user,
            debt_reserve.debt,
            receive_shares=candidate.receive_shares,
        )
        return LiquidationOpportunity(
            protocol="aave_v4",
            chain_id=candidate.chain_id,
            borrower=candidate.user,
            health_factor=account.health_factor,
            debt_to_repay=debt_reserve.debt,
            collateral_to_seize=collateral_reserve.supplied.token,
            tx=tx,
        )

    async def check_compound_v3(self, candidate: CompoundV3Candidate) -> LiquidationOpportunity | None:
        """Return an opportunity if the Compound III account is liquidatable,
        else ``None``. The opportunity's transaction is ``absorb`` — buying
        the absorbed collateral back at the discount is a separate step."""
        comet = CompoundV3.from_chain(self._w3(candidate.chain_id), candidate.chain_id, candidate.base_symbol)
        position = await comet.get_user_position(candidate.account)
        if not position.is_liquidatable:
            return None
        tx = comet.build_absorb_tx(self.liquidator, [candidate.account])
        return LiquidationOpportunity(
            protocol="compound_v3",
            chain_id=candidate.chain_id,
            borrower=candidate.account,
            health_factor=None,
            debt_to_repay=position.base_borrow,
            collateral_to_seize=None,
            tx=tx,
        )

    # ------------------------------------------------------------------
    # Batch discovery
    # ------------------------------------------------------------------

    async def _check(self, candidate: LiquidationCandidate) -> LiquidationOpportunity | None:
        """Dispatch one candidate to its protocol-specific check."""
        if isinstance(candidate, MorphoCandidate):
            return await self.check_morpho(candidate)
        if isinstance(candidate, AaveV3Candidate):
            return await self.check_aave_v3(candidate)
        if isinstance(candidate, AaveV4Candidate):
            return await self.check_aave_v4(candidate)
        if isinstance(candidate, CompoundV3Candidate):
            return await self.check_compound_v3(candidate)
        raise TypeError(f"unknown candidate type: {type(candidate).__name__}")

    async def _safe_check(self, candidate: LiquidationCandidate) -> LiquidationOpportunity | None:
        """Run :meth:`_check`, downgrading any failure to ``None`` + a log
        line so one bad RPC does not sink the whole batch."""
        if candidate.chain_id not in self.w3s:
            logger.warning("skipping %s — no RPC for chain %s", type(candidate).__name__, candidate.chain_id)
            return None
        try:
            return await self._check(candidate)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully per candidate
            logger.warning("liquidation check failed for %r: %s", candidate, exc)
            return None

    async def find_liquidatable_positions(self, candidates: list[LiquidationCandidate]) -> list[LiquidationOpportunity]:
        """Check every candidate concurrently and return the liquidatable ones.

        Healthy positions, candidates on a chain missing from ``w3s``, and
        candidates whose on-chain read fails are dropped (the latter two with a
        log line). Order is preserved — sort by
        :attr:`LiquidationOpportunity.health_factor` for most-underwater-first.
        """
        results = await asyncio.gather(*(self._safe_check(c) for c in candidates))
        return [opp for opp in results if opp is not None]


__all__ = [
    "AaveV3Candidate",
    "AaveV4Candidate",
    "CompoundV3Candidate",
    "LiquidationCandidate",
    "LiquidationOpportunity",
    "LiquidationRouter",
    "MorphoCandidate",
    "Protocol",
]

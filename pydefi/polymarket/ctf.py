"""On-chain Conditional Tokens Framework (CTF) actions for Polymarket.

Outcome tokens are ERC1155 minted by the Gnosis ConditionalTokens contract,
collateralised 1:1 by USDC. All operations are on-chain, independent of the CLOB:

* **split** — lock N USDC, mint N of each outcome token (a complete set).
* **merge** — burn a complete set, unlock N USDC (the inverse of split).
* **redeem** — after resolution, burn winning tokens for their USDC payout.
* **convert** — neg-risk only: turn No positions into the complementary Yes
  positions plus collateral.

Negative-risk markets route every operation through the NegRiskAdapter (it wraps
the USDC itself, so you approve the adapter and pass just a conditionId + amount)
instead of the raw CTF. :class:`PolymarketCTF` hides that behind a ``neg_risk``
flag and resolves all addresses from :mod:`pydefi.deployments`.

Resolve outcome-token ids from the Gamma API (``clobTokenIds``) or
:meth:`PolymarketCTF.get_position_id` — local derivation needs alt_bn128 math.

Docs: https://docs.polymarket.com/trading/ctf/overview
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from eth_contract.erc20 import ERC20
from eth_utils import keccak
from hexbytes import HexBytes
from web3 import AsyncWeb3

from pydefi._utils import to_tx
from pydefi.abi.predict import CONDITIONAL_TOKENS, NEG_RISK_ADAPTER
from pydefi.deployments import get_address, get_token
from pydefi.types import ZERO_HASH, Address, Hash, Token

#: Index sets for a Polymarket binary market: bit 0 = Yes, bit 1 = No.
#: ``splitPosition`` / ``mergePositions`` take this as the ``partition`` and
#: ``redeemPositions`` (raw CTF) takes it as the ``indexSets``.
BINARY_PARTITION: tuple[int, ...] = (1, 2)

#: Polymarket markets are never nested, so the parent collection is always zero.
PARENT_COLLECTION_ID: Hash = ZERO_HASH

Bytes32 = Hash | bytes | str


def _to_bytes32(value: Bytes32) -> bytes:
    """Normalise a bytes32-ish value (hex string, bytes, or ``HexBytes``)."""
    b = HexBytes(value)
    if len(b) != 32:
        raise ValueError(f"expected a 32-byte value, got {len(b)} bytes: {value!r}")
    return b


def compute_condition_id(oracle: Address, question_id: Bytes32, outcome_slot_count: int = 2) -> Hash:
    """``keccak256(oracle, questionId, outcomeSlotCount)`` — the CTF's conditionId.

    Only this id is a plain hash; ``collectionId`` / ``positionId`` need alt_bn128
    math, so get outcome-token ids from the Gamma API or
    :meth:`PolymarketCTF.get_position_id`.
    """
    packed = oracle + _to_bytes32(question_id) + outcome_slot_count.to_bytes(32, "big")
    return Hash(keccak(packed))


class PolymarketCTF:
    """On-chain CTF actions for Polymarket, bound to a chain.

    Resolves the ConditionalTokens / NegRiskAdapter / collateral addresses from
    :mod:`pydefi.deployments`, builds ``{to, data, value}`` calldata for split /
    merge / redeem / convert, and adds the async reads that need a provider.
    Standard and negative-risk markets share one surface — pass ``neg_risk=True``
    to route through the NegRiskAdapter instead of the raw CTF.

    The ``build_*_tx`` methods are pure (no I/O), so an instance built with
    explicit addresses works without a provider; ``w3`` is only used by the reads.

    Args:
        w3: :class:`~web3.AsyncWeb3` for the target chain.
        chain_id: EVM chain id (137 Polygon mainnet, 80002 Amoy).
        conditional_tokens: Override the CTF address (else from the registry).
        neg_risk_adapter: Override the NegRiskAdapter address (mainnet only).
        collateral: Override the collateral token (else USDC.e from the registry).
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        chain_id: int,
        conditional_tokens: Address | None = None,
        neg_risk_adapter: Address | None = None,
        collateral: Token | None = None,
    ) -> None:
        self.w3 = w3
        self.chain_id = chain_id
        self.conditional_tokens = conditional_tokens or get_address("POLYMARKET_CONDITIONAL_TOKENS", chain_id)
        self._neg_risk_adapter = neg_risk_adapter
        self.collateral = collateral or get_token("USDC.e", chain_id)

    @classmethod
    def from_chain(cls, w3: AsyncWeb3, chain_id: int) -> PolymarketCTF:
        """Construct a client with every address resolved from the registry."""
        return cls(w3=w3, chain_id=chain_id)

    @property
    def neg_risk_adapter(self) -> Address:
        """The NegRiskAdapter address, resolved lazily (mainnet only for now)."""
        if self._neg_risk_adapter is None:
            self._neg_risk_adapter = get_address("POLYMARKET_NEG_RISK_ADAPTER", self.chain_id)
        return self._neg_risk_adapter

    # ------------------------------------------------------------------
    # Writes — {to, data, value}
    # ------------------------------------------------------------------

    def build_approve_tx(self, amount: int, *, neg_risk: bool = False) -> dict[str, Any]:
        """Approve collateral to whichever contract a split pulls it from — the
        NegRiskAdapter when *neg_risk*, otherwise the CTF."""
        spender = self.neg_risk_adapter if neg_risk else self.conditional_tokens
        return to_tx(self.collateral.address, ERC20.fns.approve(spender, amount).data)

    def build_set_approval_for_all_tx(self, operator: Address, approved: bool = True) -> dict[str, Any]:
        """Build an ERC1155 ``setApprovalForAll`` on the CTF for *operator*.

        Grant the CLOB exchange (to trade outcome tokens) or the NegRiskAdapter
        (to merge / redeem / convert on your behalf) control of your tokens.
        """
        return to_tx(self.conditional_tokens, CONDITIONAL_TOKENS.fns.setApprovalForAll(operator, approved).data)

    def build_split_tx(
        self,
        condition_id: Bytes32,
        amount: int,
        *,
        neg_risk: bool = False,
        partition: Sequence[int] = BINARY_PARTITION,
    ) -> dict[str, Any]:
        """Build a ``splitPosition``: lock *amount* collateral, mint a complete set
        of outcome tokens. Requires a prior :meth:`build_approve_tx`."""
        if neg_risk:
            return self._neg_risk_position_tx(NEG_RISK_ADAPTER.fns.splitPosition, condition_id, amount)
        return self._ctf_position_tx(CONDITIONAL_TOKENS.fns.splitPosition, condition_id, amount, partition)

    def build_merge_tx(
        self,
        condition_id: Bytes32,
        amount: int,
        *,
        neg_risk: bool = False,
        partition: Sequence[int] = BINARY_PARTITION,
    ) -> dict[str, Any]:
        """Build a ``mergePositions``: burn a complete set of outcome tokens, unlock
        *amount* collateral. The inverse of :meth:`build_split_tx`."""
        if neg_risk:
            return self._neg_risk_position_tx(NEG_RISK_ADAPTER.fns.mergePositions, condition_id, amount)
        return self._ctf_position_tx(CONDITIONAL_TOKENS.fns.mergePositions, condition_id, amount, partition)

    def build_redeem_tx(
        self,
        condition_id: Bytes32,
        *,
        neg_risk: bool = False,
        amounts: Sequence[int] | None = None,
        index_sets: Sequence[int] = BINARY_PARTITION,
    ) -> dict[str, Any]:
        """Build a ``redeemPositions``: after resolution, burn winning tokens for
        their collateral payout.

        The raw CTF redeems the full balance of *index_sets*; the NegRiskAdapter
        takes explicit per-outcome *amounts* (required when *neg_risk*).
        """
        if neg_risk:
            if amounts is None:
                raise ValueError("neg-risk redeem requires explicit per-outcome amounts")
            data = NEG_RISK_ADAPTER.fns.redeemPositions(_to_bytes32(condition_id), list(amounts)).data
            return to_tx(self.neg_risk_adapter, data)
        data = CONDITIONAL_TOKENS.fns.redeemPositions(
            self.collateral.address, PARENT_COLLECTION_ID, _to_bytes32(condition_id), list(index_sets)
        ).data
        return to_tx(self.conditional_tokens, data)

    def build_convert_tx(self, market_id: Bytes32, index_set: int, amount: int) -> dict[str, Any]:
        """Build a NegRiskAdapter ``convertPositions``: convert *amount* of the No
        positions selected by *index_set* into the complementary Yes positions
        (plus collateral) across the multi-outcome *market_id*."""
        data = NEG_RISK_ADAPTER.fns.convertPositions(_to_bytes32(market_id), index_set, amount).data
        return to_tx(self.neg_risk_adapter, data)

    def _ctf_position_tx(self, fn: Any, condition_id: Bytes32, amount: int, partition: Sequence[int]) -> dict[str, Any]:
        """Encode a raw-CTF ``(collateral, parent, conditionId, partition, amount)`` split/merge."""
        data = fn(
            self.collateral.address, PARENT_COLLECTION_ID, _to_bytes32(condition_id), list(partition), amount
        ).data
        return to_tx(self.conditional_tokens, data)

    def _neg_risk_position_tx(self, fn: Any, condition_id: Bytes32, amount: int) -> dict[str, Any]:
        """Encode a NegRiskAdapter ``(conditionId, amount)`` split/merge."""
        return to_tx(self.neg_risk_adapter, fn(_to_bytes32(condition_id), amount).data)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_outcome_balance(self, owner: Address, token_id: int) -> int:
        """Return *owner*'s balance of a single outcome token (ERC1155)."""
        return await CONDITIONAL_TOKENS.fns.balanceOf(owner, token_id).call(self.w3, to=self.conditional_tokens)

    async def get_position_id(self, collection_id: Bytes32, collateral: Address | None = None) -> int:
        """Read ``getPositionId(collateral, collectionId)`` from the CTF."""
        token = collateral if collateral is not None else self.collateral.address
        return await CONDITIONAL_TOKENS.fns.getPositionId(token, _to_bytes32(collection_id)).call(
            self.w3, to=self.conditional_tokens
        )

    async def is_resolved(self, condition_id: Bytes32) -> bool:
        """Return whether the condition has been resolved on-chain.

        ``payoutDenominator`` is zero until the oracle reports, so a non-zero
        value means the market is settled and redeemable.
        """
        denom = await CONDITIONAL_TOKENS.fns.payoutDenominator(_to_bytes32(condition_id)).call(
            self.w3, to=self.conditional_tokens
        )
        return denom > 0

    async def get_payouts(self, condition_id: Bytes32, outcome_count: int = 2) -> list[int]:
        """Return the payout numerators per outcome (e.g. ``[1, 0]`` once Yes wins
        a binary market), or all zeros if not yet resolved."""
        cid = _to_bytes32(condition_id)
        return list(
            await asyncio.gather(
                *(
                    CONDITIONAL_TOKENS.fns.payoutNumerators(cid, i).call(self.w3, to=self.conditional_tokens)
                    for i in range(outcome_count)
                )
            )
        )


__all__ = [
    "BINARY_PARTITION",
    "PARENT_COLLECTION_ID",
    "PolymarketCTF",
    "compute_condition_id",
]

"""Prediction-market ABIs (Polymarket / Gnosis Conditional Tokens).

Pre-built :class:`~eth_contract.Contract` objects, bound to an address at the
call site::

    from pydefi.abi.predict import CONDITIONAL_TOKENS
    data = CONDITIONAL_TOKENS.fns.splitPosition(collateral, parent, condition_id, [1, 2], amount).data

* :data:`CONDITIONAL_TOKENS` — Gnosis ConditionalTokens: split / merge / redeem,
  the id getters, payout reads, and the ERC1155 balance / approval surface.
* :data:`NEG_RISK_ADAPTER` — Polymarket NegRiskAdapter for neg-risk markets; it
  wraps USDC internally, so callers approve the adapter and pass a
  ``conditionId`` + amount. Only the simplified overloads are declared (avoids
  overload ambiguity).

Refs:
    https://docs.polymarket.com/trading/ctf/overview
    https://github.com/Polymarket/neg-risk-ctf-adapter
"""

from __future__ import annotations

from eth_contract import Contract

# ---------------------------------------------------------------------------
# Gnosis ConditionalTokens (CTF)
# ---------------------------------------------------------------------------

#: Gnosis ConditionalTokens — the ERC1155 contract that mints Polymarket
#: outcome tokens. For Polymarket binary markets ``parentCollectionId`` is the
#: zero hash and ``partition`` / ``indexSets`` is ``[1, 2]`` (Yes = bit 0,
#: No = bit 1).
CONDITIONAL_TOKENS = Contract.from_abi(
    [
        # --- positions: mint / burn / redeem ---
        "function splitPosition(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] partition, uint256 amount)",
        "function mergePositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] partition, uint256 amount)",
        "function redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)",
        # --- id derivation (pure / view) ---
        "function getConditionId(address oracle, bytes32 questionId, uint256 outcomeSlotCount) view returns (bytes32)",
        "function getCollectionId(bytes32 parentCollectionId, bytes32 conditionId, uint256 indexSet) view returns (bytes32)",
        "function getPositionId(address collateralToken, bytes32 collectionId) view returns (uint256)",
        "function getOutcomeSlotCount(bytes32 conditionId) view returns (uint256)",
        # --- resolution / payouts ---
        "function payoutDenominator(bytes32 conditionId) view returns (uint256)",
        "function payoutNumerators(bytes32 conditionId, uint256 index) view returns (uint256)",
        # --- ERC1155 surface ---
        "function balanceOf(address owner, uint256 id) view returns (uint256)",
        "function balanceOfBatch(address[] owners, uint256[] ids) view returns (uint256[])",
        "function setApprovalForAll(address operator, bool approved)",
        "function isApprovedForAll(address owner, address operator) view returns (bool)",
    ]
)

# ---------------------------------------------------------------------------
# Polymarket NegRiskAdapter
# ---------------------------------------------------------------------------

#: Polymarket NegRiskAdapter — used for negative-risk markets. The adapter pulls
#: raw USDC from the caller and wraps it before talking to the CTF, so the
#: caller approves USDC to the adapter and passes only ``conditionId`` + amount.
#: ``convertPositions`` turns a set of No positions into the complementary Yes
#: positions (plus collateral) across a multi-outcome market.
NEG_RISK_ADAPTER = Contract.from_abi(
    [
        "function splitPosition(bytes32 conditionId, uint256 amount)",
        "function mergePositions(bytes32 conditionId, uint256 amount)",
        "function redeemPositions(bytes32 conditionId, uint256[] amounts)",
        "function convertPositions(bytes32 marketId, uint256 indexSet, uint256 amount)",
        "function getConditionId(bytes32 questionId) view returns (bytes32)",
        "function getPositionId(bytes32 questionId, bool outcome) view returns (uint256)",
    ]
)

__all__ = [
    "CONDITIONAL_TOKENS",
    "NEG_RISK_ADAPTER",
]

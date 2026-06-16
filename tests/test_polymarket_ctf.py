"""Offline tests for pydefi.polymarket.ctf (on-chain CTF actions, no live calls)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from eth_contract.erc20 import ERC20
from hexbytes import HexBytes

from pydefi._utils import decode_address
from pydefi.abi.predict import CONDITIONAL_TOKENS, NEG_RISK_ADAPTER
from pydefi.deployments import get_address, get_token
from pydefi.polymarket import BINARY_PARTITION, PolymarketCTF, compute_condition_id
from pydefi.types import ChainId

# A real (oracle, questionId) → conditionId vector cross-checked against the
# on-chain ConditionalTokens.getConditionId on Polygon mainnet.
_ORACLE = decode_address("0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74", ChainId.POLYGON)
_QUESTION_ID = "0x" + "11" * 32
_EXPECTED_CONDITION_ID = "0x1e5afef09a58c86aa251e7ad5558643165377d2dfabf8e09772f60ceba14c683"

# Sample market identifiers (arbitrary 32-byte values for calldata assertions).
_CONDITION_ID = "0x" + "ab" * 32
_MARKET_ID = "0x" + "cd" * 32

_CTF = get_address("POLYMARKET_CONDITIONAL_TOKENS", ChainId.POLYGON)
_ADAPTER = get_address("POLYMARKET_NEG_RISK_ADAPTER", ChainId.POLYGON)
_USDCE = get_token("USDC.e", ChainId.POLYGON).address

# build_* methods are pure; w3 is only used by the async reads, so None is fine.
_CTF_CLIENT = PolymarketCTF(w3=cast(Any, None), chain_id=ChainId.POLYGON)


def _selector(tx: dict[str, Any]) -> str:
    """4-byte selector of a tx dict's calldata, as lowercase hex."""
    return HexBytes(tx["data"])[:4].hex()


def _to(tx: dict[str, Any]) -> str:
    return cast(str, tx["to"]).lower()


def _approve_spender(tx: dict[str, Any]) -> str:
    """Decoded spender of an ERC20 ``approve`` tx, as lowercase hex."""
    spender, _amount = ERC20.fns.approve.decode_input(HexBytes(tx["data"]))
    return spender.lower()


class TestComputeConditionId:
    """Local conditionId derivation matches the CTF's getConditionId."""

    def test_known_vector(self):
        assert compute_condition_id(_ORACLE, _QUESTION_ID).to_0x_hex() == _EXPECTED_CONDITION_ID

    def test_returns_32_bytes(self):
        assert len(compute_condition_id(_ORACLE, _QUESTION_ID)) == 32

    def test_outcome_count_changes_id(self):
        assert compute_condition_id(_ORACLE, _QUESTION_ID, 2) != compute_condition_id(_ORACLE, _QUESTION_ID, 3)

    def test_accepts_bytes_question_id(self):
        assert compute_condition_id(_ORACLE, _QUESTION_ID) == compute_condition_id(_ORACLE, b"\x11" * 32)


# Every builder targets the right contract with the right 4-byte selector and
# carries no ETH value. Argument encoding is checked in TestBuilderCalldata.
@pytest.mark.parametrize(
    "tx, expected_to, selector",
    [
        pytest.param(_CTF_CLIENT.build_split_tx(_CONDITION_ID, 1), _CTF, "72ce4275", id="split"),
        pytest.param(_CTF_CLIENT.build_merge_tx(_CONDITION_ID, 1), _CTF, "9e7212ad", id="merge"),
        pytest.param(_CTF_CLIENT.build_redeem_tx(_CONDITION_ID), _CTF, "01b7037c", id="redeem"),
        pytest.param(_CTF_CLIENT.build_split_tx(_CONDITION_ID, 1, neg_risk=True), _ADAPTER, "a3d7da1d", id="ng_split"),
        pytest.param(_CTF_CLIENT.build_merge_tx(_CONDITION_ID, 1, neg_risk=True), _ADAPTER, "b10c5c17", id="ng_merge"),
        pytest.param(
            _CTF_CLIENT.build_redeem_tx(_CONDITION_ID, neg_risk=True, amounts=[1, 2]),
            _ADAPTER,
            "dbeccb23",
            id="ng_redeem",
        ),
        pytest.param(_CTF_CLIENT.build_convert_tx(_MARKET_ID, 2, 7), _ADAPTER, "c64748c4", id="ng_convert"),
        pytest.param(_CTF_CLIENT.build_approve_tx(1), _USDCE, "095ea7b3", id="approve"),
        pytest.param(_CTF_CLIENT.build_set_approval_for_all_tx(_ADAPTER, True), _CTF, "a22cb465", id="set_approval"),
    ],
)
def test_builder_targets_contract(tx: dict[str, Any], expected_to: Any, selector: str):
    assert _to(tx) == expected_to.to_0x_hex()
    assert _selector(tx) == selector
    assert tx["value"] == "0"


class TestBuilderCalldata:
    """Builders encode their arguments into the calldata correctly."""

    def test_split_round_trips(self):
        tx = _CTF_CLIENT.build_split_tx(_CONDITION_ID, 100_000_000)
        collateral, parent, cid, partition, amount = CONDITIONAL_TOKENS.fns.splitPosition.decode_input(
            HexBytes(tx["data"])
        )
        assert collateral.lower() == _USDCE.to_0x_hex()
        assert parent == b"\x00" * 32
        assert cid == bytes.fromhex("ab" * 32)
        assert partition == (1, 2)
        assert amount == 100_000_000

    def test_split_partition_override(self):
        tx = _CTF_CLIENT.build_split_tx(_CONDITION_ID, 1, partition=[1, 2, 4])
        *_, partition, _amount = CONDITIONAL_TOKENS.fns.splitPosition.decode_input(HexBytes(tx["data"]))
        assert partition == (1, 2, 4)

    def test_redeem_defaults_to_binary_index_sets(self):
        tx = _CTF_CLIENT.build_redeem_tx(_CONDITION_ID)
        *_, index_sets = CONDITIONAL_TOKENS.fns.redeemPositions.decode_input(HexBytes(tx["data"]))
        assert index_sets == tuple(BINARY_PARTITION)

    def test_neg_risk_split_encodes_condition_and_amount(self):
        tx = _CTF_CLIENT.build_split_tx(_CONDITION_ID, 5_000_000, neg_risk=True)
        cid, amount = NEG_RISK_ADAPTER.fns.splitPosition.decode_input(HexBytes(tx["data"]))
        assert cid == bytes.fromhex("ab" * 32)
        assert amount == 5_000_000

    def test_neg_risk_redeem_takes_per_outcome_amounts(self):
        tx = _CTF_CLIENT.build_redeem_tx(_CONDITION_ID, neg_risk=True, amounts=[10, 20])
        cid, amounts = NEG_RISK_ADAPTER.fns.redeemPositions.decode_input(HexBytes(tx["data"]))
        assert cid == bytes.fromhex("ab" * 32)
        assert amounts == (10, 20)

    def test_neg_risk_convert_encodes_market_index_amount(self):
        tx = _CTF_CLIENT.build_convert_tx(_MARKET_ID, 2, 7)
        market_id, index_set, amount = NEG_RISK_ADAPTER.fns.convertPositions.decode_input(HexBytes(tx["data"]))
        assert market_id == bytes.fromhex("cd" * 32)
        assert index_set == 2
        assert amount == 7


def test_builder_rejects_non_bytes32_id():
    with pytest.raises(ValueError, match="32-byte"):
        _CTF_CLIENT.build_split_tx("0xdeadbeef", 1, neg_risk=True)


class TestPolymarketCtfClient:
    """Address resolution + neg_risk routing (no network)."""

    def test_resolves_registry_addresses(self):
        assert _CTF_CLIENT.conditional_tokens == _CTF
        assert _CTF_CLIENT.neg_risk_adapter == _ADAPTER
        assert _CTF_CLIENT.collateral.address == _USDCE

    def test_split_routes_by_neg_risk(self):
        assert _to(_CTF_CLIENT.build_split_tx(_CONDITION_ID, 1)) == _CTF.to_0x_hex()
        assert _to(_CTF_CLIENT.build_split_tx(_CONDITION_ID, 1, neg_risk=True)) == _ADAPTER.to_0x_hex()

    def test_approve_spender_follows_neg_risk(self):
        assert _approve_spender(_CTF_CLIENT.build_approve_tx(1)) == _CTF.to_0x_hex()
        assert _approve_spender(_CTF_CLIENT.build_approve_tx(1, neg_risk=True)) == _ADAPTER.to_0x_hex()

    def test_neg_risk_redeem_requires_amounts(self):
        with pytest.raises(ValueError, match="amounts"):
            _CTF_CLIENT.build_redeem_tx(_CONDITION_ID, neg_risk=True)

    def test_neg_risk_redeem_with_amounts(self):
        tx = _CTF_CLIENT.build_redeem_tx(_CONDITION_ID, neg_risk=True, amounts=[1, 2])
        assert _to(tx) == _ADAPTER.to_0x_hex()

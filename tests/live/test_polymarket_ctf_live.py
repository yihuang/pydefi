"""Live + fork tests for pydefi.polymarket.ctf (on-chain CTF actions on Polygon).

* ``@pytest.mark.live`` — read-only id-derivation and resolution calls against
  the deployed ConditionalTokens on a public Polygon RPC.
* ``@pytest.mark.fork`` — split / merge / redeem against an Anvil Polygon fork
  (``polygon_fork_w3``). Each test prepares its own condition and funds the
  trader with real USDC.e, so it never leans on a live market's neg-risk quirks.

Run with ``pytest -m live`` or ``pytest -m fork``.
"""

from __future__ import annotations

import pytest
from eth_contract import Contract
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3, Web3

from pydefi._utils import decode_address, to_tx
from pydefi.abi.predict import CONDITIONAL_TOKENS
from pydefi.deployments import get_token
from pydefi.polymarket import BINARY_PARTITION, PolymarketCTF, compute_condition_id
from pydefi.polymarket.ctf import PARENT_COLLECTION_ID
from pydefi.types import ChainId, Hash
from tests.live.anvil_helpers import impersonate, send_ok, set_balance

# Polymarket's UMA CTF adapter — the oracle baked into every market's conditionId.
_ORACLE = decode_address("0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74", ChainId.POLYGON)

# A real RESOLVED non-neg-risk binary market ("US forces enter Iran by April 30?").
# conditionId stays prepared and settled on-chain forever, so reads are stable.
_RESOLVED_CID = "0x6d0e09d0f04572d9b1adad84703458b0297bc5603b69dccbde93147ee4443246"


@pytest.fixture
async def ctf_live(polygon_w3) -> PolymarketCTF:
    """A PolymarketCTF bound to a public Polygon RPC (read-only)."""
    return PolymarketCTF(polygon_w3, ChainId.POLYGON)


@pytest.mark.live
class TestPolymarketCtfReadsLive:
    """Read-only CTF calls against the deployed ConditionalTokens on Polygon."""

    async def test_compute_condition_id_matches_on_chain(self, ctf_live: PolymarketCTF):
        question_id = "0x" + "11" * 32
        on_chain = await CONDITIONAL_TOKENS.fns.getConditionId(_ORACLE, Web3.to_bytes(hexstr=question_id), 2).call(
            ctf_live.w3, to=ctf_live.conditional_tokens
        )
        assert bytes(on_chain) == bytes(compute_condition_id(_ORACLE, question_id))

    async def test_resolved_market_is_resolved(self, ctf_live: PolymarketCTF):
        assert await ctf_live.is_resolved(_RESOLVED_CID) is True

    async def test_resolved_market_payouts_pick_one_winner(self, ctf_live: PolymarketCTF):
        """A settled binary market reports numerators summing to one winner."""
        payouts = await ctf_live.get_payouts(_RESOLVED_CID)
        assert len(payouts) == 2
        assert sum(payouts) == 1
        assert set(payouts) == {0, 1}

    async def test_unknown_condition_is_unresolved(self, ctf_live: PolymarketCTF):
        unknown = "0x" + "ee" * 32
        assert await ctf_live.is_resolved(unknown) is False


# ---------------------------------------------------------------------------
# Fork tests — write round-trips against the real CTF on a Polygon fork
# ---------------------------------------------------------------------------

# A USDC.e whale on Polygon — the Aave amUSDC reserve custodies the underlying.
_USDCE_WHALE = decode_address("0x1a13F4Ca1d028320A707D99520AbFefca3998b7F", ChainId.POLYGON)

# A fresh EOA as the trader. Anvil's default accounts collide with contracts
# deployed at those well-known addresses on Polygon, whose code would trip the
# ERC1155 ``onERC1155Received`` check when the CTF mints — so use a clean one.
_TRADER = decode_address("0xAbcdEF0123456789AbcdEf0123456789aBCDeF01", ChainId.POLYGON)

# Our own oracle + question, so a condition can be prepared (and resolved) without
# a live market's neg-risk wrapping.
_TEST_ORACLE = decode_address("0x000000000000000000000000000000000000bEEF", ChainId.POLYGON)
_TEST_QUESTION = "0x" + "ab" * 32
_QUESTION_BYTES = Web3.to_bytes(hexstr=_TEST_QUESTION)

_GAS = 10**18  # MATIC for gas
_AMOUNT = 10_000_000  # 10 USDC.e (6 decimals)

# CTF functions used only for test setup (not part of the curated public ABI).
_CTF_SETUP = Contract.from_abi(
    [
        "function prepareCondition(address oracle, bytes32 questionId, uint256 outcomeSlotCount)",
        "function reportPayouts(bytes32 questionId, uint256[] payouts)",
    ]
)


async def _usdce_balance(w3: AsyncWeb3, owner) -> int:
    return await ERC20.fns.balanceOf(owner).call(w3, to=get_token("USDC.e", ChainId.POLYGON).address)


async def _outcome_balances(ctf: PolymarketCTF, condition_id: Hash) -> list[int]:
    """The trader's balance of each binary outcome token for *condition_id*."""
    balances = []
    for index_set in BINARY_PARTITION:
        collection = await CONDITIONAL_TOKENS.fns.getCollectionId(
            PARENT_COLLECTION_ID, bytes(condition_id), index_set
        ).call(ctf.w3, to=ctf.conditional_tokens)
        position_id = await ctf.get_position_id(collection)
        balances.append(await ctf.get_outcome_balance(_TRADER, position_id))
    return balances


async def _split_position(w3: AsyncWeb3) -> tuple[PolymarketCTF, Hash]:
    """Prepare our own condition, fund the trader, and split USDC.e into a complete
    set of outcome tokens. Returns the bound client and its ``conditionId``."""
    ctf = PolymarketCTF(w3, ChainId.POLYGON)
    usdce = get_token("USDC.e", ChainId.POLYGON).address
    for who in (_TRADER, _USDCE_WHALE, _TEST_ORACLE):
        await impersonate(w3, who)
        await set_balance(w3, who, _GAS)
    await send_ok(w3, _USDCE_WHALE, to_tx(usdce, ERC20.fns.transfer(_TRADER, _AMOUNT).data), "fund usdce")

    prepare = to_tx(ctf.conditional_tokens, _CTF_SETUP.fns.prepareCondition(_TEST_ORACLE, _QUESTION_BYTES, 2).data)
    await send_ok(w3, _TRADER, prepare, "prepareCondition")

    condition_id = compute_condition_id(_TEST_ORACLE, _TEST_QUESTION)
    await send_ok(w3, _TRADER, ctf.build_approve_tx(_AMOUNT), "approve")
    await send_ok(w3, _TRADER, ctf.build_split_tx(condition_id, _AMOUNT), "split")
    return ctf, condition_id


@pytest.mark.fork
class TestPolymarketCtfFork:
    """split / merge / redeem against the real ConditionalTokens on a Polygon fork."""

    async def test_split_then_merge_round_trips(self, polygon_fork_w3):
        """Split locks USDC.e and mints a complete set; merge reverses it exactly."""
        w3 = polygon_fork_w3
        ctf, condition_id = await _split_position(w3)

        # collateral locked, a full complete set of both outcome tokens minted
        assert await _usdce_balance(w3, _TRADER) == 0
        assert await _outcome_balances(ctf, condition_id) == [_AMOUNT, _AMOUNT]

        await send_ok(w3, _TRADER, ctf.build_merge_tx(condition_id, _AMOUNT), "merge")

        # outcome tokens burned, collateral returned
        assert await _outcome_balances(ctf, condition_id) == [0, 0]
        assert await _usdce_balance(w3, _TRADER) == _AMOUNT

    async def test_split_resolve_redeem_pays_the_winner(self, polygon_fork_w3):
        """After the oracle resolves the condition, redeem pays out the winner."""
        w3 = polygon_fork_w3
        ctf, condition_id = await _split_position(w3)

        # the oracle reports "Yes" (index 0) as the winner
        report = to_tx(ctf.conditional_tokens, _CTF_SETUP.fns.reportPayouts(_QUESTION_BYTES, [1, 0]).data)
        await send_ok(w3, _TEST_ORACLE, report, "reportPayouts")

        assert await ctf.is_resolved(condition_id) is True
        assert await ctf.get_payouts(condition_id) == [1, 0]

        # redeem burns the complete set; the winning side pays the full collateral
        await send_ok(w3, _TRADER, ctf.build_redeem_tx(condition_id), "redeem")
        assert await _outcome_balances(ctf, condition_id) == [0, 0]
        assert await _usdce_balance(w3, _TRADER) == _AMOUNT

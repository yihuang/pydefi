"""Tests for the Multicall3 batching helper.

The indexer and the tick-ladder fetch both index results positionally, so the
contract that matters is not just "did it decode" but "is result *i* the answer
to call *i*".
"""

from __future__ import annotations

import pytest
from eth_contract.erc20 import ERC20
from eth_contract.multicall3 import MULTICALL3, MULTICALL3_ADDRESS, Call3

import pydefi.multicall as multicall_module
from pydefi.deployments import get_address
from pydefi.multicall import batch_call
from pydefi.types import Address

_TOKEN = Address(b"\xab" * 20)
_MULTICALL = Address(b"\xca" * 20)


class _PreparedCall:
    """One prepared ``aggregate3`` invocation, bound to its own payload.

    Chunks are all prepared before any is awaited, so sharing state across
    invocations would make every chunk see the last payload.
    """

    def __init__(self, payload, responder):
        self._payload, self._responder = payload, responder

    async def call(self, _w3, to=None):
        return self._responder(self._payload)


class _FakeAggregate:
    """Stands in for ``MULTICALL3.fns.aggregate3(...)``."""

    def __init__(self, recorder, responder):
        self._recorder, self._responder = recorder, responder

    def __call__(self, payload):
        self._recorder.append(payload)
        return _PreparedCall(payload, self._responder)


@pytest.fixture
def fake_multicall(monkeypatch):
    """Patch aggregate3; yields the list of payloads it was handed."""

    def install(responder):
        payloads: list = []
        agg = _FakeAggregate(payloads, responder)

        class _Fns:
            aggregate3 = staticmethod(agg)

        monkeypatch.setattr(multicall_module.MULTICALL3, "fns", _Fns)
        return payloads

    return install


def _ok(value: bytes):
    return lambda payload: [(True, value) for _ in payload]


async def test_empty_calls_makes_no_request(fake_multicall):
    payloads = fake_multicall(_ok(b""))
    assert await batch_call(None, [], multicall_address=_MULTICALL) == []
    assert payloads == [], "an empty batch must not hit the network"


async def test_accepts_both_address_and_hex_string_targets(fake_multicall):
    """The indexer passes checksummed strings; the tick fetch passes Address."""
    payloads = fake_multicall(_ok((18).to_bytes(32, "big")))
    calls = [(_TOKEN, ERC20.fns.decimals()), ("0x" + "cd" * 20, ERC20.fns.decimals())]
    assert await batch_call(None, calls, multicall_address=_MULTICALL) == [18, 18]
    targets = [Address(call.target) for call in payloads[0]]
    assert targets == [_TOKEN, Address("0x" + "cd" * 20)], "either form reaches the encoder, in order"


class _ChainW3:
    """Just enough AsyncWeb3 to answer ``await w3.eth.chain_id``."""

    def __init__(self, chain_id: int):
        self.eth = self
        self._chain_id = chain_id

    @property
    def chain_id(self):
        async def _get():
            return self._chain_id

        return _get()


async def test_unknown_chain_raises_before_any_call(fake_multicall):
    """The indexer's fallback to individual reads hangs off this raise."""
    payloads = fake_multicall(_ok((18).to_bytes(32, "big")))
    with pytest.raises(KeyError, match="MULTICALL3"):
        await batch_call(_ChainW3(1337), [(_TOKEN, ERC20.fns.decimals())])
    assert payloads == [], "a chain with no deployment must not be batched into empty code"


async def test_known_chain_resolves_to_the_canonical_deployment(fake_multicall):
    """The per-chain map is there for the gate above, not to hold a second copy of the hex."""
    fake_multicall(_ok((18).to_bytes(32, "big")))
    assert await batch_call(_ChainW3(1), [(_TOKEN, ERC20.fns.decimals())]) == [18]
    assert get_address("MULTICALL3", 1) == Address(MULTICALL3_ADDRESS)


async def test_encodes_both_target_forms_identically():
    """Neither form is normalised on the way in, so aggregate3 must encode them the same."""
    as_bytes = MULTICALL3.fns.aggregate3([Call3(_TOKEN, False, b"")]).data
    as_hex = MULTICALL3.fns.aggregate3([Call3("0x" + "ab" * 20, False, b"")]).data
    assert as_bytes == as_hex


async def test_splits_into_chunks(fake_multicall):
    payloads = fake_multicall(_ok((18).to_bytes(32, "big")))
    calls = [(_TOKEN, ERC20.fns.decimals())] * 7
    results = await batch_call(None, calls, multicall_address=_MULTICALL, chunk_size=3)
    assert results == [18] * 7
    assert [len(p) for p in payloads] == [3, 3, 1], "chunked, and nothing dropped"


async def test_revert_raises_by_default(fake_multicall):
    fake_multicall(lambda payload: [(False, b"") for _ in payload])
    with pytest.raises(ValueError, match="reverted"):
        await batch_call(None, [(_TOKEN, ERC20.fns.decimals())], multicall_address=_MULTICALL)


async def test_revert_yields_none_when_allowed(fake_multicall):
    fake_multicall(lambda payload: [(False, b"") for _ in payload])
    got = await batch_call(None, [(_TOKEN, ERC20.fns.decimals())], multicall_address=_MULTICALL, allow_failure=True)
    assert got == [None]


async def test_undecodable_return_yields_none_when_allowed(fake_multicall):
    """A non-standard ERC-20 answering bytes32 from symbol() must not fail the batch."""
    fake_multicall(lambda payload: [(True, b"\x01\x02") for _ in payload])
    got = await batch_call(None, [(_TOKEN, ERC20.fns.symbol())], multicall_address=_MULTICALL, allow_failure=True)
    assert got == [None]


async def test_undecodable_return_raises_by_default(fake_multicall):
    fake_multicall(lambda payload: [(True, b"\x01\x02") for _ in payload])
    with pytest.raises(Exception):  # noqa: B017 - the decoder's own error type
        await batch_call(None, [(_TOKEN, ERC20.fns.symbol())], multicall_address=_MULTICALL)


async def test_short_response_raises_rather_than_misaligning(fake_multicall):
    """Callers index positionally, so a dropped result would shift every value.

    zip() would truncate here without complaint, handing the caller fewer
    results than calls and silently attributing one token's data to another.
    """
    fake_multicall(lambda payload: [(True, (18).to_bytes(32, "big"))] * (len(payload) - 1))
    calls = [(_TOKEN, ERC20.fns.decimals())] * 4
    with pytest.raises(ValueError, match="3 results for 4 calls"):
        await batch_call(None, calls, multicall_address=_MULTICALL)

"""Unit tests for the pydefi.indexer module.

These tests use an in-memory SQLite database and a mocked AsyncWeb3 instance
so no live RPC connection is required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydefi.indexer import PoolIndexer

# Valid 20-byte hex addresses used as test fixtures
_POOL_V2 = "0x" + "ab" * 20
_POOL_V2_2 = "0x" + "ac" * 20
_POOL_V3 = "0x" + "ba" * 20
_POOL_V3_NEG = "0x" + "bc" * 20
_POOL_EMPTY = "0x" + "cc" * 20
_POOL_EMPTY_V3 = "0x" + "dd" * 20
_POOL_LIVE = "0x" + "ee" * 20
_POOL_UNREG = "0x" + "ff" * 20
_POOL_NO_W3 = "0x" + "11" * 20
_POOL_CHECKPOINT = "0x" + "22" * 20
_FACTORY_V2 = "0x" + "fa" * 20
_FACTORY_V3 = "0x" + "fb" * 20
_NEW_PAIR = "0x" + "ca" * 20
_NEW_POOL_V3 = "0x" + "cb" * 20
_TOKEN_A = "0x" + "a0" * 20
_TOKEN_B = "0x" + "b0" * 20
_TOKEN_C = "0x" + "c0" * 20
_TOKEN_D = "0x" + "d0" * 20

_TX_A = "0x" + "aa" * 32
_TX_B = "0x" + "bb" * 32
_TX_C = "0x" + "cc" * 32
_TX_D = "0x" + "dd" * 32
_TX_E = "0x" + "ee" * 32
_TX_F = "0x" + "ff" * 32
_BLOCK_HASH = "0x" + "11" * 32


# ---------------------------------------------------------------------------
# Helpers to build fake log objects
# ---------------------------------------------------------------------------


def _make_v2_log(
    pool_address: str,
    block_number: int,
    reserve0: int,
    reserve1: int,
    tx_hash: str = _TX_A,
    log_index: int = 0,
    block_hash: str = _BLOCK_HASH,
) -> dict:
    data_bytes = reserve0.to_bytes(32, "big") + reserve1.to_bytes(32, "big")
    return {
        "address": pool_address,
        "blockNumber": block_number,
        "blockHash": block_hash,
        "transactionHash": tx_hash,
        "logIndex": log_index,
        "topics": [
            "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1",
        ],
        "data": "0x" + data_bytes.hex(),
    }


def _make_v3_log(
    pool_address: str,
    block_number: int,
    amount0: int,
    amount1: int,
    sqrt_price_x96: int,
    liquidity: int,
    tick: int,
    tx_hash: str = _TX_D,
    log_index: int = 0,
    block_hash: str = _BLOCK_HASH,
) -> dict:
    def to_int256_bytes(v: int) -> bytes:
        return v.to_bytes(32, "big", signed=True)

    data_bytes = (
        to_int256_bytes(amount0)
        + to_int256_bytes(amount1)
        + sqrt_price_x96.to_bytes(32, "big")
        + liquidity.to_bytes(32, "big")
        + to_int256_bytes(tick)
    )
    sender = "0x" + "01" * 32
    recipient = "0x" + "02" * 32
    return {
        "address": pool_address,
        "blockNumber": block_number,
        "blockHash": block_hash,
        "transactionHash": tx_hash,
        "logIndex": log_index,
        "topics": [
            "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
            sender,
            recipient,
        ],
        "data": "0x" + data_bytes.hex(),
    }


def _make_pair_created_log(
    factory_address: str,
    token0: str,
    token1: str,
    pair: str,
    block_number: int = 10,
    tx_hash: str = _TX_A,
    log_index: int = 0,
    block_hash: str = _BLOCK_HASH,
) -> dict:
    """Simulate a V2 PairCreated(address,address,address,uint) log."""

    # topics[1] = token0 (indexed), topics[2] = token1 (indexed)
    # data = pair (address, padded 32 bytes) + allPairsLength (uint256)
    def _addr_topic(addr: str) -> str:
        return "0x" + bytes(12).hex() + addr.lower()[2:]

    pair_bytes = bytes(12) + bytes.fromhex(pair.lower()[2:])
    data_bytes = pair_bytes + (1).to_bytes(32, "big")
    return {
        "address": factory_address,
        "blockNumber": block_number,
        "blockHash": block_hash,
        "transactionHash": tx_hash,
        "logIndex": log_index,
        "topics": [
            "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9",
            _addr_topic(token0),
            _addr_topic(token1),
        ],
        "data": "0x" + data_bytes.hex(),
    }


def _make_pool_created_log(
    factory_address: str,
    token0: str,
    token1: str,
    fee: int,
    pool: str,
    tick_spacing: int = 60,
    block_number: int = 10,
    tx_hash: str = _TX_A,
    log_index: int = 0,
    block_hash: str = _BLOCK_HASH,
) -> dict:
    """Simulate a V3 PoolCreated(address,address,uint24,int24,address) log."""

    def _addr_topic(addr: str) -> str:
        return "0x" + bytes(12).hex() + addr.lower()[2:]

    def _uint24_topic(v: int) -> str:
        return "0x" + v.to_bytes(32, "big").hex()

    tick_bytes = tick_spacing.to_bytes(32, "big")
    pool_bytes = bytes(12) + bytes.fromhex(pool.lower()[2:])
    data_bytes = tick_bytes + pool_bytes
    return {
        "address": factory_address,
        "blockNumber": block_number,
        "blockHash": block_hash,
        "transactionHash": tx_hash,
        "logIndex": log_index,
        "topics": [
            "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118",
            _addr_topic(token0),
            _addr_topic(token1),
            _uint24_topic(fee),
        ],
        "data": "0x" + data_bytes.hex(),
    }


def _make_mock_w3(block_number: int = 100, timestamp: int = 1_700_000_000) -> MagicMock:
    """Return a MagicMock that mimics the async web3 interface used by PoolIndexer.

    ``eth.block_number`` in web3.py async is an awaitable property, so we use
    a ``property()`` that returns a fresh coroutine on each access.
    ``eth.call`` is stubbed to raise so ERC-20 lookups fall back to defaults.
    """
    mock_eth = MagicMock()
    _ts = timestamp

    async def _block_number_coro() -> int:
        return block_number

    async def _get_block_coro(bn: int) -> dict:
        return {"timestamp": _ts, "number": bn}

    async def _call_raises(*args, **kwargs):
        raise Exception("eth_call not mocked")

    # Make block_number an awaitable property
    type(mock_eth).block_number = property(lambda self: _block_number_coro())
    mock_eth.get_block = AsyncMock(side_effect=_get_block_coro)
    mock_eth.get_logs = AsyncMock(return_value=[])
    mock_eth.call = AsyncMock(side_effect=_call_raises)

    mock_w3 = MagicMock()
    mock_w3.eth = mock_eth
    return mock_w3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoolRegistration:
    def test_add_v2_pool(self):
        pool_addr = "0x" + "a1" * 20
        token0 = "0x" + "b1" * 20
        token1 = "0x" + "b2" * 20
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="TKA",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="TKB",
            token1_decimals=6,
            chain_id=1,
            fee_bps=30,
        )
        pool = indexer.get_pool(pool_addr)
        assert pool is not None
        assert pool.pool_address == pool_addr.lower()
        assert pool.protocol == "UniswapV2"
        assert pool.chain_id == 1
        assert pool.token0_symbol == "TKA"
        assert pool.token1_symbol == "TKB"
        assert pool.fee_bps == 30

    def test_add_v3_pool(self):
        pool_addr = "0x" + "c1" * 20
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v3_pool(
            pool_address=pool_addr,
            protocol="UniswapV3",
            token0_address="0x" + "d1" * 20,
            token0_symbol="WETH",
            token0_decimals=18,
            token1_address="0x" + "d2" * 20,
            token1_symbol="USDC",
            token1_decimals=6,
            chain_id=1,
            fee_bps=5,
        )
        pool = indexer.get_pool(pool_addr)
        assert pool is not None
        assert pool.protocol == "UniswapV3"
        assert pool.fee_bps == 5

    def test_add_pool_upsert(self):
        """Re-registering a pool updates its metadata."""
        pool_addr = "0x" + "e1" * 20
        token0 = "0x" + "f1" * 20
        token1 = "0x" + "f2" * 20
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        # Update fee
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
            fee_bps=100,
        )
        pool = indexer.get_pool(pool_addr)
        assert pool is not None
        assert pool.fee_bps == 100

    def test_list_pools(self):
        indexer = PoolIndexer(db_url="sqlite://")
        token0 = "0x" + "a0" * 20
        token1 = "0x" + "b0" * 20
        for i in range(3):
            indexer.add_v2_pool(
                pool_address=f"0x{i:040x}",
                protocol="UniswapV2",
                token0_address=token0,
                token0_symbol="A",
                token0_decimals=18,
                token1_address=token1,
                token1_symbol="B",
                token1_decimals=18,
                chain_id=1,
            )
        assert len(indexer.list_pools()) == 3


class TestV2Indexing:
    @pytest.mark.asyncio
    async def test_backfill_v2(self):
        mock_w3 = _make_mock_w3(block_number=200, timestamp=1_700_001_000)
        pool_addr = _POOL_V2
        token0 = "0x" + "a2" * 20
        token1 = "0x" + "b3" * 20
        logs = [
            _make_v2_log(pool_addr, 100, 1_000, 2_000, tx_hash=_TX_A, log_index=0),
            _make_v2_log(pool_addr, 101, 1_100, 1_900, tx_hash=_TX_B, log_index=0),
        ]
        mock_w3.eth.get_logs = AsyncMock(return_value=logs)

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )

        stored = await indexer.backfill(from_block=100, to_block=200, pool_address=pool_addr)
        assert stored == 2

        state = indexer.get_latest_v2_state(pool_addr)
        assert state is not None
        # The last event at block 101 should be returned (highest block_number).
        assert state["block_number"] == 101
        assert state["reserve0"] == 1_100
        assert state["reserve1"] == 1_900

    @pytest.mark.asyncio
    async def test_backfill_idempotent(self):
        """Running backfill twice does not create duplicate rows."""
        mock_w3 = _make_mock_w3(block_number=200)
        pool_addr = _POOL_V2_2
        token0 = "0x" + "a3" * 20
        token1 = "0x" + "b4" * 20
        log = _make_v2_log(pool_addr, 100, 500, 600, tx_hash=_TX_C, log_index=0)
        mock_w3.eth.get_logs = AsyncMock(return_value=[log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        await indexer.backfill(from_block=100, to_block=100, pool_address=pool_addr)
        await indexer.backfill(from_block=100, to_block=100, pool_address=pool_addr)

        # Only 1 event should exist
        state = indexer.get_latest_v2_state(pool_addr)
        assert state is not None
        assert state["reserve0"] == 500

    @pytest.mark.asyncio
    async def test_latest_v2_state_none_when_empty(self):
        pool_addr = _POOL_EMPTY
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address="0x" + "a4" * 20,
            token0_symbol="A",
            token0_decimals=18,
            token1_address="0x" + "b5" * 20,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        assert indexer.get_latest_v2_state(pool_addr) is None


class TestV3Indexing:
    @pytest.mark.asyncio
    async def test_backfill_v3(self):
        mock_w3 = _make_mock_w3(block_number=300, timestamp=1_700_002_000)
        pool_addr = _POOL_V3
        token0 = "0x" + "a5" * 20
        token1 = "0x" + "b6" * 20
        sqrt_price = 2**96  # 1:1 price
        logs = [
            _make_v3_log(
                pool_addr,
                block_number=150,
                amount0=-1_000,
                amount1=1_000,
                sqrt_price_x96=sqrt_price,
                liquidity=5_000_000,
                tick=0,
                tx_hash=_TX_D,
                log_index=0,
            ),
        ]
        mock_w3.eth.get_logs = AsyncMock(return_value=logs)

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v3_pool(
            pool_address=pool_addr,
            protocol="UniswapV3",
            token0_address=token0,
            token0_symbol="WETH",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="USDC",
            token1_decimals=6,
            chain_id=1,
        )

        stored = await indexer.backfill(from_block=100, to_block=300, pool_address=pool_addr)
        assert stored == 1

        state = indexer.get_latest_v3_state(pool_addr)
        assert state is not None
        assert state["sqrt_price_x96"] == sqrt_price
        assert state["liquidity"] == 5_000_000
        assert state["tick"] == 0

    @pytest.mark.asyncio
    async def test_latest_v3_state_none_when_empty(self):
        pool_addr = _POOL_EMPTY_V3
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v3_pool(
            pool_address=pool_addr,
            protocol="UniswapV3",
            token0_address="0x" + "a6" * 20,
            token0_symbol="A",
            token0_decimals=18,
            token1_address="0x" + "b7" * 20,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        assert indexer.get_latest_v3_state(pool_addr) is None

    @pytest.mark.asyncio
    async def test_negative_amounts_stored_correctly(self):
        """Negative int256 amounts (ticks) are preserved correctly."""
        mock_w3 = _make_mock_w3(block_number=200)
        pool_addr = _POOL_V3_NEG
        token0 = "0x" + "a7" * 20
        token1 = "0x" + "b8" * 20
        logs = [
            _make_v3_log(
                pool_addr,
                block_number=100,
                amount0=-5_000,
                amount1=4_990,
                sqrt_price_x96=2**96,
                liquidity=10_000,
                tick=-887272,  # minimum valid tick
                tx_hash=_TX_E,
                log_index=0,
            ),
        ]
        mock_w3.eth.get_logs = AsyncMock(return_value=logs)

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v3_pool(
            pool_address=pool_addr,
            protocol="UniswapV3",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        await indexer.backfill(from_block=100, to_block=200, pool_address=pool_addr)
        state = indexer.get_latest_v3_state(pool_addr)
        assert state is not None
        assert state["tick"] == -887272


class TestLivePolling:
    @pytest.mark.asyncio
    async def test_run_stops_on_event(self):
        """run() exits when the stop_event is set before the first poll."""
        mock_w3 = _make_mock_w3(block_number=1000)
        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        stop_event = asyncio.Event()
        stop_event.set()
        # Should return immediately without raising.
        await indexer.run(poll_interval=0.01, stop_event=stop_event)

    @pytest.mark.asyncio
    async def test_poll_once_indexes_new_events(self):
        """_poll_once() fetches events for registered pools up to the current head."""
        mock_w3 = _make_mock_w3(block_number=500, timestamp=1_700_003_000)
        pool_addr = _POOL_LIVE
        token0 = "0x" + "a8" * 20
        token1 = "0x" + "b9" * 20
        log = _make_v2_log(pool_addr, 500, 9_000, 8_000, tx_hash=_TX_F, log_index=0)
        mock_w3.eth.get_logs = AsyncMock(return_value=[log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        await indexer._poll_once()

        state = indexer.get_latest_v2_state(pool_addr)
        assert state is not None
        assert state["reserve0"] == 9_000
        assert state["reserve1"] == 8_000

    @pytest.mark.asyncio
    async def test_backfill_unregistered_pool_raises(self):
        mock_w3 = _make_mock_w3()
        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        with pytest.raises(ValueError, match="has not been registered"):
            await indexer.backfill(from_block=0, pool_address=_POOL_UNREG)

    @pytest.mark.asyncio
    async def test_backfill_no_w3_raises(self):
        pool_addr = _POOL_NO_W3
        token0 = "0x" + "a9" * 20
        token1 = "0x" + "b0" * 20
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address=token0,
            token0_symbol="A",
            token0_decimals=18,
            token1_address=token1,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        with pytest.raises(RuntimeError, match="w3 must be set"):
            await indexer.backfill(from_block=0, pool_address=pool_addr)

    @pytest.mark.asyncio
    async def test_poll_once_no_address_filter(self):
        """_poll_once issues one getLogs call without a per-pool address filter."""
        mock_w3 = _make_mock_w3(block_number=100)
        pool_addr_a = "0x" + "a1" * 20
        pool_addr_b = "0x" + "a2" * 20
        log_a = _make_v2_log(pool_addr_a, 100, 1_000, 2_000, tx_hash=_TX_A, log_index=0)
        log_b = _make_v2_log(pool_addr_b, 100, 3_000, 4_000, tx_hash=_TX_B, log_index=1)
        mock_w3.eth.get_logs = AsyncMock(return_value=[log_a, log_b])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        for addr in (pool_addr_a, pool_addr_b):
            indexer.add_v2_pool(
                pool_address=addr,
                protocol="UniswapV2",
                token0_address=_TOKEN_A,
                token0_symbol="A",
                token0_decimals=18,
                token1_address=_TOKEN_B,
                token1_symbol="B",
                token1_decimals=18,
                chain_id=1,
            )

        await indexer._poll_once()

        # Only ONE getLogs call should have been made (not one per pool).
        assert mock_w3.eth.get_logs.call_count == 1
        # Verify no "address" key in the call params.
        call_kwargs = mock_w3.eth.get_logs.call_args[0][0]
        assert "address" not in call_kwargs

        state_a = indexer.get_latest_v2_state(pool_addr_a)
        state_b = indexer.get_latest_v2_state(pool_addr_b)
        assert state_a is not None and state_a["reserve0"] == 1_000
        assert state_b is not None and state_b["reserve0"] == 3_000


class TestIndexerStateCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_updated_after_backfill(self):
        mock_w3 = _make_mock_w3(block_number=150)
        pool_addr = _POOL_CHECKPOINT
        mock_w3.eth.get_logs = AsyncMock(return_value=[])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_v2_pool(
            pool_address=pool_addr,
            protocol="UniswapV2",
            token0_address="0x" + "aa" * 20,
            token0_symbol="A",
            token0_decimals=18,
            token1_address="0x" + "bb" * 20,
            token1_symbol="B",
            token1_decimals=18,
            chain_id=1,
        )
        assert indexer._get_last_indexed_block(pool_addr.lower()) is None
        await indexer.backfill(from_block=100, to_block=150, pool_address=pool_addr)
        assert indexer._get_last_indexed_block(pool_addr.lower()) == 150


class TestFactoryDiscovery:
    @pytest.mark.asyncio
    async def test_add_factory_registered(self):
        """add_factory() persists the factory and it shows in list_factories()."""
        indexer = PoolIndexer(db_url="sqlite://")
        indexer.add_factory(factory_address=_FACTORY_V2, protocol="UniswapV2", chain_id=1)
        factories = indexer.list_factories()
        assert len(factories) == 1
        assert factories[0].factory_address == _FACTORY_V2.lower()
        assert factories[0].protocol == "UniswapV2"

    @pytest.mark.asyncio
    async def test_pair_created_auto_registers_pool(self):
        """PairCreated event from a registered factory auto-registers the new pool."""
        mock_w3 = _make_mock_w3(block_number=50)
        log = _make_pair_created_log(
            factory_address=_FACTORY_V2,
            token0=_TOKEN_A,
            token1=_TOKEN_B,
            pair=_NEW_PAIR,
        )
        mock_w3.eth.get_logs = AsyncMock(return_value=[log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_factory(factory_address=_FACTORY_V2, protocol="UniswapV2", chain_id=1)

        await indexer.backfill(from_block=10, to_block=50)

        pool = indexer.get_pool(_NEW_PAIR)
        assert pool is not None
        assert pool.protocol == "UniswapV2"
        assert pool.token0_address == _TOKEN_A.lower()
        assert pool.token1_address == _TOKEN_B.lower()
        assert pool.fee_bps == 30

    @pytest.mark.asyncio
    async def test_pool_created_auto_registers_v3_pool(self):
        """PoolCreated event from a registered V3 factory auto-registers the new pool."""
        mock_w3 = _make_mock_w3(block_number=50)
        fee = 3000  # 0.3% in V3 hundredths-of-bips
        log = _make_pool_created_log(
            factory_address=_FACTORY_V3,
            token0=_TOKEN_C,
            token1=_TOKEN_D,
            fee=fee,
            pool=_NEW_POOL_V3,
        )
        mock_w3.eth.get_logs = AsyncMock(return_value=[log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_factory(factory_address=_FACTORY_V3, protocol="UniswapV3", chain_id=1)

        await indexer.backfill(from_block=10, to_block=50)

        pool = indexer.get_pool(_NEW_POOL_V3)
        assert pool is not None
        assert pool.protocol == "UniswapV3"
        assert pool.token0_address == _TOKEN_C.lower()
        assert pool.token1_address == _TOKEN_D.lower()
        assert pool.fee_bps == fee // 100  # 30

    @pytest.mark.asyncio
    async def test_pair_created_not_registered_twice(self):
        """Seeing the same PairCreated event twice does not duplicate the pool."""
        mock_w3 = _make_mock_w3(block_number=50)
        log = _make_pair_created_log(
            factory_address=_FACTORY_V2,
            token0=_TOKEN_A,
            token1=_TOKEN_B,
            pair=_NEW_PAIR,
        )
        mock_w3.eth.get_logs = AsyncMock(return_value=[log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_factory(factory_address=_FACTORY_V2, protocol="UniswapV2", chain_id=1)

        await indexer.backfill(from_block=10, to_block=50)
        await indexer.backfill(from_block=10, to_block=50)

        assert len(indexer.list_pools()) == 1

    @pytest.mark.asyncio
    async def test_factory_and_pool_events_in_one_getlogs(self):
        """backfill() processes factory + pool events from a single getLogs call."""
        mock_w3 = _make_mock_w3(block_number=50)
        factory_log = _make_pair_created_log(
            factory_address=_FACTORY_V2,
            token0=_TOKEN_A,
            token1=_TOKEN_B,
            pair=_NEW_PAIR,
            block_number=10,
            tx_hash=_TX_A,
            log_index=0,
        )
        pool_log = _make_v2_log(
            _NEW_PAIR,
            block_number=10,
            reserve0=500,
            reserve1=1_000,
            tx_hash=_TX_B,
            log_index=1,
        )
        # Simulate factory + pool event returned in the same getLogs batch.
        mock_w3.eth.get_logs = AsyncMock(return_value=[factory_log, pool_log])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_factory(factory_address=_FACTORY_V2, protocol="UniswapV2", chain_id=1)

        await indexer.backfill(from_block=10, to_block=50)

        # Pool was auto-registered from the factory event.
        pool = indexer.get_pool(_NEW_PAIR)
        assert pool is not None

        # Pool state was indexed from the Sync event in the same batch.
        state = indexer.get_latest_v2_state(_NEW_PAIR)
        assert state is not None
        assert state["reserve0"] == 500

    @pytest.mark.asyncio
    async def test_backfill_full_scan_address_filter(self):
        """backfill() without pool_address filters get_logs to all tracked addresses."""
        from web3 import Web3

        mock_w3 = _make_mock_w3(block_number=50)
        mock_w3.eth.get_logs = AsyncMock(return_value=[])

        indexer = PoolIndexer(db_url="sqlite://", w3=mock_w3)
        indexer.add_factory(factory_address=_FACTORY_V2, protocol="UniswapV2", chain_id=1)
        indexer.add_factory(factory_address=_FACTORY_V3, protocol="UniswapV3", chain_id=1)

        await indexer.backfill(from_block=10, to_block=50)

        call_kwargs = mock_w3.eth.get_logs.call_args[0][0]
        assert "address" in call_kwargs, "backfill() must include address filter to avoid exceeding node result limits"
        addresses = call_kwargs["address"]
        assert Web3.to_checksum_address(_FACTORY_V2) in addresses
        assert Web3.to_checksum_address(_FACTORY_V3) in addresses

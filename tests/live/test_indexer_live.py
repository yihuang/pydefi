"""Live integration tests for the :class:`~pydefi.indexer.PoolIndexer`.

These tests require network access to a public Ethereum JSON-RPC endpoint (no
API key needed).  They are tagged ``@pytest.mark.live`` and are excluded from
the regular test run.  Run them explicitly with::

    pytest -m live tests/live/test_indexer_live.py

All tests use an in-memory SQLite database so no files are left on disk.

Covered scenarios
-----------------
* **V2 backfill** — back-fill a short recent window for the well-known
  USDC/WETH Uniswap V2 pair; assert that at least one ``Sync`` event was
  indexed and the returned reserve state is numerically plausible.
* **V3 backfill** — back-fill a short recent window for the USDC/WETH
  Uniswap V3 0.05% pool; assert that at least one ``Swap`` event was indexed
  and ``sqrtPriceX96`` is in a plausible range.
* **Backfill idempotency** — running backfill twice over the same range must
  not create duplicate rows.
* **Checkpoint persistence** — after backfill, the stored checkpoint equals
  the requested ``to_block``.
* **Factory auto-discovery (V2)** — register the Uniswap V2 factory and
  back-fill a range over which at least one ``PairCreated`` event is known to
  have occurred; assert that the newly discovered pool was auto-registered.
* **Factory auto-discovery (V3)** — same for the Uniswap V3 factory and
  ``PoolCreated`` events.
"""

from __future__ import annotations

import pytest

from pydefi.indexer import PoolIndexer

# ---------------------------------------------------------------------------
# Well-known Ethereum mainnet addresses
# ---------------------------------------------------------------------------

# Uniswap V2 USDC/WETH pair (very active — always has Sync events)
_V2_USDC_WETH = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"

# Uniswap V3 USDC/WETH 0.05% pool (very active — always has Swap events)
_V3_USDC_WETH = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# Uniswap V2 factory (PairCreated events)
_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

# Uniswap V3 factory (PoolCreated events)
_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

# Well-known token addresses (all Ethereum mainnet)
_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
_WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

# Plausibility bounds for USDC/WETH reserves in the V2 pool:
# token0 = USDC (6 decimals), token1 = WETH (18 decimals)
# At any realistic price of $500–$10 000/ETH the reserves should be
# well within the following raw ranges.
_MIN_V2_USDC_RESERVE = 1_000 * 10**6  # at least $1 000 USDC
_MIN_V2_WETH_RESERVE = 1 * 10**16  # at least 0.01 WETH

# For V3 USDC/WETH the sqrtPriceX96 should represent a price between
# $100 and $100 000 per WETH.
# sqrtPriceX96 = sqrt(raw_WETH / raw_USDC) * 2**96
# raw_WETH per raw_USDC = (1 / price_usd) * 1e18 / 1e6 = 1e12 / price_usd
# At $100/ETH   → sqrtPriceX96 ≈ 7.9e33
# At $100000/ETH → sqrtPriceX96 ≈ 2.5e32
_MIN_V3_SQRT_PRICE = 10**32
_MAX_V3_SQRT_PRICE = 10**34

# How many recent blocks to back-fill — large enough that the active
# USDC/WETH pools almost certainly had transactions, small enough to
# complete quickly over a public RPC.
_BACKFILL_WINDOW = 50

# How many blocks to scan for factory events.  V2 pair creation is rarer, so
# we use a larger window to guarantee we find at least one event.
_FACTORY_WINDOW = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v2_indexer(w3) -> PoolIndexer:
    indexer = PoolIndexer(db_url="sqlite://", w3=w3)
    indexer.add_v2_pool(
        pool_address=_V2_USDC_WETH,
        protocol="UniswapV2",
        token0_address=_USDC,
        token0_symbol="USDC",
        token0_decimals=6,
        token1_address=_WETH,
        token1_symbol="WETH",
        token1_decimals=18,
        chain_id=1,
    )
    return indexer


def _make_v3_indexer(w3) -> PoolIndexer:
    indexer = PoolIndexer(db_url="sqlite://", w3=w3)
    indexer.add_v3_pool(
        pool_address=_V3_USDC_WETH,
        protocol="UniswapV3",
        token0_address=_USDC,
        token0_symbol="USDC",
        token0_decimals=6,
        token1_address=_WETH,
        token1_symbol="WETH",
        token1_decimals=18,
        chain_id=1,
        fee_bps=5,
    )
    return indexer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestIndexerLive:
    """Live integration tests for :class:`PoolIndexer` against Ethereum mainnet."""

    async def test_v2_backfill_indexes_events(self, eth_w3):
        """Backfilling 50 recent blocks for USDC/WETH V2 yields >= 1 Sync event."""
        head = await eth_w3.eth.block_number
        from_block = head - _BACKFILL_WINDOW

        indexer = _make_v2_indexer(eth_w3)
        stored = await indexer.backfill(from_block=from_block, to_block=head, pool_address=_V2_USDC_WETH)

        assert stored >= 1, f"Expected at least 1 Sync event in blocks {from_block}-{head}, got {stored}"

        state = indexer.get_latest_v2_state(_V2_USDC_WETH)
        assert state is not None
        assert state["block_number"] >= from_block
        assert state["timestamp"] > 0
        assert state["reserve0"] >= _MIN_V2_USDC_RESERVE, f"USDC reserve too small: {state['reserve0']}"
        assert state["reserve1"] >= _MIN_V2_WETH_RESERVE, f"WETH reserve too small: {state['reserve1']}"

    async def test_v3_backfill_indexes_events(self, eth_w3):
        """Backfilling 50 recent blocks for USDC/WETH V3 yields >= 1 Swap event."""
        head = await eth_w3.eth.block_number
        from_block = head - _BACKFILL_WINDOW

        indexer = _make_v3_indexer(eth_w3)
        stored = await indexer.backfill(from_block=from_block, to_block=head, pool_address=_V3_USDC_WETH)

        assert stored >= 1, f"Expected at least 1 Swap event in blocks {from_block}-{head}, got {stored}"

        state = indexer.get_latest_v3_state(_V3_USDC_WETH)
        assert state is not None
        assert state["block_number"] >= from_block
        assert state["timestamp"] > 0
        assert _MIN_V3_SQRT_PRICE < state["sqrt_price_x96"] < _MAX_V3_SQRT_PRICE, (
            f"sqrtPriceX96 {state['sqrt_price_x96']} outside plausible range "
            f"[{_MIN_V3_SQRT_PRICE}, {_MAX_V3_SQRT_PRICE}]"
        )
        assert state["liquidity"] > 0
        # tick for USDC/WETH at any realistic price is between -1000000 and 1000000
        assert -1_000_000 < state["tick"] < 1_000_000, f"tick {state['tick']} outside expected range"

    async def test_v2_backfill_idempotent(self, eth_w3):
        """Running backfill twice over the same range must not create duplicate rows."""
        head = await eth_w3.eth.block_number
        from_block = head - _BACKFILL_WINDOW

        indexer = _make_v2_indexer(eth_w3)
        stored_first = await indexer.backfill(from_block=from_block, to_block=head, pool_address=_V2_USDC_WETH)
        stored_second = await indexer.backfill(from_block=from_block, to_block=head, pool_address=_V2_USDC_WETH)

        assert stored_second == 0, f"Second backfill inserted {stored_second} duplicate rows"
        # Total events accessible should equal the first run count.
        state = indexer.get_latest_v2_state(_V2_USDC_WETH)
        assert state is not None
        _ = stored_first  # consumed above

    async def test_checkpoint_persisted_after_backfill(self, eth_w3):
        """After backfill the checkpoint equals the requested to_block."""
        head = await eth_w3.eth.block_number
        from_block = head - _BACKFILL_WINDOW

        indexer = _make_v2_indexer(eth_w3)
        await indexer.backfill(from_block=from_block, to_block=head, pool_address=_V2_USDC_WETH)

        last = indexer._get_last_indexed_block(_V2_USDC_WETH.lower())
        assert last == head, f"Checkpoint {last} != requested to_block {head}"

    async def test_v2_factory_discovers_new_pools(self, eth_w3):
        """Backfilling Uniswap V2 factory over a 500-block window auto-registers >= 1 pool."""
        head = await eth_w3.eth.block_number
        from_block = head - _FACTORY_WINDOW

        indexer = PoolIndexer(db_url="sqlite://", w3=eth_w3)
        indexer.add_factory(factory_address=_V2_FACTORY, protocol="UniswapV2", chain_id=1)

        await indexer.backfill(from_block=from_block, to_block=head)

        pools = indexer.list_pools()
        assert len(pools) >= 1, (
            f"Expected at least 1 pool discovered from V2 factory in blocks {from_block}-{head}, found {len(pools)}"
        )
        for pool in pools:
            assert pool.protocol == "UniswapV2"
            assert pool.chain_id == 1
            assert pool.token0_address
            assert pool.token1_address

    async def test_v3_factory_discovers_new_pools(self, eth_w3):
        """Backfilling Uniswap V3 factory over a 500-block window auto-registers >= 1 pool."""
        head = await eth_w3.eth.block_number
        from_block = head - _FACTORY_WINDOW

        indexer = PoolIndexer(db_url="sqlite://", w3=eth_w3)
        indexer.add_factory(factory_address=_V3_FACTORY, protocol="UniswapV3", chain_id=1)

        await indexer.backfill(from_block=from_block, to_block=head)

        pools = indexer.list_pools()
        assert len(pools) >= 1, (
            f"Expected at least 1 pool discovered from V3 factory in blocks {from_block}-{head}, found {len(pools)}"
        )
        for pool in pools:
            assert pool.protocol == "UniswapV3"
            assert pool.chain_id == 1
            assert pool.token0_address
            assert pool.token1_address

    async def test_combined_factory_and_pool_backfill(self, eth_w3):
        """Factory + known pool registered together; single backfill handles both."""
        head = await eth_w3.eth.block_number
        from_block = head - _BACKFILL_WINDOW

        indexer = PoolIndexer(db_url="sqlite://", w3=eth_w3)
        # Register a factory for auto-discovery.
        indexer.add_factory(factory_address=_V2_FACTORY, protocol="UniswapV2", chain_id=1)
        # Also register a well-known pool directly for state indexing.
        indexer.add_v2_pool(
            pool_address=_V2_USDC_WETH,
            protocol="UniswapV2",
            token0_address=_USDC,
            token0_symbol="USDC",
            token0_decimals=6,
            token1_address=_WETH,
            token1_symbol="WETH",
            token1_decimals=18,
            chain_id=1,
        )

        stored = await indexer.backfill(from_block=from_block, to_block=head)

        # The known pool should have Sync events.
        assert stored >= 1
        state = indexer.get_latest_v2_state(_V2_USDC_WETH)
        assert state is not None
        assert state["reserve0"] > 0

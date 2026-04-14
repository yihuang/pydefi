"""
AMM pool indexer.

The :class:`PoolIndexer` class indexes on-chain AMM pool events into a local
SQLite database (or any SQLAlchemy-compatible backend).  It supports both:

* **Back-filling** – fetching historical ``eth_getLogs`` data for a block range.
* **Live polling** – continuously fetching new blocks and storing new events.

Supported protocols
-------------------
* **Uniswap V2** (and forks) – indexed via the ``Sync(uint112,uint112)`` event.
* **Uniswap V3** (and forks) – indexed via the ``Swap(address,address,int256,int256,uint160,uint128,int24)`` event.

Quick-start example::

    from web3 import AsyncWeb3
    from pydefi.indexer import PoolIndexer

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
    indexer = PoolIndexer(db_url="sqlite:///pools.db", w3=w3)

    await indexer.add_v2_pool(
        pool_address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",  # USDC/ETH
        protocol="UniswapV2",
        token0_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        token0_symbol="USDC",
        token0_decimals=6,
        token1_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        token1_symbol="WETH",
        token1_decimals=18,
        chain_id=1,
    )

    # Back-fill 1 000 blocks of history
    current = await w3.eth.block_number
    await indexer.backfill(
        pool_address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        from_block=current - 1000,
        to_block=current,
    )

    # Start live polling (press Ctrl-C to stop)
    await indexer.run()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from sqlmodel import Session, SQLModel, create_engine, select
from web3 import AsyncWeb3
from web3.types import BlockNumber

from pydefi.indexer.models import IndexerState, Pool, V2SyncEvent, V3SwapEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event topic hashes (keccak256 of the canonical event signature)
# ---------------------------------------------------------------------------

# keccak256("Sync(uint112,uint112)")
_V2_SYNC_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
_V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# How many blocks to request per getLogs call during back-fill
_DEFAULT_BATCH_SIZE = 2_000

# Default polling interval for live mode (seconds)
_DEFAULT_POLL_INTERVAL = 12


def _to_signed(value: int, bits: int) -> int:
    """Reinterpret an unsigned integer as a two's-complement signed integer."""
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


class PoolIndexer:
    """Index AMM pool events into a local SQLAlchemy database.

    Args:
        db_url: SQLAlchemy connection string.  Defaults to an in-memory SQLite
            database (``"sqlite://"``) — useful for testing.  Pass
            ``"sqlite:///pools.db"`` to persist to a file.
        w3: An :class:`~web3.AsyncWeb3` instance connected to the target chain.

    Example::

        indexer = PoolIndexer(db_url="sqlite:///mydb.db", w3=w3)
    """

    def __init__(self, db_url: str = "sqlite://", w3: Optional[AsyncWeb3] = None) -> None:
        self._engine = create_engine(db_url)
        SQLModel.metadata.create_all(self._engine)
        self.w3 = w3
        # Track which protocol each pool uses so we know which logs to parse.
        self._pool_protocol: dict[str, str] = {}
        # Load existing registrations from DB.
        with Session(self._engine) as session:
            for pool in session.exec(select(Pool)).all():
                self._pool_protocol[pool.pool_address.lower()] = pool.protocol
            for state in session.exec(select(IndexerState)).all():
                pass  # just ensure the table is queryable

    # ------------------------------------------------------------------
    # Pool registration
    # ------------------------------------------------------------------

    def add_v2_pool(
        self,
        pool_address: str,
        protocol: str,
        token0_address: str,
        token0_symbol: str,
        token0_decimals: int,
        token1_address: str,
        token1_symbol: str,
        token1_decimals: int,
        chain_id: int,
        fee_bps: int = 30,
    ) -> None:
        """Register a Uniswap V2-compatible pool for indexing.

        If the pool is already registered its metadata is *updated*.

        Args:
            pool_address: On-chain pool contract address.
            protocol: Human-readable protocol name (e.g. ``"UniswapV2"``).
            token0_address: Address of *token0*.
            token0_symbol: Ticker symbol for *token0*.
            token0_decimals: ERC-20 decimals for *token0*.
            token1_address: Address of *token1*.
            token1_symbol: Ticker symbol for *token1*.
            token1_decimals: ERC-20 decimals for *token1*.
            chain_id: EVM chain ID.
            fee_bps: Swap fee in basis points (default ``30``).
        """
        addr = pool_address.lower()
        pool = Pool(
            pool_address=addr,
            protocol=protocol,
            chain_id=chain_id,
            token0_address=token0_address.lower(),
            token0_symbol=token0_symbol,
            token0_decimals=token0_decimals,
            token1_address=token1_address.lower(),
            token1_symbol=token1_symbol,
            token1_decimals=token1_decimals,
            fee_bps=fee_bps,
        )
        with Session(self._engine) as session:
            existing = session.get(Pool, addr)
            if existing:
                for field, value in pool.model_dump(exclude={"pool_address"}).items():
                    setattr(existing, field, value)
                session.add(existing)
            else:
                session.add(pool)
            session.commit()
        self._pool_protocol[addr] = protocol

    def add_v3_pool(
        self,
        pool_address: str,
        protocol: str,
        token0_address: str,
        token0_symbol: str,
        token0_decimals: int,
        token1_address: str,
        token1_symbol: str,
        token1_decimals: int,
        chain_id: int,
        fee_bps: int = 5,
    ) -> None:
        """Register a Uniswap V3-compatible pool for indexing.

        If the pool is already registered its metadata is *updated*.

        Args:
            pool_address: On-chain pool contract address.
            protocol: Human-readable protocol name (e.g. ``"UniswapV3"``).
            token0_address: Address of *token0*.
            token0_symbol: Ticker symbol for *token0*.
            token0_decimals: ERC-20 decimals for *token0*.
            token1_address: Address of *token1*.
            token1_symbol: Ticker symbol for *token1*.
            token1_decimals: ERC-20 decimals for *token1*.
            chain_id: EVM chain ID.
            fee_bps: Swap fee in basis points (default ``5`` = 0.05 %).
        """
        self.add_v2_pool(
            pool_address=pool_address,
            protocol=protocol,
            token0_address=token0_address,
            token0_symbol=token0_symbol,
            token0_decimals=token0_decimals,
            token1_address=token1_address,
            token1_symbol=token1_symbol,
            token1_decimals=token1_decimals,
            chain_id=chain_id,
            fee_bps=fee_bps,
        )

    # ------------------------------------------------------------------
    # Back-fill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        pool_address: str,
        from_block: int,
        to_block: Optional[int] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """Fetch and store historical events for *pool_address*.

        Splits the ``[from_block, to_block]`` range into chunks of
        *batch_size* and issues one ``eth_getLogs`` call per chunk to avoid
        exceeding node limits.

        Args:
            pool_address: Pool address to back-fill (must be registered first
                with :meth:`add_v2_pool` or :meth:`add_v3_pool`).
            from_block: First block to include (inclusive).
            to_block: Last block to include (inclusive).  Defaults to the
                current chain head when omitted.
            batch_size: Number of blocks per ``eth_getLogs`` request.

        Returns:
            Total number of events stored (new events only; duplicates are
            silently skipped).

        Raises:
            ValueError: If *pool_address* has not been registered.
            RuntimeError: If :attr:`w3` has not been set.
        """
        if self.w3 is None:
            raise RuntimeError("PoolIndexer.w3 must be set before calling backfill()")
        addr = pool_address.lower()
        if addr not in self._pool_protocol:
            raise ValueError(f"Pool {pool_address!r} has not been registered. Call add_v2_pool/add_v3_pool first.")

        if to_block is None:
            to_block = await self.w3.eth.block_number

        protocol = self._pool_protocol[addr]
        topic = _V2_SYNC_TOPIC if "v2" in protocol.lower() else _V3_SWAP_TOPIC

        total_stored = 0
        current = from_block
        while current <= to_block:
            end = min(current + batch_size - 1, to_block)
            logs = await self.w3.eth.get_logs(
                {
                    "address": AsyncWeb3.to_checksum_address(addr),
                    "topics": [topic],
                    "fromBlock": current,
                    "toBlock": end,
                }
            )
            logger.debug("Pool %s: fetched %d logs for blocks %d-%d", addr, len(logs), current, end)
            stored = await self._store_logs(addr, logs)
            total_stored += stored
            current = end + 1

        # Update the indexer state checkpoint
        self._set_last_indexed_block(addr, to_block)
        return total_stored

    # ------------------------------------------------------------------
    # Live polling
    # ------------------------------------------------------------------

    async def run(
        self,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Poll for new blocks and index events for all registered pools.

        Runs until *stop_event* is set (or forever if no event is provided).
        Each iteration fetches events from the block after the last checkpoint
        up to the current chain head.

        Args:
            poll_interval: Seconds to sleep between polling iterations.
            stop_event: Optional :class:`asyncio.Event`; when set the loop
                exits cleanly after the current iteration completes.
        """
        if self.w3 is None:
            raise RuntimeError("PoolIndexer.w3 must be set before calling run()")
        logger.info("PoolIndexer: starting live polling (interval=%.1fs)", poll_interval)

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await self._poll_once()
            except Exception as exc:
                logger.warning("PoolIndexer: poll error: %s", exc)
            if stop_event is not None and stop_event.is_set():
                break
            await asyncio.sleep(poll_interval)

    async def _poll_once(self) -> None:
        """Fetch and store events for all registered pools up to the current head."""
        if not self._pool_protocol:
            return
        current_block: int = await self.w3.eth.block_number  # type: ignore[assignment]

        for addr in list(self._pool_protocol):
            last = self._get_last_indexed_block(addr)
            if last is None:
                # No checkpoint yet — only index the single latest block.
                from_block = current_block
            else:
                from_block = last + 1

            if from_block > current_block:
                continue

            protocol = self._pool_protocol[addr]
            topic = _V2_SYNC_TOPIC if "v2" in protocol.lower() else _V3_SWAP_TOPIC

            logs = await self.w3.eth.get_logs(
                {
                    "address": AsyncWeb3.to_checksum_address(addr),
                    "topics": [topic],
                    "fromBlock": from_block,
                    "toBlock": current_block,
                }
            )
            stored = await self._store_logs(addr, logs)
            if stored:
                logger.debug("Pool %s: stored %d new events up to block %d", addr, stored, current_block)
            self._set_last_indexed_block(addr, current_block)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_latest_v2_state(self, pool_address: str) -> Optional[dict[str, Any]]:
        """Return the most-recent reserve state for a V2 pool.

        Args:
            pool_address: Pool address.

        Returns:
            A dict with keys ``block_number``, ``timestamp``, ``reserve0``,
            ``reserve1`` — or ``None`` if no events have been indexed yet.
        """
        addr = pool_address.lower()
        with Session(self._engine) as session:
            stmt = (
                select(V2SyncEvent)
                .where(V2SyncEvent.pool_address == addr)
                .order_by(V2SyncEvent.block_number.desc(), V2SyncEvent.log_index.desc())
            )  # type: ignore[arg-type]
            event = session.exec(stmt).first()
            if event is None:
                return None
            return {
                "block_number": event.block_number,
                "timestamp": event.timestamp,
                "reserve0": event.reserve0,
                "reserve1": event.reserve1,
            }

    def get_latest_v3_state(self, pool_address: str) -> Optional[dict[str, Any]]:
        """Return the most-recent swap state for a V3 pool.

        Args:
            pool_address: Pool address.

        Returns:
            A dict with keys ``block_number``, ``timestamp``, ``sqrt_price_x96``,
            ``liquidity``, ``tick`` — or ``None`` if no events have been indexed yet.
        """
        addr = pool_address.lower()
        with Session(self._engine) as session:
            stmt = (
                select(V3SwapEvent)
                .where(V3SwapEvent.pool_address == addr)
                .order_by(V3SwapEvent.block_number.desc(), V3SwapEvent.log_index.desc())
            )  # type: ignore[arg-type]
            event = session.exec(stmt).first()
            if event is None:
                return None
            return {
                "block_number": event.block_number,
                "timestamp": event.timestamp,
                "sqrt_price_x96": event.sqrt_price_x96,
                "liquidity": event.liquidity,
                "tick": event.tick,
            }

    def get_pool(self, pool_address: str) -> Optional[Pool]:
        """Return the registered :class:`~pydefi.indexer.models.Pool` metadata.

        Args:
            pool_address: Pool address.

        Returns:
            The :class:`~pydefi.indexer.models.Pool` row, or ``None`` if not
            registered.
        """
        addr = pool_address.lower()
        with Session(self._engine) as session:
            return session.get(Pool, addr)

    def list_pools(self) -> list[Pool]:
        """Return all registered pools.

        Returns:
            List of :class:`~pydefi.indexer.models.Pool` rows.
        """
        with Session(self._engine) as session:
            return list(session.exec(select(Pool)).all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_logs(self, pool_address: str, logs: list[Any]) -> int:
        """Parse and persist a batch of raw logs.

        Existing rows (identified by ``tx_hash + log_index``) are silently
        skipped to make the operation idempotent.

        Returns:
            Number of *new* rows inserted.
        """
        if not logs:
            return 0

        protocol = self._pool_protocol.get(pool_address, "")
        is_v2 = "v2" in protocol.lower()

        # Collect timestamps for unique block numbers in one batch.
        block_numbers: set[int] = {int(log["blockNumber"]) for log in logs}
        timestamps: dict[int, int] = {}
        for bn in block_numbers:
            block = await self.w3.eth.get_block(BlockNumber(bn))
            timestamps[bn] = int(block["timestamp"])

        stored = 0
        with Session(self._engine) as session:
            for log in logs:
                bn = int(log["blockNumber"])
                tx_hash = (
                    log["transactionHash"].hex()
                    if not isinstance(log["transactionHash"], str)
                    else log["transactionHash"]
                )
                block_hash = log["blockHash"].hex() if not isinstance(log["blockHash"], str) else log["blockHash"]
                log_index = int(log["logIndex"])
                ts = timestamps[bn]

                if is_v2:
                    # Sync(uint112 reserve0, uint112 reserve1)
                    # data = 32 bytes reserve0 + 32 bytes reserve1
                    data = log["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data_bytes = bytes(data)
                    else:
                        hex_str = data[2:] if isinstance(data, str) and data.startswith("0x") else data
                        data_bytes = bytes.fromhex(hex_str)
                    reserve0 = int.from_bytes(data_bytes[0:32], "big")
                    reserve1 = int.from_bytes(data_bytes[32:64], "big")

                    # Skip if already stored (idempotent)
                    existing = session.exec(
                        select(V2SyncEvent)
                        .where(V2SyncEvent.tx_hash == tx_hash)
                        .where(V2SyncEvent.log_index == log_index)
                    ).first()
                    if existing:
                        continue

                    session.add(
                        V2SyncEvent(
                            pool_address=pool_address,
                            block_number=bn,
                            block_hash=block_hash,
                            tx_hash=tx_hash,
                            log_index=log_index,
                            timestamp=ts,
                            reserve0=reserve0,
                            reserve1=reserve1,
                        )
                    )
                    stored += 1

                else:
                    # Swap(address,address,int256,int256,uint160,uint128,int24)
                    # topics[0] = event sig
                    # topics[1] = sender (indexed)
                    # topics[2] = recipient (indexed)
                    # data = amount0(int256) + amount1(int256) + sqrtPriceX96(uint160) + liquidity(uint128) + tick(int24)
                    data = log["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data_bytes = bytes(data)
                    else:
                        hex_str = data[2:] if isinstance(data, str) and data.startswith("0x") else data
                        data_bytes = bytes.fromhex(hex_str)

                    amount0 = _to_signed(int.from_bytes(data_bytes[0:32], "big"), 256)
                    amount1 = _to_signed(int.from_bytes(data_bytes[32:64], "big"), 256)
                    sqrt_price_x96 = int.from_bytes(data_bytes[64:96], "big")
                    liquidity = int.from_bytes(data_bytes[96:128], "big")
                    tick = _to_signed(int.from_bytes(data_bytes[128:160], "big"), 256)

                    existing = session.exec(
                        select(V3SwapEvent)
                        .where(V3SwapEvent.tx_hash == tx_hash)
                        .where(V3SwapEvent.log_index == log_index)
                    ).first()
                    if existing:
                        continue

                    session.add(
                        V3SwapEvent(
                            pool_address=pool_address,
                            block_number=bn,
                            block_hash=block_hash,
                            tx_hash=tx_hash,
                            log_index=log_index,
                            timestamp=ts,
                            sqrt_price_x96=sqrt_price_x96,
                            liquidity=liquidity,
                            tick=tick,
                            amount0=amount0,
                            amount1=amount1,
                        )
                    )
                    stored += 1

            session.commit()
        return stored

    def _get_last_indexed_block(self, pool_address: str) -> Optional[int]:
        """Return the last indexed block for *pool_address* from the DB."""
        with Session(self._engine) as session:
            state = session.get(IndexerState, pool_address)
            return state.last_indexed_block if state else None

    def _set_last_indexed_block(self, pool_address: str, block_number: int) -> None:
        """Persist the checkpoint for *pool_address*."""
        with Session(self._engine) as session:
            state = session.get(IndexerState, pool_address)
            if state is None:
                session.add(IndexerState(pool_address=pool_address, last_indexed_block=block_number))
            else:
                state.last_indexed_block = block_number
                session.add(state)
            session.commit()

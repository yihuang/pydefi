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

Factory auto-discovery
----------------------
Register a factory address with :meth:`~PoolIndexer.add_factory` and the indexer
will automatically register newly created pools as it sees ``PairCreated`` (V2)
or ``PoolCreated`` (V3) events.

Quick-start example::

    from web3 import AsyncWeb3
    from pydefi.indexer import PoolIndexer

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
    indexer = PoolIndexer(db_url="sqlite:///pools.db", w3=w3)

    # Register a factory – new pools are discovered automatically.
    indexer.add_factory(
        factory_address="0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        protocol="UniswapV2",
        chain_id=1,
    )

    # Or register individual pools directly.
    indexer.add_v2_pool(
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
    await indexer.backfill(from_block=current - 1000, to_block=current)

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

from pydefi.indexer.models import Factory, IndexerState, Pool, V2SyncEvent, V3SwapEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event topic hashes (keccak256 of the canonical event signature)
# ---------------------------------------------------------------------------

# keccak256("Sync(uint112,uint112)")
_V2_SYNC_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
_V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# keccak256("PairCreated(address,address,address,uint256)")
_V2_PAIR_CREATED_TOPIC = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"

# keccak256("PoolCreated(address,address,uint24,int24,address)")
_V3_POOL_CREATED_TOPIC = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"

# How many blocks to request per getLogs call during back-fill
_DEFAULT_BATCH_SIZE = 2_000

# Default polling interval for live mode (seconds)
_DEFAULT_POLL_INTERVAL = 12

# All topics monitored by the indexer in one combined getLogs query
_ALL_TOPICS = [
    _V2_SYNC_TOPIC,
    _V3_SWAP_TOPIC,
    _V2_PAIR_CREATED_TOPIC,
    _V3_POOL_CREATED_TOPIC,
]


def _to_signed(value: int, bits: int) -> int:
    """Reinterpret an unsigned integer as a two's-complement signed integer."""
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _hex_data(data: Any) -> bytes:
    """Convert a log ``data`` field (bytes or hex string) to :class:`bytes`."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    hex_str = data[2:] if isinstance(data, str) and data.startswith("0x") else data
    return bytes.fromhex(hex_str)


def _topic_str(topic: Any) -> str:
    """Normalise a log topic to a lowercase hex string."""
    if isinstance(topic, (bytes, bytearray)):
        return "0x" + bytes(topic).hex()
    return topic.lower() if isinstance(topic, str) else str(topic)


def _addr_from_topic(topic: Any) -> str:
    """Extract a 20-byte EVM address from a 32-byte indexed topic value."""
    raw = _hex_data(topic)
    return "0x" + raw[-20:].hex()


class PoolIndexer:
    """Index AMM pool events into a local SQLAlchemy database.

    Pools and factories can be registered before or during indexing.  Once a
    factory is registered the indexer will automatically discover and register
    new pools as they are created on-chain.

    All ``eth_getLogs`` calls are issued without an address filter so that
    factory events and pool events can be fetched in a single RPC call per
    block range.  Logs are then dispatched to the correct handler based on
    the emitting address and the event topic.

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
        # Track which protocol each pool/factory uses so we can dispatch logs.
        self._pool_protocol: dict[str, str] = {}
        self._factory_protocol: dict[str, str] = {}
        self._factory_chain_id: dict[str, int] = {}
        # Load existing registrations from DB.
        with Session(self._engine) as session:
            for pool in session.exec(select(Pool)).all():
                self._pool_protocol[pool.pool_address.lower()] = pool.protocol
            for factory in session.exec(select(Factory)).all():
                self._factory_protocol[factory.factory_address.lower()] = factory.protocol
                self._factory_chain_id[factory.factory_address.lower()] = factory.chain_id

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
    # Factory registration
    # ------------------------------------------------------------------

    def add_factory(self, factory_address: str, protocol: str, chain_id: int) -> None:
        """Register a factory contract for automatic pool discovery.

        When the indexer encounters a ``PairCreated`` (V2) or ``PoolCreated``
        (V3) event from this factory it will automatically call
        :meth:`add_v2_pool` / :meth:`add_v3_pool` for the new pool.

        If the factory is already registered its metadata is *updated*.

        Args:
            factory_address: On-chain factory contract address.
            protocol: Human-readable protocol name; must contain ``"v2"`` for
                Uniswap V2 forks or ``"v3"`` for V3 forks
                (e.g. ``"UniswapV2"``, ``"UniswapV3"``).
            chain_id: EVM chain identifier.
        """
        addr = factory_address.lower()
        with Session(self._engine) as session:
            existing = session.get(Factory, addr)
            if existing:
                existing.protocol = protocol
                existing.chain_id = chain_id
                session.add(existing)
            else:
                session.add(Factory(factory_address=addr, protocol=protocol, chain_id=chain_id))
            session.commit()
        self._factory_protocol[addr] = protocol
        self._factory_chain_id[addr] = chain_id

    def list_factories(self) -> list[Factory]:
        """Return all registered factories.

        Returns:
            List of :class:`~pydefi.indexer.models.Factory` rows.
        """
        with Session(self._engine) as session:
            return list(session.exec(select(Factory)).all())

    # ------------------------------------------------------------------
    # Back-fill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        from_block: int,
        to_block: Optional[int] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        pool_address: Optional[str] = None,
    ) -> int:
        """Fetch and store historical events for all registered pools and factories.

        Issues a single ``eth_getLogs`` call per block-range batch (without an
        address filter) covering all monitored event topics, then dispatches
        each log to the appropriate handler.  Pass *pool_address* to restrict
        indexing to a single pool.

        Args:
            from_block: First block to include (inclusive).
            to_block: Last block to include (inclusive).  Defaults to the
                current chain head when omitted.
            batch_size: Number of blocks per ``eth_getLogs`` request.
            pool_address: If given, only index events for this pool
                (must be registered first).  Otherwise, all registered pools
                and factories are indexed.

        Returns:
            Total number of events stored (new events only; duplicates are
            silently skipped).

        Raises:
            ValueError: If *pool_address* is given but has not been registered.
            RuntimeError: If :attr:`w3` has not been set.
        """
        if self.w3 is None:
            raise RuntimeError("PoolIndexer.w3 must be set before calling backfill()")

        # Optional single-pool mode — validate early.
        target_addr: Optional[str] = None
        if pool_address is not None:
            target_addr = pool_address.lower()
            if target_addr not in self._pool_protocol:
                raise ValueError(
                    f"Pool {pool_address!r} has not been registered. "
                    "Call add_v2_pool/add_v3_pool first."
                )

        if to_block is None:
            to_block = await self.w3.eth.block_number

        total_stored = 0
        current = from_block
        while current <= to_block:
            end = min(current + batch_size - 1, to_block)
            logs = await self.w3.eth.get_logs(
                {
                    "topics": [_ALL_TOPICS],
                    "fromBlock": current,
                    "toBlock": end,
                }
            )
            logger.debug("backfill blocks %d-%d: fetched %d logs", current, end, len(logs))

            # Filter to the target pool when in single-pool mode.
            if target_addr is not None:
                logs = [lg for lg in logs if _addr_from_topic(lg["address"]).lower() == target_addr
                        or lg["address"].lower() == target_addr]

            stored = await self._process_logs(logs)
            total_stored += stored

            # Update checkpoints for all pools/factories that had events.
            seen_addrs: set[str] = {lg["address"].lower() for lg in logs}
            for addr in seen_addrs:
                self._set_last_indexed_block(addr, end)
            current = end + 1

        # Ensure checkpoint reflects *to_block* for the target pool even when
        # no events were found.
        if target_addr is not None:
            self._set_last_indexed_block(target_addr, to_block)

        return total_stored

    # ------------------------------------------------------------------
    # Live polling
    # ------------------------------------------------------------------

    async def run(
        self,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Poll for new blocks and index events for all registered pools and factories.

        Runs until *stop_event* is set (or forever if no event is provided).
        Each iteration issues a single ``eth_getLogs`` call covering all
        monitored topics (without an address filter) and commits all new events
        in one database transaction for efficiency.

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
        """Fetch and store events for all registered pools and factories up to the current head.

        Issues a single ``eth_getLogs`` call without an address filter so that
        factory events (``PairCreated``, ``PoolCreated``) and pool events
        (``Sync``, ``Swap``) can be retrieved in one round-trip.  All new rows
        are committed in a single database transaction.
        """
        if not self._pool_protocol and not self._factory_protocol:
            return
        current_block: int = await self.w3.eth.block_number  # type: ignore[assignment]

        # Determine the earliest un-indexed block across all tracked addresses.
        all_addrs = list(self._pool_protocol) + list(self._factory_protocol)
        from_block = current_block
        for addr in all_addrs:
            last = self._get_last_indexed_block(addr)
            candidate = last + 1 if last is not None else current_block
            if candidate < from_block:
                from_block = candidate

        if from_block > current_block:
            return

        # One getLogs call, no address filter — dispatch by address + topic.
        logs = await self.w3.eth.get_logs(
            {
                "topics": [_ALL_TOPICS],
                "fromBlock": from_block,
                "toBlock": current_block,
            }
        )
        stored = await self._process_logs(logs)
        if stored:
            logger.debug(
                "PoolIndexer: stored %d new events blocks %d-%d",
                stored,
                from_block,
                current_block,
            )
        # Advance checkpoints for every tracked address.
        for addr in all_addrs:
            self._set_last_indexed_block(addr, current_block)
        # Also advance any factories that were just auto-discovered in this run.
        for addr in list(self._factory_protocol):
            if addr not in all_addrs:
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
    # Internal log processing
    # ------------------------------------------------------------------

    async def _process_logs(self, logs: list[Any]) -> int:
        """Parse, dispatch, and persist a heterogeneous batch of raw logs.

        Handles V2 ``Sync``, V3 ``Swap``, V2 ``PairCreated``, and V3
        ``PoolCreated`` events in one pass.  Collects block timestamps in bulk
        before the DB write and commits all new rows in a **single transaction**
        for efficiency.

        Existing rows (identified by ``tx_hash + log_index``) are silently
        skipped to keep the operation idempotent.

        Returns:
            Number of *new* rows inserted.
        """
        if not logs:
            return 0

        # Batch-fetch timestamps for all unique block numbers first.
        block_numbers: set[int] = {int(log["blockNumber"]) for log in logs}
        timestamps: dict[int, int] = {}
        for bn in block_numbers:
            block = await self.w3.eth.get_block(BlockNumber(bn))
            timestamps[bn] = int(block["timestamp"])

        stored = 0
        with Session(self._engine) as session:
            for log in logs:
                topic0 = _topic_str(log["topics"][0])
                emitter = log["address"].lower()
                bn = int(log["blockNumber"])
                ts = timestamps[bn]
                tx_hash = (
                    log["transactionHash"].hex()
                    if not isinstance(log["transactionHash"], str)
                    else log["transactionHash"]
                )
                block_hash = (
                    log["blockHash"].hex() if not isinstance(log["blockHash"], str) else log["blockHash"]
                )
                log_index = int(log["logIndex"])

                if topic0 == _V2_SYNC_TOPIC and emitter in self._pool_protocol:
                    stored += self._handle_v2_sync(
                        session, emitter, bn, block_hash, tx_hash, log_index, ts, log
                    )

                elif topic0 == _V3_SWAP_TOPIC and emitter in self._pool_protocol:
                    stored += self._handle_v3_swap(
                        session, emitter, bn, block_hash, tx_hash, log_index, ts, log
                    )

                elif topic0 == _V2_PAIR_CREATED_TOPIC and emitter in self._factory_protocol:
                    await self._handle_pair_created(session, emitter, log)

                elif topic0 == _V3_POOL_CREATED_TOPIC and emitter in self._factory_protocol:
                    await self._handle_pool_created(session, emitter, log)

            # Single commit for the entire batch (one transaction per call).
            session.commit()
        return stored

    def _handle_v2_sync(
        self,
        session: Session,
        pool_address: str,
        block_number: int,
        block_hash: str,
        tx_hash: str,
        log_index: int,
        timestamp: int,
        log: Any,
    ) -> int:
        """Persist a V2 ``Sync`` event; returns 1 if inserted, 0 if duplicate."""
        data_bytes = _hex_data(log["data"])
        reserve0 = int.from_bytes(data_bytes[0:32], "big")
        reserve1 = int.from_bytes(data_bytes[32:64], "big")

        existing = session.exec(
            select(V2SyncEvent)
            .where(V2SyncEvent.tx_hash == tx_hash)
            .where(V2SyncEvent.log_index == log_index)
        ).first()
        if existing:
            return 0

        session.add(
            V2SyncEvent(
                pool_address=pool_address,
                block_number=block_number,
                block_hash=block_hash,
                tx_hash=tx_hash,
                log_index=log_index,
                timestamp=timestamp,
                reserve0=reserve0,
                reserve1=reserve1,
            )
        )
        return 1

    def _handle_v3_swap(
        self,
        session: Session,
        pool_address: str,
        block_number: int,
        block_hash: str,
        tx_hash: str,
        log_index: int,
        timestamp: int,
        log: Any,
    ) -> int:
        """Persist a V3 ``Swap`` event; returns 1 if inserted, 0 if duplicate."""
        # Swap data layout (non-indexed fields):
        # amount0(int256) | amount1(int256) | sqrtPriceX96(uint160) | liquidity(uint128) | tick(int24)
        data_bytes = _hex_data(log["data"])
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
            return 0

        session.add(
            V3SwapEvent(
                pool_address=pool_address,
                block_number=block_number,
                block_hash=block_hash,
                tx_hash=tx_hash,
                log_index=log_index,
                timestamp=timestamp,
                sqrt_price_x96=sqrt_price_x96,
                liquidity=liquidity,
                tick=tick,
                amount0=amount0,
                amount1=amount1,
            )
        )
        return 1

    async def _handle_pair_created(
        self, session: Session, factory_address: str, log: Any
    ) -> None:
        """Auto-register a V2 pool discovered from a ``PairCreated`` event.

        Event ABI::

            PairCreated(address indexed token0, address indexed token1, address pair, uint)

        *token0* and *token1* are the indexed topics[1] and topics[2]; the pair
        address is the first 20 bytes of the non-indexed ``data`` field.
        """
        topics = log["topics"]
        token0_addr = _addr_from_topic(topics[1])
        token1_addr = _addr_from_topic(topics[2])
        pair_addr = "0x" + _hex_data(log["data"])[:32][-20:].hex()
        pair_addr_lower = pair_addr.lower()

        if pair_addr_lower in self._pool_protocol:
            return  # Already known

        protocol = self._factory_protocol[factory_address]
        chain_id = self._factory_chain_id[factory_address]

        # Fetch ERC-20 metadata for each token; fall back to empty strings / 18.
        t0_symbol, t0_decimals = await self._fetch_token_meta(token0_addr)
        t1_symbol, t1_decimals = await self._fetch_token_meta(token1_addr)

        pool = Pool(
            pool_address=pair_addr_lower,
            protocol=protocol,
            chain_id=chain_id,
            token0_address=token0_addr.lower(),
            token0_symbol=t0_symbol,
            token0_decimals=t0_decimals,
            token1_address=token1_addr.lower(),
            token1_symbol=t1_symbol,
            token1_decimals=t1_decimals,
            fee_bps=30,
        )
        existing = session.get(Pool, pair_addr_lower)
        if existing is None:
            session.add(pool)
        self._pool_protocol[pair_addr_lower] = protocol
        logger.info(
            "Factory %s: discovered V2 pool %s (%s/%s)",
            factory_address,
            pair_addr,
            t0_symbol,
            t1_symbol,
        )

    async def _handle_pool_created(
        self, session: Session, factory_address: str, log: Any
    ) -> None:
        """Auto-register a V3 pool discovered from a ``PoolCreated`` event.

        Event ABI::

            PoolCreated(address indexed token0, address indexed token1,
                        uint24 indexed fee, int24 tickSpacing, address pool)

        *token0*, *token1*, and *fee* are indexed (topics[1-3]); *pool* is the
        last 20 bytes of the non-indexed ``data`` field (after *tickSpacing*).
        """
        topics = log["topics"]
        token0_addr = _addr_from_topic(topics[1])
        token1_addr = _addr_from_topic(topics[2])
        fee = int.from_bytes(_hex_data(topics[3])[-3:], "big")  # uint24

        data_bytes = _hex_data(log["data"])
        # data = tickSpacing(int24, 32 bytes padded) | pool(address, 32 bytes padded)
        pool_addr = "0x" + data_bytes[32:64][-20:].hex()
        pool_addr_lower = pool_addr.lower()

        if pool_addr_lower in self._pool_protocol:
            return

        protocol = self._factory_protocol[factory_address]
        chain_id = self._factory_chain_id[factory_address]
        fee_bps = fee // 100  # V3 fee is in hundredths of a bip; convert to bps

        t0_symbol, t0_decimals = await self._fetch_token_meta(token0_addr)
        t1_symbol, t1_decimals = await self._fetch_token_meta(token1_addr)

        pool = Pool(
            pool_address=pool_addr_lower,
            protocol=protocol,
            chain_id=chain_id,
            token0_address=token0_addr.lower(),
            token0_symbol=t0_symbol,
            token0_decimals=t0_decimals,
            token1_address=token1_addr.lower(),
            token1_symbol=t1_symbol,
            token1_decimals=t1_decimals,
            fee_bps=fee_bps,
        )
        existing = session.get(Pool, pool_addr_lower)
        if existing is None:
            session.add(pool)
        self._pool_protocol[pool_addr_lower] = protocol
        logger.info(
            "Factory %s: discovered V3 pool %s (%s/%s fee=%d)",
            factory_address,
            pool_addr,
            t0_symbol,
            t1_symbol,
            fee,
        )

    async def _fetch_token_meta(self, token_address: str) -> tuple[str, int]:
        """Return ``(symbol, decimals)`` for *token_address* via ERC-20 calls.

        Falls back to ``("", 18)`` if the calls fail (e.g. non-standard tokens).
        """
        from eth_contract.erc20 import ERC20

        try:
            symbol: str = await ERC20.fns.symbol().call(self.w3, to=token_address)
        except Exception:
            symbol = ""
        try:
            decimals: int = await ERC20.fns.decimals().call(self.w3, to=token_address)
        except Exception:
            decimals = 18
        return symbol, decimals

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _get_last_indexed_block(self, address: str) -> Optional[int]:
        """Return the last indexed block for *address* from the DB."""
        with Session(self._engine) as session:
            state = session.get(IndexerState, address)
            return state.last_indexed_block if state else None

    def _set_last_indexed_block(self, address: str, block_number: int) -> None:
        """Persist the checkpoint for *address*."""
        with Session(self._engine) as session:
            state = session.get(IndexerState, address)
            if state is None:
                session.add(IndexerState(address=address, last_indexed_block=block_number))
            else:
                state.last_indexed_block = block_number
                session.add(state)
            session.commit()

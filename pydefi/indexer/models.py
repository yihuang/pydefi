"""
SQLModel data models for the AMM pool indexer.

Tables
------
- :class:`Factory`       – Factory contract metadata (auto-discovers new pools).
- :class:`Pool`          – Pool metadata (address, protocol, tokens, fee).
- :class:`V2SyncEvent`   – Uniswap V2 ``Sync`` events (reserve0, reserve1 per block).
- :class:`V3SwapEvent`   – Uniswap V3 ``Swap`` events (sqrtPriceX96, liquidity, tick per block).
- :class:`IndexerState`  – Last indexed block per pool/factory (checkpointing for back-fill / live catch-up).
"""

from __future__ import annotations

from typing import Any, Optional

import sqlalchemy as sa
from sqlmodel import Column, Field, SQLModel


class _BigInt(sa.TypeDecorator):
    """Store arbitrarily large integers as TEXT for SQLite/cross-DB compatibility.

    Values that fit in a regular 64-bit signed integer are still accepted; this
    type is required for Uniswap V3 ``uint160`` / ``uint128`` / ``int256``
    fields that can exceed SQLite's native INTEGER range.
    """

    impl = sa.Text
    cache_ok = True

    def process_bind_param(self, value: Optional[int], dialect: Any) -> Optional[str]:
        return str(value) if value is not None else None

    def process_result_value(self, value: Optional[str], dialect: Any) -> Optional[int]:
        return int(value) if value is not None else None


class Factory(SQLModel, table=True):
    """Metadata for a monitored AMM factory contract.

    When the indexer encounters a ``PairCreated`` (V2) or ``PoolCreated`` (V3)
    event emitted by this factory, it automatically registers the new pool for
    indexing.

    Attributes:
        factory_address: On-chain factory contract address (primary key, lowercase).
        protocol: Human-readable protocol name (e.g. ``"UniswapV2"`` or ``"UniswapV3"``).
        chain_id: EVM chain identifier.
    """

    factory_address: str = Field(primary_key=True)
    protocol: str
    chain_id: int


class Pool(SQLModel, table=True):
    """Metadata for a tracked AMM pool.

    Attributes:
        pool_address: On-chain pool contract address (primary key, lowercase).
        protocol: Human-readable protocol name (e.g. ``"UniswapV2"``).
        chain_id: EVM chain identifier.
        token0_address: Address of *token0* (lowercase).
        token0_symbol: Ticker symbol for *token0*.
        token0_decimals: ERC-20 decimals for *token0*.
        token1_address: Address of *token1* (lowercase).
        token1_symbol: Ticker symbol for *token1*.
        token1_decimals: ERC-20 decimals for *token1*.
        fee_bps: Swap fee in basis points (e.g. ``30`` = 0.3 %).
    """

    pool_address: str = Field(primary_key=True)
    protocol: str
    chain_id: int
    token0_address: str
    token0_symbol: str
    token0_decimals: int = 18
    token1_address: str
    token1_symbol: str
    token1_decimals: int = 18
    fee_bps: int = 30


class V2SyncEvent(SQLModel, table=True):
    """A single Uniswap V2 ``Sync(uint112 reserve0, uint112 reserve1)`` log entry.

    Attributes:
        id: Auto-increment primary key.
        pool_address: Address of the pool that emitted the event (indexed, lowercase).
        block_number: Block height at which the event was emitted.
        block_hash: Hex-encoded block hash.
        tx_hash: Hex-encoded transaction hash.
        log_index: Index of this log within the transaction.
        timestamp: Unix timestamp of the block (seconds).
        reserve0: Raw reserve of *token0* after the state change.
        reserve1: Raw reserve of *token1* after the state change.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    pool_address: str = Field(index=True)
    block_number: int = Field(index=True)
    block_hash: str
    tx_hash: str
    log_index: int
    timestamp: int
    # uint112 can exceed 64-bit integer range; store reserves via _BigInt()
    reserve0: int = Field(sa_column=Column(_BigInt(), nullable=False))
    reserve1: int = Field(sa_column=Column(_BigInt(), nullable=False))


class V3SwapEvent(SQLModel, table=True):
    """A single Uniswap V3 ``Swap`` log entry.

    Attributes:
        id: Auto-increment primary key.
        pool_address: Address of the pool that emitted the event (indexed, lowercase).
        block_number: Block height at which the event was emitted.
        block_hash: Hex-encoded block hash.
        tx_hash: Hex-encoded transaction hash.
        log_index: Index of this log within the transaction.
        timestamp: Unix timestamp of the block (seconds).
        sqrt_price_x96: Square-root price (Q64.96 fixed-point) after the swap.
        liquidity: Active liquidity after the swap.
        tick: Current tick after the swap.
        amount0: Net change in *token0* reserve (signed).
        amount1: Net change in *token1* reserve (signed).
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    pool_address: str = Field(index=True)
    block_number: int = Field(index=True)
    block_hash: str
    tx_hash: str
    log_index: int
    timestamp: int
    # uint160 / uint128 / int256 exceed SQLite INTEGER range — store as TEXT
    sqrt_price_x96: int = Field(sa_column=Column(_BigInt(), nullable=False))
    liquidity: int = Field(sa_column=Column(_BigInt(), nullable=False))
    tick: int = Field(sa_column=Column(_BigInt(), nullable=False))
    amount0: int = Field(sa_column=Column(_BigInt(), nullable=False))
    amount1: int = Field(sa_column=Column(_BigInt(), nullable=False))


class IndexerState(SQLModel, table=True):
    """Per-pool or per-factory indexer checkpoint.

    Records the most-recently indexed block number for each tracked address so
    that the indexer can resume after a restart without re-processing
    already-seen events.

    Attributes:
        address: Pool or factory address (primary key, lowercase).
        last_indexed_block: The highest block number whose logs have been stored.
    """

    address: str = Field(primary_key=True)
    last_indexed_block: int

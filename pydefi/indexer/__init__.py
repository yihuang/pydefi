"""
AMM pool indexer.

Provides local indexing of AMM pool events into a SQLite (or any SQLAlchemy-
compatible) database, supporting both historical back-fill and live polling.

Example::

    from web3 import AsyncWeb3
    from pydefi.indexer import PoolIndexer

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider("https://eth.drpc.org"))
    indexer = PoolIndexer(db_url="sqlite:///pools.db", w3=w3)

    indexer.add_v2_pool(
        pool_address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        protocol="UniswapV2",
        token0_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        token0_symbol="USDC",
        token0_decimals=6,
        token1_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        token1_symbol="WETH",
        token1_decimals=18,
        chain_id=1,
    )
    await indexer.backfill(from_block=17_000_000, to_block=17_100_000, pool_address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc")
    await indexer.run()
"""

from pydefi.indexer.indexer import PoolIndexer
from pydefi.indexer.models import Factory, IndexerState, Pool, V2SyncEvent, V3SwapEvent

__all__ = [
    "PoolIndexer",
    "Factory",
    "Pool",
    "V2SyncEvent",
    "V3SwapEvent",
    "IndexerState",
]

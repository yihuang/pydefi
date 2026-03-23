"""Live integration tests for pydefi.pool_data.

These tests hit the public GeckoTerminal REST API (no key required) and
verify that the parsed responses are structurally and numerically valid.
They are tagged ``@pytest.mark.live`` and excluded from the regular test run.

Run with::

    pytest -m live tests/live/test_pool_data_live.py
"""

import pytest

from pydefi.exceptions import PoolDataError
from pydefi.pathfinder.router import Router
from pydefi.pool_data.geckoterminal import GeckoTerminal
from pydefi.types import ChainId, TokenAmount

from .conftest import USDC, WETH

# ---------------------------------------------------------------------------
# Well-known Ethereum mainnet pool addresses for spot checks
# ---------------------------------------------------------------------------

# Uniswap V2 WETH/USDC pair
WETH_USDC_V2 = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"
# Uniswap V3 WETH/USDC 0.05% pool
WETH_USDC_V3 = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# Sanity price bounds: 1 WETH should fetch between $500 and $10 000 in USDC
MIN_USDC = 500 * 10**6  # 500 USDC (6 decimals)
MAX_USDC = 10_000 * 10**6  # 10 000 USDC


# ---------------------------------------------------------------------------
# GeckoTerminal live tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="GeckoTerminal live tests skipped — rate-limiting / availability")
@pytest.mark.live
class TestGeckoTerminalLive:
    """Live tests against the public GeckoTerminal API v2 (Ethereum mainnet)."""

    def _client(self) -> GeckoTerminal:
        return GeckoTerminal(chain_id=ChainId.ETHEREUM)

    async def test_get_pool_v2(self):
        """get_pool should return valid PoolData for a known V2 pair."""
        client = self._client()
        pool = await client.get_pool(WETH_USDC_V2)

        assert pool.pool_address.lower() == WETH_USDC_V2.lower()
        assert pool.chain_id == ChainId.ETHEREUM
        # Both token symbols should be non-empty
        assert pool.token0.symbol
        assert pool.token1.symbol
        # TVL should be non-zero for the WETH/USDC V2 pair
        assert pool.extra.get("reserve_in_usd", 0) > 0

    async def test_get_pool_v3(self):
        """get_pool should return valid PoolData for a known V3 pool."""
        client = self._client()
        pool = await client.get_pool(WETH_USDC_V3)

        assert pool.pool_address.lower() == WETH_USDC_V3.lower()
        assert pool.chain_id == ChainId.ETHEREUM
        assert pool.token0.symbol
        assert pool.token1.symbol

    async def test_get_pool_not_found_raises(self):
        """Unknown pool address should raise PoolDataError (404)."""
        client = self._client()
        with pytest.raises(PoolDataError):
            await client.get_pool("0x" + "00" * 20)

    async def test_get_top_pools_returns_nonempty_list(self):
        """get_top_pools should return at least one pool."""
        client = self._client()
        pools = await client.get_top_pools(limit=5)

        assert len(pools) > 0
        for pool in pools:
            assert pool.pool_address
            assert pool.token0.symbol
            assert pool.token1.symbol
            assert pool.chain_id == ChainId.ETHEREUM

    async def test_get_top_pools_limit_respected(self):
        """The number of pools returned should not exceed limit."""
        client = self._client()
        pools = await client.get_top_pools(limit=3)

        assert len(pools) <= 3

    async def test_get_pools_for_token_weth(self):
        """get_pools_for_token should return WETH-containing pools."""
        client = self._client()
        pools = await client.get_pools_for_token(WETH.address, limit=5)

        assert len(pools) > 0
        for pool in pools:
            addrs = {pool.token0.address.lower(), pool.token1.address.lower()}
            assert WETH.address.lower() in addrs, f"Pool {pool.pool_address} does not contain WETH"

    async def test_get_pools_for_tokens_weth_usdc(self):
        """get_pools_for_tokens should return pools for WETH and USDC."""
        client = self._client()
        pools = await client.get_pools_for_tokens([WETH.address, USDC.address], limit=10)

        assert len(pools) > 0
        # Each pool must contain at least one of the queried tokens
        queried = {WETH.address.lower(), USDC.address.lower()}
        for pool in pools:
            pool_tokens = {
                pool.token0.address.lower(),
                pool.token1.address.lower(),
            }
            assert pool_tokens & queried, f"Pool {pool.pool_address} contains neither WETH nor USDC"

    async def test_get_pools_for_tokens_deduplicated(self):
        """Querying overlapping tokens should not produce duplicate pools."""
        client = self._client()
        # WETH appears in many pools; query it twice (same token twice)
        pools = await client.get_pools_for_tokens([WETH.address, WETH.address], limit=10)

        addresses = [p.pool_address for p in pools]
        assert len(addresses) == len(set(addresses)), "Duplicate pool addresses found"

    async def test_build_graph_and_route_weth_to_usdc(self):
        """Pools fetched from GeckoTerminal should produce a navigable graph.

        Strategy:
        - Fetch the top pools for WETH and USDC (the two tokens we want to swap).
        - Build a PoolGraph from the result.
        - Run the Router and verify a plausible WETH → USDC route.
        """
        client = self._client()
        pools = await client.get_pools_for_tokens([WETH.address, USDC.address], limit=20)
        assert pools, "No pools returned from GeckoTerminal"

        graph = client.build_graph(pools)
        router = Router(graph, max_hops=3)

        amount_in = TokenAmount.from_human(WETH, "1")
        route = router.find_best_route(amount_in, USDC)

        assert route.token_in == WETH
        assert route.token_out == USDC
        assert route.amount_out.amount > 0
        # Sanity: 1 WETH should be worth between $500 and $10 000 in USDC
        assert MIN_USDC < route.amount_out.amount < MAX_USDC, (
            f"WETH→USDC via GeckoTerminal out of range: {route.amount_out.human_amount} USDC"
        )

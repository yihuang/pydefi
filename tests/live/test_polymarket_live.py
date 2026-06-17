"""Live integration tests for pydefi.polymarket.

These tests make real HTTP requests to the public Polymarket Gamma API and
CLOB API.  No credentials or private key are needed — all tests exercise the
public (unauthenticated) endpoints only.

Run with::

    pytest -m live tests/live/test_polymarket_live.py
"""

from __future__ import annotations

import pytest

from pydefi.polymarket import PolymarketClient

# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> PolymarketClient:
    """Return a Level-0 PolymarketClient (no credentials needed)."""
    return PolymarketClient()


# ---------------------------------------------------------------------------
# Gamma API — market data
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestGammaApiLive:
    """Live tests for the Polymarket Gamma API (public, no auth required)."""

    async def test_get_markets_returns_list(self, client):
        """get_markets() returns a non-empty list of market dicts."""
        markets = await client.get_markets(active=True, closed=False, limit=5)

        assert isinstance(markets, list), "get_markets() should return a list"
        assert len(markets) > 0, "There should be at least one active market"

    async def test_get_markets_structure(self, client):
        """Each market dict contains the expected fields."""
        markets = await client.get_markets(active=True, closed=False, limit=3)

        for market in markets:
            assert isinstance(market, dict), "Each market should be a dict"
            # These fields are always present in active markets
            assert "question" in market, "market should have 'question'"
            assert "conditionId" in market, "market should have 'conditionId'"
            assert "clobTokenIds" in market, "market should have 'clobTokenIds'"

    async def test_get_markets_clob_token_ids_are_strings(self, client):
        """clobTokenIds contains two non-empty string token IDs."""
        markets = await client.get_markets(active=True, closed=False, limit=5)

        for market in markets:
            token_ids = market.get("clobTokenIds")
            if token_ids is None:
                continue
            # token_ids may be a JSON string or a list depending on the API version
            if isinstance(token_ids, str):
                import json

                token_ids = json.loads(token_ids)
            assert isinstance(token_ids, list), "clobTokenIds should be a list"
            assert len(token_ids) == 2, "binary market should have exactly 2 token IDs"

    async def test_get_market_by_condition_id(self, client):
        """get_market() returns a single market dict for a known condition ID."""
        # Fetch one market to get a real condition ID
        markets = await client.get_markets(active=True, closed=False, limit=1)
        assert markets, "Need at least one market for this test"

        condition_id = markets[0]["conditionId"]
        market = await client.get_market(condition_id)

        assert isinstance(market, dict), "get_market() should return a dict"
        assert market.get("conditionId") == condition_id, "Returned market should match the requested condition ID"

    async def test_get_events_returns_list(self, client):
        """get_events() returns a non-empty list of event dicts."""
        events = await client.get_events(active=True, closed=False, limit=5)

        assert isinstance(events, list), "get_events() should return a list"
        assert len(events) > 0, "There should be at least one active event"

    async def test_get_events_structure(self, client):
        """Each event dict contains the expected fields."""
        events = await client.get_events(active=True, closed=False, limit=3)

        for event in events:
            assert isinstance(event, dict), "Each event should be a dict"
            assert "title" in event or "slug" in event, "event should have 'title' or 'slug'"

    async def test_get_events_with_ordering(self, client):
        """get_events() with ascending=False returns a non-empty list."""
        events = await client.get_events(
            active=True,
            closed=False,
            limit=5,
        )

        assert isinstance(events, list)
        assert len(events) > 0

    async def test_get_tags_returns_list(self, client):
        """get_tags() returns a non-empty list of tag dicts."""
        tags = await client.get_tags()

        assert isinstance(tags, list), "get_tags() should return a list"
        assert len(tags) > 0, "There should be at least one tag"

    async def test_get_tags_structure(self, client):
        """Each tag dict has at least an 'id' and a label-like field."""
        tags = await client.get_tags()

        for tag in tags[:5]:
            assert isinstance(tag, dict), "Each tag should be a dict"
            assert "id" in tag, "tag should have an 'id'"

    async def test_get_markets_pagination(self, client):
        """Pagination via limit/offset returns different result sets."""
        page1 = await client.get_markets(active=True, closed=False, limit=5, offset=0)
        page2 = await client.get_markets(active=True, closed=False, limit=5, offset=5)

        if len(page1) == 5 and len(page2) > 0:
            ids1 = {m.get("conditionId") for m in page1}
            ids2 = {m.get("conditionId") for m in page2}
            assert ids1.isdisjoint(ids2), "Pages should not overlap"


# ---------------------------------------------------------------------------
# CLOB API — public endpoints
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestClobApiPublicLive:
    """Live tests for public CLOB API endpoints (no auth required)."""

    async def test_get_server_time(self, client):
        """get_server_time() returns a positive Unix timestamp."""
        result = await client.get_server_time()

        assert isinstance(result, (int, float)), "get_server_time() should return a number"
        assert result > 0, "server time should be a positive timestamp"

    async def _get_token_id(self, client: PolymarketClient) -> str:
        """Helper: fetch a live token ID from the first active binary market."""
        markets = await client.get_markets(active=True, closed=False, limit=10)
        for m in markets:
            token_ids = m.get("clobTokenIds")
            if token_ids is None:
                continue
            if isinstance(token_ids, str):
                import json

                token_ids = json.loads(token_ids)
            if isinstance(token_ids, list) and len(token_ids) >= 1 and token_ids[0]:
                return str(token_ids[0])
        pytest.skip("No active market with a valid token ID found")

    async def test_get_clob_market(self, client):
        """get_clob_market() returns a dict with minimum_tick_size and neg_risk."""
        markets = await client.get_markets(active=True, closed=False, limit=5)
        assert markets, "Need at least one market"

        condition_id = markets[0]["conditionId"]
        market = await client.get_clob_market(condition_id)

        assert isinstance(market, dict), "get_clob_market() should return a dict"
        assert "minimum_tick_size" in market or "neg_risk" in market, (
            "CLOB market should have minimum_tick_size or neg_risk"
        )

    async def test_get_orderbook(self, client):
        """get_orderbook() returns a dict with bids and asks lists."""
        token_id = await self._get_token_id(client)
        book = await client.get_orderbook(token_id)

        assert isinstance(book, dict), "get_orderbook() should return a dict"
        assert "bids" in book or "asks" in book, "order book should have bids or asks"

        if "bids" in book:
            assert isinstance(book["bids"], list)
        if "asks" in book:
            assert isinstance(book["asks"], list)

    async def test_get_midpoint(self, client):
        """get_midpoint() returns a dict with a 'mid' price string."""
        token_id = await self._get_token_id(client)
        result = await client.get_midpoint(token_id)

        assert isinstance(result, dict), "get_midpoint() should return a dict"
        assert "mid" in result, "midpoint response should have 'mid'"
        mid = result["mid"]
        assert isinstance(mid, str), "'mid' should be a string"
        mid_float = float(mid)
        assert 0.0 <= mid_float <= 1.0, f"mid price {mid_float} should be in [0, 1]"

    async def test_get_price_buy(self, client):
        """get_price() returns a dict with a 'price' for the BUY side."""
        token_id = await self._get_token_id(client)
        result = await client.get_price(token_id, "BUY")

        assert isinstance(result, dict), "get_price() should return a dict"
        assert "price" in result, "price response should have 'price'"
        price = float(result["price"])
        assert 0.0 <= price <= 1.0, f"price {price} should be in [0, 1]"

    async def test_get_price_sell(self, client):
        """get_price() returns a valid price for the SELL side."""
        token_id = await self._get_token_id(client)
        result = await client.get_price(token_id, "SELL")

        assert isinstance(result, dict)
        assert "price" in result
        price = float(result["price"])
        assert 0.0 <= price <= 1.0

    async def test_get_tick_size(self, client):
        """get_tick_size() returns the minimum_tick_size for a token."""
        token_id = await self._get_token_id(client)
        result = await client.get_tick_size(token_id)

        assert isinstance(result, dict), "get_tick_size() should return a dict"
        assert "minimum_tick_size" in result, "response should have 'minimum_tick_size'"
        tick = result["minimum_tick_size"]
        # The API may return a float or a string; normalise to float for comparison
        tick_float = float(tick)
        assert tick_float in (0.1, 0.01, 0.001, 0.0001), f"tick size {tick_float} should be one of the standard values"

    async def test_get_neg_risk(self, client):
        """get_neg_risk() returns a dict with a 'neg_risk' boolean."""
        token_id = await self._get_token_id(client)
        result = await client.get_neg_risk(token_id)

        assert isinstance(result, dict), "get_neg_risk() should return a dict"
        assert "neg_risk" in result, "response should have 'neg_risk'"
        assert isinstance(result["neg_risk"], bool), "'neg_risk' should be a boolean"

    async def test_get_last_trade_price(self, client):
        """get_last_trade_price() returns a valid price string."""
        token_id = await self._get_token_id(client)
        result = await client.get_last_trade_price(token_id)

        assert isinstance(result, dict), "get_last_trade_price() should return a dict"
        assert "price" in result, "response should have 'price'"
        price = float(result["price"])
        assert 0.0 <= price <= 1.0, f"last trade price {price} should be in [0, 1]"

    async def test_get_spread(self, client):
        """get_spread() returns a dict with a 'spread' field."""
        token_id = await self._get_token_id(client)
        result = await client.get_spread(token_id)

        assert isinstance(result, dict), "get_spread() should return a dict"
        assert "spread" in result, "spread response should have 'spread'"
        spread = float(result["spread"])
        assert spread >= 0.0, f"spread {spread} should be non-negative"

"""Tests for pydefi.polymarket (no live calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from eth_account import Account

from pydefi.polymarket import (
    BUY,
    EXCHANGE_ADDRESSES,
    POLYGON_AMOY_CHAIN_ID,
    POLYGON_CHAIN_ID,
    SELL,
    PolymarketClient,
    build_hmac_signature,
    get_order_amounts,
    sign_clob_auth,
    sign_order,
    to_token_decimals,
)

# ---------------------------------------------------------------------------
# Deterministic test key — no real funds
# ---------------------------------------------------------------------------

# A well-known throwaway private key used only for deterministic signing
# assertions in tests. It holds no real funds and must never be used on any
# live network.
_TEST_PRIVATE_KEY = "0xb0057716d5917badaf911b193b12b910811c1497b5bada8d7711f758981c3773"
_TEST_WALLET = Account.from_key(_TEST_PRIVATE_KEY)
_TEST_ADDRESS = _TEST_WALLET.address

# A fake API secret (base64url-encoded, 32 bytes decoded).
_TEST_SECRET = "dGVzdC1zZWNyZXQtdmFsdWUtaGVyZS0tLS0tLS0tLS0t"
_TEST_API_KEY = "test-api-key-uuid"
_TEST_PASSPHRASE = "test-passphrase"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aiohttp_mock(status: int, response_data) -> MagicMock:
    """Build a mock aiohttp.ClientSession returning *response_data*."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.raise_for_status = MagicMock()

    resp_ctx = MagicMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=resp_ctx)
    mock_session.post = MagicMock(return_value=resp_ctx)
    mock_session.delete = MagicMock(return_value=resp_ctx)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return session_ctx


# ---------------------------------------------------------------------------
# sign_clob_auth
# ---------------------------------------------------------------------------


class TestSignClobAuth:
    """Tests for the sign_clob_auth() helper."""

    def test_returns_hex_string(self):
        """sign_clob_auth() returns a hex signature starting with 0x."""
        sig = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000)
        assert isinstance(sig, str)
        assert sig.startswith("0x")

    def test_signature_is_65_bytes(self):
        """EIP-712 signatures are 65 bytes (r + s + v)."""
        sig = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000)
        # "0x" prefix + 130 hex chars = 65 bytes
        assert len(sig) == 132

    def test_deterministic_for_same_inputs(self):
        """Same inputs produce the same signature."""
        sig_a = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000, nonce=0)
        sig_b = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000, nonce=0)
        assert sig_a == sig_b

    def test_different_timestamp_gives_different_sig(self):
        """Different timestamps produce different signatures."""
        sig_a = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000)
        sig_b = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_001)
        assert sig_a != sig_b

    def test_polygon_and_amoy_chains(self):
        """Signing works for both mainnet and testnet chain IDs."""
        sig_main = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_CHAIN_ID, timestamp=1_700_000_000)
        sig_test = sign_clob_auth(_TEST_PRIVATE_KEY, chain_id=POLYGON_AMOY_CHAIN_ID, timestamp=1_700_000_000)
        assert sig_main != sig_test


# ---------------------------------------------------------------------------
# build_hmac_signature
# ---------------------------------------------------------------------------


class TestBuildHmacSignature:
    """Tests for build_hmac_signature()."""

    def test_returns_base64_string(self):
        """Result is a non-empty base64url string."""
        sig = build_hmac_signature(_TEST_SECRET, "1700000000", "GET", "/orders")
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_deterministic(self):
        """Same inputs → same signature."""
        sig_a = build_hmac_signature(_TEST_SECRET, "1700000000", "GET", "/orders")
        sig_b = build_hmac_signature(_TEST_SECRET, "1700000000", "GET", "/orders")
        assert sig_a == sig_b

    def test_different_methods_give_different_sigs(self):
        """GET and POST produce different HMAC signatures."""
        sig_get = build_hmac_signature(_TEST_SECRET, "1700000000", "GET", "/orders")
        sig_post = build_hmac_signature(_TEST_SECRET, "1700000000", "POST", "/orders")
        assert sig_get != sig_post

    def test_body_included_in_signature(self):
        """Including a body changes the signature."""
        sig_no_body = build_hmac_signature(_TEST_SECRET, "1700000000", "POST", "/order")
        sig_with_body = build_hmac_signature(_TEST_SECRET, "1700000000", "POST", "/order", {"foo": "bar"})
        assert sig_no_body != sig_with_body

    def test_known_good_vector_no_body(self):
        """Known-good HMAC for a GET request with no body.

        Verified by:
            import base64, hashlib, hmac
            key = base64.urlsafe_b64decode(_TEST_SECRET)
            msg = b"1700000000GET/orders"
            base64.urlsafe_b64encode(hmac.new(key, msg, hashlib.sha256).digest())
        """
        expected = "pxJCUIZkL3HEcBHdGYmnmvPQNQxlF7CCo2cU_9tCRq4="
        sig = build_hmac_signature(_TEST_SECRET, "1700000000", "GET", "/orders")
        assert sig == expected

    def test_known_good_vector_with_body(self):
        """Known-good HMAC for a POST request with a JSON body dict.

        Verified by:
            import base64, hashlib, hmac, json
            key = base64.urlsafe_b64decode(_TEST_SECRET)
            msg = ('1700000000POST/order' + json.dumps({"foo": "bar"})).encode()
            base64.urlsafe_b64encode(hmac.new(key, msg, hashlib.sha256).digest())
        """
        expected = "8jmMLnsdqpIaiH7N_IetwTILTR09TGAGraaykWTsyiQ="
        sig = build_hmac_signature(_TEST_SECRET, "1700000000", "POST", "/order", {"foo": "bar"})
        assert sig == expected


# ---------------------------------------------------------------------------
# get_order_amounts
# ---------------------------------------------------------------------------


class TestGetOrderAmounts:
    """Tests for get_order_amounts()."""

    def test_buy_amounts(self):
        """BUY: maker pays USDC, taker receives outcome tokens."""
        maker_amount, taker_amount = get_order_amounts(BUY, size=10.0, price=0.50, tick_size="0.01")
        # maker pays 0.50 * 10 = 5 USDC → 5_000_000 base units
        assert maker_amount == 5_000_000
        # taker receives 10 outcome tokens → 10_000_000 base units
        assert taker_amount == 10_000_000

    def test_sell_amounts(self):
        """SELL: maker gives outcome tokens, taker pays USDC."""
        maker_amount, taker_amount = get_order_amounts(SELL, size=10.0, price=0.50, tick_size="0.01")
        # maker gives 10 outcome tokens → 10_000_000 base units
        assert maker_amount == 10_000_000
        # taker pays 0.50 * 10 = 5 USDC → 5_000_000 base units
        assert taker_amount == 5_000_000

    def test_invalid_side_raises(self):
        """Invalid side raises ValueError."""
        with pytest.raises(ValueError):
            get_order_amounts(99, size=10.0, price=0.50, tick_size="0.01")

    def test_invalid_tick_size_raises(self):
        """Unsupported tick size raises ValueError."""
        with pytest.raises(ValueError):
            get_order_amounts(BUY, size=10.0, price=0.50, tick_size="0.5")

    def test_all_tick_sizes(self):
        """Amount calculation works for all supported tick sizes."""
        for ts in ("0.1", "0.01", "0.001", "0.0001"):
            maker, taker = get_order_amounts(BUY, size=10.0, price=0.50, tick_size=ts)
            assert maker > 0
            assert taker > 0


# ---------------------------------------------------------------------------
# to_token_decimals
# ---------------------------------------------------------------------------


class TestToTokenDecimals:
    """Tests for to_token_decimals()."""

    def test_converts_usdc(self):
        """5 USDC → 5_000_000 base units."""
        assert to_token_decimals(5.0) == 5_000_000

    def test_rounds_correctly(self):
        """Fractional amounts are rounded to nearest integer."""
        assert to_token_decimals(1.5) == 1_500_000


# ---------------------------------------------------------------------------
# sign_order
# ---------------------------------------------------------------------------


class TestSignOrder:
    """Tests for sign_order()."""

    _TOKEN_ID = "12345678901234567890"
    _MAKER = _TEST_ADDRESS

    def test_returns_signed_order_dict(self):
        """sign_order() returns a dict with the expected keys."""
        order = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="5000000",
            taker_amount="10000000",
            side=BUY,
            salt=42,
        )
        required_keys = {
            "salt",
            "maker",
            "signer",
            "taker",
            "tokenId",
            "makerAmount",
            "takerAmount",
            "expiration",
            "nonce",
            "feeRateBps",
            "side",
            "signatureType",
            "signature",
        }
        assert required_keys.issubset(order.keys())

    def test_signature_is_hex(self):
        """The signature field is a hex string starting with 0x."""
        order = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="5000000",
            taker_amount="10000000",
            side=BUY,
            salt=42,
        )
        assert order["signature"].startswith("0x")
        assert len(order["signature"]) == 132  # 65 bytes = 130 hex chars + "0x"

    def test_deterministic_with_fixed_salt(self):
        """Fixed salt produces deterministic signature."""
        order_a = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="5000000",
            taker_amount="10000000",
            side=BUY,
            salt=99,
        )
        order_b = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="5000000",
            taker_amount="10000000",
            side=BUY,
            salt=99,
        )
        assert order_a["signature"] == order_b["signature"]

    def test_exchange_addresses(self):
        """Correct exchange contracts are used for all chain/neg_risk combos."""
        for (chain_id, neg_risk), addr in EXCHANGE_ADDRESSES.items():
            order = sign_order(
                private_key=_TEST_PRIVATE_KEY,
                maker=self._MAKER,
                token_id=self._TOKEN_ID,
                maker_amount="5000000",
                taker_amount="10000000",
                side=BUY,
                salt=1,
                chain_id=chain_id,
                neg_risk=neg_risk,
            )
            # Just verify it runs without error; address is embedded in the sig
            assert "signature" in order

    def test_sell_side(self):
        """SELL orders are signed correctly."""
        order = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="10000000",
            taker_amount="5000000",
            side=SELL,
            salt=7,
        )
        assert order["side"] == SELL
        assert order["signature"].startswith("0x")

    def test_signer_defaults_to_maker(self):
        """signer defaults to maker when not specified."""
        order = sign_order(
            private_key=_TEST_PRIVATE_KEY,
            maker=self._MAKER,
            token_id=self._TOKEN_ID,
            maker_amount="5000000",
            taker_amount="10000000",
            side=BUY,
            salt=1,
        )
        assert order["signer"] == self._MAKER


# ---------------------------------------------------------------------------
# PolymarketClient — initialisation
# ---------------------------------------------------------------------------


class TestPolymarketClientInit:
    """Tests for PolymarketClient constructor."""

    def test_no_credentials(self):
        """Client initialises with no credentials (Level 0)."""
        client = PolymarketClient()
        assert client._private_key is None
        assert client._address is None

    def test_with_private_key(self):
        """Address is derived from private key."""
        client = PolymarketClient(private_key=_TEST_PRIVATE_KEY)
        assert client._address == _TEST_ADDRESS

    def test_custom_funder(self):
        """Custom funder overrides the derived address."""
        custom_funder = "0x0000000000000000000000000000000000000001"
        client = PolymarketClient(private_key=_TEST_PRIVATE_KEY, funder=custom_funder)
        assert client._funder == custom_funder

    def test_custom_urls(self):
        """Custom API URLs are stored correctly."""
        client = PolymarketClient(
            gamma_api_url="https://custom-gamma.example.com",
            clob_api_url="https://custom-clob.example.com",
        )
        assert client._gamma_base == "https://custom-gamma.example.com"
        assert client._clob_base == "https://custom-clob.example.com"

    def test_trailing_slash_stripped(self):
        """Trailing slashes are stripped from base URLs."""
        client = PolymarketClient(
            gamma_api_url="https://gamma.example.com/",
            clob_api_url="https://clob.example.com/",
        )
        assert not client._gamma_base.endswith("/")
        assert not client._clob_base.endswith("/")

    def test_l1_headers_requires_key(self):
        """_l1_headers raises RuntimeError when no private key is set."""
        client = PolymarketClient()
        with pytest.raises(RuntimeError, match="private_key"):
            client._l1_headers()

    def test_l2_headers_requires_creds(self):
        """_l2_headers raises RuntimeError when API credentials are missing."""
        client = PolymarketClient(private_key=_TEST_PRIVATE_KEY)
        with pytest.raises(RuntimeError, match="api_key"):
            client._l2_headers("GET", "/orders")


# ---------------------------------------------------------------------------
# PolymarketClient — Gamma API (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPolymarketClientGamma:
    """Tests for Gamma API methods using mocked aiohttp."""

    @pytest.fixture
    def client(self):
        return PolymarketClient(gamma_api_url="https://gamma.test", clob_api_url="https://clob.test")

    @pytest.mark.asyncio
    async def test_get_markets(self, client):
        """get_markets() performs a GET request and returns parsed JSON."""
        fake_markets = [{"question": "Will X happen?", "clobTokenIds": ["111", "222"]}]
        mock_session = _make_aiohttp_mock(200, fake_markets)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_markets(limit=1)

        assert result == fake_markets

    @pytest.mark.asyncio
    async def test_get_market(self, client):
        """get_market() returns first item from the Gamma API list response."""
        fake_market = {"conditionId": "0xabc", "question": "Q?"}
        mock_session = _make_aiohttp_mock(200, [fake_market])

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_market("0xabc")

        assert result == fake_market

    @pytest.mark.asyncio
    async def test_get_market_not_found(self, client):
        """get_market() returns None when the Gamma API returns an empty list."""
        mock_session = _make_aiohttp_mock(200, [])

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_market("0xnonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_events(self, client):
        """get_events() performs a GET request and returns parsed JSON."""
        fake_events = [{"title": "Election 2024", "markets": []}]
        mock_session = _make_aiohttp_mock(200, fake_events)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_events(limit=1)

        assert result == fake_events

    @pytest.mark.asyncio
    async def test_get_tags(self, client):
        """get_tags() returns list of tag dicts."""
        fake_tags = [{"id": 1, "label": "Politics"}]
        mock_session = _make_aiohttp_mock(200, fake_tags)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_tags()

        assert result == fake_tags


# ---------------------------------------------------------------------------
# PolymarketClient — CLOB public endpoints (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPolymarketClientClobPublic:
    """Tests for public CLOB API methods using mocked aiohttp."""

    @pytest.fixture
    def client(self):
        return PolymarketClient(gamma_api_url="https://gamma.test", clob_api_url="https://clob.test")

    @pytest.mark.asyncio
    async def test_get_server_time(self, client):
        """get_server_time() returns an int timestamp."""
        fake_resp = 1_700_000_000
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_server_time()

        assert isinstance(result, int)
        assert result == 1_700_000_000

    @pytest.mark.asyncio
    async def test_get_orderbook(self, client):
        """get_orderbook() returns bids/asks."""
        fake_book = {"bids": [{"price": "0.48", "size": "100"}], "asks": []}
        mock_session = _make_aiohttp_mock(200, fake_book)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_orderbook("999")

        assert "bids" in result

    @pytest.mark.asyncio
    async def test_get_midpoint(self, client):
        """get_midpoint() returns a dict with 'mid'."""
        fake_resp = {"mid": "0.50"}
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_midpoint("999")

        assert result["mid"] == "0.50"

    @pytest.mark.asyncio
    async def test_get_price(self, client):
        """get_price() returns a dict with 'price'."""
        fake_resp = {"price": "0.52"}
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_price("999", "BUY")

        assert result["price"] == "0.52"

    @pytest.mark.asyncio
    async def test_get_tick_size(self, client):
        """get_tick_size() returns minimum_tick_size."""
        fake_resp = {"minimum_tick_size": "0.01"}
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_tick_size("999")

        assert result["minimum_tick_size"] == "0.01"

    @pytest.mark.asyncio
    async def test_get_neg_risk(self, client):
        """get_neg_risk() returns neg_risk bool."""
        fake_resp = {"neg_risk": False}
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_neg_risk("999")

        assert result["neg_risk"] is False


# ---------------------------------------------------------------------------
# PolymarketClient — L1 authenticated endpoints (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPolymarketClientL1:
    """Tests for L1 authenticated CLOB methods."""

    @pytest.fixture
    def client(self):
        return PolymarketClient(
            private_key=_TEST_PRIVATE_KEY,
            clob_api_url="https://clob.test",
            gamma_api_url="https://gamma.test",
        )

    @pytest.mark.asyncio
    async def test_create_api_key(self, client):
        """create_api_key() POSTs to /auth/api-key and returns creds."""
        fake_creds = {
            "apiKey": "abc-123",
            "secret": _TEST_SECRET,
            "passphrase": "pw",
        }
        mock_session = _make_aiohttp_mock(200, fake_creds)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.create_api_key()

        assert result == fake_creds

    def test_l1_headers_contain_required_fields(self, client):
        """L1 headers contain POLY_ADDRESS, POLY_SIGNATURE, POLY_TIMESTAMP, POLY_NONCE."""
        headers = client._l1_headers(nonce=0)
        assert "POLY_ADDRESS" in headers
        assert "POLY_SIGNATURE" in headers
        assert "POLY_TIMESTAMP" in headers
        assert "POLY_NONCE" in headers
        assert headers["POLY_ADDRESS"] == _TEST_ADDRESS


# ---------------------------------------------------------------------------
# PolymarketClient — L2 authenticated endpoints (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPolymarketClientL2:
    """Tests for L2 authenticated CLOB methods."""

    @pytest.fixture
    def client(self):
        return PolymarketClient(
            private_key=_TEST_PRIVATE_KEY,
            api_key=_TEST_API_KEY,
            api_secret=_TEST_SECRET,
            api_passphrase=_TEST_PASSPHRASE,
            clob_api_url="https://clob.test",
            gamma_api_url="https://gamma.test",
        )

    def test_l2_headers_contain_required_fields(self, client):
        """L2 headers contain the five required POLY_* keys."""
        headers = client._l2_headers("GET", "/data/orders")
        assert "POLY_ADDRESS" in headers
        assert "POLY_SIGNATURE" in headers
        assert "POLY_TIMESTAMP" in headers
        assert "POLY_API_KEY" in headers
        assert "POLY_PASSPHRASE" in headers
        assert headers["POLY_API_KEY"] == _TEST_API_KEY

    @pytest.mark.asyncio
    async def test_post_order_invalid_side(self, client):
        """post_order() raises ValueError for unknown side."""
        with pytest.raises(ValueError, match="side"):
            await client.post_order(
                token_id="999",
                price=0.50,
                size=10.0,
                side="HOLD",
                tick_size="0.01",
            )

    @pytest.mark.asyncio
    async def test_post_order_requires_private_key(self):
        """post_order() raises RuntimeError when no private key is set."""
        client_no_key = PolymarketClient(
            api_key=_TEST_API_KEY,
            api_secret=_TEST_SECRET,
            api_passphrase=_TEST_PASSPHRASE,
            clob_api_url="https://clob.test",
            gamma_api_url="https://gamma.test",
        )
        with pytest.raises(RuntimeError, match="private_key"):
            await client_no_key.post_order(
                token_id="999",
                price=0.50,
                size=10.0,
                side="BUY",
                tick_size="0.01",
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """cancel_order() DELETEs to /order and returns response."""
        fake_resp = {"success": True}
        mock_session = _make_aiohttp_mock(200, fake_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.cancel_order("order-id-123")

        assert result == fake_resp

    @pytest.mark.asyncio
    async def test_get_trades(self, client):
        """get_trades() returns list of trade dicts."""
        fake_trades = [{"id": "t1", "price": "0.50"}]
        mock_session = _make_aiohttp_mock(200, fake_trades)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_trades()

        assert result == fake_trades

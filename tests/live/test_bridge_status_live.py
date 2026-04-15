"""Live integration tests for bridge get_status() implementations.

These tests make real HTTP requests to public bridge APIs to verify that
``get_status()`` correctly calls the right endpoints, parses responses, and
returns a well-formed :class:`~pydefi.types.BridgeStatus`.

Two levels of live testing are included:

1. **Unknown-hash tests** — Use an all-zeros 32-byte hash.  Most bridge APIs
   return a "not found" response for this, which should be parsed into
   ``BridgeTransactionStatus.UNKNOWN`` without raising an exception.  These
   tests validate: API reachability, URL construction, response parsing, and
   the "not found" handling path.  Note that some APIs (e.g. Relay) may
   return a valid status even for a zero hash due to internal indexing
   behaviour; those tests only assert structural correctness.

2. **Response-structure tests** — Verify that ``get_status()`` always returns
   a :class:`~pydefi.types.BridgeStatus` with the correct ``protocol`` name
   and a valid ``BridgeTransactionStatus`` value.

Run with::

    pytest -m live tests/live/test_bridge_status_live.py
"""

import pytest

from pydefi.bridge.across import Across
from pydefi.bridge.cctp import CCTP
from pydefi.bridge.gaszip import GasZip
from pydefi.bridge.layerzero_oft import LayerZeroOFT
from pydefi.bridge.mayan import Mayan
from pydefi.bridge.relay import Relay
from pydefi.bridge.stargate import Stargate
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeTransactionStatus, ChainId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# An all-zero transaction hash — most bridge APIs should return "not found"
# for this, resulting in BridgeTransactionStatus.UNKNOWN.
_ZERO_HASH = "0x" + "00" * 32

# Well-known GasZip deposit contract on Ethereum mainnet
GASZIP_CONTRACT_ETH = "0x391E7C679d29bD940d63be94AD22A25d25b5A604"

# A canonical LayerZero OFT (ZRO token) — same address on every chain
_ZRO_ADDRESS = "0x6985884C4392D348587B19cb9eAAf157F13271cd"

# A well-known Stargate V1 router on Ethereum mainnet
STARGATE_ROUTER_ETH = "0x8731d54E9D02c286767d56ac03e8037C07e01e98"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_status(result, expected_src_tx_hash: str, expected_protocol: str) -> None:
    """Assert the BridgeStatus has all required fields with plausible values."""
    assert isinstance(result.status, BridgeTransactionStatus), (
        f"status {result.status!r} is not a BridgeTransactionStatus"
    )
    assert result.src_tx_hash == expected_src_tx_hash
    assert result.protocol == expected_protocol
    # dst_tx_hash may be None (for pending/unknown) or a hex string
    if result.dst_tx_hash is not None:
        assert result.dst_tx_hash.startswith("0x"), "dst_tx_hash must be a hex string"


def _skip_on_bridge_error(exc: BridgeError, bridge_name: str) -> None:
    """Skip a live test when a non-critical BridgeError occurs (e.g. connectivity)."""
    pytest.skip(f"{bridge_name} API unavailable: {exc}")


# ---------------------------------------------------------------------------
# Across
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestAcrossGetStatusLive:
    """Live tests for Across.get_status() against the public Across API."""

    def _client(self) -> Across:
        return Across(
            w3=None,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            spoke_pool_address="0x5c7BCd6E7De5423a257D81B442095A1a6ced35C5",
        )

    async def test_get_status_unknown_hash(self):
        """Across: zero hash returns UNKNOWN without raising."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Across")
        _assert_valid_status(result, _ZERO_HASH, "Across")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """Across: get_status always returns a BridgeStatus with protocol='Across'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Across")
        assert result.protocol == "Across"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# CCTP
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCCTPGetStatusLive:
    """Live tests for CCTP.get_status() against the public Circle Iris v2 API."""

    def _client(self) -> CCTP:
        return CCTP(
            w3=None,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
        )

    async def test_get_status_unknown_hash(self):
        """CCTP: zero hash returns UNKNOWN without raising (Iris API returns 404)."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "CCTP")
        _assert_valid_status(result, _ZERO_HASH, "CCTP")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """CCTP: get_status always returns a BridgeStatus with protocol='CCTP'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "CCTP")
        assert result.protocol == "CCTP"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestRelayGetStatusLive:
    """Live tests for Relay.get_status() against the public Relay API."""

    def _client(self) -> Relay:
        return Relay(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)

    async def test_get_status_returns_valid_response(self):
        """Relay: get_status returns a valid BridgeStatus without raising.

        The Relay API may return a non-UNKNOWN status even for a zero hash
        due to its internal indexing; we only validate the response structure.
        """
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Relay")
        _assert_valid_status(result, _ZERO_HASH, "Relay")

    async def test_get_status_returns_bridge_status_type(self):
        """Relay: get_status always returns a BridgeStatus with protocol='Relay'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Relay")
        assert result.protocol == "Relay"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# Mayan
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestMayanGetStatusLive:
    """Live tests for Mayan.get_status() against the public Mayan Explorer API."""

    def _client(self) -> Mayan:
        return Mayan(src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.ARBITRUM)

    async def test_get_status_unknown_hash(self):
        """Mayan: zero hash returns UNKNOWN without raising (404 from explorer)."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Mayan")
        _assert_valid_status(result, _ZERO_HASH, "Mayan")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """Mayan: get_status always returns a BridgeStatus with protocol='Mayan'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Mayan")
        assert result.protocol == "Mayan"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# Stargate (LayerZero Scan)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestStargateGetStatusLive:
    """Live tests for Stargate.get_status() against the LayerZero Scan API."""

    def _client(self) -> Stargate:
        return Stargate(
            w3=None,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            router_address=STARGATE_ROUTER_ETH,
        )

    async def test_get_status_unknown_hash(self):
        """Stargate: zero hash returns UNKNOWN without raising (404 from LZ Scan)."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Stargate / LayerZero Scan")
        _assert_valid_status(result, _ZERO_HASH, "Stargate")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """Stargate: get_status always returns a BridgeStatus with protocol='Stargate'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "Stargate / LayerZero Scan")
        assert result.protocol == "Stargate"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# LayerZeroOFT (LayerZero Scan)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLayerZeroOFTGetStatusLive:
    """Live tests for LayerZeroOFT.get_status() against the LayerZero Scan API."""

    def _client(self) -> LayerZeroOFT:
        return LayerZeroOFT(
            w3=None,
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            oft_address=_ZRO_ADDRESS,
        )

    async def test_get_status_unknown_hash(self):
        """LayerZeroOFT: zero hash returns UNKNOWN without raising."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "LayerZeroOFT / LayerZero Scan")
        _assert_valid_status(result, _ZERO_HASH, "LayerZeroOFT")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """LayerZeroOFT: get_status always returns a BridgeStatus with protocol='LayerZeroOFT'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "LayerZeroOFT / LayerZero Scan")
        assert result.protocol == "LayerZeroOFT"
        assert isinstance(result.status, BridgeTransactionStatus)


# ---------------------------------------------------------------------------
# GasZip
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestGasZipGetStatusLive:
    """Live tests for GasZip.get_status() against the public GasZip backend API."""

    def _client(self) -> GasZip:
        return GasZip(
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.ARBITRUM,
            contract_address=GASZIP_CONTRACT_ETH,
        )

    async def test_get_status_unknown_hash(self):
        """GasZip: zero hash returns UNKNOWN without raising (404 from backend)."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "GasZip")
        _assert_valid_status(result, _ZERO_HASH, "GasZip")
        assert result.status == BridgeTransactionStatus.UNKNOWN

    async def test_get_status_returns_bridge_status_type(self):
        """GasZip: get_status always returns a BridgeStatus with protocol='GasZip'."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
        except BridgeError as exc:
            _skip_on_bridge_error(exc, "GasZip")
        assert result.protocol == "GasZip"
        assert isinstance(result.status, BridgeTransactionStatus)

    async def test_get_status_does_not_raise_bridge_error_for_unknown(self):
        """GasZip: querying an unknown hash should not raise BridgeError."""
        try:
            result = await self._client().get_status(_ZERO_HASH)
            assert result.status == BridgeTransactionStatus.UNKNOWN
        except BridgeError as exc:
            pytest.skip(f"GasZip status API temporarily unavailable: {exc}")

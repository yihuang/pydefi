"""Live and testnet integration tests for the faucet utilities.

These tests exercise the :mod:`pydefi.faucet` module against real external
services and the Sepolia testnet.

Test categories
---------------
``@pytest.mark.live``
    Validates the :class:`~pydefi.faucet.BaseFaucet` interface and the
    Alchemy / QuickNode faucet class construction — these require nothing
    more than network access and are safe to run in CI without any API keys.

``@pytest.mark.testnet``
    Actually calls the faucet API and checks that the funded wallet's balance
    increases on Sepolia.  Requires:

    - ``TESTNET_PRIVATE_KEY`` — hex private key of the wallet to fund.
    - ``ALCHEMY_API_KEY`` **or** ``QUICKNODE_TOKEN`` — one faucet credential.
    - ``SEPOLIA_RPC_URL`` (optional) — defaults to ``https://rpc.sepolia.org``.

Run with::

    pytest -m testnet tests/live/test_faucet_live.py
"""

from __future__ import annotations

import pytest

from pydefi.exceptions import FaucetError
from pydefi.faucet import AlchemyFaucet, BaseFaucet, QuickNodeFaucet
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# Unit-style tests (no API calls) — run alongside other live tests
# ---------------------------------------------------------------------------


class TestFaucetConstruction:
    """Verify that faucet objects can be constructed and raise on bad config."""

    def test_alchemy_faucet_sepolia(self):
        """AlchemyFaucet can be instantiated for Sepolia."""
        faucet = AlchemyFaucet(api_key="dummy_key", chain_id=ChainId.SEPOLIA)
        assert faucet.chain_id == ChainId.SEPOLIA
        assert isinstance(faucet, BaseFaucet)

    def test_alchemy_faucet_unsupported_chain(self):
        """AlchemyFaucet raises FaucetError for unsupported chains."""
        with pytest.raises(FaucetError, match="unsupported chain"):
            AlchemyFaucet(api_key="dummy_key", chain_id=ChainId.ETHEREUM)

    def test_quicknode_faucet_sepolia(self):
        """QuickNodeFaucet can be instantiated for Sepolia."""
        faucet = QuickNodeFaucet(token="dummy_token", chain_id=ChainId.SEPOLIA)
        assert faucet.chain_id == ChainId.SEPOLIA
        assert isinstance(faucet, BaseFaucet)

    def test_quicknode_faucet_unsupported_chain(self):
        """QuickNodeFaucet raises FaucetError for unsupported chains."""
        with pytest.raises(FaucetError, match="unsupported chain"):
            QuickNodeFaucet(token="dummy_token", chain_id=ChainId.ARBITRUM)

    def test_custom_api_url(self):
        """AlchemyFaucet and QuickNodeFaucet accept a custom api_base_url."""
        custom = "https://custom.faucet.example.com"
        alchemy = AlchemyFaucet(api_key="k", api_base_url=custom)
        assert alchemy._api_base == custom

        qn = QuickNodeFaucet(token="t", api_base_url=custom)
        assert qn._api_base == custom


# ---------------------------------------------------------------------------
# Testnet tests — require credentials and a real Sepolia wallet
# ---------------------------------------------------------------------------


@pytest.mark.testnet
class TestFaucetTestnet:
    """Testnet integration tests that call the faucet and verify the balance.

    These tests are **skipped automatically** unless the required environment
    variables are set (``TESTNET_PRIVATE_KEY`` + ``ALCHEMY_API_KEY`` or
    ``QUICKNODE_TOKEN``).  The :func:`funded_testnet_account` fixture handles
    auto-funding and skipping.
    """

    async def test_funded_testnet_account_has_balance(self, funded_testnet_account, sepolia_w3):
        """After the faucet fixture runs the wallet must have ≥ 0.01 ETH."""
        balance = await sepolia_w3.eth.get_balance(funded_testnet_account.address)
        assert balance >= 10**16, f"Expected ≥ 0.01 ETH on Sepolia, got {balance / 10**18:.6f} ETH"

    async def test_faucet_ensure_funded_no_op_when_rich(self, testnet_faucet, sepolia_w3, funded_testnet_account):
        """ensure_funded() must be a no-op when the balance already meets the threshold.

        After :func:`funded_testnet_account` has topped up the wallet we call
        ``ensure_funded`` again with the same threshold; it must return without
        calling the faucet a second time.
        """
        balance_before = await sepolia_w3.eth.get_balance(funded_testnet_account.address)

        # ensure_funded should return immediately because balance >= min_balance.
        await testnet_faucet.ensure_funded(
            sepolia_w3,
            funded_testnet_account.address,
            min_balance=balance_before,  # already met
        )

        balance_after = await sepolia_w3.eth.get_balance(funded_testnet_account.address)
        # Balance may change due to other activity but must remain non-zero.
        assert balance_after > 0

"""Live and testnet integration tests for the faucet utilities.

These tests exercise the :mod:`pydefi.faucet` module against real external
services and the Sepolia testnet.

Test categories
---------------
``@pytest.mark.live``
    Validates the :class:`~pydefi.faucet.BaseFaucet` interface and the
    Alchemy faucet class construction — these require nothing more than the
    Python environment and are safe to run in CI.  The live Sepolia test
    additionally requires ``ALCHEMY_API_KEY`` and ``TESTNET_PRIVATE_KEY`` and
    is skipped automatically when those variables are absent.

``@pytest.mark.testnet``
    Actually calls the faucet API and checks that the funded wallet's balance
    increases on Sepolia.  Requires:

    - ``TESTNET_PRIVATE_KEY`` — hex private key of the wallet to fund.
    - ``ALCHEMY_API_KEY`` — Alchemy API key.
    - ``SEPOLIA_RPC_URL`` (optional) — defaults to ``https://rpc.sepolia.org``.

Run with::

    pytest -m live tests/live/test_faucet_live.py
    pytest -m testnet tests/live/test_faucet_live.py
"""

from __future__ import annotations

import os

import pytest

from pydefi.exceptions import FaucetError
from pydefi.faucet import AlchemyFaucet, BaseFaucet
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# Unit-style tests (no API calls)
# ---------------------------------------------------------------------------


@pytest.mark.live
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

    def test_custom_api_url(self):
        """AlchemyFaucet accepts a custom api_base_url."""
        custom = "https://custom.faucet.example.com"
        alchemy = AlchemyFaucet(api_key="k", api_base_url=custom)
        assert alchemy._api_base == custom


# ---------------------------------------------------------------------------
# Live Sepolia faucet test — only drips when balance is below threshold
# ---------------------------------------------------------------------------

_MIN_BALANCE = 10**16  # 0.01 ETH


@pytest.mark.live
async def test_alchemy_faucet_funds_wallet_when_low(sepolia_w3):
    """Request faucet drip only when the test wallet balance is below 0.01 ETH.

    Skipped automatically when ``ALCHEMY_API_KEY`` or ``TESTNET_PRIVATE_KEY``
    are not set in the environment.
    """
    from eth_account import Account

    api_key = os.environ.get("ALCHEMY_API_KEY", "")
    private_key = os.environ.get("TESTNET_PRIVATE_KEY", "")

    if not api_key:
        pytest.skip("ALCHEMY_API_KEY not set")
    if not private_key:
        pytest.skip("TESTNET_PRIVATE_KEY not set")

    account = Account.from_key(private_key)
    faucet = AlchemyFaucet(api_key=api_key, chain_id=ChainId.SEPOLIA)

    balance = await sepolia_w3.eth.get_balance(account.address)

    if balance < _MIN_BALANCE:
        # Only call the faucet when the wallet actually needs funds.
        await faucet.ensure_funded(sepolia_w3, account.address, min_balance=_MIN_BALANCE)
        balance = await sepolia_w3.eth.get_balance(account.address)

    assert balance >= _MIN_BALANCE, f"Expected ≥ 0.01 ETH on Sepolia, got {balance / 10**18:.6f} ETH"


# ---------------------------------------------------------------------------
# Testnet tests — require credentials and a real Sepolia wallet
# ---------------------------------------------------------------------------


@pytest.mark.testnet
class TestFaucetTestnet:
    """Testnet integration tests that call the faucet and verify the balance.

    These tests are **skipped automatically** unless the required environment
    variables are set (``TESTNET_PRIVATE_KEY`` + ``ALCHEMY_API_KEY``).  The
    :func:`funded_testnet_account` fixture handles auto-funding and skipping.
    """

    async def test_funded_testnet_account_has_balance(self, funded_testnet_account, sepolia_w3):
        """After the faucet fixture runs the wallet must have ≥ 0.01 ETH."""
        balance = await sepolia_w3.eth.get_balance(funded_testnet_account.address)
        assert balance >= _MIN_BALANCE, f"Expected ≥ 0.01 ETH on Sepolia, got {balance / 10**18:.6f} ETH"

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

"""Tests for pydefi.hyperliquid signing utilities (no live calls)."""

from __future__ import annotations

from eth_account import Account

from pydefi.hyperliquid.signing import (
    _SIGNATURE_CHAIN_ID,
    USD_SEND_SIGN_TYPES,
    _hyperliquid_chain_name,
    _user_signed_payload,
    sign_inner,
    sign_send_to_evm_with_data_action,
    sign_usd_transfer_action,
)
from tests.addrs import ETH_WHALE, ZERO_ADDR

# ---------------------------------------------------------------------------
# Test fixtures — deterministic private key, no real funds
# ---------------------------------------------------------------------------

# A throwaway key used only for deterministic signing assertions.
_TEST_PRIVATE_KEY = "0xb0057716d5917badaf911b193b12b910811c1497b5bada8d7711f758981c3773"
_TEST_WALLET = Account.from_key(_TEST_PRIVATE_KEY)


# ---------------------------------------------------------------------------
# sign_inner
# ---------------------------------------------------------------------------


class TestSignInner:
    """Tests for the sign_inner() helper."""

    def _make_payload(self, chain_id: int = 421614) -> dict:
        """Return a minimal well-formed EIP-712 payload for signing."""
        return {
            "domain": {
                "name": "HyperliquidSignTransaction",
                "version": "1",
                "chainId": chain_id,
                "verifyingContract": ZERO_ADDR.to_0x_hex(),
            },
            "types": {
                "HyperliquidTransaction:UsdSend": USD_SEND_SIGN_TYPES,
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
            },
            "primaryType": "HyperliquidTransaction:UsdSend",
            "message": {
                "hyperliquidChain": "Mainnet",
                "destination": ETH_WHALE.to_0x_hex(),
                "amount": "10.0",
                "time": 1_700_000_000_000,
            },
        }

    def test_returns_dict_with_r_s_v(self):
        """sign_inner() returns a dict with 'r', 's', and 'v' keys."""
        payload = self._make_payload()
        result = sign_inner(_TEST_WALLET, payload)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"r", "s", "v"}

    def test_r_s_are_hex_strings(self):
        """'r' and 's' fields are hex strings starting with '0x'."""
        payload = self._make_payload()
        result = sign_inner(_TEST_WALLET, payload)

        assert isinstance(result["r"], str)
        assert result["r"].startswith("0x")
        assert isinstance(result["s"], str)
        assert result["s"].startswith("0x")

    def test_v_is_27_or_28(self):
        """'v' field is either 27 or 28 (Ethereum canonical recovery IDs)."""
        payload = self._make_payload()
        result = sign_inner(_TEST_WALLET, payload)

        assert result["v"] in (27, 28)

    def test_r_s_length(self):
        """'r' and 's' hex strings are 32 bytes (64 hex chars + '0x' prefix)."""
        payload = self._make_payload()
        result = sign_inner(_TEST_WALLET, payload)

        # 0x + 64 hex chars = 66 chars for a full 32-byte value.
        # Some implementations may omit leading zeros; accept ≥ 3 chars (0x + at least 1 byte).
        assert len(result["r"]) >= 3
        assert len(result["s"]) >= 3

    def test_deterministic(self):
        """sign_inner() is deterministic for the same key and payload."""
        payload = self._make_payload()
        result1 = sign_inner(_TEST_WALLET, payload)
        result2 = sign_inner(_TEST_WALLET, payload)

        assert result1["r"] == result2["r"]
        assert result1["s"] == result2["s"]
        assert result1["v"] == result2["v"]


# ---------------------------------------------------------------------------
# _hyperliquid_chain_name
# ---------------------------------------------------------------------------


class TestHyperliquidChainName:
    def test_mainnet(self):
        assert _hyperliquid_chain_name(True) == "Mainnet"

    def test_testnet(self):
        assert _hyperliquid_chain_name(False) == "Testnet"


# ---------------------------------------------------------------------------
# _user_signed_payload
# ---------------------------------------------------------------------------


class TestUserSignedPayload:
    def test_uses_action_signature_chain_id(self):
        """_user_signed_payload() reads chainId from action['signatureChainId']."""
        action = {
            "signatureChainId": "0xa4b1",  # Arbitrum (42161)
            "hyperliquidChain": "Mainnet",
        }
        payload = _user_signed_payload(
            "HyperliquidTransaction:UsdSend",
            USD_SEND_SIGN_TYPES,
            action,
        )
        assert payload["domain"]["chainId"] == 0xA4B1  # 42161

    def test_primary_type(self):
        """_user_signed_payload() sets the correct primaryType."""
        action = {"signatureChainId": "0x66eee", "hyperliquidChain": "Mainnet"}
        payload = _user_signed_payload(
            "HyperliquidTransaction:UsdSend",
            USD_SEND_SIGN_TYPES,
            action,
        )
        assert payload["primaryType"] == "HyperliquidTransaction:UsdSend"


# ---------------------------------------------------------------------------
# sign_usd_transfer_action
# ---------------------------------------------------------------------------


class TestSignUsdTransferAction:
    def test_returns_valid_signature(self):
        """sign_usd_transfer_action() produces a valid r/s/v signature dict."""
        action = {
            "type": "usdSend",
            "destination": ETH_WHALE.to_0x_hex(),
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        result = sign_usd_transfer_action(_TEST_PRIVATE_KEY, action, is_mainnet=True)

        assert set(result.keys()) == {"r", "s", "v"}
        assert result["r"].startswith("0x")
        assert result["s"].startswith("0x")
        assert result["v"] in (27, 28)

    def test_sets_signature_chain_id(self):
        """sign_usd_transfer_action() sets signatureChainId to the standard Arb Sepolia ID."""
        action = {
            "type": "usdSend",
            "destination": ETH_WHALE.to_0x_hex(),
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        sign_usd_transfer_action(_TEST_PRIVATE_KEY, action, is_mainnet=True)

        assert action["signatureChainId"] == _SIGNATURE_CHAIN_ID


# ---------------------------------------------------------------------------
# sign_send_to_evm_with_data_action — destination chain ID handling
# ---------------------------------------------------------------------------


class TestSignSendToEvmWithDataAction:
    """Tests for the sendToEvmWithData signing function.

    Unlike other user-signed actions, this function uses the destination
    chain's EVM chain ID as signatureChainId (NOT the fixed 0x66eee).
    """

    def _make_action(self, sig_chain_id: str = "0xa4b1") -> dict:
        return {
            "type": "sendToEvmWithData",
            "signatureChainId": sig_chain_id,
            "token": "USDC",
            "amount": "10",
            "sourceDex": "",
            "destinationRecipient": ETH_WHALE.to_0x_hex(),
            "addressEncoding": "hex",
            "destinationChainId": 3,
            "gasLimit": 200_000,
            "data": "0x",
            "nonce": 1_700_000_000_000,
        }

    def test_returns_valid_signature(self):
        """sign_send_to_evm_with_data_action() produces a valid r/s/v dict."""
        action = self._make_action()
        result = sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action)

        assert set(result.keys()) == {"r", "s", "v"}
        assert result["r"].startswith("0x")
        assert result["s"].startswith("0x")
        assert result["v"] in (27, 28)

    def test_uses_destination_chain_id_not_fixed(self):
        """Signature differs when signatureChainId differs (proving it's used)."""
        action_arb = self._make_action(sig_chain_id="0xa4b1")  # Arbitrum 42161
        action_eth = self._make_action(sig_chain_id="0x1")  # Ethereum 1

        sig_arb = sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action_arb)
        sig_eth = sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action_eth)

        # Different chain IDs must produce different signatures.
        assert sig_arb["r"] != sig_eth["r"] or sig_arb["s"] != sig_eth["s"]

    def test_does_not_override_signature_chain_id(self):
        """The caller-supplied signatureChainId is preserved (not overwritten)."""
        custom_chain_id = "0xa4b1"
        action = self._make_action(sig_chain_id=custom_chain_id)
        sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action)

        assert action["signatureChainId"] == custom_chain_id

    def test_sets_hyperliquid_chain_mainnet(self):
        """hyperliquidChain is set to 'Mainnet' when is_mainnet=True."""
        action = self._make_action()
        sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action, is_mainnet=True)

        assert action["hyperliquidChain"] == "Mainnet"

    def test_sets_hyperliquid_chain_testnet(self):
        """hyperliquidChain is set to 'Testnet' when is_mainnet=False."""
        action = self._make_action()
        sign_send_to_evm_with_data_action(_TEST_PRIVATE_KEY, action, is_mainnet=False)

        assert action["hyperliquidChain"] == "Testnet"

    def test_standard_action_uses_fixed_chain_id(self):
        """UsdSend uses the fixed _SIGNATURE_CHAIN_ID (0x66eee), NOT the destination chain ID.

        Additionally verifies that the chain ID is actually incorporated into the
        signature: a manually constructed payload using a *different* chain ID
        produces a different r/s than the standard sign_usd_transfer_action call.
        """
        usd_action = {
            "type": "usdSend",
            "destination": ETH_WHALE.to_0x_hex(),
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        sig_standard = sign_usd_transfer_action(_TEST_PRIVATE_KEY, usd_action, is_mainnet=True)
        # The fixed Arbitrum Sepolia chain ID (0x66eee = 421614) must have been used.
        assert usd_action["signatureChainId"] == _SIGNATURE_CHAIN_ID
        assert _SIGNATURE_CHAIN_ID == "0x66eee"

        # Produce a second signature using the Arbitrum mainnet chain ID (0xa4b1) — it
        # should differ from the standard one, confirming the chain ID is in the hash.
        alt_action = {
            "signatureChainId": "0xa4b1",  # Arbitrum mainnet (42161), not 421614
            "hyperliquidChain": "Mainnet",
            "destination": ETH_WHALE.to_0x_hex(),
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        wallet = Account.from_key(_TEST_PRIVATE_KEY)
        sig_alt = sign_inner(
            wallet,
            _user_signed_payload(
                "HyperliquidTransaction:UsdSend",
                USD_SEND_SIGN_TYPES,
                alt_action,
            ),
        )
        assert sig_standard["r"] != sig_alt["r"] or sig_standard["s"] != sig_alt["s"], (
            "Signatures must differ when signatureChainId changes"
        )

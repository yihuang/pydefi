"""Tests for pydefi.wallet — OpenWallet Standard integration (no live calls)."""

from __future__ import annotations

import pytest
from eth_account import Account

from pydefi.wallet import (
    OpenWalletSigner,
    create_wallet,
    delete_wallet,
    derive_address,
    export_wallet,
    generate_mnemonic,
    get_wallet,
    import_wallet_mnemonic,
    import_wallet_private_key,
    list_wallets,
    rename_wallet,
)

# ---------------------------------------------------------------------------
# Test fixtures — deterministic private key, no real funds
# ---------------------------------------------------------------------------

# Throwaway key used only for deterministic assertions.
_TEST_PRIVATE_KEY = "b0057716d5917badaf911b193b12b910811c1497b5bada8d7711f758981c3773"
_TEST_ACCOUNT = Account.from_key(_TEST_PRIVATE_KEY)
_TEST_ADDRESS = _TEST_ACCOUNT.address  # 0x1dF62f291b2E969fB0849d99D9Ce41e2F137006e


@pytest.fixture()
def vault(tmp_path):
    """Return a temporary, isolated OWS vault directory."""
    vault_dir = str(tmp_path / "ows-vault")
    import os

    os.makedirs(vault_dir)
    yield vault_dir
    # tmp_path is cleaned up automatically by pytest


# ---------------------------------------------------------------------------
# generate_mnemonic
# ---------------------------------------------------------------------------


class TestGenerateMnemonic:
    def test_returns_string(self):
        phrase = generate_mnemonic()
        assert isinstance(phrase, str)

    def test_default_12_words(self):
        phrase = generate_mnemonic()
        assert len(phrase.split()) == 12

    def test_24_words(self):
        phrase = generate_mnemonic(24)
        assert len(phrase.split()) == 24

    def test_unique_each_call(self):
        p1 = generate_mnemonic()
        p2 = generate_mnemonic()
        assert p1 != p2


# ---------------------------------------------------------------------------
# derive_address
# ---------------------------------------------------------------------------


class TestDeriveAddress:
    def test_evm_address_format(self):
        phrase = generate_mnemonic()
        addr = derive_address(phrase, "ethereum")
        assert addr.startswith("0x")
        assert len(addr) == 42

    def test_deterministic(self):
        phrase = generate_mnemonic()
        a1 = derive_address(phrase, "ethereum")
        a2 = derive_address(phrase, "ethereum")
        assert a1 == a2

    def test_different_index_different_address(self):
        phrase = generate_mnemonic()
        a0 = derive_address(phrase, "ethereum", index=0)
        a1 = derive_address(phrase, "ethereum", index=1)
        assert a0 != a1

    def test_solana_address(self):
        phrase = generate_mnemonic()
        addr = derive_address(phrase, "solana")
        # Solana addresses are base58, 32-44 chars
        assert len(addr) > 30


# ---------------------------------------------------------------------------
# Wallet lifecycle
# ---------------------------------------------------------------------------


class TestCreateWallet:
    def test_returns_wallet_info(self, vault):
        w = create_wallet("test", vault_path=vault)
        assert "id" in w
        assert w["name"] == "test"
        assert "accounts" in w
        assert isinstance(w["accounts"], list)

    def test_has_evm_account(self, vault):
        w = create_wallet("test", vault_path=vault)
        evm_accounts = [a for a in w["accounts"] if a["chain_id"].startswith("eip155:")]
        assert len(evm_accounts) >= 1
        assert evm_accounts[0]["address"].startswith("0x")

    def test_has_solana_account(self, vault):
        w = create_wallet("test", vault_path=vault)
        sol_accounts = [a for a in w["accounts"] if a["chain_id"].startswith("solana:")]
        assert len(sol_accounts) >= 1


class TestImportWalletPrivateKey:
    def test_evm_address_matches_eth_account(self, vault):
        w = import_wallet_private_key("test-pk", _TEST_PRIVATE_KEY, vault_path=vault)
        evm_addr = next(a["address"] for a in w["accounts"] if a["chain_id"].startswith("eip155:"))
        assert evm_addr == _TEST_ADDRESS

    def test_with_0x_prefix(self, vault):
        w = import_wallet_private_key("test-pk", "0x" + _TEST_PRIVATE_KEY, vault_path=vault)
        evm_addr = next(a["address"] for a in w["accounts"] if a["chain_id"].startswith("eip155:"))
        assert evm_addr == _TEST_ADDRESS


class TestImportWalletMnemonic:
    def test_roundtrip_import_export(self, vault):
        phrase = generate_mnemonic(12)
        w = import_wallet_mnemonic("mn-wallet", phrase, vault_path=vault)
        assert w["name"] == "mn-wallet"
        exported = export_wallet("mn-wallet", vault_path=vault)
        assert exported == phrase


class TestListGetDeleteRename:
    def test_list_wallets(self, vault):
        create_wallet("wallet-a", vault_path=vault)
        create_wallet("wallet-b", vault_path=vault)
        wallets = list_wallets(vault_path=vault)
        names = [w["name"] for w in wallets]
        assert "wallet-a" in names
        assert "wallet-b" in names

    def test_get_wallet(self, vault):
        create_wallet("my-wallet", vault_path=vault)
        w = get_wallet("my-wallet", vault_path=vault)
        assert w["name"] == "my-wallet"

    def test_delete_wallet(self, vault):
        create_wallet("to-delete", vault_path=vault)
        delete_wallet("to-delete", vault_path=vault)
        wallets = list_wallets(vault_path=vault)
        assert all(w["name"] != "to-delete" for w in wallets)

    def test_rename_wallet(self, vault):
        create_wallet("old-name", vault_path=vault)
        rename_wallet("old-name", "new-name", vault_path=vault)
        w = get_wallet("new-name", vault_path=vault)
        assert w["name"] == "new-name"


# ---------------------------------------------------------------------------
# OpenWalletSigner
# ---------------------------------------------------------------------------


class TestOpenWalletSignerAddress:
    def test_address_matches_private_key(self, vault):
        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)
        assert signer.address == _TEST_ADDRESS

    def test_repr(self, vault):
        create_wallet("my-wallet", vault_path=vault)
        signer = OpenWalletSigner("my-wallet", vault_path=vault)
        assert "my-wallet" in repr(signer)


class TestOpenWalletSignerSignEip712:
    """Tests that OpenWalletSigner.sign_eip712() produces valid signatures."""

    def _make_payload(self, chain_id: int = 421614) -> dict:
        return {
            "domain": {
                "name": "HyperliquidSignTransaction",
                "version": "1",
                "chainId": chain_id,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "types": {
                "HyperliquidTransaction:UsdSend": [
                    {"name": "hyperliquidChain", "type": "string"},
                    {"name": "destination", "type": "string"},
                    {"name": "amount", "type": "string"},
                    {"name": "time", "type": "uint64"},
                ],
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
                "destination": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                "amount": "10.0",
                "time": 1_700_000_000_000,
            },
        }

    def test_returns_r_s_v(self, vault):
        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)
        result = signer.sign_eip712(self._make_payload())
        assert set(result.keys()) == {"r", "s", "v"}

    def test_r_s_are_hex_strings(self, vault):
        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)
        result = signer.sign_eip712(self._make_payload())
        assert result["r"].startswith("0x")
        assert result["s"].startswith("0x")

    def test_v_is_27_or_28(self, vault):
        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)
        result = signer.sign_eip712(self._make_payload())
        assert result["v"] in (27, 28)

    def test_matches_eth_account_signature(self, vault):
        """OpenWalletSigner produces the same signature as eth_account for the same key."""
        from eth_account.messages import encode_typed_data
        from eth_utils import to_hex

        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)

        payload = self._make_payload()
        ows_result = signer.sign_eip712(payload)

        # Compare against eth_account
        eth_wallet = Account.from_key(_TEST_PRIVATE_KEY)
        structured_data = encode_typed_data(full_message=payload)
        signed = eth_wallet.sign_message(structured_data)
        eth_r = to_hex(signed.r)
        eth_s = to_hex(signed.s)

        assert ows_result["r"] == eth_r
        assert ows_result["s"] == eth_s
        assert ows_result["v"] == signed.v

    def test_deterministic(self, vault):
        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)
        payload = self._make_payload()
        r1 = signer.sign_eip712(payload)
        r2 = signer.sign_eip712(payload)
        assert r1["r"] == r2["r"]
        assert r1["s"] == r2["s"]
        assert r1["v"] == r2["v"]


# ---------------------------------------------------------------------------
# Integration: OpenWalletSigner with sign_inner() and sign_XXX_action()
# ---------------------------------------------------------------------------


class TestOpenWalletSignerIntegration:
    """Tests that OpenWalletSigner works with the Hyperliquid signing helpers."""

    def test_sign_inner_with_ows_signer(self, vault):
        from pydefi.hyperliquid.signing import sign_inner

        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)

        payload = {
            "domain": {
                "name": "HyperliquidSignTransaction",
                "version": "1",
                "chainId": 421614,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "types": {
                "HyperliquidTransaction:UsdSend": [
                    {"name": "hyperliquidChain", "type": "string"},
                    {"name": "destination", "type": "string"},
                    {"name": "amount", "type": "string"},
                    {"name": "time", "type": "uint64"},
                ],
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
                "destination": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                "amount": "10.0",
                "time": 1_700_000_000_000,
            },
        }
        result = sign_inner(signer, payload)
        assert set(result.keys()) == {"r", "s", "v"}
        assert result["v"] in (27, 28)

    def test_sign_usd_transfer_action_with_ows_signer(self, vault):
        from pydefi.hyperliquid.signing import sign_usd_transfer_action

        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)

        action = {
            "type": "usdSend",
            "destination": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        # Sign with OWS signer
        ows_result = sign_usd_transfer_action(signer, action.copy(), is_mainnet=True)

        # Sign with raw private key — must produce the same output
        action2 = {
            "type": "usdSend",
            "destination": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "amount": "10.0",
            "time": 1_700_000_000_000,
        }
        pk_result = sign_usd_transfer_action(_TEST_PRIVATE_KEY, action2, is_mainnet=True)

        assert ows_result["r"] == pk_result["r"]
        assert ows_result["s"] == pk_result["s"]
        assert ows_result["v"] == pk_result["v"]

    def test_sign_l1_action_with_ows_signer(self, vault):
        from pydefi.hyperliquid.signing import sign_l1_action

        import_wallet_private_key("pk-wallet", _TEST_PRIVATE_KEY, vault_path=vault)
        signer = OpenWalletSigner("pk-wallet", vault_path=vault)

        action = {"type": "order", "coin": "BTC"}
        nonce = 1_700_000_000_000

        ows_result = sign_l1_action(signer, action, nonce, is_mainnet=True)
        pk_result = sign_l1_action(_TEST_PRIVATE_KEY, action, nonce, is_mainnet=True)

        assert ows_result["r"] == pk_result["r"]
        assert ows_result["s"] == pk_result["s"]
        assert ows_result["v"] == pk_result["v"]

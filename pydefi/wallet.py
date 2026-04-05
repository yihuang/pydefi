"""
OpenWallet Standard (OWS) integration for pydefi.

Provides a thin wrapper around the ``open-wallet-standard`` (``ows``) Python
package so that pydefi signing functions can use OWS-managed wallets instead
of raw private-key hex strings.

``open-wallet-standard`` keeps key material encrypted at rest and applies a
policy engine before every signing operation.  Install it with::

    pip install open-wallet-standard

Quick start::

    from pydefi.wallet import create_wallet, OpenWalletSigner
    from pydefi.hyperliquid import HyperliquidClient

    # Create (or import) a wallet once
    wallet = create_wallet("agent-treasury")

    # Build a signer and pass it wherever pydefi expects a private key
    signer = OpenWalletSigner("agent-treasury")

    client = HyperliquidClient()
    await client.usd_send(signer, destination="0x...", amount="10", nonce=...)

All ``ows.*`` management functions (``create_wallet``, ``list_wallets``, etc.)
are re-exported from this module so callers only need one import.

Note:
    The ``vault_path`` parameter accepted by every function here maps to the
    ``vault_path_opt`` keyword in the underlying ``ows`` API.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import ows as _ows

    _OWS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ows = None  # type: ignore[assignment]
    _OWS_AVAILABLE = False


def _require_ows() -> None:
    """Raise ImportError with an install hint if ``ows`` is not installed."""
    if not _OWS_AVAILABLE:
        raise ImportError(
            "The 'open-wallet-standard' package is required for wallet management. "
            "Install it with: pip install open-wallet-standard"
        )


# ---------------------------------------------------------------------------
# Re-exported wallet management functions
# ---------------------------------------------------------------------------


def generate_mnemonic(words: int = 12) -> str:
    """Generate a new BIP-39 mnemonic phrase.

    Args:
        words: Number of mnemonic words — ``12`` or ``24``.

    Returns:
        Space-separated mnemonic string.
    """
    _require_ows()
    return _ows.generate_mnemonic(words)


def derive_address(mnemonic: str, chain: str, index: int = 0) -> str:
    """Derive an address from a mnemonic without creating a wallet.

    Args:
        mnemonic: BIP-39 mnemonic phrase.
        chain: Chain name — ``"ethereum"``, ``"solana"``, ``"bitcoin"``, etc.
        index: Account index in the derivation path.

    Returns:
        Chain-native address string.
    """
    _require_ows()
    return _ows.derive_address(mnemonic, chain, index if index else None)


def create_wallet(
    name: str,
    passphrase: str | None = None,
    words: int = 12,
    vault_path: str | None = None,
) -> dict[str, Any]:
    """Create a new wallet with addresses for all supported chains.

    Args:
        name: Human-readable wallet name.
        passphrase: Optional encryption passphrase.
        words: Mnemonic word count (``12`` or ``24``).
        vault_path: Custom vault directory.  Defaults to the OWS system vault.

    Returns:
        WalletInfo dict with ``id``, ``name``, ``created_at``, and
        ``accounts`` keys.
    """
    _require_ows()
    return _ows.create_wallet(name, passphrase=passphrase, words=words, vault_path_opt=vault_path)


def import_wallet_mnemonic(
    name: str,
    mnemonic: str,
    passphrase: str | None = None,
    index: int | None = None,
    vault_path: str | None = None,
) -> dict[str, Any]:
    """Import a wallet from a BIP-39 mnemonic phrase.

    Args:
        name: Human-readable wallet name.
        mnemonic: BIP-39 mnemonic phrase.
        passphrase: Optional encryption passphrase.
        index: Account index in the derivation path.
        vault_path: Custom vault directory.

    Returns:
        WalletInfo dict.
    """
    _require_ows()
    return _ows.import_wallet_mnemonic(
        name,
        mnemonic,
        passphrase=passphrase,
        index=index,
        vault_path_opt=vault_path,
    )


def import_wallet_private_key(
    name: str,
    private_key_hex: str,
    chain: str = "ethereum",
    passphrase: str | None = None,
    vault_path: str | None = None,
    secp256k1_key: str | None = None,
    ed25519_key: str | None = None,
) -> dict[str, Any]:
    """Import a wallet from a hex-encoded private key.

    A random key is generated for the curve not covered by the provided key.

    Args:
        name: Human-readable wallet name.
        private_key_hex: Hex-encoded private key (with or without ``0x``
            prefix).  Ignored when both *secp256k1_key* and *ed25519_key* are
            provided.
        chain: Source chain that identifies the key curve — ``"ethereum"``
            (default, secp256k1) or ``"solana"`` / ``"sui"`` / ``"ton"``
            (Ed25519).
        passphrase: Optional encryption passphrase.
        vault_path: Custom vault directory.
        secp256k1_key: Explicit secp256k1 private key (hex).
        ed25519_key: Explicit Ed25519 private key (hex).

    Returns:
        WalletInfo dict.
    """
    _require_ows()
    # Strip 0x prefix if present — ows expects raw hex
    pk = private_key_hex.removeprefix("0x") if isinstance(private_key_hex, str) else private_key_hex
    return _ows.import_wallet_private_key(
        name,
        pk,
        chain=chain,
        passphrase=passphrase,
        vault_path_opt=vault_path,
        secp256k1_key=secp256k1_key,
        ed25519_key=ed25519_key,
    )


def list_wallets(vault_path: str | None = None) -> list[dict[str, Any]]:
    """List all wallets in the vault.

    Args:
        vault_path: Custom vault directory.

    Returns:
        List of WalletInfo dicts.
    """
    _require_ows()
    return _ows.list_wallets(vault_path_opt=vault_path)


def get_wallet(name_or_id: str, vault_path: str | None = None) -> dict[str, Any]:
    """Look up a wallet by name or UUID.

    Args:
        name_or_id: Wallet name or UUID v4 string.
        vault_path: Custom vault directory.

    Returns:
        WalletInfo dict.
    """
    _require_ows()
    return _ows.get_wallet(name_or_id, vault_path_opt=vault_path)


def delete_wallet(name_or_id: str, vault_path: str | None = None) -> None:
    """Delete a wallet from the vault.

    Args:
        name_or_id: Wallet name or UUID v4 string.
        vault_path: Custom vault directory.
    """
    _require_ows()
    _ows.delete_wallet(name_or_id, vault_path_opt=vault_path)


def rename_wallet(name_or_id: str, new_name: str, vault_path: str | None = None) -> None:
    """Rename a wallet.

    Args:
        name_or_id: Wallet name or UUID v4 string.
        new_name: New wallet name.
        vault_path: Custom vault directory.
    """
    _require_ows()
    _ows.rename_wallet(name_or_id, new_name, vault_path_opt=vault_path)


def export_wallet(
    name_or_id: str,
    passphrase: str | None = None,
    vault_path: str | None = None,
) -> str:
    """Export a wallet's mnemonic phrase or private keys.

    Mnemonic wallets return the phrase string.  Private-key wallets return a
    JSON string with both curve keys.

    Args:
        name_or_id: Wallet name or UUID v4 string.
        passphrase: Decryption passphrase (if the wallet was created with one).
        vault_path: Custom vault directory.

    Returns:
        Mnemonic string or JSON-encoded key dict.
    """
    _require_ows()
    return _ows.export_wallet(name_or_id, passphrase=passphrase, vault_path_opt=vault_path)


# ---------------------------------------------------------------------------
# JSON helper for EIP-712 serialization
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> str:
    """JSON serialiser for types not handled by the standard encoder.

    EIP-712 payloads may contain ``bytes`` values (e.g. ``bytes32``
    ``connectionId`` in phantom-agent signing).  These are encoded as
    ``0x``-prefixed hex strings, which is the canonical representation
    expected by typed-data signers.
    """
    if isinstance(obj, (bytes, bytearray)):
        return "0x" + obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# OpenWalletSigner — EIP-712 signer backed by an OWS-managed wallet
# ---------------------------------------------------------------------------


class OpenWalletSigner:
    """EIP-712 signer that delegates to an ``open-wallet-standard`` wallet.

    Use this anywhere pydefi accepts a private-key hex string.  The key
    material never leaves the OWS vault — only the signature bytes are
    returned.

    Args:
        wallet_name: Name (or UUID) of the OWS wallet to sign with.
        passphrase: Decryption passphrase used when the wallet was created.
        index: Account index (HD path index) to use for signing.
        vault_path: Custom vault directory.  Defaults to the system vault.

    Example::

        from pydefi.wallet import OpenWalletSigner
        from pydefi.hyperliquid import HyperliquidClient

        signer = OpenWalletSigner("agent-treasury")
        client = HyperliquidClient()
        await client.usd_send(signer, destination="0x...", amount="10", nonce=...)
    """

    def __init__(
        self,
        wallet_name: str,
        passphrase: str | None = None,
        index: int | None = None,
        vault_path: str | None = None,
    ) -> None:
        _require_ows()
        self.wallet_name = wallet_name
        self.passphrase = passphrase
        self.index = index
        self.vault_path = vault_path

    # ------------------------------------------------------------------
    # EIP-712 signing — used by pydefi.hyperliquid.signing.sign_inner()
    # ------------------------------------------------------------------

    def sign_eip712(self, data: dict[str, Any]) -> dict[str, str | int]:
        """Sign an EIP-712 payload and return ``{"r", "s", "v"}``.

        This method is called by :func:`pydefi.hyperliquid.signing.sign_inner`
        when an :class:`OpenWalletSigner` is used as the ``wallet`` argument.

        Args:
            data: Fully-formed EIP-712 payload dict with ``domain``,
                ``types``, ``primaryType``, and ``message`` keys.

        Returns:
            ``{"r": "0x...", "s": "0x...", "v": 27|28}``
        """
        result = _ows.sign_typed_data(
            self.wallet_name,
            "ethereum",
            json.dumps(data, default=_json_default),
            passphrase=self.passphrase,
            index=self.index,
            vault_path_opt=self.vault_path,
        )
        sig_bytes = bytes.fromhex(result["signature"])
        # Use int conversion so that leading-zero bytes are dropped, matching
        # the behaviour of eth_utils.to_hex() used in the eth_account path.
        r = hex(int.from_bytes(sig_bytes[:32], "big"))
        s = hex(int.from_bytes(sig_bytes[32:64], "big"))
        v: int = result["recovery_id"]
        return {"r": r, "s": s, "v": v}

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------

    @property
    def address(self) -> str:
        """Return the EVM (EIP-55 checksummed) address for this signer."""
        wallet_info = get_wallet(self.wallet_name, vault_path=self.vault_path)
        for account in wallet_info["accounts"]:
            if account["chain_id"].startswith("eip155:"):
                return account["address"]
        raise ValueError(f"No EVM account found in wallet '{self.wallet_name}'")

    def __repr__(self) -> str:
        return f"OpenWalletSigner(wallet_name={self.wallet_name!r})"


__all__ = [
    "OpenWalletSigner",
    "generate_mnemonic",
    "derive_address",
    "create_wallet",
    "import_wallet_mnemonic",
    "import_wallet_private_key",
    "list_wallets",
    "get_wallet",
    "delete_wallet",
    "rename_wallet",
    "export_wallet",
]

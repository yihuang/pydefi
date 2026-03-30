"""
Async HTTP client for Hyperliquid L1 info and exchange APIs.

Hyperliquid exposes two REST endpoints:

* ``POST /info``     — read-only queries (no authentication required).
* ``POST /exchange`` — signed action submission (requires EIP-712 signature).

The :class:`HyperliquidClient` wraps both endpoints and exposes convenience
methods for the most common operations.  For raw access use :meth:`post_info`
and :meth:`post_action`.

HyperEVM (the EVM layer of Hyperliquid) runs at a separate JSON-RPC endpoint.
Use :attr:`evm_rpc_url` to get the URL, or :meth:`make_evm_w3` to create a
ready-to-use :class:`~web3.AsyncWeb3` instance.

Docs:
    https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""

from __future__ import annotations

import time
from typing import Any

import aiohttp
from web3 import AsyncWeb3

from pydefi.hyperliquid.signing import (
    sign_l1_action,
    sign_send_asset_action,
    sign_send_to_evm_with_data_action,
    sign_spot_transfer_action,
    sign_usd_class_transfer_action,
    sign_usd_transfer_action,
    sign_withdraw_action,
)

# ---------------------------------------------------------------------------
# Well-known endpoints
# ---------------------------------------------------------------------------

_MAINNET_API: str = "https://api.hyperliquid.xyz"
_TESTNET_API: str = "https://api.hyperliquid-testnet.xyz"

# HyperEVM JSON-RPC endpoints
_HYPER_EVM_MAINNET_RPC: str = "https://rpc.hyperliquid.xyz/evm"
_HYPER_EVM_TESTNET_RPC: str = "https://rpc.hyperliquid-testnet.xyz/evm"


class HyperliquidClient:
    """Async client for the Hyperliquid L1 info and exchange APIs.

    Handles both read-only info queries and authenticated exchange actions.
    All methods are async and require an ``async with`` or ``await`` at the
    call site.

    Args:
        is_mainnet: Use mainnet (``True``, the default) or testnet
            (``False``).
        api_base_url: Override the API base URL.  Useful for testing or
            pointing at a custom proxy.
    """

    def __init__(
        self,
        is_mainnet: bool = True,
        api_base_url: str | None = None,
    ) -> None:
        self.is_mainnet = is_mainnet
        self._api_base = (api_base_url or (_MAINNET_API if is_mainnet else _TESTNET_API)).rstrip("/")

    # -----------------------------------------------------------------------
    # HyperEVM helpers
    # -----------------------------------------------------------------------

    @property
    def evm_rpc_url(self) -> str:
        """Return the HyperEVM JSON-RPC URL for this network."""
        return _HYPER_EVM_MAINNET_RPC if self.is_mainnet else _HYPER_EVM_TESTNET_RPC

    def make_evm_w3(self) -> AsyncWeb3:
        """Return an :class:`~web3.AsyncWeb3` instance connected to HyperEVM.

        The returned client is connected to the HyperEVM JSON-RPC endpoint for
        the configured network (mainnet chain ID 999, testnet 998).

        Example::

            client = HyperliquidClient()
            w3 = client.make_evm_w3()
            block = await w3.eth.get_block("latest")
        """
        return AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.evm_rpc_url))

    # -----------------------------------------------------------------------
    # Low-level API primitives
    # -----------------------------------------------------------------------

    async def post_info(self, payload: dict[str, Any]) -> Any:
        """Send a read-only query to ``POST /info``.

        Args:
            payload: JSON body, e.g. ``{"type": "meta"}`` or
                ``{"type": "userState", "user": "0x..."}``.

        Returns:
            Parsed JSON response.

        Raises:
            :exc:`aiohttp.ClientResponseError`: On non-2xx HTTP responses.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._api_base}/info",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    async def post_action(
        self,
        action: dict[str, Any],
        private_key: str,
        nonce: int | None = None,
        vault_address: str | None = None,
        expires_after: int | None = None,
    ) -> Any:
        """Sign and submit a trading action to ``POST /exchange``.

        Computes the phantom-agent signature and posts the signed request.
        Use this for trading actions (``order``, ``cancel``, ``modify``, etc.).
        For transfer/withdrawal actions that require user-signed EIP-712, use
        the dedicated helpers (:meth:`usd_send`, :meth:`spot_send`, etc.).

        Args:
            action: The L1 action dict (e.g. ``{"type": "order", ...}``).
            private_key: Hex-encoded private key for signing.
            nonce: Millisecond timestamp nonce.  Defaults to current time.
            vault_address: Address of the vault / sub-account to act on
                behalf of, or ``None`` for the master account.
            expires_after: Optional expiry timestamp in milliseconds.

        Returns:
            Parsed JSON response from the exchange API.

        Raises:
            :exc:`aiohttp.ClientResponseError`: On non-2xx HTTP responses.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        signature = sign_l1_action(
            private_key=private_key,
            action=action,
            nonce=nonce,
            vault_address=vault_address,
            expires_after=expires_after,
            is_mainnet=self.is_mainnet,
        )

        request_body: dict[str, Any] = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }
        if vault_address is not None:
            request_body["vaultAddress"] = vault_address

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._api_base}/exchange",
                json=request_body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    async def _post_user_signed_exchange(
        self,
        request_body: dict[str, Any],
    ) -> Any:
        """Post a pre-built user-signed request to ``/exchange``."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._api_base}/exchange",
                json=request_body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    # -----------------------------------------------------------------------
    # Info API convenience methods
    # -----------------------------------------------------------------------

    async def get_meta(self) -> dict[str, Any]:
        """Retrieve perps exchange metadata (universe list with coin names)."""
        return await self.post_info({"type": "meta"})

    async def get_meta_and_asset_ctxs(self) -> list[Any]:
        """Retrieve perps metadata together with live asset contexts."""
        return await self.post_info({"type": "metaAndAssetCtxs"})

    async def get_spot_meta(self) -> dict[str, Any]:
        """Retrieve spot exchange metadata (tokens and trading pairs)."""
        return await self.post_info({"type": "spotMeta"})

    async def get_spot_meta_and_asset_ctxs(self) -> list[Any]:
        """Retrieve spot metadata together with live asset contexts."""
        return await self.post_info({"type": "spotMetaAndAssetCtxs"})

    async def get_all_mids(self) -> dict[str, str]:
        """Retrieve mid prices for all actively-traded perps coins."""
        return await self.post_info({"type": "allMids"})

    async def get_user_state(self, user: str) -> dict[str, Any]:
        """Retrieve a user's perps clearinghouse state (positions, balances).

        Args:
            user: EVM address (42-character hex string).
        """
        return await self.post_info({"type": "clearinghouseState", "user": user})

    async def get_spot_clearinghouse_state(self, user: str) -> dict[str, Any]:
        """Retrieve a user's spot clearinghouse state (token balances).

        Args:
            user: EVM address (42-character hex string).
        """
        return await self.post_info({"type": "spotClearinghouseState", "user": user})

    async def get_open_orders(self, user: str) -> list[dict[str, Any]]:
        """Retrieve a user's open perps orders.

        Args:
            user: EVM address.
        """
        return await self.post_info({"type": "openOrders", "user": user})

    async def get_user_fills(self, user: str) -> list[dict[str, Any]]:
        """Retrieve a user's fill history.

        Args:
            user: EVM address.
        """
        return await self.post_info({"type": "userFills", "user": user})

    async def get_user_funding(self, user: str, start_time: int, end_time: int | None = None) -> list[dict[str, Any]]:
        """Retrieve a user's funding payment history.

        Args:
            user: EVM address.
            start_time: Unix timestamp in milliseconds.
            end_time: Optional end timestamp in milliseconds.
        """
        payload: dict[str, Any] = {"type": "userFunding", "user": user, "startTime": start_time}
        if end_time is not None:
            payload["endTime"] = end_time
        return await self.post_info(payload)

    async def get_l2_book(self, coin: str, n_sig_figs: int | None = None) -> dict[str, Any]:
        """Retrieve the L2 order book for a coin.

        Args:
            coin: Coin symbol (e.g. ``"BTC"``).
            n_sig_figs: Optional number of significant figures for price
                aggregation.
        """
        payload: dict[str, Any] = {"type": "l2Book", "coin": coin}
        if n_sig_figs is not None:
            payload["nSigFigs"] = n_sig_figs
        return await self.post_info(payload)

    # -----------------------------------------------------------------------
    # Exchange action convenience methods (user-signed)
    # -----------------------------------------------------------------------

    async def usd_send(
        self,
        private_key: str,
        destination: str,
        amount: str,
        nonce: int | None = None,
    ) -> Any:
        """Send USDC between Hyperliquid L1 accounts (Core USDC transfer).

        This transfer does not touch the EVM bridge.

        Args:
            private_key: Sender's hex-encoded private key.
            destination: Recipient EVM address.
            amount: Amount as a decimal string (e.g. ``"100.0"``).
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "usdSend",
            "destination": destination,
            "amount": amount,
            "time": nonce,
        }
        signature = sign_usd_transfer_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

    async def spot_send(
        self,
        private_key: str,
        destination: str,
        token: str,
        amount: str,
        nonce: int | None = None,
    ) -> Any:
        """Send spot tokens between Hyperliquid L1 accounts.

        This transfer does not touch the EVM bridge.  To bridge tokens
        between HyperCore and HyperEVM use :meth:`send_asset` instead.

        Args:
            private_key: Sender's hex-encoded private key.
            destination: Recipient EVM address.
            token: Token identifier (e.g. ``"USDC:0x..."`` or ``"PURR:0x..."``).
            amount: Amount as a decimal string.
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "spotSend",
            "destination": destination,
            "token": token,
            "amount": amount,
            "time": nonce,
        }
        signature = sign_spot_transfer_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

    async def withdraw(
        self,
        private_key: str,
        destination: str,
        amount: str,
        nonce: int | None = None,
    ) -> Any:
        """Initiate a withdrawal from Hyperliquid L1 to an EVM address.

        After this call the L1 validators sign and relay the withdrawal to the
        EVM bridge contract.  Expect approximately 5 minutes to finalise.

        Args:
            private_key: Sender's hex-encoded private key.
            destination: Destination EVM address to receive USDC.
            amount: Amount of USDC as a decimal string (e.g. ``"100.0"``).
                A $1 fee is deducted by the protocol.
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "withdraw3",
            "destination": destination,
            "amount": amount,
            "time": nonce,
        }
        signature = sign_withdraw_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

    async def usd_class_transfer(
        self,
        private_key: str,
        amount: str,
        to_perp: bool,
        nonce: int | None = None,
    ) -> Any:
        """Transfer USDC between the spot and perp wallets.

        Args:
            private_key: Sender's hex-encoded private key.
            amount: Amount of USDC as a decimal string.
            to_perp: ``True`` to move funds from spot → perp, ``False`` for
                perp → spot.
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "usdClassTransfer",
            "amount": amount,
            "toPerp": to_perp,
            "nonce": nonce,
        }
        signature = sign_usd_class_transfer_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

    async def send_asset(
        self,
        private_key: str,
        destination: str,
        token: str,
        amount: str,
        source_dex: str = "",
        destination_dex: str = "",
        from_sub_account: str = "",
        nonce: int | None = None,
    ) -> Any:
        """Transfer tokens between perp DEXes, spot balances, users, or sub-accounts.

        To bridge USDC from HyperCore to HyperEVM, set *destination* to the
        USDC system address ``0x2000000000000000000000000000000000000000``
        and *destination_dex* to ``"spot"``.

        Args:
            private_key: Sender's hex-encoded private key.
            destination: Recipient address, or the USDC system address
                ``0x2000000000000000000000000000000000000000`` to bridge to
                HyperEVM.
            token: Token identifier (e.g. ``"USDC"``).
            amount: Amount as a decimal string.
            source_dex: Source DEX name.  Use ``""`` for the perp balance or
                ``"spot"`` for the spot balance.
            destination_dex: Destination DEX name.  Use ``""`` for the perp
                balance or ``"spot"`` for the spot balance.  When bridging to
                HyperEVM use ``"spot"`` so USDC lands in the EVM wallet.
            from_sub_account: Sub-account address to transfer from, or ``""``
                for the master account.
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "sendAsset",
            "destination": destination,
            "sourceDex": source_dex,
            "destinationDex": destination_dex,
            "token": token,
            "amount": amount,
            "fromSubAccount": from_sub_account,
            "nonce": nonce,
        }
        signature = sign_send_asset_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

    async def send_to_evm_with_data(
        self,
        private_key: str,
        destination_recipient: str,
        amount: str,
        destination_chain_id: int,
        signature_chain_id: str,
        token: str = "USDC",
        source_dex: str = "",
        address_encoding: str = "hex",
        gas_limit: int = 200_000,
        data: str = "0x",
        nonce: int | None = None,
    ) -> Any:
        """Withdraw USDC from HyperCore to an EVM chain via CCTP.

        Initiates a CCTP-backed withdrawal from a HyperCore account to a
        recipient address on a supported EVM chain.  Circle's CCTP relayer
        mints USDC on the destination chain after the withdrawal is processed.

        The ``signatureChainId`` is the **destination** chain's EVM chain ID
        (e.g. ``"0xa4b1"`` for Arbitrum mainnet, ``"0x1"`` for Ethereum), not
        the fixed Arbitrum Sepolia ID used by other user-signed actions.

        Args:
            private_key: Sender's hex-encoded private key.
            destination_recipient: Recipient EVM address on the destination
                chain (e.g. ``"0x1234..."``) as a hex string.
            amount: Amount of USDC as a decimal string (e.g. ``"10"`` for
                10 USDC).
            destination_chain_id: CCTP domain ID of the destination chain
                (e.g. ``3`` for Arbitrum, ``0`` for Ethereum).  See
                https://developers.circle.com/cctp/concepts/supported-chains-and-domains
            signature_chain_id: EVM chain ID of the destination chain as a hex
                string (e.g. ``"0xa4b1"`` for Arbitrum).  Used for EIP-712
                replay protection.
            token: Token to withdraw.  Defaults to ``"USDC"``.
            source_dex: Source balance.  Use ``""`` (default) to withdraw from
                the perp balance, or ``"spot"`` for the spot balance.
            address_encoding: Address encoding format.  Defaults to ``"hex"``.
            gas_limit: Gas limit for the destination chain mint transaction.
                Defaults to 200,000.
            data: Optional calldata to pass to the destination contract.
                Defaults to ``"0x"`` (no calldata, direct ERC-20 transfer).
            nonce: Millisecond timestamp nonce.  Defaults to current time.

        Returns:
            API response dict.

        Example::

            # Withdraw 10 USDC from HyperCore perp to Arbitrum mainnet
            await client.send_to_evm_with_data(
                private_key=pk,
                destination_recipient="0xYourAddress",
                amount="10",
                destination_chain_id=3,        # Arbitrum CCTP domain
                signature_chain_id="0xa4b1",   # Arbitrum EVM chain ID
            )
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        action: dict[str, Any] = {
            "type": "sendToEvmWithData",
            "signatureChainId": signature_chain_id,
            "token": token,
            "amount": amount,
            "sourceDex": source_dex,
            "destinationRecipient": destination_recipient,
            "addressEncoding": address_encoding,
            "destinationChainId": destination_chain_id,
            "gasLimit": gas_limit,
            "data": data,
            "nonce": nonce,
        }
        signature = sign_send_to_evm_with_data_action(private_key, action, self.is_mainnet)
        return await self._post_user_signed_exchange({"action": action, "nonce": nonce, "signature": signature})

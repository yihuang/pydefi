"""
Circle Cross-Chain Transfer Protocol v2 (CCTP v2) USDC bridge integration.

CCTP v2 is Circle's upgraded protocol for burning USDC on one chain and
minting the equivalent amount on another chain via Circle's attestation
service.  Unlike CCTP v1, v2 introduces:

* **Fast Transfer** — two finality tiers (1000 = ~8–20 s, 2000 = ~15–19 min)
* **Relayer fees** — a small fee deducted by the attestation relayer
* **Native hooks** — ``depositForBurnWithHook`` embeds arbitrary calldata
  (the DeFiVM program) directly in the attested burn message

High-level flow
---------------
1. **Source chain**: approve ``TokenMessengerV2`` to spend USDC, then call
   ``depositForBurn`` or ``depositForBurnWithHook``.
2. **Attestation**: poll Circle's Iris v2 API until the burn is attested
   (~8–20 seconds for fast-finality transfers, or ~15–19 minutes for standard).
3. **Destination chain**: call ``MessageTransmitterV2.receiveMessage`` with
   the signed attestation to mint USDC (less any relayer fee) to the
   designated recipient.

Compose flow (with CCTPComposer)
---------------------------------
For destination-chain execution, embed the DeFiVM program as ``hookData``
when calling ``depositForBurnWithHook``.  The CCTPComposer contract on the
destination chain will:

1. Receive the CCTP attestation and call ``receiveMessage`` to mint USDC.
2. Extract the DeFiVM program from the ``hookData`` field in the burn message.
3. Forward the minted USDC to DeFiVM and execute the program.

Since the DeFiVM program is embedded in the CCTP message, it is *committed
on-chain at burn time* and cannot be altered after submission.  Set
``destinationCaller = composer_address`` so only the CCTPComposer can relay
the ``receiveMessage`` call — this prevents third parties from front-running
the mint.

Docs: https://developers.circle.com/stablecoins/cctp-getting-started
"""

from __future__ import annotations

from typing import Any

import aiohttp
from eth_contract import Contract
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

# ---------------------------------------------------------------------------
# Fast-finality threshold constant (CCTP v2)
# ---------------------------------------------------------------------------

# Use FINALITY_THRESHOLD_CONFIRMED (1000) for "fast transfer" (~8–20 seconds).
# Use FINALITY_THRESHOLD_FINALIZED (2000) for standard (~15–19 min on Ethereum).
FINALITY_THRESHOLD_CONFIRMED: int = 1000
FINALITY_THRESHOLD_FINALIZED: int = 2000

# ---------------------------------------------------------------------------
# Circle Iris v2 API
# ---------------------------------------------------------------------------

_IRIS_API_BASE = "https://iris-api.circle.com"

# ---------------------------------------------------------------------------
# ABI fragments — TokenMessengerV2
# ---------------------------------------------------------------------------

_TOKEN_MESSENGER_V2_ABI = [
    # depositForBurn — standard transfer (no compose hook)
    "function depositForBurn(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold) external",
    # depositForBurnWithHook — compose transfer; DeFiVM program passed as hookData
    "function depositForBurnWithHook(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller, uint256 maxFee, uint32 minFinalityThreshold, bytes calldata hookData) external",
]

# ---------------------------------------------------------------------------
# Well-known CCTP v2 contract addresses
# ---------------------------------------------------------------------------

# Circle CCTP domain IDs (shared between v1 and v2, unchanged).
# https://developers.circle.com/stablecoins/supported-domains
_CCTP_DOMAIN: dict[int, int] = {
    1: 0,  # Ethereum
    43114: 1,  # Avalanche
    10: 2,  # OP Mainnet
    42161: 3,  # Arbitrum
    8453: 6,  # Base
    137: 7,  # Polygon PoS
    130: 10,  # Unichain
    59144: 11,  # Linea
}

# CCTP v2 TokenMessengerV2 addresses.
# CCTP v2 is deployed at deterministic CREATE2 addresses — the same address
# on every supported EVM chain.
# https://developers.circle.com/stablecoins/evm-smart-contracts
_TOKEN_MESSENGER_V2: dict[int, str] = {
    1: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Ethereum
    43114: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Avalanche
    10: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # OP Mainnet
    42161: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Arbitrum
    8453: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Base
    137: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Polygon PoS
    130: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Unichain
    59144: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf00",  # Linea
}

# CCTP v2 MessageTransmitterV2 addresses (same address on all supported chains).
_MESSAGE_TRANSMITTER_V2: dict[int, str] = {
    1: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Ethereum
    43114: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Avalanche
    10: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # OP Mainnet
    42161: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Arbitrum
    8453: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Base
    137: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Polygon PoS
    130: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Unichain
    59144: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # Linea
}

# Native USDC addresses per chain (Circle-issued, unchanged from v1).
_USDC: dict[int, str] = {
    1: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # Ethereum
    43114: "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # Avalanche
    10: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # OP Mainnet
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # Arbitrum
    8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base
    137: "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # Polygon PoS
    130: "0x078D888E40faAe0f32594342c85940AF3949E666",  # Unichain
    59144: "0x176211869cA2b568f2A7D4EE941E073a821EE1ff",  # Linea
}


class CCTP(BaseBridge):
    """Circle CCTP v2 cross-chain USDC bridge integration.

    CCTP v2 burns USDC on the source chain and mints it on the destination chain
    via Circle's off-chain attestation service.  Transfers are always 1:1 minus
    a small relayer fee; only gas is paid on each chain.

    For an end-to-end compose flow (execute a DeFiVM program after minting)
    use :meth:`build_bridge_compose_tx` on the source chain — it encodes
    ``depositForBurnWithHook`` with the DeFiVM program embedded as ``hookData``.
    After Circle attests the burn, call
    ``CCTPComposer.receiveAndExecute(message, attestation)`` on the destination
    chain to mint USDC and execute the program in one atomic call.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the source chain.
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        token_messenger_address: Address of the ``TokenMessengerV2`` contract on
            the source chain.  Defaults to the well-known CCTP v2 address for
            ``src_chain_id`` when omitted.
        src_usdc_address: USDC token address on the source chain.  Defaults to
            the well-known native USDC address for ``src_chain_id``.
        api_base_url: Override the Circle Iris v2 API base URL.  Defaults to
            ``https://iris-api.circle.com``.  Use
            ``https://iris-api-sandbox.circle.com`` for testnet transfers.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        token_messenger_address: str | None = None,
        src_usdc_address: str | None = None,
        api_base_url: str = _IRIS_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3
        self._api_base = api_base_url.rstrip("/")

        self.token_messenger_address = token_messenger_address or _TOKEN_MESSENGER_V2.get(src_chain_id, "")
        if not self.token_messenger_address:
            raise BridgeError(
                f"CCTP: no TokenMessengerV2 address known for chain {src_chain_id}. "
                "Pass token_messenger_address explicitly."
            )

        self.src_usdc_address = src_usdc_address or _USDC.get(src_chain_id, "")
        if not self.src_usdc_address:
            raise BridgeError(
                f"CCTP: no USDC address known for chain {src_chain_id}. Pass src_usdc_address explicitly."
            )

        self._token_messenger = Contract.from_abi(_TOKEN_MESSENGER_V2_ABI, to=self.token_messenger_address)

    @property
    def protocol_name(self) -> str:
        return "CCTP"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _cctp_domain(self, chain_id: int) -> int:
        """Return the CCTP domain ID for *chain_id*."""
        domain = _CCTP_DOMAIN.get(chain_id)
        if domain is None:
            raise BridgeError(f"CCTP: unsupported chain ID {chain_id}. Provide domain ID explicitly.")
        return domain

    @staticmethod
    def _address_to_bytes32(address: str) -> bytes:
        """Left-pad an EVM address to 32 bytes."""
        return HexBytes(address).rjust(32, b"\x00")

    # -----------------------------------------------------------------------
    # Circle Iris v2 API
    # -----------------------------------------------------------------------

    async def get_fees(self) -> list[dict[str, Any]]:
        """Query the Circle Iris v2 API for minimum transfer fees.

        Calls ``GET /v2/burn/USDC/fees/{srcDomain}/{dstDomain}`` and returns
        the raw list of ``{ finalityThreshold, minimumFee }`` objects — one
        entry per supported finality tier.

        Returns:
            List of fee objects from the Iris API.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On HTTP or API error.
        """
        src_domain = self._cctp_domain(self.src_chain_id)
        dst_domain = self._cctp_domain(self.dst_chain_id)
        url = f"{self._api_base}/v2/burn/USDC/fees/{src_domain}/{dst_domain}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise BridgeError(f"CCTP Iris API error ({resp.status}): {text}")
                data = await resp.json(content_type=None)
                return data  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # BaseBridge interface
    # -----------------------------------------------------------------------

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a CCTP v2 bridge quote by querying the Circle Iris v2 fees API.

        Calls ``GET /v2/burn/USDC/fees/{srcDomain}/{dstDomain}`` to retrieve
        the minimum relayer fee for the requested finality tier, then computes
        ``amount_out = amount_in - minimum_fee``.

        Args:
            token_in: Source chain USDC token.
            token_out: Destination chain USDC token.
            amount_in: Amount to bridge.
            min_finality_threshold: Finality tier to price the quote for.
                Use :data:`FINALITY_THRESHOLD_CONFIRMED` (1000, ~8–20 s) for
                fast transfers, or :data:`FINALITY_THRESHOLD_FINALIZED` (2000)
                for standard finality.

        Returns:
            A :class:`~pydefi.types.BridgeQuote` with the fee and estimated
            time filled in from the Iris API response.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error or if no fee
                entry is found for the requested finality threshold.
        """
        fee_entries = await self.get_fees()

        # Find the entry matching the requested finality tier.
        entry = next((e for e in fee_entries if e.get("finalityThreshold") == min_finality_threshold), None)
        if entry is None:
            raise BridgeError(
                f"CCTP: no fee entry found for finalityThreshold={min_finality_threshold}. "
                f"Available thresholds: {[e.get('finalityThreshold') for e in fee_entries]}"
            )

        min_fee = int(entry.get("minimumFee", 0))
        amount_out_raw = max(0, amount_in.amount - min_fee)
        # ~20 s for fast/confirmed finality (threshold ≤ 1000);
        # ~19 minutes (1140 s) for standard finalized transfers.
        estimated_time = 20 if min_finality_threshold <= FINALITY_THRESHOLD_CONFIRMED else 1140

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=min_fee),
            estimated_time_seconds=estimated_time,
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        slippage_bps: int = 0,
        dst_domain: int | None = None,
        destination_caller: str | None = None,
        max_fee: int = 0,
        min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a CCTP v2 ``depositForBurn`` transaction.

        Burns USDC on the source chain.  After Circle attests the burn event,
        the recipient (or anyone) must submit ``receiveMessage`` on the
        destination chain to mint the USDC (minus relayer fee).

        Args:
            token_in: Source chain USDC token.
            token_out: Destination chain USDC token (used for validation only).
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.
            slippage_bps: Ignored for CCTP (accepted for API compatibility).
            dst_domain: Override the CCTP destination domain ID.
            destination_caller: Restrict who may call ``receiveMessage`` on the
                destination chain (zero-bytes = no restriction).  Defaults to
                no restriction.
            max_fee: Maximum relayer fee in USDC units (must be < amount and
                >= minimum fee if the chain has a non-zero minimum fee).
                Defaults to 0 (valid on chains where minFee == 0).
            min_finality_threshold: Minimum attestation finality.  Use
                :data:`FINALITY_THRESHOLD_CONFIRMED` (1000, ~8–20 s) for fast
                transfers or :data:`FINALITY_THRESHOLD_FINALIZED` (2000,
                ~15–19 min) for maximum security.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Note:
            The caller must separately ``approve`` the ``TokenMessengerV2``
            contract to spend ``amount_in.amount`` of USDC before submitting
            this transaction.
        """
        _dst_domain = dst_domain if dst_domain is not None else self._cctp_domain(self.dst_chain_id)
        mint_recipient = self._address_to_bytes32(recipient)
        dst_caller_bytes = self._address_to_bytes32(destination_caller) if destination_caller else b"\x00" * 32

        call_data = self._token_messenger.fns.depositForBurn(
            amount_in.amount,
            _dst_domain,
            mint_recipient,
            Web3.to_checksum_address(self.src_usdc_address),
            dst_caller_bytes,
            max_fee,
            min_finality_threshold,
        ).data

        return {
            "to": self.token_messenger_address,
            "data": "0x" + call_data.hex() if isinstance(call_data, bytes) else call_data,
            "value": "0",
            "gas": str(200_000),
        }

    async def build_bridge_compose_tx(
        self,
        amount_in: TokenAmount,
        composer_address: str,
        program: bytes,
        dst_domain: int | None = None,
        max_fee: int = 0,
        min_finality_threshold: int = FINALITY_THRESHOLD_CONFIRMED,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a CCTP v2 ``depositForBurnWithHook`` transaction for compose.

        Embeds the DeFiVM *program* as ``hookData`` in the CCTP v2 burn message
        and sets ``mintRecipient = composer_address`` so that USDC is minted
        directly into the :class:`~pydefi.bridge.CCTPComposer` contract on the
        destination chain.  ``destinationCaller`` is set to *composer_address*
        so that only the CCTPComposer can invoke ``receiveMessage``, preventing
        third parties from front-running the mint.

        The DeFiVM program is *committed on-chain at burn time*:  it is part of
        the attested CCTP message and cannot be changed after the burn is
        submitted.

        Args:
            amount_in: Amount of USDC to bridge.
            composer_address: Address of the :class:`CCTPComposer` contract on
                the destination chain.  Used as both ``mintRecipient`` and
                ``destinationCaller``.
            program: Raw DeFiVM bytecode to execute on the destination chain
                after USDC is minted.  Passed as ``hookData``.
            dst_domain: Override the CCTP destination domain ID.
            max_fee: Maximum relayer fee in USDC units.  Defaults to 0 (valid
                on chains where minFee == 0; query Iris for the exact minimum).
            min_finality_threshold: Minimum attestation finality level.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Note:
            The caller must separately ``approve`` the ``TokenMessengerV2``
            contract to spend ``amount_in.amount`` of USDC before submitting
            this transaction.
        """
        if not program:
            raise BridgeError("CCTP: program (hookData) must not be empty for compose transactions.")

        _dst_domain = dst_domain if dst_domain is not None else self._cctp_domain(self.dst_chain_id)
        mint_recipient = self._address_to_bytes32(composer_address)
        destination_caller = self._address_to_bytes32(composer_address)

        call_data = self._token_messenger.fns.depositForBurnWithHook(
            amount_in.amount,
            _dst_domain,
            mint_recipient,
            Web3.to_checksum_address(self.src_usdc_address),
            destination_caller,
            max_fee,
            min_finality_threshold,
            program,
        ).data

        return {
            "to": self.token_messenger_address,
            "data": "0x" + call_data.hex() if isinstance(call_data, bytes) else call_data,
            "value": "0",
            "gas": str(220_000),
        }

    # -----------------------------------------------------------------------
    # Class-level helpers for deployment lookups
    # -----------------------------------------------------------------------

    @classmethod
    def message_transmitter_address(cls, chain_id: int) -> str:
        """Return the well-known CCTP v2 ``MessageTransmitterV2`` address for *chain_id*.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If no address is known.
        """
        addr = _MESSAGE_TRANSMITTER_V2.get(chain_id)
        if addr is None:
            raise BridgeError(f"CCTP: no MessageTransmitterV2 address known for chain {chain_id}")
        return addr

    @classmethod
    def usdc_address(cls, chain_id: int) -> str:
        """Return the well-known native USDC address for *chain_id*.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If no address is known.
        """
        addr = _USDC.get(chain_id)
        if addr is None:
            raise BridgeError(f"CCTP: no USDC address known for chain {chain_id}")
        return addr

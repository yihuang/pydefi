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

Bridging to HyperCore (Hyperliquid L1)
---------------------------------------
To bridge USDC directly to a HyperCore address from an external EVM chain,
use ``dst_chain_id=ChainId.HYPERCORE``.  The bridge uses a two-step process:

1. USDC is burned on the source chain via ``depositForBurnWithHook``, with the
   ``mintRecipient`` and ``destinationCaller`` set to the ``CctpForwarder``
   contract address on HyperEVM.
2. The ``CctpForwarder`` receives the minted USDC on HyperEVM, reads the
   ``hookData``, and deposits the USDC into HyperCore for the designated
   recipient.

The ``hookData`` is encoded via :func:`encode_cctp_forward_hook_data` and
specifies the HyperCore recipient address and destination DEX (perp or spot).
A small forwarding fee (0.20 USDC) is charged by the ``CctpForwarder``.

By default, deposits credit the **perp** balance on HyperCore.  Pass
``hypercore_dex=HYPERCORE_DEX_SPOT`` to deposit into the spot balance.

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

Docs: https://developers.circle.com/cctp/concepts/cctp-on-hypercore
"""

from __future__ import annotations

from typing import Any

import aiohttp
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.abi.bridge import CCTP_TOKEN_MESSENGER_V2
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, ChainId, Token, TokenAmount

# ---------------------------------------------------------------------------
# Fast-finality threshold constant (CCTP v2)
# ---------------------------------------------------------------------------

# Use FINALITY_THRESHOLD_CONFIRMED (1000) for "fast transfer" (~8–20 seconds).
# Use FINALITY_THRESHOLD_FINALIZED (2000) for standard (~15–19 min on Ethereum).
FINALITY_THRESHOLD_CONFIRMED: int = 1000
FINALITY_THRESHOLD_FINALIZED: int = 2000

# ---------------------------------------------------------------------------
# HyperCore-specific constants
# ---------------------------------------------------------------------------

# HyperCore destination DEX values for the CctpForwarder hookData.
# These values are defined by the CctpForwarder contract on HyperEVM:
#   0           → perp balance (default)
#   0xFFFFFFFF  → spot balance (uint32 max value per the CctpForwarder spec)
# Docs: https://developers.circle.com/cctp/howtos/transfer-usdc-from-solana-to-hypercore
HYPERCORE_DEX_PERP: int = 0
HYPERCORE_DEX_SPOT: int = 0xFFFFFFFF  # uint32 max — spot balance on HyperCore

# ---------------------------------------------------------------------------
# Circle Iris v2 API
# ---------------------------------------------------------------------------

_IRIS_API_BASE = "https://iris-api.circle.com"

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
    999: 19,  # HyperEVM (Hyperliquid)
    # HyperCore is Hyperliquid's L1; CCTP physically mints on HyperEVM (domain 19)
    # and Hyperliquid routes funds to HyperCore automatically.
    1337: 19,  # HyperCore (Hyperliquid L1) — routes via HyperEVM
}

# CCTP v2 TokenMessengerV2 addresses.
# CCTP v2 is deployed at deterministic CREATE2 addresses — the same address
# on every supported EVM chain.
# https://developers.circle.com/stablecoins/evm-smart-contracts
_TOKEN_MESSENGER_V2: dict[int, str] = {
    1: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Ethereum
    43114: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Avalanche
    10: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # OP Mainnet
    42161: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Arbitrum
    8453: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Base
    137: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Polygon PoS
    130: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Unichain
    59144: "0x28B5a0E9c621a5BAdaa536219b3a228c8168cF00",  # Linea
    999: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",  # HyperEVM (Hyperliquid)
    1337: "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",  # HyperCore (same contract on HyperEVM)
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
    999: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # HyperEVM (Hyperliquid)
    1337: "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64",  # HyperCore (same contract on HyperEVM)
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
    999: "0xb88339CB7199b77E23DB6E890353E22632Ba630f",  # HyperEVM (Hyperliquid)
    1337: "0xb88339CB7199b77E23DB6E890353E22632Ba630f",  # HyperCore (minted on HyperEVM)
}

# CctpForwarder contract addresses on HyperEVM (keyed by is_mainnet: bool).
# The CctpForwarder receives USDC minted by CCTP on HyperEVM and forwards it
# to the designated recipient on HyperCore (Hyperliquid L1) using hookData.
# Used as both mintRecipient and destinationCaller for HyperCore-bound transfers.
# Mainnet address verified on-chain (tokenMessenger() == TokenMessengerV2).
# Docs: https://developers.circle.com/cctp/concepts/cctp-on-hypercore
_CCTP_FORWARDER: dict[bool, str] = {
    True: "0xb21d281dedb17ae5b501f6aa8256fe38c4e45757",  # HyperEVM mainnet (verified)
    False: "0x02e39ECb8368b41bF68FF99ff351aC9864e5E2a2",  # HyperEVM testnet
}


def encode_cctp_forward_hook_data(
    recipient: str | None = None,
    destination_dex: int = HYPERCORE_DEX_PERP,
) -> bytes:
    """Encode ``hookData`` for the HyperEVM ``CctpForwarder`` contract.

    The ``CctpForwarder`` on HyperEVM reads this hook data when it receives
    minted USDC via CCTP and uses it to forward the USDC to the correct
    address on HyperCore.

    Hook data format::

        Field                  Bytes   Type      Description
        ─────────────────────  ──────  ────────  ───────────────────────────
        magicBytes             24      bytes24   ASCII "cctp-forward" + padding
        version                4       uint32    Always 0
        dataLength             4       uint32    0 (no recipient) or 24 (with)
        recipient              20      address   HyperCore recipient (optional)
        destinationDex         4       uint32    DEX: 0=perp, 0xFFFFFFFF=spot

    Args:
        recipient: EVM-format HyperCore recipient address.  When ``None``
            the hook data contains only the header (no recipient), and the
            ``CctpForwarder`` will use the CCTP ``mintRecipient`` field as the
            HyperCore recipient.
        destination_dex: HyperCore destination DEX.  Use
            :data:`HYPERCORE_DEX_PERP` (0, the default) to credit the perp
            balance, or :data:`HYPERCORE_DEX_SPOT` (``0xFFFFFFFF``) for spot.

    Returns:
        Encoded ``hookData`` bytes.
    """
    # 24-byte magic: 12-char ASCII string "cctp-forward" padded to 24 bytes with nulls
    magic = b"cctp-forward" + b"\x00" * 12  # 12 ASCII + 12 padding = 24 bytes total
    version = (0).to_bytes(4, "big")

    if recipient is None:
        # Header only — no recipient data
        data_length = (0).to_bytes(4, "big")
        return magic + version + data_length

    # With recipient: 20 bytes address + 4 bytes dex = 24 bytes of data
    data_length = (24).to_bytes(4, "big")
    addr_bytes = bytes.fromhex(recipient[2:] if recipient.startswith("0x") else recipient)
    if len(addr_bytes) != 20:
        raise ValueError(f"recipient must be a 20-byte EVM address, got {len(addr_bytes)} bytes")
    dex_bytes = (destination_dex & 0xFFFFFFFF).to_bytes(4, "big")
    return magic + version + data_length + addr_bytes + dex_bytes


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
        dst_chain_id: Destination chain EVM ID.  Use :data:`~pydefi.types.ChainId.HYPERCORE`
            (1337) to bridge directly to the Hyperliquid L1.
        token_messenger_address: Address of the ``TokenMessengerV2`` contract on
            the source chain.  Defaults to the well-known CCTP v2 address for
            ``src_chain_id`` when omitted.
        src_usdc_address: USDC token address on the source chain.  Defaults to
            the well-known native USDC address for ``src_chain_id``.
        api_base_url: Override the Circle Iris v2 API base URL.  Defaults to
            ``https://iris-api.circle.com``.  Use
            ``https://iris-api-sandbox.circle.com`` for testnet transfers.
        cctp_forwarder_address: Address of the ``CctpForwarder`` contract on
            HyperEVM.  Only used when ``dst_chain_id=ChainId.HYPERCORE``.
            Defaults to the well-known mainnet address.  Pass the testnet
            address when using ``iris-api-sandbox.circle.com``.
        is_mainnet: When ``True`` (default), use the mainnet ``CctpForwarder``
            address on HyperEVM.  Set to ``False`` for testnet.  Ignored when
            *cctp_forwarder_address* is provided explicitly.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        token_messenger_address: str | None = None,
        src_usdc_address: str | None = None,
        api_base_url: str = _IRIS_API_BASE,
        cctp_forwarder_address: str | None = None,
        is_mainnet: bool = True,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3
        self._api_base = api_base_url.rstrip("/")
        self.is_mainnet = is_mainnet

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

        self.cctp_forwarder_address = cctp_forwarder_address or _CCTP_FORWARDER[is_mainnet]

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

    async def get_fees(self, hypercore_deposit: bool = False) -> list[dict[str, Any]]:
        """Query the Circle Iris v2 API for minimum transfer fees.

        Calls ``GET /v2/burn/USDC/fees/{srcDomain}/{dstDomain}`` and returns
        the raw list of fee objects — one entry per supported finality tier.

        When bridging to HyperCore (``dst_chain_id=ChainId.HYPERCORE``), the
        response also includes a ``forwardFee`` field representing the fixed
        forwarding fee charged by the ``CctpForwarder`` contract (0.20 USDC).
        The effective total fee per transfer is
        ``minimumFee + forwardFee["med"]`` in USDC sub-units.

        Args:
            hypercore_deposit: When ``True``, append
                ``?forward=true&hyperCoreDeposit=true`` query parameters to
                include the ``forwardFee`` in the response.  Automatically set
                to ``True`` when ``dst_chain_id=ChainId.HYPERCORE``.

        Returns:
            List of fee objects from the Iris API.  Each entry has at minimum
            ``{ finalityThreshold, minimumFee }``.  When *hypercore_deposit*
            is ``True``, each entry also includes
            ``{ forwardFee: { low, med, high } }``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On HTTP or API error.
        """
        src_domain = self._cctp_domain(self.src_chain_id)
        dst_domain = self._cctp_domain(self.dst_chain_id)
        url = f"{self._api_base}/v2/burn/USDC/fees/{src_domain}/{dst_domain}"
        is_hypercore = self.dst_chain_id == ChainId.HYPERCORE or hypercore_deposit
        if is_hypercore:
            url += "?forward=true&hyperCoreDeposit=true"
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
        ``amount_out = amount_in - total_fee``.

        When ``dst_chain_id=ChainId.HYPERCORE``, the total fee includes both
        the CCTP relayer fee (``minimumFee``) and the ``CctpForwarder``
        forwarding fee (``forwardFee["med"]``, typically 0.20 USDC).

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
        # For HyperCore, include the CctpForwarder forwarding fee.
        # The forwardFee is a fixed 0.20 USDC charged by the CctpForwarder contract.
        if self.dst_chain_id == ChainId.HYPERCORE and "forwardFee" in entry:
            forward_fee_info = entry["forwardFee"]
            forward_fee = int(forward_fee_info.get("med") or forward_fee_info.get("low") or 0)
            min_fee += forward_fee
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
        hypercore_dex: int = HYPERCORE_DEX_PERP,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a CCTP v2 burn transaction to bridge USDC to the destination chain.

        For most destination chains this calls ``depositForBurn`` on the source
        ``TokenMessengerV2`` contract.

        For ``dst_chain_id=ChainId.HYPERCORE`` the flow uses
        ``depositForBurnWithHook``: the USDC is minted on HyperEVM to the
        ``CctpForwarder`` contract which reads the encoded ``hookData`` and
        deposits the USDC into the recipient's HyperCore account.  See
        :func:`encode_cctp_forward_hook_data` for the hookData format.

        After Circle attests the burn event, the recipient (or anyone) must
        submit ``receiveMessage`` on the destination chain to mint the USDC
        (minus relayer fee).

        Args:
            token_in: Source chain USDC token.
            token_out: Destination chain USDC token (used for validation only).
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain (or on
                HyperCore when ``dst_chain_id=ChainId.HYPERCORE``).
            slippage_bps: Ignored for CCTP (accepted for API compatibility).
            dst_domain: Override the CCTP destination domain ID.
            destination_caller: Restrict who may call ``receiveMessage`` on the
                destination chain (zero-bytes = no restriction).  For HyperCore
                this is always set to the ``CctpForwarder`` address.
            max_fee: Maximum relayer fee in USDC units (must be < amount and
                >= minimum fee if the chain has a non-zero minimum fee).
                Defaults to 0 (valid on chains where minFee == 0).
            min_finality_threshold: Minimum attestation finality.  Use
                :data:`FINALITY_THRESHOLD_CONFIRMED` (1000, ~8–20 s) for fast
                transfers or :data:`FINALITY_THRESHOLD_FINALIZED` (2000,
                ~15–19 min) for maximum security.
            hypercore_dex: HyperCore destination DEX for the forwarded USDC.
                Use :data:`HYPERCORE_DEX_PERP` (0, default) to credit the perp
                balance, or :data:`HYPERCORE_DEX_SPOT` (``0xFFFFFFFF``) for the
                spot balance.  Only used when ``dst_chain_id=ChainId.HYPERCORE``.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Note:
            The caller must separately ``approve`` the ``TokenMessengerV2``
            contract to spend ``amount_in.amount`` of USDC before submitting
            this transaction.
        """
        _dst_domain = dst_domain if dst_domain is not None else self._cctp_domain(self.dst_chain_id)

        if self.dst_chain_id == ChainId.HYPERCORE:
            # For HyperCore: mint USDC on HyperEVM to the CctpForwarder, which
            # reads the hookData and deposits it into the recipient's HyperCore account.
            forwarder = self.cctp_forwarder_address
            mint_recipient = self._address_to_bytes32(forwarder)
            dst_caller_bytes = self._address_to_bytes32(forwarder)
            hook_data = encode_cctp_forward_hook_data(recipient, hypercore_dex)

            call_data = CCTP_TOKEN_MESSENGER_V2.fns.depositForBurnWithHook(
                amount_in.amount,
                _dst_domain,
                mint_recipient,
                Web3.to_checksum_address(self.src_usdc_address),
                dst_caller_bytes,
                max_fee,
                min_finality_threshold,
                hook_data,
            ).data
        else:
            mint_recipient = self._address_to_bytes32(recipient)
            dst_caller_bytes = self._address_to_bytes32(destination_caller) if destination_caller else b"\x00" * 32

            call_data = CCTP_TOKEN_MESSENGER_V2.fns.depositForBurn(
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

        call_data = CCTP_TOKEN_MESSENGER_V2.fns.depositForBurnWithHook(
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

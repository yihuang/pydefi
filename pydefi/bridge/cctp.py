"""
Circle Cross-Chain Transfer Protocol (CCTP) USDC bridge integration.

CCTP is Circle's native protocol for burning USDC on one chain and minting
the equivalent amount on another chain via Circle's attestation service.
Unlike wrapped-token bridges, CCTP burns the original USDC and mints freshly
issued USDC on the destination chain, so no liquidity pools are involved and
transfers are always 1:1 with zero protocol fee.

High-level flow
---------------
1. **Source chain**: approve ``TokenMessenger`` to spend USDC, then call
   ``depositForBurn`` (or ``depositForBurnWithCaller``).
2. **Attestation**: poll Circle's Iris API until the burn is attested
   (typically < 2 minutes for *fast* finality chains).
3. **Destination chain**: call ``MessageTransmitter.receiveMessage`` with the
   signed attestation to mint USDC to the designated recipient.

Compose flow (with CCTPComposer)
---------------------------------
Set ``mintRecipient`` to a deployed :class:`CCTPComposer` contract address.
After Circle attests the burn, call
``CCTPComposer.receiveAndExecute(message, attestation, program)``
on the destination chain.  The contract mints USDC to itself, then forwards
the tokens and a DeFiVM program to the DeFiVM contract for execution
(e.g. swap, lend, LP into a pool).

Use ``depositForBurnWithCaller`` with ``destinationCaller = composer_address``
to prevent front-running: the ``MessageTransmitter`` will only accept
``receiveMessage`` originating from the CCTPComposer, so no third party can
mint the USDC with a different program.

Docs: https://developers.circle.com/stablecoins/cctp-getting-started
"""

from __future__ import annotations

from typing import Any

from eth_contract import Contract
from hexbytes import HexBytes
from web3 import AsyncWeb3, Web3

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

# ---------------------------------------------------------------------------
# ABI fragments
# ---------------------------------------------------------------------------

_TOKEN_MESSENGER_ABI = [
    # depositForBurn(amount, destinationDomain, mintRecipient, burnToken)
    "function depositForBurn(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken) external returns (uint64 nonce)",
    # depositForBurnWithCaller — restricts who may call receiveMessage on dst
    "function depositForBurnWithCaller(uint256 amount, uint32 destinationDomain, bytes32 mintRecipient, address burnToken, bytes32 destinationCaller) external returns (uint64 nonce)",
]

# ---------------------------------------------------------------------------
# Well-known CCTP v1 contract addresses
# ---------------------------------------------------------------------------

# Circle CCTP domain IDs (differ from EVM chain IDs).
# https://developers.circle.com/stablecoins/supported-domains
_CCTP_DOMAIN: dict[int, int] = {
    1: 0,  # Ethereum
    43114: 1,  # Avalanche
    10: 2,  # OP Mainnet
    42161: 3,  # Arbitrum
    8453: 6,  # Base
    137: 7,  # Polygon PoS
}

# CCTP v1 TokenMessenger addresses (source-chain contract that burns USDC).
# https://developers.circle.com/stablecoins/evm-smart-contracts
_TOKEN_MESSENGER: dict[int, str] = {
    1: "0xBd3fa81B58Ba92a82136038B25aDec7066af3155",  # Ethereum
    43114: "0x6B25532e1060CE10cc3B0A99e5683b91BFDe6982",  # Avalanche
    10: "0x2B4069517957735bE00ceE0fadAE88a26365528f",  # OP Mainnet
    42161: "0x19330d10D9Cc8751218eaf51E8885D058642E08A",  # Arbitrum
    8453: "0x1682Ae6375C4E4A97e4B583BC394c861A46D8962",  # Base
    137: "0x9daF8c91AEFAE50b9c0E69629D3F6Ca40cA3B3FE",  # Polygon PoS
}

# CCTP v1 MessageTransmitter addresses (destination-chain contract that mints USDC).
_MESSAGE_TRANSMITTER: dict[int, str] = {
    1: "0x0a992d191DEeC32aFe36203Ad87D7d289a738F81",  # Ethereum
    43114: "0x8186359aF5F57FbB40c6b14A588d2A59C0C29880",  # Avalanche
    10: "0x4D41f22c5a0e5c74090899E5a8Fb597a8842b3e8",  # OP Mainnet
    42161: "0xC30362313FBBA5cf9163F0bb16a0e01f01A896ca",  # Arbitrum
    8453: "0xAD09780d193884d503182aD4588450C416D6F9D4",  # Base
    137: "0xF3be9355363857F3e001be68856A2f96b4C39Ba9",  # Polygon PoS
}

# Native USDC addresses per chain (Circle-issued, not bridged USDC.e).
_USDC: dict[int, str] = {
    1: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # Ethereum
    43114: "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # Avalanche
    10: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # OP Mainnet
    42161: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # Arbitrum
    8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base
    137: "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # Polygon PoS
}


class CCTP(BaseBridge):
    """Circle CCTP cross-chain USDC bridge integration.

    CCTP burns USDC on the source chain and mints it on the destination chain
    via Circle's off-chain attestation service.  Transfers are always 1:1 with
    no protocol fee; only gas is paid on each chain.

    For an end-to-end compose flow (execute a DeFiVM program after minting)
    use :meth:`build_bridge_compose_tx` on the source chain and then call
    ``CCTPComposer.receiveAndExecute`` on the destination chain once Circle has
    attested the burn.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the source chain.
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        token_messenger_address: Address of the ``TokenMessenger`` contract on
            the source chain.  Defaults to the well-known CCTP v1 address for
            ``src_chain_id`` when omitted.
        src_usdc_address: USDC token address on the source chain.  Defaults to
            the well-known native USDC address for ``src_chain_id``.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        token_messenger_address: str | None = None,
        src_usdc_address: str | None = None,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3

        # Resolve contract addresses, falling back to well-known defaults.
        self.token_messenger_address = token_messenger_address or _TOKEN_MESSENGER.get(src_chain_id, "")
        if not self.token_messenger_address:
            raise BridgeError(
                f"CCTP: no TokenMessenger address known for chain {src_chain_id}. "
                "Pass token_messenger_address explicitly."
            )

        self.src_usdc_address = src_usdc_address or _USDC.get(src_chain_id, "")
        if not self.src_usdc_address:
            raise BridgeError(
                f"CCTP: no USDC address known for chain {src_chain_id}. Pass src_usdc_address explicitly."
            )

        self._token_messenger = Contract.from_abi(_TOKEN_MESSENGER_ABI, to=self.token_messenger_address)

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
    # BaseBridge interface
    # -----------------------------------------------------------------------

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a CCTP bridge quote.

        CCTP transfers are 1:1 with no protocol fee.  The bridge fee is always
        zero in token terms; only gas costs are incurred on each chain.

        Args:
            token_in: Source chain USDC token.
            token_out: Destination chain USDC token (same asset, different chain).
            amount_in: Amount to bridge.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.
        """
        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_in.amount),
            bridge_fee=TokenAmount(token=token_in, amount=0),
            estimated_time_seconds=120,  # ~2 min typical attestation time
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a CCTP ``depositForBurn`` transaction.

        Burns USDC on the source chain.  After Circle attests the burn event,
        the recipient (or anyone) must submit ``receiveMessage`` on the
        destination chain to mint the USDC.

        Args:
            token_in: Source chain USDC token.
            token_out: Destination chain USDC token (used for validation only).
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.  This becomes
                the ``mintRecipient`` in the CCTP burn message (zero-padded to
                32 bytes).
            slippage_bps: Ignored for CCTP (always 1:1); accepted for API
                compatibility.
            dst_domain: Override the CCTP destination domain ID.  Defaults to
                the well-known domain for ``dst_chain_id``.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Note:
            The caller must separately ``approve`` the ``TokenMessenger``
            contract to spend ``amount_in.amount`` of USDC before submitting
            this transaction.
        """
        _dst_domain = dst_domain if dst_domain is not None else self._cctp_domain(self.dst_chain_id)
        mint_recipient = self._address_to_bytes32(recipient)

        call_data = self._token_messenger.fns.depositForBurn(
            amount_in.amount,
            _dst_domain,
            mint_recipient,
            Web3.to_checksum_address(self.src_usdc_address),
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
        dst_domain: int | None = None,
        restrict_caller: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a CCTP ``depositForBurn`` transaction targeting a CCTPComposer.

        This variant sets ``mintRecipient`` to *composer_address* so that USDC
        is minted directly into the :class:`~pydefi.bridge.CCTPComposer`
        contract on the destination chain.  After minting, the composer
        executes a user-supplied DeFiVM program.

        When *restrict_caller* is ``True`` (default), the transaction uses
        ``depositForBurnWithCaller`` with ``destinationCaller = composer_address``
        so that only the CCTPComposer contract can invoke ``receiveMessage``.
        This prevents third parties from front-running the mint with a different
        DeFiVM program.

        Args:
            amount_in: Amount of USDC to bridge.
            composer_address: Address of the :class:`CCTPComposer` contract on
                the destination chain.
            dst_domain: Override the CCTP destination domain ID.
            restrict_caller: If ``True``, use ``depositForBurnWithCaller`` so
                that only *composer_address* can call ``receiveMessage``.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Note:
            The caller must separately ``approve`` the ``TokenMessenger``
            contract to spend ``amount_in.amount`` of USDC before submitting
            this transaction.
        """
        _dst_domain = dst_domain if dst_domain is not None else self._cctp_domain(self.dst_chain_id)
        mint_recipient = self._address_to_bytes32(composer_address)

        if restrict_caller:
            destination_caller = self._address_to_bytes32(composer_address)
            call_data = self._token_messenger.fns.depositForBurnWithCaller(
                amount_in.amount,
                _dst_domain,
                mint_recipient,
                Web3.to_checksum_address(self.src_usdc_address),
                destination_caller,
            ).data
        else:
            call_data = self._token_messenger.fns.depositForBurn(
                amount_in.amount,
                _dst_domain,
                mint_recipient,
                Web3.to_checksum_address(self.src_usdc_address),
            ).data

        return {
            "to": self.token_messenger_address,
            "data": "0x" + call_data.hex() if isinstance(call_data, bytes) else call_data,
            "value": "0",
            "gas": str(200_000),
        }

    # -----------------------------------------------------------------------
    # Class-level helpers for deployment lookups
    # -----------------------------------------------------------------------

    @classmethod
    def message_transmitter_address(cls, chain_id: int) -> str:
        """Return the well-known CCTP v1 ``MessageTransmitter`` address for *chain_id*.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If no address is known.
        """
        addr = _MESSAGE_TRANSMITTER.get(chain_id)
        if addr is None:
            raise BridgeError(f"CCTP: no MessageTransmitter address known for chain {chain_id}")
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

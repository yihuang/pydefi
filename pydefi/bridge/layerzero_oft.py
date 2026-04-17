"""
LayerZero OFT (Omnichain Fungible Token) cross-chain bridge integration.

LayerZero OFT v2 is a token standard that enables fungible tokens to be
transferred natively across EVM chains using the LayerZero v2 messaging
protocol.  Each OFT-enabled token contract exposes ``quoteSend`` (fee
estimation) and ``send`` (cross-chain transfer) on every supported chain.

Docs: https://docs.layerzero.network/v2/developers/evm/oft/quickstart
"""

from __future__ import annotations

from typing import Any

from web3 import AsyncWeb3, Web3

from pydefi._utils import address_to_bytes32
from pydefi.abi.bridge import LAYERZERO_OFT, MessagingFee, OFTSendParam
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import Address, BridgeQuote, Token, TokenAmount

# LayerZero v2 endpoint IDs (EIDs) mapped from EVM chain IDs.
# See: https://docs.layerzero.network/v2/developers/evm/technical-reference/deployed-contracts
_LZ_EID: dict[int, int] = {
    1: 30101,  # Ethereum
    10: 30111,  # Optimism
    56: 30102,  # BNB Chain
    137: 30109,  # Polygon
    250: 30112,  # Fantom
    324: 30165,  # zkSync Era
    8453: 30184,  # Base
    42161: 30110,  # Arbitrum
    43114: 30106,  # Avalanche
    59144: 30183,  # Linea
    81457: 30243,  # Blast
    130: 30320,  # Unichain
    480: 30337,  # World Chain
    534352: 30214,  # Scroll
    7777777: 30195,  # Zora
}


class LayerZeroOFT(BaseBridge):
    """LayerZero OFT v2 cross-chain token bridge integration.

    This class wraps the ``IOFT`` interface used by OFT tokens built on
    LayerZero v2.  The OFT contract itself holds the bridging logic and fee
    estimation; no separate router contract is required.

    Token amounts are preserved 1:1 across chains (no protocol fee deducted
    from the token).  The only cost is the LayerZero native messaging fee
    paid on top in the source chain's native gas token.

    For OFTs deployed at the same contract address on every chain (e.g.
    USDT0 at ``0x1E4a5963aBFD975d8c9021ce480b42188849D41d``), omit
    ``dst_oft_address`` — it defaults to ``oft_address``.  For OFTs that use
    different addresses per chain, pass the destination chain contract address
    explicitly via ``dst_oft_address``.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the source chain.
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        oft_address: Address of the OFT contract on the source chain.
        dst_oft_address: Address of the OFT contract on the destination chain.
            Defaults to ``oft_address`` for unified-address OFTs.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        oft_address: str,
        dst_oft_address: str | None = None,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3
        self.oft_address = oft_address
        self.dst_oft_address = dst_oft_address or oft_address

    @property
    def protocol_name(self) -> str:
        return "LayerZeroOFT"

    def _lz_eid(self, evm_chain_id: int) -> int:
        """Map an EVM chain ID to a LayerZero v2 endpoint ID (EID)."""
        eid = _LZ_EID.get(evm_chain_id)
        if eid is None:
            raise BridgeError(f"LayerZeroOFT: unsupported chain ID {evm_chain_id}")
        return eid

    def _validate_tokens(self, token_in: Token, token_out: Token) -> None:
        """Validate that token addresses match the configured OFT contracts.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If either token address
                does not match the expected OFT contract address.
        """
        src_addr = Web3.to_checksum_address(self.oft_address)
        dst_addr = Web3.to_checksum_address(self.dst_oft_address)
        if Web3.to_checksum_address(token_in.address) != src_addr:
            raise BridgeError(
                f"LayerZeroOFT: token_in address {token_in.address!r} "
                f"does not match source OFT address {self.oft_address!r}"
            )
        if Web3.to_checksum_address(token_out.address) != dst_addr:
            raise BridgeError(
                f"LayerZeroOFT: token_out address {token_out.address!r} "
                f"does not match destination OFT address {self.dst_oft_address!r}"
            )

    async def quote_send_fee(
        self,
        amount: int,
        recipient: Address,
        slippage_bps: int = 50,
    ) -> int:
        """Estimate the native LayerZero messaging fee for a ``send`` call.

        Calls ``quoteSend`` on the OFT contract to get the exact native fee
        required for the cross-chain message.

        Args:
            amount: Token amount to bridge (raw, in local decimals).
            recipient: Recipient address on the destination chain.
            slippage_bps: Slippage tolerance in basis points used to compute
                ``minAmountLD``.

        Returns:
            Estimated native fee in wei.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On contract call failure.
        """
        dst_eid = self._lz_eid(self.dst_chain_id)
        to_bytes32 = address_to_bytes32(recipient)
        min_amount = self._apply_slippage(amount, slippage_bps)

        send_param = OFTSendParam(
            dstEid=dst_eid,
            to=to_bytes32,
            amountLD=amount,
            minAmountLD=min_amount,
            extraOptions=b"",
            composeMsg=b"",
            oftCmd=b"",
        )

        try:
            result = await LAYERZERO_OFT.fns.quoteSend(send_param, False).call(self.w3, to=self.oft_address)
            # quoteSend returns (nativeFee, lzTokenFee)
            native_fee: int = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise BridgeError(f"LayerZeroOFT: quoteSend failed: {exc}") from exc
        return native_fee

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a LayerZero OFT bridge quote.

        OFT transfers are 1:1: the full ``amount_in`` is received on the
        destination chain (``bridge_fee`` is zero in token terms).  The
        LayerZero native messaging fee is paid separately in the source
        chain's native gas token when submitting the transaction.

        Args:
            token_in: Source chain OFT token.
            token_out: Destination chain OFT token (same asset, different chain).
            amount_in: Amount to bridge.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If token addresses do not
                match the configured OFT contracts.
        """
        self._validate_tokens(token_in, token_out)
        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_in.amount),
            bridge_fee=TokenAmount(token=token_in, amount=0),
            estimated_time_seconds=30,  # LayerZero v2 is typically 15–30 seconds
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 50,
        refund_address: Address | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a LayerZero OFT ``send`` transaction.

        Fetches the current native messaging fee via ``quoteSend``, then
        encodes a ``send`` call that transfers ``amount_in`` to ``recipient``
        on the destination chain.

        Args:
            token_in: Source chain OFT token.
            token_out: Destination chain OFT token.
            amount_in: Amount to bridge.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            refund_address: Address to receive any excess native fee refund.
                Defaults to ``recipient``.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On fee estimation failure
                or if token addresses do not match the configured OFT contracts.
        """
        self._validate_tokens(token_in, token_out)
        _refund = refund_address or recipient
        dst_eid = self._lz_eid(self.dst_chain_id)
        to_bytes32 = address_to_bytes32(recipient)
        min_amount = self._apply_slippage(amount_in.amount, slippage_bps)

        send_param = OFTSendParam(
            dstEid=dst_eid,
            to=to_bytes32,
            amountLD=amount_in.amount,
            minAmountLD=min_amount,
            extraOptions=b"",
            composeMsg=b"",
            oftCmd=b"",
        )

        native_fee = await self.quote_send_fee(amount_in.amount, recipient, slippage_bps)
        messaging_fee = MessagingFee(nativeFee=native_fee, lzTokenFee=0)

        call_data = LAYERZERO_OFT.fns.send(send_param, messaging_fee, _refund).data

        return {
            "to": self.oft_address,
            "data": "0x" + call_data.hex(),
            "value": str(native_fee),
            "gas": str(300_000),
        }

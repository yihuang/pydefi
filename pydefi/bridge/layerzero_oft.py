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

from pydefi._math import apply_slippage
from pydefi._utils import address_to_bytes32
from pydefi.abi.bridge import LAYERZERO_OFT, MessagingFee, OFTSendParam
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

# LayerZero v2 executor options (``OptionsBuilder`` TYPE_3 wire format). The
# destination endpoint only runs ``lzReceive`` (OFT credit) and ``lzCompose``
# (the DeFiVM program) for the executor gas these options fund — omit them and
# the compose silently never executes.
_OPTIONS_TYPE_3 = (3).to_bytes(2, "big")
_WORKER_ID_EXECUTOR = 1
_OPTION_TYPE_LZRECEIVE = 1
_OPTION_TYPE_LZCOMPOSE = 3
_DEFAULT_LZ_RECEIVE_GAS = 150_000  # destination OFT credit/mint
_DEFAULT_COMPOSE_GAS = 800_000  # destination DeFiVM swap -> supply program
_BRIDGE_SEND_GAS = 300_000  # source ``send`` gas budget (plain transfer)
_COMPOSE_SEND_GAS = 400_000  # source ``send`` gas budget (larger compose msg)


def _executor_option(option_type: int, option_data: bytes) -> bytes:
    """One ``OptionsBuilder`` executor option: ``worker_id || uint16(len+1) ||
    option_type || option_data`` (the ``+1`` counts the type byte)."""
    return bytes([_WORKER_ID_EXECUTOR]) + (len(option_data) + 1).to_bytes(2, "big") + bytes([option_type]) + option_data


def _compose_options(lz_receive_gas: int, compose_gas: int, *, compose_index: int = 0) -> bytes:
    """LayerZero v2 ``extraOptions`` funding the destination ``lzReceive`` (OFT
    credit) and ``lzCompose`` (DeFiVM program) executor gas."""
    lz_receive = _executor_option(_OPTION_TYPE_LZRECEIVE, lz_receive_gas.to_bytes(16, "big"))
    lz_compose = _executor_option(
        _OPTION_TYPE_LZCOMPOSE, compose_index.to_bytes(2, "big") + compose_gas.to_bytes(16, "big")
    )
    return _OPTIONS_TYPE_3 + lz_receive + lz_compose


class LayerZeroOFT:
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
        self.src_chain_id = src_chain_id
        self.dst_chain_id = dst_chain_id
        self.w3 = w3
        self.oft_address = oft_address
        self.dst_oft_address = dst_oft_address or oft_address

    protocol_name: str = "LayerZeroOFT"

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

    def _send_param(
        self,
        amount: int,
        recipient: Address,
        slippage_bps: int,
        *,
        compose_msg: bytes = b"",
        extra_options: bytes = b"",
    ) -> OFTSendParam:
        """Build the ``SendParam`` for ``quoteSend`` / ``send`` — *compose_msg* and
        *extra_options* are empty for a plain transfer, set for a compose."""
        return OFTSendParam(
            dstEid=self._lz_eid(self.dst_chain_id),
            to=address_to_bytes32(recipient),
            amountLD=amount,
            minAmountLD=apply_slippage(amount, slippage_bps),
            extraOptions=extra_options,
            composeMsg=compose_msg,
            oftCmd=b"",
        )

    async def _quote_native_fee(self, send_param: OFTSendParam) -> int:
        """Call ``quoteSend`` and return the native messaging fee (wei)."""
        try:
            result = await LAYERZERO_OFT.fns.quoteSend(send_param, False).call(self.w3, to=self.oft_address)
            # quoteSend returns (nativeFee, lzTokenFee)
            return result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise BridgeError(f"LayerZeroOFT: quoteSend failed: {exc}") from exc

    async def _send_tx(self, send_param: OFTSendParam, refund: Address, gas: int) -> dict[str, Any]:
        """Quote the native fee for *send_param* and encode the ``send`` tx dict (``value`` = the fee)."""
        native_fee = await self._quote_native_fee(send_param)
        call_data = LAYERZERO_OFT.fns.send(send_param, MessagingFee(nativeFee=native_fee, lzTokenFee=0), refund).data
        return {"to": self.oft_address, "data": "0x" + call_data.hex(), "value": str(native_fee), "gas": str(gas)}

    async def quote_send_fee(self, amount: int, recipient: Address, slippage_bps: int = 50) -> int:
        """Native LayerZero messaging fee (wei) for a plain ``send`` of *amount* to *recipient*."""
        return await self._quote_native_fee(self._send_param(amount, recipient, slippage_bps))

    async def get_quote(self, token_in: Token, token_out: Token, amount_in: TokenAmount, **kwargs: Any) -> BridgeQuote:
        """OFT bridge quote: 1:1 in token terms (full *amount_in* received, zero token fee); the
        LayerZero native messaging fee is paid separately in source gas. Raises if the token
        addresses don't match the configured OFT contracts."""
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
        """A LayerZero OFT ``send`` tx moving *amount_in* to *recipient* on the destination
        chain. *refund_address* (default *recipient*) receives any excess native fee. Raises
        if the tokens don't match the configured OFT contracts."""
        self._validate_tokens(token_in, token_out)
        send_param = self._send_param(amount_in.amount, recipient, slippage_bps)
        return await self._send_tx(send_param, refund_address or recipient, _BRIDGE_SEND_GAS)

    async def build_compose_send(
        self,
        amount_in: TokenAmount,
        composer: Address,
        program: bytes,
        *,
        compose_gas: int = _DEFAULT_COMPOSE_GAS,
        lz_receive_gas: int = _DEFAULT_LZ_RECEIVE_GAS,
        slippage_bps: int = 50,
        refund_address: Address | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """``ComposeBridge`` legs: a single OFT ``send`` to *composer* (the
        :class:`OFTComposer`) carrying *program* as ``composeMsg``, with ``extraOptions``
        funding the destination ``lzReceive`` + ``lzCompose`` gas. A native OFT burns from
        the sender, so there is no approve leg. The bridged token must be the OFT itself;
        *refund_address* (default *composer*) receives any excess native fee."""
        if not program:
            raise BridgeError("LayerZeroOFT: program must not be empty for compose transactions.")
        if Web3.to_checksum_address(amount_in.token.address) != Web3.to_checksum_address(self.oft_address):
            raise BridgeError(
                f"LayerZeroOFT: amount_in token {amount_in.token.address!r} is not the OFT {self.oft_address!r}"
            )
        send_param = self._send_param(
            amount_in.amount,
            composer,
            slippage_bps,
            compose_msg=program,
            extra_options=_compose_options(lz_receive_gas, compose_gas),
        )
        return [await self._send_tx(send_param, refund_address or composer, _COMPOSE_SEND_GAS)]

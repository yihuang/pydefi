"""Eureka (IBC v2) cross-chain bridge adapter.

Builds ``ICS20Transfer.sendTransfer(SendTransferMsg)`` calldata locally from
a (transfer-addr, sourceClient) tuple.

References: `IBC v2 spec <https://github.com/cosmos/ibc/tree/main/spec/IBC_V2>`_.
"""

from __future__ import annotations

from typing import Any

from pydefi.abi.bridge import ICS20_DEFAULT_PORT, ICS20_TRANSFER
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import Address, BridgeQuote, Token, TokenAmount


def validate_send_transfer_fields(denom: Address | str, amount: int, timeout_timestamp: int) -> bytes:
    """Common validation for ``sendTransfer`` / ``sendTransferAndCompose``
    calldata encoders. Returns the 20-byte ``denom`` bytes."""
    if isinstance(denom, str):
        denom_bytes = bytes.fromhex(denom.removeprefix("0x"))
    else:
        denom_bytes = bytes(denom)
    if len(denom_bytes) != 20:
        raise ValueError(f"denom must be a 20-byte EVM address, got {len(denom_bytes)} bytes")
    if amount < 0 or amount >> 256:
        raise ValueError(f"amount {amount} out of uint256 range")
    if timeout_timestamp < 0 or timeout_timestamp >> 64:
        raise ValueError(f"timeout_timestamp {timeout_timestamp} out of uint64 range")
    return denom_bytes


def encode_send_transfer_calldata(
    *,
    denom: Address | str,
    amount: int,
    receiver: str,
    source_client: str,
    timeout_timestamp: int,
    dest_port: str = ICS20_DEFAULT_PORT,
    memo: str = "",
) -> bytes:
    """ABI-encode a call to ``ICS20Transfer.sendTransfer(SendTransferMsg)``.
    ``receiver`` is opaque to the EVM-side router (bech32 for Cosmos, hex for
    EVM destinations); ``memo`` carries PFM hops and middleware hooks."""
    denom_bytes = validate_send_transfer_fields(denom, amount, timeout_timestamp)
    return bytes(
        ICS20_TRANSFER.fns.sendTransfer(
            (denom_bytes, amount, receiver, source_client, dest_port, timeout_timestamp, memo)
        ).data
    )


class Eureka(BaseBridge):
    """IBC v2 (Eureka) bridge via :class:`ICS20Transfer.sendTransfer`.

    ``dst_chain_id`` is informational only (Cosmos chain ids are strings;
    pass ``0`` if you don't need it). ``source_client_id`` is the client on
    the source chain that points at the destination (e.g. ``"07-tendermint-0"``).
    """

    PROTOCOL = "Eureka"

    def __init__(
        self,
        src_chain_id: int,
        dst_chain_id: int,
        *,
        ics20_transfer_addr: Address,
        source_client_id: str,
        dest_port: str = ICS20_DEFAULT_PORT,
        timeout_seconds: int = 600,
        estimated_time_seconds: int = 60,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        if len(ics20_transfer_addr) != 20:
            raise ValueError(f"ics20_transfer_addr must be a 20-byte EVM address, got {len(ics20_transfer_addr)} bytes")
        self.ics20_transfer_addr = ics20_transfer_addr
        self.source_client_id = source_client_id
        self.dest_port = dest_port
        self.timeout_seconds = timeout_seconds
        self.estimated_time_seconds = estimated_time_seconds

    @property
    def protocol_name(self) -> str:
        return self.PROTOCOL

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Return a quote for an Eureka ICS-20 transfer.

        ICS-20 charges no on-chain fee and preserves the base-unit integer
        amount across the relay (``build_bridge_tx`` sends ``amount_in.amount``
        unchanged). Differing ``token_in``/``token_out`` decimals would misreport
        the human value, so they're rejected here just as ``RouteDAG.bridge``
        rejects them at build time. Pass ``amount_out_override`` for IFT-style
        wrapping that genuinely changes precision.
        """
        if amount_in.token.address != token_in.address:
            raise BridgeError(f"amount_in.token ({amount_in.token}) must match token_in ({token_in})")
        amount_out_override = kwargs.get("amount_out_override")
        if amount_out_override is not None:
            amount_out_raw = int(amount_out_override)
        elif token_in.decimals == token_out.decimals:
            amount_out_raw = amount_in.amount
        else:
            raise BridgeError(
                f"Eureka quote decimals mismatch: {token_in.symbol} ({token_in.decimals} dec) "
                f"!= {token_out.symbol} ({token_out.decimals} dec). ICS-20 preserves the "
                f"base-unit amount across the relay, so differing decimals would misreport the "
                f"quote; pass `amount_out_override` for IFT-style precision changes"
            )

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=0),
            estimated_time_seconds=kwargs.get("estimated_time_seconds", self.estimated_time_seconds),
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 50,  # noqa: ARG002 — ICS-20 doesn't quote-and-slip
        *,
        now: int | None = None,
        receiver: str | None = None,
        memo: str = "",
        gas: int = 600_000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an ``ICS20Transfer.sendTransfer`` transaction.

        ``receiver`` defaults to ``recipient`` hex-encoded — pass a bech32
        string when the destination is a Cosmos chain. ``slippage_bps`` is
        accepted for :class:`BaseBridge` parity but has no effect (ICS-20 is
        amount-preserving).
        """
        if amount_in.token.address != token_in.address:
            raise BridgeError(f"amount_in.token ({amount_in.token}) must match token_in ({token_in})")
        if token_in.is_native():
            raise BridgeError("Eureka ICS-20 transfers require an ERC-20; wrap native gas first")
        import time

        timeout_at = (now if now is not None else int(time.time())) + self.timeout_seconds
        receiver_str = receiver if receiver is not None else "0x" + bytes(recipient).hex()

        data = encode_send_transfer_calldata(
            denom=token_in.address,
            amount=amount_in.amount,
            receiver=receiver_str,
            source_client=self.source_client_id,
            timeout_timestamp=timeout_at,
            dest_port=kwargs.get("dest_port", self.dest_port),
            memo=memo,
        )

        return {
            "to": self.ics20_transfer_addr,
            "data": "0x" + data.hex(),
            "value": "0",
            "gas": str(gas),
        }

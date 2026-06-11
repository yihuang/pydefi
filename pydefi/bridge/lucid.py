"""Lucid Labs cross-chain bridge — ``AssetController.transferTo`` is 1:1, paid
in source-chain native via the Hyperlane or LayerZero adapter.

Supported sources:

* MANTRA (5888) — Hyperlane. USDC/WETH → Base, USDT → Optimism.
* Kite (2366) — LayerZero. USDC/WETH → Avalanche, USDT → Celo.

Adapter addresses are not published by Lucid; discover them via a
``TransferRelayed`` event scan against any controller.

Docs: https://docs.lucidlabs.fi/developer-reference/deployed-contracts/controller-contracts
"""

from __future__ import annotations

from typing import Any, Literal

from eth_abi import encode as abi_encode
from web3 import AsyncWeb3, Web3
from web3.types import FilterParams

from pydefi._utils import decode_address, encode_address
from pydefi.abi.bridge import (
    LUCID_ASSET_CONTROLLER,
    LUCID_HYPERLANE_ADAPTER,
    LUCID_LAYERZERO_ADAPTER,
)
from pydefi.exceptions import BridgeError
from pydefi.types import ZERO_ADDRESS, Address, BridgeQuote, ChainId, Token, TokenAmount

_TRANSFER_RELAYED_TOPIC = Web3.keccak(text="TransferRelayed(bytes32,address)")


async def discover_adapter(
    w3: AsyncWeb3,
    controller_address: Address,
    from_block: int = 0,
    to_block: int | Literal["latest"] = "latest",
    *,
    chunk_size: int = 10_000,
) -> Address:
    """Return the bridge adapter address by scanning the controller's past
    ``TransferRelayed`` events, walking backwards from ``to_block`` in
    ``chunk_size``-block windows. Picks the most recent match.

    The chunked walk is required for providers that cap log range (MANTRA's
    public RPC rejects requests with ``to - from > 10000``). Kite has no cap
    but pays only one extra eth_blockNumber round-trip.

    Only ``int`` and ``"latest"`` are accepted for ``to_block`` — other
    BlockIdentifier forms (block tags like ``"finalized"``, hash strings)
    are not supported and raise :class:`BridgeError`. ``chunk_size`` must
    be positive.
    """
    if chunk_size < 1:
        raise BridgeError(f"Lucid: chunk_size must be >= 1, got {chunk_size}")
    if isinstance(to_block, int):
        head = to_block
    elif to_block == "latest":
        head = await w3.eth.block_number
    else:
        raise BridgeError(f"Lucid: unsupported to_block {to_block!r}; pass an int or 'latest'")

    address = Web3.to_checksum_address(bytes(controller_address))
    cursor = head
    while cursor >= from_block:
        lo = max(from_block, cursor - chunk_size + 1)
        params: FilterParams = {
            "address": address,
            "topics": [_TRANSFER_RELAYED_TOPIC],
            "fromBlock": lo,
            "toBlock": cursor,
        }
        logs = await w3.eth.get_logs(params)
        if logs:
            # Event data: single 32-byte word, address right-aligned in the low 20 bytes.
            return Address(bytes(logs[-1]["data"])[-20:])
        if lo == from_block:
            break
        cursor = lo - 1
    raise BridgeError(
        f"Lucid: no TransferRelayed events on controller 0x{bytes(controller_address).hex()} "
        f"between {from_block} and {head}"
    )


# Generous: transferTo calls into the adapter, which calls the LZ endpoint
# or Hyperlane mailbox — multiple external hops.
_DEFAULT_SEND_GAS = 500_000

AdapterKind = Literal["layerzero", "hyperlane"]

_DEFAULT_ADAPTER_BY_SRC: dict[int, AdapterKind] = {
    ChainId.KITE: "layerzero",
    ChainId.KITE_TESTNET: "layerzero",
    ChainId.MANTRA: "hyperlane",
}


def _encode_transfer_message(
    nonce: int,
    dest_chain_id: int,
    recipient: "Address",
    amount: int,
    unwrap: bool,
    threshold: int = 1,
    transfer_id: bytes = b"\x00" * 32,
) -> bytes:
    """Encode a Lucid Transfer message for quoting."""
    return abi_encode(
        ["(uint256,uint256,address,uint256,bool,uint256,bytes32)"],
        [(nonce, dest_chain_id, bytes(recipient), amount, unwrap, threshold, transfer_id)],
    )


class LucidBridge:
    """One instance per (source controller, adapter). Token amounts are 1:1;
    only cost is the native fee from ``adapter.quoteMessage(..., includeFee=true)``.

    ``adapter_kind`` defaults from ``src_chain_id``; override only to test
    against a non-canonical adapter on the same source.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        controller_address: Address,
        adapter_address: Address,
        dest_gas_limit: int = 300_000,
        adapter_kind: AdapterKind | None = None,
    ) -> None:
        if src_chain_id not in _DEFAULT_ADAPTER_BY_SRC:
            raise BridgeError(
                f"LucidBridge: source chain must be Kite (2366), Kite testnet "
                f"(2368), or MANTRA (5888), got {src_chain_id}"
            )
        self.src_chain_id = src_chain_id
        self.dst_chain_id = dst_chain_id
        self.w3 = w3
        self.controller_address = controller_address
        self.adapter_address = adapter_address
        self.dest_gas_limit = dest_gas_limit
        self.adapter_kind: AdapterKind = adapter_kind or _DEFAULT_ADAPTER_BY_SRC[src_chain_id]
        if self.adapter_kind not in ("layerzero", "hyperlane"):
            raise BridgeError(f"LucidBridge: unknown adapter_kind {self.adapter_kind!r}")
        self._cached_source_token: Address | None = None

    @property
    def protocol_name(self) -> str:
        return "Lucid"

    def _encode_bridge_options(self, refund_address: Address, gas_limit: int) -> bytes:
        # LayerZero uses uint128 gasLimit, Hyperlane uses uint256 — only field
        # that differs between the two adapter Options structs.
        gas_type = "uint128" if self.adapter_kind == "layerzero" else "uint256"
        return abi_encode([f"(address,{gas_type})"], [(bytes(refund_address), gas_limit)])

    async def _assert_token_in(self, token_in: Token) -> None:
        if token_in.chain_id != self.src_chain_id:
            raise BridgeError(
                f"Lucid: token_in.chain_id={token_in.chain_id} does not match src_chain_id={self.src_chain_id}"
            )
        if self._cached_source_token is None:
            try:
                token_str = await LUCID_ASSET_CONTROLLER.fns.token().call(self.w3, to=self.controller_address)
            except Exception as exc:
                raise BridgeError(f"Lucid: controller.token() read failed: {exc}") from exc
            self._cached_source_token = decode_address(token_str, self.src_chain_id)
        if bytes(token_in.address) != bytes(self._cached_source_token):
            raise BridgeError(
                f"Lucid: token_in {encode_address(token_in.address, self.src_chain_id)} "
                f"does not match controller token "
                f"{encode_address(self._cached_source_token, self.src_chain_id)}"
            )

    async def _dest_controller(self) -> Address:
        try:
            dst = await LUCID_ASSET_CONTROLLER.fns.getControllerForChain(self.dst_chain_id).call(
                self.w3, to=self.controller_address
            )
        except Exception as exc:
            raise BridgeError(f"Lucid: getControllerForChain({self.dst_chain_id}) failed: {exc}") from exc
        if int(dst, 16) == 0:
            raise BridgeError(f"Lucid: controller has no destination configured for chain {self.dst_chain_id}")
        return decode_address(dst, self.src_chain_id)

    async def quote_native_fee(
        self,
        amount: int,
        recipient: Address,
        unwrap: bool = False,
    ) -> int:
        """Required ``msg.value`` for the upcoming ``transferTo`` call (native wei)."""
        dest_ctrl = await self._dest_controller()
        message = _encode_transfer_message(
            nonce=0,
            dest_chain_id=self.dst_chain_id,
            recipient=recipient,
            amount=amount,
            unwrap=unwrap,
        )
        adapter_abi = LUCID_LAYERZERO_ADAPTER if self.adapter_kind == "layerzero" else LUCID_HYPERLANE_ADAPTER
        try:
            fee = await adapter_abi.fns.quoteMessage(
                bytes(dest_ctrl),
                self.dst_chain_id,
                self.dest_gas_limit,
                message,
                True,
            ).call(self.w3, to=self.adapter_address)
        except Exception as exc:
            raise BridgeError(f"Lucid: quoteMessage failed: {exc}") from exc
        return int(fee)

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address | None = None,
        unwrap: bool = False,
    ) -> BridgeQuote:
        # `recipient` only affects the encoded payload length (fixed for Transfer),
        # so the zero address is fine for read-only quoting.
        await self._assert_token_in(token_in)
        _recipient = recipient if recipient is not None else ZERO_ADDRESS
        fee = await self.quote_native_fee(amount_in.amount, _recipient, unwrap)

        native_token = Token(
            chain_id=self.src_chain_id,
            address=ZERO_ADDRESS,
            symbol="NATIVE",
            decimals=18,
        )
        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_in.amount),
            bridge_fee=TokenAmount(token=native_token, amount=fee),
            estimated_time_seconds=60,
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 0,
        unwrap: bool = False,
        refund_address: Address | None = None,
        fee_buffer_bps: int = 0,
    ) -> dict[str, Any]:
        """Caller must have pre-approved ``token_in`` to the controller. Excess
        ``msg.value`` is refunded by the adapter to ``refund_address`` (defaults
        to ``recipient``). ``fee_buffer_bps`` bumps msg.value above the live quote
        for cases where broadcast lags the quote and adapter pricing drifts.

        ``slippage_bps`` and ``token_out`` are accepted for ``BaseBridge``
        compatibility and ignored: Lucid is 1:1, and the destination token is
        implicit from the controller — neither value can change the outcome.
        """
        del slippage_bps, token_out
        await self._assert_token_in(token_in)
        _refund = refund_address if refund_address is not None else recipient

        fee = await self.quote_native_fee(amount_in.amount, recipient, unwrap)
        if fee_buffer_bps:
            fee = fee * (10_000 + fee_buffer_bps) // 10_000

        bridge_options = self._encode_bridge_options(_refund, self.dest_gas_limit)
        call_data = LUCID_ASSET_CONTROLLER.fns.transferTo(
            bytes(recipient),
            amount_in.amount,
            unwrap,
            self.dst_chain_id,
            bytes(self.adapter_address),
            bridge_options,
        ).data

        return {
            "to": self.controller_address,
            "data": "0x" + call_data.hex(),
            "value": str(fee),
            "gas": str(_DEFAULT_SEND_GAS),
        }

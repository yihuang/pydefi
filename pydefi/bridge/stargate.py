"""
Stargate Finance cross-chain bridge integration.

Stargate is built on LayerZero and supports bridging native stablecoin
liquidity across chains.  This module wraps the ``IStargate`` router
interface via :class:`~eth_contract.Contract`.

Solana is supported as a *destination* chain via the LayerZero V2 endpoint
(endpoint ID ``30168``).  Bridging *from* Solana to EVM requires the
Stargate Solana program and is outside the scope of this module.

Docs: https://stargateprotocol.gitbook.io/stargate/
"""

from __future__ import annotations

from typing import Any

from web3 import AsyncWeb3

from pydefi.abi.bridge import STARGATE_ROUTER
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

# LayerZero chain IDs differ from EVM chain IDs.
# Solana uses the LayerZero V2 endpoint ID (30168).
_LZ_CHAIN_ID: dict[int, int] = {
    1: 101,  # Ethereum
    56: 102,  # BSC
    43114: 106,  # Avalanche
    137: 109,  # Polygon
    42161: 110,  # Arbitrum
    10: 111,  # Optimism
    250: 112,  # Fantom
    8453: 184,  # Base
    1399811149: 30168,  # Solana (LayerZero V2 endpoint ID)
}

# Stargate pool IDs for common tokens
_POOL_IDS: dict[str, int] = {
    "USDC": 1,
    "USDT": 2,
    "DAI": 3,
    "FRAX": 7,
    "USDD": 11,
    "ETH": 13,
    "sUSD": 14,
    "LUSD": 15,
    "MAI": 16,
    "METIS": 17,
    "metisUSDT": 19,
}


class Stargate(BaseBridge):
    """Stargate Finance cross-chain bridge integration.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the *source* chain.
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        router_address: Address of the Stargate ``Router`` contract on the
            source chain.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        router_address: str,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3
        self.router_address = router_address

    @property
    def protocol_name(self) -> str:
        return "Stargate"

    def _lz_chain_id(self, evm_chain_id: int) -> int:
        """Map an EVM chain ID to a LayerZero chain ID."""
        lz_id = _LZ_CHAIN_ID.get(evm_chain_id)
        if lz_id is None:
            raise BridgeError(f"Stargate: unsupported chain ID {evm_chain_id}")
        return lz_id

    def _pool_id(self, token: Token) -> int:
        """Return the Stargate pool ID for *token*."""
        pool_id = _POOL_IDS.get(token.symbol)
        if pool_id is None:
            raise BridgeError(f"Stargate: no pool ID for token {token.symbol}")
        return pool_id

    async def quote_lz_fee(
        self,
        dst_chain_id: int,
        recipient: str,
        dst_gas: int = 200_000,
    ) -> int:
        """Estimate the LayerZero messaging fee for a cross-chain transfer.

        Args:
            dst_chain_id: Destination EVM chain ID.
            recipient: Recipient address on the destination chain.
            dst_gas: Gas limit for the destination call.

        Returns:
            Estimated fee in native gas token wei.
        """
        lz_dst_chain = self._lz_chain_id(dst_chain_id)
        lz_tx_params = (dst_gas, 0, b"")
        to_bytes = bytes.fromhex(recipient[2:].lower().zfill(40))
        try:
            result = await STARGATE_ROUTER.fns.quoteLayerZeroFee(
                lz_dst_chain,
                1,  # TYPE_SWAP_REMOTE
                to_bytes,
                b"",
                lz_tx_params,
            ).call(self.w3, to=self.router_address)
            fee: int = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as exc:
            raise BridgeError(f"Stargate: quoteLayerZeroFee failed: {exc}") from exc
        return fee

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
    ) -> BridgeQuote:
        """Get a Stargate bridge quote.

        Args:
            token_in: Source token (must be a Stargate-supported asset).
            token_out: Destination token.
            amount_in: Amount to bridge.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.
        """
        # Stargate typically charges a 6-bp protocol fee
        PROTOCOL_FEE_BPS = 6
        fee_raw = amount_in.amount * PROTOCOL_FEE_BPS // 10_000
        amount_out_raw = amount_in.amount - fee_raw

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=fee_raw),
            estimated_time_seconds=180,  # ~3 min average
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        slippage_bps: int = 50,
        dst_gas: int = 200_000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Stargate bridge transaction.

        Args:
            token_in: Source token.
            token_out: Destination token.
            amount_in: Amount to send.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            dst_gas: Gas limit for the destination-chain call.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.
        """
        src_pool_id = self._pool_id(token_in)
        dst_pool_id = self._pool_id(token_out)
        lz_dst_chain = self._lz_chain_id(self.dst_chain_id)
        lz_fee = await self.quote_lz_fee(self.dst_chain_id, recipient, dst_gas)

        # Derive min_amount from post-fee output so it accounts for the protocol fee
        PROTOCOL_FEE_BPS = 6
        fee_raw = amount_in.amount * PROTOCOL_FEE_BPS // 10_000
        amount_out_raw = amount_in.amount - fee_raw
        min_amount = self._apply_slippage(amount_out_raw, slippage_bps)

        lz_tx_params = (dst_gas, 0, b"")
        to_bytes = bytes.fromhex(recipient[2:].lower().zfill(40))

        # Build call data via the Contract ABI
        call_data = STARGATE_ROUTER.fns.swap(
            lz_dst_chain,
            src_pool_id,
            dst_pool_id,
            recipient,
            amount_in.amount,
            min_amount,
            lz_tx_params,
            to_bytes,
            b"",
        ).data

        return {
            "to": self.router_address,
            "data": "0x" + call_data.hex() if isinstance(call_data, bytes) else call_data,
            "value": str(lz_fee),
            "gas": str(500_000),
        }

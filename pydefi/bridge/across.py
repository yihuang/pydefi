"""
Across Protocol cross-chain bridge integration.

Across uses UMA's optimistic oracle for fast (2-4 minute) bridging with
competitive fees.  This module wraps the ``SpokePool`` on-chain contract
and the Across Suggested Fees API.

Docs: https://docs.across.to/
"""

from __future__ import annotations

from typing import Any

import aiohttp
from eth_contract import Contract
from web3 import AsyncWeb3

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

# ---------------------------------------------------------------------------
# ABI fragments
# ---------------------------------------------------------------------------

_SPOKE_POOL_ABI = [
    "function depositV3(address depositor, address recipient, address inputToken, address outputToken, uint256 inputAmount, uint256 outputAmount, uint256 destinationChainId, address exclusiveRelayer, uint32 quoteTimestamp, uint32 fillDeadline, uint32 exclusivityDeadline, bytes calldata message) external payable",
    "function getCurrentTime() external view returns (uint256)",
]

_ACROSS_API_BASE = "https://app.across.to/api"


class Across(BaseBridge):
    """Across Protocol cross-chain bridge integration.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance for the source chain.
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        spoke_pool_address: Address of the ``SpokePool`` contract on the
            source chain.
        api_base_url: Override the Across Suggested Fees API base URL.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        src_chain_id: int,
        dst_chain_id: int,
        spoke_pool_address: str,
        api_base_url: str = _ACROSS_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.w3 = w3
        self.spoke_pool_address = spoke_pool_address
        self._api_base = api_base_url.rstrip("/")
        self._spoke_pool = Contract.from_abi(_SPOKE_POOL_ABI, to=spoke_pool_address)

    @property
    def protocol_name(self) -> str:
        return "Across"

    async def get_suggested_fees(
        self,
        token: Token,
        amount: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query the Across Suggested Fees API.

        Args:
            token: Token to bridge.
            amount: Raw amount to bridge.
            **kwargs: Additional query parameters (e.g. ``relayer``).

        Returns:
            Raw API response dictionary.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: On API error.
        """
        params: dict[str, Any] = {
            "token": token.address,
            "inputChainId": self.src_chain_id,
            "outputChainId": self.dst_chain_id,
            "amount": str(amount),
            **kwargs,
        }
        url = f"{self._api_base}/suggested-fees"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise BridgeError(f"Across API error ({resp.status}): {data}")
                return data  # type: ignore[return-value]

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get an Across bridge quote.

        Args:
            token_in: Source chain token.
            token_out: Destination chain token.
            amount_in: Amount to bridge.
            **kwargs: Forwarded to :meth:`get_suggested_fees`.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.
        """
        fees_data = await self.get_suggested_fees(token_in, amount_in.amount, **kwargs)

        total_relay_fee_pct = int(fees_data.get("totalRelayFee", {}).get("pct", "0"))
        # Fee pct is in units of 1e18 (1e18 = 100%)
        fee_raw = amount_in.amount * total_relay_fee_pct // (10**18)
        amount_out_raw = max(0, amount_in.amount - fee_raw)
        estimated_fill_time = int(fees_data.get("estimatedFillTimeSec", 120))

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=fee_raw),
            estimated_time_seconds=estimated_fill_time,
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: str,
        slippage_bps: int = 50,
        depositor: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an Across ``depositV3`` transaction.

        Args:
            token_in: Source token.
            token_out: Destination token (same asset on the destination chain).
            amount_in: Amount to deposit.
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points.
            depositor: Depositor address (defaults to ``recipient``).
            **kwargs: Forwarded to :meth:`get_suggested_fees`.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.
        """
        depositor = depositor or recipient
        fees_data = await self.get_suggested_fees(token_in, amount_in.amount, **kwargs)

        total_relay_fee_pct = int(fees_data.get("totalRelayFee", {}).get("pct", "0"))
        fee_raw = amount_in.amount * total_relay_fee_pct // (10**18)
        output_amount = self._apply_slippage(max(0, amount_in.amount - fee_raw), slippage_bps)
        quote_timestamp = int(fees_data.get("timestamp", 0))
        fill_deadline = quote_timestamp + 18_000  # 5 hours

        call_data = self._spoke_pool.fns.depositV3(
            depositor,
            recipient,
            token_in.address,
            token_out.address,
            amount_in.amount,
            output_amount,
            self.dst_chain_id,
            "0x" + "00" * 20,  # exclusiveRelayer (none)
            quote_timestamp,
            fill_deadline,
            0,  # exclusivityDeadline
            b"",
        ).data

        value = str(amount_in.amount) if token_in.is_native() else "0"

        return {
            "to": self.spoke_pool_address,
            "data": "0x" + call_data.hex() if isinstance(call_data, bytes) else call_data,
            "value": value,
            "gas": str(300_000),
        }

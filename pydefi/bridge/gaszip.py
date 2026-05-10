"""
GasZip cross-chain gas bridge integration.

GasZip allows users to send native gas tokens to one or many destination
chains from a single source-chain transaction.  This module wraps the
GasZip backend API and the on-chain ``IGasZip`` deposit contract.

Docs: https://docs.gas.zip/
"""

from __future__ import annotations

from typing import Any

import aiohttp

from pydefi.abi.bridge import GASZIP
from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import Address, BridgeQuote, Token, TokenAmount

_GASZIP_API_BASE = "https://backend.gas.zip/v2"

# GasZip chain IDs (their own internal numbering may differ; EVM IDs used here)
_SUPPORTED_CHAINS: set[int] = {
    1,  # Ethereum
    10,  # Optimism
    56,  # BSC
    137,  # Polygon
    8453,  # Base
    42161,  # Arbitrum
    43114,  # Avalanche
    59144,  # Linea
    534352,  # Scroll
    81457,  # Blast
    324,  # zkSync Era
    7777777,  # Zora
    130,  # Unichain
    480,  # World Chain
}


class GasZip(BaseBridge):
    """GasZip cross-chain gas bridge integration.

    GasZip bridges native gas tokens from a source chain to one or more
    destination chains in a single transaction.

    Args:
        src_chain_id: Source chain EVM ID.
        dst_chain_id: Destination chain EVM ID.
        contract_address: Address of the GasZip deposit contract on the
            source chain.
        api_base_url: Override the GasZip backend API base URL.
    """

    def __init__(
        self,
        src_chain_id: int,
        dst_chain_id: int,
        contract_address: str,
        api_base_url: str = _GASZIP_API_BASE,
    ) -> None:
        super().__init__(src_chain_id, dst_chain_id)
        self.contract_address = contract_address
        self._api_base = api_base_url.rstrip("/")

    @property
    def protocol_name(self) -> str:
        return "GasZip"

    def _check_chain(self, chain_id: int) -> None:
        """Raise :class:`~pydefi.exceptions.BridgeError` for unsupported chains."""
        if chain_id not in _SUPPORTED_CHAINS:
            raise BridgeError(f"GasZip: unsupported chain ID {chain_id}")

    async def get_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> BridgeQuote:
        """Get a GasZip bridge quote.

        GasZip only bridges *native gas tokens*; ``token_in`` and ``token_out``
        must both be native tokens (address ``0xEeee...``).

        Args:
            token_in: Source native token.
            token_out: Destination native token.
            amount_in: Amount to bridge (in native wei).
            **kwargs: Additional query parameters.

        Returns:
            A :class:`~pydefi.types.BridgeQuote`.

        Raises:
            :class:`~pydefi.exceptions.BridgeError`: If tokens are not native
                or chains are unsupported, or on API error.
        """
        self._check_chain(self.src_chain_id)
        self._check_chain(self.dst_chain_id)

        if not token_in.is_native() or not token_out.is_native():
            raise BridgeError("GasZip only supports native gas tokens for bridging")

        # GasZip quote endpoint: GET /v2/quotes/{srcChainId}/{amount}/{dstChainId}
        url = f"{self._api_base}/quotes/{self.src_chain_id}/{amount_in.amount}/{self.dst_chain_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise BridgeError(f"GasZip API error ({resp.status}): {text[:200]}")
                data = await resp.json(content_type=None)

        # Response: {"quotes": [{"chain": int, "expected": str, "speed": int, ...}]}
        quotes_list = data.get("quotes", [])
        if not quotes_list or not quotes_list[0].get("expected"):
            raise BridgeError("GasZip: no quotes returned from API")

        amount_out_raw = int(quotes_list[0]["expected"])
        fee_raw = max(0, amount_in.amount - amount_out_raw)
        estimated_time = int(quotes_list[0].get("speed", 30))

        return BridgeQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=amount_out_raw),
            bridge_fee=TokenAmount(token=token_in, amount=fee_raw),
            estimated_time_seconds=estimated_time,
            protocol=self.protocol_name,
        )

    async def build_bridge_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        recipient: Address,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a GasZip deposit transaction.

        The transaction calls ``deposit(to, chains)`` on the GasZip contract,
        where ``to`` is the recipient address encoded as a ``uint256`` and
        ``chains`` is a list containing the single destination chain ID.

        Args:
            token_in: Source native token.
            token_out: Destination native token.
            amount_in: Amount to bridge (in native wei).
            recipient: Receiver address on the destination chain.
            slippage_bps: Slippage tolerance in basis points (informational).
            **kwargs: Ignored.

        Returns:
            Transaction dict with ``to``, ``data``, ``value``, ``gas``.
        """
        self._check_chain(self.src_chain_id)
        self._check_chain(self.dst_chain_id)

        if not token_in.is_native() or not token_out.is_native():
            raise BridgeError("GasZip only supports native gas tokens for bridging")

        # Validate and encode recipient address as uint256
        to_uint256 = int.from_bytes(recipient, "big")
        call_data: bytes = GASZIP.fns.deposit(
            to_uint256,
            [self.dst_chain_id],
        ).data

        return {
            "to": self.contract_address,
            "data": "0x" + call_data.hex(),
            "value": str(amount_in.amount),
            "gas": str(100_000),
        }

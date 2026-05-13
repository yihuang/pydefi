"""Quote a lane across multiple bridges and rank by effective delivered amount.

Legacy bridges (CCTP / Stargate / Across / Mayan / LayerZeroOFT / GasZip /
Relay) deduct their fee from ``amount_out`` and set ``bridge_fee.token``
to ``token_in``, so quotes are directly comparable.  CCIP leaves the
bridged amount intact and charges a separate fee in native gas or LINK;
ranking it fairly needs a caller-supplied ``FeeConverter`` that prices
the fee in ``token_out`` sub-units.  Unconvertible quotes are tagged
``fee_unranked`` and sorted last — never treated as free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydefi.bridge.base import BaseBridge
from pydefi.exceptions import BridgeError
from pydefi.types import BridgeQuote, Token, TokenAmount

#: Returns the fee in ``quote.token_out`` sub-units, or ``None`` if unpriced.
FeeConverter = Callable[[BridgeQuote], "int | None"]


@dataclass
class RankedBridgeQuote:
    """A :class:`BridgeQuote` paired with its fee-adjusted score.

    ``effective_amount_out`` is ``amount_out`` for quotes whose fee is already
    netted into the bridged amount; otherwise it's ``amount_out`` minus the
    converted fee.  ``fee_unranked`` is set when the fee is in a third token
    and no converter could price it — such quotes sort last.
    """

    quote: BridgeQuote
    effective_amount_out: int
    fee_in_token_out_units: int
    fee_unranked: bool

    @property
    def protocol(self) -> str:
        return self.quote.protocol


def rank_bridge_quotes(
    quotes: Sequence[BridgeQuote],
    *,
    fee_converter: FeeConverter | None = None,
) -> list[RankedBridgeQuote]:
    """Rank *quotes* by effective amount delivered, best first.

    Quotes whose fee is denominated in a third token and cannot be priced by
    *fee_converter* are tagged ``fee_unranked`` and sorted last.  The sort is
    stable: input order is preserved within the rankable and unrankable
    groups.
    """
    ranked: list[RankedBridgeQuote] = []
    for quote in quotes:
        fee_addr = quote.bridge_fee.token.address
        amount_out = quote.amount_out.amount

        if fee_addr in (quote.token_in.address, quote.token_out.address):
            # Fee already netted into amount_out by the bridge.
            ranked.append(RankedBridgeQuote(quote, amount_out, 0, False))
            continue

        converted = fee_converter(quote) if fee_converter is not None else None
        if converted is None:
            ranked.append(RankedBridgeQuote(quote, amount_out, 0, True))
        else:
            ranked.append(RankedBridgeQuote(quote, amount_out - converted, converted, False))

    # Ranked first by effective_amount_out (desc), unranked last in insertion order
    ranked.sort(key=lambda r: (r.fee_unranked, 0 if r.fee_unranked else -r.effective_amount_out))
    return ranked


class BridgeRouter:
    """Fan ``get_quote`` out to a set of bridges in parallel.

    Bridges that raise :class:`BridgeError` (lane unsupported, token not on
    allowlist, …) are silently dropped; any other exception propagates.  The
    returned quotes are typically passed to :func:`rank_bridge_quotes` for
    fee-aware ranking — see the example below.

    Example::

        router = BridgeRouter([
            CCTP(w3, src_chain_id=1, dst_chain_id=8453),
            CCIP(w3, src_chain_id=1, dst_chain_id=8453),
        ])

        def fee_to_usdc(q):  # CCIP pays in native ETH; price it in token_out.
            if q.protocol != "CCIP":
                return None
            return int(q.bridge_fee.amount * eth_price_in_usdc * 10**q.token_out.decimals // 10**18)

        quotes = await router.quote_all(token_in, token_out, amount)
        ranked = rank_bridge_quotes(quotes, fee_converter=fee_to_usdc)
    """

    def __init__(self, bridges: Sequence[BaseBridge]) -> None:
        if not bridges:
            raise ValueError("BridgeRouter requires at least one bridge")
        self.bridges: list[BaseBridge] = list(bridges)

    async def quote_all(
        self,
        token_in: Token,
        token_out: Token,
        amount_in: TokenAmount,
        **kwargs: Any,
    ) -> list[BridgeQuote]:
        """Fetch quotes from every bridge in parallel; drop BridgeError responses."""
        results = await asyncio.gather(
            *(bridge.get_quote(token_in, token_out, amount_in, **kwargs) for bridge in self.bridges),
            return_exceptions=True,
        )
        quotes: list[BridgeQuote] = []
        for result in results:
            if isinstance(result, BridgeError):
                continue
            if isinstance(result, BaseException):
                raise result
            quotes.append(result)
        return quotes

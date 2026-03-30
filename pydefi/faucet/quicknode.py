"""
QuickNode testnet faucet integration.

Requires a QuickNode API token.  Set the ``QUICKNODE_TOKEN`` environment
variable or pass the token directly to :class:`QuickNodeFaucet`.

API reference: https://www.quicknode.com/docs/faucets
"""

from __future__ import annotations

import aiohttp

from pydefi.exceptions import FaucetError
from pydefi.faucet.base import BaseFaucet
from pydefi.types import ChainId

# QuickNode faucet endpoint.
_QUICKNODE_FAUCET_URL = "https://faucet.quicknode.com/drip"

# Mapping from EVM chain ID to the QuickNode chain slug.
_CHAIN_SLUGS: dict[int, str] = {
    ChainId.SEPOLIA: "ethereum-sepolia",
}


class QuickNodeFaucet(BaseFaucet):
    """QuickNode testnet faucet.

    Requests testnet ETH from QuickNode's faucet service.
    Requires a QuickNode API token with faucet access.

    Args:
        token: QuickNode API token.  Defaults to the ``QUICKNODE_TOKEN``
            environment variable when ``None`` is passed.
        chain_id: Target chain (default: :attr:`~pydefi.types.ChainId.SEPOLIA`).
        api_base_url: Override the default QuickNode faucet endpoint URL.

    Raises:
        :class:`~pydefi.exceptions.FaucetError`: On construction when
            *chain_id* is not supported.

    Example::

        import os
        from pydefi.faucet import QuickNodeFaucet

        faucet = QuickNodeFaucet(token=os.environ["QUICKNODE_TOKEN"])
        tx_hash = await faucet.request("0xYourAddress")
    """

    def __init__(
        self,
        token: str,
        chain_id: int = ChainId.SEPOLIA,
        api_base_url: str = _QUICKNODE_FAUCET_URL,
    ) -> None:
        if chain_id not in _CHAIN_SLUGS:
            raise FaucetError(f"QuickNodeFaucet: unsupported chain ID {chain_id}")
        self._token = token
        self._chain_id = chain_id
        self._api_base = api_base_url.rstrip("/")

    @property
    def chain_id(self) -> int:
        return self._chain_id

    async def request(self, address: str) -> str | None:
        """Request testnet ETH from QuickNode's faucet.

        Args:
            address: The wallet address to fund.

        Returns:
            The transaction hash returned by QuickNode, or ``None``.

        Raises:
            :class:`~pydefi.exceptions.FaucetError`: If the faucet
                returns a non-2xx HTTP status code.
        """
        chain = _CHAIN_SLUGS[self._chain_id]
        headers = {
            "x-api-key": self._token,
            "Content-Type": "application/json",
        }
        payload = {
            "address": address,
            "chain": chain,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._api_base,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise FaucetError(
                        f"QuickNode faucet error ({resp.status}): {text[:200]}"
                    )
                data = await resp.json(content_type=None)
        return data.get("txHash") or data.get("hash") or data.get("id")

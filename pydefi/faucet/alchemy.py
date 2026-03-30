"""
Alchemy testnet faucet integration.

Requires an Alchemy API key with faucet access.  Set the ``ALCHEMY_API_KEY``
environment variable or pass the key directly to :class:`AlchemyFaucet`.

API reference: https://docs.alchemy.com/reference/faucet-api
"""

from __future__ import annotations

import aiohttp

from pydefi.exceptions import FaucetError
from pydefi.faucet.base import BaseFaucet
from pydefi.types import ChainId

# Alchemy faucet endpoint (v3 token send API).
_ALCHEMY_FAUCET_URL = "https://api.alchemy.com/api/sendToken"

# Mapping from EVM chain ID to the Alchemy network name accepted by the API.
_NETWORK_NAMES: dict[int, str] = {
    ChainId.SEPOLIA: "ETH_SEPOLIA",
}


class AlchemyFaucet(BaseFaucet):
    """Alchemy testnet faucet.

    Requests testnet ETH from Alchemy's faucet service via their REST API.
    Requires an Alchemy API key with faucet drip access.

    Args:
        api_key: Alchemy API key with faucet drip access.
        chain_id: Target chain (default: :attr:`~pydefi.types.ChainId.SEPOLIA`).
        api_base_url: Override the default Alchemy faucet endpoint URL.

    Raises:
        :class:`~pydefi.exceptions.FaucetError`: On construction when
            *chain_id* is not supported.

    Example::

        import os
        from pydefi.faucet import AlchemyFaucet

        faucet = AlchemyFaucet(api_key=os.environ["ALCHEMY_API_KEY"])
        tx_hash = await faucet.request("0xYourAddress")
    """

    def __init__(
        self,
        api_key: str,
        chain_id: int = ChainId.SEPOLIA,
        api_base_url: str = _ALCHEMY_FAUCET_URL,
    ) -> None:
        if chain_id not in _NETWORK_NAMES:
            raise FaucetError(f"AlchemyFaucet: unsupported chain ID {chain_id}")
        self._api_key = api_key
        self._chain_id = chain_id
        self._api_base = api_base_url.rstrip("/")

    @property
    def chain_id(self) -> int:
        return self._chain_id

    async def request(self, address: str) -> str | None:
        """Request testnet ETH from Alchemy's faucet.

        Args:
            address: The wallet address to fund.

        Returns:
            The transaction hash returned by Alchemy, or ``None``.

        Raises:
            :class:`~pydefi.exceptions.FaucetError`: If the faucet
                returns a non-2xx HTTP status code.
        """
        network = _NETWORK_NAMES[self._chain_id]
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "network": network,
            "toAddress": address,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._api_base,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status < 200 or resp.status >= 300:
                    text = await resp.text()
                    raise FaucetError(f"Alchemy faucet error ({resp.status}): {text[:200]}")
                data = await resp.json(content_type=None)
        return data.get("txHash") or data.get("requestId")

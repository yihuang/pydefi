"""
Abstract base class for testnet faucet integrations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from web3 import AsyncWeb3


class BaseFaucet(ABC):
    """Abstract base for testnet faucet integrations.

    Subclasses implement :meth:`request` to call a specific faucet service
    and deposit testnet ETH (or other tokens) to *address*.

    Usage example::

        faucet = AlchemyFaucet(api_key=os.environ["ALCHEMY_API_KEY"])
        await faucet.ensure_funded(w3, my_address, min_balance=10**16)
    """

    @property
    @abstractmethod
    def chain_id(self) -> int:
        """The EVM chain ID this faucet serves."""

    @abstractmethod
    async def request(self, address: str) -> str | None:
        """Request testnet funds for *address*.

        Args:
            address: The wallet address to fund.

        Returns:
            A transaction hash, request ID, or any confirmation string
            returned by the faucet, or ``None`` when the faucet does not
            return a meaningful identifier.

        Raises:
            :class:`~pydefi.exceptions.FaucetError`: When the faucet
                rejects the request or an HTTP error occurs.
        """

    async def ensure_funded(
        self,
        w3: AsyncWeb3,
        address: str,
        min_balance: int = 10**16,
        timeout: float = 60.0,
        poll_interval: float = 3.0,
    ) -> None:
        """Top-up *address* if its native balance is below *min_balance*.

        The method first checks the on-chain balance.  When funding is
        needed it calls :meth:`request` and then polls until the balance
        reaches *min_balance* or *timeout* seconds elapse.

        Args:
            w3: Async Web3 instance pointing at the target network.
            address: Wallet address to fund.
            min_balance: Minimum acceptable balance in wei
                (default: 0.01 ETH = ``10**16``).
            timeout: Maximum seconds to wait for funds to arrive
                (default: 60).
            poll_interval: Seconds between balance polls (default: 3).

        Raises:
            :class:`~pydefi.exceptions.FaucetError`: When the faucet
                request fails.
            :class:`TimeoutError`: When funds do not arrive within
                *timeout* seconds.
        """
        balance = await w3.eth.get_balance(address)
        if balance >= min_balance:
            return

        await self.request(address)

        # Poll until the balance reaches min_balance or we time out.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            balance = await w3.eth.get_balance(address)
            if balance >= min_balance:
                return
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out waiting for faucet funds: address={address} balance={balance} min_balance={min_balance}"
        )

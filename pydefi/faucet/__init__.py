"""
Testnet faucet utilities for pydefi.

Provides a uniform interface for requesting funds from public testnet
faucets.  Each faucet implementation exposes :meth:`~BaseFaucet.request`
to trigger a drip and :meth:`~BaseFaucet.ensure_funded` to top-up a wallet
only when its balance falls below a specified threshold.

Supported faucets
-----------------
- :class:`AlchemyFaucet` — Alchemy Sepolia faucet (``ALCHEMY_API_KEY``).
- :class:`QuickNodeFaucet` — QuickNode Sepolia faucet (``QUICKNODE_TOKEN``).

Quick start::

    import os
    from pydefi.faucet import AlchemyFaucet

    faucet = AlchemyFaucet(api_key=os.environ["ALCHEMY_API_KEY"])

    # Top-up only when balance < 0.01 ETH
    await faucet.ensure_funded(w3, wallet_address, min_balance=10**16)
"""

from pydefi.faucet.alchemy import AlchemyFaucet
from pydefi.faucet.base import BaseFaucet
from pydefi.faucet.quicknode import QuickNodeFaucet

__all__ = ["BaseFaucet", "AlchemyFaucet", "QuickNodeFaucet"]

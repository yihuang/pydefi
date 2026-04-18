"""Common pool base interfaces used across routing and execution planning."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePool(ABC):
    """Base class for pool descriptors used by RouteDAG actions."""

    pool_address: str
    protocol: str
    fee_bps: int

    @abstractmethod
    def amount_out(self, amount_in: int) -> int:
        """Return the output amount for a given input amount."""

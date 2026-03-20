"""
Custom exceptions for pydefi.
"""

from __future__ import annotations


class PydefiError(Exception):
    """Base class for all pydefi errors."""


class InsufficientLiquidityError(PydefiError):
    """Raised when a pool does not have enough liquidity for the requested trade."""


class NoRouteFoundError(PydefiError):
    """Raised when the pathfinder cannot find a route between two tokens."""


class AggregatorError(PydefiError):
    """Raised when a DEX aggregator API returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BridgeError(PydefiError):
    """Raised when a cross-chain bridge operation fails."""


class SlippageExceededError(PydefiError):
    """Raised when the actual output is below the minimum acceptable amount."""

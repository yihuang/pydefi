"""
Custom exceptions for pydifi.
"""

from __future__ import annotations


class PydifiError(Exception):
    """Base class for all pydifi errors."""


class InsufficientLiquidityError(PydifiError):
    """Raised when a pool does not have enough liquidity for the requested trade."""


class NoRouteFoundError(PydifiError):
    """Raised when the pathfinder cannot find a route between two tokens."""


class AggregatorError(PydifiError):
    """Raised when a DEX aggregator API returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BridgeError(PydifiError):
    """Raised when a cross-chain bridge operation fails."""


class SlippageExceededError(PydifiError):
    """Raised when the actual output is below the minimum acceptable amount."""

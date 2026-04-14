"""
Common types used throughout pydefi.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from enum import IntEnum
from typing import ClassVar


class ChainId(IntEnum):
    """Well-known chain IDs (EVM and non-EVM)."""

    ETHEREUM = 1
    OPTIMISM = 10
    BSC = 56
    POLYGON = 137
    UNICHAIN = 130
    WORLDCHAIN = 480
    BASE = 8453
    ARBITRUM = 42161
    AVALANCHE = 43114
    LINEA = 59144
    HYPERCORE = 1337  # Hyperliquid L1 (HyperCore); CCTP routes through HyperEVM (domain 19)
    HYPEREVM = 999
    BLAST = 81457
    SCROLL = 534352
    ZKSYNC = 324
    ZORA = 7777777
    SEPOLIA = 11155111
    # Solana – uses the Wormhole / cross-chain convention for its "chain ID"
    SOLANA = 1399811149


@dataclass(frozen=True)
class Token:
    """Represents an ERC-20 token (or native currency) on a specific chain.

    Attributes:
        chain_id: The chain this token lives on.
        address: Checksum address of the ERC-20 contract.  Use the sentinel
            value ``"0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"`` for the
            native gas token (ETH, MATIC, BNB …).
        symbol: Human-readable ticker symbol (e.g. ``"USDC"``).
        decimals: Token precision (default 18).
        name: Optional long-form name.
    """

    chain_id: int
    address: str
    symbol: str
    decimals: int = 18
    name: str | None = None

    # Sentinel for native currency (class variable, not an instance field)
    NATIVE_ADDRESS: ClassVar[str] = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"

    def is_native(self) -> bool:
        """Return ``True`` if this token represents the native gas currency."""
        return self.address.lower() == self.NATIVE_ADDRESS.lower()

    def __str__(self) -> str:
        return f"{self.symbol}({self.chain_id})"


@dataclass
class TokenAmount:
    """A specific quantity of a :class:`Token`.

    Attributes:
        token: The token.
        amount: Raw on-chain integer amount (in the token's smallest unit).
    """

    token: Token
    amount: int

    @property
    def human_amount(self) -> Decimal:
        """Return the amount expressed in whole tokens."""
        return Decimal(self.amount) / Decimal(10**self.token.decimals)

    @classmethod
    def from_human(cls, token: Token, amount: Decimal | float | str) -> "TokenAmount":
        """Create a :class:`TokenAmount` from a human-readable decimal value.

        The value is quantized to the token's precision using ``ROUND_DOWN``
        (i.e. any sub-unit remainder is truncated, never rounded up).
        """
        raw_decimal = Decimal(str(amount)) * Decimal(10**token.decimals)
        raw = int(raw_decimal.to_integral_value(rounding=ROUND_DOWN))
        return cls(token=token, amount=raw)

    def __repr__(self) -> str:
        return f"TokenAmount({self.human_amount} {self.token.symbol})"


@dataclass
class SwapStep:
    """One hop in a multi-hop swap route.

    Attributes:
        token_in: Input token for this hop.
        token_out: Output token for this hop.
        pool_address: Address of the liquidity pool used.
        protocol: Human-readable protocol name (e.g. ``"UniswapV2"``).
        fee: Swap fee in basis points (base 10000, e.g. ``30`` = 0.3%).
    """

    token_in: Token
    token_out: Token
    pool_address: str
    protocol: str
    fee: int = 30


@dataclass
class SwapRoute:
    """A complete swap route from *input token* to *output token*.

    A route may consist of one or more :class:`SwapStep` hops.

    Attributes:
        steps: Ordered list of swap hops.
        amount_in: Exact input amount.
        amount_out: Expected output amount (best estimate).
        price_impact: Estimated price impact as a fraction (e.g. ``0.005`` = 0.5%).
    """

    steps: list[SwapStep]
    amount_in: TokenAmount
    amount_out: TokenAmount
    price_impact: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("SwapRoute must contain at least one SwapStep")

    @property
    def token_in(self) -> Token:
        return self.steps[0].token_in

    @property
    def token_out(self) -> Token:
        return self.steps[-1].token_out

    def __repr__(self) -> str:
        path = " -> ".join([self.steps[0].token_in.symbol] + [s.token_out.symbol for s in self.steps])
        return f"SwapRoute({path}, in={self.amount_in.human_amount}, out={self.amount_out.human_amount})"


@dataclass
class SwapTransaction:
    """An encoded transaction ready to submit to the Uniswap Universal Router.

    Attributes:
        to: Target contract address (the Universal Router).
        data: ABI-encoded calldata for the ``execute`` call.
        value: Amount of native ETH (in wei) to attach to the transaction.
            Typically non-zero only when wrapping ETH as part of the swap.
    """

    to: str
    data: bytes
    value: int = 0


@dataclass
class BridgeQuote:
    """A quote for bridging tokens across chains.

    Attributes:
        token_in: Source token (on the source chain).
        token_out: Destination token (on the destination chain).
        amount_in: Amount being sent.
        amount_out: Expected amount received after fees.
        bridge_fee: Fee charged by the bridge (in *amount_in* token units).
        estimated_time_seconds: Estimated bridge completion time.
        protocol: Bridge protocol name.
    """

    token_in: Token
    token_out: Token
    amount_in: TokenAmount
    amount_out: TokenAmount
    bridge_fee: TokenAmount
    estimated_time_seconds: int
    protocol: str

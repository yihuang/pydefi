"""
Common types used throughout pydefi.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from enum import Enum, IntEnum
from typing import Any, TypeAlias

from hexbytes import HexBytes

MAX_BPS = 10_000

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Address as raw bytes (canonical intermediate representation), 20 bytes for EVM, 32 bytes for Solana.
#: Use ``decode_address(addr_str, chain_id)`` to convert from a human-readable string at the periphery.
Address = HexBytes
ZERO_ADDRESS = Address(b"\x00" * 20)
NATIVE_SENTINEL = Address(b"\xee" * 20)  # "0xEeee…", used by some protocols to represent native token

NATIVE_ADDRESSES: frozenset[Address] = frozenset(
    {
        ZERO_ADDRESS,
        NATIVE_SENTINEL,
    }
)
#: 32-byte hash or log topic as raw bytes (canonical intermediate representation).
#: Use ``HexBytes(hash_str)`` to convert a 0x-prefixed hex string to Hash.
Hash = HexBytes
ZERO_HASH = Hash(b"\x00" * 32)


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
        address: Token contract address as raw bytes (:class:`~hexbytes.HexBytes`,
            i.e. ``Address``).  For EVM chains this is 20 bytes; for Solana it is
            32 bytes (public key).  Use
            :func:`~pydefi._utils.decode_address` to convert a human-readable
            string to ``Address`` at the periphery before constructing a
            :class:`Token`.  Use
            :func:`~pydefi._utils.encode_address` to format for external APIs.
        symbol: Human-readable ticker symbol (e.g. ``"USDC"``).
        decimals: Token precision (default 18).
        name: Optional long-form name.
    """

    chain_id: int
    address: Address
    symbol: str
    decimals: int = 18
    name: str | None = None

    def is_native(self) -> bool:
        """Return ``True`` if this token represents the native gas currency."""
        return self.address in NATIVE_ADDRESSES

    def __str__(self) -> str:
        return f"{self.symbol}({self.chain_id})"

    @property
    def encoded_address(self) -> str:
        """Return the chain-specific string representation of the token's address."""
        from pydefi._utils import encode_address  # lazy import to avoid circular dependency

        return encode_address(self.address, self.chain_id)


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
        pool_address: Address of the liquidity pool used, or ``None`` when the
            pool identity is unavailable (e.g. aggregator routes that do not
            expose individual pool addresses).
        protocol: Human-readable protocol name (e.g. ``"UniswapV2"``).
        fee: Swap fee in basis points (base 10000, e.g. ``30`` = 0.3%).
    """

    token_in: Token
    token_out: Token
    pool_address: Address | None
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
    dag: "RouteDAG | None" = None

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


class SwapProtocol(str, Enum):
    """Supported DEX protocols for :class:`SwapHop`.

    Both values use **direct pool/pair calls** — no router contract is involved.
    """

    UNISWAP_V2 = "uniswap_v2"
    """Uniswap V2-compatible pair: pre-transfer tokenIn, then call ``pair.swap()``.

    On-chain amountOut is computed from ``pair.getReserves()`` using the
    constant-product formula, so no off-chain quote is required.
    """

    UNISWAP_V3 = "uniswap_v3"
    """Uniswap V3-compatible pool: call ``pool.swap()`` directly.

    The pool fires a flash-swap callback (``uniswapV3SwapCallback`` or a
    compatible variant) which ``DeFiVM.fallback()`` handles automatically.
    """


class BasePool(ABC):
    """Base class for pool descriptors used by RouteDAG actions."""

    pool_address: Address
    protocol: SwapProtocol
    fee_bps: int

    def zero_for_one(self, token_out: Address) -> bool:
        raise NotImplementedError("BasePool.zero_for_one() must be implemented by subclasses")


@dataclass(frozen=True)
class RouteSwap:
    """A single swap edge in a route DAG."""

    token_out: Token
    pool: BasePool

    def zero_for_one(self) -> bool:
        return self.pool.zero_for_one(self.token_out.address)


@dataclass(frozen=True)
class RouteSplitLeg:
    """One branch in a split section of a route DAG."""

    fraction_bps: int
    actions: tuple[RouteAction, ...]


@dataclass(frozen=True)
class RouteSplit:
    """A split/merge section of a route DAG."""

    legs: tuple[RouteSplitLeg, ...]
    token_out: Token


RouteAction: TypeAlias = RouteSwap | RouteSplit


@dataclass
class _RouteSplitLegBuilder:
    fraction_bps: int
    actions: list[RouteAction] = field(default_factory=list)
    current_token: Token | None = None


@dataclass
class _RouteSplitBuilder:
    origin_token: Token
    legs: list[_RouteSplitLegBuilder] = field(default_factory=list)
    active_leg: _RouteSplitLegBuilder | None = None

    def start_leg(self, fraction_bps: int) -> None:
        leg = _RouteSplitLegBuilder(fraction_bps=fraction_bps, current_token=self.origin_token)
        self.legs.append(leg)
        self.active_leg = leg


@dataclass
class RouteDAG:
    """Fluent builder for split/merge swap routes represented as a DAG."""

    token_in: Token | None = None
    actions: list[RouteAction] = field(default_factory=list)
    _current_token: Token | None = None
    _split_stack: list[_RouteSplitBuilder] = field(default_factory=list)

    def from_token(self, token: Token) -> "RouteDAG":
        if self.token_in is not None:
            raise ValueError("RouteDAG.from_token() can only be called once")
        self.token_in = token
        self._current_token = token
        return self

    def swap(self, token_out: Token, pool: BasePool) -> "RouteDAG":
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before swap()")
        self._current_actions().append(RouteSwap(token_out=token_out, pool=pool))
        self._set_current_token(token_out)
        return self

    def split(self) -> "RouteDAG":
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before split()")
        if not self._split_stack:
            origin_token = self._current_token
        else:
            parent = self._split_stack[-1]
            if parent.active_leg is None or parent.active_leg.current_token is None:
                raise ValueError("leg() must be called before nested split()")
            origin_token = parent.active_leg.current_token

        if origin_token is None:
            raise ValueError("RouteDAG.from_token() must be called before split()")
        self._split_stack.append(_RouteSplitBuilder(origin_token=origin_token))
        return self

    def leg(self, fraction_bps: int) -> "RouteDAG":
        if not self._split_stack:
            raise ValueError("RouteDAG.leg() must be called inside split()")
        if not (0 < fraction_bps <= MAX_BPS):
            raise ValueError(f"leg fraction_bps must be in (0, {MAX_BPS}], got {fraction_bps}")
        self._split_stack[-1].start_leg(fraction_bps)
        return self

    def merge(self) -> "RouteDAG":
        if not self._split_stack:
            raise ValueError("RouteDAG.merge() called without an active split")

        split_ctx = self._split_stack.pop()
        total_bps = sum(leg.fraction_bps for leg in split_ctx.legs)
        if total_bps != MAX_BPS:
            raise ValueError(f"sum of split leg fraction_bps must be {MAX_BPS}, got {total_bps}")

        if any(not leg.actions for leg in split_ctx.legs):
            raise ValueError("each split leg must contain at least one swap() before merge()")

        end_tokens = {leg.current_token for leg in split_ctx.legs}
        if len(end_tokens) != 1:
            raise ValueError("all split legs must end at the same token before merge()")

        merged_token = next(iter(end_tokens))
        split_action = RouteSplit(
            legs=tuple(
                RouteSplitLeg(fraction_bps=leg.fraction_bps, actions=tuple(leg.actions)) for leg in split_ctx.legs
            ),
            token_out=merged_token,
        )

        if self._split_stack:
            parent = self._split_stack[-1]
            if parent.active_leg is None:
                raise ValueError("internal RouteDAG error: missing parent split leg")
            parent.active_leg.actions.append(split_action)
            parent.active_leg.current_token = merged_token
        else:
            self.actions.append(split_action)
            self._current_token = merged_token
        return self

    def to_dict(self) -> dict[str, Any]:
        if self._split_stack:
            raise ValueError("RouteDAG has unmerged split legs")
        if self.token_in is None:
            raise ValueError("RouteDAG.from_token() must be called before serialization")
        return {"token_in": self.token_in, "actions": _freeze_actions(self.actions)}

    def _current_actions(self) -> list[RouteAction]:
        if not self._split_stack:
            return self.actions
        split_ctx = self._split_stack[-1]
        if split_ctx.active_leg is None:
            raise ValueError("leg() must be called before swap() inside split()")
        return split_ctx.active_leg.actions

    def _set_current_token(self, token: Token) -> None:
        if not self._split_stack:
            self._current_token = token
            return
        split_ctx = self._split_stack[-1]
        if split_ctx.active_leg is None:
            raise ValueError("leg() must be called before swap() inside split()")
        split_ctx.active_leg.current_token = token


def _freeze_actions(actions: Sequence[RouteAction]) -> tuple[RouteAction, ...]:
    frozen: list[RouteAction] = []
    for action in actions:
        if isinstance(action, RouteSwap):
            frozen.append(action)
            continue
        frozen.append(
            RouteSplit(
                legs=tuple(
                    RouteSplitLeg(
                        fraction_bps=leg.fraction_bps,
                        actions=_freeze_actions(leg.actions),
                    )
                    for leg in action.legs
                ),
                token_out=action.token_out,
            )
        )
    return tuple(frozen)


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

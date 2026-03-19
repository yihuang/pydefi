"""
Curve Finance stableswap pool integration.

Curve uses a hybrid constant-sum / constant-product formula optimised for
pegged assets (stablecoins, wrapped tokens).  This module wraps the
``ICurvePool`` interface (``exchange``, ``get_dy``) via
:class:`~eth_contract.Contract`.
"""

from __future__ import annotations

from decimal import Decimal

from eth_contract import Contract
from web3 import AsyncWeb3

from pydifi.amm.base import BaseAMM
from pydifi.exceptions import InsufficientLiquidityError
from pydifi.types import SwapRoute, SwapStep, Token, TokenAmount

# ---------------------------------------------------------------------------
# ABI fragments
# ---------------------------------------------------------------------------

_CURVE_POOL_ABI = [
    "function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256)",
    "function get_dy_underlying(int128 i, int128 j, uint256 dx) external view returns (uint256)",
    "function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
    "function exchange_underlying(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256)",
    "function coins(uint256 i) external view returns (address)",
    "function balances(uint256 i) external view returns (uint256)",
    "function A() external view returns (uint256)",
    "function fee() external view returns (uint256)",
]

_CURVE_REGISTRY_ABI = [
    "function find_pool_for_coins(address from, address to) external view returns (address)",
    "function find_pool_for_coins(address from, address to, uint256 i) external view returns (address)",
    "function get_coin_indices(address pool, address from, address to) external view returns (int128, int128, bool)",
    "function get_best_rate(address from, address to, uint256 amount) external view returns (address, uint256)",
]


class CurvePool(BaseAMM):
    """Curve Finance stableswap pool integration.

    Each instance is bound to a *single* Curve pool.  Use
    :class:`CurveRegistry` (or query the registry on-chain) to discover the
    correct pool for a given token pair.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance.
        pool_address: Address of the Curve pool contract.
        tokens: Ordered list of tokens in the pool (matching the pool's
            ``coins`` array).
        use_underlying: If ``True`` use the ``_underlying`` variants of
            ``get_dy`` / ``exchange`` (for lending pools like Compound).
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        pool_address: str,
        tokens: list[Token],
        use_underlying: bool = False,
    ) -> None:
        super().__init__(w3, pool_address)
        self._tokens = tokens
        self._use_underlying = use_underlying
        self._pool = Contract.from_abi(_CURVE_POOL_ABI, to=pool_address)

    @property
    def protocol_name(self) -> str:
        return "Curve"

    @property
    def tokens(self) -> list[Token]:
        """Tokens registered in this pool (ordered by pool index)."""
        return list(self._tokens)

    def _coin_index(self, token: Token) -> int:
        """Return the pool index for *token*."""
        for i, t in enumerate(self._tokens):
            if t.address.lower() == token.address.lower():
                return i
        raise ValueError(f"Token {token.symbol} not found in this Curve pool")

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    async def get_dy(
        self, token_in: Token, token_out: Token, amount_in: int
    ) -> int:
        """Call ``get_dy`` to simulate a swap.

        Args:
            token_in: Input token.
            token_out: Output token.
            amount_in: Raw input amount.

        Returns:
            Expected raw output amount.

        Raises:
            :class:`~pydifi.exceptions.InsufficientLiquidityError`: On revert.
        """
        i = self._coin_index(token_in)
        j = self._coin_index(token_out)
        fn_name = "get_dy_underlying" if self._use_underlying else "get_dy"
        try:
            amount_out: int = await getattr(self._pool.fns, fn_name)(i, j, amount_in).call(self.w3)
        except Exception as exc:
            raise InsufficientLiquidityError(f"get_dy failed: {exc}") from exc
        return amount_out

    async def get_amounts_out(
        self, amount_in: TokenAmount, path: list[Token]
    ) -> list[TokenAmount]:
        """Simulate a swap along *path* through this pool.

        Curve pools typically contain 2-4 tokens.  Multi-hop routes require
        chaining calls across multiple pools.

        Args:
            amount_in: Input amount.
            path: Must be exactly two tokens from this pool.

        Returns:
            ``[amount_in, amount_out]``
        """
        if len(path) != 2:
            raise ValueError("CurvePool.get_amounts_out requires a 2-token path")
        amount_out_raw = await self.get_dy(path[0], path[1], amount_in.amount)
        return [amount_in, TokenAmount(token=path[1], amount=amount_out_raw)]

    async def get_amounts_in(
        self, amount_out: TokenAmount, path: list[Token]
    ) -> list[TokenAmount]:
        """Estimate required input to obtain *amount_out* (approximation).

        Curve does not expose a native ``get_dx`` on-chain, so this uses a
        binary-search approximation.

        Args:
            amount_out: Desired output.
            path: Must be exactly two tokens.

        Returns:
            ``[amount_in, amount_out]``
        """
        if len(path) != 2:
            raise ValueError("CurvePool.get_amounts_in requires a 2-token path")

        target = amount_out.amount
        # Start with a 1:1 estimate and double until we exceed the target
        lo, hi = 1, target * 2
        for _ in range(64):
            mid = (lo + hi) // 2
            dy = await self.get_dy(path[0], path[1], mid)
            if dy >= target:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1:
                break

        return [TokenAmount(token=path[0], amount=hi), amount_out]

    # ------------------------------------------------------------------
    # Route builder
    # ------------------------------------------------------------------

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
    ) -> SwapRoute:
        """Build a direct swap route through this Curve pool.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (must be in this pool).
            slippage_bps: Slippage tolerance in basis points.

        Returns:
            A :class:`~pydifi.types.SwapRoute`.
        """
        amount_out_raw = await self.get_dy(amount_in.token, token_out, amount_in.amount)
        amount_out = TokenAmount(token=token_out, amount=amount_out_raw)

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=self.router_address,
            protocol=self.protocol_name,
            fee=400,  # Curve base fee is 0.04% (4 bps = 400 hundredths of a bp)
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=Decimal(0),
        )

    def get_registry_contract(self, registry_address: str) -> Contract:
        """Return a :class:`~eth_contract.Contract` bound to a Curve registry."""
        return Contract.from_abi(_CURVE_REGISTRY_ABI, to=registry_address)

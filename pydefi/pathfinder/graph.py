"""
Graph representation of the DEX liquidity landscape.

Each node in the graph is a :class:`~pydefi.types.Token`.
Each directed edge represents a liquidity pool that can swap *token_in* →
*token_out* and carries the pool's reserve information for price estimation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, ClassVar, Iterator

from pydefi.types import Token


@dataclass
class PoolEdge:
    """A directed edge in the liquidity pool graph.

    Represents one *direction* of a swap through a single pool.  For a
    bidirectional pool (A ↔ B), two ``PoolEdge`` objects are added to the
    graph — one for A→B and one for B→A.

    Attributes:
        token_in: Input token for this direction.
        token_out: Output token for this direction.
        pool_address: On-chain pool / pair address.
        protocol: Human-readable protocol name.
        reserve_in: Reserve of token_in in the pool (raw units).
        reserve_out: Reserve of token_out in the pool (raw units).
        fee_bps: Swap fee in basis points (e.g. ``30`` = 0.3%).
        extra: Optional extra data (e.g. pool type, tick spacing).
    """

    token_in: Token
    token_out: Token
    pool_address: str
    protocol: str
    reserve_in: int = 0
    reserve_out: int = 0
    fee_bps: int = 30
    extra: dict = field(default_factory=dict)

    @property
    def spot_price(self) -> Decimal:
        """Spot price of token_out denominated in token_in.

        Returns:
            ``Decimal(0)`` if reserve_in is zero.
        """
        if self.reserve_in == 0:
            return Decimal(0)
        adj_in = Decimal(self.reserve_in) / Decimal(10**self.token_in.decimals)
        adj_out = Decimal(self.reserve_out) / Decimal(10**self.token_out.decimals)
        return adj_out / adj_in

    def amount_out(self, amount_in: int) -> int:
        """Estimate the output amount for *amount_in* using the constant-product formula.

        Args:
            amount_in: Raw input amount.

        Returns:
            Estimated raw output amount (0 if reserves are unavailable).
        """
        if self.reserve_in == 0 or self.reserve_out == 0:
            return 0
        fee_factor = 10_000 - self.fee_bps
        amount_in_with_fee = amount_in * fee_factor
        numerator = amount_in_with_fee * self.reserve_out
        denominator = self.reserve_in * 10_000 + amount_in_with_fee
        if denominator == 0:
            return 0
        return numerator // denominator

    def estimate_price_impact(self, amount_in: int) -> Decimal:
        """Estimate the price impact of swapping *amount_in* through this pool.

        For V2-style pools with known reserves, price impact is approximated as
        ``amount_in / (reserve_in + amount_in)`` — the fraction of pool depth
        consumed by the swap.

        Subclasses should override this method to implement pool-type-specific
        impact estimation (e.g. :class:`V3PoolEdge` uses virtual reserves
        derived from ``sqrtPriceX96`` / ``liquidity``).

        Args:
            amount_in: Raw input amount (before fees).

        Returns:
            Estimated price impact in ``[0, 1]``, or ``Decimal('NaN')`` if the
            pool does not expose enough information for an estimate.
        """
        if self.reserve_in > 0:
            return Decimal(amount_in) / Decimal(self.reserve_in + amount_in)
        return Decimal("NaN")

    def log_weight(self, amount_in: int = 1) -> float:
        """Return a log-space weight suitable for shortest-path algorithms.

        The weight is ``-log(amount_out / amount_in)`` so that maximising
        output (best rate) is equivalent to finding the minimum weight path.

        Args:
            amount_in: Reference input amount (default 1 token in smallest
                unit).

        Returns:
            Float weight; returns ``float('inf')`` if the pool has no
            liquidity.
        """

        out = self.amount_out(amount_in)
        if out == 0:
            return float("inf")
        ratio = out / amount_in
        if ratio <= 0:
            return float("inf")
        return -math.log(ratio)

    def effective_log_weight(
        self,
        amount_in: int = 1,
        gas_price_gwei: float | None = None,
        *,
        native_token_price_usd: float | None = None,
        token_out_price_usd: float | None = None,
        estimated_gas_units: int | None = None,
        default_gas_units: int = 150_000,
    ) -> float:
        """Gas-aware variant of :meth:`log_weight`.

        Computes ``effective_out = amount_out - gas_cost_out`` and returns
        ``-log(effective_out / amount_in)``. If gas/price inputs are missing
        or invalid, falls back to :meth:`log_weight`.

        Returns ``float('inf')`` when the hop is invalid or gas cost consumes
        all output.
        """

        if amount_in <= 0:
            return float("inf")

        out = self.amount_out(amount_in)
        if out <= 0:
            return float("inf")

        if gas_price_gwei is None or gas_price_gwei <= 0:
            return self.log_weight(amount_in)

        if (
            native_token_price_usd is None
            or native_token_price_usd <= 0
            or token_out_price_usd is None
            or token_out_price_usd <= 0
        ):
            return self.log_weight(amount_in)

        gas_units = estimated_gas_units
        if gas_units is None:
            gas_units = int(self.extra.get("estimated_gas", default_gas_units))
        if gas_units <= 0:
            return self.log_weight(amount_in)

        gas_cost_native = gas_units * gas_price_gwei * 1e-9
        gas_cost_out_tokens = gas_cost_native * native_token_price_usd / token_out_price_usd
        gas_cost_out_raw = int(gas_cost_out_tokens * (10**self.token_out.decimals))

        effective_out = out - gas_cost_out_raw
        if effective_out <= 0:
            return float("inf")

        ratio = effective_out / amount_in
        if ratio <= 0:
            return float("inf")
        return -math.log(ratio)


@dataclass
class V3PoolEdge(PoolEdge):
    """A directed edge representing a Uniswap V3 concentrated liquidity pool.

    Uses the V3 concentrated liquidity formula (``sqrtPriceX96`` and
    ``liquidity``) for price and amount estimation instead of the
    constant-product reserve model used by :class:`PoolEdge`.

    Attributes:
        sqrt_price_x96: Current sqrt price of the pool (from ``slot0``),
            in Q96 fixed-point format (i.e. ``sqrtPrice * 2^96``).
        liquidity: Current active liquidity of the pool.
        is_token0_in: ``True`` if ``token_in`` is ``token0`` of the V3 pool.
    """

    sqrt_price_x96: int = 0
    liquidity: int = 0
    is_token0_in: bool = True

    _Q96: ClassVar[int] = 2**96

    @property
    def spot_price(self) -> Decimal:
        """Spot price of ``token_out`` denominated in ``token_in``, adjusted
        for token decimals, derived from ``sqrtPriceX96``.
        """
        if self.sqrt_price_x96 == 0:
            return Decimal(0)
        Q96 = self._Q96
        sqrtP = Decimal(self.sqrt_price_x96) / Decimal(Q96)
        # price_raw = token1 per token0 in the pool's raw (smallest) units,
        # without any decimal adjustment for token precision.
        price_raw = sqrtP**2
        # adj converts from raw units to human-readable units
        adj = Decimal(10**self.token_in.decimals) / Decimal(10**self.token_out.decimals)
        if self.is_token0_in:
            return price_raw * adj
        else:
            if price_raw == 0:
                return Decimal(0)
            return (Decimal(1) / price_raw) * adj

    def amount_out(self, amount_in: int) -> int:
        """Estimate output using the V3 concentrated liquidity formula.

        This uses the exact Uniswap V3 ``sqrtPrice`` math for a **single-tick
        approximation** — it assumes all liquidity ``L`` is available at the
        current price and that no tick boundaries are crossed during the swap.

        **Accuracy vs trade size:**  For swaps that are small relative to
        ``L * sqrtP / Q96`` (roughly the depth of the current tick range), the
        estimate is very close to the true on-chain result.  For large swaps
        that would move the price across one or more tick boundaries, the
        formula overestimates the output because it ignores the reduced
        liquidity (and potentially zero liquidity) beyond the current tick.
        In practice this is fine for **pathfinding** — the router is comparing
        routes to pick the best one, not producing the definitive execution
        quote.  For the precise execution quote, always call
        :meth:`~pydefi.amm.uniswap_v3.UniswapV3.quote_exact_input_single`
        (backed by the on-chain QuoterV2 which simulates full tick traversal).

        Args:
            amount_in: Raw input amount.

        Returns:
            Estimated raw output amount (0 if pool state is unavailable).
        """
        if self.sqrt_price_x96 == 0 or self.liquidity == 0 or amount_in <= 0:
            return 0

        Q96 = self._Q96
        sqrtP = self.sqrt_price_x96
        L = self.liquidity

        # Deduct fee from input (fee_bps is in basis points, e.g. 30 = 0.3%)
        amount_in_net = amount_in * (10_000 - self.fee_bps) // 10_000
        if amount_in_net <= 0:
            return 0

        if self.is_token0_in:
            # Swapping token0 → token1 (price decreases):
            # new_sqrtP = sqrtP * L * Q96 / (L * Q96 + amount_in_net * sqrtP)
            denom = L * Q96 + amount_in_net * sqrtP
            if denom <= 0:
                return 0
            new_sqrtP = sqrtP * L * Q96 // denom
            # Δy = L * (sqrtP - new_sqrtP) / Q96
            if sqrtP <= new_sqrtP:
                return 0
            return L * (sqrtP - new_sqrtP) // Q96
        else:
            # Swapping token1 → token0 (price increases):
            # new_sqrtP = sqrtP + amount_in_net * Q96 / L
            new_sqrtP = sqrtP + amount_in_net * Q96 // L
            # Δx = L * Q96 * (new_sqrtP - sqrtP) / (sqrtP * new_sqrtP)
            if new_sqrtP <= sqrtP:
                return 0
            numerator = L * Q96 * (new_sqrtP - sqrtP)
            denominator = sqrtP * new_sqrtP
            if denominator == 0:
                return 0
            return numerator // denominator

    def estimate_price_impact(self, amount_in: int) -> Decimal:
        """Estimate price impact using V3 current-tick virtual reserves.

        Computes the virtual reserve (liquidity depth) at the current tick as
        a proxy for ``reserve_in`` in the impact formula
        ``amount_in / (depth + amount_in)``:

        * token0 → token1 (``is_token0_in=True``):  depth ≈ ``L × Q96 / sqrtP``
        * token1 → token0 (``is_token0_in=False``): depth ≈ ``L × sqrtP / Q96``

        These expressions correspond to the virtual reserves of the input token
        in Uniswap V3's concentrated liquidity math at the current tick.
        The pre-fee ``amount_in`` is used for the ratio (price impact measures
        depth consumption before fees are applied).

        Args:
            amount_in: Raw input amount (before fees).

        Returns:
            Estimated price impact in ``[0, 1]``, or ``Decimal('NaN')`` if the
            pool state is unavailable.
        """
        if amount_in < 0:
            return Decimal("NaN")
        if amount_in == 0:
            return Decimal(0)
        if self.sqrt_price_x96 == 0 or self.liquidity == 0:
            return Decimal("NaN")
        Q96 = self._Q96
        if self.is_token0_in:
            # Virtual depth of token0 at current tick ≈ L * Q96 / sqrtP
            depth = self.liquidity * Q96 // self.sqrt_price_x96
        else:
            # Virtual depth of token1 at current tick ≈ L * sqrtP / Q96
            depth = self.liquidity * self.sqrt_price_x96 // Q96
        if depth <= 0:
            return Decimal("NaN")
        return Decimal(amount_in) / Decimal(depth + amount_in)


class PoolGraph:
    """A directed multi-graph of token → token liquidity pools.

    Use :meth:`add_pool` to register pools, then :meth:`edges_from` to
    iterate over all pools that accept a given input token.

    Example::

        graph = PoolGraph()
        graph.add_pool(edge_usdc_to_eth)
        graph.add_pool(edge_eth_to_usdc)
        for edge in graph.edges_from(usdc):
            print(edge.token_out, edge.spot_price)
    """

    def __init__(self) -> None:
        # adjacency list: token_in_address -> list[PoolEdge]
        self._adj: defaultdict[str, list[PoolEdge]] = defaultdict(list)
        self._tokens: dict[str, Token] = {}

    def find_best_route_gas_aware(
        self,
        start: Token,
        end: Token,
        *,
        amount_in: int,
        weight_fn: Callable[[PoolEdge, int], float],
        max_hops: int = 4,
    ) -> list[PoolEdge]:
        """Return the lowest-weight path under a hop limit.

        Uses hop-bounded relaxation and applies ``weight_fn(edge, current_amount)``
        during traversal (for example, gas-aware ``effective_log_weight``).
        Returns the best edge path, or an empty list when no route is found.
        """
        if amount_in <= 0:
            return []

        src_addr = start.address.lower()
        dst_addr = end.address.lower()
        if src_addr == dst_addr:
            raise ValueError("token_in and token_out must be different")

        # state -> (cumulative_weight, current_amount, path)
        best: dict[tuple[str, int], tuple[float, int, list[PoolEdge]]] = {(src_addr, 0): (0.0, amount_in, [])}

        for hop in range(max_hops):
            current_states = [(k, v) for k, v in best.items() if k[1] == hop]
            for (token_addr, _), (cur_weight, cur_amount, path) in current_states:
                visited_tokens: set[str] = {e.token_in.address.lower() for e in path}
                visited_tokens.add(token_addr)

                token: Token = path[-1].token_out if path else start

                for edge in self.edges_from(token):
                    next_addr = edge.token_out.address.lower()
                    if next_addr in visited_tokens:
                        continue

                    next_amount = edge.amount_out(cur_amount)
                    if next_amount <= 0:
                        continue

                    edge_weight = weight_fn(edge, cur_amount)
                    next_weight = cur_weight + edge_weight

                    next_key = (next_addr, hop + 1)
                    existing = best.get(next_key)
                    if (
                        existing is None
                        or next_weight < existing[0]
                        or (next_weight == existing[0] and next_amount > existing[1])
                    ):
                        best[next_key] = (next_weight, next_amount, path + [edge])

        best_result: tuple[float, int, list[PoolEdge]] | None = None
        for h in range(1, max_hops + 1):
            entry = best.get((dst_addr, h))
            if entry is None:
                continue
            if best_result is None:
                best_result = entry
                continue
            if entry[0] < best_result[0] or (entry[0] == best_result[0] and entry[1] > best_result[1]):
                best_result = entry

        return [] if best_result is None else best_result[2]

    def add_pool(self, edge: PoolEdge) -> None:
        """Register a pool edge in the graph.

        Args:
            edge: The :class:`PoolEdge` to add.
        """
        key = edge.token_in.address.lower()
        self._adj[key].append(edge)
        self._tokens[edge.token_in.address.lower()] = edge.token_in
        self._tokens[edge.token_out.address.lower()] = edge.token_out

    def add_bidirectional_pool(
        self,
        token_a: Token,
        token_b: Token,
        pool_address: str,
        protocol: str,
        reserve_a: int = 0,
        reserve_b: int = 0,
        fee_bps: int = 30,
        **extra,
    ) -> None:
        """Add both directions of a symmetric liquidity pool.

        Args:
            token_a: First token.
            token_b: Second token.
            pool_address: Pool contract address.
            protocol: Protocol name.
            reserve_a: Reserve of token_a.
            reserve_b: Reserve of token_b.
            fee_bps: Swap fee in basis points.
            **extra: Extra metadata stored in ``PoolEdge.extra``.
        """
        self.add_pool(
            PoolEdge(
                token_in=token_a,
                token_out=token_b,
                pool_address=pool_address,
                protocol=protocol,
                reserve_in=reserve_a,
                reserve_out=reserve_b,
                fee_bps=fee_bps,
                extra=dict(extra),
            )
        )
        self.add_pool(
            PoolEdge(
                token_in=token_b,
                token_out=token_a,
                pool_address=pool_address,
                protocol=protocol,
                reserve_in=reserve_b,
                reserve_out=reserve_a,
                fee_bps=fee_bps,
                extra=dict(extra),
            )
        )

    def edges_from(self, token: Token) -> list[PoolEdge]:
        """Return all edges that depart from *token*.

        Args:
            token: Source token.

        Returns:
            List of :class:`PoolEdge` objects.
        """
        return list(self._adj[token.address.lower()])

    def edges_to(self, token: Token) -> list[PoolEdge]:
        """Return all edges that arrive at *token*.

        Args:
            token: Destination token.

        Returns:
            List of :class:`PoolEdge` objects.
        """
        result: list[PoolEdge] = []
        for edges in self._adj.values():
            for e in edges:
                if e.token_out.address.lower() == token.address.lower():
                    result.append(e)
        return result

    @property
    def tokens(self) -> list[Token]:
        """All tokens present in the graph."""
        return list(self._tokens.values())

    def __len__(self) -> int:
        return sum(len(edges) for edges in self._adj.values())

    def __iter__(self) -> Iterator[PoolEdge]:
        for edges in self._adj.values():
            yield from edges

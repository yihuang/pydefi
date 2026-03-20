"""
DEX pathfinding router.

Implements hop-bounded dynamic programming over a
:class:`~pydifi.pathfinder.graph.PoolGraph` to find the optimal swap route
between any two tokens.

The algorithm maximises the output amount by tracking the best raw output at
each ``(token, hop_depth)`` state.  Because ``edge.amount_out`` is
monotonically increasing, the state with the largest raw output at any
intermediate token always propagates to the largest output at subsequent hops,
making simple DP relaxation both correct and safe.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydifi.exceptions import NoRouteFoundError
from pydifi.pathfinder.graph import PoolEdge, PoolGraph
from pydifi.types import SwapRoute, SwapStep, Token, TokenAmount


class Router:
    """Optimal swap route finder over a :class:`~pydifi.pathfinder.graph.PoolGraph`.

    Uses hop-bounded dynamic programming to find the multi-hop route that
    maximises the output amount.

    Args:
        graph: The pool graph to search.
        max_hops: Maximum number of swap hops allowed (default ``3``).
    """

    def __init__(self, graph: PoolGraph, max_hops: int = 3) -> None:
        self.graph = graph
        self.max_hops = max_hops

    def find_best_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
    ) -> SwapRoute:
        """Find the route that maximises the output amount.

        Uses hop-bounded DP relaxation:

        * State: ``(token_address, hop_count)``
        * Value: maximum raw output amount of *token_address* reachable in
          exactly *hop_count* hops from the source token.
        * At each depth, every reachable state is expanded once; if two paths
          reach the same ``(token, hop)`` state, only the one with the larger
          raw output is kept.  This is correct because ``edge.amount_out`` is
          monotonically increasing: a higher intermediate amount always yields
          at least as much from any onward edge.

        This approach avoids the instability of Dijkstra-style log-weight
        pruning when edge weights span different token decimal scales (e.g.
        WETH 18 dec → USDC 6 dec produces a large negative log-ratio that can
        misguide Dijkstra's early-termination heuristic).

        Args:
            amount_in: Exact input token and amount.
            token_out: Desired output token.

        Returns:
            The best :class:`~pydifi.types.SwapRoute` found.

        Raises:
            :class:`~pydifi.exceptions.NoRouteFoundError`: If no path exists
                between the two tokens within ``max_hops``.
        """
        src = amount_in.token
        dst_addr = token_out.address.lower()

        if src.address.lower() == dst_addr:
            raise ValueError("token_in and token_out must be different")

        # DP table: best[(token_addr, hops)] = (max_raw_output, path_of_edges)
        # Seed with the source state at hop depth 0.
        best: dict[tuple[str, int], tuple[int, list[PoolEdge]]] = {
            (src.address.lower(), 0): (amount_in.amount, [])
        }

        for hop in range(self.max_hops):
            # Snapshot all states at the current depth to avoid processing
            # states we add during this iteration.
            current_states = [(k, v) for k, v in best.items() if k[1] == hop]

            for (token_addr, _), (current_amount, path) in current_states:
                # Tokens already on this path (cycle prevention).
                visited_tokens: set[str] = {e.token_in.address.lower() for e in path}
                visited_tokens.add(token_addr)

                # Retrieve the Token object: use the last edge's output token,
                # or the source token for the initial state.
                token: Token = path[-1].token_out if path else src

                for edge in self.graph.edges_from(token):
                    next_addr = edge.token_out.address.lower()
                    if next_addr in visited_tokens:
                        continue

                    next_amount = edge.amount_out(current_amount)
                    if next_amount <= 0:
                        continue

                    next_key = (next_addr, hop + 1)
                    existing = best.get(next_key)
                    if existing is None or next_amount > existing[0]:
                        best[next_key] = (next_amount, path + [edge])

        # Collect the best path to the destination across all hop depths.
        best_result: Optional[tuple[int, list[PoolEdge]]] = None
        for h in range(1, self.max_hops + 1):
            entry = best.get((dst_addr, h))
            if entry is not None:
                if best_result is None or entry[0] > best_result[0]:
                    best_result = entry

        if best_result is None:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol} "
                f"within {self.max_hops} hops"
            )

        final_amount, final_path = best_result
        steps = [
            SwapStep(
                token_in=edge.token_in,
                token_out=edge.token_out,
                pool_address=edge.pool_address,
                protocol=edge.protocol,
                fee=edge.fee_bps * 100,
            )
            for edge in final_path
        ]
        return SwapRoute(
            steps=steps,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=final_amount),
            price_impact=self._estimate_price_impact(final_path, amount_in.amount),
        )

    def find_all_routes(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        top_k: int = 5,
    ) -> list[SwapRoute]:
        """Find the top-*k* routes by output amount.

        Uses DFS with dominance pruning to enumerate multi-hop routes.
        Pruning rule: when two paths reach the same ``(token, depth)`` state,
        only the one with the larger intermediate amount is explored further.
        Because ``edge.amount_out`` is monotonically increasing, any
        continuation from the dominated state produces ≤ output compared to
        the corresponding continuation from the better state.

        .. note::
            This method is designed for **small, curated graphs** (typically
            fewer than ~20 tokens and ~50 edges).  Even with dominance pruning,
            DFS can grow combinatorially on dense graphs.  For large graphs or
            production routing, prefer :meth:`find_best_route`, which uses
            bounded DP and scales more predictably.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            top_k: Maximum number of routes to return.

        Returns:
            List of :class:`~pydifi.types.SwapRoute` objects, sorted by
            output amount descending.

        Raises:
            :class:`~pydifi.exceptions.NoRouteFoundError`: If no routes exist.
            :class:`ValueError`: If ``token_in`` and ``token_out`` are the same.
        """
        src = amount_in.token
        dst_addr = token_out.address.lower()
        routes: list[SwapRoute] = []

        if src.address.lower() == dst_addr:
            raise ValueError("token_in and token_out must be different")

        # Dominance pruning: best raw output seen for each (token_addr, hop_depth).
        # A path is pruned when it arrives at a state with a lower amount than
        # one already explored — any onward route will be dominated.
        best_at: dict[tuple[str, int], int] = {}

        def dfs(
            current_token: Token,
            current_amount: int,
            path: list[PoolEdge],
            visited_tokens: set[str],
        ) -> None:
            depth = len(path)
            tok_addr = current_token.address.lower()

            # Prune dominated states.
            existing = best_at.get((tok_addr, depth))
            if existing is not None and current_amount <= existing:
                return
            best_at[(tok_addr, depth)] = current_amount

            if tok_addr == dst_addr:
                steps = [
                    SwapStep(
                        token_in=e.token_in,
                        token_out=e.token_out,
                        pool_address=e.pool_address,
                        protocol=e.protocol,
                        fee=e.fee_bps * 100,
                    )
                    for e in path
                ]
                routes.append(SwapRoute(
                    steps=steps,
                    amount_in=amount_in,
                    amount_out=TokenAmount(token=token_out, amount=current_amount),
                    price_impact=self._estimate_price_impact(path, amount_in.amount),
                ))
                return

            if depth >= self.max_hops:
                return

            for edge in self.graph.edges_from(current_token):
                next_addr = edge.token_out.address.lower()
                if next_addr in visited_tokens:
                    continue
                next_amount = edge.amount_out(current_amount)
                if next_amount <= 0:
                    continue
                dfs(
                    edge.token_out,
                    next_amount,
                    path + [edge],
                    visited_tokens | {next_addr},
                )

        dfs(src, amount_in.amount, [], {src.address.lower()})

        if not routes:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol}"
            )

        routes.sort(key=lambda r: r.amount_out.amount, reverse=True)
        return routes[:top_k]

    @staticmethod
    def _estimate_price_impact(edges: list[PoolEdge], amount_in: int) -> Decimal:
        """Estimate cumulative price impact across a multi-hop path.

        Delegates per-hop impact estimation to each edge's polymorphic
        :meth:`~pydifi.pathfinder.graph.PoolEdge.estimate_price_impact` method,
        allowing each pool type (V2, V3, Curve, …) to implement its own model:

        * :class:`~pydifi.pathfinder.graph.PoolEdge` (V2-style): uses
          ``amount_in / (reserve_in + amount_in)``.
        * :class:`~pydifi.pathfinder.graph.V3PoolEdge`: uses virtual reserves
          derived from ``sqrtPriceX96`` / ``liquidity``.

        If a hop returns ``Decimal('NaN')`` (impact unestimable) and no other
        hop yields a positive estimate, the cumulative result is
        ``Decimal('NaN')`` to signal "impact unknown" rather than "zero impact".

        Args:
            edges: Ordered pool edges in the route.
            amount_in: Input amount at the first hop.

        Returns:
            Estimated cumulative price impact in ``[0, 1]``, or
            ``Decimal('NaN')`` if the entire path consists of pools where
            impact cannot be estimated.
        """
        total_impact = Decimal(0)
        current_amount = amount_in
        has_unestimated_hop = False
        for edge in edges:
            hop_impact = edge.estimate_price_impact(current_amount)
            if hop_impact.is_nan():
                has_unestimated_hop = True
            else:
                total_impact += hop_impact
            # Always propagate the simulated amount forward so that later hops
            # use the correct intermediate amount.
            current_amount = edge.amount_out(current_amount)
        if has_unestimated_hop and total_impact == Decimal(0):
            return Decimal("NaN")
        return min(total_impact, Decimal(1))

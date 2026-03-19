"""
DEX pathfinding router.

Implements a modified Dijkstra / Bellman-Ford algorithm over a
:class:`~pydifi.pathfinder.graph.PoolGraph` to find the optimal swap route
between any two tokens.

The algorithm maximises the output amount by working in log-space (minimising
``-log(output/input)`` along each hop), which converts the multiplicative
product of exchange rates into an additive sum.
"""

from __future__ import annotations

import heapq
import math
from decimal import Decimal
from typing import Optional

from pydifi.exceptions import NoRouteFoundError
from pydifi.pathfinder.graph import PoolEdge, PoolGraph
from pydifi.types import SwapRoute, SwapStep, Token, TokenAmount


class Router:
    """Optimal swap route finder over a :class:`~pydifi.pathfinder.graph.PoolGraph`.

    Uses a Dijkstra-style shortest-path algorithm in log-exchange-rate space
    to find the multi-hop route that maximises the output amount.

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

        Uses a modified Dijkstra algorithm:

        * Node state: ``(cumulative_log_weight, token, hops, path_so_far)``
        * Edge weight: ``-log(edge.amount_out(amount) / amount)``
        * We forward-simulate the actual amount at each hop so that price
          impact is included naturally.

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

        # Priority queue: (neg_log_output, current_amount, token, hops, steps)
        # We minimise neg_log_output, which maximises the total output.
        heap: list[tuple[float, int, Token, int, list[PoolEdge]]] = [
            (0.0, amount_in.amount, src, 0, [])
        ]
        # best neg_log_output seen for each (token_address, hops) state
        visited: dict[tuple[str, int], float] = {}

        best_route: Optional[SwapRoute] = None

        while heap:
            neg_log_out, current_amount, current_token, hops, path = heapq.heappop(heap)

            state_key = (current_token.address.lower(), hops)
            if state_key in visited and visited[state_key] <= neg_log_out:
                continue
            visited[state_key] = neg_log_out

            # Check if we've reached the destination
            if current_token.address.lower() == dst_addr:
                steps = [
                    SwapStep(
                        token_in=edge.token_in,
                        token_out=edge.token_out,
                        pool_address=edge.pool_address,
                        protocol=edge.protocol,
                        fee=edge.fee_bps * 100,
                    )
                    for edge in path
                ]
                route = SwapRoute(
                    steps=steps,
                    amount_in=amount_in,
                    amount_out=TokenAmount(token=token_out, amount=current_amount),
                    price_impact=self._estimate_price_impact(path, amount_in.amount),
                )
                if best_route is None or current_amount > best_route.amount_out.amount:
                    best_route = route
                continue

            if hops >= self.max_hops:
                continue

            for edge in self.graph.edges_from(current_token):
                next_amount = edge.amount_out(current_amount)
                if next_amount <= 0:
                    continue
                # Avoid cycles
                visited_tokens = {e.token_in.address.lower() for e in path}
                if edge.token_out.address.lower() in visited_tokens:
                    continue

                step_weight = -math.log(next_amount / current_amount) if current_amount > 0 else math.inf
                new_weight = neg_log_out + step_weight
                new_state = (edge.token_out.address.lower(), hops + 1)
                if new_state in visited and visited[new_state] <= new_weight:
                    continue

                heapq.heappush(
                    heap,
                    (new_weight, next_amount, edge.token_out, hops + 1, path + [edge]),
                )

        if best_route is None:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol} "
                f"within {self.max_hops} hops"
            )
        return best_route

    def find_all_routes(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        top_k: int = 5,
    ) -> list[SwapRoute]:
        """Find the top-*k* routes by output amount.

        Uses DFS with pruning to enumerate multi-hop routes.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            top_k: Maximum number of routes to return.

        Returns:
            List of :class:`~pydifi.types.SwapRoute` objects, sorted by
            output amount descending.

        Raises:
            :class:`~pydifi.exceptions.NoRouteFoundError`: If no routes exist.
        """
        src = amount_in.token
        dst_addr = token_out.address.lower()
        routes: list[SwapRoute] = []

        if src.address.lower() == dst_addr:
            raise ValueError("token_in and token_out must be different")

        def dfs(
            current_token: Token,
            current_amount: int,
            path: list[PoolEdge],
            visited_tokens: set[str],
        ) -> None:
            if current_token.address.lower() == dst_addr:
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

            if len(path) >= self.max_hops:
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

        Price impact for a single hop = ``(reserve_in * reserve_out)`` before
        vs after the swap.  We compute a simple approximation based on the
        ratio of amount_in to reserve_in.

        Args:
            edges: Ordered pool edges in the route.
            amount_in: Input amount at the first hop.

        Returns:
            Estimated price impact in [0, 1].
        """
        total_impact = Decimal(0)
        current_amount = amount_in
        for edge in edges:
            if edge.reserve_in > 0:
                impact = Decimal(current_amount) / Decimal(edge.reserve_in + current_amount)
                total_impact += impact
                current_amount = edge.amount_out(current_amount)
        return min(total_impact, Decimal(1))

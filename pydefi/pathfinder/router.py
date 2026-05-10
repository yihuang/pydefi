"""
DEX pathfinding router.

Implements hop-bounded dynamic programming over a
:class:`~pydefi.pathfinder.graph.PoolGraph` to find the optimal swap route
between any two tokens.

The algorithm maximises the output amount by tracking the best raw output at
each ``(token, hop_depth)`` state.  Because ``edge.amount_out`` is
monotonically increasing, the state with the largest raw output at any
intermediate token always propagates to the largest output at subsequent hops,
making simple DP relaxation both correct and safe.
"""

from __future__ import annotations

from decimal import Decimal

from pydefi.exceptions import NoRouteFoundError
from pydefi.pathfinder.graph import PoolEdge, PoolGraph
from pydefi.types import MAX_BPS, Address, RouteDAG, RouteSplit, RouteSwap, SwapRoute, SwapStep, Token, TokenAmount


class Router:
    """Optimal swap route finder over a :class:`~pydefi.pathfinder.graph.PoolGraph`.

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
        *,
        gas_price_gwei: float = 0.0,
        native_token_price_usd: float = 0.0,
        token_out_price_usd: float = 0.0,
        max_hops: int | None = None,
    ) -> SwapRoute:
        """Find the best route with optional gas-aware scoring.

        - ``gas_price_gwei <= 0``: use output-only routing (legacy behavior).
        - ``gas_price_gwei > 0``: use gas-aware effective-log-weight routing.

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
            gas_price_gwei: Gas price in gwei. Set to 0 to ignore gas
                (backward-compatible behavior).
            native_token_price_usd: Native gas token price in USD.
            token_out_price_usd: Output token price in USD.
            max_hops: Maximum hop depth to explore. If omitted, uses the
                router instance's ``self.max_hops``.

        Returns:
            The best :class:`~pydefi.types.SwapRoute` found.

        Raises:
            :class:`~pydefi.exceptions.NoRouteFoundError`: If no path exists
                between the two tokens within ``max_hops``.
        """
        effective_max_hops = self.max_hops if max_hops is None else max_hops

        if gas_price_gwei > 0:
            if native_token_price_usd <= 0 or token_out_price_usd <= 0:
                raise ValueError(
                    "native_token_price_usd and token_out_price_usd are required when gas_price_gwei is provided"
                )
            return self._find_best_route_gas_aware(
                amount_in,
                token_out,
                gas_price_gwei=gas_price_gwei,
                native_token_price_usd=native_token_price_usd,
                token_out_price_usd=token_out_price_usd,
                max_hops=effective_max_hops,
            )

        return self._find_best_route_no_gas(amount_in, token_out, max_hops=effective_max_hops)

    def _find_best_route_no_gas(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        max_hops: int,
    ) -> SwapRoute:
        """Find the best route by output amount only (ignoring gas costs)."""
        src = amount_in.token
        dst_addr = token_out.address

        if src.address == dst_addr:
            raise ValueError("token_in and token_out must be different")

        # DP table: best[(token_addr, hops)] = (max_raw_output, path_of_edges)
        # Seed with the source state at hop depth 0.
        best: dict[tuple[Address, int], tuple[int, list[PoolEdge]]] = {(src.address, 0): (amount_in.amount, [])}

        for hop in range(max_hops):
            # Snapshot all states at the current depth to avoid processing
            # states we add during this iteration.
            current_states = [(k, v) for k, v in best.items() if k[1] == hop]

            for (token_addr, _), (current_amount, path) in current_states:
                # Tokens already on this path (cycle prevention).
                visited_tokens: set[Address] = {e.token_in.address for e in path}
                visited_tokens.add(token_addr)

                # Retrieve the Token object: use the last edge's output token,
                # or the source token for the initial state.
                token: Token = path[-1].token_out if path else src

                for edge in self.graph.edges_from(token):
                    next_addr = edge.token_out.address
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
        best_result: tuple[int, list[PoolEdge]] | None = None
        for h in range(1, max_hops + 1):
            entry = best.get((dst_addr, h))
            if entry is not None:
                if best_result is None or entry[0] > best_result[0]:
                    best_result = entry

        if best_result is None:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol} within {max_hops} hops"
            )

        final_amount, final_path = best_result
        steps = [
            SwapStep(
                token_in=edge.token_in,
                token_out=edge.token_out,
                pool_address=edge.pool_address,
                protocol=edge.protocol,
                fee=edge.fee_bps,
            )
            for edge in final_path
        ]
        return SwapRoute(
            steps=steps,
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=final_amount),
            price_impact=self._estimate_price_impact(final_path, amount_in.amount),
            dag=self._edges_to_dag(src, final_path),
        )

    def _find_best_route_gas_aware(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        gas_price_gwei: float,
        native_token_price_usd: float,
        token_out_price_usd: float,
        max_hops: int,
    ) -> SwapRoute:
        """Find the lowest effective-log-weight route under gas-aware scoring."""
        path = self.graph.find_best_route_gas_aware(
            start=amount_in.token,
            end=token_out,
            amount_in=amount_in.amount,
            weight_fn=lambda edge, current_amount: edge.effective_log_weight(
                amount_in=current_amount,
                gas_price_gwei=gas_price_gwei,
                native_token_price_usd=native_token_price_usd,
                token_out_price_usd=token_out_price_usd,
            ),
            max_hops=max_hops,
        )

        if not path:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol} within {max_hops} hops"
            )

        final_amount = amount_in.amount
        for edge in path:
            final_amount = edge.amount_out(final_amount)

        return SwapRoute(
            steps=[
                SwapStep(
                    token_in=edge.token_in,
                    token_out=edge.token_out,
                    pool_address=edge.pool_address,
                    protocol=edge.protocol,
                    fee=edge.fee_bps,
                )
                for edge in path
            ],
            amount_in=amount_in,
            amount_out=TokenAmount(token=token_out, amount=final_amount),
            price_impact=self._estimate_price_impact(path, amount_in.amount),
            dag=self._edges_to_dag(amount_in.token, path),
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
            List of :class:`~pydefi.types.SwapRoute` objects, sorted by
            output amount descending.

        Raises:
            :class:`~pydefi.exceptions.NoRouteFoundError`: If no routes exist.
            :class:`ValueError`: If ``token_in`` and ``token_out`` are the same.
        """
        src = amount_in.token
        dst_addr = token_out.address
        routes: list[SwapRoute] = []

        if src.address == dst_addr:
            raise ValueError("token_in and token_out must be different")

        # Dominance pruning: best raw output seen for each (token_addr, hop_depth).
        # A path is pruned when it arrives at a state with a lower amount than
        # one already explored — any onward route will be dominated.
        best_at: dict[tuple[Address, int], int] = {}

        def dfs(
            current_token: Token,
            current_amount: int,
            path: list[PoolEdge],
            visited_tokens: set[Address],
        ) -> None:
            depth = len(path)
            tok_addr = current_token.address

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
                        fee=e.fee_bps,
                    )
                    for e in path
                ]
                routes.append(
                    SwapRoute(
                        steps=steps,
                        amount_in=amount_in,
                        amount_out=TokenAmount(token=token_out, amount=current_amount),
                        price_impact=self._estimate_price_impact(path, amount_in.amount),
                        dag=self._edges_to_dag(src, path),
                    )
                )
                return

            if depth >= self.max_hops:
                return

            for edge in self.graph.edges_from(current_token):
                next_addr = edge.token_out.address
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

        dfs(src, amount_in.amount, [], {src.address})

        if not routes:
            raise NoRouteFoundError(f"No route found from {amount_in.token.symbol} to {token_out.symbol}")

        routes.sort(key=lambda r: r.amount_out.amount, reverse=True)
        return routes[:top_k]

    def find_best_route_dag(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        gas_price_gwei: float = 0.0,
        native_token_price_usd: float = 0.0,
        token_out_price_usd: float = 0.0,
        max_hops: int | None = None,
    ) -> RouteDAG:
        """Find the best route and return it as :class:`~pydefi.types.RouteDAG`."""
        route = self.find_best_route(
            amount_in,
            token_out,
            gas_price_gwei=gas_price_gwei,
            native_token_price_usd=native_token_price_usd,
            token_out_price_usd=token_out_price_usd,
            max_hops=max_hops,
        )
        if route.dag is None:
            raise ValueError("internal Router error: missing DAG representation")
        return route.dag

    def find_all_routes_dag(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        top_k: int = 5,
    ) -> list[RouteDAG]:
        """Find top-*k* routes and return them as :class:`~pydefi.types.RouteDAG` objects."""
        routes = self.find_all_routes(amount_in, token_out, top_k=top_k)
        if any(route.dag is None for route in routes):
            raise ValueError("internal Router error: missing DAG representation")
        return [route.dag for route in routes if route.dag is not None]

    def find_best_split(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        max_splits: int = 2,
        step_bps: int = 1000,
        max_hops: int | None = None,
    ) -> RouteDAG:
        """Find the N-way allocation of *amount_in* across routes that maximises output.

        Distributes the input across up to *max_splits* diverse candidate routes
        to maximise aggregate output — the same strategy used by Uniswap Smart
        Path and UniRoute.  All weight vectors from 1-way up to *max_splits*-way
        are evaluated in a single pass by :meth:`_best_n_way_split`.

        Algorithm
        ---------
        1. Discover up to *max_splits* diverse candidate routes via
           :meth:`_find_top_routes` (one per distinct first-hop pool).
        2. Enumerate every weight vector ``(w_0, …, w_{n-1})`` where each
           ``w_i`` is a non-negative multiple of *step_bps* and
           ``sum(w_i) == 10 000``.
        3. Evaluate each allocation off-chain via :meth:`_follow_route`.
        4. Return the best allocation as a :class:`~pydefi.types.RouteDAG`.

        A single-leg result (no split improves on the best route) is returned
        as a linear DAG without a :class:`~pydefi.types.RouteSplit` node.

        .. note::
            Search cost is ``C(k+n-1, n-1)`` (stars-and-bars) where
            ``k = MAX_BPS // step_bps`` and ``n = max_splits``.  At the
            default ``step_bps=1000`` (k=10): 2-way → 11 evals, 3-way → 66,
            4-way → 286.

        Args:
            amount_in: Total input amount.
            token_out: Desired output token.
            step_bps: Weight granularity in basis points (default ``1000`` = 10%).
            max_splits: Maximum number of split legs to consider (default ``2``).
            max_hops: Forwarded to :meth:`_find_top_routes`.

        Returns:
            A :class:`~pydefi.types.RouteDAG` — linear when a single route
            wins, split/merge when multiple legs improve output.

        Raises:
            :class:`~pydefi.exceptions.NoRouteFoundError`: If no route exists.
            :class:`ValueError`: If *max_splits* < 1.
        """
        if max_splits < 1:
            raise ValueError("max_splits must be >= 1")
        routes = self._find_top_routes(amount_in, token_out, top_n=max_splits, max_hops=max_hops)
        edge_index: dict[tuple[Address, Address], PoolEdge] = {
            (edge.pool_address, edge.token_in.address): edge for edge in self.graph
        }
        legs = self._best_n_way_split(routes, amount_in, edge_index, step_bps)

        if len(legs) == 1:
            _, edges = legs[0]
            return self._edges_to_dag(amount_in.token, edges)

        dag = RouteDAG().from_token(amount_in.token)
        dag.split()
        for weight_bps, edges in legs:
            dag.leg(weight_bps)
            for edge in edges:
                dag.swap(edge.token_out, edge)
        dag.merge()
        return dag

    def _find_top_routes(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        top_n: int,
        max_hops: int | None = None,
    ) -> list[SwapRoute]:
        """Find the top-*n* diverse routes by output, deduplicated by first-hop pool.

        Similar to :meth:`find_best_route` but tracks the best *top_n* paths
        at each ``(token, hop)`` state, then deduplicates by first-hop pool
        address so each returned route starts through a genuinely different pool.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token.
            top_n: Maximum number of diverse routes to return.
            max_hops: Maximum hop depth. Defaults to ``self.max_hops``.

        Returns:
            List of :class:`~pydefi.types.SwapRoute` sorted by output descending.
            May contain fewer than *top_n* entries if the graph lacks diversity.

        Raises:
            :class:`~pydefi.exceptions.NoRouteFoundError`: If no path exists.
        """
        effective_max_hops = self.max_hops if max_hops is None else max_hops
        src = amount_in.token
        dst_addr: Address = token_out.address

        if src.address == dst_addr:
            raise ValueError("token_in and token_out must be different")

        best: dict[tuple[Address, int], list[tuple[int, list[PoolEdge]]]] = {(src.address, 0): [(amount_in.amount, [])]}
        # Map Address → Token, kept in sync with `best` key convention.
        token_by_addr: dict[Address, Token] = {src.address: src}

        for hop in range(effective_max_hops):
            current_states = [(k, v) for k, v in best.items() if k[1] == hop and v]
            for (token_addr, _), candidates in current_states:
                token: Token = token_by_addr[token_addr]
                updated_keys: set[tuple[Address, int]] = set()

                for current_amount, path in candidates:
                    visited: set[Address] = {e.token_in.address for e in path}
                    visited.add(token_addr)

                    for edge in self.graph.edges_from(token):
                        next_addr: Address = edge.token_out.address
                        if next_addr in visited:
                            continue
                        next_amount = edge.amount_out(current_amount)
                        if next_amount <= 0:
                            continue
                        next_key = (next_addr, hop + 1)
                        best.setdefault(next_key, []).append((next_amount, path + [edge]))
                        token_by_addr.setdefault(next_addr, edge.token_out)
                        updated_keys.add(next_key)

                for key in updated_keys:
                    lst = best[key]
                    lst.sort(key=lambda x: x[0], reverse=True)
                    best[key] = lst[:top_n]

        all_candidates: list[tuple[int, list[PoolEdge]]] = []
        for h in range(1, effective_max_hops + 1):
            all_candidates.extend(best.get((dst_addr, h), []))

        if not all_candidates:
            raise NoRouteFoundError(
                f"No route found from {amount_in.token.symbol} to {token_out.symbol} within {effective_max_hops} hops"
            )

        all_candidates.sort(key=lambda x: x[0], reverse=True)

        seen_first_pools: set[Address] = set()
        diverse: list[tuple[int, list[PoolEdge]]] = []
        for amount, path in all_candidates:
            first_pool: Address = path[0].pool_address
            if first_pool not in seen_first_pools:
                seen_first_pools.add(first_pool)
                diverse.append((amount, path))
            if len(diverse) >= top_n:
                break

        routes: list[SwapRoute] = []
        for final_amount, final_path in diverse:
            steps = [
                SwapStep(
                    token_in=edge.token_in,
                    token_out=edge.token_out,
                    pool_address=edge.pool_address,
                    protocol=edge.protocol,
                    fee=edge.fee_bps,
                )
                for edge in final_path
            ]
            routes.append(
                SwapRoute(
                    steps=steps,
                    amount_in=amount_in,
                    amount_out=TokenAmount(token=token_out, amount=final_amount),
                    price_impact=self._estimate_price_impact(final_path, amount_in.amount),
                )
            )
        return routes

    def _follow_route(
        self,
        route: SwapRoute,
        raw_amount: int,
        edge_index: dict[tuple[Address, Address], PoolEdge],
    ) -> int:
        """Walk each step of *route* at *raw_amount* and return the output amount."""
        current = raw_amount
        for step in route.steps:
            key = (step.pool_address, step.token_in.address)
            edge = edge_index.get(key)
            if edge is None:
                return 0
            current = edge.amount_out(current)
            if current <= 0:
                return 0
        return current

    def _best_n_way_split(
        self,
        routes: list[SwapRoute],
        amount_in: TokenAmount,
        edge_index: dict[tuple[Address, Address], PoolEdge],
        step_bps: int,
    ) -> list[tuple[int, list[PoolEdge]]]:
        """Return the best N-way weight allocation across *routes* at *step_bps* granularity.

        Enumerates every weight vector ``(w_0, …, w_{n-1})`` where each ``w_i``
        is a non-negative multiple of *step_bps* and the vector sums to
        ``MAX_BPS``.  Degenerate single-leg vectors are included, so the
        result is always at least as good as any one route alone.

        Integer-division rounding is corrected by adding the leftover to the
        last leg with a non-zero weight so leg amounts sum exactly to
        ``amount_in.amount``.

        Returns:
            List of ``(weight_bps, edges)`` pairs for legs with ``weight_bps > 0``,
            ordered by the original route ranking.
        """
        total = amount_in.amount
        n = len(routes)
        best_legs: list[tuple[int, list[PoolEdge]]] = []
        best_total: int = 0

        # Pre-build per-route edge lists once to avoid repeated dict lookups.
        route_edges: list[list[PoolEdge]] = []
        for route in routes:
            edges = [
                edge_index[(step.pool_address, step.token_in.address)]
                for step in route.steps
                if step.pool_address is not None
            ]
            route_edges.append(edges)

        def _enumerate(idx: int, remaining_bps: int, weights: list[int]) -> None:
            nonlocal best_legs, best_total
            if idx == n - 1:
                weights.append(remaining_bps)
                amts: list[int] = [total * w // MAX_BPS for w in weights]
                last_nz = max(i for i in range(n) if weights[i] > 0)
                amts[last_nz] += total - sum(amts)
                outs = [self._follow_route(routes[i], amts[i], edge_index) if amts[i] > 0 else 0 for i in range(n)]
                combined = sum(outs)
                if combined > best_total:
                    best_total = combined
                    best_legs = [(weights[i], route_edges[i]) for i in range(n) if weights[i] > 0]
                weights.pop()
                return
            for w in range(0, remaining_bps + 1, step_bps):
                weights.append(w)
                _enumerate(idx + 1, remaining_bps - w, weights)
                weights.pop()

        _enumerate(0, MAX_BPS, [])
        return best_legs

    def simulate(self, dag: RouteDAG, amount_in: int) -> int:
        """Simulate the output amount for *dag* at *amount_in* using off-chain edge math.

        Handles both linear DAGs and split/merge DAGs recursively.

        Args:
            dag: The route DAG to simulate.
            amount_in: Input amount in raw token units.

        Returns:
            Simulated output amount in raw token units.
        """
        payload = dag.to_dict()
        return self._simulate_actions(payload["actions"], amount_in)

    def _simulate_actions(self, actions: list | tuple, amount: int) -> int:
        for action in actions:
            if isinstance(action, RouteSwap):
                amount = action.pool.amount_out(amount)
            elif isinstance(action, RouteSplit):
                total = 0
                for leg in action.legs:
                    leg_amount = amount * leg.fraction_bps // MAX_BPS
                    total += self._simulate_actions(leg.actions, leg_amount)
                amount = total
        return amount

    @staticmethod
    def dag_leg_weights(dag: RouteDAG) -> list[int]:
        """Return ``fraction_bps`` for each split leg, or ``[10000]`` for a linear DAG."""

        payload = dag.to_dict()
        if payload["actions"] and isinstance(payload["actions"][0], RouteSplit):
            return [leg.fraction_bps for leg in payload["actions"][0].legs]
        return [MAX_BPS]

    @staticmethod
    def _edges_to_dag(token_in: Token, edges: list[PoolEdge]) -> RouteDAG:
        dag = RouteDAG().from_token(token_in)
        for edge in edges:
            dag.swap(edge.token_out, edge)
        return dag

    @staticmethod
    def _estimate_price_impact(edges: list[PoolEdge], amount_in: int) -> Decimal:
        """Estimate cumulative price impact across a multi-hop path.

        Delegates per-hop impact estimation to each edge's polymorphic
        :meth:`~pydefi.pathfinder.graph.PoolEdge.estimate_price_impact` method,
        allowing each pool type (V2, V3, Curve, …) to implement its own model:

        * :class:`~pydefi.pathfinder.graph.PoolEdge` (V2-style): uses
          ``amount_in / (reserve_in + amount_in)``.
        * :class:`~pydefi.pathfinder.graph.V3PoolEdge`: uses virtual reserves
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

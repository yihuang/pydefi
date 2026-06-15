"""Adaptive Sign Gradient Method for Token Allocation.

Implements PRIME Algorithm 3 (https://arxiv.org/abs/2603.08337). Sign-gradient
ascent on the simplex with Armijo backtracking. Outer optimises ``W_p``
across paths, inner optimises ``w_e`` within each multi-edge hop bundle —
both via a shared ``_simplex_ascent`` primitive — until marginal output
prices equalise (KKT). Marginals are finite differences (swap-function-
agnostic); sign-only updates stay robust at extreme value scales.

Tunables live as module-level constants below: callers always see a clean
2-arg ``asgm_optimize(paths, amount_in)`` and adjust the constants when
re-tuning.
"""

from __future__ import annotations

from collections.abc import Callable

from pydefi.pathfinder.multipath import MultiEdgePath, PathWeights, _distribute_int

# Armijo backtracking line search — values from PRIME Appendix B (the paper's
# sensitivity sweep over α∈[0.01, 0.9], β∈[0.3, 0.95] showed α=1e-4 is robust
# and β=0.5 is the latency / precision sweet spot).
_ARMIJO_ALPHA = 1e-4  # sufficient-decrease parameter (PRIME default)
_DECAY_BETA = 0.5  # step shrink factor on rejection (PRIME default)

# Implementation choices (PRIME doesn't specify concrete values for these).
_MAX_OUTER_ITER = 200  # cross-path ASGM iteration cap
_MAX_INNER_ITER = 50  # per-bundle _simplex_ascent iteration cap
_INITIAL_STEP = 0.1  # initial Δ in simplex units
_OUTER_TOL_BPS = 0.1  # outer convergence: (gap / avg_marginal) in bps
_INNER_TOL_REL = 1e-4  # inner convergence: gap / (|g+| + |g-|) relative
# PRIME uses analytic marginals f_p'(W_p · x); we use finite differences so
# the solver stays agnostic to swap-function form (V2 / V3 / future Curve).
_MARGINAL_EPS = 1e-3  # finite-difference perturbation in simplex units


def asgm_optimize(paths: list[MultiEdgePath], amount_in: int) -> list[PathWeights]:
    """Optimise ``(W_p, w_e)`` across *paths* to maximise total output.

    Uniform initialisation, then nested ASGM until cross-path marginals
    equalise within ``_OUTER_TOL_BPS`` of their average.
    """
    if not paths:
        return []
    if amount_in <= 0:
        raise ValueError("amount_in must be positive")

    n = len(paths)
    weights = [
        PathWeights(
            path_weight=1.0 / n,
            intra_hop_weights=[[1.0 / len(b)] * len(b) for b in p.hops],
        )
        for p in paths
    ]
    if n == 1:
        # Single path: only intra-hop matters. Trivially set path_weight=1.
        weights[0].path_weight = 1.0
        _optimize_intra_hops_inplace(paths[0], weights[0], amount_in)
        return weights

    # Refresh intra-hop weights once before the outer loop so marginals
    # reflect the best the path can do at its current allocation.
    for p, w, x in zip(paths, weights, _path_inputs(weights, amount_in)):
        if x > 0:
            _optimize_intra_hops_inplace(p, w, x)

    for _ in range(_MAX_OUTER_ITER):
        marginals = [
            _path_marginal(paths[i], weights[i], amount_in) if weights[i].path_weight > 0 else float("-inf")
            for i in range(n)
        ]
        i_plus = max(range(n), key=lambda i: marginals[i])
        # p- must currently have positive allocation.
        positive = [i for i in range(n) if weights[i].path_weight > 0]
        if not positive:
            break
        i_minus = min(positive, key=lambda i: marginals[i])
        if i_plus == i_minus:
            break
        gap = marginals[i_plus] - marginals[i_minus]
        avg = sum(m for m in marginals if m != float("-inf")) / max(1, sum(1 for m in marginals if m != float("-inf")))
        if avg > 0 and gap / avg * 10_000 < _OUTER_TOL_BPS:
            break

        delta = min(_INITIAL_STEP, weights[i_minus].path_weight)  # cap so W_{p-} stays >= 0
        if delta <= 0:
            break

        baseline = _total_output(paths, weights, amount_in)
        target_gain = _ARMIJO_ALPHA * delta * amount_in * gap
        # Snapshot both fields a trial mutates: _optimize_intra_hops_inplace
        # rewrites intra_hop_weights, so a rejected trial must restore those too
        # — not just path_weight — or it returns an inconsistent allocation.
        base_plus_pw = weights[i_plus].path_weight
        base_minus_pw = weights[i_minus].path_weight
        base_plus_ihw = [list(h) for h in weights[i_plus].intra_hop_weights]
        base_minus_ihw = [list(h) for h in weights[i_minus].intra_hop_weights]
        accepted = False
        while delta > 1e-9:
            weights[i_plus].path_weight = base_plus_pw + delta
            weights[i_minus].path_weight = base_minus_pw - delta
            # Re-tune intra-hop weights for the two paths whose share changed —
            # use the conservation-preserving distribution.
            current_inputs = _path_inputs(weights, amount_in)
            _optimize_intra_hops_inplace(paths[i_plus], weights[i_plus], current_inputs[i_plus])
            if weights[i_minus].path_weight > 0:
                _optimize_intra_hops_inplace(paths[i_minus], weights[i_minus], current_inputs[i_minus])
            new_total = _total_output(paths, weights, amount_in)
            if new_total >= baseline + target_gain:
                accepted = True
                break
            # Reject: restore path_weight AND intra-hop weights, then shrink.
            weights[i_plus].path_weight = base_plus_pw
            weights[i_minus].path_weight = base_minus_pw
            weights[i_plus].intra_hop_weights = [list(h) for h in base_plus_ihw]
            weights[i_minus].intra_hop_weights = [list(h) for h in base_minus_ihw]
            delta *= _DECAY_BETA
            target_gain = _ARMIJO_ALPHA * delta * amount_in * gap
        if not accepted:
            break

    return weights


def _path_inputs(weights: list[PathWeights], amount_in: int) -> list[int]:
    """Distribute *amount_in* across paths via largest-remainder; total is exactly conserved."""
    return _distribute_int(amount_in, [w.path_weight for w in weights])


def _total_output(paths: list[MultiEdgePath], weights: list[PathWeights], amount_in: int) -> int:
    """Sum of per-path outputs at the conserved per-path inputs."""
    return sum(p.amount_out(x, w) for p, w, x in zip(paths, weights, _path_inputs(weights, amount_in)) if x > 0)


def _path_marginal(path: MultiEdgePath, weights: PathWeights, amount_in: int) -> float:
    """Finite-difference marginal output of *path* w.r.t. its share of *amount_in*.

    Widens the perturbation so it crosses at least one wei even at small
    *amount_in* (otherwise ``x1 == x0`` and the marginal collapses to zero).
    """
    eps_share = max(_MARGINAL_EPS, 1.0 / max(1, amount_in))
    x0 = int(weights.path_weight * amount_in)
    x1 = int((weights.path_weight + eps_share) * amount_in)
    return (path.amount_out(x1, weights) - path.amount_out(x0, weights)) / max(1, x1 - x0)


def _optimize_intra_hops_inplace(path: MultiEdgePath, weights: PathWeights, path_input: int) -> None:
    """Greedy per-hop ASGM: maximise each hop's output independently.

    Hop output is monotonic in hop input and pools are unique per bundle,
    so per-hop greedy is path-optimal for the given ``path_input``.
    """
    if path_input <= 0 or not path.hops:
        return
    cur = path_input
    for i, bundle in enumerate(path.hops):
        if len(bundle) == 1:
            weights.intra_hop_weights[i] = [1.0]
        else:
            weights.intra_hop_weights[i] = _simplex_ascent(
                weights.intra_hop_weights[i],
                eval_fn=lambda ws, b=bundle, c=cur: _bundle_output(b, c, ws),
                grad_fn=lambda ws, idx, b=bundle, c=cur: _bundle_marginal(b, c, ws, idx),
            )
        cur = _bundle_output(bundle, cur, weights.intra_hop_weights[i])
        if cur <= 0:
            break


def _bundle_output(bundle: tuple, hop_input: int, ws: list[float]) -> int:
    """Sum of edge outputs across the bundle under weights *ws*."""
    if hop_input <= 0:
        return 0
    allocs = _distribute_int(hop_input, ws)
    return sum(edge.amount_out(a) for edge, a in zip(bundle, allocs) if a > 0)


def _bundle_marginal(bundle: tuple, hop_input: int, ws: list[float], idx: int) -> float:
    """Finite-difference marginal price of edge ``idx`` at its current share of *hop_input*.

    Approximates ``f_e'(w_e · hop_input)`` per PRIME Algorithm 3 line 5 (inner
    ASGM on edge weights), matching the per-coordinate marginal the outer loop
    uses. Widen the perturbation so it crosses ≥1 wei even at small inputs.
    """
    if hop_input <= 0:
        return 0.0
    eps_share = max(_MARGINAL_EPS, 1.0 / max(1, hop_input))
    edge = bundle[idx]
    x0 = int(ws[idx] * hop_input)
    x1 = int((ws[idx] + eps_share) * hop_input)
    return (edge.amount_out(x1) - edge.amount_out(x0)) / max(1, x1 - x0)


def _simplex_ascent(
    init: list[float],
    *,
    eval_fn: Callable[[list[float]], int],
    grad_fn: Callable[[list[float], int], float],
) -> list[float]:
    """Sign-gradient ascent on the unit simplex.

    Each iteration picks the highest- and lowest-gradient coordinates and
    shifts weight from low to high; Armijo backtracking sets the step size.
    Returns weights with sum-to-1 preserved.
    """
    n = len(init)
    if n <= 1:
        return [1.0] if n == 1 else []
    ws = list(init)
    for _ in range(_MAX_INNER_ITER):
        grads = [grad_fn(ws, i) for i in range(n)]
        positive = [i for i in range(n) if ws[i] > 0]
        if not positive:
            break
        i_plus = max(range(n), key=lambda i: grads[i])
        i_minus = min(positive, key=lambda i: grads[i])
        if i_plus == i_minus:
            break
        gap = grads[i_plus] - grads[i_minus]
        if gap <= 0:
            break
        if abs(gap) / max(1e-12, abs(grads[i_plus]) + abs(grads[i_minus])) < _INNER_TOL_REL:
            break

        delta = min(_INITIAL_STEP, ws[i_minus])
        if delta <= 0:
            break
        baseline = eval_fn(ws)
        accepted = False
        while delta > 1e-9:
            ws[i_plus] += delta
            ws[i_minus] -= delta
            new_val = eval_fn(ws)
            if new_val >= baseline + _ARMIJO_ALPHA * delta * gap:
                accepted = True
                break
            ws[i_plus] -= delta
            ws[i_minus] += delta
            delta *= _DECAY_BETA
        if not accepted:
            break
    return ws

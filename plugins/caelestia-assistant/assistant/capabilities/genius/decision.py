"""genius.decision — decision analysis without an oracle.

When the question is "which option", this module gives the machinery a
careful analyst would use — and shows its work:

  * weighted sum model (WSM) + weighted product model (WPM)
  * AHP: pairwise comparison matrix -> priority vector via the dominant
    eigenvalue (power iteration, from genius.linalg), consistency index
    and consistency ratio with the Saaty random-index table
  * TOPSIS: vector normalization, ideal and anti-ideal solutions,
    separation measures, closeness ranking
  * expected-value decision trees: chance nodes with probabilities,
    leaf payoffs, rollback, expected value of perfect information (EVPI)
  * regret analysis (minimax regret) and Pareto frontier extraction
    for multi-objective choices
  * minimax with alpha-beta pruning over a supplied game tree
  * weight sensitivity: how much would the weights have to change
    before the winner changes (one-at-a-time swing analysis)

Every function returns rankings WITH the intermediate tables so the
recommendation can be audited instead of trusted.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import linalg

__all__ = [
    "weighted_sum", "weighted_product", "ahp", "topsis",
    "decision_tree", "minimax_regret", "pareto_frontier",
    "minimax_game", "sensitivity", "compare",
]


def compare(matrix: Sequence[Sequence[float]], labels: Sequence[str],
            criteria: Sequence[str],
            weights: Optional[Sequence[float]] = None,
            benefits: Optional[Sequence[bool]] = None,
            method: str = "wsm") -> Dict[str, Any]:
    """One entry point for the JSON bridge: dispatch a decision method.

    Mirrors the CLI's ``genius decide`` method switch (wsm | wpm | topsis
    | pareto | regret) so the shell-side tool and the CLI agree on names
    and semantics. ``weights`` defaults to equal weighting. ``ahp`` is
    deliberately not reachable here — it consumes a Saaty pairwise
    matrix (a different input shape) and stays CLI-only.
    """
    method = (method or "wsm").strip().lower()
    if method == "pareto":
        return pareto_frontier(matrix, labels, benefits)
    if method == "regret":
        return minimax_regret(matrix, labels, criteria)
    w = [1.0] * len(criteria) if weights is None else list(weights)
    if method == "topsis":
        return topsis(matrix, labels, w, criteria, benefits)
    if method == "wpm":
        return weighted_product(matrix, labels, w, criteria, benefits)
    if method == "wsm":
        return weighted_sum(matrix, labels, w, criteria, benefits)
    raise ValueError(f"unknown method {method!r} (wsm|wpm|topsis|pareto|regret)")


def _validate(matrix: Sequence[Sequence[float]], labels: Sequence[str],
              weights: Sequence[float], criteria: Sequence[str]) -> None:
    n = len(labels)
    if not n or len(matrix) != n or any(len(r) != len(criteria) for r in matrix):
        raise ValueError("decision matrix must be alternatives x criteria")
    if len(weights) != len(criteria):
        raise ValueError("one weight per criterion required")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights)) < 1e-12:
        raise ValueError("weights sum to zero")


def _norm_weights(weights: Sequence[float]) -> List[float]:
    total = sum(weights)
    return [w / total for w in weights] if total else list(weights)


_BENEFIT_DEFAULT: Dict[str, bool] = {}


def _orient(value: float, benefit: bool) -> float:
    return value if benefit else -value


def weighted_sum(matrix: Sequence[Sequence[float]], labels: Sequence[str],
                weights: Sequence[float], criteria: Sequence[str],
                benefits: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    benefits = list(benefits) if benefits is not None else [True] * len(criteria)
    _validate(matrix, labels, weights, criteria)
    w = _norm_weights(weights)
    scores = {}
    for label, row in zip(labels, matrix):
        scores[label] = sum(_orient(v, benefits[j]) * w[j] for j, v in enumerate(row))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {"method": "WSM (weighted sum)", "weights": dict(zip(criteria, w)),
            "benefit_criteria": [c for c, b in zip(criteria, benefits) if b],
            "cost_criteria": [c for c, b in zip(criteria, benefits) if not b],
            "scores": {k: round(v, 6) for k, v in scores.items()},
            "ranking": [k for k, _ in ranked],
            "winner": ranked[0][0] if ranked else None,
            "note": "additive — one unit on any criterion is directly tradeable"}


def weighted_product(matrix: Sequence[Sequence[float]], labels: Sequence[str],
                     weights: Sequence[float], criteria: Sequence[str],
                     benefits: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    benefits = list(benefits) if benefits is not None else [True] * len(criteria)
    _validate(matrix, labels, weights, criteria)
    w = _norm_weights(weights)
    scores = {}
    for label, row in zip(labels, matrix):
        acc = 0.0
        for j, v in enumerate(row):
            oriented = v if benefits[j] else (1.0 / v if v > 0 else 1e-9)
            acc += w[j] * math.log(max(oriented, 1e-12))
        scores[label] = math.exp(acc)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {"method": "WPM (weighted product)", "weights": dict(zip(criteria, w)),
            "scores": {k: round(v, 8) for k, v in scores.items()},
            "ranking": [k for k, _ in ranked],
            "winner": ranked[0][0] if ranked else None,
            "note": "multiplicative — a zero on any criterion is fatal, ratios matter"}


# Saaty random consistency index for n = 1..15
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
       9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}


def ahp(pairwise: Sequence[Sequence[float]], labels: Sequence[str]) -> Dict[str, Any]:
    """AHP priorities from a Saaty pairwise matrix (values ~1/9..9)."""
    n = len(labels)
    if n != len(pairwise) or any(len(r) != n for r in pairwise):
        raise ValueError("pairwise matrix must be square with one row per label")
    for i in range(n):
        for j in range(n):
            if abs(pairwise[i][j] * pairwise[j][i] - 1) > 1e-6:
                raise ValueError(f"pairwise[{i}][{j}] and [{j}][{i}] are not reciprocal")
    a = [[float(v) for v in row] for row in pairwise]
    lam, vec = linalg.power_iteration(a, iterations=1000)
    total = sum(vec)
    priority = [v / total for v in vec]
    # consistency: lambda_max from Aw = lambda*w component-wise average
    aw = linalg.matmul(a, [[p] for p in priority])
    aw = [row[0] for row in aw]
    lam_est = sum(aw[i] / priority[i] for i in range(n)) / n if all(priority) else lam
    ci = (lam_est - n) / (n - 1) if n > 1 else 0.0
    ri = _RI.get(n, 1.5)
    cr = ci / ri if ri > 0 else 0.0
    ranked = sorted(zip(labels, priority), key=lambda kv: -kv[1])
    return {"method": "AHP", "priorities": {k: round(v, 6) for k, v in zip(labels, priority)},
            "ranking": [k for k, _ in ranked],
            "lambda_max": round(lam_est, 6), "consistency_index": round(ci, 6),
            "consistency_ratio": round(cr, 6),
            "consistent": cr < 0.1,
            "verdict": ("judgments are consistent (CR < 0.10)" if cr < 0.1 else
                        "judgments are inconsistent — revisit the comparisons (CR >= 0.10)")}


def topsis(matrix: Sequence[Sequence[float]], labels: Sequence[str],
           weights: Sequence[float], criteria: Sequence[str],
           benefits: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    """TOPSIS: rank by closeness to the ideal and distance from the worst."""
    benefits = list(benefits) if benefits is not None else [True] * len(criteria)
    _validate(matrix, labels, weights, criteria)
    w = _norm_weights(weights)
    n_c = len(criteria)
    # vector normalization
    norms = [math.sqrt(sum((row[j]) ** 2 for row in matrix)) for j in range(n_c)]
    norms = [x if x > 0 else 1.0 for x in norms]
    v = [[row[j] / norms[j] * w[j] for j in range(n_c)] for row in matrix]
    ideal = [max(col) if benefits[j] else min(col)
             for j, col in enumerate(zip(*v))]
    anti = [min(col) if benefits[j] else max(col)
            for j, col in enumerate(zip(*v))]
    closeness: Dict[str, float] = {}
    for label, row in zip(labels, v):
        d_pos = math.sqrt(sum((row[j] - ideal[j]) ** 2 for j in range(n_c)))
        d_neg = math.sqrt(sum((row[j] - anti[j]) ** 2 for j in range(n_c)))
        closeness[label] = d_neg / (d_pos + d_neg) if (d_pos + d_neg) > 0 else 0.0
    ranked = sorted(closeness.items(), key=lambda kv: -kv[1])
    return {"method": "TOPSIS", "normalized_weights": {c: round(x, 6) for c, x in zip(criteria, w)},
            "ideal": [round(x, 6) for x in ideal], "anti_ideal": [round(x, 6) for x in anti],
            "closeness": {k: round(x, 6) for k, x in closeness.items()},
            "ranking": [k for k, _ in ranked], "winner": ranked[0][0] if ranked else None,
            "note": "picks the alternative geometrically closest to the ideal point"}


# ---------------------------------------------------------------------------
# Expected-value decision trees
# ---------------------------------------------------------------------------

def decision_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Rollback an expected-value tree.

    Node forms:
      {"type": "leaf", "payoff": 100}
      {"type": "choice", "children": [{"label": "a", "node": {...}}, ...]}
      {"type": "chance", "children": [{"p": 0.7, "label": "good", "node": {...}}]}
    """
    def rollback(node: Dict[str, Any]) -> Tuple[float, List[str]]:
        kind = node.get("type")
        if kind == "leaf":
            return float(node.get("payoff", 0.0)), []
        children = node.get("children", [])
        if not children:
            raise ValueError("non-leaf node without children")
        if kind == "choice":
            best_ev, best_path, best_label = None, None, None
            for ch in children:
                ev, path = rollback(ch["node"])
                if best_ev is None or ev > best_ev:
                    best_ev, best_path = ev, path + [ch.get("label", "?")]
                    best_label = ch.get("label", "?")
            return best_ev, best_path
        if kind == "chance":
            ev_total = 0.0
            path_bits: List[str] = []
            probs = [ch.get("p", 0.0) for ch in children]
            if abs(sum(probs) - 1) > 1e-6:
                raise ValueError(f"chance probabilities sum to {sum(probs):.4f}, not 1")
            for ch in children:
                ev, path = rollback(ch["node"])
                ev_total += ch.get("p", 0.0) * ev
                path_bits.append(f"[p={ch.get('p')}]{ch.get('label', '?')}")
            return ev_total, path_bits
        raise ValueError(f"unknown node type {kind!r}")

    ev, path = rollback(tree)
    return {"expected_value": round(ev, 6), "optimal_path": path}


def evpi(payoffs_by_state: Dict[str, Dict[str, float]],
         probs: Dict[str, float]) -> Dict[str, Any]:
    """Expected value of perfect information for act-by-state payoffs."""
    if abs(sum(probs.values()) - 1) > 1e-6:
        raise ValueError("state probabilities must sum to 1")
    acts = list(next(iter(payoffs_by_state.values())).keys())
    evs = {a: sum(probs[s] * payoffs_by_state[s][a] for s in probs) for a in acts}
    best_act = max(evs, key=evs.get)
    ev_no_info = evs[best_act]
    ev_perfect = sum(probs[s] * max(payoffs_by_state[s].values()) for s in probs)
    return {"expected_values": {a: round(v, 4) for a, v in evs.items()},
            "best_act_without_info": best_act,
            "ev_without_info": round(ev_no_info, 4),
            "ev_with_perfect_info": round(ev_perfect, 4),
            "evpi": round(ev_perfect - ev_no_info, 4),
            "note": "the most you should pay for a perfect forecast"}


def minimax_regret(payoff_matrix: Sequence[Sequence[float]],
                   acts: Sequence[str], states: Sequence[str]) -> Dict[str, Any]:
    m = [[float(v) for v in row] for row in payoff_matrix]
    if len(m) != len(acts) or any(len(r) != len(states) for r in m):
        raise ValueError("payoff matrix must be acts x states")
    best_per_state = [max(col) for col in zip(*m)]
    regrets = [[best_per_state[j] - m[i][j] for j in range(len(states))]
               for i in range(len(acts))]
    max_regret = [max(r) for r in regrets]
    best = max(range(len(acts)), key=lambda i: -max_regret[i])
    return {"regret_matrix": {acts[i]: [round(v, 4) for v in regrets[i]]
                               for i in range(len(acts))},
            "max_regret": {acts[i]: round(max_regret[i], 4) for i in range(len(acts))},
            "choice": acts[best],
            "note": "minimizes the worst 'if only I had known' feeling"}


def pareto_frontier(matrix: Sequence[Sequence[float]], labels: Sequence[str],
                    maximize: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    """Extract the non-dominated set (all objectives maximized by default)."""
    n = len(labels)
    if len(matrix) != n:
        raise ValueError("one row per alternative")
    dims = len(matrix[0]) if matrix else 0
    maximize = list(maximize) if maximize is not None else [True] * dims
    pts = [[v if maximize[d] else -v for d, v in enumerate(row)] for row in matrix]
    frontier: List[str] = []
    why: Dict[str, List[str]] = {}
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if all(pts[j][d] >= pts[i][d] for d in range(dims)) and \
               any(pts[j][d] > pts[i][d] for d in range(dims)):
                dominated = True
                why.setdefault(labels[i], []).append(f"dominated by {labels[j]}")
                break
        if not dominated:
            frontier.append(labels[i])
    return {"frontier": frontier, "dominated": why,
            "note": "the frontier is where every trade-off lives; the rest is strictly worse"}


# ---------------------------------------------------------------------------
# Game trees
# ---------------------------------------------------------------------------

def minimax_game(node: Dict[str, Any]) -> Dict[str, Any]:
    """Minimax with alpha-beta over
      {"type": "leaf", "value": v} |
      {"type": "max"|"min", "children": [node, ...]}
    Returns the root value, optimal move index, and nodes pruned.
    """
    stats = {"nodes": 0, "pruned": 0}

    def ab(node: Dict[str, Any], alpha: float, beta: float) -> Tuple[float, Optional[int]]:
        stats["nodes"] += 1
        if node.get("type") == "leaf":
            return float(node.get("value", 0.0)), None
        kind = node.get("type")
        children = node.get("children", [])
        if not children:
            raise ValueError("internal node without children")
        best_idx = 0
        if kind == "max":
            value = -math.inf
            for i, ch in enumerate(children):
                v, _ = ab(ch, alpha, beta)
                if v > value:
                    value, best_idx = v, i
                alpha = max(alpha, value)
                if alpha >= beta:
                    stats["pruned"] += len(children) - i - 1
                    break
            return value, best_idx
        if kind == "min":
            value = math.inf
            for i, ch in enumerate(children):
                v, _ = ab(ch, alpha, beta)
                if v < value:
                    value, best_idx = v, i
                beta = min(beta, value)
                if alpha >= beta:
                    stats["pruned"] += len(children) - i - 1
                    break
            return value, best_idx
        raise ValueError(f"node type must be max|min|leaf, got {kind!r}")

    value, move = ab(node, -math.inf, math.inf)
    return {"value": value, "optimal_child": move,
            "nodes_visited": stats["nodes"], "branches_pruned": stats["pruned"]}


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def sensitivity(matrix: Sequence[Sequence[float]], labels: Sequence[str],
                weights: Sequence[float], criteria: Sequence[str],
                benefits: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    """One-at-a-time weight swings: when does the winner change?"""
    base = weighted_sum(matrix, labels, weights, criteria, benefits)
    base_winner = base["winner"]
    swings: List[Dict[str, Any]] = []
    for j, crit in enumerate(criteria):
        lo, hi = 0.0, 1.0
        flip_points: List[float] = []
        for trial in [k / 200 for k in range(201)]:
            w2 = list(weights)
            rest = sum(weights) - weights[j]
            w2[j] = trial * (rest + 0.0001) / 1.0001 if rest else trial
            w2[j] = trial
            try:
                res = weighted_sum(matrix, labels, w2, criteria, benefits)
                if res["winner"] != base_winner:
                    flip_points.append(round(trial, 4))
            except ValueError:
                continue
        w = _norm_weights(weights)[j]
        nearest = None
        for f in flip_points:
            if f != round(w, 4):
                nearest = f
                break
        swings.append({"criterion": crit, "current_share": round(w, 4),
                       "winner_flips_at_share": nearest if flip_points else None,
                       "robust": not flip_points or
                                min((abs(f - w) for f in flip_points), default=1) > 0.15})
    return {"base_winner": base_winner, "swings": swings,
            "note": "a criterion whose tiny swing flips the winner deserves a second look"}

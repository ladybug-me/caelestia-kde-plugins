"""genius.optimize — numerical & combinatorial optimization, stdlib-only.

    caelestia-assist genius optimize anneal --energy "(x-3)^2 + (y+1)^2" --x0 10 --x1 10
    caelestia-assist genius optimize pareto '[{"cpu": 5, "fps": 60}, {"cpu": 2, "fps": 30}]' --axes cpu --axes fps
    caelestia-assist do "minimize (x-2)^2 with annealing"   # via the meta router

Everything is deterministic under a fixed seed (the codebase rule: learned
or stochastic behaviour must be reproducible). No I/O, no state — pure
functions returning plain data.

| Function | Algorithm | Use it for |
| --- | --- | --- |
| `anneal` | simulated annealing with geometric schedule | discrete/rough landscapes |
| `hill_climb` | stochastic hill climbing + random restarts | cheap local refinement |
| `genetic` | steady-state GA (tournament selection, blend/gaussian crossover, gaussian mutation) | population search over vectors |
| `ternary_min` | ternary search on a unimodal function | exact 1-D continuous minimum |
| `golden_section` | golden-section search on a unimodal function | same, fewer evaluations |
| `pareto_frontier` | non-dominated set extraction | multi-objective trade-offs |
| `pareto_sort` | full non-dominated ranking (front 1, 2, ...) | tiered recommendation lists |
| `branch_and_bound` | 0/1 knapsack with the LP (fractional-greedy) bound | small integer programs |
| `tabu_search` | deterministic tabu-list local search (Glover 1986) | scheduling-shaped permutations |

The settings optimizer (assistant/settings/optimize.py) builds on the Pareto
utilities; the agent layer uses `hill_climb` for plan repair.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

__all__ = [
    "anneal", "hill_climb", "genetic", "ternary_min", "golden_section",
    "pareto_frontier", "pareto_sort", "branch_and_bound", "tabu_search",
]


# ---------------------------------------------------------------------------
# Simulated annealing / hill climbing
# ---------------------------------------------------------------------------

def anneal(energy: Callable[[Sequence[float]], float], x0: Sequence[float],
           steps: int = 2000, t0: float = 1.0, t1: float = 1e-4,
           step_scale: float = 1.0, bounds: Sequence[Tuple[float, float]] = (),
           seed: int = 7) -> Dict[str, Any]:
    """Simulated annealing over R^n with a geometric cooling schedule
    T(k) = t0 * (t1/t0)^(k/steps). Neighbour moves are gaussian with scale
    decaying alongside temperature. `bounds` clamps per-dimension.
    Deterministic for a fixed seed."""
    rng = random.Random(seed)
    x = [float(v) for v in x0]
    ex = energy(x)
    best_x, best_e = list(x), ex
    curve: List[float] = []
    for k in range(steps):
        t = t0 * (t1 / t0) ** (k / max(1, steps - 1))
        cand = [v + rng.gauss(0.0, step_scale * t) for v in x]
        for i, (lo, hi) in enumerate(bounds):
            cand[i] = min(hi, max(lo, cand[i]))
        ec = energy(cand)
        if ec <= ex or rng.random() < math.exp(-(ec - ex) / max(t, 1e-300)):
            x, ex = cand, ec
            if ec < best_e:
                best_x, best_e = list(cand), ec
        if k % max(1, steps // 20) == 0:
            curve.append(best_e)
    return {"x": best_x, "energy": best_e, "steps": steps, "curve": curve,
            "final": {"x": x, "energy": ex}}


def hill_climb(energy: Callable[[Sequence[float]], float], x0: Sequence[float],
               restarts: int = 8, iters: int = 200, step: float = 0.5,
               bounds: Sequence[Tuple[float, float]] = (),
               seed: int = 7) -> Dict[str, Any]:
    """Stochastic hill climbing with random-restart: from each start point,
    accept strictly-improving gaussian moves with an annealed step; keep the
    best across restarts. Cheap, surprisingly strong on smooth problems."""
    rng = random.Random(seed)
    best_x: List[float] = []
    best_e = math.inf
    starts: List[List[float]] = [[float(v) for v in x0]]
    for _ in range(restarts):
        starts.append([rng.uniform(lo, hi) if bounds else
                       float(v) + rng.gauss(0, 3 * step) for v in x0] if bounds else
                      [float(v) + rng.gauss(0, 3 * step) for v in x0])
    evaluated = 0
    for start in starts:
        x, ex = list(start), energy(start)
        evaluated += 1
        for it in range(iters):
            s = step * (1.0 - it / iters) + 1e-6
            cand = [v + rng.gauss(0.0, s) for v in x]
            for i, (lo, hi) in enumerate(bounds):
                cand[i] = min(hi, max(lo, cand[i]))
            ec = energy(cand)
            evaluated += 1
            if ec < ex:
                x, ex = cand, ec
        if ex < best_e:
            best_x, best_e = x, ex
    return {"x": best_x, "energy": best_e, "restarts": restarts + 1,
            "evaluations": evaluated}


# ---------------------------------------------------------------------------
# Genetic algorithm (steady-state, real-coded)
# ---------------------------------------------------------------------------

def genetic(energy: Callable[[Sequence[float]], float],
            bounds: Sequence[Tuple[float, float]], pop_size: int = 40,
            generations: int = 120, tournament: int = 3, crossover: float = 0.9,
            mutation: float = 0.15, elitism: int = 2, seed: int = 7,
            minimize: bool = True) -> Dict[str, Any]:
    """Steady-state real-coded GA. Selection: k-way tournament. Crossover:
    blend (BLX-alpha-ish) per gene. Mutation: gaussian reset scaled to the
    gene range. Elitism carries the best `elitism` individuals forward.
    Returns the best genome, its energy, and the per-generation best curve."""
    rng = random.Random(seed)
    if not bounds:
        raise ValueError("genetic() requires bounds per gene")

    def score(ind: List[float]) -> float:
        return energy(ind) if minimize else -energy(ind)

    def rand_ind() -> List[float]:
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    def mutate(ind: List[float]) -> List[float]:
        out = list(ind)
        for i, (lo, hi) in enumerate(bounds):
            if rng.random() < mutation:
                sigma = 0.1 * (hi - lo)
                out[i] = min(hi, max(lo, out[i] + rng.gauss(0.0, sigma)))
        return out

    def _cross(a: List[float], b: List[float]) -> List[float]:
        if rng.random() > crossover:
            return list(a)
        return [rng.uniform(lo, hi) if rng.random() < 0.5 else
                (a[i] + b[i]) / 2.0 for i, (lo, hi) in enumerate(bounds)]

    pop = [rand_ind() for _ in range(pop_size)]
    scores = [score(p) for p in pop]
    curve: List[float] = []
    n_children = max(1, pop_size - max(0, elitism))
    for _gen in range(generations):
        order = sorted(range(pop_size), key=lambda i: scores[i])
        curve.append(scores[order[0]])
        # steady-state: exactly n_children replacement attempts per generation,
        # each replacing the current worst individual (bounded — no spin).
        for _c in range(n_children):
            def pick() -> List[float]:
                contenders = rng.sample(range(pop_size), min(tournament, pop_size))
                winner = min(contenders, key=lambda i: scores[i])
                return pop[winner]
            p1, p2 = pick(), pick()
            child = mutate(_cross(p1, p2))
            cs = score(child)
            worst = max(range(pop_size), key=lambda i: scores[i])
            if cs < scores[worst]:
                pop[worst], scores[worst] = child, cs
    order = sorted(range(pop_size), key=lambda i: scores[i])
    best = pop[order[0]]
    best_raw = energy(best) if minimize else -energy(best)
    return {"x": best, "energy": best_raw, "generations": generations,
            "curve": curve, "population_best": scores[order[0]]}


# ---------------------------------------------------------------------------
# 1-D exact search on unimodal functions
# ---------------------------------------------------------------------------

def ternary_min(f: Callable[[float], float], lo: float, hi: float,
                iters: int = 100, tol: float = 1e-9) -> Dict[str, Any]:
    """Ternary search for the minimum of a unimodal f on [lo, hi]. Exact to
    `tol` after O(iters log((hi-lo)/tol))-ish contraction; stops early."""
    for _ in range(iters):
        if hi - lo < tol:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    x = (lo + hi) / 2.0
    return {"x": x, "value": f(x), "interval": [lo, hi]}


def golden_section(f: Callable[[float], float], lo: float, hi: float,
                   tol: float = 1e-9, max_evals: int = 200) -> Dict[str, Any]:
    """Golden-section search — ternary search with a 0.618 contraction and
    one evaluation per step (cheapest exact 1-D method)."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    evals = 2
    while abs(b - a) > tol and evals < max_evals:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        evals += 1
    x = (a + b) / 2.0
    return {"x": x, "value": f(x), "evals": evals, "interval": [a, b]}


# ---------------------------------------------------------------------------
# Multi-objective: Pareto dominance
# ---------------------------------------------------------------------------

def _dominates(a: Dict[str, float], b: Dict[str, float], axes: Sequence[str],
               directions: Sequence[str]) -> bool:
    """a dominates b iff a is at least as good everywhere and strictly
    better somewhere ('min' axes: lower is better; 'max': higher is better)."""
    at_least = True
    strictly = False
    for ax, d in zip(axes, directions):
        av, bv = a[ax], b[ax]
        better_or_equal = (av <= bv) if d == "min" else (av >= bv)
        strictly_better = (av < bv) if d == "min" else (av > bv)
        if not better_or_equal:
            at_least = False
            break
        if strictly_better:
            strictly = True
    return at_least and strictly


def pareto_frontier(points: Sequence[Dict[str, float]], axes: Sequence[str],
                    directions: Sequence[str] = ()) -> Dict[str, Any]:
    """Non-dominated set over the given axes. O(k·n^2) — honest about cost,
    fine for the hundreds of settings bundles this assistant compares.
    Directions default to 'min' for every axis."""
    dirs = list(directions) or ["min"] * len(axes)
    if len(dirs) != len(axes):
        raise ValueError("directions must match axes")
    for p in points:
        missing = [ax for ax in axes if ax not in p]
        if missing:
            raise ValueError(f"point missing axes {missing}")
    front: List[Dict[str, float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and _dominates(q, p, axes, dirs):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return {"front": front, "dominated_count": len(points) - len(front),
            "axes": list(axes), "directions": dirs}


def pareto_sort(points: Sequence[Dict[str, float]], axes: Sequence[str],
                directions: Sequence[str] = ()) -> Dict[str, Any]:
    """Full non-dominated ranking: front 1, front 2, ... (each point in an
    earlier front dominates none and is dominated by none within it). Useful
    as a tiered recommendation order."""
    dirs = list(directions) or ["min"] * len(axes)
    remaining = [dict(p) for p in points]
    fronts: List[List[Dict[str, float]]] = []
    while remaining:
        fr = pareto_frontier(remaining, axes, dirs)["front"]
        fronts.append(fr)
        front_ids = {id(p) for p in fr}
        remaining = [p for p in remaining if id(p) not in front_ids]
    return {"fronts": fronts, "tiers": len(fronts), "axes": list(axes),
            "directions": dirs}


# ---------------------------------------------------------------------------
# Branch-and-bound and tabu search (phase 2.2) — discrete optimization.
# ---------------------------------------------------------------------------


def branch_and_bound(items: Sequence[Tuple[float, float]],
                     capacity: float,
                     max_nodes: int = 200000) -> Dict[str, Any]:
    """0/1 knapsack by branch-and-bound with the LP relaxation bound.

    The canonical small integer program: each item is (weight, value),
    at most one copy each, total weight <= capacity, maximize value.
    Bound (Land & Doig 1960's branch-and-bound idea, instantiated with
    Dantzig's fractional-greedy upper bound — the classic 1957 bound):
    sort by value density, fill fractionally, allow the last item to
    split; that relaxed value upper-bounds every integer completion, so
    a subtree whose bound <= the best integer solution is pruned.

    Depth-first with best-first item ordering; deterministic (ties break
    by index). ``max_nodes`` guards against pathological instances — the
    search stops and says so honestly instead of hanging. Returns the
    chosen item indices, the optimal value/weight, and the pruning
    accounting.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    parsed = [(float(w), float(v), i) for i, (w, v) in enumerate(items)]
    for w, v, _i in parsed:
        if w < 0 or v < 0:
            raise ValueError("weights and values must be non-negative")
    # value-density order (greedy relaxation order)
    order = sorted(parsed, key=lambda t: (-(t[1] / t[0]) if t[0] > 0 else -t[1], t[2]))

    def bound(index: int, weight: float, value: float) -> float:
        """LP relaxation of the remaining items (fractional greedy)."""
        best = value
        remaining = capacity - weight
        for w, v, _i in order[index:]:
            if remaining <= 0:
                break
            if w <= remaining:
                best += v
                remaining -= w
            else:
                best += v * (remaining / w)  # fractional fill
                remaining = 0.0
        return best

    best_value = 0.0
    best_taken: List[int] = []
    nodes = 0
    pruned = 0
    stopped = False

    # iterative DFS: stack entries (item index in `order`, weight, value,
    # taken-list) — the two children are take/skip
    stack: List[Tuple[int, float, float, List[int]]] = [(0, 0.0, 0.0, [])]
    while stack:
        if nodes >= max_nodes:
            stopped = True
            break
        index, weight, value, taken = stack.pop()
        nodes += 1
        if index >= len(order):
            if value > best_value:
                best_value = value
                best_taken = taken
            continue
        if bound(index, weight, value) <= best_value:
            pruned += 1
            continue
        w, v, original = order[index]
        # skip child first so the take child (deeper, better bound) is
        # explored last-but-first from the stack -> value-greedy dive
        stack.append((index + 1, weight, value, taken))
        if weight + w <= capacity:
            stack.append((index + 1, weight + w, value + v,
                          taken + [original]))

    total_weight = sum(items[i][0] for i in best_taken)
    out: Dict[str, Any] = {
        "chosen": sorted(best_taken),
        "total_value": round(best_value, 6),
        "total_weight": round(total_weight, 6),
        "capacity": capacity,
        "nodes_explored": nodes,
        "subtrees_pruned": pruned,
        "algorithm": ("branch-and-bound (Land & Doig 1960) with the "
                      "fractional-greedy LP bound (Dantzig 1957)"),
    }
    if stopped:
        out["note"] = (f"node budget {max_nodes} reached — the answer is "
                       "the best found, not a proven optimum")
    return out


def tabu_search(score: Callable[[Sequence[Any]], float],
                initial: Sequence[Any],
                neighbors: Callable[[Sequence[Any]], Sequence[Sequence[Any]]],
                rounds: int = 60,
                tabu_size: int = 12,
                minimize: bool = True) -> Dict[str, Any]:
    """Deterministic tabu-list local search (Glover 1986, "Future paths
    for integer programming and links to artificial intelligence",
    Comput. & OR 13(5)).

    ``score(candidate)``: the objective (minimized by default);
    ``neighbors(candidate)``: the move set (a list of candidates);
    ``tabu_size``: the recency list length — the last N accepted moves'
    signatures cannot be re-entered, which is what escapes the local
    optima plain hill climbing gets stuck in.

    Deterministic: neighbors are evaluated in the given order, the best
    non-tabu (or aspiration-better) move wins, ties break to the first.
    Returns the best candidate seen, its score, and the trajectory's
    honest accounting (moves taken, tabu blocks, aspiration overrides).
    """
    current = list(initial)
    current_score = score(current)
    best = list(current)
    best_score = current_score
    tabu: List[str] = []
    blocks = 0
    aspirations = 0
    moves = 0

    def signature(candidate: Sequence[Any]) -> str:
        return repr(list(candidate))

    for _ in range(rounds):
        candidates = list(neighbors(current))
        if not candidates:
            break
        chosen = None
        chosen_score = None
        for cand in candidates:
            s = score(cand)
            sig = signature(cand)
            better = (s < current_score) if minimize else (s > current_score)
            is_tabu = sig in tabu
            aspiration = (s < best_score) if minimize else (s > best_score)
            if is_tabu and not aspiration:
                blocks += 1
                continue
            if aspiration and is_tabu:
                aspirations += 1
            if (chosen_score is None
                    or ((s < chosen_score) if minimize else (s > chosen_score))):
                chosen, chosen_score = list(cand), s
        if chosen is None:
            break  # every move is tabu without aspiration: stop honestly
        moves += 1
        current, current_score = chosen, chosen_score
        tabu.append(signature(chosen))
        if len(tabu) > tabu_size:
            tabu.pop(0)
        if (current_score < best_score) if minimize else (current_score > best_score):
            best, best_score = list(current), current_score

    return {"best": best, "score": best_score,
            "rounds_run": moves,
            "tabu_blocks": blocks, "aspiration_overrides": aspirations,
            "algorithm": (f"tabu search (Glover 1986), deterministic "
                          f"order, tabu list {tabu_size}")}

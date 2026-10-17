"""Daily plan (0/1 knapsack) and project sequencing (Critical Path Method)."""
import math
from collections import deque

UNIT = 5  # minutes per knapsack cell


def knapsack(items, budget_min):
    """items: [{id, minutes, value}]. Maximises total value within budget_min.

    Exact DP over UNIT-minute cells with backtracking. Returns (ids, value, used).
    """
    W = int(budget_min // UNIT)
    weights = [max(1, math.ceil(it["minutes"] / UNIT)) for it in items]
    values = [float(it["value"]) for it in items]
    n = len(items)
    if n == 0 or W <= 0:
        return [], 0.0, 0
    if n * W > 5_000_000:
        return _greedy(items, budget_min)
    best = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for c in range(W + 1):
            best[i][c] = best[i - 1][c]
            if wi <= c and best[i - 1][c - wi] + vi > best[i][c]:
                best[i][c] = best[i - 1][c - wi] + vi
    chosen, c = [], W
    for i in range(n, 0, -1):
        if best[i][c] != best[i - 1][c]:
            chosen.append(items[i - 1]["id"])
            c -= weights[i - 1]
    chosen.reverse()
    used = sum(int(math.ceil(it["minutes"])) for it in items if it["id"] in chosen)
    return chosen, best[n][W], used


def _greedy(items, budget_min):
    ranked = sorted(items, key=lambda it: -it["value"] / max(it["minutes"], 1))
    chosen, used, value = [], 0, 0.0
    for it in ranked:
        if used + it["minutes"] <= budget_min:
            chosen.append(it["id"])
            used += it["minutes"]
            value += it["value"]
    return chosen, value, used


def cpm(tasks):
    """tasks: {id: {"minutes": m, "deps": [ids]}}.

    Returns {id: {es, ef, slack, critical}} in minutes from project start.
    Raises ValueError on a dependency cycle or unknown dependency.
    """
    for tid, t in tasks.items():
        for d in t.get("deps", []):
            if d not in tasks:
                raise ValueError(f"{tid} depends on unknown task {d}")
    indeg = {t: len(tasks[t].get("deps", [])) for t in tasks}
    succ = {t: [] for t in tasks}
    for t, spec in tasks.items():
        for d in spec.get("deps", []):
            succ[d].append(t)
    q = deque(sorted(t for t in tasks if indeg[t] == 0))
    order = []
    while q:
        t = q.popleft()
        order.append(t)
        for s in sorted(succ[t]):
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != len(tasks):
        raise ValueError("dependency cycle detected")
    es, ef = {}, {}
    for t in order:
        es[t] = max((ef[d] for d in tasks[t].get("deps", [])), default=0)
        ef[t] = es[t] + tasks[t]["minutes"]
    total = max(ef.values(), default=0)
    lf = {t: total for t in tasks}
    for t in reversed(order):
        if succ[t]:
            lf[t] = min(lf[s] - tasks[s]["minutes"] for s in succ[t])
    out = {}
    for t in tasks:
        slack = lf[t] - ef[t]
        out[t] = {"es": es[t], "ef": ef[t], "slack": slack, "critical": slack == 0}
    return out

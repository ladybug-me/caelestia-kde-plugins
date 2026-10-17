"""genius.graphs — classical graph algorithms, stdlib-only.

    caelestia-assist genius dijkstra '{"0": {"1": 4, "2": 1}, "2": {"1": 2}}' --source 0
    caelestia-assist genius topsort '[["a","b"], ["a","c"], ["b","d"]]'
    caelestia-assist genius assign --rows 4,2,7,9 --costs "4,1,3;2,0,5;3,2,2;9,8,9"
    caelestia-assist do "shortest path from a to g in ..."   # via the meta router

Every function is a pure, deterministic, JSON-serialisable computation: plain
data in, plain data out, no I/O, no RNG (all algorithms here are exact).

Algorithms implemented (each real, named, and honest about complexity):

| Function | Algorithm | Complexity |
| --- | --- | --- |
| `dijkstra` | Dijkstra with a binary heap | O((V+E) log V) |
| `astar` | A* with an admissible heuristic callback | O(E) best case |
| `toposort` | Kahn's algorithm + cycle detection | O(V+E) |
| `longest_path_dag` | topological relaxation | O(V+E) |
| `UnionFind` | union-find with path halving + union by size | α(V) amortized |
| `hungarian` | Jonker-Volgenant shortest augmenting path (the O(n^3) assignment algorithm) | O(n^2 m) |
| `min_spanning_tree` | Kruskal over UnionFind | O(E log E) |
| `bellman_ford` | Bellman-Ford with negative-cycle reporting | O(V·E) |
| `articulation_points` | Tarjan low-link DFS | O(V+E) |
| `edmonds_karp` | Ford-Fulkerson with BFS augmentation (Edmonds & Karp 1972) | O(V·E^2) |
| `communities` | label propagation over the shared Graph (Raghavan et al. 2007) | O(V+E) per pass |

These power the agent layer (dependency-resolved task graphs), the settings
optimizer (constraint graphs) and any `do` request that mentions shortest
paths, ordering, or optimal assignment.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "dijkstra", "astar", "toposort", "longest_path_dag", "UnionFind",
    "hungarian", "min_spanning_tree", "bellman_ford", "articulation_points",
    "edmonds_karp", "communities",
]


# ---------------------------------------------------------------------------
# Dijkstra / A*
# ---------------------------------------------------------------------------

def dijkstra(graph: Dict[str, Dict[str, float]], source: str,
             target: Optional[str] = None) -> Dict[str, Any]:
    """Shortest paths from `source` over adjacency {node: {neighbor: weight}}.

    Weights must be non-negative (Dijkstra's invariant); negative edges are
    rejected explicitly rather than producing silently wrong trees.
    Returns {"dist": {...}, "prev": {...}, "path": [...] } — `path` only when
    `target` is given and reachable.
    """
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            if w < 0:
                raise ValueError(f"negative weight {u}->{v} ({w}); use bellman_ford")
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, str] = {}
    done: Dict[str, bool] = {}
    heap: List[Tuple[float, str]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done[u] = True
        if target is not None and u == target:
            break
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    result: Dict[str, Any] = {
        "dist": {k: v for k, v in dist.items() if v != math.inf},
        "prev": prev,
    }
    if target is not None:
        if target in dist:
            path, node = [target], target
            while node != source:
                node = prev[node]
                path.append(node)
            result["path"] = path[::-1]
            result["cost"] = dist[target]
        else:
            result["path"] = None
            result["cost"] = None
    return result


def astar(graph: Dict[str, Dict[str, float]], source: str, target: str,
          heuristic: Callable[[str], float]) -> Dict[str, Any]:
    """A* search. `heuristic(node)` must be admissible (never overestimate);
    with an admissible heuristic the returned path is optimal. The f-score
    tie-break prefers larger g (deeper nodes) which shrinks the explored set
    on grid-like graphs."""
    open_heap: List[Tuple[float, float, str]] = [(heuristic(source), 0.0, source)]
    g_score: Dict[str, float] = {source: 0.0}
    prev: Dict[str, str] = {}
    closed: Dict[str, bool] = {}
    while open_heap:
        _, g, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed[u] = True
        if u == target:
            path, node = [target], target
            while node != source:
                node = prev[node]
                path.append(node)
            return {"cost": g, "path": path[::-1], "expanded": len(closed)}
        for v, w in graph.get(u, {}).items():
            ng = g + w
            if ng < g_score.get(v, math.inf):
                g_score[v] = ng
                prev[v] = u
                heapq.heappush(open_heap, (ng + heuristic(v), ng, v))
    return {"cost": None, "path": None, "expanded": len(closed)}


# ---------------------------------------------------------------------------
# Orderings and DAG analysis
# ---------------------------------------------------------------------------

def toposort(edges: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    """Kahn topological order from edge pairs (a, b) meaning "a before b".

    Returns {"order": [...], "cycle": [...] } — `cycle` lists the nodes that
    could not be ordered (empty iff the graph is a DAG). The order is
    deterministic: among ready nodes the lexicographically smallest is taken
    (a heap, not a set), so the same graph always yields the same answer.
    """
    preds: Dict[str, int] = {}
    succs: Dict[str, List[str]] = {}
    nodes: set = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
        succs.setdefault(a, []).append(b)
        preds[b] = preds.get(b, 0) + 1
        preds.setdefault(a, 0)
    import heapq as _hq
    ready = [n for n, d in preds.items() if d == 0]
    _hq.heapify(ready)
    order: List[str] = []
    while ready:
        n = _hq.heappop(ready)
        order.append(n)
        for m in succs.get(n, []):
            preds[m] -= 1
            if preds[m] == 0:
                _hq.heappush(ready, m)
    cycle = sorted(n for n in nodes if n not in set(order))
    return {"order": order, "cycle": cycle, "is_dag": not cycle}


def longest_path_dag(edges: Iterable[Tuple[str, str]],
                     weights: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, Any]:
    """Longest path in a DAG (weights default 1 per edge) via topological
    relaxation. Reports the critical path — the chain of gates that decides
    total duration. Refuses cyclic graphs with the cycle members."""
    edges = list(edges)
    ts = toposort(edges)
    if not ts["is_dag"]:
        return {"critical_path": None, "length": None, "cycle": ts["cycle"]}
    w = weights or {}
    succs: Dict[str, List[str]] = {}
    for a, b in edges:
        succs.setdefault(a, []).append(b)
    best: Dict[str, float] = {}
    prev: Dict[str, str] = {}
    for node in ts["order"]:
        best.setdefault(node, 0.0)
        for m in succs.get(node, []):
            ew = w.get((node, m), 1.0)
            if best[node] + ew > best.get(m, -math.inf):
                best[m] = best[node] + ew
                prev[m] = node
    if not best:
        return {"critical_path": [], "length": 0.0, "cycle": []}
    end = max(best, key=lambda k: best[k])
    path, node = [end], end
    while node in prev:
        node = prev[node]
        path.append(node)
    return {"critical_path": path[::-1], "length": best[end], "cycle": []}


class UnionFind:
    """Union-find with path halving and union by size (α(n) amortized)."""

    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}
        for it in items:
            self.add(it)

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path halving
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def components(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for item in self.parent:
            groups.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in groups.items()}


# ---------------------------------------------------------------------------
# Minimum spanning tree / negative-weight shortest paths / cut vertices
# ---------------------------------------------------------------------------

def min_spanning_tree(nodes: Sequence[str],
                      edges: Sequence[Tuple[str, str, float]]) -> Dict[str, Any]:
    """Kruskal MST. Returns the chosen edge list and total weight; for a
    disconnected graph the result is a minimum spanning FOREST (components
    reported)."""
    uf = UnionFind(nodes)
    chosen: List[Tuple[str, str, float]] = []
    total = 0.0
    for a, b, w in sorted(edges, key=lambda e: (e[2], e[0], e[1])):
        if uf.union(a, b):
            chosen.append((a, b, w))
            total += w
    return {"edges": chosen, "total": total, "components": uf.components()}


def bellman_ford(graph: Dict[str, Dict[str, float]], source: str) -> Dict[str, Any]:
    """Bellman-Ford: shortest paths allowing negative weights; reports a
    negative cycle (by reconstruction) instead of returning garbage."""
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, str] = {}
    edges = [(u, v, w) for u, nbrs in graph.items() for v, w in nbrs.items()]
    nodes = set(graph) | {v for _, v, _ in edges}
    for _ in range(max(0, len(nodes) - 1)):
        changed = False
        for u, v, w in edges:
            if u in dist and dist[u] + w < dist.get(v, math.inf):
                dist[v] = dist[u] + w
                prev[v] = u
                changed = True
        if not changed:
            break
    negative_cycle: Optional[List[str]] = None
    for u, v, w in edges:
        if u in dist and dist[u] + w < dist.get(v, math.inf):
            # walk back n steps inside the cycle to land on it
            node = v
            for _ in range(len(nodes)):
                node = prev.get(node, node)
            cycle, cur = [node], prev.get(node, node)
            while cur != node and len(cycle) <= len(nodes):
                cycle.append(cur)
                cur = prev.get(cur, cur)
            negative_cycle = cycle[::-1]
            break
    return {"dist": dist, "prev": prev, "negative_cycle": negative_cycle}


def articulation_points(graph: Dict[str, List[str]]) -> List[str]:
    """Tarjan low-link DFS articulation points (single points of failure:
    removing the node disconnects its component). Iterative to survive deep
    graphs on modest machines."""
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, str] = {}
    points: set = set()
    counter = 0
    for start in graph:
        if start in disc:
            continue
        root_children = 0
        stack: List[Tuple[str, int]] = [(start, 0)]
        disc[start] = low[start] = counter
        counter += 1
        while stack:
            node, idx = stack[-1]
            nbrs = graph.get(node, [])
            if idx < len(nbrs):
                stack[-1] = (node, idx + 1)
                n = nbrs[idx]
                if n not in disc:
                    parent[n] = node
                    if node == start:
                        root_children += 1
                    disc[n] = low[n] = counter
                    counter += 1
                    stack.append((n, 0))
                elif n != parent.get(node) and disc[n] < low[node]:
                    low[node] = disc[n]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[node])
                    if p != start and low[node] >= disc[p]:
                        points.add(p)
        if root_children > 1:
            points.add(start)
    return sorted(points)


# ---------------------------------------------------------------------------
# Optimal assignment (Hungarian / Jonker-Volgenant)
# ---------------------------------------------------------------------------

def hungarian(cost: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Minimum-cost perfect assignment of n rows to m columns (n <= m).

    The O(n^2 m) shortest-augmenting-path algorithm with potentials (the
    standard JV/e-maxx formulation): exact, no epsilon, deterministic.
    Non-square inputs are solved as a rectangular problem; assignment of
    every row is returned. Use negated costs for maximum-weight problems.
    """
    n = len(cost)
    if n == 0:
        return {"assignment": [], "total": 0.0}
    m = len(cost[0])
    if any(len(row) != m for row in cost):
        raise ValueError("cost matrix must be rectangular")
    if n > m:
        raise ValueError("more rows than columns; transpose (or pad) first")
    INF = math.inf
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row matched to column j (1-based)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 < 0:
                break  # unreachable with a consistent matrix; guard anyway
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment: List[Optional[int]] = [None] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))  # type: ignore[index]
    return {"assignment": assignment, "total": total}


# ---------------------------------------------------------------------------
# Max-flow (Edmonds-Karp) and label-propagation communities (phase 2.2).
# ---------------------------------------------------------------------------


def edmonds_karp(capacity: Dict[str, Dict[str, float]], source: str,
                 sink: str) -> Dict[str, Any]:
    """Maximum flow: the Ford-Fulkerson method with BFS augmentation
    (Edmonds & Karp 1972, "Theoretical improvements in algorithmic
    efficiency for network flow problems", JACM 19(2) — the BFS choice
    bounds the augmentations at O(V·E), each costing O(E), for the
    O(V·E^2) total in the table above).

    ``capacity``: {u: {v: capacity}} — residual-capacity semantics; the
    reverse direction of an edge may carry capacity too (both keys are
    independent). Capacities must be non-negative. Returns the flow
    value, a per-edge flow map (only NONZERO flows are reported), and
    the min-cut (the BFS-reachable residual set at termination — the
    max-flow/min-cut theorem says it is a minimum cut).
    """
    if source == sink:
        raise ValueError("source and sink must differ")
    nodes = set(capacity)
    for u, nbrs in capacity.items():
        if u not in nodes:
            nodes.add(u)
        for v, c in nbrs.items():
            nodes.add(v)
            if c < 0:
                raise ValueError(f"negative capacity {u}->{v} ({c})")
    # residual graph as a flat dict of dicts, symmetric seats allocated
    residual: Dict[str, Dict[str, float]] = {u: {} for u in nodes}
    for u, nbrs in capacity.items():
        for v, c in nbrs.items():
            residual[u][v] = residual[u].get(v, 0.0) + float(c)
            residual[v].setdefault(u, 0.0)

    from collections import deque

    def bfs(parent: Dict[str, str]) -> bool:
        seen = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, cap in residual[u].items():
                if cap > 0 and v not in seen:
                    seen.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    parent: Dict[str, str] = {}
    flow = 0.0
    while bfs(parent):
        # bottleneck along the BFS path
        bottleneck = math.inf
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, residual[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= bottleneck
            residual[v][u] += bottleneck
            v = u
        flow += bottleneck
        parent = {}

    flows: Dict[str, Dict[str, float]] = {}
    for u, nbrs in capacity.items():
        for v in nbrs:
            pushed = residual[v].get(u, 0.0)  # reverse residual = net flow
            if pushed > 0:
                flows.setdefault(u, {})[v] = round(pushed, 6)
    # min cut: residual-reachable set from source at termination
    reachable = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, cap in residual[u].items():
            if cap > 0 and v not in reachable:
                reachable.add(v)
                queue.append(v)
    cut_edges = [(u, v) for u in reachable
                 for v in capacity.get(u, {}) if v not in reachable]
    return {"max_flow": round(flow, 6), "flows": flows,
            "min_cut_set": sorted(reachable),
            "min_cut_edges": cut_edges,
            "algorithm": "edmonds-karp (BFS-augmented Ford-Fulkerson, "
                         "O(V*E^2)); min cut by max-flow/min-cut theorem"}


def communities(adjacency: Dict[str, Any]) -> Dict[str, Any]:
    """Label-propagation community detection over an adjacency map.

    Reuses ``brain.personal.graph.Graph`` verbatim — its ``communities()``
    is the deterministic label propagation (Raghavan, Albert & Kumara
    2007, "Near linear time algorithm to detect community structures",
    Phys. Rev. E 76) the note-vault analysis already ships, with sorted
    node visits and smallest-label tie-breaks. The adjacency is adapted
    (never reimplemented) into the Graph's add() shape; lists and dict
    neighbor maps are both accepted.
    """
    from assistant.capabilities.brain.personal.graph import Graph

    graph = Graph()
    for node, neighbors in adjacency.items():
        if isinstance(neighbors, dict):
            targets = list(neighbors.keys())
        else:
            targets = [str(n) for n in neighbors]
        graph.add(str(node), [str(t) for t in targets])
    groups = graph.communities()
    return {"communities": groups,
            "n_communities": len(groups),
            "n_nodes": len(graph.nodes),
            "orphans": graph.orphans(),
            "algorithm": "label propagation (Raghavan et al. 2007) via "
                         "brain.personal.graph.Graph, reused verbatim"}

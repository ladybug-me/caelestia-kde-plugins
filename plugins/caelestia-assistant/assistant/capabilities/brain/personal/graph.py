"""Note link graph: PageRank, orphans, and label-propagation communities.

Deterministic: nodes are visited in sorted order and ties break to the smallest
label, so the same vault always yields the same clusters.

exponential-build-3 D adds two link-graph views alongside PageRank:
- HITS (Kleinberg 1999, "Authoritative Sources in a Hyperlinked
  Environment", JACM 46(5), 604-632) — hub/authority scores from
  mutually-reinforcing power iteration, L1-normalized each step,
  reported side by side with PageRank wherever both are surfaced;
- TIME-DECAY EDGE WEIGHTS (``decay_weights``): exponential half-life
  weighting of a note's links by the note's age, feedable into the
  weighted PageRank (and HITS) so "which notes matter NOW" can differ
  from "which notes matter all-time". Both default to the unweighted
  behavior when no weights are given.
"""
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set

LINK = re.compile(r"\[\[([^\]|#]+)")

DEFAULT_HALF_LIFE_DAYS = 90.0


def links(text):
    return {m.strip().lower() for m in LINK.findall(text)}


def decay_weights(notes, now, half_life_days=DEFAULT_HALF_LIFE_DAYS):
    """Time-decay edge weights for a scanned vault: every wiki-link edge
    (source, target) carries ``2 ** (-age_days(source) / half_life_days)``
    — an exponential half-life on the SOURCE note's file age (a note
    edited today links at full strength; a note untouched for one
    half-life links at half strength). ``now`` is a caller-supplied
    epoch timestamp (fixture-injectable for determinism; pass
    ``datetime.now(timezone.utc).timestamp()`` in live code — the same
    injection pattern health_report uses). Read-only: touches nothing
    but mtime. Notes whose mtime cannot be read weigh as fresh (the
    same OSError convention as health_report). Pure function of
    (notes, now, half_life_days); ``half_life_days <= 0`` is refused.
    """
    if half_life_days <= 0:
        raise ValueError(
            f"half_life_days must be positive (got {half_life_days!r})")
    weights = {}
    for rel, note in notes.items():
        try:
            age_days = max(
                (float(now) - os.path.getmtime(note["path"])) / 86400.0,
                0.0)
        except OSError:
            age_days = 0.0
        weight = 2.0 ** (-age_days / half_life_days)
        for target in links(note["text"]):
            weights[(rel.lower(), target)] = weight
    return weights


class Graph:
    def __init__(self):
        self.out = {}

    def add(self, node, targets):
        self.out.setdefault(node, set()).update(targets)
        for t in targets:
            self.out.setdefault(t, set())

    @property
    def nodes(self):
        return sorted(self.out)

    def in_degree(self):
        deg = Counter()
        for src, tgts in self.out.items():
            for t in tgts:
                deg[t] += 1
        return deg

    def pagerank(self, damping=0.85, iters=50, weights=None):
        """PageRank with OPTIONAL edge-strength weights (exponential-
        build-3 D). Weights are EDGE STRENGTHS relative to an
        unweighted edge: 1.0 = full strength, values in (0, 1] (larger
        is refused — a weight can weaken an edge, never amplify it),
        so time-decay weights (which only shrink) fit exactly. The
        out-share of v is ``rank[v] * w(v, t) / len(out(v))`` — NOT
        renormalized by the weight sum, because per-source renormal-
        ization would cancel any per-source weighting (all of a
        source's edges decay together; normalizing them against each
        other hides the decay entirely — the flaw this formulation
        exists to avoid). The un-followed slack
        ``rank[v] * (1 - mean_weight(v))`` joins the dangling mass and
        teleports uniformly, so total rank stays 1: a stale note's
        links are followed with lower probability and the walker
        teleports instead — the time-aware random surfer. With
        ``weights=None`` (or all weights 1.0) this is byte-identical
        to the original implementation (pinned by test)."""
        nodes = self.nodes
        n = len(nodes)
        if n == 0:
            return {}
        if weights is not None:
            for (src, tgt), w in weights.items():
                if tgt in self.out.get(src, ()):
                    if w <= 0:
                        raise ValueError(
                            f"edge weight for ({src}, {tgt}) must be "
                            f"positive (got {w!r}); refusing rather "
                            "than clamping")
                    if w > 1.0:
                        raise ValueError(
                            f"edge weight for ({src}, {tgt}) exceeds "
                            f"full strength 1.0 (got {w!r}); weights "
                            "weaken edges (e.g. time decay), they "
                            "never amplify them")
        rank = {v: 1.0 / n for v in nodes}
        for _ in range(iters):
            dangling = sum(rank[v] for v in nodes if not self.out[v])
            new = {v: (1 - damping) / n + damping * dangling / n
                   for v in nodes}
            for v in nodes:
                if not self.out[v]:
                    continue
                out_v = self.out[v]
                if weights is None:
                    share = damping * rank[v] / len(out_v)
                    for t in out_v:
                        new[t] += share
                    continue
                # sorted: cross-process determinism (set order follows
                # PYTHONHASHSEED; the weighted accumulation must not)
                per = [(t, weights.get((v, t), 1.0))
                       for t in sorted(out_v)]
                strength = sum(w for _t, w in per) / len(per)
                for t, w in per:
                    new[t] += damping * rank[v] * w / len(per)
                slack = rank[v] * (1.0 - strength)
                if slack > 0:
                    # the un-followed share teleports uniformly
                    for t in new:
                        new[t] += damping * slack / n
            rank = new
        return rank

    def hits(self, max_iter=100, tol=1e-10, weights=None):
        """Kleinberg 1999, "Authoritative Sources in a Hyperlinked
        Environment", JACM 46(5), 604-632 — HITS: authorities are the
        notes good hubs point at; hubs are the notes that point at good
        authorities. Classic power iteration (auth = weighted sum of
        in-neighbors' hub scores, hub = weighted sum of out-neighbors'
        authority scores, L1-normalized each step, nodes visited in
        sorted order), with the same OPTIONAL positive edge weights as
        pagerank (e.g. time-decay; ``None`` = unweighted). Converged is
        an honest flag, not an assumption: it reports whether the
        hub+authority delta fell below ``tol`` within ``max_iter``.
        Returns {"hub", "authority", "iterations", "converged"}."""
        nodes = self.nodes
        if not nodes:
            return {"hub": {}, "authority": {}, "iterations": 0,
                    "converged": True}
        if weights is not None:
            for (src, tgt), w in weights.items():
                if tgt in self.out.get(src, ()):
                    if w <= 0:
                        raise ValueError(
                            f"edge weight for ({src}, {tgt}) must be "
                            f"positive (got {w!r}); refusing rather "
                            "than clamping")
                    if w > 1.0:
                        raise ValueError(
                            f"edge weight for ({src}, {tgt}) exceeds "
                            f"full strength 1.0 (got {w!r}); weights "
                            "weaken edges (e.g. time decay), they "
                            "never amplify them")
        incoming = {v: [] for v in nodes}
        for src in nodes:
            for t in self.out[src]:
                incoming[t].append(src)
        hub = {v: 1.0 / len(nodes) for v in nodes}
        auth = {v: 1.0 / len(nodes) for v in nodes}
        for it in range(1, max_iter + 1):
            new_auth = {
                v: sum(hub[u] * (1.0 if weights is None
                                 else weights.get((u, v), 1.0))
                       for u in incoming[v])
                for v in nodes}
            s = sum(new_auth.values())
            if s > 0:
                new_auth = {v: x / s for v, x in new_auth.items()}
            new_hub = {
                v: sum(new_auth[t] * (1.0 if weights is None
                                      else weights.get((v, t), 1.0))
                       for t in sorted(self.out[v]))
                for v in nodes}
            s = sum(new_hub.values())
            if s > 0:
                new_hub = {v: x / s for v, x in new_hub.items()}
            delta = sum(abs(new_auth[v] - auth[v])
                        + abs(new_hub[v] - hub[v]) for v in nodes)
            auth, hub = new_auth, new_hub
            if delta < tol:
                return {"hub": hub, "authority": auth, "iterations": it,
                        "converged": True}
        return {"hub": hub, "authority": auth, "iterations": max_iter,
                "converged": False}

    def orphans(self):
        indeg = self.in_degree()
        return [v for v in self.nodes if not self.out[v] and indeg[v] == 0]

    def communities(self, max_iter=30):
        undirected = {v: set() for v in self.out}
        for src, tgts in self.out.items():
            for t in tgts:
                undirected[src].add(t)
                undirected[t].add(src)
        labels = {v: v for v in self.out}
        for _ in range(max_iter):
            changed = False
            for v in self.nodes:
                if not undirected[v]:
                    continue
                counts = Counter(labels[u] for u in undirected[v])
                best = max(counts.values())
                new = min(lbl for lbl, c in counts.items() if c == best)
                if new != labels[v]:
                    labels[v] = new
                    changed = True
            if not changed:
                break
        groups = {}
        for v, lbl in labels.items():
            groups.setdefault(lbl, []).append(v)
        return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


# ---------------------------------------------------------------------------
# Betweenness centrality + bridge notes (exponential-build-4 C)
# ---------------------------------------------------------------------------


def betweenness(graph) -> Dict[str, Dict[str, float]]:
    """Brandes 2001, "A Faster Algorithm for Betweenness Centrality",
    J. Mathematical Sociology 25(2): the standard accumulation over
    single-source shortest-path DAGs — O(V*E), unweighted (a wiki link
    is an edge; there is no edge length in a vault). Betweenness of v
    = the share of shortest paths between OTHER pairs that pass
    THROUGH v — the "this note connects otherwise-separate clusters"
    quantity, Freeman 1977, "A Set of Measures of Centrality Based on
    Betweenness", Sociometry 40(1). Normalized by (n-1)(n-2)/2, the
    undirected maximum, so the value is in [0, 1] and comparable
    across vault sizes. Deterministic: nodes visited in sorted order,
    ties accumulate identically. Pairs counted once per undirected
    pair; self-pairs excluded (v is never an endpoint of its own
    betweenness)."""
    nodes = graph.nodes
    und: Dict[str, Set[str]] = {v: set() for v in nodes}
    for src in nodes:
        for t in graph.out[src]:
            und[src].add(t)
            und[t].add(src)
    betw: Dict[str, float] = {v: 0.0 for v in nodes}
    for s in sorted(nodes):
        # Brandes single-source pass
        stack: List[str] = []
        preds: Dict[str, List[str]] = {v: [] for v in nodes}
        sigma: Dict[str, float] = {v: 0.0 for v in nodes}
        sigma[s] = 1.0
        dist: Dict[str, int] = {v: -1 for v in nodes}
        dist[s] = 0
        queue: List[str] = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            for w in sorted(und[v]):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta: Dict[str, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betw[w] += delta[w]
    n = len(nodes)
    norm = ((n - 1) * (n - 2) / 2.0) if n > 2 else 1.0
    # undirected graphs count each pair twice in the Brandes delta
    return {v: {"raw": round(betw[v] / 2.0, 6),
                "normalized": round((betw[v] / 2.0) / norm, 6)}
            for v in nodes}


def bridge_notes(graph, top: int = 5) -> Dict[str, Any]:
    """The betweenness-vs-PageRank cross view: notes whose BETWEENNESS
    is high while their PageRank and HITS authority are unremarkable —
    the ones connecting separate clusters rather than the ones
    everything points at (Burt 1992, "Structural Holes": the brokerage
    position, not the popularity position). This is a DIFFERENT signal
    from the existing importance views, presented as such: a bridge
    note is not "more important", it is differently important — it is
    the note whose removal would disconnect topics.

    Ranking: betweenness percentile MINUS mean of (pagerank
    percentile, authority percentile); positive gaps are bridges.
    Percentiles are rank-based over the vault's own distribution — a
    small vault makes percentiles coarse, and the report says so."""
    from .topics import _percentile_ranks  # shared helper (see topics.py)
    bet = betweenness(graph)
    rank = graph.pagerank()
    hits = graph.hits()
    auth = hits["authority"]
    nodes = graph.nodes
    if len(nodes) < 5:
        return {"bridges": [], "n_nodes": len(nodes),
                "note": "fewer than 5 notes: percentiles are too coarse "
                        "to name bridges honestly (rerun as the vault grows)",
                "algorithm": "Brandes 2001 betweenness vs PageRank/HITS "
                             "percentile gap"}
    b_pct = _percentile_ranks({v: bet[v]["normalized"] for v in nodes})
    p_pct = _percentile_ranks({v: rank.get(v, 0.0) for v in nodes})
    a_pct = _percentile_ranks({v: auth.get(v, 0.0) for v in nodes})
    rows = []
    for v in nodes:
        gap = b_pct[v] - (p_pct[v] + a_pct[v]) / 2.0
        rows.append({"note": v,
                     "betweenness": bet[v]["normalized"],
                     "pagerank": round(rank.get(v, 0.0), 6),
                     "gap": round(gap, 3)})
    # the GATE: a bridge note must actually sit on shortest paths —
    # strictly positive betweenness. (A percentile-median gate would
    # not do: tied zero-betweenness nodes share a mean-rank percentile
    # and would sail through on the gap arithmetic alone.)
    rows.sort(key=lambda r: (-r["gap"], r["note"]))
    bridges = [r for r in rows if r["betweenness"] > 0][:top]
    return {"bridges": bridges, "n_nodes": len(nodes),
            "converged": hits["converged"],
            "note": "bridge = high betweenness, unremarkable PageRank/"
                    "authority — a brokerage position (Burt 1992), not a "
                    "fourth flavor of important",
            "algorithm": "Brandes 2001 betweenness percentile minus mean "
                         "PageRank/authority percentile"}

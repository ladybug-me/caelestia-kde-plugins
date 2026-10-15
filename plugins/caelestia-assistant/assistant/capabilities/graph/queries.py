"""assistant.capabilities.graph.queries — the graph's query surface (F12).

Every structural algorithm here REUSES the shipped engines rather than
re-implementing them:

  * shortest explanation path  -> ``genius.graphs.dijkstra`` over
    ``-log(confidence)`` weights (shortest = highest-confidence)
  * "what breaks if Y"         -> ``genius.graphs.articulation_points``
    (cut vertices of the curated interaction graph) and
    ``genius.graphs.edmonds_karp`` (confidence-capacity influence flow)
  * PageRank                   -> a personalized-teleport variant of the
    SAME formulation as ``brain.personal.graph.Graph.pagerank`` (which is
    pinned by its own tests to UNIFORM teleport and cannot take a seed
    vector). This function keeps that class's weight semantics —
    ``rank * w / outdegree`` with un-followed slack joining the dangling
    mass — but teleports to a user seed set when one is given, uniform
    otherwise (where it matches the personal-graph result exactly in
    spirit). It is documented here rather than smuggled into the
    personal-graph class, whose behavior is test-pinned.

Honesty contract: a query whose answer has no cited path ABSTAINS
(``NO_CITED_PATH``); a path the registry never heard of is
``UNKNOWN_PATH``; the interaction universe is the curated table, and
every result says so — structural claims from five curated edges are
labeled as exactly that, never as a model of the shell.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from assistant.capabilities.genius import graphs as _graphs

__all__ = ["what_affects", "what_breaks", "explanation_path", "related",
           "personalized_pagerank", "resolve_config"]

_AFFECTS_KINDS = ("affects", "co_changes")

_BOUNDED_UNIVERSE_NOTE = (
    "the interaction universe is exactly the curated consequences table "
    "(bounded, citation-backed, growable by reviewed diffs) plus, when a "
    "ledger was supplied, co-change co-occurrence which is correlation, "
    "never cause — no claim here generalizes beyond cited edges")


def resolve_config(graph: Dict[str, Any], key: str) -> Optional[str]:
    """Resolve a config path OR a tool name to its config node id."""
    if f"config:{key}" in {n["id"] for n in graph["nodes"]}:
        return f"config:{key}"
    for e in graph["edges"]:
        if e["kind"] == "sets" and e["src"] == f"tool:{key}":
            return e["dst"]
    return None


def _affects_adjacency(graph: Dict[str, Any],
                       kinds: Sequence[str] = _AFFECTS_KINDS
                       ) -> Dict[str, List[str]]:
    """Undirected adjacency over the interaction kinds (for articulation).
    Self-annotation edges (a -> a) are skipped: they annotate a state,
    they do not connect anything."""
    adj: Dict[str, List[str]] = {}
    for e in graph["edges"]:
        if e["kind"] in kinds and e["src"] != e["dst"]:
            adj.setdefault(e["src"], []).append(e["dst"])
            adj.setdefault(e["dst"], []).append(e["src"])
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj


def _closure(graph: Dict[str, Any], start: str, forward: bool,
             kinds: Sequence[str] = _AFFECTS_KINDS) -> Dict[str, List[Dict]]:
    """BFS closure over directed interaction edges; returns node -> hops."""
    seen: Dict[str, List[Dict]] = {}
    frontier = [start]
    while frontier:
        nxt: List[str] = []
        for cur in frontier:
            for e in graph["edges"]:
                if e["kind"] not in kinds:
                    continue
                src, dst = (e["src"], e["dst"]) if forward else (e["dst"], e["src"])
                if src != cur or dst == start or dst in seen:
                    continue
                seen.setdefault(dst, []).append(e)
                nxt.append(dst)
        frontier = nxt
    return seen


def what_affects(graph: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Reverse question: which curated interactions act ON this key?

    Answer = reverse closure over affects edges (with every citation,
    confidence word, arming condition and effect carried through), plus
    the explain_rules that mention the key. Abstains on unknown keys."""
    cid = resolve_config(graph, key)
    if cid is None:
        return {"verdict": "UNKNOWN_PATH", "key": key}
    upstream = _closure(graph, cid, forward=False, kinds=("affects",))
    items = []
    for node, hops in sorted(upstream.items()):
        for e in hops:
            p = e["provenance"]
            items.append({"from": e["src"].split(":", 1)[1],
                          "confidence": p.get("confidence"),
                          "when": p.get("when"),
                          "effect": p.get("effect"),
                          "citation": p.get("citation"),
                          "claimed_content": p.get("claimed_content")})
    rules = [n for n in graph["nodes"]
             if n["type"] == "explain_rule"
             and any(e["kind"] == "explained_by" and e["dst"] == n["id"]
                     and e["src"] == cid for e in graph["edges"])]
    return {"verdict": "OK", "key": cid.split(":", 1)[1],
            "upstream": items,
            "explain_rules": [r["label"] for r in rules],
            "note": _BOUNDED_UNIVERSE_NOTE}


def what_breaks(graph: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Forward question: what changes downstream if this key moves?

    Three honest layers, all clearly labeled:
      1. downstream — the conditional closure over affects edges (each
         with its arming condition; WITHOUT live state we cannot know
         which are armed, so every one is reported as conditional);
      2. structure — articulation points of the curated interaction
         graph: is this key a cut vertex, and which node groups part
         if it is removed (structure of the TABLE, not of the shell);
      3. influence flow — max-flow (Edmonds–Karp) with confidence
         capacities from the key to the TERMINAL nodes of the curated
         interaction graph (nodes that are only ever effects, never
         triggers): a bounded impact score that grows with the table.
         A zero with non-empty downstream means the downstream keys
         are themselves triggers (intermediate, not terminals) — the
         closure above is the substance, the flow is a summary.
    Plus co-change neighbors when a ledger was supplied (correlation)."""
    cid = resolve_config(graph, key)
    if cid is None:
        return {"verdict": "UNKNOWN_PATH", "key": key}
    downstream = _closure(graph, cid, forward=True)
    items = []
    for node, hops in sorted(downstream.items()):
        for e in hops:
            p = e["provenance"]
            if e["kind"] == "co_changes":
                items.append({"to": e["dst"].split(":", 1)[1],
                              "kind": "co_change (correlation, never cause)",
                              "weight": e["weight"],
                              "evidence_ids": p.get("evidence_ids")})
            else:
                items.append({"to": e["dst"].split(":", 1)[1],
                              "kind": "conditional effect",
                              "confidence": p.get("confidence"),
                              "when": p.get("when"),
                              "effect": p.get("effect"),
                              "citation": p.get("citation")})
    # structural layer: articulation over the interaction graph
    adj = _affects_adjacency(graph)
    arts = _graphs.articulation_points(adj)
    structure: Dict[str, Any] = {
        "is_articulation_point": cid in arts,
        "articulation_points": arts,
    }
    if cid in arts:
        # which pairs part: components of the graph minus this node
        reduced = {k: [t for t in v if t != cid]
                   for k, v in adj.items() if k != cid}
        comps = _components(reduced)
        structure["disconnected_groups"] = comps
    # influence flow: confidence-capacity max-flow to each sink
    affects_edges = [e for e in graph["edges"] if e["kind"] == "affects"]
    caps: Dict[str, Dict[str, float]] = {}
    for e in affects_edges:
        caps.setdefault(e["src"], {})[e["dst"]] = e["weight"]
    srcs = {e["src"] for e in affects_edges}
    dsts = {e["dst"] for e in affects_edges}
    sinks = sorted(dsts - srcs)  # terminal nodes of the interaction graph
    flow_total = 0.0
    for s in sinks:
        if s == cid:
            continue
        res = _graphs.edmonds_karp(caps, cid, s)
        flow_total += float(res.get("max_flow", 0.0))
    return {"verdict": "OK", "key": cid.split(":", 1)[1],
            "downstream": items, "structure": structure,
            "influence_flow": round(flow_total, 4),
            "note": _BOUNDED_UNIVERSE_NOTE}


def _components(adj: Dict[str, List[str]]) -> List[List[str]]:
    seen: set = set()
    out: List[List[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(t for t in adj.get(cur, ()) if t not in seen)
        out.append(sorted(comp))
    return out


def explanation_path(graph: Dict[str, Any], src: str,
                     dst: str) -> Dict[str, Any]:
    """Shortest CITED explanation path src -> dst over affects edges.

    Dijkstra on ``-log(confidence)``: the shortest walk is the
    highest-confidence chain. No chain exists -> ``NO_CITED_PATH``
    (abstention — the prompt's rule: never invent an uncited path)."""
    s = resolve_config(graph, src)
    d = resolve_config(graph, dst)
    if s is None or d is None:
        return {"verdict": "UNKNOWN_PATH", "src": src, "dst": dst}
    w: Dict[str, Dict[str, float]] = {}
    by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    for e in graph["edges"]:
        if e["kind"] == "affects":
            w.setdefault(e["src"], {})[e["dst"]] = -math.log(e["weight"])
            by_pair.setdefault((e["src"], e["dst"]), []).append(e)
    res = _graphs.dijkstra(w, s, target=d)
    path = res.get("path")
    if not path:
        return {"verdict": "NO_CITED_PATH", "src": s, "dst": d,
                "note": "no chain of cited interactions connects these "
                        "keys in the curated table; abstaining rather "
                        "than guessing"}
    hops = []
    for a, b in zip(path, path[1:]):
        e = by_pair[(a, b)][0]
        hops.append({"from": a.split(":", 1)[1], "to": b.split(":", 1)[1],
                     "confidence": e["provenance"].get("confidence"),
                     "when": e["provenance"].get("when"),
                     "effect": e["provenance"].get("effect"),
                     "citation": e["provenance"].get("citation")})
    return {"verdict": "OK", "src": s, "dst": d, "path": path,
            "hops": hops, "path_confidence": math.exp(-res["dist"][d]),
            "note": _BOUNDED_UNIVERSE_NOTE}


def related(graph: Dict[str, Any], node_id: str, hops: int = 2,
            top: int = 12) -> Dict[str, Any]:
    """Spreading activation over ALL edge kinds (undirected view).

    Activation decays 0.5 per hop; results are the top-k neighbors with
    their hop distance and the edge kinds that reached them. Bounded by
    construction (frontier shrinks as seen grows); no relevance claim
    beyond graph proximity is made."""
    if not any(n["id"] == node_id for n in graph["nodes"]):
        return {"verdict": "UNKNOWN_NODE", "node": node_id}
    act: Dict[str, float] = {node_id: 1.0}
    reached_by: Dict[str, set] = {}
    visited = {node_id}
    frontier = [node_id]
    for _depth in range(1, max(1, hops) + 1):
        nxt: List[str] = []
        for cur in frontier:
            for e in graph["edges"]:
                if e["src"] == cur:
                    other = e["dst"]
                elif e["dst"] == cur:
                    other = e["src"]
                else:
                    continue
                if other == node_id:
                    continue
                # max-propagate decaying activation (0.5 per hop)
                act[other] = max(act.get(other, 0.0),
                                 act[cur] * 0.5 * e["weight"])
                reached_by.setdefault(other, set()).add(e["kind"])
                if other not in visited:
                    visited.add(other)
                    nxt.append(other)
        frontier = nxt
        if not frontier:
            break
    ranked = sorted(((a, n) for n, a in act.items() if n != node_id),
                    key=lambda t: (-round(t[0], 9), t[1]))[:top]
    return {"verdict": "OK", "node": node_id, "hops": hops,
            "neighbors": [{"node": n, "activation": round(a, 4),
                           "kinds": sorted(reached_by.get(n, set()))}
                          for a, n in ranked]}


def personalized_pagerank(graph: Dict[str, Any],
                          seeds: Optional[Sequence[str]] = None,
                          damping: float = 0.85,
                          iters: int = 50) -> Dict[str, Any]:
    """Personalized PageRank over the SEMANTIC graph (file nodes excluded
    — citation hubs would otherwise absorb all rank; the choice is
    documented, not silent).

    Teleport is uniform over the seed set (tool/config/group/preset/
    rule ids the user cares about) when seeds are given, uniform over
    all nodes otherwise. Edge weights merge parallel edges by MEAN
    (average of evidence strengths). Same weight semantics as
    brain.personal.graph.Graph.pagerank (see module docstring for why
    this variant lives here)."""
    semantic = {n["id"] for n in graph["nodes"] if n["type"] != "file"}
    out: Dict[str, Dict[str, float]] = {n: {} for n in semantic}
    sums: Dict[str, Dict[str, float]] = {n: {} for n in semantic}
    counts: Dict[str, Dict[str, int]] = {n: {} for n in semantic}
    for e in graph["edges"]:
        if e["src"] in semantic and e["dst"] in semantic:
            sums[e["src"]][e["dst"]] = sums[e["src"]].get(e["dst"], 0.0) + e["weight"]
            counts[e["src"]][e["dst"]] = counts[e["src"]].get(e["dst"], 0) + 1
    for s in sums:
        for t, total in sums[s].items():
            out[s][t] = total / counts[s][t]
    nodes = sorted(semantic)
    n = len(nodes)
    if n == 0:
        return {"ranks": {}, "seeds": list(seeds or [])}
    seed_ids = [s for s in (seeds or ()) if s in semantic] or None
    if seed_ids:
        teleport = {v: (1.0 / len(seed_ids) if v in seed_ids else 0.0)
                    for v in nodes}
    else:
        teleport = {v: 1.0 / n for v in nodes}
    rank = dict(teleport)
    for _ in range(iters):
        nxt = {v: 0.0 for v in nodes}
        dangling = 0.0
        for v in nodes:
            targets = out[v]
            if not targets:
                dangling += rank[v]
                continue
            share = rank[v] / len(targets)
            contributed = 0.0
            for t, wgt in targets.items():
                add = share * wgt
                nxt[t] += add
                contributed += add
            # un-followed slack (weights < 1 weaken edges) joins the
            # dangling mass and teleports — brain.personal's semantics
            dangling += rank[v] - contributed
        for v in nodes:
            nxt[v] += damping * dangling * teleport[v] \
                + (1 - damping) * teleport[v]
        total = sum(nxt.values())
        if total > 0:
            for v in nodes:
                nxt[v] /= total
        rank = nxt
    ranks = {v: rank[v] for v in nodes if rank[v] > 1e-9}
    return {"ranks": dict(sorted(ranks.items(), key=lambda t: (-t[1], t[0]))),
            "seeds": seed_ids or [],
            "personalized": bool(seed_ids)}

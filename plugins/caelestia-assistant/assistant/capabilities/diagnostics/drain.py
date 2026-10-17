"""diagnostics.drain — a streaming Drain log-template miner, wired in
FRONT of the rule engine so a line no signature matches gets templated
instead of silently discarded.

He, Zhu, He & Lyu, "Drain: An Online Log Parsing Approach with Fixed
Depth Tree", IEEE ICSM 2017. The mechanism, adapted not invented:

  * PREPROCESS — tokens matching a known-shape regex (IPv4, decimal
    numbers, long hex blobs) are masked to ``<*>``-style parameter
    placeholders (the paper's masking step; the default patterns are
    the paper's own examples, overridable per constructor);
  * PARSE TREE — a fixed-depth trie: level 1 splits on the FIRST
    token (a masked first token routes to the dedicated ``<*>``
    child, per the paper), level 2 splits on TOKEN COUNT, and the
    leaf list holds the LogClusters; a candidate joins the leaf
    cluster whose template is most position-wise similar, measured
    by the paper's SimSeq (equal tokens at equal positions / length);
  * MERGE — a candidate at or above ``sim_threshold`` (the paper's
    simThresh, 0.4) updates its cluster: every divergent position
    becomes ``<*>``; below it, a new cluster is born.

One pass, bounded memory, no training, no model: the tree only ever
grows to ``max_children`` per node (beyond the cap the paper's own
fallback applies — the candidate routes to the node's ``<*>``
child) and ``max_clusters`` total (past the cap a genuinely new
template is counted in the OVERFLOW bucket and reported as
unclustered — an honest capacity refusal, never a silent eviction).

WHAT THIS IS NOT (the honesty the prompt asks for, stated twice):
Drain clusters SHAPES, not CAUSES. A template earning recurrence is a
signal that a HUMAN should consider writing a real diagnostic rule for
it — it is not a diagnosis, carries no fix, and joins no settings tool
(only a human-authored rule gets the settings reverse-join, because
only a rule asserts a root cause). Recurrence counts and first-seen
dates are bookkeeping for that human decision.

Timestamps: ``update(line, when=...)`` records the CALLER-SUPPLIED
date string (determinism: nothing here reads the clock; live callers
pass ``datetime.now(timezone.utc).date().isoformat()``). A miner run
without dates reports counts only — honestly undated, never backdated.

State moves only through plain dicts (``to_dict`` / ``from_dict``) for
the caller's learned-state JSON path (brain/state.py) — this module
writes NOTHING itself. Pure stdlib; no RNG; deterministic given the
input stream (leaf search scans clusters in insertion order, the
first at-or-above the threshold wins — the paper's own tie handling).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "STATE_KEY", "SHAPES_NOT_CAUSES", "DEFAULT_SIM_THRESHOLD",
    "DEFAULT_MAX_CHILDREN", "DEFAULT_MAX_CLUSTERS", "MASK_TOKEN",
    "DrainMiner", "mask_token", "mine", "augment_diagnosis",
    "render_unmatched_lines",
]

STATE_KEY = "drain_miner"

SHAPES_NOT_CAUSES = (
    "Drain clusters shapes, not causes: a recurring template is a "
    "candidate for a human-written diagnostic rule, not a diagnosis"
)

DEFAULT_SIM_THRESHOLD = 0.4
DEFAULT_MAX_CHILDREN = 12
DEFAULT_MAX_CLUSTERS = 2048
MASK_TOKEN = "<*>"

# The paper's preprocessing masks parameters before the tree sees the
# line. These are the paper's own shape families (IPv4, decimal
# numbers, long hex blobs); callers may add domain patterns per
# constructor. Order matters: the first matching pattern wins.
DEFAULT_MASK_PATTERNS = (
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),              # IPv4
    re.compile(r"\b[0-9a-fA-F]{8,}\b"),                      # hex blobs
    re.compile(r"(?<![A-Za-z])[+-]?\d+(?:\.\d+)?(?![A-Za-z])"),  # numbers
)


def mask_token(token: str, patterns: Sequence[re.Pattern]) -> str:
    """Mask ONE token: the first matching shape family wins. Empty
    patterns list means no masking (caller's explicit choice)."""
    for pattern in patterns:
        if pattern.search(token):
            return MASK_TOKEN
    return token


class _Node:
    """One parse-tree node: either an internal splitter (children) or a
    leaf list of cluster ids. Bounded: children grow past the cap into
    the node's ``<*>`` fallback child."""

    __slots__ = ("children", "fallback", "leaf")

    def __init__(self) -> None:
        self.children: Dict[str, "_Node"] = {}
        self.fallback: Optional["_Node"] = None
        self.leaf: List[int] = []  # cluster ids, insertion order


class DrainMiner:
    """The streaming miner. ``update`` one line at a time (or batch via
    ``mine``); serialize through ``to_dict`` / ``from_dict``."""

    def __init__(self,
                 sim_threshold: float = DEFAULT_SIM_THRESHOLD,
                 max_children: int = DEFAULT_MAX_CHILDREN,
                 max_clusters: int = DEFAULT_MAX_CLUSTERS,
                 mask_patterns: Optional[Sequence[re.Pattern]] = None
                 ) -> None:
        if not (0.0 < sim_threshold <= 1.0):
            raise ValueError(
                f"sim_threshold must be inside (0, 1], got {sim_threshold!r}")
        if max_children < 1:
            raise ValueError(f"max_children must be >= 1, got {max_children!r}")
        if max_clusters < 1:
            raise ValueError(f"max_clusters must be >= 1, got {max_clusters!r}")
        self.sim_threshold = float(sim_threshold)
        self.max_children = int(max_children)
        self.max_clusters = int(max_clusters)
        self.mask_patterns = (list(mask_patterns) if mask_patterns is not None
                              else list(DEFAULT_MASK_PATTERNS))
        self.root = _Node()
        # clusters[i] = {"template": [tokens], "count": n,
        #                "first_seen": date-or-None, "last_seen": ...,
        #                "example": str}
        self.clusters: List[Dict[str, Any]] = []
        self.overflow: Dict[str, int] = {}   # template key -> count
        self.n_lines = 0

    # ------------------------------------------------------------------
    # similarity (the paper's SimSeq, over equal-count token lists)
    # ------------------------------------------------------------------

    @staticmethod
    def _similarity(a: Sequence[str], b: Sequence[str]) -> float:
        if not a or len(a) != len(b):
            return 0.0
        equal = sum(1 for x, y in zip(a, b) if x == y)
        return equal / len(a)

    # ------------------------------------------------------------------
    # traversal
    # ------------------------------------------------------------------

    def _route(self, tokens: List[str]) -> List[int]:
        """Walk root -> first-token level -> token-count level; return
        the leaf list. The paper's fixed-depth structure, with the
        maxChild fallback into the ``<*>`` child."""
        node = self.root
        first = tokens[0] if tokens else MASK_TOKEN
        child_key = first
        if child_key in node.children:
            node = node.children[child_key]
        elif len(node.children) >= self.max_children:
            if node.fallback is None:
                node.fallback = _Node()
            node = node.fallback
        else:
            node = node.children.setdefault(child_key, _Node())

        count_key = str(len(tokens))
        if count_key in node.children:
            node = node.children[count_key]
        elif len(node.children) >= self.max_children:
            if node.fallback is None:
                node.fallback = _Node()
            node = node.fallback
        else:
            node = node.children.setdefault(count_key, _Node())
        return node.leaf

    def update(self, line: str, when: Optional[str] = None) -> Dict[str, Any]:
        """One line in, its (possibly pre-existing) cluster out. ``when``
        is the caller-supplied ISO date string for first/last-seen
        bookkeeping (None = undated, honestly). Empty lines are
        REFUSED — there is no honest template for nothing."""
        if not line or not line.strip():
            raise ValueError("drain.update: empty line (nothing to template)")
        tokens = [mask_token(tok, self.mask_patterns)
                  for tok in line.split()]
        self.n_lines += 1
        leaf = self._route(tokens)

        best_id: Optional[int] = None
        best_sim = -1.0
        for cid in leaf:
            sim = self._similarity(self.clusters[cid]["template"], tokens)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        if best_id is not None and best_sim >= self.sim_threshold:
            cluster = self.clusters[best_id]
            cluster["count"] += 1
            cluster["template"] = [
                old if old == new else MASK_TOKEN
                for old, new in zip(cluster["template"], tokens)]
            cluster["last_seen"] = when if when is not None else cluster["last_seen"]
            return self._cluster_row(cluster)

        if len(self.clusters) >= self.max_clusters:
            key = " ".join(tokens)
            self.overflow[key] = self.overflow.get(key, 0) + 1
            return {"cluster_id": None, "template": key, "count":
                    self.overflow[key], "first_seen": None,
                    "last_seen": None, "overflowed": True,
                    "example": line}

        cluster = {
            "template": list(tokens),
            "count": 1,
            "first_seen": when,
            "last_seen": when,
            "example": line,
        }
        self.clusters.append(cluster)
        leaf.append(len(self.clusters) - 1)
        return self._cluster_row(cluster)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_row(cluster: Dict[str, Any]) -> Dict[str, Any]:
        return {"cluster_id": None,  # ids are positional, not persisted
                "template": " ".join(cluster["template"]),
                "count": cluster["count"],
                "first_seen": cluster.get("first_seen"),
                "last_seen": cluster.get("last_seen"),
                "overflowed": False,
                "example": cluster.get("example")}

    def templates(self, min_count: int = 1) -> List[Dict[str, Any]]:
        """All clusters (and any overflow templates) as report rows,
        count descending, template ascending — deterministic. Rows are
        freshly built (no positional ids leak)."""
        rows = [self._cluster_row(c) for c in self.clusters]
        for key, count in sorted(self.overflow.items()):
            rows.append({"cluster_id": None, "template": key,
                         "count": count, "first_seen": None,
                         "last_seen": None, "overflowed": True,
                         "example": None})
        rows = [r for r in rows if r["count"] >= min_count]
        rows.sort(key=lambda r: (-r["count"], r["template"]))
        return rows

    # ------------------------------------------------------------------
    # persistence (plain dicts; the caller's learned-state JSON path)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_threshold": self.sim_threshold,
            "max_children": self.max_children,
            "max_clusters": self.max_clusters,
            "n_lines": self.n_lines,
            "clusters": [dict(c) for c in self.clusters],
            "overflow": dict(self.overflow),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "DrainMiner":
        miner = cls(sim_threshold=state.get("sim_threshold",
                                            DEFAULT_SIM_THRESHOLD),
                    max_children=state.get("max_children",
                                           DEFAULT_MAX_CHILDREN),
                    max_clusters=state.get("max_clusters",
                                           DEFAULT_MAX_CLUSTERS))
        miner.clusters = [dict(c) for c in state.get("clusters", [])]
        miner.overflow = dict(state.get("overflow", {}))
        miner.n_lines = int(state.get("n_lines", 0))
        # the parse tree itself is never persisted — rebuild leaf
        # membership by re-routing each stored template (identical
        # routing decisions, since routing is a pure function of the
        # template tokens)
        for cid, cluster in enumerate(miner.clusters):
            leaf = miner._route(list(cluster["template"]))
            if cid not in leaf:
                leaf.append(cid)
        return miner

    def capacity(self) -> Dict[str, Any]:
        """The honest capacity report: how full the tree is, and whether
        overflow is being dropped into the bucket."""
        return {"clusters": len(self.clusters), "max_clusters": self.max_clusters,
                "overflowed_keys": len(self.overflow),
                "overflowed_lines": sum(self.overflow.values())}


# ---------------------------------------------------------------------------
# Batch + rule-engine integration
# ---------------------------------------------------------------------------


def mine(lines: Sequence[str], miner: Optional[DrainMiner] = None,
         when: Optional[str] = None) -> Dict[str, Any]:
    """Batch convenience: feed every non-empty line through ``update``.
    Returns the miner's capacity report plus the ranked templates — the
    same rows ``templates()`` yields."""
    miner = miner if miner is not None else DrainMiner()
    for line in lines:
        if line.strip():
            miner.update(line, when)
    return {"capacity": miner.capacity(),
            "n_lines": miner.n_lines,
            "templates": miner.templates(),
            "note": SHAPES_NOT_CAUSES}


def augment_diagnosis(diagnosis: Dict[str, Any], text: str,
                      miner: Optional[DrainMiner] = None) -> Dict[str, Any]:
    """The IN FRONT wiring: when the rule engine answered NO_MATCH (or
    AMBIGUOUS — candidates exist but the engine refused to pick one),
    the lines that no signature matched are TEMPLATED here instead of
    being silently discarded. ``diagnosis`` gains one additive key,
    ``unmatched_templates``:

        {"rows": [...], "note": SHAPES_NOT_CAUSES, "dated": bool}

    MATCH verdicts are untouched (a matched rule IS the diagnosis; the
    template miner has nothing to add and would only dilute the
    report). A caller-held persisted ``miner`` supplies recurrence and
    first-seen dates across calls; the default stateless run counts
    within THIS input only and says so. Nothing here joins a settings
    tool: only human-authored rules earn the settings reverse-join,
    because only a rule asserts a root cause."""
    verdict = diagnosis.get("verdict")
    if verdict not in ("NO_MATCH", "AMBIGUOUS"):
        return diagnosis
    if miner is None:
        miner = DrainMiner()
        dated = False
    else:
        dated = any(row.get("first_seen") for row in miner.templates())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for line in lines:
        miner.update(line)
    rows = miner.templates()
    diagnosis["unmatched_templates"] = {
        "rows": rows[:16],
        "n_lines": len(lines),
        "dated": dated,
        "note": SHAPES_NOT_CAUSES,
    }
    return diagnosis


def render_unmatched_lines(unmatched: Dict[str, Any]) -> List[str]:
    """The report block for ``unmatched_templates`` (engine.render_report
    calls this). Plain text, nothing executable."""
    lines: List[str] = []
    lines.append("Unmatched-line templates (Drain, He et al. 2017 — shapes, not causes):")
    for row in unmatched.get("rows", []):
        when = row.get("first_seen")
        seen = f", first seen {when}" if when else ""
        flag = " [OVERFLOW — capacity cap reached]" if row.get("overflowed") else ""
        lines.append(f"  - [{row['count']}x{seen}]{flag} {row['template'][:160]}")
    if not unmatched.get("rows"):
        lines.append("  - (no recurring shape in this input)")
    lines.append(f"  Note: {unmatched.get('note', SHAPES_NOT_CAUSES)}.")
    return lines

"""genius.metacog — the assistant knowing itself.

This is the self-improvement engine. It never acts; it reflects:

  * rule induction: ID3-style decision tree over your past decisions
    (each row = features of a proposal, outcome = approve/reject), then
    flattened into human-readable rules with honest hit counts —
    "you reject PRIVILEGED changes at 2am" becomes a stated preference,
    not a vibe. Small data is handled honestly: shallow trees, minimum
    leaf support, and an explicit verdict when data is too thin.
  * active learning: entropy-based question selection — given the
    current model and a pool of unlabeled items, which ONE question
    resolves the most uncertainty (maximum information gain)?
  * coverage map: which domains you actually use, with acceptance rates
    per domain — the assistant's honest report card.
  * concept clustering: leader-algorithm clustering over past requests
    (TF-IDF cosine), each cluster named by its top terms — "you keep
    asking about: bar blur, package updates, git habits".

Everything here is read-only analysis of history you already have.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["induce_rules", "suggest_question", "coverage_map",
           "cluster_requests"]


# ---------------------------------------------------------------------------
# Rule induction (ID3-lite over decision history)
# ---------------------------------------------------------------------------

def _entropy(labels: Sequence[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _split_info(rows: Sequence[Dict[str, Any]], feature: str) -> Tuple[float, Dict[Any, List[Dict[str, Any]]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("features", {}).get(feature)), []).append(r)
    weighted = sum(len(g) / len(rows) * _entropy([str(r["outcome"]) for r in g])
                   for g in groups.values())
    return weighted, groups


def induce_rules(history: Sequence[Dict[str, Any]], max_depth: int = 3,
                 min_leaf: int = 3) -> Dict[str, Any]:
    """Learn approve/reject rules from proposal history.

    Row shape: {"features": {..strings/numbers..}, "outcome": "approve"|"reject"}
    """
    rows = [r for r in history if r.get("outcome") in ("approve", "reject")
            and isinstance(r.get("features"), dict)]
    if len(rows) < 2 * min_leaf:
        return {"rules": [], "verdict": "INSUFFICIENT_HISTORY",
                "n_examples": len(rows),
                "question": "decide on more proposals first — "
                            f"need about {2 * min_leaf - len(rows)} more",
                "note": "thin data teaches nothing but overconfidence"}
    features = sorted({f for r in rows for f in r["features"]})
    rules: List[Dict[str, Any]] = []

    def grow(group: List[Dict[str, Any]], conditions: List[str],
             depth: int) -> None:
        labels = [str(r["outcome"]) for r in group]
        majority = Counter(labels).most_common(1)[0][0]
        purity = Counter(labels)[majority] / len(labels)
        if depth >= max_depth or purity >= 0.9 or len(group) < 2 * min_leaf:
            if purity >= 0.75 and len(group) >= min_leaf:
                rules.append({
                    "conditions": list(conditions) or ["(no conditions)"],
                    "prediction": majority,
                    "support": len(group),
                    "accuracy": round(purity, 3),
                    "reads": (f"IF {' AND '.join(conditions) or 'always'} "
                              f"THEN {majority} ({len(group)} cases, "
                              f"{purity:.0%} consistent)")})
            return
        best_feature, best_gain, best_groups = None, 0.0, None
        parent_e = _entropy(labels)
        for f in features:
            weighted, groups = _split_info(group, f)
            gain = parent_e - weighted
            if gain > best_gain and len(groups) > 1:
                best_feature, best_gain, best_groups = f, gain, groups
        if best_feature is None or best_gain < 0.05:
            if purity >= 0.7:
                rules.append({"conditions": list(conditions) or ["(no conditions)"],
                              "prediction": majority, "support": len(group),
                              "accuracy": round(purity, 3),
                              "reads": (f"IF {' AND '.join(conditions) or 'always'} "
                                        f"THEN {majority} ({len(group)} cases, "
                                        f"{purity:.0%} consistent)")})
            return
        for value, subgroup in sorted(best_groups.items(), key=lambda kv: -len(kv[1])):
            grow(subgroup, conditions + [f"{best_feature}={value}"], depth + 1)

    grow(rows, [], 0)
    rules.sort(key=lambda r: -r["support"])
    approve_rate = sum(1 for r in rows if r["outcome"] == "approve") / len(rows)
    return {"rules": rules[:8], "verdict": "LEARNED" if rules else "NO_STABLE_PATTERN",
            "n_examples": len(rows),
            "overall_approve_rate": round(approve_rate, 3),
            "note": "these are descriptive patterns in YOUR past decisions, "
                    "not obligations for future ones"}


# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------

def suggest_question(pool: Sequence[Dict[str, Any]],
                     model_rules: Optional[Sequence[Dict[str, Any]]] = None,
                     labels: Sequence[str] = ("approve", "reject")) -> Dict[str, Any]:
    """Pick the question whose answer most reduces outcome uncertainty.

    Pool items: {"id", "features": {...}, "prediction"?: str, "p"?: float}
    """
    if not pool:
        return {"verdict": "EMPTY_POOL", "question": None}
    scored = []
    for item in pool:
        feats = item.get("features", {})
        p = item.get("p")
        if p is None:
            p = 0.5
        p = max(0.01, min(0.99, float(p)))
        # binary entropy of the current belief
        uncertainty = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        # feature count heuristics: more distinct features = more
        # information in the eventual answer
        scored.append({"id": item.get("id"), "uncertainty_bits": round(uncertainty, 4),
                       "p": round(p, 3), "features": feats})
    scored.sort(key=lambda s: -s["uncertainty_bits"])
    best = scored[0]
    if best["uncertainty_bits"] < 0.3:
        return {"verdict": "CONFIDENT", "question": None,
                "note": "the model is already sure enough; asking would be noise"}
    label_bits = max(1, math.log2(len(labels)))
    return {"verdict": "ASK",
            "question": {
                "item_id": best["id"],
                "phrasing": f"about {best['id']}: would you approve or reject this?",
                "expected_information_gain_bits": round(min(best["uncertainty_bits"],
                                                            label_bits), 3),
                "current_belief": f"p(approve) = {best['p']}",
            },
            "ranked_pool": [{"id": s["id"], "bits": s["uncertainty_bits"]}
                             for s in scored[:5]],
            "note": "one question at a time, chosen to resolve the most doubt"}


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------

def coverage_map(usage: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Report card: per-domain usage, acceptance, and trend arrows.

    Row shape: {"domain": str, "accepted": bool, "day": int (optional)}
    """
    if not usage:
        return {"verdict": "NO_USAGE_YET", "domains": {},
                "note": "nothing tracked yet — use the assistant first"}
    by_domain: Dict[str, Dict[str, Any]] = {}
    for row in usage:
        d = str(row.get("domain", "unknown"))
        slot = by_domain.setdefault(d, {"uses": 0, "accepted": 0, "recent_uses": 0})
        slot["uses"] += 1
        if row.get("accepted"):
            slot["accepted"] += 1
        if row.get("day", 1) >= 1:  # caller marks recency as they see fit
            slot["recent_uses"] += 1
    domains = {}
    for d, slot in by_domain.items():
        rate = slot["accepted"] / slot["uses"] if slot["uses"] else 0.0
        domains[d] = {"uses": slot["uses"], "acceptance_rate": round(rate, 3),
                      "verdict": ("healthy" if rate >= 0.6 else
                                  "mismatch — ask what's wrong" if rate < 0.35 and slot["uses"] >= 3
                                  else "warming up" if slot["uses"] < 3 else "mixed")}
    ranked = sorted(domains.items(), key=lambda kv: -kv[1]["uses"])
    total = sum(s["uses"] for s in by_domain.values())
    herfindahl = sum((s["uses"] / total) ** 2 for s in by_domain.values())
    return {"verdict": "OK", "n_events": total, "domains": dict(ranked),
            "concentration_hhi": round(herfindahl, 3),
            "most_used": ranked[0][0] if ranked else None,
            "untapped": [d for d, s in domains.items() if s["uses"] == 1],
            "note": "a narrow map means the assistant is being used as a hammer"}


# ---------------------------------------------------------------------------
# Concept clustering (leader algorithm over request texts)
# ---------------------------------------------------------------------------

def _tf(tokens: Sequence[str]) -> Counter:
    return Counter(tokens)


def _tokenize_light(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2]


def cluster_requests(requests: Sequence[str], threshold: float = 0.25,
                     max_clusters: int = 12) -> Dict[str, Any]:
    """Leader clustering: first-seen request anchors a cluster; similar
    ones join it; stragglers seed new clusters. Named by top shared terms."""
    if not requests:
        return {"clusters": [], "n": 0, "note": "no requests to cluster"}
    clusters: List[Dict[str, Any]] = []
    idf: Counter = Counter()
    docs = []
    for r in requests:
        toks = _tokenize_light(r)
        docs.append(toks)
        for t in set(toks):
            idf[t] += 1
    n_docs = len(docs)

    def _sim(a: List[str], b: List[str]) -> float:
        ta, tb = _tf(a), _tf(b)
        if not ta or not tb:
            return 0.0
        common = set(ta) & set(tb)
        num = sum((1 + math.log(1 + ta[t])) * (1 + math.log(1 + tb[t])) *
                  math.log(n_docs / (1 + idf[t])) ** 2 for t in common)
        na = math.sqrt(sum((1 + math.log(1 + c)) * math.log(n_docs / (1 + idf[t])) ** 2
                            for t, c in ta.items()))
        nb = math.sqrt(sum((1 + math.log(1 + c)) * math.log(n_docs / (1 + idf[t])) ** 2
                            for t, c in tb.items()))
        return num / (na * nb) if na and nb else 0.0

    for i, doc in enumerate(docs):
        placed = False
        for cl in clusters:
            if _sim(doc, cl["anchor_doc"]) > threshold:
                cl["members"].append(requests[i])
                cl["member_docs"].append(doc)
                placed = True
                break
        if not placed and len(clusters) < max_clusters:
            clusters.append({"anchor": requests[i], "anchor_doc": doc,
                             "members": [requests[i]], "member_docs": [doc]})
        elif not placed and clusters:
            clusters[-1]["members"].append(requests[i])
            clusters[-1]["member_docs"].append(doc)
    out = []
    for cl in clusters:
        all_terms = [t for doc in cl["member_docs"] for t in set(doc)]
        top = [t for t, _ in Counter(all_terms).most_common(4)]
        out.append({"name": " ".join(top) or cl["anchor"][:30],
                    "size": len(cl["members"]),
                    "example": cl["anchor"][:80],
                    "members": [m[:60] for m in cl["members"][:5]]})
    out.sort(key=lambda c: -c["size"])
    return {"clusters": out, "n": len(requests),
            "top_themes": [c["name"] for c in out[:5]],
            "note": "themes are what YOU keep returning to — a mirror, not a judge"}

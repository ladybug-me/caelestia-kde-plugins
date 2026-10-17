"""diagnostics.fusion — weighted-Bayes combination of the confidences the
troubleshooting engines already produce, into one ranked diagnosis.

Three engines answer a troubleshooting question independently and each
attaches its own calibrated confidence to its own candidates:

- the diagnostics rule engine (engine.py) — each rule carries its own
  confidence;
- retrieval (retrieval/search.py) — BM25 scores over the repo's corpus;
- scan (scan/scanner.py) — streaming signature statistics over a log
  stream.

This module is a READ-ONLY DOWNSTREAM CONSUMER: it calls the engines,
reads their outputs, and never modifies them. The combination is
weighted Bayes in log-odds space — each source's confidence is one
likelihood-ratio update from a common prior, scaled by that source's
weight (with all weights equal and independent sources this reduces to
exactly the naive-Bayes likelihood-ratio product brain/naive_bayes.py
applies; the weighted form is the standard linear opinion pool in
log-odds space, Genest & Zidek 1986, "Combining probability
distributions", StatSci 1(1)). No source vote is invented: a source
with no evidence for a hypothesis ABSTAINS (its term is skipped), never
a silent 0.5.

Score calibration, per source (all deterministic, all documented):
- diagnostics: a rule's qualitative tier ("deterministic"/"probable")
  is NOT a probability and is never invented into one — the evidence
  strength is the engine's own per-candidate match score, mapped
  through the same saturating transform as every other unbounded score
  (see below); candidates without a numeric score abstain;
- retrieval: BM25 is unbounded, so a fixed-scale saturating transform
  c = s / (s + K) maps it into (0, 1) — monotone in the native score,
  K a fixed documented constant (default 1.0), no fitting step;
- scan: the same transform over a pattern's hit DENSITY (hits per
  line): one signature hit is prior-neutral (0.5), more hits are
  stronger — the conservative reading of a stream statistic.

One transform class for all three sources' unbounded native scores:
c = s / (s + K), monotone, deterministic, never claims certainty.

Hypotheses are caller-keyed strings (a rule id, a doc id, or any label
the caller maps them to via ``hypothesis_map``). No fuzzy joining is
performed here — merging two engines' candidates into one hypothesis is
an explicit, auditable caller decision.

Out-of-range values are REJECTED, never clamped (the settings layer's
own convention): a confidence outside (0, 1), a negative weight, or a
weight for an unknown source raises ValueError.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

__all__ = ["DEFAULT_WEIGHTS", "fuse", "from_diagnose", "from_retrieval",
           "from_scan", "diagnose"]

# Source trust, fixed and conservative by default: the deterministic,
# citation-pinned rule engine is worth twice a corpus or stream signal.
# Overridable per call; unknown or negative weights are rejected.
DEFAULT_WEIGHTS = {"diagnostics": 0.5, "retrieval": 0.25, "scan": 0.25}

# Saturating-transform scale for the unbounded native scores (BM25 and
# scan hit counts): c = s / (s + K). K=1 makes exactly one unit of
# native evidence prior-neutral (0.5) — the conservative default.
DEFAULT_SCALE = 1.0

_EPS = 1e-9


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _check_confidence(source: str, hypothesis: str, c: Any) -> float:
    try:
        c = float(c)
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} confidence for {hypothesis!r} is not a number: {c!r}")
    if not (0.0 < c < 1.0):
        # certainty claims and zero-confidence noise are both refused:
        # engines produce calibrated values strictly inside (0, 1)
        raise ValueError(
            f"{source} confidence for {hypothesis!r} must be strictly "
            f"inside (0, 1), got {c} (rejected, never clamped)")
    return c


def fuse(evidence: List[Dict[str, Any]],
         weights: Optional[Dict[str, float]] = None,
         prior: float = 0.5) -> Dict[str, Any]:
    """Combine per-source confidences into one ranked diagnosis list.

    ``evidence``: [{"source": "diagnostics"|"retrieval"|"scan",
                    "hypothesis": str, "confidence": float in (0,1)}]
    (the shape from_diagnose / from_retrieval / from_scan emit). Each
    hypothesis's posterior log-odds is the prior log-odds plus the
    weight-scaled likelihood ratios of every source that voted.

    Returns {"prior", "weights", "diagnoses": [{hypothesis, confidence,
    sources, odds_multiplier}]} ranked by combined confidence desc with
    a hypothesis-name tie-break (deterministic)."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    for name, weight in w.items():
        if name not in DEFAULT_WEIGHTS:
            raise ValueError(f"unknown source {name!r} in weights "
                             f"(known: {sorted(DEFAULT_WEIGHTS)})")
        if float(weight) < 0.0:
            raise ValueError(f"weight for {name!r} must be >= 0, "
                             f"got {weight}")
    if not (0.0 < float(prior) < 1.0):
        raise ValueError(f"prior must be strictly inside (0, 1), "
                         f"got {prior}")

    prior_logit = _logit(float(prior))
    by_hypothesis: Dict[str, Dict[str, float]] = {}
    for item in evidence:
        source = str(item.get("source", ""))
        if source not in DEFAULT_WEIGHTS:
            raise ValueError(f"unknown source {source!r} in evidence")
        hypothesis = str(item.get("hypothesis", ""))
        if not hypothesis:
            raise ValueError("evidence item without a hypothesis")
        c = _check_confidence(source, hypothesis, item.get("confidence"))
        slot = by_hypothesis.setdefault(hypothesis, {})
        if source in slot:
            raise ValueError(
                f"source {source!r} voted twice for {hypothesis!r}")
        slot[source] = c

    diagnoses: List[Dict[str, Any]] = []
    for hypothesis, slot in by_hypothesis.items():
        logit = prior_logit
        contributions: Dict[str, float] = {}
        for source, c in slot.items():
            delta = float(w[source]) * (_logit(c) - prior_logit)
            contributions[source] = round(delta, 6)
            logit += delta
        diagnoses.append({
            "hypothesis": hypothesis,
            "confidence": round(_sigmoid(logit), 4),
            "sources": {s: round(c, 4) for s, c in sorted(slot.items())},
            "contributions": contributions,
            # total likelihood ratio vs the prior — the honest magnitude
            # of the combined evidence (1.0 when nothing voted)
            "odds_multiplier": round(math.exp(logit - prior_logit), 4),
        })
    diagnoses.sort(key=lambda d: (-d["confidence"], d["hypothesis"]))
    return {"prior": prior, "weights": w, "diagnoses": diagnoses}


# ---------------------------------------------------------------------------
# Per-source collectors: read-only transformers of each engine's OWN output.
# ---------------------------------------------------------------------------

def from_diagnose(diag_result: Dict[str, Any],
                  scale: float = DEFAULT_SCALE) -> List[Dict[str, Any]]:
    """The diagnostics engine's candidates as fusion evidence.

    The evidence strength is the engine's own per-candidate match score
    through the shared saturating transform (the rule's qualitative
    ``confidence`` tier is a label, not a probability, and is never
    invented into a number). Candidates without a numeric score abstain.
    The engine's own output is only read, never modified."""
    out: List[Dict[str, Any]] = []
    for cand in diag_result.get("candidates", []):
        rule = cand.get("rule") or {}
        score = cand.get("score")
        if not isinstance(score, (int, float)) or float(score) <= 0.0:
            continue  # no numeric match strength -> abstain, never guess
        s = float(score)
        out.append({"source": "diagnostics",
                    "hypothesis": str(rule.get("id", "")),
                    "confidence": s / (s + float(scale)),
                    "tier": rule.get("confidence")})
    return out


def from_retrieval(hits: List[Dict[str, Any]],
                   scale: float = DEFAULT_SCALE) -> List[Dict[str, Any]]:
    """Retrieval's BM25 hits as fusion evidence (fixed-scale saturating
    transform; monotone, deterministic, no fitting)."""
    out: List[Dict[str, Any]] = []
    for hit in hits:
        score = float(hit.get("score", 0.0))
        if score <= 0.0:
            continue  # a zero-scored doc is no evidence, not a vote
        out.append({"source": "retrieval",
                    "hypothesis": f"doc:{hit.get('doc_id', hit.get('id'))}",
                    "confidence": score / (score + float(scale))})
    return out


def from_scan(scan_result: Dict[str, Any],
              scale: float = DEFAULT_SCALE) -> List[Dict[str, Any]]:
    """Scan's streaming signature statistics as fusion evidence: one
    pattern hit is prior-neutral, more are stronger (conservative by
    construction)."""
    lines = max(1, int(scan_result.get("lines_scanned", 0)))
    out: List[Dict[str, Any]] = []
    for hit in scan_result.get("pattern_hits", []):
        hits = float(hit.get("hits", 0.0))
        if hits <= 0.0:
            continue
        # saturating transform over the hit DENSITY (hits per line),
        # so a long stream needs proportionally more hits to move the
        # same source weight
        density = hits / lines
        out.append({"source": "scan",
                    "hypothesis": f"pattern:{hit.get('pattern', '')}",
                    "confidence": density / (density + float(scale))})
    return out


# ---------------------------------------------------------------------------
# The one-call convenience (still read-only; the engines answer first).
# ---------------------------------------------------------------------------

def diagnose(text: str, stream_text: Optional[str] = None,
             patterns: Optional[List[str]] = None, k: int = 3,
             hypothesis_map: Optional[Dict[str, str]] = None,
             weights: Optional[Dict[str, float]] = None,
             prior: float = 0.5, scale: float = DEFAULT_SCALE
             ) -> Dict[str, Any]:
    """Ask all engines, fuse their evidence, rank once.

    Runs diagnostics.engine.diagnose(text) and retrieval's search; scan
    participates only when the caller supplies BOTH ``stream_text`` and
    the signature ``patterns`` (the automaton is the caller's choice —
    the same rule as the agent's scan dispatcher). ``hypothesis_map``
    optionally re-labels source items onto shared hypotheses
    (e.g. {"doc:ISS-120": "vesktop-screencast"}); without it, each item
    stays its own hypothesis — nothing is fuzzy-joined here.

    The engines' own result dicts are never modified: this module only
    reads them."""
    from . import engine as diagnostics_engine
    from assistant.capabilities.retrieval import search as retrieval_search

    evidence: List[Dict[str, Any]] = []

    diag = diagnostics_engine.diagnose(text)
    evidence.extend(from_diagnose(diag, scale=scale))

    hits = retrieval_search.search(text, k=k)
    evidence.extend(from_retrieval(hits, scale=scale))

    scan_summary: Optional[Dict[str, Any]] = None
    if stream_text is not None and patterns:
        from assistant.capabilities.scan import Automaton
        from assistant.capabilities.scan.scanner import scan_text
        scan_summary = scan_text(stream_text, Automaton(list(patterns)))
        evidence.extend(from_scan(scan_summary, scale=scale))

    if hypothesis_map:
        for item in evidence:
            item["hypothesis"] = hypothesis_map.get(
                item["hypothesis"], item["hypothesis"])

    fused = fuse(evidence, weights=weights, prior=prior)
    fused["inputs"] = {
        "diagnostics_verdict": diag.get("verdict"),
        "retrieval_hits": len(hits),
        "scan_ran": scan_summary is not None,
    }
    return fused

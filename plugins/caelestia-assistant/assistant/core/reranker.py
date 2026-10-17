"""cortex.reranker — a LEARNED pairwise re-ranker layered on the existing
router ranking, never replacing it.

What exists already (grep-first, none of it replaced):
- ``cortex/router.py`` — the four-signal hybrid ranker; every candidate
  already carries its per-signal components in ``RouteResult.features``
  (lex / sem / fuzz / noun / cue / coverage) — the per-PAIR features
  this module learns over, no new feature engineering;
- ``cortex/conformal.py::ConformalCalibrator`` — the repo's ONE
  uncertainty mechanism, the same gate slot_tagger uses;
- ``cortex/slot_tagger.py`` — the structural template this module
  mirrors: averaged perceptron, 2-best margin as confidence, conformal
  gate, honest fallback with the reason attached.

The gap this closes: the router's signal WEIGHTS are global (one
RouterState for every request). A user whose history consistently
prefers a different trade-off between the signals than the hand-set
prior expresses has no way for that preference to move the RANKING —
until now. This module trains a pairwise max-margin perceptron
(Rosenblatt 1958's perceptron update applied to pairwise preference
constraints; the pairwise max-margin ranking formulation is Herbrich,
Graepel & Obermayer 2000, "Large Margin Rank Boundaries for Ordinal
Regression"; averaged weights per Collins 2002) over the router's own
per-candidate features, using ONLY local approve/reject routing
history — the same source cortex/learn.py and slot_tagger.py train
from. No external corpus exists anywhere in this module:

- APPROVED rows supervise: the surface the user accepted must outrank
  the alternatives the router scored for the SAME request (one pairwise
  constraint per competitor); the update is w += phi(approved) -
  phi(competitor) on every violated pair — a weighted update in the
  perceptron family, not a new model;
- REJECTED rows supervise NOTHING here (the slot_tagger rule, same
  reason: a request the user refused says nothing about what they
  wanted instead — it is skipped, never fabricated into a constraint).

The gate (the safety property): the re-ranked top-1-vs-top-2 margin
becomes a confidence through the same fixed-scale logistic transform
class slot_tagger uses, and the conformal calibrator decides whether
that confidence is covered — fed from THIS signal's own (confidence,
outcome) stream, as every calibrator consumer feeds its own. Not
covered — or NO calibration data at all — and the request falls back
to the router's own ranking with the calibrator's reason attached. The
re-ranker adds preference; it never guesses.

Pure: no I/O, no clock, no randomness (fixed update order, averaged
weights); identical input bytes -> identical output. Persistence is a
plain dict for the caller to store in learned-state JSON (an existing
write path); this module writes nothing.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from .conformal import ConformalCalibrator

__all__ = ["FEATURES", "Reranker", "supervision_from_history", "rerank_gated"]

# The router's own per-candidate signal components (router.py builds
# exactly these keys into RouteResult.features). A fixed feature space:
# unknown keys in a candidate's dict are ignored; missing keys read 0.0
# (the no-evidence value, never a guess).
FEATURES = ("lex", "sem", "fuzz", "noun", "cue", "coverage")


def _margin_confidence(margin: float) -> float:
    """Top1-vs-top2 re-ranker margin -> (0, 1) confidence through the
    same fixed-scale logistic class slot_tagger documents (monotone,
    deterministic, never claims certainty). Gamma pinned so a margin of
    ~1.0 score unit — a full signal's worth — is already confident."""
    gamma = 2.0
    return 1.0 / (1.0 + math.exp(-gamma * margin))


class Reranker:
    """Averaged pairwise perceptron over the router's feature space."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data if isinstance(data, dict) else {}
        self.weights: Dict[str, float] = {f: 0.0 for f in FEATURES}
        self._totals: Dict[str, float] = {f: 0.0 for f in FEATURES}
        self._steps = 0
        self.updates = int(data.get("updates", 0))
        self.trained_examples = int(data.get("trained_examples", 0))
        saved = data.get("weights") or {}
        for f, v in saved.items():
            if f in self.weights:
                self.weights[f] = float(v)
        self._averaged_cache: Optional[Dict[str, float]] = None

    # -- scoring --------------------------------------------------------------

    def score(self, feats: Dict[str, float],
              use_averaged: bool = True) -> float:
        table = self._averaged() if use_averaged else self.weights
        return sum(table[f] * float(feats.get(f, 0.0)) for f in FEATURES)

    # -- training -------------------------------------------------------------

    def update(self, approved_feats: Dict[str, float],
               competitor_feats: Sequence[Dict[str, float]]) -> bool:
        """One pairwise max-margin update (Collins 2002's averaged
        scheme over Herbrich et al.'s pairwise constraints): for every
        competitor the approved surface fails to beat under the CURRENT
        weights, w += phi(approved) - phi(competitor). Returns True when
        at least one pair was violated (the update changed something)."""
        competitors = list(competitor_feats)
        if not competitors:
            return False
        violated = False
        self._steps += 1
        for comp in competitors:
            gap = (self.score(approved_feats, use_averaged=False)
                   - self.score(comp, use_averaged=False))
            if gap <= 0.0:
                violated = True
                for f in FEATURES:
                    delta = (float(approved_feats.get(f, 0.0))
                             - float(comp.get(f, 0.0)))
                    if delta:
                        self._bump(f, delta)
        self.updates += 1
        self.trained_examples += 1
        self._averaged_cache = None
        return violated

    def train(self, rows: Sequence[Dict[str, Any]],
              epochs: int = 12) -> int:
        """Deterministic multi-epoch training over
        ``{"approved": feats, "competitors": [feats, ...]}`` rows: fixed
        row order, full pass each epoch, early stop when an epoch makes
        no mistakes (the perceptron convergence guarantee holds when the
        preferences are separable; otherwise the averaged weights still
        bound the damage). Returns the number of epochs actually run."""
        rows = list(rows)
        if not rows:
            return 0
        for epoch in range(1, max(1, epochs) + 1):
            mistakes = sum(
                1 for row in rows
                if self.update(row["approved"], row["competitors"]))
            if mistakes == 0:
                return epoch
        return max(1, epochs)

    def _bump(self, feature: str, delta: float) -> None:
        self.weights[feature] += delta
        self._totals[feature] += self._steps * delta

    def _averaged(self) -> Dict[str, float]:
        """Averaged weights (Collins 2002 section 3.3): w_avg = w_total/t.
        Cached because scoring reads every feature of every candidate."""
        if self._averaged_cache is not None:
            return self._averaged_cache
        if self._steps == 0:
            self._averaged_cache = dict(self.weights)
            return self._averaged_cache
        self._averaged_cache = {
            f: self._totals[f] / self._steps for f in FEATURES}
        return self._averaged_cache

    # -- persistence (learned-state JSON; the caller writes it) ---------------

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": dict(self.weights),
                "totals": dict(self._totals),
                "steps": self._steps,
                "updates": self.updates,
                "trained_examples": self.trained_examples}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Reranker":
        reranker = cls()
        data = data if isinstance(data, dict) else {}
        for f, v in (data.get("weights") or {}).items():
            if f in reranker.weights:
                reranker.weights[f] = float(v)
        for f, v in (data.get("totals") or {}).items():
            if f in reranker._totals:
                reranker._totals[f] = float(v)
        reranker._steps = int(data.get("steps", 0))
        reranker.updates = int(data.get("updates", 0))
        reranker.trained_examples = int(data.get("trained_examples", 0))
        reranker._averaged_cache = None
        return reranker


# ---------------------------------------------------------------------------
# Supervision from LOCAL history only (the cortex/learn.py data source).
# ---------------------------------------------------------------------------

def supervision_from_history(examples: Sequence[Dict[str, Any]],
                             route_fn: Optional[Callable[
                                 [str], Dict[str, Dict[str, float]]]] = None
                             ) -> List[Dict[str, Any]]:
    """Turn approved history rows into pairwise training rows.

    ``examples``: cortex/learn.py's own example rows ({text, surface,
    label, outcome, ...}). ONLY approved rows (label 1 / outcome applied
    or approved) supervise; rejected or undecided rows are skipped — a
    refused request says nothing about what should have ranked instead.
    For each approved row the router re-scores the request: the row
    supervises only when the approved surface appears among the scored
    candidates (a surface the router did not score produces NO row —
    nothing is fabricated to fill the gap). Competitors are every OTHER
    scored candidate's feature dict.

    ``route_fn`` maps text -> {surface: feature-dict} (the caller passes
    ``lambda t: route(t, k=8).features`` for the real router; tests pass
    stubs). Default is the router's own top-8 with the default state.
    """
    if route_fn is None:
        from .router import route as _route

        def route_fn(text: str) -> Dict[str, Dict[str, float]]:
            return _route(text, k=8).features

    out: List[Dict[str, Any]] = []
    for row in examples:
        if not isinstance(row, dict):
            continue
        approved = (row.get("label") == 1
                    or row.get("outcome") in ("applied", "approved"))
        if not approved:
            continue
        text = str(row.get("text", ""))
        surface = str(row.get("surface", ""))
        if not text or not surface:
            continue
        features = route_fn(text)
        if surface not in features:
            continue  # the router never scored it — no constraint exists
        competitors = [f for name, f in features.items() if name != surface]
        if not competitors:
            continue  # a single candidate carries no ranking information
        out.append({"text": text, "approved": features[surface],
                    "competitors": competitors, "surface": surface})
    return out


# ---------------------------------------------------------------------------
# The gated entry point: re-ranker first, router ranking on low confidence.
# ---------------------------------------------------------------------------

def rerank_gated(features_map: Dict[str, Dict[str, float]],
                 router_order: Sequence[str],
                 reranker: Reranker,
                 calibrator: Optional[ConformalCalibrator] = None
                 ) -> Dict[str, Any]:
    """One route, one answer, one honest path label.

    - the re-ranker answers when it is trained AND the conformal
      calibrator (with actual calibration data) covers its margin
      confidence;
    - everything else — untrained re-ranker, a single candidate (nothing
      to re-rank), confidence below the conformal threshold, or NO
      calibration data at all — returns the router's OWN ranking
      unchanged, carrying the calibrator's own reason.

    ``features_map``: the RouteResult's own ``features`` dict;
    ``router_order``: the router's own candidate order (the fallback and
    the deterministic tie-break). Neither is mutated.
    """
    order = [n for n in router_order if n in features_map]
    if len(order) < 2:
        return {"used": "router-fallback",
                "reason": "fewer than two scored candidates — nothing to "
                          "re-rank",
                "order": list(router_order)}
    if reranker.trained_examples == 0 or not any(reranker.weights.values()):
        return {"used": "router-fallback", "reason": "re-ranker untrained",
                "order": list(router_order)}
    scores = [(reranker.score(features_map[n]), -i, n)
              for i, n in enumerate(order)]
    # deterministic: descending score, then the router's own order
    scores.sort(key=lambda t: (-t[0], -t[1]))
    best_score, _i, best = scores[0]
    second_score, _j, _second = scores[1]
    margin = best_score - second_score
    confidence = _margin_confidence(margin)
    verdict = (calibrator.verdict(confidence)
               if calibrator is not None and calibrator.scores
               else {"covered": None,
                     "reason": ("insufficient calibration data — no "
                                "distribution-free claim available")})
    if verdict.get("covered") is not True:
        return {"used": "router-fallback", "reason": str(verdict.get("reason")),
                "order": list(router_order), "confidence": round(confidence, 4),
                "margin": round(margin, 4)}
    return {"used": "reranker",
            "order": [t[2] for t in scores],
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "top": best,
            "conformal": verdict.get("reason")}

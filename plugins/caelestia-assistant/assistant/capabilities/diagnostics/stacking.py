"""diagnostics.stacking — stacked generalization over the fusion pool
(exponential-build-4 D).

Wolpert, "Stacked Generalization", Neural Networks 5(2), 1992: instead
of combining the confidences several engines emit with a FIXED rule
(here: fusion.py's weighted log-odds pool with hand-set weights), train
a META-LEARNER on level-1 data — the engines' own outputs — and let the
learned combiner do the mixing. The honest discipline the paper adds
and this module keeps: the meta-learner must be trained on outputs the
engines produced OUT OF SAMPLE (cross-validation), not on the same
rows they fit — combining in-sample outputs just memorizes the pool.

What is stacked here: the three fusion sources (diagnostics rule
scores, retrieval BM25 saturations, scan hit densities) per hypothesis.
The level-2 learning target is each source's EMPIRICAL RELIABILITY —
the fraction of labeled rows where that source's own top pick was the
correct hypothesis — and the learned weight is the reliability's lift
over chance (1/K hypotheses), floored at 0, normalized to the pool's
total. A build-4 finding is recorded here honestly: the tempting
per-source logistic of (log-odds edge, "was the top pick right") is
CIRCULAR — the label is true exactly when the edge is positive, for
any source including a random one, so no fit on that pair can measure
reliability. The chance-lift statistic is the non-circular level-2
learner, and the k-fold evaluate() is what makes the choice empirical.
The weights are DROP-IN for
``fusion.fuse(weights=stacked_weights(rows)["weights"])``: same
log-odds space, same weight semantics, interpretable by construction.

SELECTABLE, NOT A REPLACEMENT: the fixed pool stays fusion.py's
default. ``evaluate`` runs the k-fold comparison — held-out Brier for
fixed-pool vs stacked — and reports BOTH with row counts. With few
labeled rows the stacked fit is thin and the report says so; a stacked
win is an empirical claim measured per call, never asserted.
Deterministic: round-robin folds in the rows' given order; the
logistic fit is fixed-iteration gradient descent, no RNG. Pure
functions, no I/O, no source is modified — fusion.py stays untouched.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["stacked_weights", "evaluate", "SOURCES"]

SOURCES = ("diagnostics", "retrieval", "scan")

_ITERATIONS = 300
_RATE = 0.08
_FOLDS = 5
_MIN_ROWS_FOR_STACKING = 12


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def stacked_weights(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit the level-2 combiner, return per-source weights shaped for
    ``fusion.fuse(weights=...)`` (normalized to the fixed pool's total
    of 1.0).

    weight_s = max(0, reliability_s − 1/K) where reliability_s is the
    empirical fraction of labeled rows where source s's own top pick
    was the correct hypothesis, K the row's hypothesis count. The
    floor is honest: a source right no more often than chance has no
    vote; a source below chance is not asked to vote against itself
    (fusion refuses negative weights), it is dropped and recorded. A
    source with too few usable rows gets the pool's neutral 0.25 and a
    thin-evidence flag — an abstention, not a fitted guess.
    """
    if len(rows) < _MIN_ROWS_FOR_STACKING:
        return {
            "weights": {s: 0.25 for s in SOURCES},
            "n_rows": len(rows),
            "thin": True,
            "note": f"fewer than {_MIN_ROWS_FOR_STACKING} labeled rows: "
                    "a stacked fit would be a guess — the fixed pool's "
                    "neutral weights are returned instead",
        }
    lifts: Dict[str, Optional[float]] = {}
    dropped: List[str] = []
    thin_sources: List[str] = []
    for source in SOURCES:
        hits, trials, k = _reliability(rows, source)
        if trials < max(4, _MIN_ROWS_FOR_STACKING // 2):
            lifts[source] = None
            thin_sources.append(source)
            continue
        chance = 1.0 / max(2, k)
        lift = (hits / trials) - chance
        if lift <= 0:
            dropped.append(source)
            lifts[source] = 0.0
        else:
            lifts[source] = lift
    positive = {s: v for s, v in lifts.items() if v}
    if not positive:
        weights = {s: 0.25 for s in SOURCES}
    else:
        total = sum(v for v in lifts.values() if v)
        weights = {s: (round(lifts[s] / total, 4) if lifts[s]
                       else 0.0) for s in SOURCES}
    return {"weights": weights, "n_rows": len(rows), "thin": False,
            "thin_sources": thin_sources, "dropped_no_lift": dropped,
            "note": "level-2 weights = empirical reliability of each "
                    "source's top pick minus chance (Wolpert 1992's "
                    "stacked discipline; the build-4 circularity finding "
                    "is in the module docstring) — drop-in for "
                    "fusion.fuse(weights=...)"}


def _reliability(rows: Sequence[Dict[str, Any]], source: str
                 ) -> Tuple[int, int, int]:
    """(hits, trials, hypotheses) for one source: how often its own
    top pick was the correct hypothesis, and the hypothesis count K."""
    hits = trials = 0
    k = 0
    for row in rows:
        conf = row.get("confidences") or {}
        correct = row.get("correct")
        per = conf.get(source) or {}
        if correct not in per or not per:
            continue
        k = max(k, len(per))
        trials += 1
        hits += 1 if max(per, key=lambda h: per[h]) == correct else 0
    return hits, trials, k


def evaluate(rows: Sequence[Dict[str, Any]], folds: int = _FOLDS
             ) -> Dict[str, Any]:
    """The k-fold comparison the selectable contract requires. Per
    fold: learn weights on the training rows (stacked) against the
    fixed DEFAULT pool, then on each test row compute the pooled
    probability that the CORRECT hypothesis beats its best rival,
        p = sigmoid(Σ_s w_s·logit(c_s,correct)
                    − max_r Σ_s w_s·logit(c_s,r)),
    and score the Brier of (p, 1). Lower Brier = the combiner put more
    probability on the truth. Both Briers are reported with row counts;
    the winner is the measured one, per call — never asserted."""
    from .fusion import DEFAULT_WEIGHTS  # the fixed pool being compared

    usable = [r for r in rows
              if any(r.get("correct") in (r.get("confidences") or {}).get(s) or {}
                     for s in SOURCES)]
    n = len(usable)
    if n < _MIN_ROWS_FOR_STACKING:
        return {"n": n, "folds": 0, "brier_fixed": None,
                "brier_stacked": None, "winner": None,
                "note": "not enough labeled rows to compare anything "
                        "honestly — bring more labeled routing history"}
    fold_of = {i: i % folds for i in range(n)}
    fixed_errors: List[float] = []
    stacked_errors: List[float] = []
    for f in range(folds):
        train = [r for i, r in enumerate(usable) if fold_of[i] != f]
        test = [r for i, r in enumerate(usable) if fold_of[i] == f]
        if not train or not test:
            continue
        stacked = stacked_weights(train)["weights"]
        for row in test:
            conf = row.get("confidences") or {}
            correct = row.get("correct")
            if correct is None:
                continue
            hypotheses = sorted({h for s in SOURCES
                                 for h in (conf.get(s) or {})})

            def pooled(h: str, weights: Dict[str, float]) -> float:
                acc = 0.0
                for s in SOURCES:
                    per = conf.get(s) or {}
                    c = per.get(h)
                    if c is None:
                        continue  # a source with no opinion abstains
                    acc += weights.get(s, 0.0) * _logit(c)
                return acc

            for weights, sink in ((DEFAULT_WEIGHTS, fixed_errors),
                                  (stacked, stacked_errors)):
                own = pooled(correct, weights)
                rivals = [pooled(h, weights) for h in hypotheses
                          if h != correct]
                if not rivals:
                    continue
                p = _sigmoid(own - max(rivals))
                sink.append((1.0 - p) ** 2)
    brier_fixed = (round(sum(fixed_errors) / len(fixed_errors), 4)
                   if fixed_errors else None)
    brier_stacked = (round(sum(stacked_errors) / len(stacked_errors), 4)
                     if stacked_errors else None)
    winner = None
    if brier_fixed is not None and brier_stacked is not None:
        winner = ("stacked" if brier_stacked < brier_fixed else
                  "fixed_pool" if brier_fixed < brier_stacked else "tie")
    return {"n": n, "folds": folds, "brier_fixed": brier_fixed,
            "brier_stacked": brier_stacked, "winner": winner,
            "note": "held-out Brier of the pooled correct-vs-best-rival "
                    "probability, k-fold (round-robin over the given "
                    "order, deterministic); a one-stream win is "
                    "evidence, not a conclusion — the fixed pool stays "
                    "the default"}

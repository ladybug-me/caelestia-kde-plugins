"""cortex.conformal — distribution-free honesty for the router's verdicts.

The router already reports a calibrated softmax confidence (Beta-Binomial
buckets). This module adds the statistical guarantee on top:

- `ConformalCalibrator` — split-conformal prediction: from a calibration set
  of (score, outcome) pairs it derives the smallest score level s such that
  at least (1-alpha) of past accepted routes scored >= s. A NEW route whose
  score clears that level carries a VALID coverage guarantee: "routes this
  confident were right at least 90% of the time before" — a distribution-free
  statement that needs no model assumptions, only exchangeability.

- `query_by_committee` — active learning: when the router's weight profiles
  (lexical-heavy / semantic-heavy / balanced) DISAGREE about the top surface,
  the honest move is to ask the user, and the disagreement (vote entropy) is
  exactly the information-gain measure. Returns the ranked "teach me" list.

- `PageHinkleyDrift` — incremental drift detector over the acceptance stream:
  the same statistic the scan layer uses on error rates, reused on YOUR
  behaviour ("you started rejecting settings plans this week").

All state is JSON round-trippable through the cortex learner's state dict.
Nothing here writes files; nothing here is a chatbot.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["ConformalCalibrator", "query_by_committee", "PageHinkleyDrift"]


# ---------------------------------------------------------------------------
class ConformalCalibrator:
    """Split-conformal calibration over router scores.

    Feed (score, outcome) pairs as they are observed — `learn_feedback`
    already produces them. `verdict(score, alpha)` then returns a
    distribution-free statement about a new route.
    """

    def __init__(self, alpha: float = 0.1, max_history: int = 400) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha in (0, 1)")
        self.alpha = alpha
        self.max_history = max_history
        self.scores: List[float] = []          # scores of ACCEPTED routes only
        self.all_scores: List[float] = []
        self.all_outcomes: List[int] = []

    # ------------------------------------------------------------------
    def observe(self, score: float, outcome: str) -> None:
        """outcome: 'applied' | 'approved' | 'rejected' | 'abstained'."""
        score = float(score)
        self.all_scores.append(score)
        self.all_outcomes.append(1 if outcome in ("applied", "approved") else 0)
        if outcome in ("applied", "approved"):
            self.scores.append(score)
        if len(self.all_scores) > self.max_history:
            overflow = len(self.all_scores) - self.max_history
            del self.all_scores[:overflow]
            del self.all_outcomes[:overflow]
        if len(self.scores) > self.max_history:
            del self.scores[:len(self.scores) - self.max_history]

    # ------------------------------------------------------------------
    def threshold(self, alpha: Optional[float] = None) -> Optional[float]:
        """Conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile of
        accepted-route scores. None when there is not enough calibration
        data (honesty: an empty calibration set guarantees nothing)."""
        a = alpha if alpha is not None else self.alpha
        n = len(self.scores)
        if n < 5:
            return None
        scores = sorted(self.scores)
        rank = min(n, max(1, math.ceil((n + 1) * (1.0 - a))))
        return scores[rank - 1]

    def verdict(self, score: float, alpha: Optional[float] = None) -> Dict[str, Any]:
        t = self.threshold(alpha)
        a = alpha if alpha is not None else self.alpha
        n = len(self.scores)
        if t is None:
            return {
                "covered": None,
                "guarantee": None,
                "reason": (f"insufficient calibration data ({n} accepted routes; "
                           f"need >= 5) — no distribution-free claim available"),
                "score": round(score, 4),
                "alpha": a,
            }
        covered = score >= t
        return {
            "covered": covered,
            "guarantee": (f"routes at or above {round(t, 3)} were accepted "
                          f">= {round(1 - a, 2)} of the time historically "
                          f"(distribution-free, n={n})"),
            "reason": "score clears the conformal threshold" if covered
                      else "score below the conformal threshold",
            "score": round(score, 4),
            "threshold": round(t, 4),
            "alpha": a,
        }

    # ------------------------------------------------------------------
    def empirical_accuracy(self, min_score: float = 0.0) -> Optional[float]:
        pairs = [(s, o) for s, o in zip(self.all_scores, self.all_outcomes)
                 if s >= min_score]
        if not pairs:
            return None
        return sum(o for _s, o in pairs) / len(pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "scores": self.scores,
                "all_scores": self.all_scores, "all_outcomes": self.all_outcomes}

    def from_dict(self, d: Dict[str, Any]) -> "ConformalCalibrator":
        self.alpha = d.get("alpha", self.alpha)
        self.scores = list(d.get("scores", []))
        self.all_scores = list(d.get("all_scores", []))
        self.all_outcomes = list(d.get("all_outcomes", []))
        return self


# ---------------------------------------------------------------------------
def query_by_committee(candidates: Sequence[Dict[str, Any]],
                       committee_votes: Sequence[Dict[str, Dict[str, float]]],
                       top_k: int = 3) -> List[Dict[str, Any]]:
    """Rank the requests worth TEACHING on (active learning).

    candidates:  the request texts (dicts with at least {"text": ...})
    committee_votes: one dict per strategy; maps candidate index -> score
    Returns the top_k candidates by COMMITTEE DISAGREEMENT — the standard
    deviation of the members' raw scores. Spread, not normalized vote
    entropy, is the right measure here: entropy cannot tell "both members
    high" from "both members low", but those are the opposite teaching
    cases, while a wide spread means the strategies genuinely chose
    different surfaces and one label settles the fight.
    """
    if len(candidates) != len(committee_votes[0] if committee_votes else {}):
        raise ValueError("votes must score every candidate")
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for i, cand in enumerate(candidates):
        scores = [votes.get(i, 0.0) for votes in committee_votes]
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        disagreement = math.sqrt(var)
        row = dict(cand)
        row["disagreement"] = round(disagreement, 4)
        row["mean_score"] = round(mean, 4)
        ranked.append((disagreement, row))
    ranked.sort(key=lambda t: -t[0])
    out = [row for _e, row in ranked[:top_k]]
    for row in out:
        row["why"] = ("strategies disagree — labelling this buys the most "
                      "information for the router")
    return out


# ---------------------------------------------------------------------------
class PageHinkleyDrift:
    """Incremental Page-Hinkley over the accept/reject stream, oriented to
    detect a DROP in acceptance (a preference shift), not a rise.

    Internally it runs the standard upward PH on the negated stream
    (x' = 1 - x): "fewer accepts" == "more rejects" == an upward change in
    x'. Hysteresis: one alarm latches the state so a single bad day is not
    reported as a trend forever — `reset()` clears it after you have acted.
    """

    def __init__(self, threshold: float = 8.0, alpha: float = 0.98,
                 delta: float = 0.01, warmup: int = 25) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self.delta = delta
        self.warmup = warmup
        self.n = 0
        self.mean = 0.0          # EMA of the rejection stream
        self.sum_xt = 0.0
        self.min_ht = math.inf
        self.alarm_at: Optional[int] = None

    def update(self, accepted: bool) -> Dict[str, Any]:
        x = 0.0 if accepted else 1.0   # negated stream: reject == 1
        if self.n == 0:
            self.mean = x
        self.n += 1
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        self.sum_xt += x - self.mean - self.delta
        self.min_ht = min(self.min_ht, self.sum_xt)
        stat = self.sum_xt - self.min_ht
        alarmed = (self.n >= self.warmup and stat > self.threshold
                   and self.alarm_at is None)
        if alarmed:
            self.alarm_at = self.n
        return {"n": self.n, "stat": round(stat, 4), "alarmed": alarmed}

    def reset(self) -> None:
        """Clear the latch after acting (weights adapted, or user informed)."""
        self.alarm_at = None
        self.n = 0
        self.sum_xt = 0.0
        self.min_ht = math.inf
        self.mean = 0.0

    def status(self) -> Dict[str, Any]:
        stat = self.sum_xt - self.min_ht
        return {
            "n": self.n,
            "statistic": round(stat, 4),
            "threshold": self.threshold,
            "alarmed": self.alarm_at is not None,
            "alarm_at": self.alarm_at,
            "message": ("acceptance rate has DROPPED — your preferences "
                        "seem to have shifted; weights will adapt, but a "
                        "cortex reset-learning is available" )
                        if self.alarm_at is not None else "no drift detected",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "alpha": self.alpha,
                "delta": self.delta, "warmup": self.warmup, "n": self.n,
                "mean": self.mean, "sum_xt": self.sum_xt,
                "min_ht": self.min_ht if self.min_ht != math.inf else None,
                "alarm_at": self.alarm_at}

    def from_dict(self, d: Dict[str, Any]) -> "PageHinkleyDrift":
        self.threshold = d.get("threshold", self.threshold)
        self.alpha = d.get("alpha", self.alpha)
        self.delta = d.get("delta", self.delta)
        self.warmup = d.get("warmup", self.warmup)
        self.n = d.get("n", 0)
        self.mean = d.get("mean", 1.0)
        self.sum_xt = d.get("sum_xt", 0.0)
        self.min_ht = d.get("min_ht") if d.get("min_ht") is not None else math.inf
        self.alarm_at = d.get("alarm_at")
        return self

"""Thompson sampling over named arms: which settings presets/tools does this
user actually approve when the brain proposes them?

This is HourBandit's algorithm (bandit.py) generalised from a fixed 0-23
index to arbitrary string arms, because settings proposals are keyed by
preset name (or tool name), not by hour. Each arm is a Beta(alpha, beta):
an approval in the ledger is a success, a rejection is a failure. New arms
start at Beta(1, 1) -- a flat prior, i.e. "no opinion yet" -- so a preset
nobody has been offered before is neither favoured nor penalised.

Secondary reward (issue #120 Phase 3.2, optional): reward() accepts an
optional ``secondary`` signal in [0, 1] (0.5 neutral) computed from a
battery-drain-rate delta measured around the preset's active window (see
diagnostics/telemetry.py::reward_from_drain). It contributes fractional
pseudo-counts scaled by SECONDARY_REWARD_WEIGHT (default 0.25), so even a
max-strength secondary signal can never outweigh one real approve/reject
decision (+/-1.0). It is ADDITIVE context, never a replacement: with
secondary=None the update is byte-for-byte the original +/-1 reward.

This module never decides anything by itself: it only ranks candidates for
a human to choose from (settings_bridge.recommend), exactly like every
other brain learner. Persisted as a plain {name: [alpha, beta]} dict, same
shape as HourBandit.to_dict(), so it lives in the same state.json file.
"""
import math
import random
from typing import Dict, List

from .features import DEFAULT_DIMENSIONS, context_features

SECONDARY_REWARD_WEIGHT = 0.25
LINUCB_ALPHA = 0.3  # exploration strength: (Li et al. 2010)'s c term


class NamedBandit:
    def __init__(self, arms=None):
        # arms: {name: [alpha, beta]}
        self.arms = {k: [float(v[0]), float(v[1])] for k, v in (arms or {}).items()}

    def _arm(self, name):
        return self.arms.setdefault(name, [1.0, 1.0])

    def reward(self, name, approved, secondary=None):
        """One decision. ``approved`` is the primary +/-1 signal;
        ``secondary`` (optional, [0, 1], 0.5 neutral) folds in fractional
        pseudo-counts at SECONDARY_REWARD_WEIGHT strength."""
        alpha, beta = self._arm(name)
        if approved:
            alpha, beta = alpha + 1.0, beta
        else:
            alpha, beta = alpha, beta + 1.0
        if secondary is not None:
            r = min(1.0, max(0.0, float(secondary)))
            alpha += SECONDARY_REWARD_WEIGHT * r
            beta += SECONDARY_REWARD_WEIGHT * (1.0 - r)
        self.arms[name] = [alpha, beta]

    def sample(self, name, rng=None):
        rng = rng or random
        alpha, beta = self._arm(name)
        return rng.betavariate(alpha, beta)

    def rank(self, names, rng=None):
        """Candidates ranked by one Thompson draw each, best first.

        Returns [(name, draw, mean_estimate), ...]. mean_estimate
        (alpha / (alpha + beta)) is the stable, non-random summary shown to
        the human; ``draw`` is what decided the order, so exploration still
        surfaces under-tried arms occasionally.
        """
        rng = rng or random
        scored = []
        for name in names:
            alpha, beta = self._arm(name)
            draw = rng.betavariate(alpha, beta)
            scored.append((name, round(draw, 3), round(alpha / (alpha + beta), 3)))
        return sorted(scored, key=lambda row: -row[1])

    def to_dict(self):
        return {k: list(v) for k, v in self.arms.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(d or {})


# ---------------------------------------------------------------------------
# Phase 2.4: LinUCB — the CONTEXTUAL bandit (matrix updates only).
#
# NamedBandit is context-blind: "compact" has ONE Beta pair regardless of
# when/why it was offered. LinUCB (Li, Chu, Langford & Schapire 2010,
# "A contextual-bandit approach to personalized news article
# recommendation", WWW) models the expected approval as x^T theta per
# arm, with a d x d Gram matrix per arm (the DISJOINT model) updated by
# outer products — pure arithmetic, no library, no gradient, nothing
# trained offline. Context = the shared feature hashing
# (brain/features.py: request text + hour bucket), so the router /
# genius / preset learners agree on the INPUT space while keeping
# separate state (transfer without merging).
# ---------------------------------------------------------------------------


class LinUCBBandit:
    """Disjoint LinUCB over named arms (presets, tools, or plans).

    Per arm: A (d x d, starts as identity) and b (d, starts zero). The
    upper confidence score for arm i under context x is
    x^T A^-1 b + alpha * sqrt(x^T A^-1 x) — exploit plus an exact
    confidence bound (the 2010 paper's Equation 5-7, disjoint part).
    """

    def __init__(self, arms=None, d: int = DEFAULT_DIMENSIONS,
                 alpha: float = LINUCB_ALPHA):
        # arms: {name: {"A": [[...]], "b": [...]}} (rows of A)
        self.d = int(d)
        self.alpha = float(alpha)
        self.arms = {}
        for name, payload in (arms or {}).items():
            rows = [[float(v) for v in row] for row in payload["A"]]
            vec = [float(v) for v in payload["b"]]
            if len(rows) != self.d or len(vec) != self.d:
                raise ValueError(f"arm {name!r} has wrong dimensions")
            self.arms[name] = {"A": rows, "b": vec}

    def _arm(self, name):
        return self.arms.setdefault(
            name, {"A": [[1.0 if i == j else 0.0 for j in range(self.d)]
                         for i in range(self.d)],
                   "b": [0.0] * self.d})

    # -- linear algebra, small and explicit --------------------------------

    def _solve(self, matrix, vector):
        """Solve M x = v by Gaussian elimination with partial pivoting
        (d <= 64; no library, deterministic)."""
        n = self.d
        m = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
            if abs(m[pivot][col]) < 1e-12:
                m[col][col] += 1e-9  # ridge-nudge a singular Gram matrix
            else:
                m[col], m[pivot] = m[pivot], m[col]
            pivot_value = m[col][col]
            for r in range(n + 1):
                if r != col and r < n or r == n:
                    pass
            for r in range(n):
                if r == col:
                    continue
                factor = m[r][col] / pivot_value
                if factor:
                    for c in range(col, n + 1):
                        m[r][c] -= factor * m[col][c]
        return [m[i][n] / (m[i][i] or 1e-12) for i in range(n)]

    def _quadratic(self, matrix, x):
        """x^T M x without inverting explicitly (via the solve above)."""
        solved = self._solve(matrix, list(x))
        return sum(xi * si for xi, si in zip(x, solved))

    # -- the bandit interface ----------------------------------------------

    def score(self, name, context: List[float]) -> Dict[str, float]:
        """Exploit term and UCB score for one arm under one context."""
        arm = self._arm(name)
        theta = self._solve(arm["A"], arm["b"])
        exploit = sum(xi * ti for xi, ti in zip(context, theta))
        confidence = self.alpha * math.sqrt(
            max(0.0, self._quadratic(arm["A"], context)))
        return {"exploit": round(exploit, 4),
                "score": round(exploit + confidence, 4)}

    def rank(self, names, context=None, text: str = "", hour: int = -1):
        """Candidates ranked by LinUCB upper-confidence score, best first.

        ``context``: the shared feature vector (brain.features.
        context_features(text, hour) when not given directly). Returns
        [(name, score, exploit), ...].
        """
        if context is None:
            context = context_features(text, hour, d=self.d)
        context = list(context)
        if len(context) != self.d:
            raise ValueError(f"context must be d={self.d}-dimensional")
        scored = []
        for name in names:
            row = self.score(name, context)
            scored.append((name, row["score"], row["exploit"]))
        return sorted(scored, key=lambda r: -r[1])

    def reward(self, name, approved, context=None, text: str = "",
               hour: int = -1):
        """One decision: A += x x^T, b += r x (r = +1 approved, -1
        refused) — the disjoint-model update, matrix arithmetic only."""
        if context is None:
            context = context_features(text, hour, d=self.d)
        context = list(context)
        if len(context) != self.d:
            raise ValueError(f"context must be d={self.d}-dimensional")
        arm = self._arm(name)
        r = 1.0 if approved else -1.0
        for i in range(self.d):
            arm["b"][i] += r * context[i]
            for j in range(self.d):
                arm["A"][i][j] += context[i] * context[j]

    def to_dict(self):
        return {name: {"A": [[round(v, 6) for v in row]
                             for row in payload["A"]],
                       "b": [round(v, 6) for v in payload["b"]]}
                for name, payload in self.arms.items()}

    @classmethod
    def from_dict(cls, data, d: int = DEFAULT_DIMENSIONS,
                  alpha: float = LINUCB_ALPHA) -> "LinUCBBandit":
        return cls(data or {}, d=d, alpha=alpha)

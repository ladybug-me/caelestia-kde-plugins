"""Behavioural signals: z-score anomalies, deferral flags, focus entropy,
and — the multivariate addition — the Isolation Forest (§6.4).

Non-redundancy by construction (checked by tests, not asserted in
comments): the existing detectors are all UNIVARIATE and/or TEMPORAL —
``zscore`` flags one odd value against a single signal's history;
``scan.sketch.page_hinkley`` and the diagnostics layer's CUSUM catch a
signal DRIFTING over time. What none of them can see is ONE event that
is weird across several dimensions at once while every single dimension
stays individually plausible: a settings spree at 3 a.m. touching six
groups in one apply; a tidy run whose shape matches no previous run. The
Isolation Forest (Liu, Ting & Zhou 2008, "Isolation Forest", ICDM 2008:
iTrees over subsamples, path-length averaging, s = 2^(-E(h)/c(n)))
exists here for exactly that case, and is deliberately NOT wired into
any univariate surface.
"""
import math
import random
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


def zscore(history, x):
    if len(history) < 2:
        return 0.0
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / (len(history) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0 if x == mean else math.inf
    return (x - mean) / sd


def deferral_flag(defer_count, threshold=3):
    return defer_count >= threshold


def shannon_bits(labels):
    """Entropy of task-switch labels. High entropy = fragmented attention."""
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ---------------------------------------------------------------------------
# Isolation Forest (Liu, Ting & Zhou, ICDM 2008).
# ---------------------------------------------------------------------------

def _harmonic(i: int) -> float:
    """H(i) = 1 + 1/2 + ... + 1/i (H(0) = 0)."""
    return sum(1.0 / k for k in range(1, i + 1))


def _c(n: int) -> float:
    """Average unsuccessful-search path length of a binary search tree with
    n external nodes — the paper's normalization constant c(n)."""
    if n <= 1:
        return 0.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n


class IsolationForest:
    """The paper's parameters by default: 100 trees over subsamples of 256,
    tree depth capped at ceil(log2(subsample)). Seeded RNG — reproducible
    by the repo's determinism rule, so the same data always yields the
    same scores. Scores follow the paper's scale: s -> 1 anomalous,
    s << 0.5 clearly normal, s ~ 0.5 no structure; 0.6 is the paper's
    practical cut (above it, "there are apparent anomalies")."""

    def __init__(self, n_trees: int = 100, subsample: int = 256,
                 seed: int = 42) -> None:
        if n_trees < 1 or subsample < 2:
            raise ValueError("n_trees >= 1 and subsample >= 2 required")
        self.n_trees = int(n_trees)
        self.subsample = int(subsample)
        self.seed = int(seed)
        self._trees: List[Any] = []
        self._n = 0

    # -- construction ------------------------------------------------------

    def _build_tree(self, rows: List[Sequence[float]], rng: random.Random,
                    depth: int, depth_limit: int) -> Dict[str, Any]:
        if depth >= depth_limit or len(rows) <= 1:
            return {"leaf": True, "n": len(rows)}
        n_features = len(rows[0])
        feature = rng.randrange(n_features)
        values = [row[feature] for row in rows]
        lo, hi = min(values), max(values)
        if lo == hi:
            return {"leaf": True, "n": len(rows)}
        threshold = rng.uniform(lo, hi)
        left = [row for row in rows if row[feature] < threshold]
        right = [row for row in rows if row[feature] >= threshold]
        if not left or not right:
            # degenerate draw (floating-point landed on lo or hi): the
            # midpoint is guaranteed to split lo < hi into both sides
            threshold = (lo + hi) / 2.0
            left = [row for row in rows if row[feature] < threshold]
            right = [row for row in rows if row[feature] >= threshold]
        return {"leaf": False, "feature": feature, "threshold": threshold,
                "left": self._build_tree(left, rng, depth + 1, depth_limit),
                "right": self._build_tree(right, rng, depth + 1, depth_limit)}

    def fit(self, rows: Sequence[Sequence[float]]) -> "IsolationForest":
        if not rows:
            raise ValueError("no rows to fit")
        rng = random.Random(self.seed)
        depth_limit = max(1, math.ceil(math.log2(max(2, self.subsample))))
        self._trees = []
        for _ in range(self.n_trees):
            sample = list(rows)
            if len(sample) > self.subsample:
                sample = rng.sample(sample, self.subsample)
            self._trees.append(self._build_tree(sample, rng, 0, depth_limit))
        self._n = len(rows)
        return self

    # -- scoring -------------------------------------------------------------

    def _path_length(self, row: Sequence[float], node: Dict[str, Any],
                     depth: int) -> float:
        if node.get("leaf"):
            return depth + _c(node["n"])
        if row[node["feature"]] < node["threshold"]:
            return self._path_length(row, node["left"], depth + 1)
        return self._path_length(row, node["right"], depth + 1)

    def score(self, row: Sequence[float]) -> float:
        """s(x) = 2^(-E(h(x)) / c(n)) — the paper's anomaly score."""
        if not self._trees:
            raise ValueError("fit() first")
        e_h = sum(self._path_length(row, tree, 0) for tree in self._trees)
        e_h /= len(self._trees)
        return 2.0 ** (-e_h / max(_c(self._n), 1e-12))

    def scores(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [self.score(row) for row in rows]


# ---------------------------------------------------------------------------
# Feature adapters: the two behavioural event streams this exists for.
# ---------------------------------------------------------------------------

SETTINGS_FEATURE_NAMES = ["hour_sin", "hour_cos", "ops", "paths",
                          "minutes_since_previous"]
TIDY_FEATURE_NAMES = ["scanned", "type_moves", "duplicates", "stale",
                      "big_files", "empty_dirs", "space_recoverable"]


def settings_change_features(entries: Sequence[Dict[str, Any]]
                              ) -> List[List[float]]:
    """Settings-history entries -> behavioural feature rows: when the
    change happened (circular hour), how big it was (op count, distinct
    paths), and how long after the previous change. ``entries`` are the
    undo-history shape ({"at": ISO-8601, "ops": [{"path", ...}]}), taken
    OLDEST FIRST for the interval feature (the history store's own order
    is newest first — reverse it first)."""
    rows: List[List[float]] = []
    previous: Optional[datetime] = None
    for entry in sorted(entries, key=lambda e: str(e.get("at", ""))):
        try:
            at = datetime.fromisoformat(str(entry.get("at", "")))
        except ValueError:
            continue
        ops = entry.get("ops") or []
        paths = {str(op.get("path", "")) for op in ops}
        angle = 2.0 * math.pi * (at.hour % 24) / 24.0
        gap = 0.0 if previous is None else max(
            0.0, (at - previous).total_seconds() / 60.0)
        rows.append([round(math.sin(angle), 4), round(math.cos(angle), 4),
                     float(len(ops)), float(len(paths)), round(gap, 2)])
        previous = at
    return rows


def tidy_run_features(surveys: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """tidy.survey() result dicts -> one row per run: the run's shape
    (scanned, type_moves, duplicates, stale, big_files, empty_dirs,
    space_recoverable), log-scaled where the raw magnitude spans decades
    (the tree's random thresholds care about ratios, not tonnes).
    Survey fields arrive as lists of records from survey() but plain
    counts from serialized/partial sources; both are accepted."""
    def _count(value: Any) -> float:
        if isinstance(value, (list, tuple)):
            return float(len(value))
        try:
            return max(0.0, float(value or 0))
        except (TypeError, ValueError):
            return 0.0

    rows: List[List[float]] = []
    for survey in surveys:
        if not isinstance(survey, dict):
            continue
        rows.append([
            math.log1p(_count(survey.get("scanned"))),
            math.log1p(_count(survey.get("type_moves"))),
            math.log1p(_count(survey.get("duplicates"))),
            math.log1p(_count(survey.get("stale"))),
            math.log1p(_count(survey.get("big_files"))),
            math.log1p(_count(survey.get("empty_dirs"))),
            math.log1p(_count(survey.get("space_recoverable"))),
        ])
    return rows

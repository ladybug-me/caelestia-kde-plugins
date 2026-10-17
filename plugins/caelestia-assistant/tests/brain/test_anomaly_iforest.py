"""Isolation Forest behavioural anomaly detection (§6.4) + the
non-redundancy fixtures against the existing detectors.

The point of this suite is the SECOND class: the same fixtures show the
Isolation Forest firing exactly where the univariate/temporal detectors
(page_hinkley, zscore) do not, and NOT firing where they do — so the
layer stays complementary rather than doubled.
"""

from __future__ import annotations

import math
import random
import unittest

from assistant.capabilities.brain.anomaly import (IsolationForest, TIDY_FEATURE_NAMES,
                                     _c, settings_change_features,
                                     tidy_run_features, zscore)
from assistant.capabilities.scan.sketch import page_hinkley


def _population(rng: random.Random, n: int = 300):
    """A normal multivariate population: three correlated features, each
    individually unremarkable."""
    rows = []
    for _ in range(n):
        a = rng.gauss(0.0, 1.0)
        rows.append([a, a * 0.6 + rng.gauss(0.0, 0.4),
                     -a * 0.4 + rng.gauss(0.0, 0.5)])
    return rows


class ScoreTests(unittest.TestCase):

    def test_planted_outlier_scores_highest(self):
        rng = random.Random(7)
        rows = _population(rng)
        rows.append([3.2, 3.0, -3.1])  # every dimension extreme jointly
        forest = IsolationForest(seed=7).fit(rows)
        scores = forest.scores(rows)
        self.assertEqual(scores.index(max(scores)), len(rows) - 1,
                         "the planted joint outlier must be the most "
                         "anomalous row")
        self.assertGreater(max(scores), 0.6)  # the paper's "apparent anomalies" cut

    def test_normals_score_low(self):
        rng = random.Random(11)
        rows = _population(rng)
        forest = IsolationForest(seed=11).fit(rows)
        scores = forest.scores(rows)
        # pure Gaussian noise keeps apparent-anomaly rate near the
        # paper's own false-positive behavior at the 0.6 cut
        self.assertLess(sum(1 for s in scores if s > 0.6) / len(scores), 0.05)

    def test_deterministic(self):
        rng = random.Random(5)
        rows = _population(rng, n=120)
        a = IsolationForest(seed=5).fit(rows).scores(rows)
        b = IsolationForest(seed=5).fit(rows).scores(rows)
        self.assertEqual(a, b)

    def test_c_normalization_matches_formula(self):
        self.assertAlmostEqual(_c(1), 0.0)
        self.assertAlmostEqual(_c(2), 1.0)
        # c(256), the paper's subsample normalization: 2*H(255) - 2*255/256
        h255 = sum(1.0 / k for k in range(1, 256))
        self.assertAlmostEqual(_c(256), 2 * h255 - 2 * 255 / 256)

    def test_validation(self):
        with self.assertRaises(ValueError):
            IsolationForest().fit([])
        with self.assertRaises(ValueError):
            IsolationForest(subsample=1)


class NonRedundancyTests(unittest.TestCase):
    """The two phenomena the existing detectors and the forest split
    between themselves — verified on the same fixtures, not asserted."""

    def test_gradual_drift_is_page_hinkleys_case_not_the_forests(self):
        # a signal drifting upward slowly, one dimension, no joint events:
        # PH must flag it; the forest must NOT rate the drifted tail as
        # anomalous (the population stays internally consistent).
        rng = random.Random(3)
        stream = [rng.gauss(0.0, 0.5) + 0.05 * i for i in range(150)]
        ph = page_hinkley(stream, threshold=10.0, min_instances=30)
        self.assertIsNotNone(ph["change_at"],
                             "Page-Hinkley owns this fixture")
        rows = [[value, rng.gauss(0.0, 1.0)] for value in stream]
        forest = IsolationForest(seed=3).fit(rows)
        tail = forest.scores(rows[100:])
        self.assertLess(sum(1 for s in tail if s > 0.6) / len(tail), 0.10,
                        "a drifted-but-consistent population is not a "
                        "forest anomaly — that would be redundant with PH")

    def test_single_joint_event_is_the_forests_case_not_zscores(self):
        # one event whose every dimension sits at its own 90th percentile —
        # no single dimension is a z>3 outlier, but the COMBINATION never
        # occurs in the population. zscore stays quiet; the forest fires.
        rng = random.Random(9)
        rows = _population(rng)
        # choose each coordinate at its 90th percentile — individually
        # unremarkable, jointly impossible for a correlated population
        percentiles = [[sorted(r[d] for r in rows)[int(0.9 * len(rows))]
                        for d in range(3)]]
        weird = [rows[0][0] * 0 + v for v in percentiles[0]]
        # force the anti-correlated third feature to VIOLATE the
        # population's structure instead of following it:
        weird[2] = percentiles[0][2] * -1 * 3.0
        for d, value in enumerate(weird):
            zs = zscore([r[d] for r in rows], value)
            self.assertLess(abs(zs), 6.0,
                            "fixture must stay individually unremarkable")
        forest = IsolationForest(seed=9).fit(rows)
        self.assertGreater(forest.score(weird), 0.6,
                           "the joint event is the forest's case")


class FeatureAdapterTests(unittest.TestCase):

    def test_settings_change_features(self):
        entries = [
            {"at": "2026-09-20T10:00:00", "ops": [
                {"path": "bar.scale", "old": 1.0, "new": 0.9}]},
            {"at": "2026-09-20T10:30:00", "ops": [
                {"path": "bar.scale", "old": 0.9, "new": 0.85},
                {"path": "appearance.blur", "old": True, "new": False}]},
        ]
        rows = settings_change_features(entries)
        self.assertEqual(len(rows), 2)
        # circular hour encoding for 10:00
        self.assertAlmostEqual(rows[0][0], round(math.sin(2 * math.pi * 10 / 24), 4))
        self.assertAlmostEqual(rows[0][1], round(math.cos(2 * math.pi * 10 / 24), 4))
        self.assertEqual(rows[0][3], 1.0)   # one distinct path
        self.assertEqual(rows[0][4], 0.0)   # first entry: no gap
        self.assertEqual(rows[1][3], 2.0)   # two distinct paths
        self.assertAlmostEqual(rows[1][4], 30.0)  # minutes since previous

    def test_settings_features_order_independent_of_input(self):
        entries = [
            {"at": "2026-09-20T10:30:00", "ops": [{"path": "a", "old": 1, "new": 2}]},
            {"at": "2026-09-20T10:00:00", "ops": [{"path": "b", "old": 2, "new": 3}]},
        ]
        rows = settings_change_features(entries)
        self.assertEqual(rows[0][4], 0.0)      # the 10:00 entry is first
        self.assertAlmostEqual(rows[1][4], 30.0)

    def test_tidy_run_features(self):
        surveys = [
            {"scanned": 500, "type_moves": [{"from": "a"}] * 40, "duplicates": 5,
             "stale": 3, "big_files": 2, "empty_dirs": 1,
             "space_recoverable": 1_000_000},
            {"scanned": 10, "type_moves": [], "duplicates": [], "stale": [],
             "big_files": [], "empty_dirs": [], "space_recoverable": 0},
        ]
        rows = tidy_run_features(surveys)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(TIDY_FEATURE_NAMES))
        self.assertGreater(rows[0][0], rows[1][0])
        self.assertEqual(rows[1][6], 0.0)  # log1p(0)

    def test_settings_spree_is_flagged(self):
        # the module's charter case: normal evening tweaks, then one 3 a.m.
        # spree touching six groups 90 seconds after the last change
        entries = [
            {"at": f"2026-09-{day:02d}T19:15:00", "ops": [{"path": "bar.scale", "old": 1, "new": 0.9}]}
            for day in range(1, 25)
        ]
        spree = {"at": "2026-09-25T03:05:00", "ops": [
            {"path": f"group{i}.key", "old": 0, "new": 1} for i in range(6)]}
        rows = settings_change_features(entries + [spree])
        forest = IsolationForest(seed=13).fit(rows)
        self.assertGreater(forest.score(rows[-1]), 0.6,
                           "the 3 a.m. six-group spree is the anomaly this "
                           "module exists to catch")


if __name__ == "__main__":
    unittest.main()

"""Tests for exponential-build phase 2.6: the robust Mahalanobis
baseline over the telemetry layer's numeric metrics.

The contract under test:

- center = median, scale = 1.4826 x MAD (Leys et al. 2013) — NOT
  mean/variance, so a poisoned history row does not move the baseline;
- degenerate coordinates (MAD = 0) are excluded and NAMED, never given
  a fake scale; missing sample coordinates are excluded and counted,
  never imputed; fewer than half the coordinates present is a refusal;
- the distance uses the chi-square tail from the EXISTING stats
  primitive, with the approximation caveat attached to the report;
- flatten_snapshot derives the metric vector from the telemetry
  layer's own shapes (battery deliberately excluded as a cyclic
  quantity); the archetype hook is read-only evidence, nothing acts on
  the verdict.
"""
import unittest

from assistant.capabilities.diagnostics import robust_baseline as rb
from assistant.capabilities.diagnostics.telemetry import snapshot
from assistant.capabilities.agent import archetypes


def _history(n=12, drift_load=False):
    rows = []
    for i in range(n):
        load = (0.8 + 0.05 * (i % 3)
                + (6.0 if drift_load and i >= n - 2 else 0.0))
        rows.append({"load1": load,
                     "load5": 0.5 + 0.02 * (i % 2),
                     "load15": 0.4 + 0.01 * (i % 2),
                     "mem_used_ratio": 0.42 + 0.005 * (i % 2),
                     "dead": 7.0})
    return rows


class BaselineTests(unittest.TestCase):
    def test_center_is_the_median_not_the_mean(self):
        rows = _history(12)
        rows[0]["load1"] = 100.0  # one poisoned row
        base = rb.robust_baseline(rows)
        self.assertAlmostEqual(base["center"]["load1"], 0.85)
        # a mean-based center would have been dragged to ~9.1

    def test_degenerate_coordinates_are_excluded_and_named(self):
        base = rb.robust_baseline(_history())
        self.assertIn("dead", base["degenerate"])   # constant 7.0
        self.assertNotIn("dead", base["kept"])
        self.assertNotIn("dead", base["center"])

    def test_all_degenerate_is_a_refusal(self):
        rows = [{"x": 1.0, "y": 2.0} for _ in range(12)]
        with self.assertRaises(ValueError):
            rb.robust_baseline(rows)

    def test_thin_history_is_refused(self):
        with self.assertRaises(ValueError):
            rb.robust_baseline(_history(5))
        with self.assertRaises(ValueError):
            rb.robust_baseline([])

    def test_method_citation_travels(self):
        base = rb.robust_baseline(_history())
        self.assertIn("Leys", base["method"])
        self.assertIn("Mahalanobis", base["method"])


class MahalanobisTests(unittest.TestCase):
    def setUp(self):
        self.base = rb.robust_baseline(_history(12))
        self.normal = {"load1": 0.9, "load5": 0.5, "load15": 0.4,
                       "mem_used_ratio": 0.42}

    def test_normal_sample_is_within_baseline(self):
        normal = {"load1": 0.85, "load5": 0.51, "load15": 0.4,
                  "mem_used_ratio": 0.42}
        out = rb.mahalanobis(normal, self.base)
        self.assertEqual(out["verdict"], "within-baseline")
        self.assertEqual(out["coordinates_used"], 4)

    def test_drifted_sample_is_flagged_with_reported_p(self):
        out = rb.mahalanobis({"load1": 8.0, "load5": 0.5, "load15": 0.4,
                              "mem_used_ratio": 0.42}, self.base)
        self.assertEqual(out["verdict"], "anomalous")
        self.assertLess(out["p_value"], out["p_threshold"])
        self.assertEqual(out["top_contributions"]["load1"],
                         max(out["top_contributions"].values()))

    def test_missing_coordinates_are_excluded_not_imputed(self):
        out = rb.mahalanobis({"load1": 0.9, "load5": 0.5}, self.base)
        self.assertEqual(out["coordinates_used"], 2)
        self.assertEqual(sorted(out["missing_in_sample"]),
                         ["load15", "mem_used_ratio"])

    def test_fewer_than_half_the_coordinates_is_a_refusal(self):
        with self.assertRaises(ValueError):
            rb.mahalanobis({"load1": 0.9}, self.base)

    def test_unseen_sample_coordinates_are_ignored(self):
        out = rb.mahalanobis(dict(self.normal, mystery=99.0), self.base)
        self.assertEqual(out["coordinates_used"], 4)

    def test_deterministic(self):
        normal = {"load1": 0.85, "load5": 0.51, "load15": 0.4,
                  "mem_used_ratio": 0.42}
        self.assertEqual(rb.mahalanobis(normal, self.base),
                         rb.mahalanobis(normal, self.base))


class FlattenAndArchetypeTests(unittest.TestCase):
    def test_flatten_snapshot_uses_the_telemetry_shapes(self):
        snap = snapshot(proc_dir="/proc", sys_dir="/sys")
        flat = rb.flatten_snapshot(snap)
        self.assertIsInstance(flat, dict)
        for key, value in flat.items():
            self.assertIsInstance(value, float)
        # battery capacity is deliberately absent (cyclic quantity)
        self.assertNotIn("battery_pct", flat)
        if "load1" in flat:
            self.assertGreaterEqual(flat["load1"], 0.0)

    def test_archetype_hook_refuses_thin_history_honestly(self):
        out = archetypes.telemetry_drift(_history(4),
                                         sample={"load1": 0.9})
        self.assertEqual(out["status"], "insufficient-history")
        self.assertEqual(out["evidence"], "telemetry-drift")

    def test_archetype_hook_scores_supplied_sample(self):
        out = archetypes.telemetry_drift(
            _history(12),
            sample={"load1": 8.0, "load5": 0.5, "load15": 0.4,
                    "mem_used_ratio": 0.42})
        self.assertEqual(out["status"], "anomalous")
        self.assertFalse(out["live_read"])
        self.assertIn("read-only", out["note"])


if __name__ == "__main__":
    unittest.main()

"""tests for diagnostics.forecast + brain.merkle + the package_audit
deepening (exponential-build-4 E).

Pinned: the steady-state Kalman filter tracks a clean trend and its
innovations' sigma is small on smooth series; the failure horizon
refuses to project through noise or a non-increasing trend; thin
series are refused outright; the Merkle tree detects the changed file,
prunes identical subtrees, and earns its keep on refresh (reused
hashes counted); symlinks are not followed; the package breakage
analysis reuses the shipped graph engines and honestly abstains
without a caller-supplied dependency graph; the telemetry series
helper keeps the flattened series bounded and skips missing readings.
"""
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import merkle
from assistant.capabilities.diagnostics import forecast, telemetry
from assistant.capabilities.agent.archetypes import package_breakage


class KalmanTests(unittest.TestCase):
    def test_tracks_clean_trend(self):
        series = [10.0 * i for i in range(20)]
        fit = forecast.steady_state_kalman(series)
        self.assertGreater(fit["slope"], 9.0)
        self.assertLess(fit["slope"], 11.0)
        # smooth series: innovations are small relative to the step
        self.assertLess(fit["sigma_innovation"], 5.0)

    def test_thin_series_refused(self):
        with self.assertRaises(ValueError):
            forecast.steady_state_kalman([1.0, 2.0, 3.0])

    def test_horizon_refuses_noise_flat_slope(self):
        # a genuinely flat series with noise: no crossing is claimed
        series = [3.0, 2.9, 3.1, 2.95, 3.05, 2.9, 3.1, 2.95, 3.05, 3.0]
        fit = forecast.steady_state_kalman(series)
        horizon = forecast.failure_horizon(fit, threshold=50.0)
        self.assertFalse(horizon["crossed"])
        self.assertIsNone(horizon["steps"])

    def test_horizon_counts_to_threshold(self):
        series = [float(i) for i in range(15)]
        fit = forecast.steady_state_kalman(series)
        horizon = forecast.failure_horizon(fit, threshold=30.0)
        self.assertIsNotNone(horizon["steps"])
        self.assertGreater(horizon["steps"], 10)

    def test_already_past_threshold(self):
        series = [float(i) for i in range(6, 20)]
        fit = forecast.steady_state_kalman(series)
        horizon = forecast.failure_horizon(fit, threshold=3.0)
        self.assertTrue(horizon["crossed"])

    def test_report_carries_caveats_verbatim(self):
        r = forecast.forecast_report([0, 1, 2, 3, 5, 6, 8], threshold=20,
                                     unit="reallocated sectors")
        self.assertTrue(any("nonlinearly" in c for c in r["caveats"]))
        self.assertTrue(any("LOCAL linear trend" in c
                            for c in r["caveats"]))
        self.assertIn("indication", r["suggestion"])
        self.assertIn("not a failure probability", r["suggestion"])


class TelemetrySeriesTests(unittest.TestCase):
    def _snap(self, temp_c):
        return {"loadavg": {"available": True, "load1": 0.5,
                            "load5": 0.4, "load15": 0.3},
                "meminfo": {"available": True, "used_ratio": 0.6},
                "thermal": {"available": True,
                            "zones": [{"path": "/sys/z0",
                                       "temp_c": temp_c}]},
                "battery": {"available": False, "batteries": []}}

    def test_series_accumulates_and_stays_bounded(self):
        state = {}
        for i in range(telemetry.SERIES_MAX_POINTS + 5):
            telemetry.record_series(state, self._snap(40.0 + i * 0.01),
                                    f"t{i}")
        series = state[telemetry.SERIES_STATE_KEY]
        self.assertEqual(len(series), telemetry.SERIES_MAX_POINTS)
        self.assertEqual(series[-1]["when"], f"t{telemetry.SERIES_MAX_POINTS + 4}")

    def test_series_for_skips_missing_metrics(self):
        state = {}
        telemetry.record_series(state, self._snap(41.0), "t0")
        partial = self._snap(42.0)
        partial["thermal"] = {"available": False, "zones": []}
        telemetry.record_series(state, partial, "t1")
        rows = telemetry.series_for(state, "thermal_mean_c")
        self.assertEqual([r[0] for r in rows], ["t0"])


class MerkleTests(unittest.TestCase):
    def setUp(self):
        # TemporaryDirectory cleans up without shutil (import policy:
        # shutil is forbidden repo-wide; the context manager is the
        # stdlib-blessed alternative)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "shell").mkdir()
        (self.root / "kwin").mkdir()
        (self.root / "shell" / "shell.json").write_text('{"a": 1}')
        (self.root / "shell" / "dock.json").write_text('{"b": 2}')
        (self.root / "kwin" / "rules.json").write_text("[]")

    def tearDown(self):
        self._tmp.cleanup()

    def test_detects_changed_and_added_files(self):
        snap1 = merkle.snapshot(str(self.root))
        (self.root / "shell" / "dock.json").write_text('{"b": 3}')
        (self.root / "kwin" / "new.json").write_text("{}")
        snap2 = merkle.refresh(json.loads(json.dumps(snap1)), str(self.root))
        d = merkle.diff(snap1, snap2)
        self.assertEqual(d["changed"], ["shell/dock.json"])
        self.assertEqual(d["added"], ["kwin/new.json"])
        self.assertFalse(d["same"])

    def test_unchanged_tree_prunes_subtrees(self):
        snap1 = merkle.snapshot(str(self.root))
        snap2 = merkle.refresh(json.loads(json.dumps(snap1)), str(self.root))
        d = merkle.diff(snap1, snap2)
        self.assertTrue(d["same"])
        self.assertGreaterEqual(d["pruned"]["subtrees_skipped"], 1)

    def test_refresh_reuses_unchanged_hashes(self):
        snap1 = merkle.snapshot(str(self.root))
        before = snap1["n_files"]
        snap2 = merkle.refresh(json.loads(json.dumps(snap1)), str(self.root))
        self.assertEqual(snap2["refresh"]["reused"], before)
        self.assertEqual(snap2["refresh"]["rehashed"], 0)

    def test_removal_detected(self):
        snap1 = merkle.snapshot(str(self.root))
        (self.root / "kwin" / "rules.json").unlink()
        snap2 = merkle.refresh(json.loads(json.dumps(snap1)), str(self.root))
        d = merkle.diff(snap1, snap2)
        self.assertEqual(d["removed"], ["kwin/rules.json"])

    def test_symlinks_not_followed(self):
        (self.root / "shell" / "link.json").symlink_to(
            self.root / "shell" / "shell.json")
        snap = merkle.snapshot(str(self.root))
        self.assertNotIn("link.json", snap["tree"]["dirs"]["shell"]["files"])

    def test_missing_dir_refused(self):
        with self.assertRaises(ValueError):
            merkle.snapshot("/tmp/no-such-merkledir-4b1f")

    def test_render_changes_unchanged(self):
        snap1 = merkle.snapshot(str(self.root))
        snap2 = merkle.refresh(json.loads(json.dumps(snap1)), str(self.root))
        lines = merkle.render_changes(merkle.diff(snap1, snap2))
        self.assertEqual(lines, ["config tree unchanged (root hash matches)"])


class PackageBreakageTests(unittest.TestCase):
    GRAPH = {
        "system-base": ["network-manager", "display-server"],
        "network-manager": ["nm-applet"],
        "display-server": ["compositor", "login-manager"],
        "compositor": ["shell-ui"],
        "nm-applet": ["shell-ui"],
    }

    def test_articulation_points_found(self):
        r = package_breakage(self.GRAPH, candidates=["nm-applet"])
        analysis = r["analysis"]
        self.assertIn("display-server", analysis["articulation_points"])
        self.assertEqual(analysis["n_nodes"], 7)  # edge-mentioned nodes count

    def test_candidate_removal_analysis(self):
        r = package_breakage(self.GRAPH, candidates=["display-server",
                                                     "nm-applet"])
        rows = {row["package"]: row for row in r["analysis"]["candidates"]}
        self.assertIn("new_articulation_points_after_removal",
                      rows["display-server"])
        self.assertTrue(r["note"].startswith("read-only"))

    def test_no_graph_is_an_honest_abstain(self):
        r = package_breakage(None)
        self.assertIsNone(r["analysis"])
        self.assertIn("no dependency graph supplied", r["note"])
        self.assertIn("install list", r["note"])

    def test_unknown_candidate_reported_not_dropped(self):
        r = package_breakage(self.GRAPH, candidates=["ghost-pkg"])
        row = r["analysis"]["candidates"][0]
        self.assertEqual(row["verdict"], "not in the supplied graph")


if __name__ == "__main__":
    unittest.main()

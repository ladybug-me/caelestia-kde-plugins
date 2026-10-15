"""Unit tests for the eval arena itself (stats, engine, ratchet)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from assistant.core.eval import stats
from assistant.core.eval.engine import (
    FOOTPRINT_BUDGETS,
    SUITES,
    _percentile,
    _values_equal,
    load_set,
    run_suite,
)


class StatsTests(unittest.TestCase):
    def test_bootstrap_ci_deterministic_and_bracketing(self):
        data = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        a = stats.bootstrap_ci(data, seed=7)
        b = stats.bootstrap_ci(data, seed=7)
        self.assertEqual(a, b)  # deterministic
        point, lo, hi = a
        self.assertAlmostEqual(point, 0.7, places=6)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_bootstrap_ci_extremes(self):
        for data in ([1.0] * 10, [0.0] * 10):
            point, lo, hi = stats.bootstrap_ci(data, seed=3)
            self.assertEqual(point, data[0])
            self.assertEqual(lo, point)
            self.assertEqual(hi, point)

    def test_paired_bootstrap_excludes_zero_for_clear_delta(self):
        a = [1.0] * 30
        b = [0.0] * 30
        point, lo, hi = stats.paired_bootstrap_delta(a, b)
        self.assertGreater(lo, 0.0)
        self.assertGreater(point, 0.0)

    def test_paired_bootstrap_aligned_required(self):
        with self.assertRaises(AssertionError):
            stats.paired_bootstrap_delta([1.0, 2.0], [1.0])

    def test_brier_perfect_and_worst(self):
        self.assertAlmostEqual(stats.brier([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(stats.brier([0.0, 1.0], [1, 0]), 1.0)

    def test_ece_perfectly_calibrated_and_overconfident(self):
        # 70% confident, 70% accurate over 10 items -> ECE 0
        probs = [0.7] * 10
        outs = [1] * 7 + [0] * 3
        self.assertAlmostEqual(stats.ece(probs, outs), 0.0, places=9)
        # 100% confident, 50% accurate -> ECE 0.5
        self.assertAlmostEqual(stats.ece([1.0] * 10, [1] * 5 + [0] * 5), 0.5)

    def test_benjamini_hochberg(self):
        # one tiny p survives; one large p does not
        reject = stats.benjamini_hochberg([0.001, 0.8], q=0.05)
        self.assertEqual(reject, [True, False])


class EngineHelperTests(unittest.TestCase):
    def test_values_equal_semantics(self):
        self.assertTrue(_values_equal(True, True))
        self.assertFalse(_values_equal(True, 1))          # bool identity strict
        self.assertTrue(_values_equal(0.8, 0.8000000001))  # float tolerance
        self.assertTrue(_values_equal("TwentyFourHour", "twentyfourhour"))
        self.assertTrue(_values_equal("anything", None))   # unpinned expectation
        self.assertFalse(_values_equal(None, "x"))          # missing value vs pinned

    def test_percentile_and_order(self):
        self.assertEqual(_percentile([1, 2, 3, 4, 5], 0.5), 3)

    def test_load_set_validates_items(self):
        data = load_set("routing", "dev")
        self.assertGreaterEqual(len(data["items"]), 40)
        for item in data["items"]:
            self.assertIn("id", item)
            self.assertIn("text", item)

    def test_suites_declared(self):
        self.assertEqual(
            set(SUITES),
            {"routing", "nlplan", "abstention", "diagnosis", "calibration",
             "footprint", "metamorphic"},
        )
        self.assertGreaterEqual(FOOTPRINT_BUDGETS["route_p50_ms"], 1.0)


class ArenaSmokeTests(unittest.TestCase):
    """Each suite runs end to end on its dev set and reports intervals."""

    @classmethod
    def setUpClass(cls):
        # hermetic: the arena is deterministic under DEFAULT_STATE; pin
        # HOME so ambient learned state can never leak in.
        cls._tmp = tempfile.TemporaryDirectory()
        cls._orig_home = os.environ.get("HOME")
        os.environ["HOME"] = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        if cls._orig_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = cls._orig_home
        cls._tmp.cleanup()

    def test_routing_suite(self):
        rep = run_suite("routing", split="dev")
        self.assertGreaterEqual(rep["n"], 80)
        point, lo, hi = rep["metrics"]["top1_rate"]
        self.assertTrue(0.0 <= lo <= point <= hi <= 1.0)

    def test_nlplan_suite(self):
        rep = run_suite("nlplan", split="dev")
        self.assertGreaterEqual(rep["n"], 25)
        self.assertIn("exact_rate", rep["metrics"])

    def test_abstention_suite(self):
        rep = run_suite("abstention", split="dev")
        self.assertGreaterEqual(rep["n"], 15)
        self.assertIn("abstain_rate", rep["metrics"])

    def test_diagnosis_suite(self):
        rep = run_suite("diagnosis", split="dev")
        self.assertGreaterEqual(rep["n"], 15)

    def test_calibration_suite(self):
        rep = run_suite("calibration", split="dev")
        self.assertGreaterEqual(rep["n"], 5)
        self.assertTrue(0.0 <= rep["metrics"]["brier"] <= 1.0)

    def test_unknown_suite_rejected(self):
        with self.assertRaises(ValueError):
            run_suite("nope")


class RatchetTests(unittest.TestCase):
    """eval/baseline.json floors hold (accuracy suites only)."""

    BASELINE = (Path(__file__).resolve().parents[3] / "assistant" /
                "core" / "eval" / "baseline.json")

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._orig_home = os.environ.get("HOME")
        os.environ["HOME"] = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        if cls._orig_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = cls._orig_home
        cls._tmp.cleanup()

    def test_dev_metrics_do_not_regress_below_recorded_floor(self):
        data = json.loads(self.BASELINE.read_text())
        recorded = data["metrics"]
        self.assertTrue(recorded, "baseline.json must carry metrics")
        current = {}
        for suite in ("routing", "nlplan", "abstention", "diagnosis"):
            rep = run_suite(suite, split="dev")
            for name, val in rep["metrics"].items():
                if isinstance(val, tuple):
                    current[f"{suite}.{name}"] = val[0]
        regressions = []
        for key, cur in sorted(current.items()):
            floor = recorded.get(key)
            if floor is None:
                continue
            if cur < floor["lo"] - 1e-9:
                regressions.append(
                    f"{key}: {cur:.4f} < recorded floor {floor['lo']:.4f} "
                    f"(point {floor['point']:.4f} recorded {data['recorded']})"
                )
        self.assertEqual(
            regressions, [],
            "dev arena regressed beyond its recorded interval — if the change "
            "is intentional, update eval/baseline.json in a reviewed commit "
            "that states why:\n" + "\n".join(regressions),
        )


    def test_sealed_sets_load_and_manifest_pins_content(self):
        # Stage B regression: the sealed CLI could never run as shipped —
        # load_set demanded `{suite}_sealed.json` while the files were
        # committed as `sealed_{suite}.json` (F1 birth defect, found and
        # fixed at Stage B). This test pins BOTH halves of that contract:
        # every suite's sealed split loads through the real loader, and
        # the sealed manifest's sha256 entries still match the content
        # (so the loader fix provably did not touch sealed content).
        import hashlib
        from assistant.core.eval import engine
        sealed = engine.available("sealed")
        self.assertIn("routing", sealed)
        for suite in sealed:
            data = engine.load_set(suite, "sealed")
            self.assertGreaterEqual(len(data.get("items", [])), 1)
        manifest = engine.SET_DIR / "sealed_manifest.sha256"
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            path = engine.SET_DIR / name.strip()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                digest, actual,
                f"sealed content drift in {name}: the sealed set must "
                f"never change without re-authoring discipline")


if __name__ == "__main__":
    unittest.main()

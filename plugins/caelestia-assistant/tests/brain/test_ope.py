"""Tests for exponential-build phase 3.2: the off-policy evaluation
gate for capability kill-switches (`brain/ope.py`, CLI `brain ope`).

The contract under test:

- the importance-weighted replay prints the estimated accept rate of a
  candidate kill-switch policy over the historical approve/reject log,
  with support and coverage accounting (Horvitz-Thompson weights;
  Dudik/Langford/Li 2011 framing);
- honest refusals: empty log, zero support (pure extrapolation), thin
  support (< 5) — all labeled, never a shrug of a number;
- the gate produces NO autonomy change: the ledger is byte-identical
  after the report, and the standing note says the switch stays a
  human file edit;
- an unknown switch name lists the known ones; the CLI wires the whole
  thing read-only end to end.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import ope
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.ope import evaluate_policy, ope_report


def _eps(approvals, rejections, kind="settings", target="dbus",
         diff=None):
    diff = diff if diff is not None else {"tool": "setDBus"}
    rows = []
    for _ in range(approvals):
        rows.append({"context": {"kind": kind, "target": target,
                                 "diff": diff},
                     "approved": True})
    for _ in range(rejections):
        rows.append({"context": {"kind": kind, "target": target,
                                 "diff": diff},
                     "approved": False})
    return rows


class EstimatorTests(unittest.TestCase):
    def test_ips_estimate_is_the_supported_accept_rate(self):
        eps = _eps(6, 2) + _eps(3, 0, kind="agent", target="other")
        out = evaluate_policy(eps, lambda ctx: ctx.get("target") == "dbus")
        self.assertIsNone(out["refused"])
        self.assertAlmostEqual(out["estimated_accept_rate"], 0.75)
        self.assertEqual(out["support"], 8)
        self.assertEqual(out["n_episodes"], 11)
        self.assertAlmostEqual(out["coverage"], 8 / 11, places=4)

    def test_full_coverage_equals_the_raw_rate(self):
        eps = _eps(3, 1) + _eps(1, 0, target="other")  # 5 >= min support
        out = evaluate_policy(eps, lambda ctx: True)
        self.assertAlmostEqual(out["estimated_accept_rate"], 0.8)
        self.assertEqual(out["coverage"], 1.0)

    def test_out_of_support_episodes_are_weighted_zero(self):
        eps = _eps(5, 0) + _eps(0, 10, target="never-proposed-by-candidate")
        out = evaluate_policy(eps, lambda ctx: ctx.get("target") == "dbus")
        self.assertAlmostEqual(out["estimated_accept_rate"], 1.0)
        self.assertEqual(out["support"], 5)
        self.assertLess(out["coverage"], 0.4)  # the gap is REPORTED


class RefusalTests(unittest.TestCase):
    def test_empty_log_refused(self):
        out = evaluate_policy([], lambda ctx: True)
        self.assertIn("empty log", out["refused"])
        self.assertIsNone(out["estimated_accept_rate"])

    def test_zero_support_is_pure_extrapolation_and_refused(self):
        out = evaluate_policy(_eps(4, 4), lambda ctx: False)
        self.assertIn("extrapolation", out["refused"])
        self.assertIsNone(out["estimated_accept_rate"])

    def test_thin_support_is_labeled_thin(self):
        out = evaluate_policy(_eps(4, 0), lambda ctx: True)  # 4 < 5
        self.assertIn("thin", out["refused"])
        self.assertIsNone(out["estimated_accept_rate"])
        self.assertEqual(out["support"], 4)


class SwitchPolicyTests(unittest.TestCase):
    def test_known_switch_filters_by_kind_and_diff(self):
        eps = _eps(5, 1) + _eps(2, 0, target="wallpaper",
                                diff={"tool": "setWallpaper"})
        out = ope_report(eps, "dbus_surface")
        self.assertIsNone(out["refused"])
        self.assertEqual(out["support"], 6)  # wallpaper rows: no 'dbus' diff

    def test_unknown_switch_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            ope_report([], "warp_drive")
        self.assertIn("known:", str(ctx.exception))
        self.assertIn("dbus_surface", str(ctx.exception))


class NoAutonomyChangeTests(unittest.TestCase):
    def test_report_writes_nothing_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = Ledger(path)
            for _ in range(6):
                pid = ledger.propose("settings", "dbus",
                                     {"tool": "setDBus"}, "r", 0.7)
                ledger.decide(pid, True)
            before = path.read_text(encoding="utf-8")
            episodes = [{"context": {"kind": i["kind"],
                                     "target": i["target"],
                                     "diff": i["diff"]},
                         "approved": i["status"] == "approved"}
                        for i in ledger.labeled()]
            out = ope_report(episodes, "dbus_surface")
            self.assertIsNone(out["refused"])
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertIn("never writes", out["note"])
            self.assertIn("file edit", out["note"])


class CliTests(unittest.TestCase):
    def test_cli_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = Ledger(path)
            for _ in range(5):
                pid = ledger.propose("settings", "dbus",
                                     {"tool": "setDBus"}, "r", 0.7)
                ledger.decide(pid, True)
            from assistant.capabilities.brain.cli import main as brain_main
            buf = io.StringIO()
            rc = brain_main(["ope", "dbus_surface", "--ledger", str(path)],
                            buf)
            self.assertEqual(rc, 0)
            self.assertIn("estimated accept rate if proposed: 1.0",
                          buf.getvalue())
            self.assertIn("evidence only", buf.getvalue())

    def test_cli_lists_switches(self):
        from assistant.capabilities.brain.cli import main as brain_main
        buf = io.StringIO()
        rc = brain_main(["ope", "x", "--list-switches"], buf)
        self.assertEqual(rc, 0)
        self.assertIn("dbus_surface", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

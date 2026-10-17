"""F30 tests — context-aware recommendations (cortex/context.py).

Under test:

- power probe: injectable root; absent batteries -> None (a desktop
  gets NO battery advice); low+discharging -> the battery-saver
  proposal with the cited evidence and an inert command; low+charging
  -> an honest no-change note; unreadable -> reported, not guessed;
- the clock is never evidence: no model -> no time finding even though
  now is passed; a fitted brain.prefs model's qualifying pattern is
  SELECTED by the current hour bucket (other buckets stay quiet);
- determinism: same injections -> same findings;
- the CLI verb renders the honest no-findings line when nothing
  qualifies.
"""

from __future__ import annotations

import io
import contextlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.capabilities.brain.prefs import PreferenceModel
from assistant.core import context as context_mod
from assistant.core import cli as cortex_cli


def _fake_sysroot(tmp: Path, percent: str = "14", status: str = "Discharging"):
    bat = tmp / "BAT0"
    bat.mkdir(parents=True, exist_ok=True)
    (bat / "capacity").write_text(percent + "\n", encoding="utf-8")
    (bat / "status").write_text(status + "\n", encoding="utf-8")
    return str(tmp / "BAT*")


class PowerProbeTests(unittest.TestCase):
    def test_no_battery_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                context_mod.read_power_state(str(Path(tmp) / "BAT*")))

    def test_low_discharging_proposes_battery_saver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sysroot(Path(tmp), percent="14",
                                 status="Discharging")
            power = context_mod.read_power_state(root)
        view = context_mod.recommend_contextual(power=power)
        kinds = [f["kind"] for f in view["findings"]]
        self.assertEqual(kinds, ["battery_low"])
        finding = view["findings"][0]
        self.assertIn("14%", finding["evidence"])
        self.assertIn("discharging", finding["evidence"])
        self.assertIn("SUGGESTED_NOT_EXECUTED", finding["command"])
        self.assertIn("battery-saver", finding["command"])

    def test_low_charging_is_an_honest_no_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sysroot(Path(tmp), percent="18", status="Charging")
            power = context_mod.read_power_state(root)
        view = context_mod.recommend_contextual(power=power)
        self.assertEqual(view["findings"][0]["kind"], "battery_low_charging")
        self.assertEqual([f for f in view["findings"] if f.get("command")],
                         [], "charging must not produce a change command")

    def test_healthy_battery_is_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sysroot(Path(tmp), percent="80", status="Discharging")
            power = context_mod.read_power_state(root)
        view = context_mod.recommend_contextual(power=power)
        self.assertEqual(view["findings"], [])

    def test_unreadable_battery_is_reported(self) -> None:
        power = {"percent": None, "charging": False,
                 "source": "/sys/class/power_supply/BAT0",
                 "error": "unreadable"}
        view = context_mod.recommend_contextual(power=power)
        self.assertEqual(view["findings"][0]["kind"], "power_unreadable")


class ClockIsNotEvidenceTests(unittest.TestCase):
    def test_no_model_no_time_finding(self) -> None:
        view = context_mod.recommend_contextual(
            now=datetime(2026, 9, 28, 21, 0))
        self.assertEqual([f for f in view["findings"]
                          if f["kind"] == "hour_pattern"], [])

    def test_model_evidence_selected_by_clock(self) -> None:
        model = PreferenceModel()
        for _ in range(5):
            model.observe("animations", "-1", True, 21)   # bucket 3
        for _ in range(5):
            model.observe("bar", "+1", True, 9)           # bucket 1
        # evening: only the bucket-3 pattern is relevant
        view = context_mod.recommend_contextual(
            now=datetime(2026, 9, 28, 21, 0), model=model)
        patterns = [f for f in view["findings"]
                    if f["kind"] == "hour_pattern"]
        self.assertEqual(len(patterns), 1)
        self.assertIn("animations", patterns[0]["evidence"])
        # with now=None every qualifying pattern is reported
        all_view = context_mod.recommend_contextual(model=model)
        self.assertEqual(len([f for f in all_view["findings"]
                              if f["kind"] == "hour_pattern"]), 2)

    def test_determinism(self) -> None:
        model = PreferenceModel()
        for _ in range(6):
            model.observe("dock", "-1", True, 2)
        now = datetime(2026, 9, 28, 3, 0)
        a = context_mod.recommend_contextual(now=now, model=model)
        b = context_mod.recommend_contextual(now=now, model=model)
        self.assertEqual(a["findings"], b["findings"])


class CliTests(unittest.TestCase):
    def test_context_verb_renders_honest_empty(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cortex_cli.main(["context", "--now", "2026-09-28T21:00"])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertIn("no contextual recommendations", out.getvalue())


if __name__ == "__main__":
    unittest.main()

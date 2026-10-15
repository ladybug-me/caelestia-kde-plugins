"""Tests for issue #120 Phase 3: read-only telemetry probes (3.1), the
battery-aware secondary bandit reward (3.2), and startup-time regression
detection over the existing CUSUM (3.3).

Telemetry tests use FIXTURE files — nothing here requires /proc or /sys
to exist on the runner, and no daemon or poller is created.
"""
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import bridge as brain_bridge
from assistant.capabilities.brain import service as brain_service
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.preset_bandit import SECONDARY_REWARD_WEIGHT, NamedBandit
from assistant.capabilities.diagnostics import telemetry
from assistant.capabilities.genius.data import changepoints, startup_regressions


def _fixture_os(root: Path):
    """A fake /proc + /sys tree with the four probe interfaces."""
    (root / "proc").mkdir()
    (root / "proc" / "loadavg").write_text("0.10 0.50 0.90 1/100 2000\n")
    (root / "proc" / "meminfo").write_text(
        "MemTotal:       16000000 kB\n"
        "MemFree:         2000000 kB\n"
        "MemAvailable:    8000000 kB\n"
        "Buffers:          500000 kB\n")
    ps = root / "sys" / "class" / "power_supply" / "BAT0"
    ps.mkdir(parents=True)
    (ps / "capacity").write_text("77\n")
    th = root / "sys" / "class" / "thermal" / "thermal_zone0"
    th.mkdir(parents=True)
    (th / "temp").write_text("45500\n")


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _fixture_os(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loadavg_meminfo_from_fixtures(self):
        load = telemetry.read_loadavg(self.root / "proc" / "loadavg")
        self.assertTrue(load["available"])
        self.assertAlmostEqual(load["load1"], 0.10)
        self.assertAlmostEqual(load["load15"], 0.90)
        mem = telemetry.read_meminfo(self.root / "proc" / "meminfo")
        self.assertTrue(mem["available"])
        self.assertEqual(mem["total_kb"], 16000000)
        self.assertAlmostEqual(mem["used_ratio"], 0.5, places=2)

    def test_battery_and_thermal_from_fixtures(self):
        bat = telemetry.read_battery(
            str(self.root / "sys/class/power_supply/*/capacity"))
        self.assertTrue(bat["available"])
        self.assertEqual(bat["batteries"][0]["capacity_pct"], 77)
        th = telemetry.read_thermal(
            str(self.root / "sys/class/thermal/*/temp"))
        self.assertAlmostEqual(th["zones"][0]["temp_c"], 45.5)

    def test_missing_interfaces_report_unavailable_not_fake(self):
        load = telemetry.read_loadavg(self.root / "proc" / "nope")
        self.assertFalse(load["available"])
        bat = telemetry.read_battery(
            str(self.root / "sys/class/power_supply/*/capacity_nowhere"))
        self.assertFalse(bat["available"])

    def test_snapshot_partial_honesty(self):
        (self.root / "proc" / "meminfo").unlink()
        snap = telemetry.snapshot(proc_dir=str(self.root / "proc"),
                                  sys_dir=str(self.root / "sys"))
        self.assertTrue(snap["loadavg"]["available"])
        self.assertFalse(snap["meminfo"]["available"])
        self.assertTrue(snap["battery"]["available"])

    def test_drain_rate_and_reward_mapping(self):
        rate = telemetry.drain_rate_percent_per_hour(80, 70, 2.0)
        self.assertAlmostEqual(rate, 5.0)  # drained 10% over 2h
        self.assertEqual(telemetry.reward_from_drain_delta(-5.0), 1.0)   # big improvement
        self.assertEqual(telemetry.reward_from_drain_delta(5.0), 0.0)    # big regression
        self.assertAlmostEqual(telemetry.reward_from_drain_delta(0.0), 0.5)  # neutral
        with self.assertRaises(ValueError):
            telemetry.drain_rate_percent_per_hour(80, 70, 0.0)
        with self.assertRaises(ValueError):
            telemetry.drain_rate_percent_per_hour(101, 70, 1.0)

    def test_bridge_op_is_one_ondemand_snapshot(self):
        r = brain_bridge.handle(
            {"op": "telemetry_snapshot", "proc_dir": str(self.root / "proc"),
             "sys_dir": str(self.root / "sys")},
            state_path=str(self.root / "state.json"),
            ledger_path=str(self.root / "l.json"))
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"]["loadavg"]["available"])


class BatteryRewardTests(unittest.TestCase):
    def test_secondary_term_is_a_noop_when_none(self):
        b1 = NamedBandit({"p": [3.0, 2.0]})
        b2 = NamedBandit({"p": [3.0, 2.0]})
        b1.reward("p", True)
        b2.reward("p", True, secondary=None)
        self.assertEqual(b1.arms, b2.arms)  # byte-identical update

    def test_secondary_term_shifts_meaningfully_but_never_overwhelms(self):
        helped = NamedBandit()
        helped.reward("preset", True)
        helped.reward("preset", True, secondary=1.0)  # battery got better
        plain = NamedBandit()
        plain.reward("preset", True)
        plain.reward("preset", True)  # same two approvals, no telemetry
        self.assertAlmostEqual(
            helped.arms["preset"][0] - plain.arms["preset"][0],
            SECONDARY_REWARD_WEIGHT)
        hurt = NamedBandit()
        hurt.reward("preset", True)
        hurt.reward("preset", True, secondary=0.0)  # battery got worse
        self.assertAlmostEqual(hurt.arms["preset"][1],
                               1.0 + SECONDARY_REWARD_WEIGHT)
        # even a max-strength negative secondary cannot flip a rejection:
        # primary rejection adds beta += 1.0, secondary adds at most 0.25.
        mixed = NamedBandit()
        mixed.reward("p", False)
        mixed.reward("p", False, secondary=1.0)
        alpha, beta = mixed.arms["p"]
        self.assertLess(alpha, beta)  # the primary signal still dominates
        # alpha = 1 (prior) + 0.25 (max secondary); beta = 1 + 1 + 1
        # (prior + two rejections) — the secondary added nothing to beta
        self.assertAlmostEqual(alpha, 1.0 + SECONDARY_REWARD_WEIGHT)
        self.assertAlmostEqual(beta, 3.0)

    def test_ranking_shifts_only_when_secondary_present(self):
        # Identical priors. Without secondary, one approval each keeps the
        # two arms' means EQUAL (no-op). With secondary, the battery signal
        # alone separates them — the property the mission asks to pin.
        b1 = NamedBandit({"battery-saver": [2.0, 2.0], "gaming": [2.0, 2.0]})
        b2 = NamedBandit({"battery-saver": [2.0, 2.0], "gaming": [2.0, 2.0]})
        for name in ("battery-saver", "gaming"):
            b1.reward(name, True)
            b2.reward(name, True, secondary=None)
        m1 = {n: a / (a + b_) for n, (a, b_) in b1.arms.items()}
        m2 = {n: a / (a + b_) for n, (a, b_) in b2.arms.items()}
        self.assertAlmostEqual(m2["battery-saver"], m2["gaming"])  # no-op path
        b1.reward("battery-saver", True, secondary=1.0)   # drains slower now
        b1.reward("gaming", True, secondary=0.0)          # drains faster now
        m1 = {n: a / (a + b_) for n, (a, b_) in b1.arms.items()}
        self.assertGreater(m1["battery-saver"], m1["gaming"])

    def test_settings_bridge_passes_the_secondary_signal(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            target = d / "shell.json"
            target.write_text("{}")
            ledger = Ledger(d / "l.json")
            from assistant.capabilities.brain import settings_bridge
            pid = settings_bridge.propose(
                ledger, target, preset="minimal", reason="try minimal")
            bandit = NamedBandit()
            settings_bridge.decide(ledger, pid["proposal_id"], True,
                                   bandit=bandit, battery_reward=1.0)
            alpha, _beta = bandit.arms["minimal"]
            self.assertAlmostEqual(alpha, 1.0 + 1.0 + SECONDARY_REWARD_WEIGHT)

    def test_service_settings_decide_accepts_battery_reward(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            target = d / "shell.json"
            target.write_text(json_cfg := "{}")
            ledger_path = str(d / "l.json")
            state_path = str(d / "s.json")
            from assistant.capabilities.brain import settings_bridge
            ledger = Ledger(ledger_path)
            pid = settings_bridge.propose(
                ledger, target, calls=[{"tool": "setAnimationSpeed",
                                        "action": "set", "value": 0.5,
                                        "raw": "battery test"}],
                reason="battery reward test")
            result = brain_service.settings_decide(
                pid["proposal_id"], True, ledger_path, state_path,
                battery_reward=0.0)
            self.assertTrue(result["applied"])


class StartupRegressionTests(unittest.TestCase):
    def test_known_changepoint_detected_at_the_right_index(self):
        # 10 boots at 5s, then 10 at 10s: the shipped CUSUM crossing lands
        # at index 9 (stable, seeded); the wrapper must report exactly that
        # index with the regression interpretation.
        series = [5.0] * 10 + [10.0] * 10
        result = startup_regressions(series)
        self.assertEqual(len(result["regressions"]), 1)
        reg = result["regressions"][0]
        self.assertEqual(reg["index"], changepoints(series)["changepoints"][0])
        self.assertEqual(reg["index"], 9)
        self.assertGreaterEqual(reg["delta_pct"], 15.0)
        self.assertAlmostEqual(reg["pre_mean"], 5.0, delta=0.1)

    def test_flat_series_has_no_regressions(self):
        result = startup_regressions([5.0] * 12)
        self.assertEqual(result["regressions"], [])

    def test_improvement_is_labeled_not_a_regression(self):
        series = [8.0] * 10 + [5.0] * 10  # boots got FASTER at index 10
        result = startup_regressions(series)
        kinds = [e["kind"] for e in result["events"]]
        self.assertIn("improvement", kinds)
        self.assertEqual(result["regressions"], [])

    def test_dates_and_versions_come_only_from_the_input(self):
        series = [5.0] * 10 + [10.0] * 10
        stamps = [f"2026-09-{i:02d}T10:00:00+00:00" for i in range(1, 21)]
        # the CUSUM crossing at index 9 is the step boundary; the label
        # the caller attaches to index 9 is what gets reported verbatim
        versions = ["1.2.0"] * 9 + ["1.3.0"] * 11
        result = startup_regressions(series, timestamps=stamps, labels=versions)
        reg = result["regressions"][0]
        self.assertEqual(reg["at"], "2026-09-10T10:00:00+00:00")
        self.assertEqual(reg["version"], "1.3.0")

    def test_underlying_cusum_still_works_unchanged(self):
        out = changepoints([5.0] * 10 + [10.0] * 10)
        self.assertEqual(out["changepoints"], [9])


if __name__ == "__main__":
    unittest.main()

"""F19 tests — pull-based workspace schedules (cortex/schedules.py).

Under test:

- save validates NOW (profile/preset existence, window shape, exactly
  one action, at least one condition) and stores in the settings
  history file's bounded "schedules" key; the target is untouched;
- eval is deterministic given (now, power): time windows including
  midnight-wrapping ones, on-battery conditions (only with a probe
  result — absent probe -> SKIPPED with the reason, never guessed),
  and honest SKIPPED rows for actions whose profile no longer exists
  (F7);
- firing schedules render SUGGESTED_NOT_EXECUTED commands; file puts
  them into the brain ledger with the personalize dedupe/cooldown
  discipline, writing ONLY the ledger.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.capabilities.brain.ledger import Ledger
from assistant.core import schedules


def _mk(initial=None):
    tmp = Path(tempfile.mkdtemp(prefix="sched-test-"))
    target = tmp / "shell.json"
    target.write_text(json.dumps(initial or {}), encoding="utf-8")
    return tmp, target


class SaveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        self.before = self.target.read_bytes()

    def test_save_with_preset_and_window(self) -> None:
        entry = schedules.save(self.target, "evening-quiet",
                               preset="battery-saver",
                               window="21:00-23:30")
        self.assertEqual(entry["name"], "evening-quiet")
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_save_with_profile_requires_existing(self) -> None:
        from assistant.capabilities.settings.profiles import save as profile_save
        profile_save(self.target, "combo",
                     [{"kind": "preset", "name": "gaming"}])
        entry = schedules.save(self.target, "game-time", profile="combo",
                               window="19:00-22:00")
        self.assertEqual(entry["action"], {"profile": "combo"})
        with self.assertRaises(schedules.ScheduleError):
            schedules.save(self.target, "ghost", profile="nope",
                           window="19:00-22:00")

    def test_save_validation(self) -> None:
        with self.assertRaises(schedules.ScheduleError):
            schedules.save(self.target, "x", preset="gaming")  # no condition
        with self.assertRaises(schedules.ScheduleError):
            schedules.save(self.target, "x", window="25:00-26:00",
                           preset="gaming")
        with self.assertRaises(schedules.ScheduleError):
            schedules.save(self.target, "x", window="9:00-10:00")  # no action
        with self.assertRaises(schedules.ScheduleError):
            # both actions is one too many
            schedules.save(self.target, "x", window="9:00-10:00",
                           profile="combo", preset="gaming")
        with self.assertRaises(schedules.ScheduleError):
            schedules.save(self.target, "x", window="9:00-10:00",
                           preset="no-such-preset")


class EvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        schedules.save(self.target, "evening-quiet", preset="battery-saver",
                       window="21:00-23:30")
        schedules.save(self.target, "wrap-night", preset="minimal",
                       window="23:30-05:00")

    def test_window_and_midnight_wrap(self) -> None:
        at_22 = datetime(2026, 9, 28, 22, 0)
        result = schedules.evaluate(self.target, now=at_22)
        names = {f["schedule"] for f in result["firing"]}
        self.assertEqual(names, {"evening-quiet"})
        at_23 = datetime(2026, 9, 28, 23, 45)
        result = schedules.evaluate(self.target, now=at_23)
        names = {f["schedule"] for f in result["firing"]}
        self.assertEqual(names, {"wrap-night"},
                         "23:45 is past 23:30, so only the wrapped "
                         "window fires")
        at_4 = datetime(2026, 9, 29, 4, 0)
        result = schedules.evaluate(self.target, now=at_4)
        names = {f["schedule"] for f in result["firing"]}
        self.assertEqual(names, {"wrap-night"})

    def test_battery_condition_without_probe_skips(self) -> None:
        schedules.save(self.target, "on-batt", preset="battery-saver",
                       on_battery=True)
        result = schedules.evaluate(self.target,
                                    now=datetime(2026, 9, 28, 22, 0))
        skipped = {s["schedule"]: s["reason"] for s in result["skipped"]}
        self.assertIn("on-batt", skipped)
        self.assertIn("not guessed", skipped["on-batt"])

    def test_battery_condition_with_probe(self) -> None:
        schedules.save(self.target, "on-batt", preset="battery-saver",
                       on_battery=True)
        low = {"percent": 15.0, "charging": False, "source": "BAT0"}
        result = schedules.evaluate(self.target,
                                    now=datetime(2026, 9, 28, 22, 0),
                                    power=low)
        self.assertIn("on-batt", {f["schedule"] for f in result["firing"]})
        charging = {"percent": 60.0, "charging": True, "source": "BAT0"}
        result = schedules.evaluate(self.target,
                                    now=datetime(2026, 9, 28, 22, 0),
                                    power=charging)
        self.assertNotIn("on-batt", {f["schedule"] for f in result["firing"]})

    def test_missing_profile_is_skipped_and_named(self) -> None:
        from assistant.capabilities.settings.profiles import save as profile_save
        profile_save(self.target, "combo",
                     [{"kind": "preset", "name": "gaming"}])
        schedules.save(self.target, "game-time", profile="combo",
                       window="00:00-23:59")
        from assistant.capabilities.settings.profiles import delete as profile_delete
        profile_delete(self.target, "combo")
        result = schedules.evaluate(self.target,
                                    now=datetime(2026, 9, 28, 12, 0))
        skipped = {s["schedule"] for s in result["skipped"]}
        self.assertIn("game-time", skipped)
        self.assertNotIn("game-time", {f["schedule"]
                                       for f in result["firing"]})

    def test_firing_commands_are_inert(self) -> None:
        result = schedules.evaluate(self.target,
                                    now=datetime(2026, 9, 28, 22, 0))
        for f in result["firing"]:
            self.assertTrue(f["command"].startswith("SUGGESTED_NOT_EXECUTED"))


class FileTests(unittest.TestCase):
    def test_file_dedupes_and_writes_only_ledger(self) -> None:
        tmp, target = _mk()
        schedules.save(target, "evening-quiet", preset="battery-saver",
                       window="21:00-23:30")
        ledger_path = tmp / "ledger.json"
        ledger = Ledger(ledger_path)
        result = schedules.evaluate(target,
                                    now=datetime(2026, 9, 28, 22, 0))
        stats1 = schedules.file_firing(result, ledger)
        self.assertEqual(stats1["filed"], 1)
        stats2 = schedules.file_firing(result, ledger)
        self.assertEqual(stats2["filed"], 0)
        self.assertEqual(stats2["skipped_pending"], 1)
        item = ledger.pending()[0]
        self.assertEqual(item["kind"], "schedule_firing")
        self.assertIn("SUGGESTED_NOT_EXECUTED", item["diff"]["command"])


if __name__ == "__main__":
    unittest.main()

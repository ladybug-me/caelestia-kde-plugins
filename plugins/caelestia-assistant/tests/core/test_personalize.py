"""F16 tests — personalized suggestions wired into the #120 proposal
surface (cortex/personalize.py).

Under test:

- recurring-apply mining: the same normalized label applied MIN_RECUR
  times yields one evidence-cited suggestion group; below the floor
  yields nothing; labels differing only by ids normalize together;
- hour-pattern mining: a brain.prefs model with enough effective
  observations and a mean above the floor yields a pattern; a weak or
  thin posterior does not;
- filing: into the brain LEDGER (the existing #120 proposal surface) —
  pending duplicates never re-file, recently-decided targets keep
  quiet for REJECT_COOLDOWN_DAYS, the pending suggestion set is
  bounded (MAX_PENDING, oldest evicted), and ONLY the ledger is
  written (history bytes untouched);
- the CLI verbs: personalize list is read-only, personalize file
  reports its dedupe stats, both accept explicit paths (no HOME
  surprises).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.prefs import PreferenceModel
from assistant.core import personalize
from assistant.capabilities.settings.history import entries as history_entries


def _mk() -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="personalize-test-"))
    target = tmp / "shell.json"
    target.write_text("{}\n", encoding="utf-8")
    return tmp, target


def _hist(target: Path, labels_with_hours):
    """Fabricate an applied history ring (newest first) with timestamps."""
    base = datetime(2026, 9, 28, 12, 0)
    entries = []
    eid = len(labels_with_hours)
    for label, hours_ago in labels_with_hours:
        entries.append({
            "id": eid, "at": (base - timedelta(hours=hours_ago)).isoformat(
                timespec="seconds"),
            "label": label,
            "ops": [{"path": "bar.scale", "old": 0.9, "new": 1.0}],
        })
        eid -= 1
    data = {"entries": entries}
    hist_path = target.parent / (target.name + ".assistant-history.json")
    hist_path.write_text(json.dumps(data), encoding="utf-8")
    return hist_path


class RecurringTests(unittest.TestCase):
    def test_recurring_label_yields_one_group(self) -> None:
        labels = [("preset: battery-saver", 1), ("preset: battery-saver #3", 5),
                  ("preset:  battery-saver", 9), ("preset: compact", 2)]
        groups = personalize.recurring_applies(labels, min_recur=3) \
            if isinstance(labels, list) and isinstance(labels[0], dict) else []
        # (the helper above takes dicts; use the real shape:)
        entries = [
            {"id": 3, "at": "2026-09-28T11:00:00", "label": "preset: battery-saver",
             "ops": []},
            {"id": 2, "at": "2026-09-28T07:00:00", "label": "preset: battery-saver #3",
             "ops": []},
            {"id": 1, "at": "2026-09-28T03:00:00", "label": "preset:  battery-saver",
             "ops": []},
        ]
        groups = personalize.recurring_applies(entries)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["label"], "preset: battery-saver")
        self.assertEqual(group["count"], 3)
        self.assertEqual(group["evidence_ids"], [3, 2, 1])
        self.assertEqual(group["hours"], {3: 1, 7: 1, 11: 1})

    def test_below_floor_yields_nothing(self) -> None:
        entries = [
            {"id": 1, "at": "2026-09-28T11:00:00", "label": "preset: compact",
             "ops": []},
            {"id": 2, "at": "2026-09-28T10:00:00", "label": "preset: compact",
             "ops": []},
        ]
        self.assertEqual(personalize.recurring_applies(entries), [])

    def test_ordering_count_desc(self) -> None:
        entries = [
            {"id": i, "at": f"2026-09-28T0{i % 9}:00:00",
             "label": "preset: a" if i < 5 else "preset: b", "ops": []}
            for i in range(1, 8)
        ]
        groups = personalize.recurring_applies(entries)
        self.assertEqual([g["count"] for g in groups], [4, 3])


class HourPatternTests(unittest.TestCase):
    def test_strong_pattern_qualifies(self) -> None:
        model = PreferenceModel()
        for _ in range(5):
            model.observe("animations", "-1", True, 21)
        patterns = personalize.hour_patterns(model)
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        self.assertEqual((p["group"], p["direction"], p["bucket"]),
                         ("animations", "-1", 3))
        self.assertGreaterEqual(p["mean"], personalize.BIAS_FLOOR)

    def test_thin_or_weak_posteriors_do_not_qualify(self) -> None:
        model = PreferenceModel()
        for _ in range(2):  # below MIN_OBS
            model.observe("bar", "+1", True, 9)
        for _ in range(6):  # enough obs, but rejected -> mean low
            model.observe("dock", "+1", False, 9)
        self.assertEqual(personalize.hour_patterns(model), [])


class FileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        self.before = self.target.read_bytes()
        self.ledger_path = self.tmp / "ledger.json"
        self.ledger = Ledger(self.ledger_path)
        self.entries = [
            {"id": i, "at": f"2026-09-28T1{i}:00:00",
             "label": "preset: battery-saver", "ops": []}
            for i in range(1, 4)
        ]
        # HERMETIC anchor (fixes a time bomb): Ledger.decide stamps the
        # REAL UTC clock, and the reject-cooldown compares that stamp
        # against the injected ``now``. A fixed 2026-09-28 anchor meant
        # ``later - decided_at`` shrank below REJECT_COOLDOWN_DAYS as
        # the real calendar advanced, so this suite started failing on
        # its own date. Anchor ``now`` to the real clock instead: the
        # decision stamp always lands ~0 days after ``now`` and ~15
        # days before ``later`` — the relations under test hold at any
        # real date.
        real = datetime.now(timezone.utc).replace(tzinfo=None)
        self.now = real
        self.later = real + timedelta(days=personalize.REJECT_COOLDOWN_DAYS + 1)

    def test_files_once_and_dedupes(self) -> None:
        mined = personalize.mine(self.entries)
        stats1 = personalize.file_suggestions(mined, self.ledger, now=self.now)
        self.assertEqual(stats1["filed"], 1)
        stats2 = personalize.file_suggestions(mined, self.ledger, now=self.now)
        self.assertEqual(stats2["filed"], 0)
        self.assertEqual(stats2["skipped_pending"], 1)
        pending = self.ledger.pending()
        self.assertEqual(pending[0]["kind"], "personalized_suggestion")
        self.assertIn("SUGGESTED_NOT_EXECUTED",
                      pending[0]["diff"]["command"])
        self.assertIn("--profile-save", pending[0]["diff"]["command"])
        # only the ledger grew; the settings target untouched
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_cooldown_after_rejection(self) -> None:
        mined = personalize.mine(self.entries)
        personalize.file_suggestions(mined, self.ledger, now=self.now)
        item = self.ledger.pending()[0]
        self.ledger.decide(item["id"], approve=False)
        stats = personalize.file_suggestions(mined, self.ledger, now=self.now)
        self.assertEqual(stats["filed"], 0)
        self.assertEqual(stats["skipped_cooldown"], 1)
        stats_later = personalize.file_suggestions(mined, self.ledger,
                                                   now=self.later)
        self.assertEqual(stats_later["filed"], 1)

    def test_bounded_pending_evicts_oldest(self) -> None:
        # file many DISTINCT suggestions to hit MAX_PENDING
        for i in range(personalize.MAX_PENDING + 2):
            entries = [{"id": j, "at": f"2026-09-2{i % 8}T10:00:00",
                        "label": f"profile: habit-{i}", "ops": []}
                       for j in range(1, 4)]
            mined = personalize.mine(entries)
            personalize.file_suggestions(mined, self.ledger, now=self.now)
        pending = [p for p in self.ledger.pending()
                   if p.get("kind") == "personalized_suggestion"]
        self.assertLessEqual(len(pending), personalize.MAX_PENDING)


class HistoryIntegrationTests(unittest.TestCase):
    def test_mine_over_real_history_entries(self) -> None:
        tmp, target = _mk()
        _hist(target, [("preset: battery-saver", 1),
                       ("preset: battery-saver", 5),
                       ("preset: battery-saver", 25)])
        mined = personalize.mine(history_entries(target))
        self.assertEqual(len(mined["recurring"]), 1)
        self.assertEqual(mined["recurring"][0]["count"], 3)


class CliTests(unittest.TestCase):
    def test_list_and_file_verbs(self) -> None:
        from assistant.core import cli as cortex_cli
        tmp, target = _mk()
        _hist(target, [("preset: battery-saver", 1),
                       ("preset: battery-saver", 5),
                       ("preset: battery-saver", 9)])
        ledger_path = tmp / "ledger.json"
        for action, expect in (("list", "recurring applies"),
                               ("file", "filed 1 suggestion")):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                rc = cortex_cli.main(["personalize", action,
                                      "--file", str(target),
                                      "--ledger", str(ledger_path)])
            self.assertEqual(rc, 0, err.getvalue())
            self.assertIn(expect, out.getvalue())
        # second file run dedupes
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = cortex_cli.main(["personalize", "file",
                                  "--file", str(target),
                                  "--ledger", str(ledger_path)])
        self.assertEqual(rc, 0)
        self.assertIn("filed 0", out.getvalue())


if __name__ == "__main__":
    unittest.main()

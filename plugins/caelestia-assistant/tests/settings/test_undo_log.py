"""A3 — the undo log: a first-class, PII-stripped negative-example store.

Contract under test (history.py + brain/calibrate.py + settings_bridge):

- EVERY undo (steps and by-id) appends one record per undone registry op:
  {"tool", "magnitude", "direction"} and NOTHING else — no timestamps, no
  labels, no values, no raw text. The PII-strip rule is enforced as a
  KEY SET assertion, not a spot check.
- The store lives inside the EXISTING history file (same atomic _save
  path — no new write surface: asserted by checking the written document
  and the sibling path inventory) and survives ring evictions (a 12-entry
  ring can never again hide an undo).
- The store is bounded (FIFO at UNDO_LOG_MAX) — the config directory
  cannot grow without limit.
- fold_undo_negatives turns records into explicit Beta-Binomial negative
  evidence at BOTH the kind level and per-tool posteriors; the settings
  bridge's default confidence reads the folded posterior (an undone
  change lowers the next proposal's confidence, the same lesson a
  rejected proposal teaches).
- schema_lint stays clean (the trust moat: no new imports, no new writes).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import calibrate
from assistant.capabilities.settings import history
from assistant.capabilities.settings.applier import apply
from assistant.adapters.caelestia.registry import tool_by_path


def _plan(path: str, new) -> dict:
    spec = tool_by_path(path)
    assert spec is not None
    return {"entries": [{"tool": spec.name, "path": path, "action": "set",
                         "raw": f"{path}={new}", "new": new}],
            "apply_blocked": False}


class UndoLogStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "shell.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _apply(self, path: str, new, label: str = "") -> dict:
        return apply(_plan(path, new), self.target, write=True, label=label)

    def test_undo_appends_pii_stripped_records(self) -> None:
        self._apply("bar.scale", 1.3, label="my labelled change")
        history.undo(self.target, 1)
        log = history.undo_log(self.target)
        self.assertEqual(len(log), 1)
        record = log[0]
        # PII-strip rule: exactly these keys, nothing else. ``quick`` is
        # a relative fact computed from the ring at undo time (the
        # undone apply was still the head), not a stored time.
        self.assertEqual(set(record),
                         {"tool", "magnitude", "direction", "quick"})
        self.assertIs(record["quick"], True)  # undo() reverts the head
        self.assertEqual(record["tool"], "setBarScale")
        # old=None (key absent): the registry default (1.0) is the honest
        # baseline — |1.3 - 1.0|, direction up, not |1.3 - 0|.
        self.assertAlmostEqual(record["magnitude"], 0.3)
        self.assertEqual(record["direction"], 1)

    def test_by_id_undo_also_appends(self) -> None:
        self._apply("appearance.blur", True)
        entry_id = history.entries(self.target)[0]["id"]
        history.undo_by_id(self.target, entry_id)
        log = history.undo_log(self.target)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["tool"], "setBlurEnabled")
        self.assertEqual(log[0]["direction"], 1)  # the apply turned blur ON
        self.assertIs(log[0]["quick"], True)  # it was still the head
        self.assertEqual(set(log[0]),
                         {"tool", "magnitude", "direction", "quick"})

    def test_by_id_undo_of_a_buried_entry_is_not_quick(self) -> None:
        # Two applies, then revert the OLDER one ('restore yesterday's
        # theme'): a deliberate selective restore, not an immediate
        # reaction to the newest change — quick must be False.
        self._apply("bar.scale", 1.2)
        self._apply("bar.scale", 1.4)
        old_id = history.entries(self.target)[-1]["id"]  # oldest survives
        self.assertEqual(len(history.entries(self.target)), 2)
        history.undo_by_id(self.target, old_id)
        record = history.undo_log(self.target)[-1]
        self.assertIs(record["quick"], False)

    def test_direction_and_magnitude_track_the_apply(self) -> None:
        self._apply("bar.scale", 1.0)          # first apply (from unset)
        self._apply("bar.scale", 0.6)         # second: 1.0 -> 0.6
        history.undo(self.target, 1)           # undo the 0.6 apply
        record = history.undo_log(self.target)[-1]
        self.assertEqual(record["direction"], -1)  # new < old
        self.assertAlmostEqual(record["magnitude"], 0.4)

    def test_log_survives_ring_eviction(self) -> None:
        # Fill the 12-entry ring well past capacity, then undo once: the
        # undo log must persist even as early applies are evicted.
        for i in range(history.MAX_ENTRIES + 4):
            self._apply("bar.scale", round(1.0 + (i % 5) * 0.05, 2))
        self.assertGreaterEqual(len(history.entries(self.target)),
                                history.MAX_ENTRIES)
        history.undo(self.target, 1)
        entries = history.entries(self.target)
        self.assertLessEqual(len(entries), history.MAX_ENTRIES)
        log = history.undo_log(self.target)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["tool"], "setBarScale")

    def test_log_is_bounded(self) -> None:
        # Drive far more undos than UNDO_LOG_MAX through a small ring.
        for round_no in range(history.UNDO_LOG_MAX + 10):
            self._apply("bar.scale", 1.0 + (round_no % 3) * 0.1)
            history.undo(self.target, 1)
        log = history.undo_log(self.target)
        self.assertEqual(len(log), history.UNDO_LOG_MAX)

    def test_no_new_write_surface(self) -> None:
        # The store lives INSIDE the existing history file: the target dir
        # contains exactly the pre-existing surfaces (target, the
        # applier's backup sibling, the history sibling) — nothing new.
        self._apply("bar.scale", 1.2)
        history.undo(self.target, 1)
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(names, ["shell.json", "shell.json.assistant-backup",
                                 "shell.json.assistant-history.json"])
        doc = json.loads((self.dir / "shell.json.assistant-history.json")
                         .read_text())
        self.assertIn("undo_log", doc)
        self.assertEqual(set(doc), {"next_id", "entries", "undo_log"})

    def test_multi_op_entry_yields_one_record_per_op(self) -> None:
        apply({"entries": [
            {"tool": "setBarScale", "path": "bar.scale", "action": "set",
             "raw": "a", "new": 1.2},
            {"tool": "setBlurEnabled", "path": "appearance.blur",
             "action": "set", "raw": "b", "new": True},
        ], "apply_blocked": False}, self.target, write=True)
        history.undo(self.target, 1)
        tools = [rec["tool"] for rec in history.undo_log(self.target)]
        self.assertEqual(tools, ["setBarScale", "setBlurEnabled"])

    def test_non_registry_paths_are_skipped(self) -> None:
        # The applier REFUSES non-registry paths outright (pinned by its
        # own tests); the pure _negative_records must still skip such an
        # op if one ever reaches the log through a hand-written entry.
        records = history._negative_records({
            "id": 1, "at": "2026-01-01T00:00:00Z", "label": "x",
            "ops": [
                {"path": "bar.scale", "old": 1.0, "new": 1.2},
                {"path": "no.such.path", "old": 1, "new": 5},
            ],
        })
        self.assertEqual([r["tool"] for r in records], ["setBarScale"])


class FoldUndoNegativesTests(unittest.TestCase):
    def test_empty_log_is_a_noop(self) -> None:
        stats = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "approved"}])
        before = json.dumps(stats, sort_keys=True)
        calibrate.fold_undo_negatives(stats, [])
        self.assertEqual(json.dumps(stats, sort_keys=True), before)

    def test_records_lower_the_posterior_mean(self) -> None:
        labeled = [{"kind": "settings", "status": "approved"} for _ in range(3)]
        stats = calibrate.acceptance_rate(labeled)
        before = stats["settings"]["mean"]
        calibrate.fold_undo_negatives(
            stats, [{"tool": "setBarScale", "magnitude": 0.3, "direction": 1}])
        self.assertLess(stats["settings"]["mean"], before)
        self.assertEqual(stats["settings"]["n"], 4)  # 3 labeled + 1 folded

    def test_per_tool_posteriors_are_created(self) -> None:
        stats = calibrate.acceptance_rate([])
        calibrate.fold_undo_negatives(stats, [
            {"tool": "setBarScale", "magnitude": 0.2, "direction": -1},
            {"tool": "setBarScale", "magnitude": 0.1, "direction": 1},
            {"tool": "setBlurEnabled", "magnitude": 1.0, "direction": 1},
        ])
        self.assertEqual(stats["tool:setBarScale"]["n"], 2)
        self.assertEqual(stats["tool:setBlurEnabled"]["n"], 1)
        # Beta(1,3) posterior mean = 0.25 for the two-record tool.
        self.assertEqual(stats["tool:setBarScale"]["mean"], 0.25)
        self.assertEqual(stats["tool:setBarScale"]["alpha"], 1.0)
        self.assertEqual(stats["tool:setBarScale"]["beta"], 3.0)

    def test_magnitude_does_not_scale_the_evidence(self) -> None:
        # An undo is ONE negative observation regardless of how big the
        # reverted change was — documented determinism, pinned here.
        small = calibrate.acceptance_rate([])
        big = calibrate.acceptance_rate([])
        calibrate.fold_undo_negatives(
            small, [{"tool": "setBarScale", "magnitude": 0.01, "direction": 1}])
        calibrate.fold_undo_negatives(
            big, [{"tool": "setBarScale", "magnitude": 5.0, "direction": 1}])
        self.assertEqual(small["tool:setBarScale"], big["tool:setBarScale"])


class QuickUndoWeightTests(unittest.TestCase):
    """Exponential-build 1.2: an approve-then-quick-undo (reverted while
    still the head of the bounded undo ring) is a STRONGER negative
    signal than a plain reject — a weighted update in the same
    Beta-Binomial model, with the weight pinned by arithmetic."""

    def test_quick_undo_shifts_posterior_further_than_a_plain_reject(self):
        # Approve-then-quick-undo: +1 alpha, then QUICK_UNDO_WEIGHT beta.
        approve_then_undo = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "approved"}])
        calibrate.fold_undo_negatives(approve_then_undo, [
            {"tool": "setBarScale", "magnitude": 0.3, "direction": 1,
             "quick": True}])
        # A plain reject: +1 beta, nothing else.
        plain_reject = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "rejected"}])
        self.assertLess(approve_then_undo["settings"]["mean"],
                        plain_reject["settings"]["mean"])
        # The exact hand-computed posteriors: 2/7 vs 1/3.
        self.assertEqual(approve_then_undo["settings"]["mean"],
                         round(2.0 / 7.0, 3))
        self.assertEqual(plain_reject["settings"]["mean"], 0.333)
        # The per-tool posterior carries the same weighted lesson.
        self.assertEqual(approve_then_undo["tool:setBarScale"]["mean"],
                         round(1.0 / (1.0 + 1.0 + calibrate.QUICK_UNDO_WEIGHT), 3))

    def test_quick_weight_is_the_smallest_integer_that_shifts_further(self):
        # w=3 exactly TIES the plain reject (2/(3+3) == 1/3) — the pin
        # demands strictly further, so the constant must be 4.
        self.assertEqual(calibrate.QUICK_UNDO_WEIGHT, 4.0)
        tie = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "approved"}])
        calibrate.fold_undo_negatives(tie, [
            {"tool": "setBarScale", "magnitude": 0.3, "direction": 1,
             "quick": True}], quick_weight=3.0)
        self.assertEqual(tie["settings"]["mean"], 0.333)

    def test_slow_undo_keeps_the_plain_weight(self):
        # A selective restore of an old entry (quick=False) stays at one
        # negative observation: it cancels the approval it reverts
        # (2/4 = 0.5) but does not overshoot a plain reject (0.333) —
        # the previous conservative behavior, kept.
        approve_then_slow_undo = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "approved"}])
        calibrate.fold_undo_negatives(approve_then_slow_undo, [
            {"tool": "setBarScale", "magnitude": 0.3, "direction": 1,
             "quick": False}])
        plain_reject = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "rejected"}])
        self.assertEqual(approve_then_slow_undo["settings"]["mean"], 0.5)
        self.assertGreater(approve_then_slow_undo["settings"]["mean"],
                           plain_reject["settings"]["mean"])

    def test_records_without_the_field_stay_plain(self):
        # Older logs (pre-quick) fold exactly as before: one negative
        # observation, never a guess about whether they were quick.
        stats = calibrate.acceptance_rate(
            [{"kind": "settings", "status": "approved"}])
        calibrate.fold_undo_negatives(stats, [
            {"tool": "setBarScale", "magnitude": 0.3, "direction": 1}])
        self.assertEqual(stats["settings"]["mean"], 0.5)


class BridgeWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "shell.json"
        self.ledger_path = self.dir / "ledger.json"
        # A ledger with three approved settings proposals: acceptance 1.0.
        from assistant.capabilities.brain.ledger import Ledger
        ledger = Ledger(str(self.ledger_path))
        for i in range(3):
            pid = ledger.propose("settings", "bar.scale", {}, "t", 0.7)
            ledger.decide(pid, True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_undo_lowers_the_next_proposals_confidence(self) -> None:
        from assistant.capabilities.brain import settings_bridge
        from assistant.capabilities.brain.ledger import Ledger

        ledger = Ledger(str(self.ledger_path))
        # No undo log yet: calibrated confidence is the approval rate, 1.0.
        clean = settings_bridge.propose(ledger, self.target,
                                         calls=[{"tool": "setBarScale",
                                                 "action": "set",
                                                 "value": 1.1, "raw": "x"}])
        ledger2 = Ledger(str(self.ledger_path))
        # Now the user applies and undoes a change.
        apply(_plan("bar.scale", 1.2), self.target, write=True)
        history.undo(self.target, 1)
        after = settings_bridge.propose(ledger2, self.target,
                                        calls=[{"tool": "setBarScale",
                                                "action": "set",
                                                "value": 1.1, "raw": "x"}])
        clean_conf = self._confidence_of(clean["proposal_id"])
        after_conf = self._confidence_of(after["proposal_id"])
        self.assertIsNotNone(clean_conf)
        self.assertIsNotNone(after_conf)
        self.assertLess(after_conf, clean_conf)

    def _confidence_of(self, proposal_id):
        if proposal_id is None:
            return None
        from assistant.capabilities.brain.ledger import Ledger
        pending = {i["id"]: i for i in Ledger(str(self.ledger_path)).pending()}
        if proposal_id in pending:
            return pending[proposal_id].get("confidence")
        for item in Ledger(str(self.ledger_path)).labeled("settings"):
            if item.get("id") == proposal_id:
                return item.get("confidence")
        return None


if __name__ == "__main__":
    unittest.main()

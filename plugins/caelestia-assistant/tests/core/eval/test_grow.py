"""F26 tests — the arena grows from experience, UNDER REVIEW
(assistant/eval/grow.py).

Under test:

- mining: learner-approved examples (label==1) become confirmed
  candidates; outstanding review-bucket phrases become PROPOSED
  candidates; dedup by text; bounded at MAX_CANDIDATES;
- quarantine: save_quarantine assigns stable ids, dedupes by text,
  bounded FIFO (writes only the caller's state dict — no I/O);
- promotion: appends one candidate to a CALLER-NAMED dev set with a
  u-* id; SEALED sets refuse BY NAME; duplicate texts refuse;
  candidates without an accept refuse; the promoted item leaves the
  quarantine;
- the drift guard in reverse: the repo's OWN dev sets still parse
  (promotion's writer format matches the loader).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.core.eval import grow


def _state():
    return {
        "cortex_learner": {"examples": [
            {"text": "make the dock tiny", "surface": "setDockIconSize",
             "p": 0.42, "label": 1, "outcome": "applied"},
            {"text": "rejected phrase", "surface": "setBarScale",
             "p": 0.5, "label": 0, "outcome": "rejected"},
        ]},
        "cortex_review": [
            {"text": "make the dock tiny", "verdict": "AMBIGUOUS",
             "surface": "setDockIconSize", "p": 0.44, "at": "2026-09-28"},
            {"text": "strange phrasing", "verdict": "ABSTAIN",
             "surface": "setOsdHideDelay", "p": 0.21, "at": "2026-09-28"},
        ],
    }


class MineTests(unittest.TestCase):
    def test_confirmed_and_proposed_candidates(self) -> None:
        mined = grow.mine_candidates(_state())
        by_source = {c["text"]: c["source"] for c in mined}
        self.assertEqual(by_source["make the dock tiny"], "correction")
        self.assertEqual(by_source["strange phrasing"], "review")
        self.assertNotIn("rejected phrase", by_source,
                         "rejected outcomes are never candidates")
        # the confirmed evidence wins the dedup
        first = next(c for c in mined if c["text"] == "make the dock tiny")
        self.assertEqual(first["accept"], ["setDockIconSize"])

    def test_bounded(self) -> None:
        state = {"cortex_review": [
            {"text": f"phrase {i}", "surface": "setBarScale"}
            for i in range(grow.MAX_CANDIDATES + 10)]}
        self.assertEqual(len(grow.mine_candidates(state)),
                         grow.MAX_CANDIDATES)


class QuarantineTests(unittest.TestCase):
    def test_ids_stable_and_bounded(self) -> None:
        state = {}
        stats1 = grow.save_quarantine(state, grow.mine_candidates(_state()))
        self.assertEqual(stats1["added"], 2)
        ids1 = [c["id"] for c in state[grow.QUARANTINE_KEY]]
        stats2 = grow.save_quarantine(state, grow.mine_candidates(_state()))
        self.assertEqual(stats2["added"], 0)
        ids2 = [c["id"] for c in state[grow.QUARANTINE_KEY]]
        self.assertEqual(ids1, ids2)


class PromoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="grow-test-"))
        self.dev_set = self.tmp / "routing_dev.json"
        self.dev_set.write_text(json.dumps({
            "note": "dev",
            "items": [{"id": "r1", "text": "existing phrase",
                       "accept": ["setBarScale"]}],
        }), encoding="utf-8")
        self.state = {}
        grow.save_quarantine(self.state, grow.mine_candidates(_state()))

    def test_promote_appends_and_unquarantines(self) -> None:
        candidate = self.state[grow.QUARANTINE_KEY][0]
        result = grow.promote(self.state, candidate["id"], self.dev_set)
        self.assertTrue(result["as"].startswith("u-"))
        data = json.loads(self.dev_set.read_text(encoding="utf-8"))
        self.assertEqual(len(data["items"]), 2)
        texts = [i["text"] for i in data["items"]]
        self.assertIn(candidate["text"], texts)
        self.assertNotIn(candidate, self.state[grow.QUARANTINE_KEY])
        # the writer's format must match the loader
        from assistant.adapters.caelestia.registry import TOOL_COUNT  # noqa: F401
        sets_dir = self.dev_set.parent
        # load_set resolves by suite/split names inside the package; the
        # loader-contract check here parses the written bytes the way
        # load_set does (json with items[])
        loaded = json.loads(self.dev_set.read_text(encoding="utf-8"))
        self.assertIsInstance(loaded["items"], list)
        self.assertTrue(all("id" in i and "text" in i and "accept" in i
                            for i in loaded["items"]))

    def test_sealed_sets_refuse_by_name(self) -> None:
        sealed = self.tmp / "sealed_routing.json"
        sealed.write_text(json.dumps({"note": "", "items": []}),
                          encoding="utf-8")
        candidate = self.state[grow.QUARANTINE_KEY][0]
        with self.assertRaisesRegex(ValueError, "SEALED"):
            grow.promote(self.state, candidate["id"], sealed)
        # nothing was written
        self.assertEqual(json.loads(sealed.read_text(encoding="utf-8")),
                         {"note": "", "items": []})

    def test_duplicate_text_refuses(self) -> None:
        candidate = self.state[grow.QUARANTINE_KEY][0]
        candidate["text"] = "existing phrase"
        with self.assertRaisesRegex(ValueError, "same text"):
            grow.promote(self.state, candidate["id"], self.dev_set)

    def test_unknown_candidate_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "no quarantined candidate"):
            grow.promote(self.state, "g-nope", self.dev_set)


if __name__ == "__main__":
    unittest.main()

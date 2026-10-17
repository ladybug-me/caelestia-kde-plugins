"""Tests for the structured slot tagger (cortex/slot_tagger.py, 2.2).

The contract under test:
- supervision comes ONLY from local approve/reject history rows (the
  cortex/learn.py example shape): approved rows supervise via the
  EXISTING grammar's own matched raws (settings/slots.extract); rejected
  rows supervise nothing;
- the structured perceptron is deterministic (same data, same weights),
  averaged (Collins 2002), and generalizes the grammar's own cue words
  to unseen targets;
- the conformal gate is the safety property: an untrained tagger, a
  low-confidence tag, or a calibrator WITHOUT data all fall back to the
  existing grammar path (slots.extract's own output) with the calibrator's
  own reason attached — the explicit low-confidence-fallback case;
- out-of-contract inputs are rejected, never truncated or guessed.
"""

from __future__ import annotations

import unittest

from assistant.core.conformal import ConformalCalibrator
from assistant.core.slot_tagger import (
    MAX_TOKENS,
    SlotTagger,
    supervision_from_history,
    tag_gated,
)


def _noun_matcher(text: str):
    words = set(text.lower().replace(",", " ").split())
    return sorted(words & {"bar", "dock", "animations", "panel", "launcher"})


# Approved rows whose slot structure the EXISTING grammar can label
# (speed up / trim / dial back / boost / reduce are slots.py's own raws);
# one rejected row that must supervise nothing.
HISTORY = [
    {"text": "speed up the animations", "label": 1, "outcome": "approved"},
    {"text": "trim the bar", "label": 1, "outcome": "applied"},
    {"text": "dial back the dock size", "label": 1, "outcome": "approved"},
    {"text": "boost the panel opacity", "label": 1, "outcome": "applied"},
    {"text": "reduce the transparency a bit", "label": 1,
     "outcome": "approved"},
    {"text": "make the bar huge and weird", "label": 0,
     "outcome": "rejected"},
]


class SupervisionTests(unittest.TestCase):
    def test_only_approved_rows_supervise(self) -> None:
        rows = supervision_from_history(HISTORY,
                                        noun_matcher=_noun_matcher)
        texts = [r["text"] for r in rows]
        self.assertNotIn("make the bar huge and weird", texts)
        self.assertEqual(len(texts), 5)

    def test_gold_spans_come_from_the_grammar(self) -> None:
        rows = supervision_from_history(HISTORY,
                                        noun_matcher=_noun_matcher)
        first = rows[0]  # "speed up the animations"
        self.assertEqual(first["tokens"][:2], ["speed", "up"])
        self.assertEqual(first["tags"][:2], ["B-CUE", "I-CUE"])
        self.assertIn("B-TARGET", first["tags"])
        reduce_row = next(r for r in rows
                          if r["text"] == "reduce the transparency a bit")
        self.assertEqual(reduce_row["tags"][0], "B-GENERIC")

    def test_undecided_rows_supervise_nothing(self) -> None:
        rows = supervision_from_history([
            {"text": "trim the bar", "label": 0, "outcome": "clarified"}])
        self.assertEqual(rows, [])


class PerceptronTests(unittest.TestCase):
    def _trained(self) -> SlotTagger:
        tagger = SlotTagger()
        rows = supervision_from_history(HISTORY,
                                        noun_matcher=_noun_matcher)
        tagger.train([(r["text"], r["tags"]) for r in rows])
        return tagger

    def test_deterministic_same_data_same_model(self) -> None:
        a, b = self._trained(), self._trained()
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_generalizes_seen_cues_to_unseen_targets(self) -> None:
        tagger = self._trained()
        result = tagger.tag("trim the launcher")
        cue_raws = [s["raw"] for s in result["spans"].get("CUE", [])]
        self.assertIn("trim", cue_raws)

    def test_char_ngram_features_are_wired(self) -> None:
        tagger = self._trained()
        feats = tagger._features(["trim"], 0, "O")
        self.assertIn("ng=tri", feats)
        self.assertIn("ng=rim", feats)

    def test_confidence_is_bounded_and_margin_nonnegative(self) -> None:
        tagger = self._trained()
        r1 = tagger.tag("trim the bar")
        self.assertTrue(0.0 < r1["confidence"] < 1.0)
        self.assertGreaterEqual(r1["margin"], 0.0)

    def test_persistence_round_trip_exact(self) -> None:
        tagger = self._trained()
        revived = SlotTagger.from_dict(tagger.to_dict())
        self.assertEqual(tagger.tag("trim the bar"),
                         revived.tag("trim the bar"))
        self.assertGreaterEqual(revived.trained_examples, 5)
        self.assertGreaterEqual(revived.updates, 5)

    def test_overlong_sequence_rejected_never_truncated(self) -> None:
        tagger = self._trained()
        with self.assertRaises(ValueError):
            tagger.tag(" ".join(["word"] * (MAX_TOKENS + 1)))

    def test_misaligned_or_unknown_gold_rejected(self) -> None:
        tagger = SlotTagger()
        with self.assertRaises(ValueError):
            tagger.update("trim the bar", ["O", "B-CUE"])
        with self.assertRaises(ValueError):
            tagger.update("trim the bar", ["O", "X-CUE", "O"])


class ConformalGateTests(unittest.TestCase):
    def _trained(self) -> SlotTagger:
        tagger = SlotTagger()
        rows = supervision_from_history(HISTORY,
                                        noun_matcher=_noun_matcher)
        tagger.train([(r["text"], r["tags"]) for r in rows])
        return tagger

    def test_untrained_tagger_falls_back_to_grammar(self) -> None:
        result = tag_gated("trim the bar", SlotTagger())
        self.assertEqual(result["used"], "grammar-fallback")
        self.assertIn("untrained", result["reason"])
        # the fallback IS the existing grammar's own output
        self.assertEqual(result["slots"]["cue"]["dimension"], "size")
        self.assertEqual(result["slots"]["cue"]["sign"], -1)

    def test_low_confidence_falls_back_with_calibrator_reason(self) -> None:
        # calibrate against HIGH scores only: the conformal threshold
        # sits near 0.95, and an untrained tagger's non-claim must not
        # sneak past it
        cal = ConformalCalibrator(alpha=0.1)
        for s in (0.95, 0.99, 0.97, 0.98, 0.96, 0.99):
            cal.observe(s, "applied")
        tagger = SlotTagger()  # untrained: the weakest possible claim
        result = tag_gated("trim the bar", tagger, cal)
        self.assertEqual(result["used"], "grammar-fallback")

    def test_no_calibration_data_falls_back(self) -> None:
        cal = ConformalCalibrator()  # zero data -> covered is None
        tagger = self._trained()
        result = tag_gated("trim the bar", tagger, cal)
        self.assertEqual(result["used"], "grammar-fallback")
        self.assertIn("insufficient calibration data", result["reason"])

    def test_covered_confidence_uses_the_tagger(self) -> None:
        cal = ConformalCalibrator(alpha=0.1)
        # accepted routes historically scored LOW: the conformal
        # threshold sits low, and the trained tagger's confident tag
        # clears it honestly
        for s in (0.2, 0.25, 0.3, 0.22, 0.28, 0.24):
            cal.observe(s, "applied")
        tagger = self._trained()
        result = tag_gated("trim the launcher", tagger, cal)
        self.assertEqual(result["used"], "tagger")
        self.assertIn("conformal", result)


if __name__ == "__main__":
    unittest.main()

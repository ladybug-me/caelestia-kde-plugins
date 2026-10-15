"""Metamorphic ratchet (Group A1): floors pinned at the first measured
levels, deliberately slightly below them. A regression under the floor
fails the suite; improvements are expected to RAISE the floors in the
same commit that measures them — never silently.

The floors are on CONSISTENCY (the metamorphic invariant held), per
transform, over the auto-generated dev variants. The suite is derived
(no committed set file); the same deterministic generator runs here, so
a future registry re-pin re-measures automatically.
"""

from __future__ import annotations

import unittest

from assistant.core.eval.engine import load_set
from assistant.core.eval.metamorphic import TRANSFORMS, generate_variants, \
    run_metamorphic

# (floor, measured-at-introduction) — raise only WITH the improvement.
FLOORS = {
    "paraphrase": 0.95,     # measured 1.0000
    "synonym": 0.85,        # measured 0.9231
    "typo": 0.60,           # measured 0.6250 (33 breaks: correction misses
    #                          + sibling confusions the suite now exposes)
    "polarity_flip": 0.85,  # measured 0.9091
    # unit_change: no unit-bearing dev item existed at introduction; the
    # floor arms the moment the A5 items give it coverage (asserted below).
}


class MetamorphicRatchet(unittest.TestCase):
    def test_suite_runs_and_reports_every_transform(self) -> None:
        report = run_metamorphic("dev")
        self.assertEqual(report["suite"], "metamorphic")
        self.assertGreater(report["n_variants"], 0)
        for t in TRANSFORMS:
            self.assertIn(f"{t}_consistency", report["metrics"])

    def test_ratchet_floors(self) -> None:
        report = run_metamorphic("dev")
        for t, floor in FLOORS.items():
            point = report["metrics"][f"{t}_consistency"][0]
            self.assertGreaterEqual(
                point, floor,
                f"{t} consistency {point} fell under the ratchet floor "
                f"{floor}")

    def test_polarity_flip_pairs_are_real_direction_words(self) -> None:
        # every flip pair must be two REAL shipped direction words with
        # opposite signs — the flip is never invented vocabulary
        from assistant.core.lexicon import DIRECTION_WORDS
        from assistant.core.eval.metamorphic import _FLIP
        for a, b in _FLIP.items():
            self.assertEqual(DIRECTION_WORDS.get(a),
                             -DIRECTION_WORDS.get(b),
                             f"flip pair {a}/{b} not opposite-signed")

    def test_unit_relation_has_coverage_or_is_armed(self) -> None:
        # A5 added unit-bearing items; from then on the unit relation
        # must have observations AND hold at or over its floor.
        report = run_metamorphic("dev")
        items = load_set("routing", "dev")["items"]
        variants = [v for it in items
                    for v in generate_variants(it["text"])["unit_change"]]
        if variants:
            point = report["metrics"]["unit_change_consistency"][0]
            self.assertGreaterEqual(point, 0.85)


if __name__ == "__main__":
    unittest.main()

"""brain.align — Needleman-Wunsch divergence mining for tidy (§6.5).

Pins: the alignment DP on hand-checkable sequences; divergence extraction
(same source, different destination); the minimum-consistency floor (the
same 3-support floor the workspace profiles and ontology gaps use); the
ledger round-trip (proposals only, never auto-applied); determinism; and
the structural purity guarantee — this module touches no filesystem,
which is how it inherits tidy's never-delete/journaled/rollback-able
discipline: by being incapable of anything else.

All fixtures are SYNTHETIC move records (§9.3: no real user filesystem
data is processed, even in testing — the dry run came first).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import align as align_mod
from assistant.capabilities.brain.align import (
    align_moves,
    divergence_report,
    move_similarity,
    propose_corrections,
)
from assistant.capabilities.brain.ledger import Ledger


def _move(src, dst, cat):
    return {"from": src, "to": dst, "category": cat}


class MoveSimilarityTests(unittest.TestCase):

    def test_identical_source(self):
        self.assertEqual(
            move_similarity(_move("a/b.png", "x", "images"),
                           _move("a/b.png", "y", "images")), 1.0)

    def test_same_basename_different_path(self):
        self.assertGreater(
            move_similarity(_move("a/b.png", "x", "images"),
                            _move("c/b.png", "y", "images")), 0.0)
        self.assertLess(
            move_similarity(_move("a/b.png", "x", "images"),
                            _move("c/b.png", "y", "images")), 1.0)

    def test_unrelated(self):
        self.assertEqual(
            move_similarity(_move("a/b.png", "x", "images"),
                           _move("c/d.pdf", "y", "documents")), 0.0)


class AlignmentTests(unittest.TestCase):

    def test_identical_sequences_align_diagonally(self):
        proposed = [_move(f"src/{i}.png", f"img/{i}.png", "images")
                    for i in range(4)]
        actual = [_move(f"src/{i}.png", f"img/{i}.png", "images")
                  for i in range(4)]
        result = align_moves(proposed, actual)
        self.assertEqual(result["aligned"], 4)
        self.assertEqual(result["gaps"], 0)
        self.assertEqual(result["trace"],
                         [(i, i) for i in range(4)])

    def test_hand_checkable_indel(self):
        # proposed has one extra file; the alignment should gap it out
        proposed = [_move("a", "x", "c"), _move("b", "y", "c"),
                   _move("extra", "z", "c")]
        actual = [_move("a", "x", "c"), _move("b", "y", "c")]
        result = align_moves(proposed, actual)
        self.assertEqual(result["aligned"], 2)
        self.assertEqual(result["gaps"], 1)
        # the extra proposed move is gapped, the real ones align
        self.assertIn((2, None), result["trace"])
        self.assertIn((0, 0), result["trace"])
        self.assertIn((1, 1), result["trace"])

    def test_alignment_is_order_preserving(self):
        # NW is a GLOBAL, order-preserving alignment: a swapped pair
        # cannot cross-match both elements (that would need two local
        # alignments). The honest behavior: the best single pairing wins
        # and the other element gaps out — monotone indices, always.
        proposed = [_move("a", "x", "c"), _move("b", "y", "c")]
        actual = [_move("b", "y", "c"), _move("a", "x", "c")]
        result = align_moves(proposed, actual)
        pairs = [(i, j) for i, j in result["trace"]
                 if i is not None and j is not None]
        self.assertTrue(pairs)
        indexes = [i for i, _j in pairs]
        self.assertEqual(indexes, sorted(indexes),
                          "aligned pairs must be monotone in the proposed side")
        # and every aligned pair is a real content match, never a crossing one
        for i, j in pairs:
            self.assertEqual(proposed[i]["from"], actual[j]["from"])

    def test_empty(self):
        self.assertEqual(align_moves([], [])["aligned"], 0)
        self.assertEqual(align_moves([_move("a", "b", "c")], [])["gaps"], 1)


class DivergenceTests(unittest.TestCase):

    def test_same_source_different_destination_diverges(self):
        proposed = [_move(f"src/shot{i}.png", f"Pictures/images/shot{i}.png",
                          "images") for i in range(4)]
        actual = [_move(f"src/shot{i}.png", f"shots/shot{i}.png",
                        "images") for i in range(4)]
        report = divergence_report(proposed, actual)
        self.assertEqual(report["aligned"], 4)
        self.assertEqual(report["aligned_same_destination"], 0)
        self.assertEqual(report["divergent"], 4)
        self.assertEqual(len(report["candidates"]), 1)
        cand = report["candidates"][0]
        self.assertEqual(cand["category"], "images")
        self.assertEqual(cand["proposed_dir"], "Pictures/images")
        self.assertEqual(cand["actual_dir"], "shots")
        self.assertEqual(cand["support"], 4)

    def test_agreeing_moves_produce_no_candidates(self):
        proposed = [_move(f"src/{i}.png", f"img/{i}.png", "images")
                    for i in range(6)]
        actual = list(proposed)
        report = divergence_report(proposed, actual)
        self.assertEqual(report["divergent"], 0)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["aligned_same_destination"], 6)

    def test_below_floor_produces_no_candidates(self):
        # two divergent moves: a choice, not a rule (the floor is 3)
        proposed = [_move("src/a.png", "Pictures/images/a.png", "images"),
                    _move("src/b.png", "Pictures/images/b.png", "images")]
        actual = [_move("src/a.png", "shots/a.png", "images"),
                 _move("src/b.png", "shots/b.png", "images")]
        report = divergence_report(proposed, actual)
        self.assertEqual(report["divergent"], 2)
        self.assertEqual(report["candidates"], [])
        self.assertGreaterEqual(report["rejected_patterns"], 1)

    def test_at_floor_produces_the_candidate(self):
        proposed = [_move(f"src/{c}.png", f"Pictures/images/{c}.png", "images")
                    for c in "abc"]
        actual = [_move(f"src/{c}.png", f"shots/{c}.png", "images")
                  for c in "abc"]
        report = divergence_report(proposed, actual)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertEqual(report["candidates"][0]["support"], 3)

    def test_scattered_divergences_do_not_cluster(self):
        # three divergences, three DIFFERENT actual destinations: no
        # consistent pattern -> no candidate (the cluster-purity analog)
        proposed = [_move(f"src/{c}.png", f"Pictures/images/{c}.png", "images")
                    for c in "abc"]
        actual = [_move("src/a.png", "shots/a.png", "images"),
                  _move("src/b.png", "screens/b.png", "images"),
                  _move("src/c.png", "wall/c.png", "images")]
        report = divergence_report(proposed, actual)
        self.assertEqual(report["divergent"], 3)
        self.assertEqual(report["candidates"], [])


class ProposalTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = Ledger(Path(self._tmp.name) / "ledger.json")

    def _divergent_pair(self, n=3):
        proposed = [_move(f"src/shot{i}.png", f"Pictures/images/shot{i}.png",
                          "images") for i in range(n)]
        actual = [_move(f"src/shot{i}.png", f"shots/shot{i}.png",
                        "images") for i in range(n)]
        return proposed, actual

    def test_qualifying_pattern_becomes_pending_tidy_rule_proposal(self):
        proposed, actual = self._divergent_pair()
        res = propose_corrections(proposed, actual, self.ledger)
        self.assertEqual(len(res["proposals"]), 1)
        pending = self.ledger.pending()
        self.assertEqual(pending[0]["kind"], "tidy_rule")
        self.assertEqual(pending[0]["status"], "pending")
        self.assertIn("moved 3", pending[0]["reason"])

    def test_nothing_is_ever_auto_applied(self):
        proposed, actual = self._divergent_pair()
        propose_corrections(proposed, actual, self.ledger)
        # approving records a DECISION about future proposal targets; the
        # diff it carries is the observed correction, nothing executable
        pid = self.ledger.pending()[0]["id"]
        self.ledger.decide(pid, True)
        approved = [p for p in self.ledger.items if p["id"] == pid][0]
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["diff"]["correction"]["actual_dir"], "shots")

    def test_below_floor_never_proposes(self):
        proposed, actual = self._divergent_pair(n=2)
        res = propose_corrections(proposed, actual, self.ledger)
        self.assertEqual(res["proposals"], [])
        self.assertEqual(self.ledger.pending(), [])

    def test_duplicate_pending_target_not_stacked(self):
        proposed, actual = self._divergent_pair()
        first = propose_corrections(proposed, actual, self.ledger)
        second = propose_corrections(proposed, actual, self.ledger)
        self.assertEqual(len(first["proposals"]), 1)
        self.assertEqual(second["proposals"], [])
        self.assertEqual(len(self.ledger.pending()), 1)

    def test_deterministic(self):
        proposed, actual = self._divergent_pair(n=5)
        a = divergence_report(proposed, actual)
        b = divergence_report(proposed, actual)
        self.assertEqual(a, b)


class PurityTests(unittest.TestCase):
    """§6.5's safety posture, made structural: no filesystem surface."""

    def test_module_touches_no_files(self):
        source = Path(align_mod.__file__).read_text(encoding="utf-8")
        for banned in ("import os", "import pathlib", "from pathlib",
                       "open(", ".rename(", ".remove(", "os.remove",
                       "shutil", "mkdir", "write_text", "read_text"):
            self.assertNotIn(banned, source,
                             f"align.py must stay filesystem-pure; found {banned!r}")
        # and behaviorally: aligning moves produces data, nothing else
        report = divergence_report([_move("a", "b", "c")],
                                   [_move("a", "d", "c")])
        self.assertIsInstance(report, dict)

    def test_tidy_discipline_unchanged(self):
        # tidy.py's own contract pins: never deletes, journals, rolls back.
        # This module adds NO new file operation to that contract — the
        # source-level check above is the guarantee; this behavioral one
        # pins that alignment output feeds only ledger proposals.
        ledger = Ledger(Path(tempfile.mkdtemp()) / "ledger.json")
        propose_corrections([_move("a", "b", "c")], [_move("a", "d", "c")],
                            ledger, min_support=1)
        for item in ledger.items:
            self.assertIn(item["kind"], ("tidy_rule",))
            self.assertIn("status", item)


if __name__ == "__main__":
    unittest.main()

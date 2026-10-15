"""Tests for exponential-build phase 1.4: the discriminative pairwise
re-ranker, conformal-gated over the router's own ranking.

The contract under test:

- a pairwise max-margin perceptron (Rosenblatt 1958; Herbrich et al.
  2000 pairwise formulation; Collins 2002 averaging) over the router's
  OWN per-candidate signal components — approved rows supervise, the
  rejected ones supervise nothing, and a surface the router never
  scored fabricates no constraint;
- the gated entry point degrades exactly like tag_gated: untrained
  model, single candidate, below-threshold confidence, or no
  calibration data all return the router's OWN ranking with the reason
  attached;
- deterministic end to end; persistence round-trips exactly; the real
  router supplies the features (no second feature implementation).
"""
import unittest

from assistant.core.conformal import ConformalCalibrator
from assistant.core.reranker import (FEATURES, Reranker,
                                       rerank_gated,
                                       supervision_from_history)


def _feats(lex=0.0, sem=0.0, fuzz=0.0, noun=0.0, cue=0.0, coverage=0.0):
    return {"lex": lex, "sem": sem, "fuzz": fuzz, "noun": noun,
            "cue": cue, "coverage": coverage}


def _calibrator_with(scores):
    cal = ConformalCalibrator()
    for s in scores:
        cal.observe(s, "applied")
    return cal


class PairwiseUpdateTests(unittest.TestCase):
    def test_violated_pair_moves_weights_toward_the_approved_surface(self):
        r = Reranker()
        approved = _feats(lex=1.0)
        competitor = _feats(sem=1.0)
        changed = r.update(approved, [competitor])
        self.assertTrue(changed)
        self.assertEqual(r.weights["lex"], 1.0)
        self.assertEqual(r.weights["sem"], -1.0)
        # the approved surface now outscores the competitor
        self.assertGreater(r.score(approved), r.score(competitor))

    def test_satisfied_pair_is_a_noop(self):
        r = Reranker()
        r.update(_feats(lex=1.0), [_feats(sem=0.5)])
        before = dict(r.weights)
        self.assertFalse(r.update(_feats(lex=1.0), [_feats(sem=0.5)]))
        self.assertEqual(r.weights, before)
        # ...but the example still counted (an effective sample size
        # you can see)
        self.assertEqual(r.trained_examples, 2)

    def test_no_competitors_supervises_nothing(self):
        r = Reranker()
        self.assertFalse(r.update(_feats(lex=1.0), []))
        self.assertEqual(r.trained_examples, 0)

    def test_separable_preferences_are_learned(self):
        # The user's history prefers the NOUN signal over the LEXICAL
        # one: approved surfaces carry noun=1, competitors lex=1.
        r = Reranker()
        rows = [{"approved": _feats(noun=1.0),
                 "competitors": [_feats(lex=1.0), _feats(sem=0.8)]}
                for _ in range(3)]
        r.train(rows)
        self.assertGreater(r.score(_feats(noun=1.0)),
                           r.score(_feats(lex=1.0)))
        self.assertGreater(r.score(_feats(noun=1.0)),
                           r.score(_feats(sem=0.8)))

    def test_deterministic_same_rows_same_model(self):
        rows = [{"approved": _feats(noun=1.0, cue=0.3),
                 "competitors": [_feats(lex=1.0), _feats(fuzz=0.4)]}]
        a, b = Reranker(), Reranker()
        a.train(rows * 2)
        b.train(rows * 2)
        self.assertEqual(a.to_dict(), b.to_dict())


class SupervisionTests(unittest.TestCase):
    def test_only_approved_rows_supervise(self):
        rows = [{"text": "make the bar scale 1.3", "surface": "setBarScale",
                 "label": 1},
                {"text": "bar a smidge shorter please",
                 "surface": "setBarScale", "label": 0}]
        seen = []

        def fake_route(text):
            seen.append(text)
            return {"setBarScale": _feats(lex=1.0),
                    "setBarPosition": _feats(sem=1.0)}

        out = supervision_from_history(rows, route_fn=fake_route)
        self.assertEqual(seen, ["make the bar scale 1.3"])  # rejected: skipped
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["surface"], "setBarScale")
        competitors = out[0]["competitors"]
        self.assertEqual(len(competitors), 1)

    def test_unscored_surface_fabricates_no_constraint(self):
        rows = [{"text": "make the bar scale 1.3", "surface": "setBarScale",
                 "label": 1}]
        out = supervision_from_history(
            rows, route_fn=lambda t: {"setOther": _feats(lex=1.0)})
        self.assertEqual(out, [])

    def test_real_router_supplies_the_features(self):
        # The default route_fn is the REAL router: an approved row for a
        # phrase the router ranks confidently produces a training row
        # whose features carry the router's own signal components.
        rows = [{"text": "make the bar scale 1.3", "surface": "setBarScale",
                 "outcome": "applied"}]
        out = supervision_from_history(rows)
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["surface"], "setBarScale")
        for key in FEATURES:
            self.assertIn(key, row["approved"])


class GatedEntryTests(unittest.TestCase):
    FEATURES_MAP = {"setBarScale": _feats(lex=1.0, noun=1.0),
                    "setBarPosition": _feats(sem=1.0)}
    ORDER = ["setBarPosition", "setBarScale"]  # the router preferred the other

    def test_untrained_reranker_falls_back(self):
        result = rerank_gated(self.FEATURES_MAP, self.ORDER, Reranker())
        self.assertEqual(result["used"], "router-fallback")
        self.assertEqual(result["reason"], "re-ranker untrained")
        self.assertEqual(result["order"], self.ORDER)

    def test_single_candidate_falls_back(self):
        r = Reranker()
        r.update(_feats(lex=1.0), [_feats(sem=1.0)])
        result = rerank_gated({"setBarScale": _feats(lex=1.0)},
                              ["setBarScale"], r)
        self.assertEqual(result["used"], "router-fallback")
        self.assertIn("nothing to re-rank", result["reason"])

    def test_covered_confidence_uses_the_reranker(self):
        r = Reranker()
        for _ in range(3):
            r.update(_feats(lex=1.0, noun=1.0), [_feats(sem=1.0)])
        # margin 1.0 -> confidence ~0.881; this calibrator's quantile
        # (0.7) is below it, so the re-ranked answer is covered
        cal = _calibrator_with([0.3, 0.4, 0.5, 0.6, 0.7])
        result = rerank_gated(self.FEATURES_MAP, self.ORDER, r, cal)
        self.assertEqual(result["used"], "reranker")
        self.assertEqual(result["order"][0], "setBarScale")
        self.assertEqual(result["conformal"],
                         "score clears the conformal threshold")

    def test_below_threshold_falls_back_with_calibrator_reason(self):
        r = Reranker()
        for _ in range(3):
            r.update(_feats(lex=1.0, noun=1.0), [_feats(sem=1.0)])
        # a calibrator fed only HIGH scores: the tie-ish margin below
        # cannot clear it
        cal = _calibrator_with([0.95, 0.96, 0.97, 0.98, 0.99])
        near_tie = {"setBarScale": _feats(lex=0.6, noun=1.0),
                    "setBarPosition": _feats(lex=0.6, noun=0.95)}
        result = rerank_gated(near_tie, self.ORDER, r, cal)
        self.assertEqual(result["used"], "router-fallback")
        self.assertEqual(result["order"], self.ORDER)  # unchanged
        self.assertIn("below the conformal threshold", result["reason"])

    def test_no_calibration_data_falls_back(self):
        r = Reranker()
        for _ in range(3):
            r.update(_feats(lex=1.0), [_feats(sem=1.0)])
        result = rerank_gated(self.FEATURES_MAP, self.ORDER, r,
                              ConformalCalibrator())
        self.assertEqual(result["used"], "router-fallback")
        self.assertIn("insufficient calibration data", result["reason"])

    def test_deterministic_and_non_mutating(self):
        r = Reranker()
        for _ in range(3):
            r.update(_feats(lex=1.0), [_feats(sem=1.0)])
        cal = _calibrator_with([0.95, 0.9, 0.92, 0.88, 0.94])
        a = rerank_gated(self.FEATURES_MAP, self.ORDER, r, cal)
        b = rerank_gated(self.FEATURES_MAP, self.ORDER, r, cal)
        self.assertEqual(a, b)
        self.assertEqual(self.ORDER, ["setBarPosition", "setBarScale"])


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_behavior(self):
        r = Reranker()
        r.update(_feats(lex=1.0), [_feats(sem=1.0)])
        r.update(_feats(noun=1.0), [_feats(fuzz=0.5)])
        clone = Reranker.from_dict(r.to_dict())
        self.assertEqual(clone.to_dict(), r.to_dict())
        probe = _feats(lex=0.5, noun=0.5)
        self.assertEqual(clone.score(probe), r.score(probe))

    def test_from_dict_ignores_unknown_features(self):
        clone = Reranker.from_dict(
            {"weights": {"bogus": 5.0, "lex": 0.25}})
        self.assertEqual(clone.weights["lex"], 0.25)
        self.assertNotIn("bogus", clone.weights)


if __name__ == "__main__":
    unittest.main()

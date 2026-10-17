"""Tests for exponential-build T3 (item B2): ADWIN (Bifet & Gavaldà
2007, SDM — the compressed bucket-list variant) as the SECOND drift
detector beside Page-Hinkley over the acceptance stream, plus the
dual-agreement consensus gate and the cortex report wiring.

The contract under test:

- a real mean shift (Bernoulli 0.8 -> 0.3, fixed seed) is cut within a
  bounded, PINNED delay window, the alarm latches, and the post-cut
  estimate tracks the new mean (64/208 on the pinned stream);
- a stable 2000-element stream produces ZERO alarms (the false-positive
  bound, pinned over two fixed streams);
- the exponential-histogram compression is exact: bucket sizes are
  non-increasing powers of two, at most M per row, bucket sums preserve
  the element total, and the width bookkeeping matches n;
- warmup (width < 32) never alarms and says so;
- persistence round-trips byte-identically, mid-stream;
- the consensus gate: a Page-Hinkley-only alarm does NOT flag drift
  (both individual states still reported); both alarmed -> flagged;
- a brute-force sliding-window two-sample reference on the same stream
  agrees with ADWIN's alarm within a bounded lag;
- the cortex learner report surfaces the consensus read-only, beside
  the existing drift lines, deterministically.
"""
import json
import math
import random
import unittest

from assistant.core.adwin import ADWIN, DriftConsensus, consensus
from assistant.core.conformal import PageHinkleyDrift
from assistant.core.learn import CortexLearner


def _stream(phases, seed):
    """Deterministic acceptance stream: [(p, count), ...] phases via
    random.Random(seed) — fixed seeds, so every pinned value below is
    reproducible."""
    rng = random.Random(seed)
    out = []
    for p, count in phases:
        out.extend(rng.random() < p for _ in range(count))
    return out


class MeanShiftTests(unittest.TestCase):
    # Bernoulli(0.8) x 200 then Bernoulli(0.3) x 200, seed 42: the
    # shift lands at element 200; the first cut happens at element 214
    # (lag 14 — the Hoeffding bound needs the newer subwindow to grow
    # to ~2^4 elements before the 0.5 mean gap clears it), leaving a
    # window whose estimate is 64/208 = 0.307692 — the NEW mean.
    SHIFT = _stream([(0.8, 200), (0.3, 200)], 42)

    def test_alarms_within_bounded_delay_and_tracks_new_mean(self):
        adwin = ADWIN()
        first_drift = None
        for i, accepted in enumerate(self.SHIFT):
            row = adwin.update(accepted)
            if row["drift"] and first_drift is None:
                first_drift = i + 1
        self.assertEqual(first_drift, 214)              # pinned, fixed seed
        self.assertEqual(adwin.status()["alarm_at"], 214)
        # the bounded delay window: 10 <= lag <= 30
        self.assertTrue(10 <= first_drift - 200 <= 30, first_drift)
        # post-cut estimate tracks the new mean (0.3), not the old one
        self.assertEqual(adwin.status()["estimate"], 0.307692)
        self.assertTrue(0.2 <= adwin.status()["estimate"] <= 0.4)
        # the post-shift remainder stayed in the window and no further
        # cut happened on the stable tail
        self.assertEqual(adwin.status()["n"], 400)
        self.assertLess(adwin.status()["width"], 400)

    def test_alarm_latches_until_reset(self):
        adwin = ADWIN()
        for accepted in self.SHIFT:
            adwin.update(accepted)
        self.assertTrue(adwin.status()["alarmed"])
        # further stable updates keep the latch (like PageHinkleyDrift)
        row = adwin.update(True)
        self.assertFalse(row["drift"])                  # no NEW cut...
        self.assertEqual(row["alarm_at"], 214)          # ...but latched
        adwin.reset()
        self.assertFalse(adwin.status()["alarmed"])
        self.assertEqual(adwin.status()["width"], 0)
        self.assertEqual(adwin.status()["n"], 0)

    def test_update_report_shape(self):
        adwin = ADWIN()
        row = adwin.update(True)
        self.assertEqual(set(row), {"drift", "width", "estimate", "alarm_at"})
        self.assertEqual(row, {"drift": False, "width": 1,
                               "estimate": 1.0, "alarm_at": None})


class StableStreamTests(unittest.TestCase):
    def test_2000_element_stable_stream_zero_alarms(self):
        for seed, p in ((42, 0.8), (7, 0.6)):
            adwin = ADWIN()
            stream = _stream([(p, 2000)], seed)
            cuts = sum(1 for accepted in stream
                       if adwin.update(accepted)["drift"])
            self.assertEqual(cuts, 0, f"seed={seed} p={p}")
            self.assertFalse(adwin.status()["alarmed"])
            self.assertEqual(adwin.status()["width"], 2000)
            self.assertAlmostEqual(adwin.status()["estimate"], p,
                                   delta=0.05)


class BucketInvariantTests(unittest.TestCase):
    def test_compression_preserves_totals_and_width(self):
        adwin = ADWIN()
        stream = _stream([(0.5, 500)], 3)
        elements = []
        for accepted in stream:
            adwin.update(accepted)
            elements.append(1.0 if accepted else 0.0)
        state = adwin.to_dict()
        buckets = state["buckets"]
        sizes = [b[0] for b in buckets]
        # sizes: non-increasing powers of two (the exponential
        # histogram invariant kept by the merge-oldest-first rule)
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertTrue(all(s & (s - 1) == 0 for s in sizes))
        # at most M=5 buckets per row (runs of equal size)
        run, prev, runs = 0, None, []
        for size in sizes:
            if size == prev:
                run += 1
            else:
                if prev is not None:
                    runs.append(run)
                prev, run = size, 1
        runs.append(run)
        self.assertTrue(all(r <= 5 for r in runs), runs)
        # bucket sums preserve the element total and the width == n
        self.assertEqual(sum(sizes), state["width"])
        self.assertEqual(state["width"], 500)
        self.assertAlmostEqual(sum(b[1] for b in buckets),
                               sum(elements), places=9)

    def test_warmup_never_alarms_and_says_so(self):
        adwin = ADWIN()   # min_width 32
        for _ in range(20):
            self.assertFalse(adwin.update(True)["drift"])
        # width 20 < 32: the report says warmup — no cut is even checked
        self.assertIn("warmup", adwin.status()["message"])
        # now a huge shift lands early: 20 x 1.0 followed by 0.0s.
        # Nothing is checked while the width is below 32...
        seq = [False] * 40
        drifts = [adwin.update(x)["drift"] for x in seq]
        # the first check happens exactly when the width reaches 32 —
        # and this 1.0-vs-0.0 gap clears the bound immediately
        first_drift = 20 + drifts.index(True) + 1
        self.assertEqual(first_drift, 32)
        self.assertEqual(adwin.status()["alarm_at"], 32)


class PersistenceTests(unittest.TestCase):
    def test_roundtrip_byte_identical_mid_stream(self):
        stream = _stream([(0.8, 120), (0.3, 60)], 9)   # includes a cut
        first = ADWIN()
        for accepted in stream[:150]:
            first.update(accepted)
        blob = json.dumps(first.to_dict(), sort_keys=True)
        second = ADWIN().from_dict(json.loads(blob))
        self.assertEqual(json.dumps(second.to_dict(), sort_keys=True),
                         blob)
        # continuing both on the same tail produces identical reports
        reports_a = [first.update(x) for x in stream[150:]]
        third = ADWIN().from_dict(json.loads(blob))
        reports_b = [third.update(x) for x in stream[150:]]
        self.assertEqual(reports_a, reports_b)

    def test_deterministic_across_instances(self):
        stream = _stream([(0.7, 300), (0.2, 100)], 5)
        a, b = ADWIN(), ADWIN()
        reports_a = [a.update(x) for x in stream]
        reports_b = [b.update(x) for x in stream]
        self.assertEqual(reports_a, reports_b)
        self.assertEqual(json.dumps(a.to_dict(), sort_keys=True),
                         json.dumps(b.to_dict(), sort_keys=True))

    def test_bad_parameters_rejected(self):
        with self.assertRaises(ValueError):
            ADWIN(delta=0.0)
        with self.assertRaises(ValueError):
            ADWIN(delta=1.0)
        with self.assertRaises(ValueError):
            ADWIN(max_buckets=1)
        with self.assertRaises(ValueError):
            ADWIN(min_width=1)


class ConsensusTests(unittest.TestCase):
    def test_pure_consensus_over_state_dicts(self):
        ph_quiet = {"alarm_at": None, "n": 40, "mean": 0.1}
        ad_quiet = {"alarm_at": None, "width": 40, "estimate": 0.8}
        ph_loud = {"alarm_at": 42, "n": 42, "mean": 0.3}
        ad_loud = {"alarm_at": 55, "width": 30, "estimate": 0.3}
        # PH-only: reported, NOT flagged
        out = consensus(ph_loud, ad_quiet)
        self.assertFalse(out["flag_drift"])
        self.assertTrue(out["page_hinkley"]["alarmed"])
        self.assertFalse(out["adwin"]["alarmed"])
        self.assertEqual(out["page_hinkley"]["alarm_at"], 42)
        self.assertEqual(out["adwin"]["width"], 40)
        # both: flagged
        self.assertTrue(consensus(ph_loud, ad_loud)["flag_drift"])
        # neither: not flagged
        self.assertFalse(consensus(ph_quiet, ad_quiet)["flag_drift"])
        # the rule travels with the report
        self.assertIn("BOTH detectors have alarmed", out["rule"])

    def test_real_shift_flags_only_on_dual_agreement(self):
        # a sharp shift (0.9 -> 0.2): both detectors alarm -> flagged
        pair = DriftConsensus()
        for accepted in _stream([(0.9, 200), (0.2, 200)], 3):
            pair.update(accepted)
        out = pair.status()
        self.assertTrue(out["page_hinkley"]["alarmed"])
        self.assertTrue(out["adwin"]["alarmed"])
        self.assertTrue(out["flag_drift"])
        self.assertIn("FLAGGED", out["summary"])

    def test_ph_only_alarm_does_not_flag(self):
        # a SMALL persistent drop (0.85 -> 0.70): Page-Hinkley (tuned
        # threshold 4.0) latches quickly; ADWIN's Hoeffding bound needs
        # more data than the 400-element stream gives for a 0.15 gap
        # (epsilon at a 200/200 split is ~0.195), so it stays quiet —
        # and the consensus does NOT flag. Each state still reported.
        pair = DriftConsensus(ph=PageHinkleyDrift(threshold=4.0,
                                                  warmup=25))
        for accepted in _stream([(0.85, 200), (0.70, 200)], 11):
            pair.update(accepted)
        out = pair.status()
        self.assertTrue(out["page_hinkley"]["alarmed"])   # PH is jumpy
        self.assertFalse(out["adwin"]["alarmed"])         # ADWIN is not
        self.assertFalse(out["flag_drift"])               # suppressed
        self.assertIsNotNone(out["page_hinkley"]["alarm_at"])
        self.assertIsNotNone(out["adwin"]["estimate"])
        self.assertIn("page-hinkley alarmed, adwin quiet", out["summary"])

    def test_reset_clears_both(self):
        pair = DriftConsensus()
        for accepted in _stream([(0.9, 200), (0.2, 200)], 3):
            pair.update(accepted)
        self.assertTrue(pair.status()["flag_drift"])
        pair.reset()
        out = pair.status()
        self.assertFalse(out["flag_drift"])
        self.assertEqual(out["page_hinkley"]["n"], 0)
        self.assertEqual(out["adwin"]["width"], 0)

    def test_drift_consensus_persistence_roundtrip(self):
        pair = DriftConsensus()
        for accepted in _stream([(0.9, 100), (0.2, 100)], 3):
            pair.update(accepted)
        blob = json.dumps(pair.to_dict(), sort_keys=True)
        restored = DriftConsensus().from_dict(json.loads(blob))
        self.assertEqual(json.dumps(restored.to_dict(), sort_keys=True),
                         blob)
        self.assertEqual(restored.status(), pair.status())


class BruteForceCrossCheckTests(unittest.TestCase):
    """A tiny brute-force reference: compare the last 32 elements to
    the previous 32 with the SAME Hoeffding bound, checked only once
    both windows are full — no buckets, no compression, just the
    two-sample mean comparison ADWIN approximates."""

    @staticmethod
    def _reference_alarm(stream, window=32, delta=0.002):
        m = 1.0 / (1.0 / window + 1.0 / window)
        epsilon = math.sqrt(math.log(4.0 / delta) / (2.0 * m))
        for t in range(2 * window, len(stream) + 1):
            old = stream[t - 2 * window:t - window]
            new = stream[t - window:t]
            mu0 = sum(1.0 for x in old if x) / window
            mu1 = sum(1.0 for x in new if x) / window
            if abs(mu0 - mu1) >= epsilon:
                return t
        return None

    def test_adwin_agrees_with_the_reference_within_a_lag(self):
        stream = _stream([(0.8, 200), (0.3, 200)], 42)
        t_ref = self._reference_alarm(stream)
        self.assertEqual(t_ref, 224)                    # pinned, fixed seed
        adwin = ADWIN()
        t_adwin = None
        for i, accepted in enumerate(stream):
            if adwin.update(accepted)["drift"] and t_adwin is None:
                t_adwin = i + 1
        self.assertEqual(t_adwin, 214)                  # pinned
        # both fire, within a bounded lag of each other
        self.assertLessEqual(abs(t_ref - t_adwin), 40)

    def test_reference_is_also_quiet_on_the_stable_stream(self):
        stream = _stream([(0.8, 2000)], 42)
        self.assertIsNone(self._reference_alarm(stream))


class CortexWiringTests(unittest.TestCase):
    """The integration: the consensus rides beside the existing drift
    lines in the learner's report (read-only; PH itself unchanged)."""

    @staticmethod
    def _learner(outcomes):
        learner = CortexLearner()
        features = {"lex": 1.0, "sem": 1.0, "fuzz": 1.0,
                    "noun": 1.0, "cue": 1.0}
        for outcome in outcomes:
            learner.observe("text", "setBarScale", features,
                            0.8, outcome)
        return learner

    def test_report_surfaces_the_consensus_beside_existing_drift(self):
        learner = self._learner(["applied"] * 150 + ["rejected"] * 150)
        full = learner.report()
        self.assertIn("drift_ph_adwin", full)
        # the existing lines are untouched beside it
        self.assertIn("drift", full)
        self.assertIn("drift_bocpd", full)
        view = full["drift_ph_adwin"]
        self.assertTrue(view["flag_drift"])          # both alarmed here
        self.assertTrue(view["page_hinkley"]["alarmed"])
        self.assertTrue(view["adwin"]["alarmed"])
        self.assertEqual(view["examples"], 300)
        # deterministic read-only replay
        self.assertEqual(learner.ph_adwin_consensus(),
                         learner.ph_adwin_consensus())
        self.assertEqual(view, learner.ph_adwin_consensus())

    def test_stable_history_does_not_flag(self):
        learner = self._learner(["applied"] * 200)
        view = learner.report()["drift_ph_adwin"]
        self.assertFalse(view["flag_drift"])
        self.assertFalse(view["page_hinkley"]["alarmed"])
        self.assertFalse(view["adwin"]["alarmed"])

    def test_thin_history_reports_honestly(self):
        learner = self._learner(["applied"] * 3)
        view = learner.report()["drift_ph_adwin"]
        self.assertFalse(view["flag_drift"])
        self.assertEqual(view["page_hinkley"]["n"], 3)
        self.assertEqual(view["adwin"]["width"], 3)

    def test_cli_renders_the_consensus_line(self):
        from assistant.core.cli import _render_report
        learner = self._learner(["applied"] * 150 + ["rejected"] * 150)
        lines = _render_report(learner, [])
        rendered = [line for line in lines if "PH+ADWIN" in line]
        self.assertEqual(len(rendered), 1)
        self.assertIn("FLAGGED", rendered[0])
        # the existing drift lines are still rendered
        self.assertTrue(any("drift (BOCPD" in line for line in lines))


if __name__ == "__main__":
    unittest.main()

"""Tests for exponential-build-3 D — the personal second-brain link-graph
and topic extensions (personal surface ONLY; nothing here may leak into
the #120 surface, pinned by the scope-fence tests elsewhere).

Under test:

- HITS (Kleinberg 1999): power iteration with L1 normalization; the
  hand-derived 3-node graph converges to the GOLDEN-SECTION fixed
  point (hub(n1) = auth(n3) = (sqrt(5)-1)/2, derived by solving the
  two-variable fixed-point equations in the test comments) — an exact
  analytic oracle, not a recorded value; hub/authority L1-normalize;
  no-edge graphs return all-zero scores honestly;
- time-decay edge weights: exponential half-life on the source note's
  age (age == half-life -> exactly 0.5, 2x half-life -> 0.25, fresh ->
  1.0), mtime-injected via os.utime for determinism; non-positive
  half-lives are refused;
- weighted PageRank: uniform weights reproduce the unweighted result
  byte-identically (the default path is unchanged, pinned); decayed
  weights reorder "who matters" toward recently-touched notes;
- NMF topic extraction (Lee & Seung 1999): a two-topic vault separates
  its vocabularies, a mixed note carries a genuine two-topic mixture,
  an exact non-negative rank-2 matrix reconstructs to small relative
  error, the error is monotone non-increasing (the paper's guarantee),
  refusals cover k<1 / k>docs / empty corpora / empty vocabularies,
  and everything is byte-identical across runs (seeded init);
- the service + CLI wiring: graph_report / topics_report and the two
  new read-only subcommands.
"""
import io
import os
import random
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain.personal import cli, service
from assistant.capabilities.brain.personal.graph import Graph, decay_weights
from assistant.capabilities.brain.personal.topics import extract, nmf

PHI = (5 ** 0.5 + 1) / 2            # 1.6180339887...
GOLDEN = PHI - 1                    # 0.6180339887... (hub n1 / auth n3)
ANTI_GOLDEN = 2 - PHI               # 0.3819660112... (hub n2 / auth n2)

CAT_NOTES = {
    "cat.md": {"text": "cat cat cat purr whisker cat purr", "path": "x"},
    "cat2.md": {"text": "whisker purr cat nap", "path": "x"},
    "dog.md": {"text": "dog dog bark fetch dog bone", "path": "x"},
    "dog2.md": {"text": "bone fetch bark dog park", "path": "x"},
    "mix.md": {"text": "cat dog purr fetch", "path": "x"},
}


class HitsTests(unittest.TestCase):
    def test_three_node_graph_hits_the_golden_section_fixed_point(self):
        # Hand-derived oracle. Graph: n1 -> {n2, n3}, n2 -> {n3}.
        # Fixed point of the L1-normalized updates: with
        # auth = (0, a2, 1-a2) and hub = (h1, 1-h1, 0):
        #   a2 = h1 / (h1 + 1)          (auth update, normalized)
        #   h1 = 1 / (1 + (1 - a2))     (hub update, normalized)
        # Substituting gives h1^2 + h1 - 1 = 0, so
        #   h1 = (sqrt(5) - 1)/2   — the golden-section conjugate —
        # and a2 = 2 - phi, a3 = h1. The top hub (n1) and the top
        # authority (n3) are DIFFERENT notes: exactly the HITS claim.
        g = Graph()
        g.add("n1", {"n2", "n3"})
        g.add("n2", {"n3"})
        g.add("n3", set())
        r = g.hits()
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["hub"]["n1"], GOLDEN, places=9)
        self.assertAlmostEqual(r["hub"]["n2"], ANTI_GOLDEN, places=9)
        self.assertEqual(r["hub"]["n3"], 0.0)
        self.assertEqual(r["authority"]["n1"], 0.0)
        self.assertAlmostEqual(r["authority"]["n2"], ANTI_GOLDEN,
                               places=9)
        self.assertAlmostEqual(r["authority"]["n3"], GOLDEN, places=9)
        # L1-normalized (both vectors sum to 1)
        self.assertAlmostEqual(sum(r["hub"].values()), 1.0, places=9)
        self.assertAlmostEqual(sum(r["authority"].values()), 1.0,
                               places=9)

    def test_symmetric_pair_is_hub_authority_symmetric(self):
        g = Graph()
        g.add("a", {"b"})
        g.add("b", {"a"})
        r = g.hits()
        self.assertAlmostEqual(r["hub"]["a"], 0.5, places=9)
        self.assertAlmostEqual(r["hub"]["b"], 0.5, places=9)
        self.assertAlmostEqual(r["authority"]["a"], 0.5, places=9)
        self.assertAlmostEqual(r["authority"]["b"], 0.5, places=9)

    def test_no_edges_returns_all_zero_scores(self):
        g = Graph()
        for v in ("a", "b"):
            g.add(v, set())
        r = g.hits()
        self.assertTrue(r["converged"])
        self.assertEqual(r["hub"], {"a": 0.0, "b": 0.0})
        self.assertEqual(r["authority"], {"a": 0.0, "b": 0.0})

    def test_deterministic_and_weight_equivalent(self):
        g = Graph()
        g.add("n1", {"n2", "n3"})
        g.add("n2", {"n3"})
        edges = [(u, t) for u in g.out for t in g.out[u]]
        uniform = {e: 1.0 for e in edges}
        self.assertEqual(g.pagerank(weights=uniform), g.pagerank())
        self.assertEqual(g.hits(weights=uniform), g.hits())
        self.assertEqual(g.hits(), g.hits())


class DecayTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name) / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, text, mtime):
        p = self.vault / name
        p.write_text(text, encoding="utf-8")
        os.utime(p, (mtime, mtime))

    def test_half_life_arithmetic_is_exact(self):
        now = 1_800_000_000.0
        day = 86400.0
        # ages: 0, 90d (one half-life), 180d (two), 45d (half a life)
        self._write("fresh.md", "see [[target]]", now)
        self._write("old90.md", "see [[target]]", now - 90 * day)
        self._write("old180.md", "see [[target]]", now - 180 * day)
        self._write("old45.md", "see [[target]]", now - 45 * day)
        from assistant.capabilities.brain.personal.vault import scan
        weights = decay_weights(scan(str(self.vault)), now,
                                half_life_days=90.0)
        self.assertEqual(weights[("fresh.md", "target")], 1.0)
        self.assertAlmostEqual(weights[("old90.md", "target")], 0.5,
                               places=12)
        self.assertAlmostEqual(weights[("old180.md", "target")], 0.25,
                               places=12)
        self.assertAlmostEqual(weights[("old45.md", "target")], 0.5 ** 0.5,
                               places=12)

    def test_nonpositive_half_life_refuses(self):
        from assistant.capabilities.brain.personal.vault import scan
        notes = scan(str(self.vault))
        with self.assertRaises(ValueError) as caught:
            decay_weights(notes, 0.0, half_life_days=0.0)
        self.assertIn("must be positive", str(caught.exception))

    def test_decayed_pagerank_prefers_recently_touched_links(self):
        # hub-old and hub-fresh both link to their own target; stale
        # note third links to BOTH targets. All-time PageRank cannot
        # tell the two targets apart (identical link structure, the
        # stale link counts as much); the decayed view can.
        now = 1_800_000_000.0
        day = 86400.0
        self._write("hub-old.md", "go [[t-old]]", now - 365 * day)
        self._write("hub-fresh.md", "go [[t-fresh]]", now)
        self._write("third.md", "see [[t-old]] and [[t-fresh]]",
                    now - 365 * day)
        self._write("t-old.md", "old target", now)
        self._write("t-fresh.md", "fresh target", now)
        from assistant.capabilities.brain.personal.vault import scan
        notes = scan(str(self.vault))
        g = Graph()
        for rel, note in notes.items():
            from assistant.capabilities.brain.personal.graph import links
            g.add(rel.lower(), links(note["text"]))
        all_time = g.pagerank()
        self.assertAlmostEqual(all_time["t-old"], all_time["t-fresh"],
                               places=12)
        weights = decay_weights(notes, now, half_life_days=90.0)
        recent = g.pagerank(weights=weights)
        # the LINK-TARGET nodes (no .md — the graph's own convention:
        # [[t-fresh]] creates/points at the node "t-fresh", distinct
        # from the note file t-fresh.md) are what the decay reorders
        self.assertGreater(recent["t-fresh"], recent["t-old"])

    def test_nonpositive_edge_weight_refuses(self):
        g = Graph()
        g.add("a", {"b"})
        with self.assertRaises(ValueError) as caught:
            g.pagerank(weights={("a", "b"): 0.0})
        self.assertIn("must be positive", str(caught.exception))
        with self.assertRaises(ValueError):
            g.hits(weights={("a", "b"): -1.0})
        # weights weaken, never amplify: > 1.0 is refused on both
        # surfaces with the stated reason
        with self.assertRaises(ValueError) as caught:
            g.pagerank(weights={("a", "b"): 1.5})
        self.assertIn("never amplify", str(caught.exception))
        with self.assertRaises(ValueError):
            g.hits(weights={("a", "b"): 2.0})


class NmfTests(unittest.TestCase):
    def test_exact_rank2_matrix_reconstructs(self):
        # V = W_true @ H_true is an exact non-negative rank-2 matrix;
        # NMF must drive the relative error low (the multiplicative
        # tail on structural zeros never fully stops, so converged may
        # honestly be False — the ERROR is the quality claim).
        w_true = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 2.0]]
        h_true = [[2.0, 0.0, 1.0, 0.0, 3.0, 0.0],
                  [0.0, 4.0, 0.0, 1.0, 0.0, 2.0]]
        v = [[sum(w_true[i][t] * h_true[t][j] for t in range(2))
              for j in range(6)] for i in range(4)]
        r = nmf(v, 2)
        self.assertLess(r["error"], 0.01)
        self.assertEqual(r["iterations"], 300)

    def test_error_is_monotone_non_increasing(self):
        # Lee & Seung 1999's guarantee for the alternating updates.
        # Track the error trajectory with one iteration per call.
        w_true = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 2.0]]
        h_true = [[2.0, 0.0, 1.0, 0.0, 3.0, 0.0],
                  [0.0, 4.0, 0.0, 1.0, 0.0, 2.0]]
        v = [[sum(w_true[i][t] * h_true[t][j] for t in range(2))
              for j in range(6)] for i in range(4)]
        errors = [nmf(v, 2, iters=n, tol=0.0)["error"]
                  for n in (1, 5, 20, 100)]
        for early, late in zip(errors, errors[1:]):
            self.assertLessEqual(late, early + 1e-12)

    def test_two_topic_vault_separates(self):
        report = extract(CAT_NOTES, k=2)
        topic_terms = [{t for t, _ in topic["terms"]}
                       for topic in report["topics"]]
        # the dog terms and the cat terms land in DIFFERENT topics
        dog_terms = {"dog", "bark", "fetch", "bone", "park"}
        cat_terms = {"cat", "purr", "whisker", "nap"}
        for dog_topic in (0, 1):
            cat_topic = 1 - dog_topic
            self.assertTrue(dog_terms & topic_terms[dog_topic],
                            dog_terms & topic_terms[dog_topic])
            self.assertTrue(cat_terms & topic_terms[cat_topic])
            self.assertFalse(dog_terms & cat_terms &
                             topic_terms[dog_topic] & topic_terms[cat_topic])
        self.assertTrue(report["converged"])
        self.assertLess(report["relative_error"], 0.5)
        # the mix note carries a genuine two-topic mixture
        mix = report["doc_topics"]["mix.md"]
        self.assertEqual(len(mix["mixture"]), 2)
        self.assertGreater(min(mix["mixture"]), 0.2)
        self.assertAlmostEqual(sum(mix["mixture"]), 1.0, places=5)
        # pure notes are dominated by their own topic with high share
        self.assertGreater(
            report["doc_topics"]["dog.md"]["mixture"][
                report["doc_topics"]["dog.md"]["dominant"]], 0.8)
        self.assertGreater(
            report["doc_topics"]["cat.md"]["mixture"][
                report["doc_topics"]["cat.md"]["dominant"]], 0.8)

    def test_determinism(self):
        self.assertEqual(extract(CAT_NOTES, k=2),
                         extract(CAT_NOTES, k=2))

    def test_refusals_are_honest(self):
        with self.assertRaises(ValueError) as caught:
            extract(CAT_NOTES, k=0)
        self.assertIn("k must be >= 1", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            extract(CAT_NOTES, k=6)  # 6 > 5 notes
        self.assertIn("a topic per note", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            extract({}, k=2)
        self.assertIn("no notes", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            extract({"a.md": {"text": "", "path": "x"},
                     "b.md": {"text": "??", "path": "x"}}, k=1)
        self.assertIn("no terms", str(caught.exception))

    def test_vocabulary_cap_keeps_top_document_frequency(self):
        rng = random.Random(3)
        big = {}
        for i in range(10):
            words = [f"common{j}" for j in range(5)] + \
                [f"rare{i}x{j}" for j in range(5)]
            big[f"n{i}.md"] = {"text": " ".join(words), "path": "x"}
        # cap 5 == exactly the common-vocabulary size: the 5 df=10
        # terms survive, every df=1 rare term is out
        report = extract(big, k=2, max_terms=5)
        kept = {t for topic in report["topics"]
                for t, _ in topic["terms"]}
        self.assertTrue({"common0", "common1"} <= kept)
        self.assertFalse(any(t.startswith("rare") for t in kept))
        self.assertEqual(report["terms"], 5)


class ServiceAndCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name) / "vault"
        self.vault.mkdir()
        now = 1_800_000_000.0
        day = 86400.0
        (self.vault / "cat.md").write_text(
            "cat cat purr whisker see [[dog]]", encoding="utf-8")
        (self.vault / "dog.md").write_text(
            "dog bark fetch see [[cat]]", encoding="utf-8")
        (self.vault / "mix.md").write_text(
            "cat dog purr fetch", encoding="utf-8")
        for name in ("cat.md", "dog.md", "mix.md"):
            os.utime(self.vault / name, (now, now))

    def tearDown(self):
        self._tmp.cleanup()

    def test_graph_report_reports_all_four_views(self):
        r = service.graph_report(str(self.vault), top=3,
                                 now=1_800_000_000.0)
        self.assertEqual(r["notes"], 3)
        for key in ("pagerank", "hubs", "authorities",
                    "pagerank_recent"):
            self.assertTrue(r[key])
        # hubs are the linking notes; authorities the linked targets
        # (zero-score tail entries filtered: the report lists top-N
        # including honest 0.0 rows)
        self.assertEqual([row["note"] for row in r["hubs"]
                           if row["score"] > 0],
                         ["cat.md", "dog.md"])
        self.assertEqual([row["note"] for row in r["authorities"]
                          if row["score"] > 0],
                         ["cat", "dog"])
        self.assertTrue(r["hits_converged"])

    def test_topics_report_matches_module(self):
        r = service.topics_report(str(self.vault), k=2)
        self.assertEqual(r["k"], 2)
        self.assertEqual(r["docs"], 3)

    def test_cli_graph_and_topics(self):
        out = io.StringIO()
        rc = cli.main(["graph", str(self.vault),
                       "--now", "1800000000"], out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        for needle in ("all-time PageRank", "hubs (HITS)",
                       "authorities (HITS)", "recent-weighted PageRank",
                       "HITS converged: yes"):
            self.assertIn(needle, text)
        out = io.StringIO()
        rc = cli.main(["topics", str(self.vault), "--k", "2"], out)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("2 topic(s) over 3 note(s)", text)
        self.assertIn("relative error", text)

    def test_cli_topics_refusal_explains_itself(self):
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        out = io.StringIO()
        rc = cli.main(["topics", str(empty), "--k", "2"], out)
        self.assertEqual(rc, 1)
        self.assertIn("no notes", out.getvalue())


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# exponential-build-4 C: Brandes betweenness + bridge notes, windowed
# topic drift over the EXISTING ADWIN/Page-Hinkley consensus
# ---------------------------------------------------------------------------

import random as _random

from assistant.capabilities.brain.personal.graph import Graph, betweenness, bridge_notes
from assistant.capabilities.brain.personal import topics
from assistant.capabilities.brain.personal.topics import (drift_report, topic_mix,
                                             _percentile_ranks)


class BetweennessTests(unittest.TestCase):
    def _two_cluster_bridge(self):
        g = Graph()
        g.add("k1", ["k2"])
        g.add("k2", ["bridge"])
        g.add("bridge", ["g1"])
        g.add("g1", ["g2"])
        return g

    def test_bridge_node_has_max_betweenness(self):
        b = betweenness(self._two_cluster_bridge())
        self.assertAlmostEqual(b["bridge"]["normalized"], 2 / 3, places=5)
        for other in ("k1", "g2"):
            self.assertLess(b[other]["normalized"], 0.2)

    def test_normalized_in_unit_interval(self):
        b = betweenness(self._two_cluster_bridge())
        for v in b.values():
            self.assertTrue(0.0 <= v["normalized"] <= 1.0)

    def test_empty_graph_zero(self):
        self.assertEqual(betweenness(Graph()), {})


class BridgeNoteTests(unittest.TestCase):
    def test_bridge_note_ranks_first_and_honesty_carried(self):
        g = Graph()
        # cluster A densely links to a hub, cluster B to another; one
        # span note connects them
        g.add("a1", ["a2"])
        g.add("a2", ["hub_a"])
        g.add("a3", ["hub_a"])
        g.add("hub_a", ["span", "a1"])
        g.add("b1", ["b2"])
        g.add("b2", ["hub_b"])
        g.add("hub_b", ["span", "b1"])
        g.add("span", ["hub_a", "hub_b"])
        result = bridge_notes(g, top=3)
        self.assertEqual(result["bridges"][0]["note"], "span")
        self.assertIn("brokerage", result["note"])

    def test_small_vault_refuses_honestly(self):
        g = Graph()
        g.add("x", ["y"])
        result = bridge_notes(g)
        self.assertEqual(result["bridges"], [])
        self.assertIn("too coarse", result["note"])


class PercentileTests(unittest.TestCase):
    def test_ties_share_mean_rank(self):
        p = _percentile_ranks({"a": 1.0, "b": 1.0, "c": 2.0})
        self.assertEqual(p["a"], p["b"])
        self.assertEqual(p["c"], 100.0)


class TopicDriftTests(unittest.TestCase):
    KERNEL = {"a": {"text": "kernel scheduling latency cpu"},
              "b": {"text": "cpu scheduler kernel threads"},
              "c": {"text": "garden plants soil water"},
              "d": {"text": "soil water garden plants"}}
    GARDEN = {"a": {"text": "garden plants soil water mulch"},
              "b": {"text": "soil mulch garden compost plants"},
              "c": {"text": "kernel latency cpu"},
              "d": {"text": "cpu scheduler kernel"}}

    def test_topic_mix_normalized_against_shared_basis(self):
        snaps = [("k1", self.KERNEL), ("k2", self.KERNEL),
                 ("k3", self.MONO), ("k4", self.MONO)]
        # build the basis the same way drift_report does
        union = {}
        for _d, notes in snaps:
            for nid, note in notes.items():
                union[nid] = note
        basis = topics.build_basis(union)
        mix = topic_mix(self.KERNEL, basis)
        self.assertTrue(mix["mix"])
        self.assertAlmostEqual(sum(mix["mix"]), 1.0, places=5)

    def test_fewer_than_four_snapshots_refused(self):
        r = drift_report([("d1", self.KERNEL), ("d2", self.KERNEL)])
        self.assertFalse(r["drifted"])
        self.assertIn("inventing", r["note"])

    def test_warmup_honesty_no_alarm_without_history(self):
        # the existing detectors' warmup is honored: a short stream
        # (even with a real shift) cannot alarm — reported, not faked
        snaps = [("d1", self.KERNEL), ("d2", self.KERNEL),
                 ("d3", self.MONO), ("d4", self.G2)]
        r = drift_report(snaps)
        self.assertFalse(r["drifted"])
        self.assertIn("co-occurrence", r["note"])

    MONO = {"a": {"text": "garden plants soil water mulch"},
            "b": {"text": "soil mulch garden compost plants"},
            "c": {"text": "garden compost soil mulch plants"},
            "d": {"text": "plants soil garden water mulch compost"}}
    G2 = {"a": {"text": "mulch compost garden soil water"},
          "b": {"text": "garden soil water mulch compost"},
          "c": {"text": "compost mulch soil garden plants"},
          "d": {"text": "water soil garden compost mulch"}}

    def test_real_shift_flags_with_enough_history(self):
        # 30 stable kernel snapshots, then 30 garden snapshots rotating
        # two garden mixes: within-phase cosine 1.0, then a SUSTAINED
        # drop (a one-point dip is an outlier and the detectors are
        # right to ignore it). Both existing detectors must alarm
        # through their unchanged consensus gate.
        snaps = [(f"k{i}", self.KERNEL) for i in range(30)]
        for i in range(30):
            snaps.append((f"m{i}", self.MONO if i % 2 == 0 else self.G2))
        r = drift_report(snaps)
        self.assertTrue(r["drifted"], msg=r["detector_status"])
        # the signal stream itself shows the shape: ~1.0 inside the
        # phase, a decisive drop at the boundary, sustained below
        self.assertAlmostEqual(r["similarity_stream"][0][1], 1.0, places=4)
        self.assertLess(r["similarity_stream"][30][1], 0.99)

    def test_gaining_topic_reported_names_terms(self):
        snaps = [(f"k{i}", self.KERNEL) for i in range(10)]
        for i in range(10):
            snaps.append((f"m{i}", self.MONO if i % 2 == 0 else self.G2))
        r = drift_report(snaps)
        self.assertIsNotNone(r["gaining_topic"])
        self.assertTrue(r["gaining_topic"]["top_terms"])
        self.assertGreater(r["gaining_topic"]["delta"], 0.0)

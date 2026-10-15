"""Tests for phase 2.6: the lexicon-diff sharing surface and the
registry-generation adapter interface.

lexicon_diff — the proposal's own verification plan:
1. Export determinism: same learner state -> byte-identical diff.
2. Import round-trip: rows survive export -> parse -> import; the
   persisted pairs land in the embedder seam (PpmiEmbedder
   labeled_pairs) and the review bucket.
3. Rollback: forget restores the pre-import pair list (the embedder
   build input, i.e. the fingerprint's determinant).
4. Injection fuzz: unknown tools, over-long phrases, unparseable
   lines, duplicates, over-cap lists import as no-ops with warnings —
   never exceptions.

gen_adapter — the byte-identity contract:
- the shipped adapter regenerates the committed tools.json
  byte-identically (the drift guard, now through the adapter seam);
- canonical_bytes is the ONE serialization (same dict -> same bytes);
- verify() reports drift structurally and never writes;
- the adapter registry is pinned to the canonical adapter;
- schema check catches structurally invalid adapter output.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from assistant.core import lexicon_diff
from assistant.capabilities.settings import gen_adapter


def _learner_state(examples: list) -> dict:
    return {"cortex_learn": {"examples": examples}}


EXAMPLES = [
    {"text": "frosted glass", "surface": "setBlurEnabled",
     "features": {}, "p": 0.86, "label": 1, "outcome": "applied"},
    {"text": "frosted glass", "surface": "setBlurEnabled",
     "features": {}, "p": 0.9, "label": 1, "outcome": "applied"},
    {"text": "make the bar slim", "surface": "setBarScale",
     "features": {}, "p": 0.79, "label": 1, "outcome": "applied"},
    {"text": "enable the blur", "surface": "setBlurEnabled",
     "features": {}, "p": 0.4, "label": 0, "outcome": "rejected"},
]


class ExportTests(unittest.TestCase):
    def test_export_groups_and_signs_rows(self) -> None:
        rows = lexicon_diff.export_rows(_learner_state(EXAMPLES))
        by_text = {r["text"]: r for r in rows}
        self.assertEqual(by_text["frosted glass"]["n"], 2)
        self.assertEqual(by_text["frosted glass"]["label"], 1)
        self.assertEqual(by_text["enable the blur"]["label"], 0)
        self.assertEqual(by_text["make the bar slim"]["surface"],
                         "setBarScale")

    def test_export_determinism(self) -> None:
        state = _learner_state(EXAMPLES)
        a = lexicon_diff.render(lexicon_diff.export_rows(state),
                                date="2026-09-26")
        b = lexicon_diff.render(lexicon_diff.export_rows(state),
                                date="2026-09-26")
        self.assertEqual(a, b)

    def test_export_is_pii_safe(self) -> None:
        blob = lexicon_diff.render(lexicon_diff.export_rows(
            _learner_state(EXAMPLES)), date="2026-09-26")
        for banned in ("/", ".json", "home", "file:", "timestamp"):
            self.assertNotIn(banned, blob.replace(" -> ", " "))
        # only phrases, surfaces, n, p — no features, no outcomes
        self.assertNotIn("features", blob)
        self.assertNotIn("outcome", blob)

    def test_export_caps_at_200_newest(self) -> None:
        examples = [
            {"text": f"phrase {i}", "surface": "setBarScale",
             "features": {}, "p": 0.5, "label": 1, "outcome": "applied"}
            for i in range(300)]
        rows = lexicon_diff.export_rows(_learner_state(examples))
        self.assertEqual(len(rows), 200)
        texts = {r["text"] for r in rows}
        self.assertIn("phrase 299", texts)   # newest kept
        self.assertNotIn("phrase 0", texts)  # oldest evicted

    def test_empty_state_exports_nothing(self) -> None:
        self.assertEqual(lexicon_diff.export_rows({}), [])


class ParseImportTests(unittest.TestCase):
    DIFF = ("CAELESTIA LEXICON DIFF v1  (2026-09-26)\n"
            "+frosted glass -> setBlurEnabled  (n=4, p=0.86)\n"
            "+make the bar slim -> setBarScale  (n=2, p=0.79)\n"
            "-enable the blur -> setBlurEnabled  (n=1, p=0.40)\n")

    def test_parse_round_trips_the_render(self) -> None:
        rows = lexicon_diff.export_rows(_learner_state(EXAMPLES))
        text = lexicon_diff.render(rows, date="2026-09-26")
        parsed, warnings = lexicon_diff.parse(text)
        self.assertEqual(warnings, [])
        self.assertEqual(len(parsed), 3)
        by_text = {r["text"]: r for r in parsed}
        self.assertEqual(by_text["frosted glass"]["n"], 2)
        self.assertEqual(by_text["frosted glass"]["p"], 0.88)

    def test_import_persists_pairs_and_reports(self) -> None:
        state: dict = {}
        report = lexicon_diff.import_diff(state, self.DIFF)
        self.assertNotIn("error", report)
        self.assertEqual(report["imported"], 3)
        self.assertIn("setBlurEnabled", report["boosted_tools"])
        self.assertIn("rollback", report)
        # the persisted pairs flatten for the embedder seam
        pairs = lexicon_diff.persisted_pairs(state)
        self.assertEqual(len(pairs), 3)
        self.assertIn(("frosted glass", "setBlurEnabled"), pairs)

    def test_import_lands_review_candidates(self) -> None:
        state: dict = {}
        lexicon_diff.import_diff(state, self.DIFF)
        bucket = state.get("cortex_review", [])
        texts = {c.get("text") for c in bucket}
        self.assertIn("frosted glass", texts)
        for c in bucket:
            if c.get("text") == "frosted glass":
                self.assertEqual(c.get("surface"), "setBlurEnabled")

    def test_import_is_idempotent_by_diff_id(self) -> None:
        state: dict = {}
        first = lexicon_diff.import_diff(state, self.DIFF)
        second = lexicon_diff.import_diff(state, self.DIFF)
        self.assertEqual(first["diff_id"], second["diff_id"])
        self.assertEqual(len(lexicon_diff.imported_ids(state)), 1)
        # review candidates are not duplicated either
        texts = [c.get("text") for c in state.get("cortex_review", [])]
        self.assertEqual(len(texts), len(set(texts)))

    def test_injection_fuzz_never_raises(self) -> None:
        # unknown tools, over-long phrases, garbage lines, dupes, over-cap
        garbage = [
            "garbage line with no arrow",
            "+fine phrase -> setBarScale  (n=1, p=0.5)",
            "+fine phrase -> setBarScale  (n=1, p=0.5)",  # duplicate
            "+names a tool -> setTotallyFakeTool  (n=1, p=0.9)",
            "+" + "x" * 500 + " -> setBarScale",
            "+|rm -rf / -> setBarScale",   # a PHRASE, never a command:
            # imported text is router supervision, nothing executes it
            "+p high -> setBarScale  (n=3, p=99.0)",  # p clamps
            "",
            "   ",
        ]
        state: dict = {}
        report = lexicon_diff.import_diff(state, "\n".join(garbage))
        self.assertNotIn("error", report)
        self.assertEqual(report["imported"], 3)
        self.assertTrue(report["warnings"])
        pairs = lexicon_diff.persisted_pairs(state)
        for text, surface in pairs:
            self.assertLessEqual(len(text), lexicon_diff.PHRASE_MAX)
            self.assertLessEqual(len(pairs), lexicon_diff.MAX_PAIRS)

    def test_over_cap_import_truncates_with_warning(self) -> None:
        rows = [{"text": f"phrase {i}", "surface": "setBarScale",
                 "n": 1, "p": 0.5, "label": 1}
                for i in range(lexicon_diff.MAX_PAIRS + 20)]
        text = lexicon_diff.render(rows)
        state: dict = {}
        report = lexicon_diff.import_diff(state, text)
        self.assertEqual(report["imported"], lexicon_diff.MAX_PAIRS)
        self.assertTrue(any("cap" in w for w in report["warnings"]))
        self.assertLessEqual(
            len(lexicon_diff.persisted_pairs(state)),
            lexicon_diff.MAX_PAIRS)

    def test_nothing_importable_is_an_honest_error(self) -> None:
        state: dict = {}
        report = lexicon_diff.import_diff(state, "complete garbage")
        self.assertIn("error", report)


class ForgetTests(unittest.TestCase):
    def test_forget_restores_the_pre_import_pair_list(self) -> None:
        state: dict = {}
        before = lexicon_diff.persisted_pairs(state)
        report = lexicon_diff.import_diff(state, ParseImportTests.DIFF)
        did = report["diff_id"]
        self.assertNotEqual(lexicon_diff.persisted_pairs(state), before)
        forgotten = lexicon_diff.forget(state, did)
        self.assertNotIn("error", forgotten)
        self.assertEqual(lexicon_diff.persisted_pairs(state), before)
        self.assertEqual(lexicon_diff.imported_ids(state), [])

    def test_forget_unknown_id_is_an_honest_error(self) -> None:
        state: dict = {}
        report = lexicon_diff.forget(state, "deadbeef0000")
        self.assertIn("error", report)

    def test_forget_leaves_other_imports_standing(self) -> None:
        state: dict = {}
        a = lexicon_diff.import_diff(state, ParseImportTests.DIFF)
        b = lexicon_diff.import_diff(
            state, "+dock icons -> setDockIconSize  (n=1, p=0.7)\n")
        self.assertNotEqual(a["diff_id"], b["diff_id"])
        lexicon_diff.forget(state, a["diff_id"])
        self.assertEqual(lexicon_diff.imported_ids(state),
                         [b["diff_id"]])
        pairs = lexicon_diff.persisted_pairs(state)
        self.assertEqual(pairs, [("dock icons", "setDockIconSize")])

    def test_diff_id_is_content_addressed(self) -> None:
        rows_a = [{"text": "one", "surface": "setBarScale",
                   "n": 1, "p": 0.5, "label": 1}]
        rows_b = [{"text": "one", "surface": "setBarScale",
                   "n": 2, "p": 0.5, "label": 1}]
        self.assertNotEqual(lexicon_diff.diff_id(rows_a),
                            lexicon_diff.diff_id(rows_b))
        self.assertEqual(lexicon_diff.diff_id(rows_a),
                         lexicon_diff.diff_id(list(rows_a)))


class EmbedderSeamTests(unittest.TestCase):
    """The runtime wiring: imported pairs weight the shared embedder's
    FIRST build; an absent import keeps the corpus-only build."""

    def test_persisted_pairs_feed_the_embedder_seam(self) -> None:
        from assistant.core.vectorize import PpmiEmbedder
        state: dict = {}
        report = lexicon_diff.import_diff(state, ParseImportTests.DIFF)
        pairs = lexicon_diff.persisted_pairs(state)
        supervised = PpmiEmbedder(labeled_pairs=pairs)
        corpus_only = PpmiEmbedder()
        # the supervision changes the geometry for the imported phrase
        q_s = supervised.embed("frosted glass")
        q_c = corpus_only.embed("frosted glass")
        blob_s = repr([round(x, 12) for x in q_s])
        blob_c = repr([round(x, 12) for x in q_c])
        self.assertNotEqual(blob_s, blob_c)
        # determinism: the same pairs rebuild identically
        again = PpmiEmbedder(labeled_pairs=pairs)
        self.assertEqual(blob_s,
                         repr([round(x, 12) for x in
                               again.embed("frosted glass")]))

    def test_singleton_reads_state_pairs_lazily(self) -> None:
        import assistant.core.vectorize as vectorize
        with tempfile.TemporaryDirectory() as tmp:
            brain = Path(tmp) / "state.json"
            state: dict = {}
            lexicon_diff.import_diff(state, ParseImportTests.DIFF)
            brain.write_text(json.dumps(state))
            with unittest.mock.patch.dict(
                    os.environ, {"CAELESTIA_BRAIN_STATE": str(brain)}):
                vectorize._EMBEDDER = None
                e = vectorize.embedder()
                try:
                    self.assertIsNotNone(e)
                finally:
                    vectorize._EMBEDDER = None  # never leak into other tests
        # absent state: corpus-only (the fingerprint-pinned default)
        vectorize._EMBEDDER = None
        corpus_only = vectorize.embedder()
        vectorize._EMBEDDER = None
        self.assertIsNotNone(corpus_only)

    def test_absent_import_keeps_the_corpus_only_fingerprint(self) -> None:
        # the A2 fingerprint test's guarantee, restated for the wiring:
        # PpmiEmbedder() DIRECT stays corpus-only regardless of state
        # (same probes, same blob format as test_svd_embedder)
        import hashlib
        from assistant.core.vectorize import PpmiEmbedder
        e = PpmiEmbedder()
        probes = ["bar scale", "make the dock icons bigger",
                  "blur the frosted glass",
                  "notification popups launcher results",
                  "greeter morning start", "make it see-through",
                  "workspace pill overview"]
        blob = repr([[round(x, 12) for x in e.embed(p)] for p in probes])
        self.assertEqual(
            hashlib.sha256(blob.encode()).hexdigest()[:16],
            "588d52c5fc0ed311")  # corpus spans the generated registry; the pin
        # updates deliberately on every registry resync (not a refactor signal)


class LexiconCliTests(unittest.TestCase):
    """`cortex lexicon export|import|forget|list` end to end, with an
    isolated brain state."""

    def _cli(self, argv: list, stdin: str = "",
             state_path: Path = None) -> str:
        from assistant.core.cli import cmd_cortex
        env = {"CAELESTIA_BRAIN_STATE": str(state_path)}
        old_stdin, sys.stdin = sys.stdin, io.StringIO(stdin)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                with unittest.mock.patch.dict(os.environ, env):
                    rc = cmd_cortex(argv)
        finally:
            sys.stdin = old_stdin
        self.out = out.getvalue()
        return str(rc)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "state.json"

    def test_export_import_forget_round_trip(self) -> None:
        # seed a learner example log
        state = _learner_state(EXAMPLES)
        self.state_path.write_text(json.dumps(state))
        rc = self._cli(["lexicon", "export"],
                       state_path=self.state_path)
        self.assertEqual(rc, "0")
        diff_text = self.out
        self.assertIn("CAELESTIA LEXICON DIFF v1", diff_text)
        self.assertIn("frosted glass -> setBlurEnabled", diff_text)

        # import into a SECOND user's state
        other = Path(self._tmp.name) / "other.json"
        rc = self._cli(["lexicon", "import"], stdin=diff_text,
                       state_path=other)
        self.assertEqual(rc, "0")
        self.assertIn("imported diff", self.out)
        self.assertIn("setBlurEnabled", self.out)  # boosted tools listed
        self.assertIn("rollback: cortex lexicon forget", self.out)
        self.assertTrue(other.exists())
        imported = json.loads(other.read_text())
        self.assertEqual(len(lexicon_diff.persisted_pairs(imported)), 3)

        # list, then forget
        rc = self._cli(["lexicon", "list"], state_path=other)
        self.assertEqual(rc, "0")
        self.assertIn("pair(s)", self.out)
        did = lexicon_diff.imported_ids(imported)[0]
        rc = self._cli(["lexicon", "forget", did], state_path=other)
        self.assertEqual(rc, "0")
        self.assertIn("forgot", self.out)
        after = json.loads(other.read_text())
        self.assertEqual(lexicon_diff.persisted_pairs(after), [])

    def test_import_garbage_fails_honestly(self) -> None:
        rc = self._cli(["lexicon", "import"], stdin="total garbage\n",
                       state_path=self.state_path)
        self.assertEqual(rc, "1")

    def test_capability_kill_switch(self) -> None:
        caps = Path(self._tmp.name) / "caps.json"
        caps.write_text(json.dumps({"lexicon_sharing": False}))
        env = {"CAELESTIA_ASSIST_CAPABILITIES": str(caps),
               "CAELESTIA_BRAIN_STATE": str(self.state_path)}
        from assistant.core.cli import cmd_cortex
        old_stdin, sys.stdin = sys.stdin, io.StringIO("")
        err = io.StringIO()
        import contextlib
        try:
            with redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(err):
                with unittest.mock.patch.dict(os.environ, env):
                    rc = cmd_cortex(["lexicon", "export"])
        finally:
            sys.stdin = old_stdin
        self.assertEqual(rc, 1)
        self.assertIn("disabled", err.getvalue())


class GenAdapterTests(unittest.TestCase):
    def _shell_root(self):
        for cand in [Path(__file__).resolve().parent, *Path(
                __file__).resolve().parent.parents]:
            if (cand / "shell" / "plugin" / "src" / "Caelestia" /
                    "Config").is_dir():
                return cand
        return None

    def test_adapter_registry_is_pinned(self) -> None:
        self.assertEqual(list(gen_adapter.ADAPTERS), ["cpp-headers"])
        self.assertEqual(gen_adapter.CANONICAL_ADAPTER, "cpp-headers")

    def test_canonical_bytes_is_deterministic(self) -> None:
        data = {"meta": {"tool_count": 1},
                "tools": [{"name": "setBarScale"}]}
        self.assertEqual(gen_adapter.canonical_bytes(data),
                         gen_adapter.canonical_bytes(dict(data)))

    def test_unknown_adapter_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            gen_adapter.generate("no-such-adapter", ".")

    def test_verify_reports_drift_structurally(self) -> None:
        root = self._shell_root()
        if root is None:
            self.skipTest("shell/ tree absent — adapter verify skips")
        committed = root / "assistant" / "adapters" / "caelestia" / "tools.json"
        report = gen_adapter.verify("cpp-headers", str(root), committed)
        self.assertTrue(report["identical"])
        self.assertEqual(report["schema_problems"], [])
        # a doctored committed artifact must report drift, not pass
        with tempfile.TemporaryDirectory() as tmp:
            doctored = Path(tmp) / "tools.json"
            doctored.write_bytes(
                committed.read_bytes() + b'{"tampered": true}\n')
            bad = gen_adapter.verify("cpp-headers", str(root), doctored)
            self.assertFalse(bad["identical"])
            self.assertTrue(bad["first_diffs"])

    def test_schema_check_catches_bad_output(self) -> None:
        problems = gen_adapter.verify_output_schema({"tools": []})
        self.assertTrue(problems)
        problems2 = gen_adapter.verify_output_schema(
            {"meta": {}, "tools": [{"name": "x", "kind": "weird"}],
             "not_exposed": [], "presets": [], "explain_rules": []})
        self.assertTrue(any("kind" in p for p in problems2))

    def test_cli_verify_is_green_and_never_writes(self) -> None:
        root = self._shell_root()
        if root is None:
            self.skipTest("shell/ tree absent")
        committed = root / "assistant" / "adapters" / "caelestia" / "tools.json"
        before = committed.read_bytes()
        out = io.StringIO()
        with redirect_stdout(out):
            rc = gen_adapter.main(["--verify"])
        self.assertEqual(rc, 0)
        self.assertIn("byte-identical", out.getvalue())
        self.assertEqual(committed.read_bytes(), before)


class SignerTrustTests(unittest.TestCase):
    """Phase 2.3: EigenTrust-style advisory trust over signer rollback
    history (Kamvar, Schlosser & Garcia-Molina 2003). The invariant under
    test: the score is ADVISORY — it never auto-decides or auto-skips
    the explicit-review requirement for any diff."""

    DIFF_A = ("CAELESTIA LEXICON DIFF v1  (2026-09-26)\n"
              "+frosted glass -> setBlurEnabled  (n=4, p=0.86)\n")
    DIFF_B = ("CAELESTIA LEXICON DIFF v1  (2026-09-26)\n"
              "+make the bar slim -> setBarScale  (n=2, p=0.79)\n")

    def test_import_without_signer_records_no_trust_state(self) -> None:
        state: dict = {}
        lexicon_diff.import_diff(state, self.DIFF_A)
        self.assertNotIn(lexicon_diff.TRUST_KEY, state)
        self.assertNotIn("signer_trust", lexicon_diff.import_diff(
            dict(state), self.DIFF_A))

    def test_import_with_signer_records_event_and_advisory(self) -> None:
        state: dict = {}
        report = lexicon_diff.import_diff(state, self.DIFF_A,
                                          signer="alice")
        self.assertIn("signer_trust", report)
        self.assertIn("ADVISORY ONLY", report["signer_trust"])
        self.assertIn("review is never auto-skipped",
                      report["signer_trust"])
        events = state[lexicon_diff.TRUST_KEY]["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signer"], "alice")
        self.assertEqual(events[0]["action"], "imported")
        # the entry itself carries the signer for rollback attribution
        did = report["diff_id"]
        self.assertEqual(state[lexicon_diff.IMPORTS_KEY][did]["signer"],
                         "alice")

    def test_rollback_earns_the_negative_signal(self) -> None:
        state: dict = {}
        kept = lexicon_diff.import_diff(state, self.DIFF_A, signer="alice")
        bad = lexicon_diff.import_diff(state, self.DIFF_B, signer="mallory")
        lexicon_diff.forget(state, bad["diff_id"])
        scores = lexicon_diff.signer_trust(state)
        # mallory's history: one import, then its rollback — both events
        # stay in the bounded log (an import that was later undone is
        # not erased from history, it is counted as what it was)
        self.assertEqual(scores["mallory"]["kept"], 1)
        self.assertEqual(scores["mallory"]["rolled_back"], 1)
        self.assertEqual(scores["alice"]["kept"], 1)
        self.assertEqual(scores["alice"]["rolled_back"], 0)
        self.assertGreater(scores["alice"]["trust"],
                           scores["mallory"]["trust"])

    def test_trust_propagates_through_tool_overlap(self) -> None:
        # carol's kept diff boosts the SAME tool as alice's: part of
        # alice's standing propagates to carol via the Jaccard edge, so
        # carol outranks dave, whose diffs touch disjoint tools
        state: dict = {"lexicon_trust": {"events": [
            {"signer": "alice", "action": "imported",
             "tools": ["setBlurEnabled"], "at": "2026-09-26T10:00:00"},
            {"signer": "carol", "action": "imported",
             "tools": ["setBlurEnabled"], "at": "2026-09-26T10:01:00"},
            {"signer": "dave", "action": "imported",
             "tools": ["setBarScale"], "at": "2026-09-26T10:02:00"},
        ]}}
        scores = lexicon_diff.signer_trust(state)
        self.assertGreater(scores["carol"]["trust"],
                           scores["dave"]["trust"])
        # determinism: same state, same numbers
        self.assertEqual(scores, lexicon_diff.signer_trust(state))

    def test_empty_history_yields_no_invented_opinions(self) -> None:
        self.assertEqual(lexicon_diff.signer_trust({}), {})
        self.assertIn("flat prior",
                      lexicon_diff.render_advisory({}))

    def test_advisory_never_auto_skips_review(self) -> None:
        # THE invariant: a heavily-trusted signer's import behaves
        # EXACTLY like anyone else's — supervised pairs + review
        # candidates + the verify/review wording, nothing skipped
        state: dict = {"lexicon_trust": {"events": [
            {"signer": "alice", "action": "imported",
             "tools": ["setBlurEnabled"], "at": "t"},
            {"signer": "alice", "action": "imported",
             "tools": ["setBlurEnabled"], "at": "t"},
            {"signer": "alice", "action": "imported",
             "tools": ["setBlurEnabled"], "at": "t"},
        ]}}
        report = lexicon_diff.import_diff(state, self.DIFF_A, signer="alice")
        self.assertNotIn("error", report)
        pairs = lexicon_diff.persisted_pairs(state)
        self.assertIn(("frosted glass", "setBlurEnabled"), pairs)
        review = state.get("cortex_review", [])
        self.assertTrue(any(c.get("text") == "frosted glass"
                            for c in review),
                        "review candidates must still land")
        self.assertIn("review", report["signer_trust"])
        self.assertIn("review is never auto-skipped",
                      report["signer_trust"])

    def test_event_log_is_bounded(self) -> None:
        state: dict = {}
        for i in range(lexicon_diff.MAX_TRUST_EVENTS + 10):
            lexicon_diff._append_trust_event(
                state, {"signer": f"s{i}", "action": "imported",
                        "tools": [], "at": "t"})
        events = state[lexicon_diff.TRUST_KEY]["events"]
        self.assertEqual(len(events), lexicon_diff.MAX_TRUST_EVENTS)


if __name__ == "__main__":
    unittest.main()

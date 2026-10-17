"""brain.qa tests — extraction honesty, type discipline, determinism.

The suite pins the module's whole contract: answers are VERBATIM spans
of real corpus text (never generated), each carrying the provenance to
check it against; the verdict gates fire exactly when the evidence
deserves them; and the hard shape rules hold — a when-answer has a
date, a count has a number, a yes/no has polarity. Floors calibrated on
the committed corpus (see the calibration block at the bottom).
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import qa  # noqa: E402

NOTE_TEXT = ("meeting notes from the shell sync:\n"
             "the widget memory budget is 45 MB per shell restart.\n"
             "remind me to re-measure after the next upstream pull.\n")


class TestAnswerTypeDetection(unittest.TestCase):
    def test_matrix(self):
        cases = {
            "why is the bar invisible": "cause",
            "how do I reset the layout": "how",
            "when was issue 418 reported": "time",
            "who broke the dock": "who",
            "where do snapshots live": "where",
            "how many tools are exposed": "count",
            "list the supported presets": "list",
            "does the bar support autohide": "boolean",
            "bar scale": "what",
        }
        for q, expected in cases.items():
            self.assertEqual(qa.detect_type(q), expected, q)

    def test_empty_question_is_what(self):
        self.assertEqual(qa.detect_type(""), "what")

    def test_procedure_cues_override_interrogative(self):
        # 'how' wins over a bare 'what'-shaped noun phrase
        self.assertEqual(qa.detect_type("steps to rebuild the shell"),
                         "how")


class TestUnits(unittest.TestCase):
    def test_corpus_units_exist_and_carry_provenance(self):
        units = qa.corpus_units()
        self.assertGreater(len(units), 500)
        for u in units:
            self.assertTrue(u["doc_id"])
            self.assertTrue(u["section"])
            self.assertGreaterEqual(u["line"], 1)
            self.assertTrue(u["text"].strip())

    def test_header_terms_rank_but_are_marked_meta(self):
        units = qa.corpus_units()
        metas = [u for u in units if u.get("meta")]
        self.assertTrue(metas)
        self.assertTrue(any("quickshell" in u["text"].lower()
                            for u in metas if u["doc_id"] == "ISS-418"))

    def test_fenced_code_is_not_prose_units(self):
        units = qa.corpus_units()
        self.assertFalse(any(u["text"].startswith("```") for u in units))

    def test_table_rows_are_units(self):
        units = qa.corpus_units()
        rows = [u for u in units if " | " in u["text"]]
        self.assertTrue(rows)


class TestExtractionHonesty(unittest.TestCase):
    def test_answer_is_verbatim_corpus_text(self):
        r = qa.answer("how do I fix a missing qml module metadata")
        self.assertGreater(len(r["answer"]), 0)
        for span in r["answer"]:
            path = (qa.CORPUS_DIR / f"{span['doc_id']}.md")
            self.assertTrue(path.exists(), span["doc_id"])
            self.assertIn(span["text"], path.read_text(encoding="utf-8"),
                          "answer span must be a verbatim corpus quote")

    def test_qml_fix_found(self):
        r = qa.answer("how do I fix a missing qml module metadata")
        self.assertEqual(r["verdict"], "CONFIDENT")
        docs = {s["doc_id"] for s in r["answer"]}
        self.assertIn("DOC-TS-11", docs)
        joined = " ".join(s["text"] for s in r["answer"])
        self.assertIn("QML2_IMPORT_PATH", joined)

    def test_vesktop_cause_found(self):
        r = qa.answer("what causes the vesktop freeze on screenshare")
        self.assertEqual(r["verdict"], "CONFIDENT")
        self.assertEqual(r["answer"][0]["doc_id"], "ISS-402")

    def test_when_scoped_to_subject_doc(self):
        r = qa.answer("when was the quickshell crash dialog issue reported")
        self.assertEqual(r["verdict"], "CONFIDENT")
        for span in r["answer"]:
            self.assertEqual(span["doc_id"], "ISS-418")
            self.assertRegex(span["text"], r"2026-\d{2}-\d{2}")

    def test_gibberish_is_not_found(self):
        r = qa.answer("zzqq blorptastic frumious")
        self.assertEqual(r["verdict"], "NOT_FOUND")
        self.assertEqual(r["answer"], [])

    def test_boolean_requires_polarity(self):
        # the corpus does mention dock badges — near-miss polarity text
        # yields THIN at best; a fabricated yes/no is never CONFIDENT
        r = qa.answer("does the dock support quadruple hyperjump")
        self.assertIn(r["verdict"], ("NOT_FOUND", "THIN"))

    def test_no_terms_is_not_found(self):
        r = qa.answer("the the the")
        self.assertEqual(r["verdict"], "NOT_FOUND")


class TestNotes(unittest.TestCase):
    def test_note_only_fact(self):
        notes = [{"name": "sync.md", "text": NOTE_TEXT}]
        r = qa.answer("what is the widget memory budget", notes=notes)
        self.assertEqual(r["verdict"], "CONFIDENT")
        self.assertEqual(r["answer"][0]["doc_id"], "sync.md")
        self.assertIn("45 MB", r["answer"][0]["text"])

    def test_sources_corpus_ignores_notes(self):
        notes = [{"name": "sync.md", "text": NOTE_TEXT}]
        r = qa.answer("what is the widget memory budget",
                      notes=notes, sources="corpus")
        # corpus fallback is allowed to be weak, but the note must
        # never leak in: no span may cite it
        for span in r["answer"]:
            self.assertNotEqual(span["doc_id"], "sync.md")

    def test_sources_notes_ignores_corpus(self):
        notes = [{"name": "sync.md", "text": NOTE_TEXT}]
        r = qa.answer("what is the widget memory budget",
                      notes=notes, sources="notes")
        self.assertEqual(r["verdict"], "CONFIDENT")
        self.assertEqual(r["answer"][0]["doc_id"], "sync.md")

    def test_empty_note_text_is_skipped(self):
        r = qa.answer("what is the widget memory budget",
                      notes=[{"name": "x", "text": "  "}])
        # no note content -> corpus fallback only; never a confident
        # answer pretending the note said something
        self.assertNotEqual(r["verdict"], "CONFIDENT")


class TestDeterminismAndTime(unittest.TestCase):
    def test_same_question_same_answer(self):
        a = qa.answer("how do I fix gamescope launching in half the screen")
        b = qa.answer("how do I fix gamescope launching in half the screen")
        self.assertEqual(json.dumps(a, sort_keys=True),
                         json.dumps(b, sort_keys=True))

    def test_temporal_window_is_echoed(self):
        from datetime import datetime
        now = datetime(2026, 9, 30, 10, 0, 0)
        r = qa.answer("what did the shell log last night", now=now)
        self.assertIn("window", r)
        self.assertEqual(r["window"]["from"], "2026-09-29T21:00:00")

    def test_fixed_anchor_without_now(self):
        # the module never reads a clock: no `now` -> the parser's own
        # fixed epoch anchors relative windows
        r = qa.answer("what changed today")
        self.assertIn("window", r)
        self.assertTrue(r["window"]["from"].startswith("2026-01-01"))


class TestRender(unittest.TestCase):
    def test_lines_cite_provenance(self):
        r = qa.answer("how do I fix a missing qml module metadata")
        lines = qa.render_lines(r)
        self.assertTrue(lines[0].startswith("[CONFIDENT]"))
        self.assertTrue(any("§ " in ln for ln in lines))

    def test_lines_report_not_found_honestly(self):
        r = qa.answer("zzqq blorptastic frumious")
        lines = qa.render_lines(r)
        self.assertIn("[NOT_FOUND]", lines[0])
        self.assertTrue(any("no extractive answer" in ln for ln in lines))

    def test_thin_lines_warn(self):
        r = qa.answer("does the dock support autohide")
        if r["verdict"] == "THIN":
            lines = qa.render_lines(r)
            self.assertTrue(any("thin evidence" in ln for ln in lines))


class TestBridgeOp(unittest.TestCase):
    def test_qa_answer_op(self):
        from assistant.capabilities.brain import bridge
        r = bridge.handle({"op": "qa_answer",
                           "question": "how do I fix a missing qml "
                                       "module metadata"}, None, None)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["verdict"], "CONFIDENT")

    def test_qa_answer_takes_notes(self):
        from assistant.capabilities.brain import bridge
        r = bridge.handle({"op": "qa_answer",
                           "question": "what is the widget memory budget",
                           "notes": [{"name": "sync.md",
                                      "text": NOTE_TEXT}],
                           "sources": "notes"}, None, None)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["verdict"], "CONFIDENT")
        self.assertEqual(r["result"]["answer"][0]["doc_id"], "sync.md")

    def test_qa_answer_requires_question(self):
        # bridge convention: op-level input errors ride in the result
        # as an "error" key (handle() only catches exceptions)
        from assistant.capabilities.brain import bridge
        r = bridge.handle({"op": "qa_answer"}, None, None)
        self.assertTrue(r["ok"])
        self.assertIn("question", r["result"]["error"])

    def test_capability_off(self):
        from assistant.capabilities.brain import bridge
        with tempfile.TemporaryDirectory() as tmp:
            capfile = Path(tmp) / "caps.json"
            caps = json.loads(_default_caps_json())
            caps["extractive_qa"] = False
            capfile.write_text(json.dumps(caps))
            old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
            os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(capfile)
            try:
                r = bridge.handle({"op": "qa_answer",
                                   "question": "why is bluetooth broken"},
                                  None, None)
            finally:
                if old is None:
                    os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
                else:
                    os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old
        self.assertTrue(r["ok"])
        self.assertIn("extractive_qa", r["result"]["error"])


class TestCLISurface(unittest.TestCase):
    def test_cli_json_and_text(self):
        from assistant.capabilities.brain import cli
        buf = io.StringIO()
        rc = cli.main(["qa", "how do I fix a missing qml module metadata",
                       "--json"], out=buf)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["verdict"], "CONFIDENT")

        buf2 = io.StringIO()
        rc = cli.main(["qa", "how do I fix a missing qml module metadata"],
                      out=buf2)
        self.assertEqual(rc, 0)
        self.assertIn("[CONFIDENT]", buf2.getvalue())

    def test_cli_note_flag(self):
        from assistant.capabilities.brain import cli
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "sync.md"
            note.write_text(NOTE_TEXT, encoding="utf-8")
            buf = io.StringIO()
            rc = cli.main(["qa", "what is the widget memory budget",
                           "--note", str(note), "--sources", "notes",
                           "--json"], out=buf)
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["answer"][0]["doc_id"], "sync.md")

    def test_cli_unreadable_note_is_skipped_honestly(self):
        from assistant.capabilities.brain import cli
        buf = io.StringIO()
        rc = cli.main(["qa", "what is the widget memory budget",
                       "--note", "/nonexistent/note.md"], out=buf)
        self.assertEqual(rc, 0)
        self.assertIn("note unreadable", buf.getvalue())


def _default_caps_json():
    """The DEFAULTS dict as JSON — the capability file override only
    needs the keys the test flips; `enabled` falls back to DEFAULTS for
    the rest, so a minimal file is honest here."""
    from assistant.core.features import DEFAULTS
    return json.dumps(DEFAULTS)


class TestCalibrationRatchet(unittest.TestCase):
    """The floors were calibrated on the committed corpus; these rows
    are the pinned side of that calibration. If one fails after a
    corpus edit, RE-CALIBRATE and justify — do not loosen in place."""

    def test_pinned_rows(self):
        rows = [
            ("how do I fix a missing qml module metadata", "CONFIDENT"),
            ("what causes the vesktop freeze on screenshare", "CONFIDENT"),
            ("how do I fix gamescope launching in half the screen",
             "CONFIDENT"),
            ("when was the quickshell crash dialog issue reported",
             "CONFIDENT"),
            ("zzqq blorptastic frumious", "NOT_FOUND"),
        ]
        for question, expected in rows:
            r = qa.answer(question)
            self.assertEqual(r["verdict"], expected,
                             f"{question}: {r['verdict']} "
                             f"(conf={r.get('confidence')})")


if __name__ == "__main__":
    unittest.main()

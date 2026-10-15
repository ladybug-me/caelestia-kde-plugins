"""genius.fsbrain — the filesystem second-brain layer (phase 2.1).

Pinned here:
- READ-ONLY: staleness/dupes/graph analyses never modify a byte of the
  surveyed tree (bytes compared before/after);
- REUSE, not reimplementation: the frecency formula is sysintel's
  generalized to stat() times; the fingerprints are scan.simhash's with
  Manku banding; PageRank/communities come from brain.personal.graph;
  keyword extraction from brain.textmine + genius.language; the magic
  matcher is scan.ac's automaton; the fallback is brain.naive_bayes;
- the filetype fallback is CORRECTABLE: a correction becomes a bounded,
  reviewable doc in the brain state and future unknowns of the same byte
  shape rank the taught type;
- no watcher, no sleep-loop, nothing resident (the module ships none —
  see the package rationale's known-gaps entry for the measured reason).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from assistant.capabilities.genius import fsbrain


def _age(path: Path, days: float, now: datetime) -> None:
    stamp = (now - timedelta(days=days)).timestamp()
    os.utime(path, (stamp, stamp))


class StalenessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.now = datetime(2026, 9, 26, tzinfo=timezone.utc)

    def _make(self, name: str, days: float, payload: bytes = b"data") -> Path:
        path = self.dir / name
        path.write_bytes(payload)
        _age(path, days, self.now)
        return path

    def test_warm_outranks_cold_and_cold_outranks_ancient(self):
        self._make("warm.txt", 1.0)
        self._make("cold.txt", 90.0)
        self._make("ancient.txt", 400.0)
        report = fsbrain.staleness_report(str(self.dir), top=3, now=self.now)
        self.assertEqual(report["n_files"], 3)
        self.assertEqual(
            [Path(r["path"]).name for r in report["warmest"]],
            ["warm.txt", "cold.txt", "ancient.txt"])
        self.assertEqual(
            [Path(r["path"]).name for r in report["coldest"]],
            ["ancient.txt", "cold.txt", "warm.txt"])

    def test_frecency_formula_is_sysintels_generalized(self):
        # one file, one write event 30 days ago (== one half-life):
        # activity = (1 + 0.5^1) + (1 + 0.5^1) = 3.0 exactly.
        self._make("one.txt", 30.0)
        report = fsbrain.staleness_report(
            str(self.dir), top=1, half_life_days=30.0, now=self.now)
        self.assertAlmostEqual(report["warmest"][0]["activity"], 3.0, places=3)

    def test_directory_entropy_scores_mixed_piles(self):
        (self.dir / "pure").mkdir()
        (self.dir / "pure" / "a.png").write_bytes(b"x")
        (self.dir / "pure" / "b.png").write_bytes(b"y")
        (self.dir / "mixed").mkdir()
        (self.dir / "mixed" / "a.png").write_bytes(b"x")
        (self.dir / "mixed" / "b.md").write_bytes(b"y")
        (self.dir / "mixed" / "c.zip").write_bytes(b"z")
        report = fsbrain.staleness_report(str(self.dir), top=5, now=self.now)
        by_dir = {e["dir"].split("/")[-1]: e for e in report["mixed_directories"]}
        self.assertAlmostEqual(by_dir["pure"]["entropy_bits"], 0.0, places=3)
        self.assertAlmostEqual(by_dir["mixed"]["entropy_bits"], 1.585,
                               places=2)  # log2(3) over three categories

    def test_report_is_read_only(self):
        self._make("warm.txt", 1.0)
        before = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        fsbrain.staleness_report(str(self.dir), now=self.now)
        after = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        self.assertEqual(before, after)

    def test_missing_root_is_a_clean_error(self):
        with self.assertRaises(ValueError):
            fsbrain.staleness_report("/definitely/not/here")


class NearDuplicateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_same_name_across_dirs_groups(self):
        for subdir in ("a", "b", "c"):
            (self.dir / subdir).mkdir()
            (self.dir / subdir / "IMG_0421.jpg").write_bytes(b"jpegdata")
        report = fsbrain.metadata_near_duplicates(str(self.dir))
        self.assertEqual(report["n_groups"], 1)
        group = report["groups"][0]
        self.assertEqual(group["n"], 3)
        self.assertTrue(all("IMG_0421.jpg" in f for f in group["files"]))

    def test_distinct_files_do_not_group(self):
        (self.dir / "a").mkdir()
        (self.dir / "a" / "IMG_0421.jpg").write_bytes(b"one")
        (self.dir / "a" / "report-2026.pdf").write_bytes(b"two")
        (self.dir / "a" / "budget-final.xlsx").write_bytes(b"three")
        report = fsbrain.metadata_near_duplicates(str(self.dir))
        self.assertEqual(report["n_groups"], 0)

    def test_read_only(self):
        (self.dir / "a").mkdir()
        (self.dir / "a" / "dup.png").write_bytes(b"x")
        (self.dir / "b").mkdir()
        (self.dir / "b" / "dup.png").write_bytes(b"y")
        before = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        fsbrain.metadata_near_duplicates(str(self.dir))
        after = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        self.assertEqual(before, after)


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _notes(self):
        (self.dir / "rust.md").write_text(
            "rust learning plan; compile borrow checker rust compiler notes")
        (self.dir / "cargo.md").write_text(
            "cargo build rust package manager compile crates registry")
        (self.dir / "kde.md").write_text(
            "kde plasma widgets wayland panel desktop shell theming")
        (self.dir / "wayland.md").write_text(
            "wayland compositors kde plasma rendering protocol surface")

    def test_top_terms_and_related_docs(self):
        self._notes()
        result = fsbrain.knowledge_graph([str(self.dir)])
        self.assertEqual(result["n_docs"], 4)
        terms = [t["term"] for t in result["top_terms"]]
        self.assertIn("rust", terms)
        # rust.md and cargo.md share the rust/compile vocabulary
        pair_names = {(Path(r["a"]).name, Path(r["b"]).name)
                      for r in result["related_docs"]}
        self.assertIn(("cargo.md", "rust.md"), pair_names)
        self.assertIn(("kde.md", "wayland.md"), pair_names)
        for row in result["related_docs"]:
            self.assertEqual(len(row["shared"]) <= 8, True)

    def test_communities_exist_for_clustered_notes(self):
        self._notes()
        result = fsbrain.knowledge_graph([str(self.dir)])
        # the rust/cargo vocabulary and the kde/wayland vocabulary form
        # separate connected components — communities larger than one
        self.assertGreaterEqual(len(result["communities"]), 1)

    def test_non_doc_files_are_ignored(self):
        self._notes()
        (self.dir / "image.png").write_bytes(b"\x89PNG" + b"0" * 16)
        result = fsbrain.knowledge_graph([str(self.dir)])
        self.assertEqual(result["n_docs"], 4)

    def test_read_only(self):
        self._notes()
        before = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        fsbrain.knowledge_graph([str(self.dir)])
        after = sorted(
            (str(p), p.read_bytes()) for p in self.dir.rglob("*")
            if p.is_file())
        self.assertEqual(before, after)


class FileTypeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _file(self, name: str, payload: bytes) -> str:
        path = self.dir / name
        path.write_bytes(payload)
        return str(path)

    def test_signature_table(self):
        cases = [
            ("img.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "png"),
            ("img.jpg", b"\xff\xd8\xff\xe0" + b"j" * 16, "jpeg"),
            ("doc.pdf", b"%PDF-1.7\n" + b"%" * 16, "pdf"),
            ("bin.zip", b"PK\x03\x04" + b"z" * 16, "zip"),
            ("prog.elf", b"\x7fELF\x02\x01\x01" + b"\0" * 16, "elf"),
            ("run.sh", b"#!/bin/sh\necho hi\n", "script"),
            ("data.gz", b"\x1f\x8b\x08" + b"g" * 16, "gzip"),
            ("arc.tar", b"\0" * 257 + b"ustar\x00" + b"t" * 100, "tar"),
        ]
        for name, payload, expected in cases:
            result = fsbrain.infer_filetype(self._file(name, payload))
            self.assertEqual(result["verdict"], "signature", name)
            self.assertEqual(result["type"], expected, name)

    def test_offset_anchoring_blocks_midfile_magic(self):
        # "PK\x03\x04" deep inside a text file must NOT claim zip.
        payload = b"# just a plain text note\nnothing magic\nPK\x03\x04tail"
        result = fsbrain.infer_filetype(self._file("note.txt", payload))
        self.assertNotEqual(result["type"], "zip")

    def test_unknown_is_honest(self):
        result = fsbrain.infer_filetype(
            self._file("mystery.dat", b"CUSTOMBLOB\x01\x02\xff\xff"))
        self.assertEqual(result["verdict"], "unknown")
        self.assertIsNone(result["type"])
        self.assertIn("teach", result)

    def test_correction_teaches_the_fallback(self):
        mystery = self._file(
            "mystery.dat", b"CUSTOMBLOB\x01\x02\x03" + b"\xff" * 20)
        # before: unknown
        self.assertEqual(
            fsbrain.infer_filetype(mystery)["verdict"], "unknown")
        # teach once
        doc = fsbrain.record_correction(mystery, "custom-blob")
        state = fsbrain.append_correction({}, doc)
        classifier = fsbrain.load_filetype_classifier(state)
        taught = fsbrain.infer_filetype(mystery, classifier=classifier)
        self.assertEqual(taught["verdict"], "bayes")
        self.assertEqual(taught["type"], "custom-blob")

    def test_correction_docs_are_bounded(self):
        mystery = self._file("m.dat", b"BYTES")
        state: dict = {}
        for i in range(fsbrain._NB_MAX_DOCS + 10):
            doc = fsbrain.record_correction(mystery, f"kind{i % 3}")
            state = fsbrain.append_correction(state, doc)
        self.assertEqual(len(state[fsbrain.DOCS_KEY]), fsbrain._NB_MAX_DOCS)


class CliTests(unittest.TestCase):
    """The genius fsbrain command surface (in-process, stdout captured)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        (self.dir / "note.md").write_text(
            "rust compile borrow checker planning notes")
        (self.dir / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    def _run(self, argv):
        import contextlib
        import io
        from assistant.capabilities.genius import cli as genius_cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = genius_cli.main(argv)
        return code, buffer.getvalue()

    def test_stale_json_round_trip(self):
        code, out = self._run(["fsbrain", "stale", str(self.dir),
                               "--top", "2", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["n_files"], 2)
        self.assertTrue(payload["warmest"])

    def test_dupes(self):
        code, out = self._run(["fsbrain", "dupes", str(self.dir)])
        self.assertEqual(code, 0)
        self.assertIn("n groups", out)

    def test_graph(self):
        code, out = self._run(["fsbrain", "graph", str(self.dir), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["n_docs"], 1)

    def test_filetype_signature(self):
        code, out = self._run(["fsbrain", "filetype",
                               str(self.dir / "img.png"), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["type"], "png")

    def test_filetype_correction_persists_bounded_docs(self):
        mystery = self.dir / "weird.dat"
        mystery.write_bytes(b"OPAQUEBLOB\x01\x02\x03\xff")
        with mock.patch("assistant.capabilities.brain.state.load", return_value={}), \
                mock.patch("assistant.capabilities.brain.state.save") as save:
            code, out = self._run(["fsbrain", "filetype", str(mystery),
                                   "--correct", "opaque-blob", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["type"], "opaque-blob")
        saved_state = save.call_args[0][0]
        docs = saved_state[fsbrain.DOCS_KEY]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["label"], "opaque-blob")
        self.assertEqual(docs[0]["features"][0], "b4f")  # 'O' = 0x4f

    def test_filetype_uses_taught_state(self):
        mystery = self.dir / "weird.dat"
        mystery.write_bytes(b"OPAQUEBLOB\x01\x02\x03\xff")
        doc = fsbrain.record_correction(str(mystery), "opaque-blob")
        state = fsbrain.append_correction({}, doc)
        with mock.patch("assistant.capabilities.brain.state.load", return_value=state):
            code, out = self._run(["fsbrain", "filetype", str(mystery),
                                   "--json"])
        payload = json.loads(out)
        self.assertEqual(payload["verdict"], "bayes")
        self.assertEqual(payload["type"], "opaque-blob")

    def test_missing_path_is_a_clean_error(self):
        import contextlib
        import io
        import sys as _sys
        from assistant.capabilities.genius import cli as genius_cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = genius_cli.main(["fsbrain", "stale", "/no/such/dir"])
        self.assertEqual(code, 1)
        self.assertIn("no such directory", err.getvalue())


if __name__ == "__main__":
    unittest.main()

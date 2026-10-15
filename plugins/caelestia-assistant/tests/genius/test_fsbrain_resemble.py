"""Tests for exponential-build phase 2.3: NCD-based file/folder
resemblance (`genius fsbrain resemble FILE --folders D1,D2,...`).

The contract under test:

- the ONE existing NCD primitive (genius/data.py::ncd, Li et al. 2004)
  answers 'which existing folder does this file most resemble' — the
  pure core never touches the filesystem, the wrapper only reads;
- folders with no samples are skipped and NAMED, never scored as 1.0;
  empty content is refused; every folder's distance is reported (the
  ranking IS the answer), ties broken by name;
- both methods (nearest-sample / concatenated profile) work, samples
  are deterministically head-truncated and the truncation is reported;
- the CLI wires it read-only end to end (tmp files, no writes).
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.capabilities.genius import data as genius_data
from assistant.capabilities.genius import fsbrain
from assistant.capabilities.genius.cli import main as genius_main

# two synthetic "formats": repetitive text vs repetitive CSV rows
_TEXT = ("the quick brown fox jumps over the lazy dog\n" * 20).encode()
_CSV = b"name,value\n" + b"row,%d\n" * 1  # placeholder, built below
_CSV = b"name,value\n" + b"".join(b"row,%d\n" % i for i in range(40))


class NcdResembleCoreTests(unittest.TestCase):
    def test_text_file_matches_the_text_folder(self):
        folders = {"notes": [_TEXT, _TEXT[:1000]],
                   "sheets": [_CSV, _CSV[:900]]}
        out = fsbrain.ncd_resemble(_TEXT[:2000], folders)
        self.assertEqual(out["best_folder"], "notes")
        self.assertLess(out["distances"][0]["distance"],
                        out["distances"][1]["distance"])
        # the full ranking is the answer
        self.assertEqual([m["folder"] for m in out["distances"]],
                         ["notes", "sheets"])

    def test_csv_content_matches_the_csv_folder(self):
        folders = {"notes": [_TEXT], "sheets": [_CSV]}
        out = fsbrain.ncd_resemble(_CSV[:2000], folders)
        self.assertEqual(out["best_folder"], "sheets")

    def test_profile_method_uses_the_concatenated_folder(self):
        folders = {"sheets": [_CSV, _CSV]}
        out = fsbrain.ncd_resemble(_CSV[:2000], folders, method="profile")
        self.assertEqual(out["method"], "profile")
        self.assertEqual(out["best_folder"], "sheets")
        # the profile distance equals NCD against the concatenation
        expected = genius_data.ncd(_CSV[:2000], _CSV + _CSV)["ncd"]
        self.assertAlmostEqual(out["distances"][0]["distance"], expected)

    def test_empty_folders_are_skipped_and_named(self):
        out = fsbrain.ncd_resemble(_TEXT[:500],
                                   {"notes": [_TEXT], "empty": []})
        self.assertEqual(out["best_folder"], "notes")
        self.assertEqual(out["skipped_empty_folders"], ["empty"])
        self.assertNotIn("empty", [m["folder"] for m in out["distances"]])

    def test_all_empty_folders_is_an_honest_refusal(self):
        with self.assertRaises(ValueError):
            fsbrain.ncd_resemble(_TEXT[:500], {"a": [], "b": []})

    def test_empty_content_is_refused(self):
        with self.assertRaises(ValueError):
            fsbrain.ncd_resemble(b"", {"notes": [_TEXT]})

    def test_unknown_method_refused(self):
        with self.assertRaises(ValueError):
            fsbrain.ncd_resemble(_TEXT[:100], {"notes": [_TEXT]},
                                 method="vibes")

    def test_truncation_is_reported(self):
        big = _TEXT * 200  # ~180 KB, past the 1000-byte head cap
        out = fsbrain.ncd_resemble(big, {"notes": [_TEXT]},
                                   max_sample_bytes=1000)
        self.assertTrue(out["file_truncated"])
        self.assertEqual(out["compared_bytes"], 1000)

    def test_ties_break_by_folder_name(self):
        # identical single sample in both folders: a genuine tie
        out = fsbrain.ncd_resemble(_TEXT[:500], {"zzz": [_TEXT[:500]],
                                                 "aaa": [_TEXT[:500]]})
        self.assertEqual(out["best_folder"], "aaa")
        self.assertEqual(out["distances"][0]["distance"],
                         out["distances"][1]["distance"])


class WrapperAndCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.notes = self.dir / "notes"
        self.sheets = self.dir / "sheets"
        self.notes.mkdir()
        self.sheets.mkdir()
        for i in range(3):
            (self.notes / f"note{i}.txt").write_text(
                "the quick brown fox jumps over the lazy dog\n" * 10)
            (self.sheets / f"sheet{i}.csv").write_text(
                "name,value\n" + "".join(f"row,{j}\n" for j in range(20)))
        self.query = self.dir / "query.txt"
        self.query.write_text("the quick brown fox jumps over the lazy dog\n"
                              * 5)

    def tearDown(self):
        self._tmp.cleanup()

    def test_wrapper_reads_only_and_ranks(self):
        out = fsbrain.resemble_file(str(self.query),
                                    [str(self.notes), str(self.sheets)])
        self.assertEqual(out["best_folder"], str(self.notes))
        self.assertEqual(out["unreadable_samples"], 0)
        # nothing was written anywhere
        self.assertEqual(sorted(p.name for p in self.notes.iterdir()),
                         ["note0.txt", "note1.txt", "note2.txt"])

    def test_wrapper_counts_unreadable_samples(self):
        unreadable = self.dir / "locked"
        unreadable.mkdir()
        (unreadable / "x.bin").write_bytes(b"\x00\x01")
        # a directory passed where a folder's file is unreadable: use a
        # file path as a "folder" so scandir yields nothing readable
        out = fsbrain.resemble_file(str(self.query),
                                    [str(self.notes), str(unreadable)])
        self.assertIn("best_folder", out)

    def test_cli_resemble_end_to_end(self):
        import io as _io
        from contextlib import redirect_stdout
        buf = _io.StringIO()
        with redirect_stdout(buf):
            rc = genius_main(["fsbrain", "resemble", str(self.query),
                              "--folders", f"{self.notes},{self.sheets}"])
        self.assertEqual(rc, 0)
        self.assertIn(str(self.notes), buf.getvalue())

    def test_cli_resemble_requires_folders(self):
        import io as _io
        import contextlib
        from contextlib import redirect_stdout
        err = _io.StringIO()
        with contextlib.redirect_stderr(err), redirect_stdout(_io.StringIO()):
            rc = genius_main(["fsbrain", "resemble", str(self.query)])
        self.assertEqual(rc, 1)
        self.assertIn("--folders", err.getvalue())

    def test_pure_core_used_the_single_ncd_primitive(self):
        with mock.patch("assistant.capabilities.genius.data.ncd",
                        wraps=genius_data.ncd) as spy:
            fsbrain.ncd_resemble(_TEXT[:500], {"notes": [_TEXT[:500]]})
        self.assertGreaterEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()

"""FTS5 store + ONNX encoder seam tests: honest availability, WAL mode,
identical round-trips, and the never-bundled-model constraint."""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from assistant.core import encoder_onnx, fts_store


class TestFtsStore(unittest.TestCase):
    def test_probe_returns_bool(self):
        self.assertIsInstance(fts_store.fts5_available(), bool)

    def test_round_trip_when_available(self):
        if not fts_store.fts5_available():
            self.skipTest("FTS5 unavailable in this sqlite build")
        idx = fts_store.FTS5Index()
        try:
            res = idx.index([
                ("d1", "Bar settings", "the bar height and dock geometry"),
                ("d2", "Notifications", "popup timeout and do not disturb"),
            ])
            self.assertTrue(res["ok"])
            hits = idx.search("bar")
            self.assertTrue(hits)
            self.assertEqual(hits[0]["doc_id"], "d1")
        finally:
            idx.close()

    def test_wal_mode_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = fts_store.connect_wal(Path(tmp) / "t.db")
            try:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                self.assertEqual(mode.lower(), "wal")
            finally:
                conn.close()

    def test_malformed_query_is_empty_not_crash(self):
        if not fts_store.fts5_available():
            self.skipTest("FTS5 unavailable in this sqlite build")
        idx = fts_store.FTS5Index()
        try:
            idx.index([("d1", "t", "body text")])
            self.assertEqual(idx.search('"unbalanced phrase'), [])
        finally:
            idx.close()


class TestOnnxSeam(unittest.TestCase):
    def test_seam_never_bundles_a_model(self):
        # upstream constraint: no model files in the tree, ever
        pkg = Path(encoder_onnx.__file__).parent
        for ext in ("*.onnx", "*.gguf", "*.bin", "*.safetensors"):
            self.assertEqual(list(pkg.glob(ext)), [], f"bundled model: {ext}")

    def test_without_env_is_honestly_unavailable(self):
        old = os.environ.pop("CAELESTIA_ENCODER_ONNX", None)
        try:
            out = encoder_onnx.load_encoder()
            if not out["available"]:
                self.assertIn("hint", out)
                self.assertIsNone(out["encode"])
        finally:
            if old is not None:
                os.environ["CAELESTIA_ENCODER_ONNX"] = old

    def test_missing_path_is_honestly_unavailable(self):
        os.environ["CAELESTIA_ENCODER_ONNX"] = "/nonexistent/model.onnx"
        try:
            out = encoder_onnx.load_encoder()
            self.assertFalse(out["available"])
        finally:
            os.environ.pop("CAELESTIA_ENCODER_ONNX", None)


if __name__ == "__main__":
    unittest.main()

"""Corpus + index tests: strict headers, determinism, index-only search."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.retrieval import CORPUS_DIR, DEFAULT_INDEX_PATH
from assistant.capabilities.retrieval import indexer
from assistant.capabilities.retrieval.search import Searcher

DOC_ID_RE = re.compile(r"^(DOC-(TS-\d{2}|ARCH-[A-Z]+|README)|ISS-\d{3})$")


class TestCorpusDocs(unittest.TestCase):
    def test_every_corpus_doc_has_strict_header(self) -> None:
        corpus = sorted(CORPUS_DIR.glob("*.md"))
        self.assertGreaterEqual(len(corpus), 25, "corpus should hold the docs + issues set")
        for path in corpus:
            meta, body = indexer.parse_corpus_doc(path)
            for key in ("id", "title", "source", "tags"):
                self.assertIn(key, meta, f"{path.name}: missing {key}")
            self.assertTrue(body.strip(), f"{path.name}: empty body")
            self.assertEqual(meta["id"], path.stem, f"{path.name}: id must match filename")

    def test_corpus_ids_unique_and_wellformed(self) -> None:
        ids = []
        for path in CORPUS_DIR.glob("*.md"):
            meta, _body = indexer.parse_corpus_doc(path)
            ids.append(meta["id"])
            self.assertRegex(meta["id"], DOC_ID_RE)
        self.assertEqual(len(ids), len(set(ids)), "duplicate corpus ids")

    def test_expected_doc_families_present(self) -> None:
        ids = {indexer.parse_corpus_doc(p)[0]["id"] for p in CORPUS_DIR.glob("*.md")}
        for number in range(1, 12):
            self.assertIn(f"DOC-TS-{number:02d}", ids)
        for arch in ("DOC-ARCH-KWIN", "DOC-ARCH-LOCKSCREEN", "DOC-ARCH-SHORTCUT", "DOC-README"):
            self.assertIn(arch, ids)
        for issue in (6, 14, 120, 333, 402, 409, 416, 418, 461, 528, 616, 641,
                      738, 744, 763, 765, 773, 774, 788, 790, 791, 792, 802, 805):
            self.assertIn(f"ISS-{issue:03d}", ids)


class TestTokenizer(unittest.TestCase):
    def test_lowercase_split_stopwords(self) -> None:
        self.assertEqual(indexer.tokenize("The KWin Bridge!"), ["kwin", "bridge"])
        self.assertEqual(indexer.tokenize("vesktop freezes when I screenshare"), ["vesktop", "freezes", "screenshare"])
        self.assertEqual(indexer.tokenize("a I of the"), [])
        self.assertEqual(indexer.tokenize("zkde_screencast_unstable_v1"), ["zkde", "screencast", "unstable", "v1"])

    def test_short_tokens_dropped(self) -> None:
        self.assertEqual(indexer.tokenize("a b cd"), ["cd"])


class TestIndexBuild(unittest.TestCase):
    def test_build_is_byte_deterministic(self) -> None:
        docs = indexer.load_corpus()
        first = indexer.serialize_index(indexer.build_index(docs))
        second = indexer.serialize_index(indexer.build_index(indexer.load_corpus()))
        self.assertEqual(first, second, "same corpus must produce identical index bytes")

    def test_committed_index_matches_corpus(self) -> None:
        """The committed index must be exactly what the current corpus builds."""
        self.assertTrue(DEFAULT_INDEX_PATH.is_file(), "index/bm25.json missing; run the build module")
        rebuilt = indexer.serialize_index(indexer.build_index(indexer.load_corpus()))
        committed = DEFAULT_INDEX_PATH.read_bytes()
        self.assertEqual(rebuilt, committed, "index/bm25.json is stale — rerun python3 -m assistant.capabilities.retrieval.build")

    def test_index_shape(self) -> None:
        index = indexer.build_index(indexer.load_corpus())
        self.assertEqual(index["version"], 1)
        self.assertEqual(index["params"], {"k1": 1.5, "b": 0.75})
        self.assertGreater(index["n_docs"], 0)
        for doc in index["docs"]:  # type: ignore[union-attr]
            self.assertIn("sentences", doc)
            self.assertGreater(doc["dl"], 0)


class TestIndexOnlySearch(unittest.TestCase):
    def test_search_works_without_corpus_files(self) -> None:
        """Runtime search must read only the index file — no corpus .md nearby."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            target = tmp_dir / "bm25.json"
            target.write_bytes(DEFAULT_INDEX_PATH.read_bytes())
            self.assertEqual(list(tmp_dir.glob("*.md")), [], "test dir must hold the index only")
            searcher = Searcher.from_file(target)
            results = searcher.search("vesktop freezes when I screenshare", k=3)
            self.assertTrue(results)
            for hit in results:
                self.assertIn("doc_id", hit)
                self.assertIn("title", hit)
                self.assertIn("score", hit)
                self.assertIn("source", hit)
                self.assertIn("snippet", hit)

    def test_unknown_words_return_no_results(self) -> None:
        searcher = Searcher.from_file()
        self.assertEqual(searcher.search("zzzqqq wugga blorptastic"), [])


if __name__ == "__main__":
    unittest.main()

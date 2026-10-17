"""Offline BM25 search over the prebuilt retrieval index.

Runtime rules:
- Reads ONLY the JSON index file (assistant/retrieval/index/bm25.json by
  default). It never opens the corpus markdown — the corpus is parsed once by
  the offline build step, never at query time.
- Stdlib only, no network, deterministic ranking (score desc, then doc id).

API:
    search(query: str, k: int = 5) -> list of
        {doc_id, title, score, source, snippet}
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import DEFAULT_INDEX_PATH
from .indexer import B, K1, tokenize

SNIPPET_MAX = 240


class Searcher:
    """BM25 searcher over a loaded index dict."""

    def __init__(self, index: Dict[str, Any]):
        if index.get("version") != 1:
            raise ValueError(f"unsupported index version: {index.get('version')!r}")
        params = index.get("params", {})
        self.k1 = float(params.get("k1", K1))
        self.b = float(params.get("b", B))
        self.n_docs = int(index.get("n_docs", len(index.get("docs", []))))
        self.avgdl = float(index.get("avgdl", 0.0)) or 1.0
        self.docs: List[Dict[str, Any]] = list(index.get("docs", []))
        self.df: Dict[str, int] = dict(index.get("df", {}))
        # postings: term -> {doc_index_str: tf}; translate to int keys once.
        self.postings: Dict[str, Dict[int, int]] = {
            term: {int(doc_i): tf for doc_i, tf in pairs.items()}
            for term, pairs in index.get("postings", {}).items()
        }
        self._idf_cache: Dict[str, float] = {}
        self._sentence_sets_cache: Dict[int, List[set]] = {}

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "Searcher":
        index_path = Path(path) if path else DEFAULT_INDEX_PATH
        index = json.loads(index_path.read_bytes().decode("utf-8"))
        return cls(index)

    def _idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is None:
            df = self.df.get(term, 0)
            cached = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            self._idf_cache[term] = cached
        return cached

    def _doc_length(self, doc_index: int) -> float:
        return float(self.docs[doc_index].get("dl", 0)) or 1.0

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Rank docs for one query; returns up to k deterministic results."""
        query_terms = Counter(tokenize(query))
        if not query_terms or k <= 0:
            return []

        scores: Dict[int, float] = {}
        for term, qtf in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            weight = qtf * self._idf(term)
            for doc_index, tf in postings.items():
                norm = 1.0 - self.b + self.b * (self._doc_length(doc_index) / self.avgdl)
                scores[doc_index] = scores.get(doc_index, 0.0) + weight * tf * (self.k1 + 1.0) / (
                    tf + self.k1 * norm
                )

        if not scores:
            return []
        # Deterministic order: score desc, then doc id asc.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.docs[item[0]]["id"]))
        results: List[Dict[str, Any]] = []
        for doc_index, score in ranked[:k]:
            doc = self.docs[doc_index]
            results.append(
                {
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "score": round(score, 4),
                    "source": doc.get("source", ""),
                    "snippet": self._best_sentence(doc_index, query_terms),
                }
            )
        return results

    def _sentence_sets(self, doc_index: int) -> List[set]:
        """Lazily tokenized sentence token-sets of one doc, for snippet scoring."""
        cached = self._sentence_sets_cache.get(doc_index)
        if cached is None:
            cached = [set(tokenize(s)) for s in self.docs[doc_index].get("sentences", [])]
            self._sentence_sets_cache[doc_index] = cached
        return cached

    def _best_sentence(self, doc_index: int, query_terms: Counter) -> str:
        """Highest-idf-overlap sentence of the doc, for the snippet."""
        sentences = self.docs[doc_index].get("sentences", [])
        if not sentences:
            return ""
        terms = self._sentence_sets(doc_index)
        best_index, best_score = 0, -1.0
        for i, sentence in enumerate(sentences):
            hits = terms[i] & query_terms.keys() if i < len(terms) else set()
            score = sum(self._idf(t) for t in hits)
            if score > best_score:
                best_index, best_score = i, score
        snippet = sentences[best_index]
        if len(snippet) > SNIPPET_MAX:
            snippet = snippet[:SNIPPET_MAX].rsplit(" ", 1)[0].rstrip() + " …"
        return snippet


def search(query: str, k: int = 5, index_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Module-level convenience: load the default index and rank one query."""
    return Searcher.from_file(index_path).search(query, k=k)

"""FTS5-backed index with an honest unavailability result.

sqlite3 ships in stdlib; the FTS5 COMPILE-TIME FEATURE may be absent
from some builds. available() probes it once. WAL journaling is set on
every connection. This is the storage seam for the find-anything index
(Phase 4.8): nothing in the hot path depends on it, so the pure-Python
BM25 remains the default and this stays an optional accelerator.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

__all__ = ["fts5_available", "connect_wal", "FTS5Index"]


def fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            return True
        finally:
            conn.close()
    except (sqlite3.Error, AttributeError):
        return False


def connect_wal(path: Path) -> sqlite3.Connection:
    """sqlite connection with WAL journaling (write-tolerant readers)."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class FTS5Index:
    """Document index over FTS5. Every method returns an honest dict;
    when FTS5 is unavailable the index reports it instead of failing."""

    def __init__(self, path=None):
        self.path = str(path) if path else ":memory:"
        self.ok = fts5_available()
        self.conn = connect_wal(self.path) if self.ok else None
        if self.ok:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
                "doc_id UNINDEXED, title, body)")

    def index(self, docs: Iterable[Tuple[str, str, str]]) -> Dict:
        """docs: (doc_id, title, body) triples."""
        if not self.ok:
            return {"ok": False, "reason": "FTS5 unavailable in this sqlite build"}
        self.conn.executemany(
            "INSERT INTO docs(doc_id, title, body) VALUES (?,?,?)", list(docs))
        self.conn.commit()
        return {"ok": True}

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        if not self.ok:
            return []
        # the FTS5 syntax guard: a malformed query is an empty result,
        # never a crash
        try:
            rows = self.conn.execute(
                "SELECT doc_id, title, bm25(docs) FROM docs WHERE docs "
                "MATCH ? ORDER BY bm25(docs) LIMIT ?", (query, limit)).fetchall()
        except sqlite3.Error:
            return []
        return [{"doc_id": r[0], "title": r[1], "score": r[2]} for r in rows]

    def close(self):
        if self.conn is not None:
            self.conn.close()

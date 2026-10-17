"""shellkb.howto — offline how-to answers from a self-authored cheat
sheet (B8).

"how do I take a screenshot of just a region?" deserves an answer that
works on a plane. This module is BM25 over a small, committed,
self-authored corpus of task -> command-sequence entries (40 of them
at birth), plus the EXISTING time-expression parser (cortex.nlhistory)
for the temporal half of a question: "how do I read what the shell was
logging last night" parses the window and ranks the log-reading entry
up, reporting the exact span it understood (21:00 -> 05:00 for "last
night" — midnight-spanning, like nlhistory's own grammar).

License, stated once and honestly: every entry is ORIGINAL text
authored for this repository and dedicated CC0 — nothing is lifted
from man pages, wikis, or third-party cheat sheets, so there is no
third-party license to verify. The corpus is a committed artifact;
adding entries is a reviewed commit, never a runtime scrape.

Scoring: plain BM25 (k1=1.5, b=0.75) over the entry's task + keyword +
step + note tokens; an entry flagged ``temporal`` gets its score
boosted when the query parses a time window (and the window is echoed
back verbatim). Deterministic tie-breaks by (score, id).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CORPUS_PATH", "answer", "render_lines"]

CORPUS_PATH = Path(__file__).resolve().parent / "howto_corpus.json"

_K1 = 1.5
_B = 0.75
_TOP_K = 5
_TEMPORAL_BOOST = 1.35  # a window was asked for and this entry cares
_SCORE_FLOOR = 4.0      # below this the match is noise, honestly reported


def _load_corpus() -> Dict[str, Any]:
    try:
        return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": [], "license": "corpus missing"}


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """A deliberately tiny suffix stripper (len>3; two bounded rounds)
    so 'logging' meets 'log' and 'recordings' meets 'record'. Doubled
    final consonants collapse after a strip ('logg' -> 'log'), except
    the legitimate ones (ll/ss/ff/zz: 'missed' stays 'miss'). No Porter
    machinery — corpus and query get the SAME rule, which is all BM25
    needs for a 40-document collection."""
    for _ in range(2):
        for suffix in _SUFFIXES:
            if len(word) > 3 and word.endswith(suffix) and \
                    len(word) - len(suffix) >= 3:
                if suffix == "s" and word[-2:] in ("ll", "ss", "ff", "zz"):
                    continue  # a legitimate double, not a plural artifact
                if suffix == "es" and (len(word) < 3 or
                                       word[-3] not in "sxzh"):
                    continue  # 'schemes' strips 's', not the stem's own 'e'
                word = word[:len(word) - len(suffix)]
                if len(word) >= 2 and word[-1] == word[-2] and \
                        word[-2:] not in ("ll", "ss", "ff", "zz"):
                    word = word[:-1]
                break
        else:
            break  # no suffix applied this round: done
    return word


def _tokens(text: str) -> List[str]:
    return [_stem(w) for w in _TOKEN_RE.findall((text or "").lower())]


class _BM25:
    """The standard Okapi BM25 over a tiny corpus — 30 lines, no
    dependencies, deterministic. Built per call (40 entries: the build
    is microseconds; the module never caches user state)."""

    def __init__(self, docs: List[Tuple[str, List[str]]]) -> None:
        self.ids = [did for did, _ in docs]
        self.docs = [toks for _, toks in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) \
            if self.n else 0.0
        self.df: Dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.tfs = [self._tf(doc) for doc in self.docs]

    @staticmethod
    def _tf(doc: List[str]) -> Dict[str, int]:
        tf: Dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        return tf

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: List[str], index: int) -> float:
        s = 0.0
        doc = self.tfs[index]
        dl = len(self.docs[index])
        norm = _K1 * (1.0 - _B + _B * dl / self.avgdl) if self.avgdl \
            else _K1
        for term in query:
            f = doc.get(term, 0)
            if not f:
                continue
            s += self.idf(term) * (f * (_K1 + 1.0)) / (f + norm)
        return s

    def search(self, query: List[str], k: int = _TOP_K) \
            -> List[Tuple[str, float]]:
        scored = [(self.ids[i], self.score(query, i))
                  for i in range(self.n)]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]


def _entry_tokens(entry: Dict[str, Any]) -> List[str]:
    keywords = entry.get("keywords", "")
    if isinstance(keywords, list):
        keywords = " ".join(keywords)
    parts = [entry.get("task", ""), keywords]
    parts.extend(entry.get("steps", []))
    parts.append(entry.get("notes", ""))
    return _tokens(" ".join(parts))


def parse_window(text: str, now=None) -> Tuple[Optional[Dict[str, Any]],
                                               Optional[str]]:
    """The temporal half of the question, via the EXISTING parser
    (cortex.nlhistory — a caller, not a second implementation). Returns
    (window-dict | None, note). ``now`` defaults to nlhistory's own
    fixed epoch so parsing stays deterministic."""
    from datetime import datetime
    from assistant.core import nlhistory

    try:
        q = nlhistory.parse_query(text, now=now or datetime(2026, 1, 1, 12))
    except Exception:  # the reuse must never break the answer path
        return None, None
    window = getattr(q, "window", None)
    if not window:
        return None, None
    lo, hi = window
    return {"from": lo.isoformat(), "to": hi.isoformat()}, \
        f"applies to {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M}"


def answer(query: str, k: int = _TOP_K, now=None) -> Dict[str, Any]:
    """The how-to answer: parsed window, ranked entries with their
    steps (each step labeled SUGGESTED_NOT_EXECUTED), honest empty case."""
    corpus = _load_corpus()
    entries = corpus.get("entries", [])
    window, window_note = parse_window(query, now=now)
    qtoks = _tokens(query)
    results: List[Dict[str, Any]] = []
    if entries and qtoks:
        index = _BM25([(e["id"], _entry_tokens(e)) for e in entries])
        for eid, score in index.search(qtoks, k=max(k, 8)):
            if score < _SCORE_FLOOR:
                continue  # honest floor: a weak BM25 match is noise
            entry = next(e for e in entries if e["id"] == eid)
            s = score
            if window and entry.get("temporal"):
                s *= _TEMPORAL_BOOST
            results.append({
                "id": eid,
                "task": entry["task"],
                "steps": [f"SUGGESTED_NOT_EXECUTED: {step}"
                          for step in entry["steps"]],
                "notes": entry.get("notes", ""),
                "see": entry.get("see", []),
                "score": round(s, 4),
                "temporal": bool(entry.get("temporal")),
            })
        results.sort(key=lambda r: (-r["score"], r["id"]))
        results = results[:k]
    return {
        "query": query,
        "window": window,
        "window_note": window_note,
        "license": corpus.get("license", ""),
        "n_entries": len(entries),
        "results": results,
        "empty_note": None if results else
        "no entry matched; the corpus is deliberately small and honest — "
        "try the keywords an entry would use ('wallpaper', 'scheme', "
        "'screenshot', 'record', 'update', 'undo', ...)",
    }


def render_lines(data: Dict[str, Any]) -> List[str]:
    lines = [f"how-to (offline, CC0 self-authored corpus, "
             f"{data['n_entries']} entries) — nothing is executed"]
    if data.get("window_note"):
        lines.append(f"  time: {data['window_note']}")
    if not data["results"]:
        lines.append(f"  ({data.get('empty_note')})")
        return lines
    for r in data["results"]:
        lines.append(f"  [{r['id']}] {r['task']}  (score {r['score']})")
        for step in r["steps"]:
            lines.append(f"    {step}")
        if r.get("notes"):
            lines.append(f"    note: {r['notes']}")
        if r.get("see"):
            lines.append(f"    see also: {', '.join(r['see'])}")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: howto \"your question\"")
        return 2
    data = answer(" ".join(argv))
    for line in render_lines(data):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

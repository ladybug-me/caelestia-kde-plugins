"""Offline BM25 indexer for the assistant retrieval corpus.

This module is used by the offline build step (`python3 -m assistant.capabilities.retrieval.build`)
and by the corpus tools. Runtime search (search.py) only reads the JSON index
this module writes — it never parses the corpus markdown.

Corpus doc format (strict):
    id: DOC-TS-01
    title: 1. Build & Compilation Issues
    source: docs/TROUBLESHOOTING.md — section 1
    tags: build, compile, cmake
    synonyms: install fails cmake missing   (optional)

    <body — everything after the first blank line>

Tokenization: lowercase, split on non-alphanumeric, keep tokens of >= 2
characters, drop a small hardcoded English stopword list. Title tokens are
weighted x3 and tag/synonym tokens x2 by repetition in the token stream, so
header text counts more than body text without any extra math.

Index JSON layout (deterministic; sort_keys):
    version, params {k1, b}, avgdl,
    docs  [{id, title, source, path, dl, sentences[]}]
    df    {term: doc frequency}
    postings {term: {doc_index: term frequency}}
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import CORPUS_DIR, DEFAULT_INDEX_PATH

K1 = 1.5
B = 0.75
VERSION = 1

TITLE_WEIGHT = 3
TAG_WEIGHT = 2

HEADER_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*): (.*)$")
TOKEN_RE = re.compile(r"[^0-9a-z]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Tiny hardcoded English stopword list (no external data files).
STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been before being both but by can could did
    do does doing down during each few for from further had has have having he her here hers him his how i if in
    into is it its itself just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, drop 1-char tokens and stopwords."""
    tokens: List[str] = []
    for raw in TOKEN_RE.split(text.lower()):
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        tokens.append(raw)
    return tokens


def parse_corpus_doc(path: Path) -> Tuple[Dict[str, str], str]:
    """Parse one strict 'key: value' header block + body; raise on malformed."""
    lines = path.read_text(encoding="utf-8").split("\n")
    meta: Dict[str, str] = {}
    idx = 0
    for idx, line in enumerate(lines):
        if line.strip() == "":
            break
        match = HEADER_KEY_RE.match(line)
        if not match:
            raise ValueError(f"{path.name}: malformed header line {line!r}")
        meta[match.group(1)] = match.group(2).strip()
    else:
        raise ValueError(f"{path.name}: header block never terminated by a blank line")
    for key in ("id", "title", "source", "tags"):
        if key not in meta:
            raise ValueError(f"{path.name}: header missing required key {key!r}")
    body = "\n".join(lines[idx + 1:]).strip()
    return meta, body


def split_sentences(body: str) -> List[str]:
    """Split a body into sentence-ish strings (newline-aware) for snippets."""
    sentences: List[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= 240:
            sentences.append(line)
            continue
        sentences.extend(s.strip() for s in SENTENCE_SPLIT_RE.split(line) if s.strip())
    return [s[:400] for s in sentences]


def _weighted_stream(meta: Dict[str, str], body: str) -> List[str]:
    tokens: List[str] = []
    tokens.extend(tokenize(meta.get("title", "")) * TITLE_WEIGHT)
    header_extra = f"{meta.get('tags', '')} {meta.get('synonyms', '')}"
    tokens.extend(tokenize(header_extra) * TAG_WEIGHT)
    tokens.extend(tokenize(body))
    return tokens


def load_corpus(corpus_dir: Optional[Path] = None) -> List[Tuple[Dict[str, str], str, Path]]:
    """Load every corpus/*.md sorted by filename (deterministic doc order)."""
    directory = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    docs: List[Tuple[Dict[str, str], str, Path]] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_corpus_doc(path)
        docs.append((meta, body, path))
    if not docs:
        raise ValueError(f"no corpus markdown found in {directory}")
    return docs


def build_index(docs: List[Tuple[Dict[str, str], str, Path]]) -> Dict[str, object]:
    """Build the BM25 index structure over pre-loaded corpus docs."""
    postings: Dict[str, Dict[int, int]] = {}
    df: Dict[str, int] = {}
    doc_entries: List[Dict[str, object]] = []
    total_tokens = 0

    for doc_index, (meta, body, path) in enumerate(docs):
        stream = _weighted_stream(meta, body)
        tf = Counter(stream)
        for term, count in tf.items():
            postings.setdefault(term, {})[doc_index] = count
        for term in tf:
            df[term] = df.get(term, 0) + 1
        total_tokens += len(stream)
        doc_entries.append(
            {
                "id": meta["id"],
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "path": str(path.name),
                "dl": len(stream),
                "sentences": split_sentences(body),
            }
        )

    n_docs = len(docs)
    avgdl = total_tokens / n_docs if n_docs else 0.0
    return {
        "version": VERSION,
        "params": {"k1": K1, "b": B},
        "n_docs": n_docs,
        "avgdl": avgdl,
        "docs": doc_entries,
        "df": df,
        # JSON keys must be strings; keep doc indices as strings in postings.
        "postings": {term: {str(i): count for i, count in pairs.items()} for term, pairs in postings.items()},
    }


def serialize_index(index: Dict[str, object]) -> bytes:
    """Deterministic JSON bytes for an index (sorted keys, stable floats)."""
    text = json.dumps(index, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def write_index(index: Dict[str, object], path: Optional[Path] = None) -> Path:
    target = Path(path) if path else DEFAULT_INDEX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(serialize_index(index))
    return target


def idf(df: int, n_docs: int) -> float:
    """Standard BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5))."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

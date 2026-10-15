"""Build the retrieval index from the corpus (offline step).

Usage:
    python3 -m assistant.capabilities.retrieval.build [--corpus DIR] [--out PATH] [--quiet]

Parses assistant/retrieval/corpus/*.md and writes index/bm25.json. Runtime
search never parses the corpus; it reads only the JSON this writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import CORPUS_DIR, DEFAULT_INDEX_PATH
from .indexer import build_index, load_corpus, serialize_index, write_index


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m assistant.capabilities.retrieval.build",
        description="Build the offline BM25 index for the retrieval corpus.",
    )
    parser.add_argument(
        "--corpus", default=str(CORPUS_DIR), help="corpus directory (default: assistant/retrieval/corpus)"
    )
    parser.add_argument("--out", default=str(DEFAULT_INDEX_PATH), help="index output path (default: index/bm25.json)")
    parser.add_argument("--quiet", action="store_true", help="only print the output path")
    args = parser.parse_args(argv)

    try:
        docs = load_corpus(Path(args.corpus))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    index = build_index(docs)
    payload = serialize_index(index)
    target = write_index(index, Path(args.out))

    if args.quiet:
        print(target)
        return 0
    unique_terms = len(index["df"])
    total_tokens = sum(doc["dl"] for doc in index["docs"])  # type: ignore[index]
    print(f"indexed {len(docs)} corpus docs, {unique_terms} unique terms, {total_tokens} tokens")
    print(f"index: {target} ({len(payload)} bytes)")
    print("docs:")
    for doc in index["docs"]:  # type: ignore[union-attr]
        print(f"  {doc['id']:16s} {doc['title']}")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

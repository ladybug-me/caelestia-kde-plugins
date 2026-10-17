"""Layer 2: local retrieval over the repo's own docs and resolved issues.

When the deterministic rules (Layer 1) do not match, the assistant searches
this corpus for the closest real past resolution and returns it with a clear
banner: these are pointers, not verified diagnoses.

Everything here is offline, stdlib-only, and index-driven at runtime: the
corpus markdown is parsed only by the offline build step
(`python3 -m assistant.capabilities.retrieval.build`); search reads index/bm25.json only.
"""

from pathlib import Path

RETRIEVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = RETRIEVAL_DIR.parents[1] / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DEFAULT_INDEX_PATH = DATA_DIR / "index" / "bm25.json"

__all__ = ["CORPUS_DIR", "DEFAULT_INDEX_PATH", "RETRIEVAL_DIR"]

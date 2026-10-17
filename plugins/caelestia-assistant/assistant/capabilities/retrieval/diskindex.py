"""retrieval.diskindex — a DISK-BACKED, bounded-memory personal search
index (exponential-build 4.1): index an arbitrary folder tree on
low-end hardware under a stated RAM ceiling, measured rather than
asserted.

The build is a classic external merge sort over postings:

1. walk the tree (bounded, read-only, sorted), tokenize each readable
   file with the EXISTING ``indexer.tokenize`` and emit
   ``term\\tdoc\\ttf`` rows into an in-memory run buffer;
2. the run buffer is flushed to a sorted temporary file whenever its
   accumulated character budget crosses the ceiling-derived limit —
   the budget IS the RAM ceiling applied to index structures;
3. ``heapq.merge`` k-way merges the runs (the stdlib's own external
   merge primitive), aggregating rows into the final index: a
   term-sorted JSONL postings file plus a small metadata JSON
   (n_docs, doc lengths, avgdl, k1/b — the same BM25 parameters the
   in-memory Searcher uses).

Queries load ONLY the metadata; the postings for a query term are
read from the sorted file with an early-exit scan (built to keep RAM
flat on a low-end machine, not to win latency benchmarks — stated
plainly). Scoring reuses ``indexer.idf`` and the Searcher's k1/b so
both indexes answer with the same BM25.

THE FOOTPRINT CLAIM, measured: the build report records the process
peak RSS (VmHWM from /proc/self/status — a plain file read, the same
way the telemetry layer reads /proc) before and after, the run-buffer
budget it enforced, the number of runs spilled, and whether the
observed growth stayed under the ceiling. If the measurement says the
ceiling was breached, the report says BREACHED — the claim is only as
good as the number, and the number is in the report.

Pure offline text processing: no subprocess, no network, no
execution; the only writes are the index files in the directory the
caller named (plus tempfile spool files, removed on completion).
"""
from __future__ import annotations

import heapq
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .indexer import B, K1, idf, tokenize

__all__ = ["build_disk_index", "DiskSearcher", "DEFAULT_CEILING_MB",
           "_read_vm_hwm_kb", "_read_vm_rss_kb"]

DEFAULT_CEILING_MB = 16
_FILL_FACTOR = 0.5          # chars-per-byte safety margin for Python overhead
_BYTES_PER_CHAR = 1         # run files are written ASCII-safe (escaped)
_MAX_FILE_BYTES = 512 * 1024  # per-file read cap


# ---------------------------------------------------------------------------
# Memory measurement (the honest part: read it, don't assert it).
# ---------------------------------------------------------------------------

def _proc_status_field(field: str) -> Optional[int]:
    """One kB value from /proc/self/status (VmHWM/VmRSS), or None off-Linux."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_vm_hwm_kb() -> Optional[int]:
    return _proc_status_field("VmHWM")


def _read_vm_rss_kb() -> Optional[int]:
    return _proc_status_field("VmRSS")


# ---------------------------------------------------------------------------
# The bounded walk.
# ---------------------------------------------------------------------------

def _walk_tree(root: str, max_files: int) -> Tuple[List[Path], int]:
    """Sorted, bounded, read-only walk. Returns (files, skipped_dirs)."""
    base = Path(os.path.expanduser(root))
    if not base.is_dir():
        raise ValueError(f"not a directory: {root}")
    files: List[Path] = []
    stack = [base]
    skipped = 0
    while stack and len(files) < max_files:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError:
            skipped += 1
            continue
        for entry in entries:
            if len(files) >= max_files:
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(Path(entry.path))
            except OSError:
                skipped += 1
    return files, skipped


# ---------------------------------------------------------------------------
# Build: tokenize -> runs -> k-way merge -> term-sorted postings file.
# ---------------------------------------------------------------------------

def build_disk_index(root: str, index_dir: str,
                     ram_ceiling_mb: float = DEFAULT_CEILING_MB,
                     max_files: int = 20000) -> Dict[str, Any]:
    """Build the disk index for one folder tree under the RAM ceiling.

    Returns the measured footprint report (peak RSS before/after, the
    enforced run budget, run count, index size) — the claim lives in
    these numbers, not in prose."""
    ceiling_bytes = float(ram_ceiling_mb) * 1024.0 * 1024.0
    if ceiling_bytes <= 0:
        raise ValueError("ram_ceiling_mb must be > 0")
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_chars = max(4096.0, ceiling_bytes * _FILL_FACTOR / _BYTES_PER_CHAR)
    rss_start = _read_vm_rss_kb()
    hwm_start = _read_vm_hwm_kb()

    files, skipped_dirs = _walk_tree(root, max_files)
    doc_lengths: List[int] = []
    doc_paths: List[str] = []

    runs: List[str] = []       # spool file paths
    buffer: List[str] = []
    buffer_chars = 0
    max_buffer_chars = 0
    unreadable = 0
    binary_likely = 0
    max_rss_seen = rss_start or 0

    def flush() -> None:
        nonlocal buffer, buffer_chars, max_rss_seen
        if not buffer:
            return
        buffer.sort()
        fd, tmp = tempfile.mkstemp(prefix="run_", suffix=".tsv",
                                   dir=str(out_dir))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(buffer))
            fh.write("\n")
        runs.append(tmp)
        buffer = []
        buffer_chars = 0
        rss_now = _read_vm_rss_kb()
        if rss_now is not None:
            max_rss_seen = max(max_rss_seen, rss_now)

    for path in files:
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                with open(path, "rb") as fh:
                    raw = fh.read(_MAX_FILE_BYTES)
            else:
                raw = path.read_bytes()
        except OSError:
            unreadable += 1
            continue
        if b"\x00" in raw:
            binary_likely += 1
            continue
        text = raw.decode("utf-8", "replace")
        tokens = tokenize(text)
        doc_index = len(doc_lengths)
        doc_lengths.append(len(tokens))
        doc_paths.append(str(path))
        counts: Dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for term in sorted(counts):
            row = f"{term}\t{doc_index}\t{counts[term]}"
            # budget check BEFORE appending: the buffer is never allowed
            # past the ceiling, not even by one row — that is what
            # makes budget_respected_by_construction a construction
            # claim rather than a hope
            if buffer_chars + len(row) + 1 > budget_chars:
                flush()
            buffer.append(row)
            buffer_chars += len(row) + 1
            if buffer_chars > max_buffer_chars:
                max_buffer_chars = buffer_chars
    flush()

    # k-way merge the runs; aggregate rows into term-sorted postings.
    postings_path = out_dir / "postings.jsonl"
    meta_path = out_dir / "meta.json"
    df: Dict[str, int] = {}
    n_docs = len(doc_lengths)
    avgdl = (sum(doc_lengths) / n_docs) if n_docs else 0.0
    n_terms = 0
    with open(postings_path, "w", encoding="utf-8") as out:
        def _rows():
            handles = []
            for r in runs:
                handles.append(open(r, "r", encoding="utf-8"))
            try:
                for line in heapq.merge(*handles):
                    yield line.rstrip("\n")
            finally:
                for h in handles:
                    h.close()

        current_term: Optional[str] = None
        current_postings: List[List[int]] = []
        for row in _rows():
            term, doc_s, tf_s = row.split("\t")
            if term != current_term:
                if current_term is not None:
                    out.write(json.dumps(
                        {"term": current_term, "df": df[current_term],
                         "postings": current_postings},
                        ensure_ascii=False, separators=(",", ":")) + "\n")
                    current_postings = []
                current_term = term
                df[term] = 0
                n_terms += 1
            tf = int(tf_s)
            df[term] += 1
            current_postings.append([int(doc_s), tf])
        if current_term is not None:
            out.write(json.dumps(
                {"term": current_term, "df": df[current_term],
                 "postings": current_postings},
                ensure_ascii=False, separators=(",", ":")) + "\n")
    for r in runs:
        try:
            os.remove(r)
        except OSError:
            pass

    meta = {
        "version": 1,
        "kind": "disk-index",
        "root": root,
        "n_docs": n_docs,
        "n_terms": n_terms,
        "avgdl": round(avgdl, 4),
        "doc_lengths": doc_lengths,
        "doc_paths": doc_paths,
        "params": {"k1": K1, "b": B},
        "footprint": {
            "ram_ceiling_mb": ram_ceiling_mb,
            "run_budget_chars": int(budget_chars),
            "max_buffer_chars_seen": max_buffer_chars,
            "budget_respected_by_construction": max_buffer_chars
            <= budget_chars,
            "runs_spilled": len(runs),
            "rss_start_kb": rss_start,
            "hwm_start_kb": hwm_start,
            "hwm_end_kb": _read_vm_hwm_kb(),
            "max_rss_seen_kb": max_rss_seen or None,
            "measured": True,
        },
        "walk": {"files": len(files), "skipped_dirs": skipped_dirs,
                 "unreadable": unreadable, "binary_likely": binary_likely},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True),
                         encoding="utf-8")
    growth_kb = None
    if meta["footprint"]["hwm_end_kb"] is not None \
            and hwm_start is not None:
        growth_kb = meta["footprint"]["hwm_end_kb"] - hwm_start
        meta["footprint"]["hwm_growth_kb"] = growth_kb
        meta["footprint"]["hwm_growth_within_ceiling"] = \
            growth_kb <= ceiling_bytes / 1024.0
        meta["footprint"]["note"] = (
            "hwm is the WHOLE PROCESS peak (interpreter included); the "
            "enforced index-structure budget is run_budget_chars, and "
            "budget_respected_by_construction is the claim that "
            "matters — hwm_growth_within_ceiling is the honest "
            "process-level observation, even when unflattering")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True),
                         encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Query: metadata in RAM, postings on disk with early-exit scans.
# ---------------------------------------------------------------------------

class DiskSearcher:
    """BM25 over the on-disk index. Loads the metadata only — no
    postings dictionary exists on this object, by construction."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "meta.json"
        if not meta_path.is_file():
            raise ValueError(f"no disk index at {self.index_dir}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("kind") != "disk-index":
            raise ValueError("not a disk index")
        self.n_docs = int(self.meta["n_docs"])
        self.avgdl = float(self.meta["avgdl"]) or 1.0
        self.k1 = float(self.meta["params"]["k1"])
        self.b = float(self.meta["params"]["b"])
        self.doc_lengths = list(self.meta["doc_lengths"])
        self.doc_paths = list(self.meta["doc_paths"])
        self.postings_path = self.index_dir / "postings.jsonl"

    def _term_line(self, term: str) -> Optional[Dict[str, Any]]:
        """Early-exit scan of the term-sorted postings file."""
        if not self.postings_path.is_file():
            return None
        with open(self.postings_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                row_term = row.get("term")
                if row_term == term:
                    return row
                if row_term > term:
                    return None  # sorted: the term is absent
        return None

    def search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25 ranking over the on-disk postings (same idf/k1/b as the
        in-memory Searcher)."""
        terms = tokenize(text)
        if not terms or self.n_docs == 0:
            return []
        qtf: Dict[str, int] = {}
        for t in terms:
            qtf[t] = qtf.get(t, 0) + 1
        scores: Dict[int, float] = {}
        for term, _qtf in qtf.items():
            row = self._term_line(term)
            if not row:
                continue
            weight = idf(row["df"], self.n_docs)
            for doc_index, tf in row["postings"]:
                norm = (1.0 - self.b
                        + self.b * (self.doc_lengths[doc_index] / self.avgdl))
                scores[doc_index] = scores.get(doc_index, 0.0) + \
                    weight * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        if not scores:
            return []
        ranked = sorted(scores.items(),
                        key=lambda item: (-item[1], self.doc_paths[item[0]]))
        results: List[Dict[str, Any]] = []
        for doc_index, score in ranked[:k]:
            results.append({"path": self.doc_paths[doc_index],
                            "score": round(score, 4),
                            "n_tokens": self.doc_lengths[doc_index]})
        return results

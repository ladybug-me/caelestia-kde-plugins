"""genius.fsbrain — the filesystem second-brain layer (read-only analysis).

Four classical analyses over the user's own directories, generalizing
machinery the assistant already ships (nothing here is a second
implementation of anything):

1. ``staleness_report`` — directory staleness/entropy scoring. The
   frecency math of ``sysintel.analyze_history`` (frequency + recency
   boost ``0.5 ** (age / half_life)``) generalized from shell-history
   event positions to filesystem ``stat()`` event times (mtime = last
   write, atime = last read): each observed event contributes its
   half-life-decayed weight, so a file touched yesterday outranks a
   file touched last quarter at equal "event count". Directory-level
   Shannon entropy over the file-category distribution scores how much
   of a mixed pile each directory is. Read-only tidy-style REPORT —
   propose-only, never deletes, never moves (``brain.tidy.apply_moves``
   stays the only mover, behind its own journal + rollback).

2. ``metadata_near_duplicates`` — near-duplicate files by METADATA
   fingerprints: each file's (name words, suffix, size bucket, mtime
   day) is SimHashed with ``scan.simhash`` (64-bit, digit-masked) and
   candidates are banded by 16-bit blocks before Hamming verification
   (Manku, Das & Sarma 2007, "Detecting Near-Duplicates for Web
   Crawling", §5.2's banding trick — the same operating point
   ``scan.simhash.DEFAULT_THRESHOLD = 3`` documents). Read-only.

3. ``knowledge_graph`` — a term-overlap knowledge graph over notes and
   docs: keywords per document (RAKE, ``genius.language.rake_keywords``),
   an unweighted term co-occurrence graph built through
   ``brain.personal.graph.Graph`` (its PageRank — Page et al. 1999 —
   and its label-propagation communities — Raghavan, Albert & Kumara
   2007 — are reused verbatim, not reimplemented), plus document
   relatedness by term-set Jaccard overlap. Classical co-occurrence
   only: no embeddings, no model file, nothing trained.

4. ``infer_filetype`` — byte-signature file-type inference: an
   Aho-Corasick automaton (``scan.ac.Automaton`` — Cormick 1975 /
   the module's own docstring) over the classic magic-byte table,
   offset-anchored (a match counts only at its declared offset); when
   the table misses, an online-correctable multinomial Naive Bayes
   (``brain.naive_bayes.NaiveBayes``) over byte-histogram features
   ranks the types the USER has corrected before — corrections persist
   in the brain state (key ``fsbrain_nb``, bounded), the same
   atomic-save discipline every learner uses.

Event-driven watching (the §5 invariant): this module ships NO watcher
and no sleep-loop of any kind — see the package rationale (see module docstrings)'s known-gaps entry for
the measured reason (select() on a directory fd is always "ready" on
this platform, and ctypes — the only stdlib route to inotify(7) — is
rejected by the behavioral safety tests). Every scan here is a
user-invoked one-shot; steady-state RSS when idle is zero because
nothing is resident.

Safety spine: read-only analysis, propose-only reports, bounded walks
(max_files cap, same discipline as sysintel's find_duplicates), the one
write is learned-state persistence through the brain's atomic save.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from assistant.capabilities.brain.naive_bayes import NaiveBayes
from assistant.capabilities.brain.personal.graph import Graph
from assistant.capabilities.brain.textmine import keywords as tfidf_keywords
from assistant.capabilities.scan.ac import Automaton
from assistant.capabilities.scan.simhash import hamming, simhash
from .language import rake_keywords

__all__ = [
    "staleness_report", "metadata_near_duplicates", "knowledge_graph",
    "infer_filetype", "load_filetype_classifier", "record_correction",
    "append_correction", "DEFAULT_HALFLIFE_DAYS", "DOCS_KEY",
]

# The persisted shape of the filetype corrections: a bounded list of
# correction DOCS (byte features + human label), not a trained model
# blob — reviewable line by line, the same posture as every other
# learned state in the assistant.
DOCS_KEY = "fsbrain_docs"
_NB_MAX_DOCS = 500

# ---------------------------------------------------------------------------
# Shared walk (bounded, read-only).
# ---------------------------------------------------------------------------

DEFAULT_HALFLIFE_DAYS = 30.0
_MAX_FILES = 20000

# Suffix → coarse category for entropy scoring (tidy.py's _category_for's
# spirit, extended for entropy purposes; categories are report labels,
# not move targets).
_CATEGORIES: Dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".svg": "image", ".bmp": "image",
    ".mp4": "video", ".mkv": "video", ".webm": "video", ".avi": "video",
    ".mp3": "audio", ".flac": "audio", ".ogg": "audio", ".wav": "audio",
    ".pdf": "document", ".md": "document", ".txt": "document",
    ".doc": "document", ".docx": "document", ".odt": "document",
    ".epub": "document",
    ".zip": "archive", ".tar": "archive", ".gz": "archive",
    ".bz2": "archive", ".xz": "archive", ".7z": "archive", ".rar": "archive",
    ".iso": "archive",
    ".py": "code", ".sh": "code", ".js": "code", ".ts": "code", ".c": "code",
    ".cpp": "code", ".h": "code", ".rs": "code", ".go": "code",
    ".json": "code", ".yml": "code", ".yaml": "code", ".toml": "code",
    ".desktop": "code", ".qml": "code",
    ".deb": "package", ".rpm": "package", ".pkg": "package", ".apk": "package",
    ".ttf": "font", ".otf": "font",
    ".sqlite": "database", ".db": "database",
}


def _walk_files(root: str, max_files: int = _MAX_FILES) -> List[Path]:
    """Bounded read-only walk (sysintel's find_duplicates discipline)."""
    base = Path(os.path.expanduser(root))
    if not base.exists():
        raise ValueError(f"no such directory: {root}")
    out: List[Path] = []
    if base.is_file():
        return [base]
    stack = [base]
    while stack and len(out) < max_files:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if len(out) >= max_files:
                break
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                out.append(Path(entry.path))
    return out


def _category_of(path: Path) -> str:
    return _CATEGORIES.get(path.suffix.lower(), "other")


def _shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (bits) over a category distribution — 0 for a
    uniform pile of one kind, log2(k) for k evenly-mixed kinds."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# 1. Staleness / entropy report (frecency generalized to stat() data).
# ---------------------------------------------------------------------------


def _decay(age_days: float, half_life_days: float) -> float:
    """sysintel.analyze_history's recency boost: 0.5 ** (age / half_life)."""
    return 0.5 ** (max(0.0, age_days) / max(1e-9, half_life_days))


def staleness_report(root: str, top: int = 15,
                     half_life_days: float = DEFAULT_HALFLIFE_DAYS,
                     max_files: int = _MAX_FILES,
                     now: Optional[datetime] = None) -> Dict[str, Any]:
    """Read-only tidy-style report: warmest/coldest files by generalized
    frecency plus the most-mixed directories by category entropy.

    Frecency generalization (sysintel.py:107's ``1 + 0.5**(age/(n*0.25))``
    from shell-history positions to stat() times): a file's activity is
    the sum over its OBSERVABLE events — mtime (last write) and atime
    (last read) — of ``0.5 ** (age_days / half_life_days)``; each event
    also carries the sysintel "+1" count term, so two recent events
    outrank one. Files with no readable stat fall through with a note,
    never a crash. NOTHING here deletes, moves, or proposes a move —
    the tidy module's journal/rollback path stays the only mover.
    """
    now = now or datetime.now(timezone.utc)
    files = _walk_files(root, max_files=max_files)
    rows: List[Dict[str, Any]] = []
    unreadable = 0
    dir_counts: Dict[str, Dict[str, int]] = {}
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            unreadable += 1
            continue
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        atime = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)
        m_age = (now - mtime).total_seconds() / 86400.0
        a_age = (now - atime).total_seconds() / 86400.0
        # one write event + one read event, each half-life weighted
        activity = (1.0 + _decay(m_age, half_life_days)) \
            + (1.0 + _decay(a_age, half_life_days))
        rows.append({
            "path": str(path),
            "size": stat.st_size,
            "days_since_write": round(m_age, 1),
            "days_since_read": round(a_age, 1),
            "activity": round(activity, 4),
            "category": _category_of(path),
        })
        bucket = dir_counts.setdefault(str(path.parent), {})
        bucket[_category_of(path)] = bucket.get(_category_of(path), 0) + 1

    rows.sort(key=lambda r: (-r["activity"], r["path"]))
    warmest = rows[:top]
    coldest = sorted(rows, key=lambda r: (r["activity"], r["path"]))[:top]

    entropies = [{
        "dir": d,
        "n_files": sum(counts.values()),
        "entropy_bits": round(_shannon_entropy(counts), 3),
        "categories": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    } for d, counts in dir_counts.items()]
    # the most-mixed directories first (highest entropy), files-first tie
    entropies.sort(key=lambda e: (-e["entropy_bits"], -e["n_files"], e["dir"]))
    entropies = entropies[:top]

    return {
        "root": root,
        "n_files": len(rows),
        "n_dirs": len(dir_counts),
        "unreadable": unreadable,
        "half_life_days": half_life_days,
        "warmest": warmest,
        "coldest": coldest,
        "mixed_directories": entropies,
        "algorithm": ("frecency = sum over stat() events of "
                      "1 + 0.5**(age_days/half_life) "
                      "(sysintel.analyze_history's formula, generalized); "
                      "dir entropy = Shannon bits over category mix"),
        "note": ("read-only report; nothing is deleted, moved, or written — "
                 "tidy's journaled apply path stays the only mover"),
    }


# ---------------------------------------------------------------------------
# 2. Metadata near-duplicates (SimHash + Manku banding).
# ---------------------------------------------------------------------------

_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _metadata_tokens(path: Path, stat: os.stat_result) -> List[str]:
    """The fingerprint vocabulary: name words, suffix, size bucket
    (log2 — 'same order of magnitude'), and mtime day. Digits masked by
    scan.simhash's own mask (volatile fields: IMG_0421 vs IMG_0999)."""
    name = path.stem.lower()
    words = [w for w in _NAME_SPLIT_RE.split(name) if w]
    size_bucket = f"size{int(math.log2(max(1, stat.st_size))):d}"
    mtime_day = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc) \
        .strftime("day%Y%m%d")
    tokens = words + [path.suffix.lower().lstrip("."), size_bucket, mtime_day]
    return tokens


def metadata_near_duplicates(root: str, threshold: int = 3,
                             max_files: int = _MAX_FILES
                             ) -> Dict[str, Any]:
    """Near-duplicate files by SimHash metadata fingerprints.

    Two files are near-duplicates when their 64-bit fingerprints sit
    within ``threshold`` Hamming bits (scan.simhash's DEFAULT_THRESHOLD
    operating point, Manku et al. 2007). Candidate generation uses the
    crawler banding trick: the 64 bits split into four 16-bit bands, and
    only files sharing at least one EXACT band are verified pairwise —
    O(n) buckets instead of O(n^2) comparisons, while a distance-3 pair
    misses a shared band with probability (1 - (1 - 3/64)^16... ) small
    enough for report purposes (documented honestly, not exact).
    """
    files = _walk_files(root, max_files=max_files)
    fingerprinted: List[Tuple[Path, os.stat_result, int]] = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size == 0 and not path.suffix:
            # zero-byte unnamed-ish artifacts fingerprint as everything
            continue
        text = " ".join(_metadata_tokens(path, stat))
        fp = simhash(text, mask_digits=True)
        fingerprinted.append((path, stat, fp))

    # Banding: band index -> band value -> file indices.
    bands: Dict[Tuple[int, int], List[int]] = {}
    for idx, (_path, _stat, fp) in enumerate(fingerprinted):
        for band in range(4):
            value = (fp >> (16 * band)) & 0xFFFF
            bands.setdefault((band, value), []).append(idx)

    # Candidate pairs from shared bands, verified by Hamming distance.
    seen_pairs: set = set()
    groups: List[List[int]] = []
    assigned: Dict[int, int] = {}
    for members in bands.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if hamming(fingerprinted[a][2],
                                       fingerprinted[b][2]) <= threshold:
                    # union the two files' groups
                    ga = assigned.get(a)
                    gb = assigned.get(b)
                    if ga is None and gb is None:
                        groups.append([a, b])
                        assigned[a] = assigned[b] = len(groups) - 1
                    elif ga is None:
                        groups[gb].append(a)
                        assigned[a] = gb
                    elif gb is None:
                        groups[ga].append(b)
                        assigned[b] = ga
                    elif ga != gb:
                        merged = groups[ga] + groups[gb]
                        groups[gb] = []
                        groups[ga] = merged
                        for member in merged:
                            assigned[member] = ga

    out_groups = []
    for group in sorted(groups, key=lambda g: (-len(g),
                                               fingerprinted[g[0]][0].name)):
        if len(group) < 2:
            continue
        members = [fingerprinted[i][0] for i in sorted(group)]
        shared_name = os.path.commonprefix(
            [m.name for m in members]).rstrip("_.- ")
        out_groups.append({
            "files": [str(m) for m in members],
            "n": len(members),
            "shared_prefix": shared_name,
            "sizes": sorted({fingerprinted[i][1].st_size for i in group}),
        })
    return {
        "root": root,
        "n_files": len(fingerprinted),
        "n_groups": len(out_groups),
        "groups": out_groups[:25],
        "algorithm": ("SimHash (scan.simhash, 64-bit, digit-masked) over "
                      "name/suffix/size-bucket/mtime tokens; candidates via "
                      "16-bit banding (Manku, Das & Sarma 2007); Hamming <= "
                      f"{threshold}"),
        "note": "read-only report; no file is touched",
    }


# ---------------------------------------------------------------------------
# 3. Term-overlap knowledge graph (co-occurrence + PageRank, reused).
# ---------------------------------------------------------------------------

_DOC_SUFFIXES = (".md", ".txt", ".markdown")


def knowledge_graph(paths: Sequence[str], top_terms: int = 25,
                    keywords_per_doc: int = 8,
                    max_files: int = 2000) -> Dict[str, Any]:
    """Term co-occurrence knowledge graph over notes/docs.

    Per document: TF-IDF keywords against the other documents
    (brain.textmine.keywords — words, corpus-aware, the extractor built
    for exactly this) plus RAKE phrases (genius.language.rake_keywords,
    Rose et al. 2010) as multi-word term atoms when a note is rich
    enough. The graph: one node per term, one edge per co-occurring
    pair inside a document — built THROUGH brain.personal.graph.Graph
    so its PageRank (Page et al. 1999, damping 0.85) and its
    label-propagation communities (Raghavan et al. 2007) run verbatim.
    Document relatedness: Jaccard overlap of the term sets. No
    embeddings, no model file, nothing trained — classical
    co-occurrence only.
    """
    docs: List[Dict[str, Any]] = []
    for root in paths:
        base = Path(os.path.expanduser(root))
        if base.is_file():
            candidates = [base]
        elif base.is_dir():
            candidates = [p for p in _walk_files(str(base),
                                                 max_files=max_files)
                          if p.suffix.lower() in _DOC_SUFFIXES]
        else:
            raise ValueError(f"no such path: {root}")
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            docs.append({"path": str(path), "text": text})

    # Terms per doc: TF-IDF words against the OTHER docs (the extractor's
    # own corpus semantics) + RAKE phrases as multi-word atoms.
    texts = [doc["text"] for doc in docs]
    for index, doc in enumerate(docs):
        corpus = [t for i, t in enumerate(texts) if i != index]
        words = tfidf_keywords(doc["text"], corpus, top=keywords_per_doc)
        rake = rake_keywords(doc["text"],
                             top=max(2, keywords_per_doc // 2))
        phrases = [row["phrase"] for row in rake.get("keywords") or []]
        terms = [str(t).lower().strip() for t in words + phrases
                 if str(t).strip()]
        doc["terms"] = sorted(set(terms))
        del doc["text"]  # terms only from here; text never persisted

    # Term co-occurrence graph via the EXISTING Graph implementation.
    graph = Graph()
    for doc in docs:
        terms = doc["terms"]
        for term in terms:
            graph.add(term, [t for t in terms if t != term])
    ranks = graph.pagerank()
    communities = graph.communities()

    top = sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0]))[:top_terms]
    related: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = set(docs[i]["terms"]), set(docs[j]["terms"])
            if not a or not b:
                continue
            overlap = len(a & b)
            if overlap < 2:
                continue
            union = len(a | b)
            related.append({
                "a": docs[i]["path"], "b": docs[j]["path"],
                "shared": sorted(a & b)[:8],
                "jaccard": round(overlap / union, 3),
            })
    related.sort(key=lambda r: (-r["jaccard"], r["a"], r["b"]))
    return {
        "n_docs": len(docs),
        "n_terms": len(graph.nodes),
        "top_terms": [{"term": t, "pagerank": round(s, 5)} for t, s in top],
        "communities": [c for c in communities if len(c) > 1][:20],
        "related_docs": related[:25],
        "algorithm": ("TF-IDF words (brain.textmine.keywords, corpus = the "
                      "other docs) + RAKE phrases (language.rake_keywords) "
                      "-> term co-occurrence graph -> PageRank + label "
                      "propagation (brain.personal.graph, reused verbatim); "
                      "doc-doc Jaccard overlap"),
        "note": "read-only analysis over the given paths",
    }


# ---------------------------------------------------------------------------
# 4. Byte-signature file-type inference (Aho-Corasick + correctable NB).
# ---------------------------------------------------------------------------

# The classic magic-byte table. Offsets are anchored: a match counts only
# at its declared offset (post-filtered out of the AC scan), so a "PK"
# deep inside a text file never claims ZIP.
_SIGNATURES: Tuple[Tuple[str, int, str], ...] = (
    ("\x89PNG", 0, "png"),
    ("\xff\xd8\xff", 0, "jpeg"),
    ("GIF8", 0, "gif"),
    ("%PDF", 0, "pdf"),
    ("PK\x03\x04", 0, "zip"),
    ("\x7fELF", 0, "elf"),
    ("#!", 0, "script"),
    ("\x1f\x8b", 0, "gzip"),
    ("BZh", 0, "bzip2"),
    ("ustar", 257, "tar"),
    ("SQLite format 3", 0, "sqlite"),
    ("ID3", 0, "mp3"),
    ("OggS", 0, "ogg"),
    ("fLaC", 0, "flac"),
    ("<?xml", 0, "xml"),
)



def _signature_automaton() -> Automaton:
    return Automaton([sig for sig, _off, _kind in _SIGNATURES],
                     ignore_case=False)


def _byte_features(head: bytes) -> List[str]:
    """Byte-histogram + structure tokens (the NB's vocabulary)."""
    features: List[str] = []
    for b in head[:64]:
        features.append(f"b{b:02x}")
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    nulls = sum(1 for b in head if b == 0)
    high = sum(1 for b in head if b >= 128)
    n = max(1, len(head))
    features.append("printable" if printable / n > 0.85 else "nonprintable")
    features.append("nulls" if nulls > n * 0.05 else "no-nulls")
    features.append("highbytes" if high > n * 0.1 else "asciiish")
    if b"\x00\x00\x00" in head:
        features.append("zero-run")
    if head.count(b"\n") >= 2:
        features.append("multiline")
    return features


def load_filetype_classifier(state: Optional[Dict[str, Any]]) -> NaiveBayes:
    """Rebuild the correction-trained Naive Bayes from the brain state's
    bounded correction docs (state[DOCS_KEY])."""
    nb = NaiveBayes()
    docs = (state or {}).get(DOCS_KEY) or []
    for doc in [d for d in docs if isinstance(d, dict)]:
        nb.train(doc.get("features") or [], [doc.get("label") or "unknown"])
    return nb


def record_correction(path: str, true_type: str,
                      head: int = 512) -> Dict[str, Any]:
    """One user correction as a training doc: byte features of the file
    plus the human-supplied type. The caller appends it to the state's
    bounded docs list (see append_correction) and persists through the
    brain's atomic save — the same caller-owns-persistence contract
    every learner here uses."""
    target = Path(os.path.expanduser(path))
    if not target.is_file():
        raise ValueError(f"no such file: {path}")
    with target.open("rb") as handle:
        blob = handle.read(head)
    return {
        "path": str(target),
        "label": str(true_type).strip().lower() or "unknown",
        "features": _byte_features(blob),
    }


def append_correction(state: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    """Bound the correction list to the most recent _NB_MAX_DOCS (oldest
    dropped — the review-bucket discipline) and return the updated state."""
    out = dict(state or {})
    docs = [d for d in (out.get(DOCS_KEY) or []) if isinstance(d, dict)]
    docs.append(doc)
    out[DOCS_KEY] = docs[-_NB_MAX_DOCS:]
    return out


def infer_filetype(path: str, head: int = 512,
                   classifier: Optional[NaiveBayes] = None
                   ) -> Dict[str, Any]:
    """Byte-signature inference with a correctable Naive Bayes fallback.

    Order of authority: (1) the anchored magic table via Aho-Corasick
    (scan.ac — one pass over the head, O(len) regardless of table size);
    (2) when no signature matches, the NB ranks types the user has
    TAUGHT it via corrections (it starts empty and says so — no invented
    priors); (3) otherwise unknown, honestly.
    """
    target = Path(os.path.expanduser(path))
    if not target.is_file():
        raise ValueError(f"no such file: {path}")
    try:
        with target.open("rb") as handle:
            blob = handle.read(head)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not blob:
        return {"path": path, "verdict": "empty", "type": None,
                "evidence": ["zero-byte file"]}

    # 1. Anchored signatures (latin-1 keeps byte->char 1:1 for the str-based
    #    AC automaton; offsets are preserved exactly).
    text = blob.decode("latin-1")
    automaton = _signature_automaton()
    best: Optional[Tuple[int, str, str]] = None  # (end_index, type, sig)
    for end_index, pid in automaton.scan(text):
        signature, offset, kind = _SIGNATURES[pid]
        if end_index - len(signature) == offset:
            if best is None or end_index < best[0]:
                best = (end_index, kind, signature)
    if best is not None:
        end_index, kind, signature = best
        return {
            "path": path, "verdict": "signature", "type": kind,
            "evidence": [f"magic bytes {signature!r} at offset "
                         f"{end_index - len(signature)}"],
        }

    # 2. Correctable Naive Bayes over byte features.
    features = _byte_features(blob)
    if classifier is not None and classifier.class_docs:
        ranked = classifier.rank(features)
        top_label, confidence = ranked[0]
        return {
            "path": path, "verdict": "bayes", "type": top_label,
            "confidence": round(confidence, 3),
            "evidence": [f"byte features {', '.join(features[:8])} ... "
                         f"({len(classifier.class_docs)} taught types)"],
        }

    # 3. Honest unknown.
    return {
        "path": path, "verdict": "unknown", "type": None,
        "evidence": ["no anchored magic match and the correction-trained "
                     "classifier has no data for this shape yet"],
        "teach": f"teach it: genius fsbrain filetype {path} --correct TYPE",
    }



# ---------------------------------------------------------------------------
# NCD file/folder resemblance (exponential-build 2.3) — read-only.
# ---------------------------------------------------------------------------

_MAX_SAMPLE_BYTES = 65536       # per compared file (deterministic head)
_MAX_SAMPLES_PER_FOLDER = 8     # bounded sampling per candidate folder


def ncd_resemble(content: bytes, folders: Dict[str, List[bytes]],
                 method: str = "nearest", compressor: str = "zlib",
                 max_sample_bytes: int = _MAX_SAMPLE_BYTES
                 ) -> Dict[str, Any]:
    """'Which existing folder does this file most resemble?' — answered
    with the ONE existing NCD primitive (``genius/data.py::ncd``, Li et
    al. 2004; no new compressor, no new distance). ``folders`` maps a
    folder name to its sample file contents (the caller owns the
    reading; the pure core never touches the filesystem).

    - ``method='nearest'`` (default): the folder's distance is its BEST
      sample — one genuinely similar resident is enough to resemble;
    - ``method='profile'``: the folder's samples are concatenated into
      one profile and compared as a whole.

    Honesty: empty content is refused (nothing to compare); folders
    with no samples are skipped and NAMED, never scored as 1.0;
    compared files are deterministically head-truncated to
    ``max_sample_bytes`` and the truncation is reported; every folder's
    distance is returned (the ranking IS the answer), ties broken by
    folder name. Deterministic end to end."""
    from . import data as data_mod  # the ONE NCD implementation
    if not content:
        raise ValueError("the file's content is empty — nothing to compare")
    if method not in ("nearest", "profile"):
        raise ValueError(f"unknown method {method!r} (nearest|profile)")
    truncated = len(content) > max_sample_bytes
    content = content[:max_sample_bytes]
    matches: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name in sorted(folders):
        samples = [s[:max_sample_bytes] for s in (folders[name] or []) if s]
        if not samples:
            skipped.append(name)
            continue
        if method == "profile":
            dist = data_mod.ncd(content, b"".join(samples),
                                compressor=compressor)["ncd"]
            matches.append({"folder": name, "distance": dist,
                            "n_samples": len(samples)})
        else:
            dists = [data_mod.ncd(content, s, compressor=compressor)["ncd"]
                     for s in samples]
            matches.append({"folder": name, "distance": min(dists),
                            "n_samples": len(dists)})
    if not matches:
        raise ValueError("no folder had a readable sample to compare against")
    matches.sort(key=lambda m: (m["distance"], m["folder"]))
    return {
        "best_folder": matches[0]["folder"],
        "distances": matches,
        "method": method,
        "compared_bytes": min(len(content), max_sample_bytes),
        "file_truncated": truncated,
        "skipped_empty_folders": skipped,
        "algorithm": ("NCD folder resemblance (Li et al. 2004 via "
                      "genius.data.ncd); read-only"),
    }


def resemble_file(path: str, folder_paths: Sequence[str],
                  method: str = "nearest",
                  max_samples: int = _MAX_SAMPLES_PER_FOLDER
                  ) -> Dict[str, Any]:
    """The read-only filesystem wrapper: the query file plus bounded,
    deterministic samples from each candidate folder (sorted walk, the
    first ``max_samples`` readable files; unreadable files are counted,
    never faked). Nothing is written, moved, or executed."""
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        raise ValueError(f"no such file: {path}")
    with open(p, "rb") as fh:
        content = fh.read(_MAX_SAMPLE_BYTES)
    folders: Dict[str, List[bytes]] = {}
    unreadable = 0
    for root in folder_paths:
        samples: List[bytes] = []
        for f in _walk_files(root, max_files=max_samples):
            try:
                with open(f, "rb") as fh:
                    samples.append(fh.read(_MAX_SAMPLE_BYTES))
            except OSError:
                unreadable += 1
        folders[str(root)] = samples
    result = ncd_resemble(content, folders, method=method)
    result["query_file"] = path
    result["unreadable_samples"] = unreadable
    return result

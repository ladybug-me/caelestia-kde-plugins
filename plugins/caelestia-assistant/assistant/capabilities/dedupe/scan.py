"""dedupe.scan — duplicate detection with a three-stage funnel.

size -> partial hash (first 4 KiB) -> full hash. Identical-result
discipline: the funnel can only ever return a SUPERSET at each stage,
so the full-hash stage decides. Deterministic ordering everywhere
(groups sorted by total size desc, members sorted by path).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = ["scan_duplicates", "survival_scores", "quarantine"]

_PARTIAL_BYTES = 4096


def _hash_file(path: Path, partial: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if partial:
            h.update(fh.read(_PARTIAL_BYTES))
        else:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    return h.hexdigest()


def scan_duplicates(roots: List[str], min_size: int = 1) -> List[Dict]:
    """Duplicate groups under ``roots``. Returns a list of
    {"size", "hash", "paths"} sorted by total wasted size desc."""
    by_size: Dict[int, List[Path]] = {}
    for root in roots:
        base = Path(root)
        if base.is_file():
            entries = [base]
        elif base.is_dir():
            entries = sorted(base.rglob("*"))
        else:
            continue
        for p in entries:
            try:
                if p.is_file() and not p.is_symlink():
                    st = p.stat()
                    if st.st_size >= min_size:
                        by_size.setdefault(st.st_size, []).append(p)
            except OSError:
                continue  # unreadable entries are honestly skipped
    # stage 2: partial hash
    by_partial: Dict[Tuple[int, str], List[Path]] = {}
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for p in paths:
            try:
                by_partial.setdefault((size, _hash_file(p, partial=True)),
                                      []).append(p)
            except OSError:
                continue
    # stage 3: full hash decides
    groups: List[Dict] = []
    for (size, _part), paths in sorted(by_partial.items(),
                                       key=lambda kv: (-kv[0][0], kv[0][1])):
        if len(paths) < 2:
            continue
        by_full: Dict[str, List[str]] = {}
        for p in paths:
            try:
                by_full.setdefault(_hash_file(p), []).append(str(p))
            except OSError:
                continue
        for digest, members in by_full.items():
            if len(members) > 1:
                groups.append({
                    "size": size,
                    "wasted": size * (len(members) - 1),
                    "hash": digest,
                    "paths": sorted(members),
                })
    groups.sort(key=lambda g: (-g["wasted"], g["hash"]))
    return groups


def survival_scores(paths: List[str], now: float) -> Dict[str, float]:
    """Last-access-based survival score in [0,1]: 1.0 for entries
    accessed within a day, decaying by half every 30 days after that.
    Honest about atime-unreliability: the note field of the caller's
    report should carry that this is a PRIOR, not a verdict."""
    scores: Dict[str, float] = {}
    for p in paths:
        try:
            atime = os.stat(p).st_atime
        except OSError:
            scores[p] = 0.0
            continue
        age_days = max(0.0, (now - atime) / 86400.0)
        if age_days <= 1.0:
            scores[p] = 1.0
        else:
            scores[p] = round(0.5 ** ((age_days - 1.0) / 30.0), 4)
    return scores


def quarantine(paths: List[str], quarantine_dir: str, now: float,
               ttl_days: int = 30) -> Dict[str, Any]:
    """TTL quarantine: move files into quarantine_dir with a manifest
    recording origin + expiry; restore() undoes it; a purge after the
    TTL is a SEPARATE explicit action. Journaled by construction (the
    manifest is the journal). Returns {"moved": [...], "errors": [...]}."""
    import json
    import time
    qdir = Path(quarantine_dir)
    qdir.mkdir(parents=True, exist_ok=True)
    manifest_path = qdir / "manifest.json"
    try:
        manifest: Dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {"entries": []}
    moved, errors = [], []
    for p in paths:
        src = Path(p)
        if not src.is_file():
            errors.append({"path": p, "error": "not a regular file"})
            continue
        dest = qdir / f"{int(now * 1000)}-{src.name}"
        try:
            os.replace(src, dest)
        except OSError as exc:
            errors.append({"path": p, "error": f"{type(exc).__name__}: {exc}"})
            continue
        manifest["entries"].append({
            "original": str(src),
            "quarantined_at": now,
            "purge_after": now + ttl_days * 86400.0,
            "file": dest.name,
        })
        moved.append(str(dest))
    manifest["entries"].sort(key=lambda e: (e["quarantined_at"], e["file"]))
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"moved": moved, "errors": errors,
            "manifest": str(manifest_path)}


def restore(quarantine_dir: str, now: float) -> Dict[str, Any]:
    """Undo every quarantine entry whose TTL has expired or that is due
    to be purged later — restore ALL entries the caller asks for via
    ttl_expired_only=False; default restores only expired ones."""
    import json
    qdir = Path(quarantine_dir)
    manifest_path = qdir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"restored": [], "errors": [{"error": "no manifest"}]}
    restored, errors, keep = [], [], []
    for entry in manifest.get("entries", []):
        dest = qdir / entry["file"]
        src = Path(entry["original"])
        if entry["purge_after"] > now:
            keep.append(entry)  # TTL not yet reached: leave quarantined
            continue
        try:
            if dest.is_file():
                src.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, src)
            restored.append(entry["original"])
        except OSError as exc:
            errors.append({"path": entry["original"],
                           "error": f"{type(exc).__name__}: {exc}"})
            keep.append(entry)
    manifest["entries"] = keep
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"restored": restored, "errors": errors}

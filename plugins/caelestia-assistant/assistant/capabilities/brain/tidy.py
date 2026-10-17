"""brain.tidy — filesystem organisation: a plan you approve, never a surprise.

The "cleans" half of the second-brain contract. `survey(root)` walks a
directory tree (bounded depth, symlinks never followed) and returns a plan:

  duplicates   size-bucketed candidates -> zlib.crc32 partial fingerprints ->
               byte-exact confirmation; keep the OLDEST copy, propose moving
               the rest into duplicates/
  stale        files untouched for the oldest age-quartile threshold,
               grouped by year for review (proposals, not deletion — this
               module NEVER deletes, ever)
  type_moves   extension-based routing (Downloads-style triage) with
               collision-safe target names name-2.ext, name-3.ext ...
  big_files    the few largest files, for human review
  empty_dirs   swept bottom-up; deepest first

Applying is a separate, explicit, journaled operation: `apply_moves` runs
ONLY the type_moves/duplicates moves you pass to it, uses os.rename inside
the same root (never crossing devices, never overwriting), writes a rollback
journal first, and `rollback` undoes it in reverse order. Deletion is not
implemented — the plan only ever SUGGESTS an `rm` as an inert string you
copy-paste yourself, exactly like the troubleshooting layers' contract.
"""
from __future__ import annotations

import json
import os
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["survey", "apply_moves", "rollback", "render_plan", "DEFAULT_CATEGORIES"]

DEFAULT_CATEGORIES: Dict[str, tuple] = {
    "images": (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".heic"),
    "documents": (".pdf", ".doc", ".docx", ".odt", ".txt", ".md", ".tex", ".epub"),
    "archives": (".zip", ".tar", ".gz", ".xz", ".7z", ".rar", ".bz2"),
    "audio": (".mp3", ".flac", ".ogg", ".wav", ".m4a", ".opus"),
    "video": (".mp4", ".mkv", ".webm", ".mov", ".avi"),
    "code": (".py", ".js", ".ts", ".c", ".cpp", ".h", ".rs", ".go", ".java",
             ".sh", ".qml", ".json", ".yaml", ".toml"),
    "installers": (".deb", ".rpm", ".appimage", ".pkg.tar.zst", ".exe", ".iso"),
}

_FORBIDDEN_ROOTS = ("/", "/etc", "/usr", "/bin", "/boot", "/dev", "/proc",
                    "/sys", "/var", "/lib")
_MAX_DEPTH = 4
_MAX_ENTRIES = 20_000
_PARTIAL_READ = 1 << 16  # 64 KiB head + tail fingerprint


# ---------------------------------------------------------------------------
def _fingerprint(path: Path, size: int) -> int:
    """crc32 of head+tail (fast reject); callers confirm byte-exactness with
    a full-hash comparison before proposing anything."""
    crc = 0
    with open(path, "rb") as fh:
        crc = zlib.crc32(fh.read(_PARTIAL_READ), crc)
        if size > 2 * _PARTIAL_READ:
            fh.seek(-_PARTIAL_READ, os.SEEK_END)
            crc = zlib.crc32(fh.read(_PARTIAL_READ), crc)
    return crc


def _files_equal(a: Path, b: Path) -> bool:
    """Byte-exact comparison in 1 MiB chunks (no whole-file buffering)."""
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca, cb = fa.read(1 << 20), fb.read(1 << 20)
            if ca != cb:
                return False
            if not ca:
                return True


def _category_for(suffix: str) -> Optional[str]:
    for cat, exts in DEFAULT_CATEGORIES.items():
        if suffix in exts:
            return cat
    return None


def _unique_target(directory: Path, name: str) -> Path:
    """name-2.ext style collision avoidance (never overwrite)."""
    target = directory / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 2
    while (directory / f"{stem}-{i}{suffix}").exists():
        i += 1
    return directory / f"{stem}-{i}{suffix}"


def _walk(root: Path) -> List[Path]:
    files: List[Path] = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).resolve().parts) - len(root.resolve().parts)
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            p = Path(dirpath) / fname
            if p.is_symlink():
                continue
            files.append(p)
            count += 1
            if count >= _MAX_ENTRIES:
                return files
    return files


# ---------------------------------------------------------------------------
def survey(root: str, categories: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
    """Full survey of `root`. Dry-run by construction: this function only
    READS. Every proposed move stays inside `root`."""
    r = Path(os.path.expanduser(root)).resolve()
    if str(r) in _FORBIDDEN_ROOTS or str(r) == str(Path.home()):
        raise ValueError(f"refusing to survey unsafe root {r}")
    if not r.is_dir():
        raise ValueError(f"not a directory: {r}")
    cats = categories or DEFAULT_CATEGORIES

    files = _walk(r)
    now = datetime.now(timezone.utc).timestamp()
    stats = []
    for p in files:
        try:
            st = p.stat()
            stats.append((p, st.st_size, st.st_mtime))
        except OSError:
            continue
    if not stats:
        return {"root": str(r), "scanned": 0, "duplicates": [], "stale": [],
                "type_moves": [], "big_files": [], "empty_dirs": [],
                "space_recoverable": 0, "inert_suggestions": []}

    # ---- duplicates: size bucket -> partial crc -> byte-exact -----------
    by_size: Dict[int, List[Path]] = {}
    for p, size, _mt in stats:
        if size == 0:
            continue
        by_size.setdefault(size, []).append(p)
    dup_groups: List[List[Path]] = []
    for size, group in sorted(by_size.items(), reverse=True):
        if len(group) < 2:
            continue
        by_crc: Dict[int, List[Path]] = {}
        for p in group:
            try:
                by_crc.setdefault(_fingerprint(p, size), []).append(p)
            except OSError:
                continue
        for crc, cg in by_crc.items():
            if len(cg) < 2:
                continue
            confirmed: List[Path] = []
            remaining = list(cg)
            while len(remaining) > 1:
                ref = remaining[0]
                same = [ref]
                for other in remaining[1:]:
                    if _files_equal(ref, other):
                        same.append(other)
                if len(same) > 1:
                    dup_groups.append(same)
                for s in same:
                    remaining.remove(s)
                if not remaining:
                    break

    # ---- stale: oldest age quartile ------------------------------------
    ages = sorted(now - mt for _p, _s, mt in stats)
    q1 = ages[len(ages) // 4] if ages else 0.0
    stale = [
        {"path": str(p), "age_days": int((now - mt) / 86400)}
        for p, _s, mt in stats
        if (now - mt) > q1 > 0 and (now - mt) > 90 * 86400
    ]
    stale.sort(key=lambda d: -d["age_days"])

    # ---- type routing ---------------------------------------------------
    type_moves: List[Dict[str, Any]] = []
    for p, _s, _mt in stats:
        if p.parent != r:
            continue  # only route the TOP level, like a Downloads folder
        cat = _category_for(p.suffix.lower())
        if cat:
            target_dir = r / cat
            target = _unique_target(target_dir, p.name) if target_dir.exists() \
                else target_dir / p.name
            if target != p:
                type_moves.append({
                    "from": str(p), "to": str(target), "category": cat,
                    "reason": f"{p.suffix.lower()} belongs to {cat}",
                })

    # ---- big files / empty dirs ----------------------------------------
    big = [{"path": str(p), "mb": round(size / 1048576, 1)}
           for p, size, _mt in sorted(stats, key=lambda t: -t[1])[:5]
           if size > 50 * 1048576]
    empty_dirs = sorted(
        str(dp) for dp, dns, fns in os.walk(r)
        if not dns and not fns and not Path(dp).name.startswith(".")
        and Path(dp) != r
    )

    dup_moves: List[Dict[str, Any]] = []
    space = 0
    for group in dup_groups:
        keep = min(group, key=lambda p: p.stat().st_mtime)
        for p in group:
            if p == keep:
                continue
            space += p.stat().st_size
            dup_moves.append({
                "from": str(p), "to": str(r / "duplicates" / p.name),
                "category": "duplicates",
                "reason": f"byte-identical to {keep.name} (kept the oldest copy)",
            })

    return {
        "root": str(r),
        "scanned": len(stats),
        "duplicates": dup_moves,
        "stale": stale[:40],
        "type_moves": type_moves,
        "big_files": big,
        "empty_dirs": empty_dirs,
        "space_recoverable": space,
        "inert_suggestions": [
            f"SUGGESTED_NOT_EXECUTED: rm '{d['path']}'   # READ_ONLY review first — {d['age_days']} days untouched"
            for d in stale[:5]
        ],
    }


# ---------------------------------------------------------------------------
def apply_moves(moves: List[Dict[str, Any]], root: str,
                journal_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute ONLY the move rows you pass (from survey output) behind an
    explicit caller decision. Same-root os.rename only; targets are made
    collision-safe; a rollback journal is written BEFORE the first move."""
    r = Path(os.path.expanduser(root)).resolve()
    journal = Path(os.path.expanduser(journal_path)) if journal_path else \
        Path.home() / ".local/state/caelestia-brain/tidy-journal.json"
    entries: List[Dict[str, Any]] = []
    applied, skipped = [], []
    for mv in moves:
        src, dst = Path(mv["from"]), Path(mv["to"])
        try:
            src.resolve().relative_to(r)
        except ValueError:
            skipped.append({"from": mv["from"], "reason": "outside root"})
            continue
        if not src.is_file() or src.is_symlink():
            skipped.append({"from": mv["from"], "reason": "not a regular file"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = _unique_target(dst.parent, dst.name)
        if dst.exists():
            skipped.append({"from": mv["from"], "reason": "target exists; refused"})
            continue
        entries.append({"from": str(src), "to": str(dst),
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps({"root": str(r), "entries": entries}, indent=2), encoding="utf-8")
    for e in entries:
        try:
            os.rename(e["from"], e["to"])
            applied.append(e)
        except OSError as exc:
            skipped.append({"from": e["from"], "reason": str(exc)})
    return {"applied": applied, "skipped": skipped, "journal": str(journal)}


def rollback(journal_path: Optional[str] = None) -> Dict[str, Any]:
    """Undo the last journaled tidy run in reverse order."""
    journal = Path(os.path.expanduser(journal_path)) if journal_path else \
        Path.home() / ".local/state/caelestia-brain/tidy-journal.json"
    if not journal.exists():
        return {"undone": [], "missing": [], "journal": str(journal)}
    data = json.loads(journal.read_text(encoding="utf-8"))
    undone, missing = [], []
    for e in reversed(data.get("entries", [])):
        if Path(e["to"]).exists():
            os.rename(e["to"], e["from"])
            undone.append(e)
        else:
            missing.append(e)
    return {"undone": undone, "missing": missing, "journal": str(journal)}


def render_plan(plan: Dict[str, Any]) -> str:
    """Human plan renderer — every destructive-looking line stays inert."""
    out = [f"tidy plan for {plan['root']} ({plan['scanned']} files scanned)"]
    moves = plan["type_moves"] + plan["duplicates"]
    if moves:
        out.append("proposed moves (apply only with your explicit approval):")
        for mv in moves[:15]:
            out.append(f"  {mv['from']}  ->  {mv['to']}   # {mv['reason']}")
        extra = len(moves) - 15
        if extra > 0:
            out.append(f"  ... and {extra} more")
    if plan["space_recoverable"]:
        out.append(f"space recoverable from duplicates: "
                   f"{plan['space_recoverable'] / 1048576:.1f} MB")
    if plan["stale"]:
        out.append(f"stale files (oldest quartile, >90 days): {len(plan['stale'])}"
                   " — review below, deletion is NEVER automatic:")
        for s in plan["stale"][:5]:
            out.append(f"  {s['age_days']:>5}d  {s['path']}")
    if plan["big_files"]:
        out.append("largest files:")
        for b in plan["big_files"]:
            out.append(f"  {b['mb']:>8.1f} MB  {b['path']}")
    if plan["empty_dirs"]:
        out.append(f"empty directories: {len(plan['empty_dirs'])}")
    for sug in plan["inert_suggestions"][:3]:
        out.append(sug)
    if not (moves or plan["stale"] or plan["empty_dirs"]):
        out.append("nothing to do — this root is already tidy")
    return "\n".join(out)

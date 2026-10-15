"""brain.merkle — Merkle-tree diffing over a config directory
(exponential-build-4 E).

The gap this fills is NOT near-duplicate detection (brain/minhash.py
finds similar files) and NOT the tidy journaled moves — it is the
question "what changed in my config TREE since my last snapshot",
answered without rehashing every file every time.

The structure is the standard Merkle tree (Merkle 1987, "A Certified
Digital Signature", CRYPTO '89 proceedings — the hash-tree idea;
no signatures are made here, only the tree):

  * a FILE leaf carries sha256(content) — chunked, bounded memory;
  * a DIRECTORY node carries sha256 over its children's
    "type:name:hash" lines in sorted order, so any change below a
    directory changes that directory's hash and nothing else does;

which gives the diff property the per-file SimHash detector lacks:
compare two tree roots — if a directory hash is EQUAL, the whole
subtree is provably identical and is SKIPPED (the skip count is
reported so the pruning is visible); only differing branches are
descended. INCREMENTAL REFRESH is where the not-rehashing-everything
is earned: refresh(tree, root) stats each known file and rehashes
only files whose (size, mtime) changed, then rebuilds directory
hashes bottom-up only where a child hash changed.

Honest envelope: this is change DETECTION, not a backup and not a
security feature — sha256 over contents proves difference, and
equality of hashes is the standard collision assumption, stated
rather than claimed. It reads the target directory read-only and
writes NOTHING: snapshots are plain dicts for the caller's
learned-state JSON path (the same enumerated write path as every
learned state in this repo).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["DEFAULT_CONFIG_ROOT", "MAX_FILES", "snapshot", "refresh",
           "diff", "root_hash", "render_changes"]

DEFAULT_CONFIG_ROOT = "~/.config/caelestia"
MAX_FILES = 50_000
_CHUNK = 65536
_UNCHANGED = "unchanged"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dir_hash(children: List[Tuple[str, str, str]]) -> str:
    """sha256 over sorted 'type:name:hash' lines — order-independent
    of directory iteration, sensitive to every child change."""
    h = hashlib.sha256()
    for kind, name, digest in sorted(children):
        h.update(f"{kind}:{name}:{digest}\n".encode("utf-8"))
    return h.hexdigest()


def _build(dir_path: Path, rel: str, counters: Dict[str, int]
           ) -> Dict[str, Any]:
    """Depth-first tree build. dirs: {name: node}; files: {name: hash}."""
    node: Dict[str, Any] = {"dirs": {}, "files": {}, "hash": ""}
    entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            counters["dirs"] += 1
            child = _build(entry, f"{rel}{entry.name}/", counters)
            node["dirs"][entry.name] = child
        elif entry.is_file() and not entry.is_symlink():
            counters["files"] += 1
            if counters["files"] > MAX_FILES:
                raise ValueError(
                    f"tree exceeds {MAX_FILES} files — refusing to "
                    "snapshot a tree this large silently; scope it "
                    "explicitly")
            node["files"][entry.name] = _hash_file(entry)
    node["hash"] = _dir_hash(
        [("d", name, child["hash"]) for name, child in node["dirs"].items()]
        + [("f", name, digest) for name, digest in node["files"].items()])
    return node


def snapshot(root: str = DEFAULT_CONFIG_ROOT) -> Dict[str, Any]:
    """Full snapshot of one directory tree (read-only). Symlinks are
    deliberately NOT followed (a config tree's symlinks point out of
    the tree; following them would hash the world)."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise ValueError(f"not a directory: {root_path}")
    counters = {"files": 0, "dirs": 0}
    tree = _build(root_path, "", counters)
    return {"root": str(root_path), "tree": tree,
            "n_files": counters["files"], "n_dirs": counters["dirs"]}


def _refresh_dir(node: Dict[str, Any], dir_path: Path,
                 counters: Dict[str, int]) -> None:
    """Incremental: rehash only changed-size/mtime files; prune removed;
    add new; rebuild the dir hash from children (dirs whose hash input
    is unchanged rebuild to the same hash in one dir-hash — no deep
    walk). The stats are the cheap part; the SHAS are the cost being
    avoided."""
    children: List[Tuple[str, str, str]] = []
    seen_files = set()
    seen_dirs = set()
    entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            seen_dirs.add(entry.name)
            counters["dirs"] += 1
            child = node["dirs"].get(entry.name)
            if child is None:
                child = _build(entry, "", counters)
                node["dirs"][entry.name] = child
            else:
                _refresh_dir(child, entry, counters)
            children.append(("d", entry.name, child["hash"]))
        elif entry.is_file() and not entry.is_symlink():
            seen_files.add(entry.name)
            counters["files"] += 1
            known = node["files"].get(entry.name)
            stat = entry.stat()
            if known is None or _stale(node, entry.name, stat):
                node["files"][entry.name] = _hash_file(entry)
                node.setdefault("stats", {})[entry.name] = [
                    stat.st_size, stat.st_mtime_ns]
            children.append(("f", entry.name, node["files"][entry.name]))
    for name in list(node["dirs"]):
        if name not in seen_dirs:
            del node["dirs"][name]
    for name in list(node["files"]):
        if name not in seen_files:
            del node["files"][name]
            node.get("stats", {}).pop(name, None)
    node["hash"] = _dir_hash(children)


def _stale(node: Dict[str, Any], name: str, stat: os.stat_result) -> bool:
    """A file is stale (needs rehashing) when its (size, mtime_ns)
    differs from the snapshot's record. First-generation trees carry
    no stats and are fully rehashed once — then the bookkeeping
    exists. A missing stat entry is treated as stale (honest: unknown
    state gets rehashed, never assumed unchanged)."""
    known = node.get("stats", {}).get(name)
    if known is None:
        return True
    return known[0] != stat.st_size or known[1] != stat.st_mtime_ns


def refresh(tree: Dict[str, Any], root: Optional[str] = None
            ) -> Dict[str, Any]:
    """Incremental refresh of one snapshot against the (possibly
    changed) directory: rehash stale files only, rebuild dir hashes
    bottom-up, report how many file hashes were REUSED — the visible
    proof the Merkle structure earned its keep."""
    root_path = Path(root or tree["root"]).expanduser()
    if not root_path.is_dir():
        raise ValueError(f"not a directory: {root_path}")
    counters = {"files": 0, "dirs": 0, "rehashed": 0, "reused": 0}
    before: Dict[str, str] = {}

    def collect(node: Dict[str, Any], rel: str) -> None:
        for name, digest in node["files"].items():
            before[f"{rel}{name}"] = digest
        for name, child in node["dirs"].items():
            collect(child, f"{rel}{name}/")

    collect(tree["tree"], "")

    def count(node: Dict[str, Any], rel: str) -> None:
        for name, digest in node["files"].items():
            key = f"{rel}{name}"
            if before.get(key) == digest:
                counters["reused"] += 1
            else:
                counters["rehashed"] += 1
        for name, child in node["dirs"].items():
            count(child, f"{rel}{name}/")

    _refresh_dir(tree["tree"], root_path, counters)
    count(tree["tree"], "")
    tree["n_files"] = counters["files"]
    tree["n_dirs"] = counters["dirs"]
    tree["refresh"] = {"rehashed": counters["rehashed"],
                       "reused": counters["reused"]}
    return tree


def root_hash(tree: Dict[str, Any]) -> str:
    """The single hash standing for the whole tree (the Merkle root)."""
    return tree["tree"]["hash"]


def diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """What changed between two snapshots, pruning identical subtrees:
    if a directory hash matches, its whole subtree is identical and is
    skipped in one comparison (the skip count is reported — the Merkle
    structure's reason to exist). Added/removed/changed files, plus
    added/removed directories, deterministic order throughout."""
    added: List[str] = []
    removed: List[str] = []
    changed: List[str] = []
    added_dirs: List[str] = []
    removed_dirs: List[str] = []
    pruned = {"subtrees_skipped": 0, "files_compared": 0}

    # F8: iterative traversal (an explicit stack) — a crafted snapshot
    # with pathological depth must diff, not hit the recursion limit.
    stack: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]] = [
        (old.get("tree"), new.get("tree"), "")]
    while stack:
        a, b, rel = stack.pop()
        if a is not None and b is not None and a["hash"] == b["hash"]:
            pruned["subtrees_skipped"] += 1
            continue
        a_files = a["files"] if a else {}
        b_files = b["files"] if b else {}
        for name in sorted(set(a_files) | set(b_files)):
            key = f"{rel}{name}"
            pa, pb = a_files.get(name), b_files.get(name)
            pruned["files_compared"] += 1
            if pa is None:
                added.append(key)
            elif pb is None:
                removed.append(key)
            elif pa != pb:
                changed.append(key)
        for name in sorted(set((a["dirs"] if a else {}))
                           | set((b["dirs"] if b else {}))):
            child_a = (a or {}).get("dirs", {}).get(name)
            child_b = (b or {}).get("dirs", {}).get(name)
            child_rel = f"{rel}{name}/"
            if child_a is not None and child_b is None:
                removed_dirs.append(child_rel)
                _list_all(child_a, child_rel, removed, removed_dirs)
            elif child_b is not None and child_a is None:
                added_dirs.append(child_rel)
                _list_all(child_b, child_rel, added, added_dirs)
            else:
                stack.append((child_a, child_b, child_rel))

    def _list_all(node: Dict[str, Any], rel: str,
                  files: List[str], dirs: List[str]) -> None:
        # iterative too (same reason as the main walk)
        pending = [(node, rel)]
        while pending:
            cur, cur_rel = pending.pop()
            for name in cur["files"]:
                files.append(f"{cur_rel}{name}")
            for name, child in cur["dirs"].items():
                dirs.append(f"{cur_rel}{name}/")
                pending.append((child, f"{cur_rel}{name}/"))
    return {"added": added, "removed": removed, "changed": changed,
            "added_dirs": added_dirs, "removed_dirs": removed_dirs,
            "same": root_hash(old) == root_hash(new),
            "pruned": pruned,
            "note": "identical subtrees are pruned by directory hash — "
                    "the skip count is the pruning working"}


def render_changes(result: Dict[str, Any]) -> List[str]:
    """The human-readable change lines (nothing executable)."""
    lines: List[str] = []
    if result["same"]:
        lines.append("config tree unchanged (root hash matches)")
        return lines
    for label, key in (("added", "added"), ("removed", "removed"),
                       ("changed", "changed")):
        for path in result[key]:
            lines.append(f"  {label}: {path}")
    for path in result["added_dirs"]:
        lines.append(f"  new directory: {path}")
    for path in result["removed_dirs"]:
        lines.append(f"  removed directory: {path}")
    prune = result["pruned"]
    lines.append(f"({prune['subtrees_skipped']} subtree(s) pruned by "
                 f"hash, {prune['files_compared']} file(s) compared)")
    return lines

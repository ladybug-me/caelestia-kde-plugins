"""devflow.todo — TODO/FIXME triage over a source tree, via scan's Aho-Corasick.

The scan layer's automaton was built for one-pass matching of many known
signatures over huge logs; a source tree is the same problem with fewer
lines. This module reuses it exactly (no second matcher): one automaton
over the marker vocabulary, one scan per file, marker + line number +
the comment's own text out the other end.

Bounds: file count capped (default 2000), per-file size capped, symlinks
never followed, hidden and build directories skipped. Reads only what the
caller names; writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from assistant.capabilities.scan import Automaton

__all__ = ["MARKERS", "triage_tree", "render", "MAX_FILES", "MAX_FILE_BYTES"]

MARKERS: Tuple[str, ...] = ("TODO", "FIXME", "XXX", "HACK", "BUG")
MAX_FILES = 2000
MAX_FILE_BYTES = 2_000_000
SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules",
             ".venv", "venv", "build", "dist", ".mypy_cache"}

_EXTENSIONS = (".py", ".qml", ".js", ".ts", ".sh", ".c", ".cpp", ".cc",
               ".hpp", ".h", ".rs", ".go", ".java", ".md", ".json",
               ".yaml", ".yml", ".toml")


def triage_tree(root: str, markers: Tuple[str, ...] = MARKERS,
                extensions: Tuple[str, ...] = _EXTENSIONS) -> Dict[str, Any]:
    """Walk ``root`` (bounded) and report every marker line, grouped for
    triage: {"root", "files", "items": [{file, line, marker, text}],
    "by_marker": {marker: count}, "skipped": [...]}."""
    root_path = Path(root)
    automaton = Automaton(list(markers))
    items: List[Dict[str, Any]] = []
    files = 0
    skipped: List[str] = []
    if root_path.is_file():
        candidates = [root_path]
    else:
        candidates = sorted(
            p for p in root_path.rglob("*")
            if p.is_file() and not any(part in SKIP_DIRS
                                       for part in p.parts))
    for path in candidates:
        if files >= MAX_FILES:
            skipped.append(f"file cap reached ({MAX_FILES})")
            break
        if path.suffix.lower() not in extensions:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped.append(f"too large: {path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append(f"unreadable: {path} ({exc.__class__.__name__})")
            continue
        files += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            hits = automaton.scan(line)
            if not hits:
                continue
            # scan() returns (end_index_exclusive, pid); recover the
            # marker's original spelling from the line, group canonically
            end, pid = min(hits, key=lambda h: (h[0], -len(automaton.pattern(h[1]))))
            marker_len = len(automaton.pattern(pid))
            marker = line[max(0, end - marker_len):end].upper()
            tail = line[end:].strip().lstrip(":-— ").strip()
            items.append({
                "file": str(path.relative_to(root_path)),
                "line": lineno,
                "marker": marker,
                "text": tail[:100],
            })
    by_marker: Dict[str, int] = {m: 0 for m in markers}
    for item in items:
        by_marker[item["marker"]] = by_marker.get(item["marker"], 0) + 1
    return {"root": str(root), "files": files, "items": items,
            "by_marker": by_marker, "skipped": skipped}


def render(report: Dict[str, Any]) -> str:
    """Plain-text triage list, grouped by marker, most numerous first."""
    if not report["items"]:
        return (f"no TODO/FIXME markers under {report['root']} "
                f"({report['files']} file(s) scanned)")
    lines = [f"{len(report['items'])} marker(s) in {report['files']} "
             f"file(s) under {report['root']}"]
    order = sorted(report["by_marker"].items(),
                   key=lambda kv: (-kv[1], kv[0]))
    for marker, count in order:
        if not count:
            continue
        lines.append("")
        lines.append(f"{marker} ({count}):")
        for item in report["items"]:
            if item["marker"] == marker:
                lines.append(f"  {item['file']}:{item['line']}  {item['text']}")
    if report["skipped"]:
        lines.append("")
        lines.append("skipped: " + "; ".join(report["skipped"][:5]))
    return "\n".join(lines)

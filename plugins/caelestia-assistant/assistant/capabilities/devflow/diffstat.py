"""devflow.diffstat — commit-message and PR skeletons from git diff stats.

Input contract (caller-supplied text, never a subprocess — the import
policy forbids one anyway): the standard ``git diff --numstat`` line
format, tab-separated::

    <added>\t<deleted>\t<path>

with the usual special cases honored: ``-`` for binary files and
``{old => new}`` renames. Everything derived below is a mechanical
template over those numbers and paths — the tool never generates prose
and never invents a "why"; the human supplies that.

Classification (deterministic keyword table, most-specific rule wins):

======================  ==========================================
rule                    conventional type
======================  ==========================================
path under tests/ or    test
matching test_*.py
*.md / docs/ / LICENSE  docs
.github/ / *ci* /       ci
  scripts/ci*
new file with only      feat
  additions
deletions dominate      refactor
(>= 2:1 del:add)        chore (fallback)
======================  ==========================================

Scope: the most common top-level directory among changed files, when one
directory covers at least half of them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

__all__ = ["parse_numstat", "classify_change", "dominant_type",
           "commit_message", "pr_skeleton", "subject_limit"]

SUBJECT_LIMIT = 72  # the classic git convention

_NUMSTAT_RE = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
_RENAME_RE = re.compile(r"\{(.+) => (.+)\}")


def parse_numstat(numstat_text: str) -> List[Dict[str, object]]:
    """``git diff --numstat`` text -> one row per file:
    {"path", "added", "deleted", "binary", "renamed_from"}."""
    rows: List[Dict[str, object]] = []
    for line in numstat_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _NUMSTAT_RE.match(line)
        if not match:
            continue
        added_s, deleted_s, raw_path = match.groups()
        binary = added_s == "-" or deleted_s == "-"
        renamed_from: Optional[str] = None
        rename = _RENAME_RE.search(raw_path)
        if rename:
            renamed_from = _RENAME_RE.sub(rename.group(1), raw_path, count=1)
        rows.append({
            "path": raw_path,
            "renamed_from": renamed_from,
            "added": 0 if binary else int(added_s),
            "deleted": 0 if binary else int(deleted_s),
            "binary": binary,
        })
    return rows


def classify_change(row: Dict[str, object]) -> str:
    """The conventional-commit type one file most likely belongs to
    (deterministic keyword rules; the least-specific rule is the honest
    fallback, never a guess about intent)."""
    path = str(row.get("path", "")).lower()
    if "/tests/" in f"/{path}" or path.startswith("tests/") \
            or re.search(r"(^|/)(test_[^/]+\.py|[^/]+_test\.[a-z]+)$", path):
        return "test"
    if path.endswith((".md", ".rst", ".txt")) or path.startswith(("docs/", "doc/")) \
            or "/license" in path:
        return "docs"
    if ".github/" in path or "/ci/" in path or "workflow" in path:
        return "ci"
    if row.get("renamed_from") is None and row.get("added", 0) > 0 \
            and row.get("deleted", 0) == 0:
        return "feat"
    if row.get("deleted", 0) >= 2 * max(1, row.get("added", 0)) \
            and row.get("deleted", 0) > 0:
        return "refactor"
    return "chore"


def dominant_type(rows: List[Dict[str, object]]) -> str:
    """The most common classification, ties broken alphabetically —
    deterministic under every reordering of the input."""
    counts = Counter(classify_change(row) for row in rows)
    if not counts:
        return "chore"
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _scope(rows: List[Dict[str, object]]) -> Optional[str]:
    """The common top-level directory when one covers >= half the files."""
    if not rows:
        return None
    tops: Counter = Counter()
    for row in rows:
        path = str(row.get("path", ""))
        parts = path.split("/")
        if len(parts) > 1 and parts[0] not in (".", "..", ""):
            tops[parts[0]] += 1
    if not tops:
        return None
    name, count = sorted(tops.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return name if count >= (len(rows) + 1) // 2 else None


def _area(path: str) -> str:
    """The area a file belongs to: first directory, else the file itself."""
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def commit_message(numstat_text: str) -> str:
    """A deterministic commit-message SKELETON from numstat text:
    conventional-commit header with real deltas, body grouped by area.
    The ``<describe the what and why>`` slots are the human's to fill."""
    rows = parse_numstat(numstat_text)
    if not rows:
        return "(no changes in the numstat input)"
    kind = dominant_type(rows)
    scope = _scope(rows)
    added = sum(int(r["added"]) for r in rows)
    deleted = sum(int(r["deleted"]) for r in rows)
    header_scope = f"({scope})" if scope else ""
    subject = (f"{kind}{header_scope}: <describe the what and why> "
               f"[{len(rows)} file(s), +{added}/-{deleted}]")
    if len(subject) > SUBJECT_LIMIT:
        subject = (f"{kind}{header_scope}: <describe the what and why> "
                   f"[{len(rows)} files +{added}/-{deleted}]")

    areas: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        areas.setdefault(_area(str(row["path"])), []).append(row)

    lines = [subject, "",
             "<why this change — one or two sentences a reviewer needs>"]
    for area in sorted(areas):
        area_rows = areas[area]
        area_add = sum(int(r["added"]) for r in area_rows)
        area_del = sum(int(r["deleted"]) for r in area_rows)
        lines.append("")
        lines.append(f"- {area}: {len(area_rows)} file(s) "
                     f"(+{area_add}/-{area_del})")
        for row in area_rows[:8]:
            note = " (binary)" if row.get("binary") else ""
            rename = ""
            if row.get("renamed_from"):
                rename = f" (renamed from {row['renamed_from']})"
            lines.append(f"    {row['path']}{note}{rename}")
        if len(area_rows) > 8:
            lines.append(f"    ... and {len(area_rows) - 8} more")
    return "\n".join(lines)


def pr_skeleton(numstat_text: str, base: str = "main",
                head: str = "dev") -> str:
    """A PR-description skeleton from the same stats, shaped like the
    upstream project's PR template (What does this change / How did you
    test it / Type / Notes), with the mechanical parts filled in and the
    judgment parts left as explicit placeholders. Deterministic."""
    rows = parse_numstat(numstat_text)
    if not rows:
        return "(no changes in the numstat input)"
    kind = dominant_type(rows)
    added = sum(int(r["added"]) for r in rows)
    deleted = sum(int(r["deleted"]) for r in rows)
    areas: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        areas.setdefault(_area(str(row["path"])), []).append(row)

    type_map = {
        "feat": "New feature", "fix": "Bug fix", "test": "Other",
        "docs": "Docs", "ci": "Other", "refactor": "Refactor",
        "chore": "Other",
    }
    out: List[str] = [
        f"## What does this change?",
        "",
        f"<one or two sentences> ({len(rows)} file(s), +{added}/-{deleted}, "
        f"`{base}...{head}`)",
        "",
        "## Changes by area",
        "",
    ]
    for area in sorted(areas):
        area_rows = areas[area]
        area_add = sum(int(r["added"]) for r in area_rows)
        area_del = sum(int(r["deleted"]) for r in area_rows)
        out.append(f"- **{area}**: {len(area_rows)} file(s) "
                   f"(+{area_add}/-{area_del})")
    out.extend([
        "",
        "## How did you test it?",
        "",
        "- [ ] <describe the verification you ran>",
        "",
        "## Type",
        "",
        f"- [ ] Bug fix",
        f"- [x] {type_map.get(kind, 'Other')}  (classified from the diff "
        f"stats: {kind}; uncheck and correct if wrong)",
        "- [ ] Config / theme",
        "- [ ] Docs",
        "- [ ] Refactor",
        "- [ ] Other",
        "",
        "## Notes for reviewers",
        "",
        "<anything tricky, trade-offs, context — or delete this section>",
        "",
    ])
    return "\n".join(out)

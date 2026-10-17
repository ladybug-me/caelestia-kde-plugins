"""shellkb.jsonmerge — structured diff and three-way merge for shell
config JSON (B9).

shell.json edits rarely happen in one place at one time: the assistant
proposes changes, the user edits by hand, an update ships new defaults —
and then three versions of one config exist. This module answers the
two questions that follow, deterministically and without writing
anything:

1. WHAT CHANGED between two documents?  The documents become ordered
   trees; a Myers diff (O(ND), the same algorithm behind GNU diff)
   runs over each tree's flattened key-paths, and a Zhang-Shasha
   ordered tree-edit distance summarizes the result as N inserts +
   M deletes + K relabels (bounded: documents past the node cap get an
   honest ABSTAIN rather than a slow guess).

2. CAN THE THREE VERSIONS COMBINE?  A path-wise three-way merge over
   the flattened (dotted-path -> value) views: one side changed,
   the other did not -> take the change; both changed identically ->
   take it once; both changed differently -> a CONFLICT with the three
   values shown and a typed explanation (value-vs-value, add-vs-add,
   delete-vs-edit). The merged document and its conflicts are OUTPUT,
   a proposal — shell.json is only ever written through the settings
   layer's own --apply gate.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["flatten", "myers_ops", "tree_edit_distance", "diff_docs",
           "merge_three_way", "render_diff", "render_merge"]

# documents bigger than this get an honest ABSTAIN on the TED metric
_TED_NODE_CAP = 2000
_MAX_DEPTH = 24

# the absent marker: identity-compared, so a REAL JSON null is never
# confused with 'the key is not there'
class _Absent:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "<absent>"


_ABSENT = _Absent()


def _get(flat: Dict[str, Any], path: str) -> Any:
    return flat.get(path, _ABSENT)


# ---------------------------------------------------------------------------
# flattening: dotted paths in deterministic order
# ---------------------------------------------------------------------------

def flatten(doc: Any, prefix: str = "", depth: int = 0) \
        -> List[Tuple[str, Any]]:
    """The document as sorted [(dotted.path, leaf-value)]. Lists index
    as .0/.1 — merge decisions are per leaf; structure disagreements
    show up as add/delete of indexed paths. Depth is capped."""
    rows: List[Tuple[str, Any]] = []
    if depth > _MAX_DEPTH:
        return rows
    if isinstance(doc, dict):
        for key in sorted(doc):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten(doc[key], path, depth + 1))
    elif isinstance(doc, list):
        for i, item in enumerate(doc):
            path = f"{prefix}.{i}" if prefix else str(i)
            rows.extend(flatten(item, path, depth + 1))
    else:
        rows.append((prefix, doc))
    return rows


# ---------------------------------------------------------------------------
# Myers diff over the path sequence (O(ND) with full backtrack)
# ---------------------------------------------------------------------------

def myers_ops(a: List[str], b: List[str]) -> List[Tuple[str, int, int]]:
    """Minimal edit script as ops ('=', '-', '+') with (i, j) indices
    into a/b. Deterministic: ties prefer deletions first, matching
    classic diff's reading of 'the line moved'."""
    n, m = len(a), len(b)
    max_d = n + m
    v = {1: 0}
    trace: List[Dict[int, int]] = []
    found_d = None
    for d in range(max_d + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v.get(k - 1, -max_d) <
                           v.get(k + 1, -max_d)):
                x = v.get(k + 1, 0)
            else:
                x = v.get(k - 1, 0) + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x, y = x + 1, y + 1
            v[k] = x
            if x >= n and y >= m:
                found_d = d
                break
        if found_d is not None:
            break
    if found_d is None:  # pragma: no cover - d always reaches n+m
        found_d = max_d
    # backtrack
    ops: List[Tuple[str, int, int]] = []
    x, y = n, m
    for d in range(found_d, 0, -1):
        vprev = trace[d]
        k = x - y
        if k == -d or (k != d and vprev.get(k - 1, -max_d) <
                       vprev.get(k + 1, -max_d)):
            prev_k = k + 1
        else:
            prev_k = k - 1
        prev_x = vprev.get(prev_k, 0)
        prev_y = prev_x - prev_k
        while x > prev_x and y > prev_y and x > 0 and y > 0:
            ops.append(("=", x - 1, y - 1))
            x, y = x - 1, y - 1
        if prev_k == k + 1:  # insertion (from b)
            ops.append(("+", x - 1, y - 1))
            y -= 1
        else:                # deletion (from a)
            ops.append(("-", x - 1, y - 1))
            x -= 1
    while x > 0 and y > 0 and a[x - 1] == b[y - 1]:
        ops.append(("=", x - 1, y - 1))
        x, y = x - 1, y - 1
    while y > 0:
        ops.append(("+", x - 1, y - 1))
        y -= 1
    while x > 0:
        ops.append(("-", x - 1, y - 1))
        x -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# tree edit distance (Zhang-Shasha, ordered labeled trees)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label: str, children: List["_Node"]) -> None:
        self.label = label
        self.children = children


def _to_tree(doc: Any, label: str = "", depth: int = 0,
             count: List[int] = None) -> _Node:
    """JSON value -> labeled ordered tree. Internal nodes carry the
    key; leaves carry key=value."""
    if count is None:
        count = [0]
    count[0] += 1
    if count[0] > _TED_NODE_CAP:
        raise ValueError("node cap exceeded")
    if not isinstance(doc, (dict, list)) or depth > _MAX_DEPTH:
        return _Node(f"{label}={doc!r}" if label else repr(doc), [])
    children = []
    if isinstance(doc, dict):
        for key in sorted(doc):
            children.append(_to_tree(doc[key], str(key), depth + 1, count))
    else:
        for i, item in enumerate(doc):
            children.append(_to_tree(item, str(i), depth + 1, count))
    return _Node(label, children)


def _postorder(node: _Node) -> List[_Node]:
    out: List[_Node] = []
    for child in node.children:
        out.extend(_postorder(child))
    out.append(node)
    return out


def _subtree_size(node: _Node) -> int:
    return 1 + sum(_subtree_size(c) for c in node.children)


class _TED:
    """Ordered tree-edit distance, unit costs (insert subtree, delete
    subtree, relabel node), via the classic recursion over child
    prefixes with pair memoization — the same optimum Zhang-Shasha
    computes, expressed recursively because it is verifiable line by
    line: two subtrees' children are matched in ORDER by an LCS-style
    DP whose third option recurses."""

    def __init__(self, pa: List[_Node], pb: List[_Node]) -> None:
        self.pa = pa
        self.pb = pb
        self.pos_a = {id(n): i for i, n in enumerate(pa)}
        self.pos_b = {id(n): i for i, n in enumerate(pb)}
        self.sizes = {id(n): _subtree_size(n) for n in pa}
        self.sizes.update({id(n): _subtree_size(n) for n in pb})
        self.memo: Dict[Tuple[int, int], int] = {}

    def td(self, ai: int, bi: int) -> int:
        key = (ai, bi)
        if key in self.memo:
            return self.memo[key]
        a, b = self.pa[ai], self.pb[bi]
        ca, cb = a.children, b.children
        m, n = len(ca), len(cb)
        # dp[i][j]: forest distance over the first i children of a and
        # the first j children of b
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        sizes_a = [self.sizes[id(c)] for c in ca]
        sizes_b = [self.sizes[id(c)] for c in cb]
        pairs_a = [self.pos_a[id(c)] for c in ca]
        pairs_b = [self.pos_b[id(c)] for c in cb]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + sizes_a[i - 1]
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + sizes_b[j - 1]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                child_pair = self.td(pairs_a[i - 1], pairs_b[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + sizes_a[i - 1],
                    dp[i][j - 1] + sizes_b[j - 1],
                    dp[i - 1][j - 1] + child_pair)
        result = dp[m][n] + (0 if a.label == b.label else 1)
        self.memo[key] = result
        return result


def tree_edit_distance(doc_a: Any, doc_b: Any) -> Optional[int]:
    """Ordered TED between two JSON docs (unit insert/delete/relabel);
    None (ABSTAIN) past the node cap."""
    try:
        ta = _to_tree(doc_a)
        tb = _to_tree(doc_b)
    except ValueError:
        return None
    solver = _TED(_postorder(ta), _postorder(tb))
    root_a = len(solver.pa) - 1
    root_b = len(solver.pb) - 1
    return solver.td(root_a, root_b)


# ---------------------------------------------------------------------------
# the two operations
# ---------------------------------------------------------------------------

def diff_docs(a: Any, b: Any) -> Dict[str, Any]:
    """Structured two-way diff: Myers over path sequences, TED summary."""
    fa, fb = flatten(a), flatten(b)
    paths_a = [p for p, _ in fa]
    paths_b = [p for p, _ in fb]
    values_a = dict(fa)
    values_b = dict(fb)
    ops = myers_ops(paths_a, paths_b)
    changed: List[Dict[str, Any]] = []
    for op, i, j in ops:
        if op == "=":
            if values_a[paths_a[i]] != values_b[paths_b[j]]:
                changed.append({"op": "changed", "path": paths_a[i],
                                "was": values_a[paths_a[i]],
                                "now": values_b[paths_b[j]]})
        elif op == "-":
            changed.append({"op": "removed", "path": paths_a[i],
                            "was": values_a[paths_a[i]]})
        elif op == "+":
            changed.append({"op": "added", "path": paths_b[j],
                            "now": values_b[paths_b[j]]})
    changed.sort(key=lambda c: (c["path"], c["op"]))
    ted = tree_edit_distance(a, b)
    return {
        "n_paths": {"a": len(fa), "b": len(fb)},
        "ted": ted,
        "ted_note": None if ted is not None else
        f"documents exceed {_TED_NODE_CAP} nodes: the tree-edit "
        f"distance abstains instead of guessing",
        "changes": changed,
        "n_changes": len(changed),
    }


def _entries_by_side(base: Any, ours: Any, theirs: Any) \
        -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return dict(flatten(base)), dict(flatten(ours)), dict(flatten(theirs))


def merge_three_way(base: Any, ours: Any, theirs: Any) -> Dict[str, Any]:
    """Path-wise three-way merge. Returns the merged document, the
    taken changes, and the conflicts (each with the three values and a
    typed explanation). A PROPOSAL: the caller decides what to do."""
    fb, fo, ft = _entries_by_side(base, ours, theirs)
    merged: Dict[str, Any] = {}
    taken: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    for path in sorted(set(fb) | set(fo) | set(ft)):
        b, o, t = _get(fb, path), _get(fo, path), _get(ft, path)
        if o is t or o == t:
            if o is not _ABSENT and o != b:
                taken.append({"path": path, "source": "both",
                              "value": o})
            if o is _ABSENT and b is not _ABSENT:
                taken.append({"path": path, "source": "both",
                              "value": "deleted"})
            merged = _set_path(merged, path, o)
            continue
        if o is b or o == b:  # only theirs moved
            taken.append({"path": path, "source": "theirs",
                          "value": "deleted" if t is _ABSENT else t})
            merged = _set_path(merged, path, t)
            continue
        if t is b or t == b:  # only ours moved
            taken.append({"path": path, "source": "ours",
                          "value": "deleted" if o is _ABSENT else o})
            merged = _set_path(merged, path, o)
            continue
        # both moved, differently: the honest conflict. The merged
        # proposal keeps the BASE value (or nothing for add-vs-add) so
        # what it proposes is always a valid document on top of base.
        kind = ("add-vs-add" if b is _ABSENT else
                "delete-vs-edit" if _ABSENT in (o, t) else
                "value-vs-value")
        if b is not _ABSENT:
            merged = _set_path(merged, path, b)
        conflicts.append({
            "path": path,
            "kind": kind,
            "base": None if b is _ABSENT else b,
            "ours": None if o is _ABSENT else o,
            "theirs": None if t is _ABSENT else t,
            "base_present": b is not _ABSENT,
            "ours_present": o is not _ABSENT,
            "theirs_present": t is not _ABSENT,
            "explanation": _conflict_text(kind, path, o, t),
        })
    return {
        "n_paths": {"base": len(fb), "ours": len(fo),
                    "theirs": len(ft)},
        "merged": merged,
        "taken": taken,
        "n_taken": len(taken),
        "conflicts": conflicts,
        "n_conflicts": len(conflicts),
        "note": "a merge PROPOSAL — nothing was written anywhere",
    }


def _conflict_text(kind: str, path: str, o: Any, t: Any) -> str:
    if kind == "add-vs-add":
        return (f"both sides added '{path}' with different values "
                f"(ours={o!r}, theirs={t!r}): pick one")
    if kind == "delete-vs-edit":
        side = "ours" if o is _ABSENT else "theirs"
        return (f"one side deleted '{path}' while the other edited it "
                f"({side} kept a value): deletion or edit, not both")
    return (f"'{path}' was changed to {o!r} by ours and to {t!r} by "
            f"theirs: pick a value")


def _set_path(doc: Any, path: str, value: Any) -> Any:
    """Write (or remove) one dotted path in the merged tree. '' names
    the root itself (scalar documents). Removal drops the key/entry."""
    if not path:
        return None if value is _ABSENT else value  # root scalar
    parts = path.split(".")
    node: Any = doc
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        want_list = nxt.isdigit()
        if isinstance(node, dict):
            if part not in node or not isinstance(node[part],
                                                  (dict, list)):
                node[part] = [] if want_list else {}
            node = node[part]
        elif isinstance(node, list):
            idx = int(part) if part.isdigit() else 0
            while len(node) <= idx:
                node.append([] if want_list else {})
            node = node[idx]
        else:
            return doc  # a scalar in the way: the path is unreachable
    last = parts[-1]
    if value is _ABSENT:
        if isinstance(node, dict) and last in node:
            del node[last]
        elif isinstance(node, list) and last.isdigit() and \
                int(last) < len(node):
            del node[int(last)]
        return doc
    if isinstance(node, dict):
        node[last] = value
    elif isinstance(node, list) and last.isdigit():
        idx = int(last)
        while len(node) <= idx:
            node.append(None)
        node[idx] = value
    return doc


def render_diff(data: Dict[str, Any]) -> List[str]:
    lines = [f"structured diff: {data['n_paths']['a']} -> "
             f"{data['n_paths']['b']} paths; "
             f"tree-edit distance "
             f"{'ABSTAIN' if data['ted'] is None else data['ted']}"
             f"{'' if data['ted'] is None else ' node edits'}"]
    if data.get("ted_note"):
        lines.append(f"  ({data['ted_note']})")
    for c in data["changes"]:
        if c["op"] == "changed":
            lines.append(f"  ~ {c['path']}: {c['was']!r} -> {c['now']!r}")
        elif c["op"] == "removed":
            lines.append(f"  - {c['path']} (was {c['was']!r})")
        else:
            lines.append(f"  + {c['path']} = {c['now']!r}")
    if not data["changes"]:
        lines.append("  (identical)")
    return lines


def render_merge(data: Dict[str, Any]) -> List[str]:
    lines = [f"three-way merge PROPOSAL — {data['n_taken']} changes "
             f"taken, {data['n_conflicts']} conflicts; nothing written"]
    for t in data["taken"]:
        lines.append(f"  take [{t['source']}] {t['path']} = {t['value']!r}")
    for c in data["conflicts"]:
        lines.append(f"  CONFLICT [{c['kind']}] {c['path']}")
        if c.get("base_present"):
            lines.append(f"    base:   {c['base']!r}")
        lines.append(f"    ours:   {c['ours']!r}"
                     f"{'' if c['ours_present'] else ' (deleted)'}")
        lines.append(f"    theirs: {c['theirs']!r}"
                     f"{'' if c['theirs_present'] else ' (deleted)'}")
        lines.append(f"    {c['explanation']}")
    if not data["taken"] and not data["conflicts"]:
        lines.append("  (all three documents agree)")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import sys
    from pathlib import Path
    ap = argparse.ArgumentParser(
        prog="caelestia-assist shellkb",
        description="structured diff / three-way merge for config JSON "
                    "(read-only; prints proposals, never writes)")
    sub = ap.add_subparsers(dest="verb")
    d = sub.add_parser("diff", help="diff two JSON files")
    d.add_argument("a")
    d.add_argument("b")
    m = sub.add_parser("merge", help="three-way merge (proposal only)")
    m.add_argument("--base", required=True)
    m.add_argument("--ours", required=True)
    m.add_argument("--theirs", required=True)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.verb == "diff":
        a = json.loads(Path(args.a).read_text())
        b = json.loads(Path(args.b).read_text())
        for line in render_diff(diff_docs(a, b)):
            print(line)
        return 0
    if args.verb == "merge":
        base = json.loads(Path(args.base).read_text())
        ours = json.loads(Path(args.ours).read_text())
        theirs = json.loads(Path(args.theirs).read_text())
        for line in render_merge(merge_three_way(base, ours, theirs)):
            print(line)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

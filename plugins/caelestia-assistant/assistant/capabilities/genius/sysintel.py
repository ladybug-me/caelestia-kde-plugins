"""genius.sysintel — system intelligence, strictly read-only.

Nothing in this module writes, executes, or deletes anything. It reads:

  * shell history (bash/zsh/fish formats) and mines it: frecency
    ranking, hour-of-day heat, co-occurrence association rules with
    lift, and a next-command Markov predictor
  * duplicate files: same-size shortlist then SHA-256 confirmation
    (streamed, bounded memory)
  * disk usage hotspots: recursive size aggregation with a depth cut
    and the biggest-wins tree
  * log files: Drain-style template mining (token masking -> template
    clustering -> frequency z-scores for template-level anomalies;
    exponential-build-4 adds the streaming fixed-depth DRAIN TREE
    proper — diagnostics/drain.py, wired in front of the rule
    engine — this function remains the batch z-score view; the two
    are different mechanisms, not duplicates)
  * JSON configs: duplicate keys, type oddities, NaN/Infinity,
    deep-shadowed settings

and proposes — through the caller's ledger — what a careful human
operator would do next. It never does it itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "parse_shell_history", "analyze_history", "find_duplicates",
    "disk_hotspots", "mine_log_templates", "lint_json_config",
]


# ---------------------------------------------------------------------------
# Shell history
# ---------------------------------------------------------------------------

_HISTORY_PATTERNS = [
    re.compile(r"^: \d+:\d+;(.*)$"),          # zsh EXTENDED_HISTORY
    re.compile(r"^- cmd: (.*)$", ),           # fish
]


def parse_shell_history(path: str) -> Dict[str, Any]:
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        return {"commands": [], "n_lines": 0, "format": "unknown",
                "note": f"no history file at {path}"}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    commands: List[str] = []
    stamps: List[Optional[int]] = []
    fmt = "bash"
    for line in lines:
        if not line.strip():
            continue
        matched = False
        for pat in _HISTORY_PATTERNS:
            m = pat.match(line)
            if m:
                fmt = "zsh" if pat.pattern.startswith(":") else "fish"
                commands.append(m.group(1))
                if fmt == "zsh":
                    try:
                        stamps.append(int(line.split(":")[1]))
                    except (ValueError, IndexError):
                        stamps.append(None)
                matched = True
                break
        if not matched and line.startswith("- cmd:") is False and line.startswith(":") is False:
            commands.append(line)
            stamps.append(None)
    return {"commands": commands, "n_lines": len(lines), "format": fmt,
            "timestamps": stamps, "path": str(p)}


def _base_command(cmd: str) -> str:
    """First two tokens (sudo stripped) — the 'what am I doing' key."""
    tokens = cmd.strip().split()
    while tokens and tokens[0] in ("sudo", "env", "nohup", "\\"):
        tokens.pop(0)
        if tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens.pop(0)
    if not tokens:
        return ""
    head = tokens[0]
    arg = tokens[1] if len(tokens) > 1 and not tokens[1].startswith("-") else ""
    return f"{head} {arg}".strip()


def analyze_history(commands: Sequence[str], timestamps: Optional[Sequence[Optional[int]]] = None,
                    top: int = 10) -> Dict[str, Any]:
    if not commands:
        raise ValueError("no commands to analyze")
    from .markov import predict_next, sequence_surprise
    bases = [_base_command(c) for c in commands]
    bases = [b for b in bases if b]
    counts = Counter(bases)
    n = len(bases)
    # frecency: frequency + recency boost (0.5^(age/n))
    freq = {}
    for i, b in enumerate(bases):
        age = n - i
        freq[b] = freq.get(b, 0.0) + 1 + 0.5 ** (age / max(1, n * 0.25))
    frecency_rank = sorted(freq.items(), key=lambda kv: -kv[1])[:top]
    # hour-of-day heat
    hours = []
    if timestamps:
        for ts in timestamps:
            if ts:
                hours.append(datetime.fromtimestamp(ts).hour)
    hour_hist = Counter(hours) if hours else {}
    peak_hour = hour_hist.most_common(1)[0][0] if hour_hist else None
    # association rules: cmd_a -> cmd_b within the same session window (5 cmds)
    pairs = Counter()
    single = Counter()
    window = 5
    for i in range(len(bases)):
        single[bases[i]] += 1
        for j in range(i + 1, min(i + window, len(bases))):
            if bases[i] != bases[j]:
                pairs[(bases[i], bases[j])] += 1
    rules = []
    for (a, b), c in pairs.most_common(500):
        if c < 3:
            continue
        support = c / n
        conf = c / single[a]
        lift = conf / (single[b] / n) if single[b] else 0
        if lift > 1.2 and conf > 0.15:
            rules.append({"after": a, "comes": b, "support": round(support, 4),
                          "confidence": round(conf, 3), "lift": round(lift, 2)})
        if len(rules) >= 10:
            break
    nxt = predict_next(bases[-4:]) if len(bases) > 4 else None
    surprise = None
    if len(bases) >= 12:
        surprise = [s for s in sequence_surprise(bases)["anomalies"]][-3:]
    return {"n_commands": len(commands),
            "n_distinct": len(counts),
            "top_commands": [{"command": b, "count": counts[b],
                              "share": round(counts[b] / n, 3)} for b, _ in counts.most_common(top)],
            "frecency_rank": [{"command": b, "score": round(s, 2)} for b, s in frecency_rank],
            "peak_hour": peak_hour,
            "hour_histogram": dict(sorted(hour_hist.items())) if hour_hist else None,
            "association_rules": rules,
            "next_command": nxt,
            "recent_surprises": surprise,
            "note": "history mining is read-only; suggestions go through the ledger"}


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def find_duplicates(root: str, min_size: int = 1024,
                    max_files: int = 20000) -> Dict[str, Any]:
    """Duplicate detection by size shortlist + streamed SHA-256."""
    base = Path(os.path.expanduser(root))
    if not base.is_dir():
        raise ValueError(f"{root} is not a directory")
    by_size: Dict[int, List[str]] = {}
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in
                       (".git", "node_modules", "__pycache__", ".cache")]
        for name in filenames:
            try:
                full = Path(dirpath) / name
                size = full.stat().st_size
            except OSError:
                continue
            scanned += 1
            if size >= min_size:
                by_size.setdefault(size, []).append(str(full))
            if scanned >= max_files:
                break
        if scanned >= max_files:
            break
    candidates = [files for files in by_size.values() if len(files) > 1]
    hashes: Dict[str, List[str]] = {}
    hashed = 0
    for group in candidates:
        for f in group:
            try:
                hashes.setdefault(_sha256(Path(f)), []).append(f)
                hashed += 1
            except OSError:
                continue
    dups = {h: files for h, files in hashes.items() if len(files) > 1}
    wasted = sum(_size_of(files[0]) * (len(files) - 1)
                 for files in dups.values() if files)
    groups = [{"size_bytes": _size_of(files[0]), "count": len(files),
               "files": sorted(files)} for files in dups.values()]
    groups.sort(key=lambda g: -g["size_bytes"] * g["count"])
    return {"root": str(base), "scanned_files": scanned, "hashed_files": hashed,
            "duplicate_groups": len(groups),
            "wasted_bytes": wasted,
            "wasted_human": _human_bytes(wasted),
            "groups": groups[:20],
            "note": "read-only scan; deletion is a human decision, not ours"}


def _size_of(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Disk hotspots
# ---------------------------------------------------------------------------

def disk_hotspots(root: str, depth: int = 3, top: int = 12) -> Dict[str, Any]:
    """Recursive directory sizes, cut at `depth`, biggest branches first."""
    base = Path(os.path.expanduser(root))
    if not base.is_dir():
        raise ValueError(f"{root} is not a directory")

    def dir_size(path: Path, cur_depth: int) -> Tuple[int, Dict[str, int]]:
        total = 0
        children: Dict[str, int] = {}
        try:
            entries = list(path.iterdir())
        except (OSError, PermissionError):
            return 0, {}
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    if cur_depth < depth:
                        sub_total, _ = dir_size(entry, cur_depth + 1)
                    else:
                        sub_total = _flat_size(entry, budget=3000)
                    children[str(entry)] = sub_total
                    total += sub_total
            except (OSError, PermissionError):
                continue
        return total, children

    total, children = dir_size(base, 0)
    ranked = sorted(children.items(), key=lambda kv: -kv[1])[:top]
    return {"root": str(base), "total_bytes": total,
            "total_human": _human_bytes(total),
            "hotspots": [{"path": p, "bytes": s, "human": _human_bytes(s),
                          "share": round(s / total, 4) if total else 0.0}
                         for p, s in ranked],
            "note": "sizes are cumulative; the top list is where the bytes are"}


def _flat_size(path: Path, budget: int = 5000) -> int:
    total = 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
            count += 1
            if count >= budget:
                return total
    return total


# ---------------------------------------------------------------------------
# Log template mining (Drain-style)
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^\d+$")
_FLOAT_RE = re.compile(r"^\d+\.\d+$")
_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def mine_log_templates(lines: Sequence[str], max_tokens: int = 32) -> Dict[str, Any]:
    """Cluster log lines into templates; flag frequency anomalies."""
    if not lines:
        raise ValueError("no log lines")
    templates: List[Dict[str, Any]] = []
    counts: List[int] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        toks = line.split()[:max_tokens]
        sig = tuple(_mask(t) for t in toks)
        matched = False
        for i, t in enumerate(templates):
            if t["signature"] == sig:
                counts[i] += 1
                matched = True
                break
        if not matched:
            templates.append({"signature": sig, "example": line})
            counts.append(1)
    total = sum(counts) or 1
    mean = sum(counts) / len(counts)
    sd = (sum((c - mean) ** 2 for c in counts) / max(1, len(counts) - 1)) ** 0.5
    out = []
    for t, c in zip(templates, counts):
        z = (c - mean) / sd if sd > 0 else 0.0
        out.append({"template": " ".join(t["signature"]),
                    "example": t["example"], "count": c,
                    "share": round(c / total, 4), "z_score": round(z, 2)})
    out.sort(key=lambda t: -t["count"])
    rare = [t for t in out if t["count"] <= 1]
    dominant = [t for t in out if t["z_score"] > 1.5]
    return {"n_lines": len(lines), "n_templates": len(templates),
            "compression_ratio": round(len(templates) / max(1, len(lines)), 4),
            "templates": out[:20],
            "rare_templates": rare[:10],
            "dominant_templates": dominant[:5],
            "note": "templates generalize your logs; rare ones deserve eyes"}


def _mask(token: str) -> str:
    if _NUMBER_RE.match(token) or _FLOAT_RE.match(token):
        return "<N>"
    if _HEX_RE.match(token) or _UUID_RE.match(token):
        return "<ID>"
    if _IP_RE.match(token):
        return "<IP>"
    # host:port and IPv4:port forms
    if ":" in token:
        head, _, tail = token.rpartition(":")
        if _IP_RE.match(head) and tail.isdigit():
            return "<IP:PORT>"
        if head and tail.isdigit() and "." in head:
            return "<HOST:PORT>"
    if len(token) > 24 and any(c.isdigit() for c in token):
        return "<LONG>"
    if token.count("/") >= 2:
        return "<PATH>"
    return token


# ---------------------------------------------------------------------------
# JSON config lint
# ---------------------------------------------------------------------------

def lint_json_config(path: str) -> Dict[str, Any]:
    """Read a JSON file and report what a careful reviewer would flag."""
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        raise ValueError(f"no config file at {path}")
    raw = p.read_text(encoding="utf-8", errors="replace")
    issues: List[Dict[str, Any]] = []
    try:
        # duplicate-key detection via object_pairs_hook
        def _hook(pairs):
            seen = {}
            for k, v in pairs:
                if k in seen:
                    issues.append({"kind": "duplicate_key", "key": k,
                                   "detail": f"later value silently overrides the earlier one"})
                seen[k] = v
            return seen
        data = json.loads(raw, object_pairs_hook=_hook,
                          parse_constant=lambda c: issues.append(
                              {"kind": "non_finite_number", "key": str(c),
                               "detail": f"{c} is not valid strict JSON"}) or float("nan"))
    except json.JSONDecodeError as exc:
        return {"path": str(p), "valid_json": False,
                "parse_error": str(exc), "issues": [],
                "note": "the file is not parseable JSON; nothing else checked"}
    _walk_json(data, "$", issues, seen_paths=set())
    return {"path": str(p), "valid_json": True,
            "top_level_type": type(data).__name__,
            "n_issues": len(issues), "issues": issues,
            "note": "read-only lint; fixes are proposed, never applied"}


def _walk_json(node: Any, path: str, issues: List[Dict[str, Any]],
               seen_paths: set) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{path}.{k}"
            if isinstance(v, (dict, list)) and child in seen_paths:
                issues.append({"kind": "repeated_structure", "key": child,
                               "detail": "same path shape occurs again — check for copy-paste"})
            seen_paths.add(child)
            if v is None:
                issues.append({"kind": "explicit_null", "key": child,
                               "detail": "explicit null — intentional or leftover?"})
            _walk_json(v, child, issues, seen_paths)
    elif isinstance(node, list):
        if len(node) > 1:
            kinds = {type(x).__name__ for x in node}
            if len(kinds) > 1:
                issues.append({"kind": "mixed_types", "key": path,
                               "detail": f"list mixes {sorted(kinds)}"})
        for i, v in enumerate(node[:200]):
            _walk_json(v, f"{path}[{i}]", issues, seen_paths)
    elif isinstance(node, float):
        if math.isnan(node) or math.isinf(node):
            issues.append({"kind": "non_finite_number", "key": path,
                           "detail": "NaN/Infinity in a config value"})

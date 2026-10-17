"""devflow.risk — a commit-risk score: McCabe cyclomatic complexity x
recent churn, both from CALLER-PIPED TEXT (the diffstat.py input
contract: no subprocess, ever — the import policy forbids one and the
caller already has the data)::

    git log --numstat > /tmp/log.txt
    python3 -m assistant.capabilities.devflow risk --source changed.py < /tmp/log.txt

The two factors:

- COMPLEXITY: cyclomatic complexity per function from the ``ast`` of
  the supplied Python source — 1 + decision points, counting each
  if/elif, loop, except handler, assert, ternary, comprehension clause
  and (on 3.10+) match case, and each additional and/or operand
  (McCabe 1976, "A Complexity Measure", IEEE TSE 4). The PEAK function
  complexity is the risk driver.
- CHURN: the ``git log --numstat`` text parsed into commits (newest
  first, the order git prints); per-file churn decays by commit
  recency with a fixed half-life (default 5 commits), so yesterday's
  hot file outweighs a touch from last month. The parse reuses
  ``diffstat.parse_numstat`` for the file rows — one numstat
  implementation, not two.

    risk = peak_cc * log2(1 + decayed_churn)

Deterministic, pure, dimensionless. The score is a REPORT for the
human's review, not a gate: nothing blocks, nothing executes. A
non-Python or unparsable source is an honest refusal of the complexity
half (churn is still reported) — never an invented number.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional

from .diffstat import parse_numstat

__all__ = ["cyclomatic_complexity", "parse_log_numstat", "recent_churn",
           "commit_risk"]

_COMMIT_RE = re.compile(r"^commit\s+([0-9a-f]{7,40})", re.I)
_RISK_TIERS = ((20.0, "low"), (60.0, "medium"))  # else high


def cyclomatic_complexity(source: str) -> Dict[str, Any]:
    """Per-function cyclomatic complexity (McCabe 1976) from the AST.

    Raises ValueError for unparsable source — the honest refusal, never
    a guessed score."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"source does not parse as Python: {exc}") from exc
    rows: List[Dict[str, Any]] = []

    def decisions(node: ast.AST) -> int:
        total = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                  ast.ExceptHandler, ast.IfExp, ast.Assert)):
                total += 1
            elif isinstance(child, ast.BoolOp):
                total += max(0, len(child.values) - 1)
            elif isinstance(child, ast.comprehension):
                total += 1 + len(child.ifs)
            elif hasattr(ast, "Match") and isinstance(child, ast.Match):
                total += len(child.cases)
        return total

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                # decisions belonging to NESTED functions are the
                # nested function's own; walk() recurses and the inner
                # entry carries them, so exclude nested defs here
                inner = sum(decisions(d) for d in ast.walk(child)
                            if isinstance(d, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))
                            and d is not child)
                rows.append({"name": name,
                             "cc": 1 + decisions(child) - inner,
                             "lineno": child.lineno})
                walk(child, prefix=name + ".")
            else:
                walk(child, prefix)

    walk(tree)
    rows.sort(key=lambda r: (-r["cc"], r["name"]))
    if not rows:
        raise ValueError("source defines no functions — nothing to score")
    return {
        "functions": rows,
        "peak_cc": rows[0]["cc"],
        "peak_function": rows[0]["name"],
        "n_functions": len(rows),
        "algorithm": ("cyclomatic complexity (McCabe 1976): 1 + decision "
                      "points, from the stdlib ast"),
    }


def parse_log_numstat(log_text: str) -> List[Dict[str, Any]]:
    """``git log --numstat`` text -> one entry per commit, newest first
    (the order git prints). File rows reuse diffstat.parse_numstat —
    the ONE numstat parser."""
    commits: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in log_text.splitlines():
        m = _COMMIT_RE.match(line.strip())
        if m:
            current = {"commit": m.group(1), "lines": []}
            commits.append(current)
            continue
        if current is not None and line.strip():
            current["lines"].append(line)
    for c in commits:
        c["rows"] = parse_numstat("\n".join(c.pop("lines")))
        c["added"] = sum(int(r["added"]) for r in c["rows"])
        c["deleted"] = sum(int(r["deleted"]) for r in c["rows"])
    return commits


def recent_churn(log_text: str, halflife_commits: int = 5
                 ) -> Dict[str, Any]:
    """Per-file churn with recency decay: commit i (0 = newest) weighs
    0.5 ** (i / halflife_commits). Deterministic; no timestamps needed
    (git's own newest-first order IS the recency signal)."""
    if halflife_commits <= 0:
        raise ValueError("halflife_commits must be >= 1")
    commits = parse_log_numstat(log_text)
    churn: Dict[str, float] = {}
    for idx, c in enumerate(commits):
        w = 0.5 ** (idx / float(halflife_commits))
        for r in c["rows"]:
            path = str(r["path"])
            churn[path] = churn.get(path, 0.0) + \
                (int(r["added"]) + int(r["deleted"])) * w
    files = sorted(({"path": p, "churn": round(v, 2)}
                    for p, v in churn.items()),
                   key=lambda d: (-d["churn"], d["path"]))
    return {"files": files, "total": round(sum(churn.values()), 2),
            "n_commits": len(commits),
            "halflife_commits": halflife_commits}


def commit_risk(log_text: str, source: str,
                source_before: Optional[str] = None,
                halflife_commits: int = 5) -> Dict[str, Any]:
    """The product the score is named for: peak cyclomatic complexity of
    the CURRENT source x log2(1 + decayed churn), tiered low/medium/
    high at stated thresholds. With ``source_before`` the CC delta is
    reported too. Unparsable source reports churn alone with the
    refusal named — the score stays None, never a guess."""
    churn = recent_churn(log_text, halflife_commits)
    out: Dict[str, Any] = {
        "churn": churn,
        "algorithm": ("risk = peak_cc (McCabe 1976) x log2(1 + "
                      "recency-decayed churn); a report, not a gate"),
    }
    try:
        cc = cyclomatic_complexity(source)
        cc_before = (cyclomatic_complexity(source_before)
                     if source_before is not None else None)
    except ValueError as exc:
        out.update({"risk": None, "tier": None,
                    "complexity_refused": str(exc),
                    "note": "churn reported alone — supply parsable "
                            "Python source for the complexity half"})
        return out
    delta = None
    if cc_before is not None:
        delta = cc["peak_cc"] - cc_before["peak_cc"]
    h = math.log2(1.0 + churn["total"])
    score = round(cc["peak_cc"] * h, 2)
    tier = "high"
    for bound, name in _RISK_TIERS:
        if score < bound:
            tier = name
            break
    out.update({
        "risk": score, "tier": tier,
        "peak_cc": cc["peak_cc"], "peak_function": cc["peak_function"],
        "cc_delta": delta,
        "churn_factor": round(h, 3),
        "thresholds": {"low_below": _RISK_TIERS[0][0],
                       "medium_below": _RISK_TIERS[1][0]},
    })
    return out

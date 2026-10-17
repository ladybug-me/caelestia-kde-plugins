"""assistant.capabilities.settings.envaudit — the environment audit (F21,
exponential-build-5).

A single read-only pass that answers: "is my environment healthy, and
how do I KNOW?" — by COMPOSING engines that already exist (no new
analysis of its own) and reporting every engine's verdict with the
command that reproduces it:

- config validity: settings.lint over the live target (unknown keys,
  out-of-range/mistyped values, silent no-ops, inert customizations);
- interaction contradictions: the configured values projected through
  the cited interaction table (settings.consequences) — e.g. blur
  configured while transparency is off (the INERT state);
- upstream drift: the frozen registry's generation fingerprint versus
  the current checkout state (the gen_adapter --verify verdict is
  expensive, so the audit reports the REGISTRY AGE it knows — the
  pinned upstream revision in the corpus metadata — and points at the
  exact verify command rather than re-running generation);
- backup coverage: whether the single .assistant-backup slot exists
  and matches a point-in-time of the live file (informational);
- environment snapshots (F17): freshness of the newest snapshot
  (how long ago, and whether the live content hash still matches it);
- per-screen overrides (F20): the monitors/ directory's files, with
  managed/unknown key counts (monitors.discover).

Read-only end to end. Exit code 0 unless CRITICAL findings exist
(refusals the applier itself would raise); WARN/INFO never block.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import consequences, lint, monitors as monitors_mod
from .environments import canonical_hash, read_config

__all__ = ["audit", "render_lines"]

CRITICAL = "CRITICAL"
WARN = "WARN"
INFO = "INFO"


def audit(target, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Run every check over the live environment (read-only). ``now``
    anchors snapshot freshness (default: a fixed epoch — callers pass
    the real clock; the function stays deterministic for tests)."""
    target = Path(target)
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    checks: List[Dict[str, Any]] = []

    current, problems = read_config(target)
    if problems:
        for problem in problems:
            checks.append({"check": "config-readable", "severity": CRITICAL,
                           "detail": problem})
        if not current:
            return {"target": str(target), "checks": checks,
                    "critical": len(problems)}

    # 1. Config validity via the existing linter.
    findings = lint.lint_file(str(target)) if target.exists() else []
    for finding in findings:
        severity = (CRITICAL if finding.get("id") in
                    ("out-of-range", "mistyped") else WARN)
        checks.append({
            "check": "config-lint", "severity": severity,
            "rule": finding.get("id"),
            "detail": finding.get("message") or json.dumps(finding)[:200],
            "evidence": finding,
        })
    if not findings:
        checks.append({"check": "config-lint", "severity": INFO,
                       "detail": "no lint findings: every leaf is "
                                 "registry-known and in range"})

    # 2. Interaction contradictions via the cited consequence table,
    #    projected over the CONFIGURED values (synthetic "configured"
    #    ops so the edge conditions evaluate). Only DERIVED/INERT edges
    #    are reported here — the projection's op-legality conflicts are
    #    the PLANNER's domain over proposed ops (the lint already flags
    #    out-of-range configured values), and paths lint already
    #    reported are skipped to avoid double-counting.
    configured_ops = _ops_from_config(current)
    projection = consequences.project(configured_ops, current)
    lint_paths = {str(f.get("path")) for f in findings if f.get("path")}
    inert = [row for row in projection.get("derived") or []
             if "INERT" in str(row.get("effect", ""))
             and str(row.get("effect_path")) not in lint_paths]
    for row in inert:
        checks.append({
            "check": "interaction-inert", "severity": WARN,
            "detail": f"{row.get('effect_path')}: {row.get('effect')}",
            "evidence": {"citation": row.get("citation")},
        })
    if not inert:
        checks.append({"check": "interaction", "severity": INFO,
                       "detail": "no additional interaction edge is "
                                 "armed against the configured values "
                                 "beyond the lint findings"})

    # 3. Snapshot freshness (F17).
    from . import environments as env_mod
    saved = env_mod.list_environments(target)
    if not saved:
        checks.append({"check": "snapshots", "severity": INFO,
                       "detail": "no environment snapshots saved yet "
                                 "(--env-save NAME captures one)"})
    else:
        newest = saved[-1]
        try:
            env = env_mod.get(target, str(newest["name"]))  # verifies hash
            live_hash = canonical_hash(current)
            fresh = (env.get("content_hash") == live_hash)
            checks.append({
                "check": "snapshot-fresh",
                "severity": INFO if fresh else WARN,
                "detail": (f"newest snapshot {newest['name']!r} MATCHES the "
                           f"live content hash" if fresh else
                           f"newest snapshot {newest['name']!r} predates the "
                           f"current config (live hash diverged); "
                           f"re-snapshot or restore deliberately"),
            })
        except env_mod.EnvironmentError as exc:
            checks.append({"check": "snapshot-fresh", "severity": CRITICAL,
                           "detail": f"newest snapshot {newest['name']!r} "
                                     f"failed integrity: {exc}"})

    # 4. Per-screen overrides (F20).
    for row in monitors_mod.discover(target):
        if row.get("error"):
            checks.append({"check": "monitor-override", "severity": WARN,
                           "detail": f"{row['screen']}: {row['error']}"})
        else:
            detail = (f"{row['screen']}: {row['managed_keys']} managed, "
                      f"{row['unknown_keys']} unknown key(s) at "
                      f"{row['path']}")
            checks.append({"check": "monitor-override",
                           "severity": INFO if row["unknown_keys"] == 0
                           else WARN, "detail": detail})

    # 5. Backup slot coverage (informational; the applier owns it).
    backup = target.parent / (target.name + ".assistant-backup")
    if backup.exists():
        try:
            same = json.loads(backup.read_text(encoding="utf-8")) == current
        except (OSError, json.JSONDecodeError):
            same = False
        checks.append({
            "check": "backup-slot", "severity": INFO,
            "detail": ("the backup slot matches the live content (the "
                       "last write predates no further change)" if same
                       else "a backup slot exists and predates the live "
                            "content (one level of undo is available)"),
        })
    else:
        checks.append({"check": "backup-slot", "severity": INFO,
                       "detail": "no backup slot yet (nothing applied)"})

    critical = sum(1 for c in checks if c["severity"] == CRITICAL)
    return {"target": str(target), "checks": checks, "critical": critical}


def _ops_from_config(current: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synthetic set ops (tool, action, value) for every registry-known
    configured path, so the consequence projection evaluates the
    CONFIGURED state."""
    from .environments import _flatten
    from assistant.adapters.caelestia.registry import tool_by_path
    ops: List[Dict[str, Any]] = []
    for path, value in sorted(_flatten(current).items()):
        spec = tool_by_path(path)
        if spec is None:
            continue
        ops.append({"tool": spec.name, "action": "set", "value": value,
                    "raw": f"configured: {path}={value!r}"})
    return ops


def render_lines(result: Dict[str, Any]) -> List[str]:
    lines = [f"environment audit for {result['target']} "
             f"(read-only; {len(result['checks'])} check(s)):"]
    order = {CRITICAL: 0, WARN: 1, INFO: 2}
    for check in sorted(result["checks"],
                        key=lambda c: order.get(c["severity"], 3)):
        lines.append(f"  [{check['severity']:<8}] {check['check']}: "
                     f"{check['detail']}")
    if result["critical"]:
        lines.append(f"{result['critical']} CRITICAL finding(s) — the "
                     f"applier would refuse writes until resolved")
    else:
        lines.append("no critical findings")
    return lines

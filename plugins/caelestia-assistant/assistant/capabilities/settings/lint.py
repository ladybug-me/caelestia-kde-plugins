"""Config health linter for shell.json (issue #120 Phase 1.1).

A deterministic rule engine over the target config — the same discipline as
assistant/diagnostics/engine.py, but applied to the live shell.json instead
of pasted logs. Read-only: it never writes, it only reports findings.

Checks (each grounded in the frozen registry, never guessed):

1. unknown-key      — a leaf present in shell.json that is neither a registry
                      tool path nor a known not_exposed path. Likely stale,
                      renamed upstream, or hand-edited; the shell's ConfigObject
                      would silently ignore it.
2. out-of-range     — a leaf whose value is outside the shipped Nexus control
                      range transcribed in tools.json (the same range the
                      planner enforces on every apply; a hand edit is not
                      validated that way, so drift lands here).
3. type-mismatch    — a leaf whose JSON type differs from the registry kind
                      (bool/int/float/string/enum).
4. silent-noop      — a leaf that is currently having no effect because a
                      gate it depends on is off. Grounded in the registry's
                      own EXPLAIN_RULES: the blur regions gate on
                      transparency.enabled && blur together
                      (shell/modules/drawers/blur/BlurOffsets.qml:17), so
                      blur=true with transparency disabled is inert.
5. inert-customized — a strength/value leaf that differs from its default
                      while its master switch is off: the customization is
                      currently inert (reported as info, not an error).

Every finding carries: id, path, severity (error|warning|info), message,
cite (the registry fact that grounds it) and a fix hint. Exit semantics are
the CLI's decision; nothing here fails or writes.

lint_rules_ok() validates this module's own rule table (used by
`caelestia-assist selfcheck`, so a malformed lint rule fails the selfcheck
the same way a malformed rules.d file does).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from assistant.adapters.caelestia.registry import NOT_EXPOSED, TOOL_COUNT, tool_by_path

FileTarget = Union[str, Path]

# Gates for the silent-noop / inert-customized rules, stated as
# (gate_path, gated_path, gate_value, grounded_cite). The blur gate is the
# registry's own explain rule (see module docstring) — the only cross-key
# gate the shipped config documents in terms shell.json itself can express.
_NOOP_GATES = (
    {
        "id": "silent-noop-blur-without-transparency",
        "gate": "appearance.transparency.enabled",
        "gate_value": False,
        "gated": "appearance.blur",
        "gated_value": True,
        "severity": "warning",
        "cite": ("assistant/settings/tools.json explain_rules[1] — "
                 "shell/modules/drawers/blur/BlurOffsets.qml:17"),
        "message": ("blur is enabled but has no effect while transparency is "
                    "disabled: the blur regions gate on transparency.enabled "
                    "&& blur together"),
        "fix": "enable appearance.transparency.enabled, or leave blur off",
    },
    {
        "id": "inert-customized-transparency-base",
        "gate": "appearance.transparency.enabled",
        "gate_value": False,
        "gated": "appearance.transparency.base",
        "gated_value": "__non_default__",
        "severity": "info",
        "cite": ("assistant/settings/registry.py — appearance.transparency.base "
                 "default 0.85 (shipped Nexus control, SliderRow)"),
        "message": ("transparency.base is customized but currently inert: "
                    "transparency is disabled"),
        "fix": ("re-enable appearance.transparency.enabled, or accept the "
                "customization is dormant"),
    },
)


def flatten_leaves(node: Any, prefix: str = "") -> List[tuple]:
    """Flatten a nested JSON object to [(dotted_path, value)].

    Public since exponential-build-3 Phase 2: brain/topology.py
    flattened the same shape with its own identical private walker
    (found by the repo's own MinHash scan) — this is the one
    implementation now (brain importing from settings is the
    precedented direction: brain/service.py::settings_bridge)."""
    leaves: List[tuple] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                leaves.extend(flatten_leaves(value, path))
            else:
                leaves.append((path, value))
    return leaves


def _known_non_tool_paths(specs: Optional[List[Any]]) -> List[str]:
    """Registry paths that exist in shell.json but are not exposed as tools
    (the not_exposed table: real ConfigObject leaves the assistant must not
    manage — credentials, endpoints, internal state)."""
    from assistant.adapters.caelestia.registry import NOT_EXPOSED  # (path, reason) pairs from tools.json

    return [path for path, _reason in NOT_EXPOSED]


def _read_config(target: FileTarget) -> Dict[str, Any]:
    path = Path(target)
    if not path.exists():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} is not a JSON object")
    return parsed


def lint(config: Dict[str, Any], specs: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Lint one parsed shell.json (dict of dicts) and return findings.

    Deterministic, read-only, and honest: every finding names the registry
    fact it is grounded in. No findings does not mean "perfect config" —
    it means "nothing this rule set can check is wrong".
    """
    from assistant.adapters.caelestia.registry import TOOL_SPECS  # the frozen table, import-cheap

    specs = specs if specs is not None else TOOL_SPECS
    known_paths = {s.path for s in specs}
    findings: List[Dict[str, Any]] = []

    non_tool_paths = _known_non_tool_paths(specs)
    leaves = flatten_leaves(config)

    # Pass 1: per-leaf registry checks.
    for path, value in leaves:
        spec = tool_by_path(path)
        if spec is None:
            if path in non_tool_paths:
                continue  # known ConfigObject leaf, deliberately unmanaged
            findings.append({
                "id": "unknown-key",
                "path": path,
                "severity": "warning",
                "message": ("not a registry tool path and not a known "
                            "non-tool leaf: stale, renamed upstream, or "
                            "hand-edited; the shell ignores unknown keys"),
                "cite": ("assistant/settings/tools.json (%d tool paths, %d not_exposed paths)"
                          % (TOOL_COUNT, len(NOT_EXPOSED))),
                "fix": "remove the key, or check the upstream config name",
            })
            continue
        # kind check
        kind = spec.kind
        valid = {
            "bool": lambda v: isinstance(v, bool),
            "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "enum": lambda v: isinstance(v, str),
            "string": lambda v: isinstance(v, str),
        }[kind]
        if not valid(value):
            findings.append({
                "id": "type-mismatch",
                "path": path,
                "severity": "warning",
                "message": f"registry kind is {kind}, found {type(value).__name__}",
                "cite": f"assistant/settings/tools.json — {path} kind={kind}",
                "fix": f"set {path} to a {kind} value",
            })
            continue
        # range check (numeric only; enums are frozen at build time and
        # strings carry no range)
        if kind in ("int", "float"):
            lo, hi = spec.minimum, spec.maximum
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                findings.append({
                    "id": "out-of-range",
                    "path": path,
                    "severity": "warning",
                    "message": (f"value {value} is outside the shipped control "
                                f"range {lo}-{hi}; the next planned apply of "
                                "this key would be rejected"),
                    "cite": f"assistant/settings/tools.json — {path} range {lo}-{hi}",
                    "fix": f"set {path} between {lo} and {hi}",
                })

    # Pass 2: cross-key gate rules (silent no-ops / inert customizations).
    flat = dict(leaves)
    by_path: Dict[str, Any] = {}
    for path, value in leaves:
        by_path.setdefault(path, value)
    for rule in _NOOP_GATES:
        gate_value = by_path.get(rule["gate"])
        gated_value = by_path.get(rule["gated"])
        if gate_value != rule["gate_value"] or gated_value is None:
            continue
        if rule["gated_value"] == "__non_default__":
            spec = tool_by_path(rule["gated"])
            if spec is None or gated_value == spec.default:
                continue
        elif gated_value != rule["gated_value"]:
            continue
        findings.append({
            "id": rule["id"],
            "path": rule["gated"],
            "severity": rule["severity"],
            "message": rule["message"],
            "cite": rule["cite"],
            "fix": rule["fix"],
        })

    return findings


def lint_file(target: FileTarget) -> List[Dict[str, Any]]:
    """Lint the JSON file at target (missing file -> no findings)."""
    return lint(_read_config(target))


def render_findings(findings: List[Dict[str, Any]]) -> List[str]:
    """Plain-text rendering, one line per finding plus its fix hint."""
    if not findings:
        return ["config lint: no findings"]
    lines = [f"config lint: {len(findings)} finding(s)"]
    for f in findings:
        lines.append(f"- [{f['severity']}] {f['id']} @ {f['path']}: {f['message']}")
        lines.append(f"    grounded in: {f['cite']}")
        lines.append(f"    fix: {f['fix']}")
    return lines


def lint_rules_ok() -> List[str]:
    """Structural self-validation of the rule table above (same spirit as
    schema_lint.check_rule_files): every gate rule must name registry paths
    that actually exist, have a severity the renderer understands, and carry
    message/cite/fix strings. Returns a list of failures (empty = OK)."""
    from assistant.adapters.caelestia.registry import NOT_EXPOSED, TOOL_COUNT, tool_by_path as _tbp

    failures: List[str] = []
    severities = {"error", "warning", "info"}
    for rule in _NOOP_GATES:
        rid = rule.get("id", "<missing id>")
        for key in ("gate", "gated", "message", "cite", "fix", "severity"):
            if key not in rule:
                failures.append(f"lint rule {rid}: missing {key!r}")
        if rule.get("severity") not in severities:
            failures.append(f"lint rule {rid}: bad severity {rule.get('severity')!r}")
        if _tbp(str(rule.get("gate", ""))) is None:
            failures.append(f"lint rule {rid}: gate {rule.get('gate')!r} is not a registry path")
        if _tbp(str(rule.get("gated", ""))) is None:
            failures.append(f"lint rule {rid}: gated {rule.get('gated')!r} is not a registry path")
    return failures

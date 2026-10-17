"""Reverse lookup: matched diagnostic rule -> settings tools (A4, Tier A).

The closed loop this module closes: Layer 1 (diagnostics) answers "what is
broken"; the settings layer (issue #120) answers "change this setting".
When a rule's ROOT CAUSE is addressable by a settings tool, the report can
now say so — "the settings tools addressing this root cause are X, Y" —
instead of leaving the user to guess the connection.

The join is EXACT and VOLUNTEERED-ONLY, by design:

- a rule's ``references`` entries of kind ``doc`` whose anchor names
  ``assistant/settings/tools.json`` carry tool names explicitly
  (e.g. CL-idle-throttle-001's anchor: "assistant/settings/tools.json —
  battery-saver and gaming presets (setBlurEnabled/setAnimationSpeed
  citations)");
- tool NAMES (``setFooBar``) and registry PATHS (``bar.scale``) matched
  word-exactly inside the rule's fix texts and notes.

It deliberately does NOT join on noun vocabulary or fuzzy similarity:
"Bar/taskbar thumbnails ... broken" (CL-kde-thumbs-001) mentions
"thumbnails", a noun of ``setLivePreviews`` — but that rule's root cause
is desktop-cache staleness, and suggesting "turn live previews off" would
be a wrong fix delivered confidently. A join that guesses is worse than
no join; only references the rule itself volunteered are trusted.

Every mention resolves through the settings layer's own citation
registry (``tool_by_name`` / ``tool_by_path`` over tools.json — the
generated, per-tool-citation-verified, drift-guarded artifact), so every
cross-reference that leaves this module is a real tools.json id by
construction. Mentions that do NOT resolve are reported separately
(``unresolved``) rather than silently dropped — a rule naming a
nonexistent tool is a content bug someone should see.

PURE module: reads nothing from disk (callers pass the rule dicts the
engine already loaded); deterministic; no I/O, no execution.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from assistant.adapters.caelestia.registry import tool_by_name, tool_by_path

__all__ = ["settings_tools_for_rule", "rules_for_tool", "SETTINGS_REGISTRY_ANCHOR"]

SETTINGS_REGISTRY_ANCHOR = "assistant/settings/tools.json"

_TOOL_NAME_RE = re.compile(r"\b(set[A-Z][A-Za-z0-9]+)\b")
_TOOL_PATH_RE = re.compile(
    r"\b((?:bar|appearance|launcher|notifs|dock|overview|osd|sidebar|greeter)"
    r"(?:\.[a-z][a-z0-9]*)+)\b"
)

# Keys of the rule whose text participates in the join (fix step texts and
# notes — the places a rule explains its own fix). Matchers and titles are
# deliberately excluded: they describe the SYMPTOM, not the fix, and a
# symptom word joining to a tool is the wrong-fix hazard above.
_TEXT_KEYS = ("notes",)


def _texts(rule: Dict[str, Any]) -> List[str]:
    """The rule's own fix/notes text surfaces as a flat string list."""
    out: List[str] = []
    for step in rule.get("fix", []) or []:
        if isinstance(step, dict):
            out.append(str(step.get("text", "")))
    for key in _TEXT_KEYS:
        value = rule.get(key)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(str(v) for v in value)
    return out


def _anchor_texts(rule: Dict[str, Any]) -> List[str]:
    return [
        str(ref.get("anchor", ""))
        for ref in rule.get("references", []) or []
        if isinstance(ref, dict) and ref.get("kind") == "doc"
    ]


def settings_tools_for_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """The reverse lookup for ONE rule.

    Returns ``{"tools": [{"tool", "path", "found_in"}...],
               "unresolved": [mention, ...]}`` — ``tools`` entries resolve
    to real tools.json ids (verified via the citation registry); mentions
    that did not resolve land in ``unresolved`` for human review.
    """
    joined: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []

    def _join(name: str, found_in: str) -> None:
        spec = tool_by_name(name)
        if spec is None:
            unresolved.append(name)
            return
        if name not in joined:
            joined[name] = {"tool": spec.name, "path": spec.path,
                            "found_in": found_in}

    # 1. References that point INTO the settings registry itself.
    for anchor in _anchor_texts(rule):
        if SETTINGS_REGISTRY_ANCHOR not in anchor:
            continue
        for match in _TOOL_NAME_RE.finditer(anchor):
            _join(match.group(1), f"reference: {anchor[:60]}...")

    # 2. Exact tool names / registry paths volunteered in fix + notes text.
    for text in _texts(rule):
        for match in _TOOL_NAME_RE.finditer(text):
            _join(match.group(1), "fix/notes text")
        for match in _TOOL_PATH_RE.finditer(text):
            path = match.group(1)
            spec = tool_by_path(path)
            if spec is None:
                unresolved.append(path)
                continue
            if spec.name not in joined:
                joined[spec.name] = {"tool": spec.name, "path": spec.path,
                                     "found_in": "fix/notes text"}

    return {"tools": [joined[k] for k in sorted(joined)],
            "unresolved": sorted(set(unresolved))}


def rules_for_tool(tool_name: str,
                   rules: List[Dict[str, Any]]) -> List[str]:
    """The other direction: which rules' fixes reference this tool.

    Useful for impact analysis before changing a tool ("which diagnostic
    rules mention it?"). Returns rule ids, sorted."""
    out: List[str] = []
    for rule in rules:
        joined = settings_tools_for_rule(rule)
        if any(entry["tool"] == tool_name for entry in joined["tools"]):
            out.append(str(rule.get("id", "?")))
    return sorted(out)

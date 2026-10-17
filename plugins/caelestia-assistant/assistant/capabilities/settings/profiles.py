"""assistant.capabilities.settings.profiles — profile algebra (F9, exponential-build-5).

Issue #120's Phase 3 asks for "optimization profiles" beyond the five
shipped presets; the A9 conformance table records that gap honestly as
b6b PARTIAL ("preset bundles + the learned pairwise ladder; no profile
algebra/composition"). This module is that algebra — built entirely
from engines that already exist, with no new write surface:

- a PROFILE is a named, ordered list of SOURCES. A source is a preset
  (presets.py — registry-validated bundles), a macro (macros.py — an
  APPROVED captured apply), or an explicit single call (registry tool
  = JSON value);
- COMPOSITION is deterministic and conflict-reporting: sources apply
  in order, LATER WINS per tool, and every overridden earlier value is
  reported, never silently dropped (the F7 no-silent-drop invariant);
- the composed ops come out in the plain planner-op shape (the same
  shape presets.preset_ops and macros.macro_ops produce), so applying a
  profile rides the ordinary plan -> preview -> consent -> applier path
  with NO bespoke writer. Multi-change profiles need the standard
  --confirm second consent, exactly like presets and macros;
- SAVE re-resolves every source against the LIVE registry and the
  saved stores at save time: a preset that no longer exists, a deleted
  macro, or a call naming an unknown tool refuses the whole save
  (a partially stale composite is worse than none). Value-range
  validation stays with the planner at apply time — a saved profile
  whose call is now out of range produces an honest plan rejection,
  never a silent write.

STORAGE (no new write surface): profiles live INSIDE the existing
undo-history file at ``<target>.assistant-history.json`` under a
bounded ``"profiles"`` key — the same file, the same ``history._save``
atomic write path, the same sibling path as the ``macros`` (C3) and
``undo_log`` (A3) precedents. Save and delete are explicit user
commands; the target shell.json is never touched by this module.

Each stored profile is {"name", "label" (free text), "created_at"
(ISO-8601), "sources": [...]} where sources are
{"kind": "preset"|"macro", "name": ...} or
{"kind": "call", "tool": ..., "value": ...}.

Pure module apart from the documented history-file reads/writes: no
execution, no network, no RNG; same inputs -> same stored bytes
(created_at is the one field that differs per call, by precedent).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from . import history
from . import presets as presets_mod
from assistant.adapters.caelestia.registry import TOOL_SPECS

__all__ = ["ProfileError", "MAX_PROFILES", "PROFILES_KEY", "NAME_RE",
           "parse_source", "list_profiles", "save", "delete",
           "compose_ops", "profile_ops", "diff", "render_profile_lines"]

MAX_PROFILES = 16
PROFILES_KEY = "profiles"

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,31}")

TOOL_NAMES = frozenset(t.name for t in TOOL_SPECS)

FileTarget = Union[str, Path]


class ProfileError(ValueError):
    """Raised for every refusal (unknown profile/source, bad name,
    duplicate); the caller renders the reason — nothing is guessed."""


def parse_source(spec: str) -> Dict[str, Any]:
    """Parse one CLI source token: ``preset:NAME`` | ``macro:NAME`` |
    ``TOOL=VALUE`` (VALUE is JSON, or a plain string on JSON failure —
    the --call convention). Pure. Raises ProfileError on a malformed
    token; unknown names are checked at save/apply time against the
    live registry, not here."""
    spec = str(spec or "").strip()
    if spec.startswith("preset:"):
        name = spec[len("preset:"):].strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", name):
            raise ProfileError(
                f"preset source needs a preset slug (lowercase letters, "
                f"digits, dashes): got {name!r}")
        return {"kind": "preset", "name": name}
    if spec.startswith("macro:"):
        name = spec[len("macro:"):].strip()
        if not name:
            raise ProfileError("macro source needs a name: macro:NAME")
        return {"kind": "macro", "name": name}
    if "=" in spec:
        tool, _, raw = spec.partition("=")
        tool = tool.strip()
        if not tool:
            raise ProfileError(f"call source needs a tool name: {spec!r}")
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            value = raw.strip()
        return {"kind": "call", "tool": tool, "value": value}
    raise ProfileError(
        f"cannot parse source {spec!r}; expected preset:NAME, macro:NAME, "
        f"or TOOL=VALUE")


def _load_profiles(target) -> List[Dict[str, Any]]:
    return [dict(p) for p in history._load(target).get(PROFILES_KEY, [])]


def list_profiles(target) -> List[Dict[str, Any]]:
    """The saved profiles, oldest first (a copy). Read-only."""
    return _load_profiles(target)


def save(target, name: str, sources: List[Dict[str, Any]],
         label: str = "") -> Dict[str, Any]:
    """Save a named profile. Every source is resolved NOW (preset in
    the registry's preset table; macro in the saved macro store; call
    naming a live registry tool) — an unresolvable source refuses the
    whole save. Writes only the history file's "profiles" key; never
    the target file. Duplicate names refuse (delete first)."""
    if not NAME_RE.fullmatch(str(name or "")):
        raise ProfileError(
            f"profile names must match {NAME_RE.pattern!r} (start with a "
            f"letter or digit; letters, digits, spaces, '_', '.', '-' "
            f"afterwards; at most 32 characters) — got {name!r}")
    if not sources:
        raise ProfileError("a profile needs at least one source")
    # Resolve every source now (honest refusal beats a stale composite).
    composed, _conflicts = compose_ops(target, sources)
    if not composed:
        raise ProfileError("a profile must resolve to at least one tool call")
    data = history._load(target)
    profiles = data.get(PROFILES_KEY, [])
    if any(p.get("name") == name for p in profiles):
        raise ProfileError(
            f"profile {name!r} already exists; delete it first "
            f"(--profile-delete {name})")
    entry = {
        "name": name,
        "label": str(label or "")[:200],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [dict(s) for s in sources],
    }
    profiles.append(entry)  # FIFO eviction below keeps the store bounded
    data[PROFILES_KEY] = profiles[-MAX_PROFILES:]
    history._save(target, data)
    return dict(entry)


def delete(target, name: str) -> Dict[str, Any]:
    """Remove one profile by name (an explicit user command; writes
    only the history file's "profiles" key). Raises ProfileError for
    an unknown name — never a silent no-op delete."""
    data = history._load(target)
    profiles = data.get(PROFILES_KEY, [])
    kept = [p for p in profiles if p.get("name") != name]
    if len(kept) == len(profiles):
        saved = [str(p.get("name")) for p in profiles]
        raise ProfileError(
            f"unknown profile {name!r}; saved profiles: "
            + (", ".join(saved) if saved else "(none)"))
    data[PROFILES_KEY] = kept
    history._save(target, data)
    gone = next(p for p in profiles if p.get("name") == name)
    return {"deleted": name, "sources": len(gone.get("sources", []))}


def compose_ops(target, sources: List[Dict[str, Any]]
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve and compose sources in order into planner-shaped ops.

    Later wins per tool; every overridden earlier value is reported as
    a conflict entry {tool, kept, dropped} (F7: nothing silently
    dropped). Read-only. Raises ProfileError for an unresolvable
    source (unknown preset/macro/tool)."""
    ops_by_tool: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    conflicts: List[Dict[str, Any]] = []
    for src in sources:
        kind = src.get("kind")
        if kind == "preset":
            name = str(src.get("name", ""))
            try:
                preset_ops = presets_mod.preset_ops(name)
            except presets_mod.PresetError as exc:
                raise ProfileError(f"preset source: {exc}") from None
            for op in preset_ops:
                _absorb(op, f"preset:{name}", ops_by_tool, order, conflicts)
        elif kind == "macro":
            name = str(src.get("name", ""))
            from . import macros as macros_mod
            try:
                macro_ops = macros_mod.macro_ops(target, name)
            except macros_mod.MacroError as exc:
                raise ProfileError(f"macro source: {exc}") from None
            for op in macro_ops:
                _absorb(op, f"macro:{name}", ops_by_tool, order, conflicts)
        elif kind == "call":
            tool = str(src.get("tool", ""))
            if tool not in TOOL_NAMES:
                raise ProfileError(
                    f"call source: {tool!r} is not a registry tool "
                    f"(the planner validates values at apply time)")
            op = {"tool": tool, "action": "set",
                  "value": src.get("value"),
                  "raw": f"call: {tool}={src.get('value')!r}"}
            _absorb(op, f"call:{tool}", ops_by_tool, order, conflicts)
        else:
            raise ProfileError(f"unknown source kind {kind!r}")
    return [ops_by_tool[t] for t in order], conflicts


def _absorb(op: Dict[str, Any], provenance: str,
            ops_by_tool: Dict[str, Dict[str, Any]], order: List[str],
            conflicts: List[Dict[str, Any]]) -> None:
    tool = str(op.get("tool"))
    prev = ops_by_tool.get(tool)
    if prev is not None and prev.get("value") != op.get("value"):
        conflicts.append({
            "tool": tool,
            "dropped": {"value": prev.get("value"), "from": prev.get("_from")},
            "kept": {"value": op.get("value"), "from": provenance},
        })
    elif prev is not None:
        return  # same value re-asserted: not a conflict, not a new op
    ops_by_tool[tool] = dict(op)
    ops_by_tool[tool]["_from"] = provenance
    if tool not in order:
        order.append(tool)


def profile_ops(target, name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """A saved profile's composed ops + conflicts (planner shape).
    Raises ProfileError for an unknown profile."""
    for profile in _load_profiles(target):
        if profile.get("name") == name:
            return compose_ops(target, profile.get("sources", []))
    saved = [str(p.get("name")) for p in _load_profiles(target)]
    raise ProfileError(
        f"unknown profile {name!r}; saved profiles: "
        + (", ".join(saved) if saved else "(none yet; save one with "
           "--profile-save)") + f" — {name!r} is not one of them")


def diff(target, name_a: str, name_b: str) -> Dict[str, Any]:
    """Effective-value diff of two saved profiles: per tool — only A,
    only B, or changed (value A -> value B). Deterministic; read-only."""
    ops_a, _ = profile_ops(target, name_a)
    ops_b, _ = profile_ops(target, name_b)
    va = {op["tool"]: op.get("value") for op in ops_a}
    vb = {op["tool"]: op.get("value") for op in ops_b}
    rows = []
    for tool in list(va) + [t for t in vb if t not in va]:
        a, b = va.get(tool), vb.get(tool)
        if tool not in vb:
            rows.append({"tool": tool, "kind": "only_a", "a": a})
        elif tool not in va:
            rows.append({"tool": tool, "kind": "only_b", "b": b})
        elif a != b:
            rows.append({"tool": tool, "kind": "changed", "a": a, "b": b})
    return {"a": name_a, "b": name_b, "rows": rows}


def render_profile_lines(target, name: str) -> List[str]:
    """The --profile-show table: sources with provenance, the composed
    ops, and every composition conflict. Read-only."""
    for profile in _load_profiles(target):
        if profile.get("name") == name:
            ops, conflicts = compose_ops(target, profile.get("sources", []))
            lines = [f"profile {name!r} — {len(profile.get('sources', []))} "
                     f"source(s):"]
            for src in profile.get("sources", []):
                if src.get("kind") == "call":
                    lines.append(f"  source: call {src.get('tool')}="
                                 f"{src.get('value')!r}")
                else:
                    lines.append(f"  source: {src.get('kind')} "
                                 f"{src.get('name')}")
            lines.append(f"composed ops ({len(ops)} tool(s), later source "
                         f"wins per tool):")
            for op in ops:
                lines.append(f"  {op['tool']} = {op.get('value')!r} "
                             f"[{op.get('_from')}]")
            if conflicts:
                lines.append(f"conflicts resolved by order "
                             f"({len(conflicts)}, later wins — none "
                             f"silently dropped):")
                for c in conflicts:
                    lines.append(
                        f"  {c['tool']}: {c['dropped']['value']!r} "
                        f"(from {c['dropped']['from']}) overridden by "
                        f"{c['kept']['value']!r} (from {c['kept']['from']})")
            return lines
    saved = [str(p.get("name")) for p in _load_profiles(target)]
    raise ProfileError(
        f"unknown profile {name!r}; saved profiles: "
        + (", ".join(saved) if saved else "(none yet; save one with "
           "--profile-save)"))

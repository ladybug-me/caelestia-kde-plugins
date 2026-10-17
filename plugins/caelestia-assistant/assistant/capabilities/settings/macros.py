"""Settings macros — capture an APPROVED proposal sequence as a named,
replayable template (exponential-build-3 C3).

Cypher (ed.) 1993, "Watch What I Do: Programming by Demonstration",
MIT Press — the programming-by-demonstration pattern: a demonstrated
action sequence (here: a settings change the user previewed, consented
to, and had applied) is recorded under a name and replayed on demand.
The capture/replay machinery itself is not novel and is not claimed to
be; what is engineered here is the CONSENT DISCIPLINE around it, which
is the whole point in a layer whose only writer is a gated applier.

THE CONSENT CONTRACT (read before trusting a replay):

- a macro is captured ONLY from an undo-history entry — i.e. from a
  proposal sequence that already passed the full preview -> consent ->
  apply gate once. There is deliberately no way to capture a macro
  from an unapplied plan: a template is a record of something the user
  actually approved, not something they merely asked about;
- REPLAY NEVER WRITES BY ITSELF. ``macro_ops`` returns plain planner
  ops (the same shape ``presets.preset_ops`` produces), so a replay
  rides the ordinary plan/render path: dry-run preview by default, and
  even with ``--apply`` the CLI demands the second consent
  (``--confirm`` or the interactive y/N) for EVERY macro — single-op
  macros included, deliberately stronger than the plain >1-change
  rule, because a replay's contents may no longer be in the user's
  head. The applied replay records its OWN history entry
  (label "macro: NAME"), so bounded undo works per replay;
- replay re-validates against the LIVE registry: ops carry tool names
  and the planner resolves them against the current tool table and the
  current file state. A macro recorded against an older registry (a
  renamed/removed tool) produces an honest error entry and a blocked
  apply, never a silent write; a macro whose values are already in
  place is an honest no-op ("no changes needed").

STORAGE (no new write surface): the macro store lives INSIDE the
existing undo-history file at ``<target>.assistant-history.json``,
under a bounded ``"macros"`` key — the same file, the same
``history._save`` atomic write path, the same sibling path as the A3
``undo_log`` precedent, so the applier's documented four-path write
scope (and test_safety.py's directory-snapshot assertion of it) is
unchanged. Capture and delete are explicit user commands (like undo
itself); the target file is never touched by this module — the
applier behind ``--apply``/``--confirm`` remains the only writer of
shell.json. The store is bounded (MAX_MACROS, FIFO: oldest capture
evicted first) so the config directory cannot grow without limit, the
same discipline the history ring applies.

Each stored macro is {"name", "label" (the captured request sentence),
"from_id" (the source history entry), "captured_at" (ISO-8601), "ops":
[{"tool", "path", "value"}]} — the APPLIED values only; the old values
stay in the history entry where they belong (a macro is not an undo
artifact).

Pure module apart from the documented history-file reads/writes: no
execution, no network, no RNG; same inputs -> same stored bytes
(the captured_at timestamp is the one field that differs per call, by
design and by precedent — history.record does the same).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import history
from assistant.adapters.caelestia.registry import tool_by_path

__all__ = ["MacroError", "MAX_MACROS", "list_macros", "save",
           "macro_ops", "delete"]

MAX_MACROS = 16
MACROS_KEY = "macros"

# Macro names: start alnum, then alnum/space/underscore/dot/dash, at
# most 32 chars — generous enough for "focus mode" or
# "evening-gaming.v2", narrow enough to stay unambiguous in a CLI flag.
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,31}")

FileTarget = Union[str, Path]


class MacroError(ValueError):
    """Raised for every refusal (unknown macro, bad name, empty or
    stale source entry, duplicate name); the caller renders the reason
    — nothing is guessed, nothing is silently overwritten."""


def list_macros(target: FileTarget) -> List[Dict[str, Any]]:
    """The saved macros, oldest capture first (a copy). Read-only."""
    return [dict(m) for m in history._load(target).get(MACROS_KEY, [])]


def save(target: FileTarget, name: str,
         from_id: Optional[int] = None) -> Dict[str, Any]:
    """Capture one APPROVED apply (an undo-history entry) as a macro.

    ``from_id`` selects the entry (see ``--history`` for ids); the
    default is the NEWEST entry — "save what I just did". The entry's
    applied ops are re-resolved against the live registry at capture
    time: a path the registry no longer knows refuses the whole capture
    (named), because a partially stale template is worse than none.
    Writes only the history file's "macros" key (see the module
    docstring); never the target."""
    if not NAME_RE.fullmatch(str(name or "")):
        raise MacroError(
            f"macro names must match {NAME_RE.pattern!r} (start with a "
            f"letter or digit; letters, digits, spaces, '_', '.', '-' "
            f"afterwards; at most 32 characters) — got {name!r}")
    data = history._load(target)
    entries = data.get("entries", [])
    if not entries:
        raise MacroError(
            "nothing to capture: the undo history is empty (a macro "
            "records an apply you already approved; apply something "
            "first, then capture it)")
    if from_id is None:
        entry = entries[0]
        from_id = entry.get("id")
    else:
        entry = next((e for e in entries
                      if int(e.get("id", -1)) == int(from_id)), None)
        if entry is None:
            raise MacroError(
                f"no history entry with id {from_id}; see --history "
                "for the ids in the bounded ring")
    if any(m.get("name") == name for m in data.get(MACROS_KEY, [])):
        raise MacroError(
            f"a macro named {name!r} already exists; delete it first "
            f"(--macro-delete {name}) — captures never silently "
            "overwrite")
    ops: List[Dict[str, Any]] = []
    stale = []
    for op in entry.get("ops", []):
        path = str(op.get("path", ""))
        spec = tool_by_path(path)
        if spec is None:
            stale.append(path)
            continue
        ops.append({"tool": spec.name, "path": spec.path,
                    "value": op.get("new")})
    if stale:
        raise MacroError(
            f"history entry #{from_id} references path(s) no longer in "
            f"the registry: {', '.join(sorted(set(stale)))}; refusing "
            "to capture a partially stale macro (delete nothing, fix "
            "nothing — the sequence as applied cannot be replayed "
            "honestly)")
    if not ops:
        raise MacroError(
            f"history entry #{from_id} references no registry paths; "
            "there is nothing replayable to capture")
    macro = {
        "name": name,
        "label": str(entry.get("label") or ""),
        "from_id": from_id,
        "captured_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "ops": ops,
    }
    data.setdefault(MACROS_KEY, []).append(macro)
    data[MACROS_KEY] = data[MACROS_KEY][-MAX_MACROS:]  # FIFO eviction
    history._save(target, data)
    return macro


def macro_ops(target: FileTarget, name: str) -> List[Dict[str, Any]]:
    """A macro's replay ops, in the plain planner-op shape (the same
    shape ``presets.preset_ops`` produces: tool / action / value /
    raw), so the replay rides the ordinary plan -> preview -> consent
    path with no bespoke writer. Read-only; raises MacroError for an
    unknown name (listing the saved ones)."""
    for macro in history._load(target).get(MACROS_KEY, []):
        if macro.get("name") == name:
            return [
                {"tool": str(op.get("tool")),
                 "action": "set",
                 "value": op.get("value"),
                 "raw": f"macro {name}: {op.get('tool')}="
                        f"{op.get('value')!r}"}
                for op in macro.get("ops", [])
            ]
    saved = [str(m.get("name")) for m in
             history._load(target).get(MACROS_KEY, [])]
    hint = (", ".join(saved) if saved
            else "(no macros saved yet; capture one with --macro-save)")
    raise MacroError(f"unknown macro {name!r}; saved macros: {hint}")


def delete(target: FileTarget, name: str) -> Dict[str, Any]:
    """Remove one macro by name (an explicit user command; writes only
    the history file's "macros" key). Raises MacroError for an unknown
    name — never a silent no-op delete."""
    data = history._load(target)
    macros = data.get(MACROS_KEY, [])
    kept = [m for m in macros if m.get("name") != name]
    if len(kept) == len(macros):
        saved = [str(m.get("name")) for m in macros]
        raise MacroError(
            f"unknown macro {name!r}; saved macros: "
            + (", ".join(saved) if saved else "(none)"))
    removed = next(m for m in macros if m.get("name") == name)
    if kept:
        data[MACROS_KEY] = kept
    else:
        data.pop(MACROS_KEY, None)
    history._save(target, data)
    return {"deleted": name, "from_id": removed.get("from_id"),
            "ops": len(removed.get("ops", []))}

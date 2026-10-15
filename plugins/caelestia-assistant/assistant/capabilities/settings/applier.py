"""Applier for the settings layer (the settings design notes (see module docstrings) §5): plan -> written file.

This is the ONLY module in the whole assistant that writes anything, and it
writes ONLY when the caller passes ``write=True`` (the CLI's ``--apply``
gate — the equivalent of #120's confirmation step). Its write scope is
exactly four paths, all siblings of the explicit target file:

- the target file itself (via a transient ``.assistant-tmp`` + atomic
  ``os.replace``, i.e. POSIX rename(2));
- ``<file>.assistant-backup`` — the single-slot one-level undo (a
  zero-byte backup means "the target did not exist before the apply");
- ``<file>.assistant-tmp`` — removed after the rename; best-effort
  unlinked on any failure;
- ``<file>.assistant-history.json`` — the bounded undo history
  (history.py owns the schema; the applier appends after each successful
  write; oldest entries are evicted beyond history.MAX_ENTRIES).

NEVER: any other path, any parent-directory creation (a missing config
directory is an error, not a mkdir), any command execution (suggested
commands are inert ``SUGGESTED_NOT_EXECUTED:`` strings), any network, any
monitor-override file. Dry-run (the default) writes nothing at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from assistant.adapters.caelestia.registry import tool_by_path

BACKUP_SUFFIX = ".assistant-backup"
TMP_SUFFIX = ".assistant-tmp"


class ApplierError(RuntimeError):
    """Raised when a write or restore cannot be completed safely."""


def _resolved(file_path: Union[str, Path]) -> Path:
    """§5.3: resolve symlinks so the rename replaces the real file and the
    backup/tmp siblings sit next to the *resolved* file."""
    return Path(os.path.realpath(str(file_path)))


def _sibling(target: Path, suffix: str) -> Path:
    return target.with_name(target.name + suffix)


def _parse_current(target: Path) -> Dict[str, Any]:
    """Parse the current target file for merging. Missing -> {} (the file
    will be created on apply). Invalid JSON / non-object -> ApplierError,
    with nothing written."""
    if not target.exists():
        return {}
    try:
        raw = target.read_bytes()
        current = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ApplierError(
            f"target file {target} cannot be parsed for merging ({exc}); nothing was written"
        ) from exc
    if not isinstance(current, dict):
        raise ApplierError(
            f"target file {target} is valid JSON but not an object; nothing was written"
        )
    return current


def _deep_set(data: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set one whitelisted registry leaf, creating intermediate dicts along
    the registry path only (§5.3 step 3). Every other key is preserved."""
    segments = dotted.split(".")
    node: Dict[str, Any] = data
    for segment in segments[:-1]:
        nxt = node.get(segment)
        if nxt is None:
            child: Dict[str, Any] = {}
            node[segment] = child
            node = child
        elif isinstance(nxt, dict):
            node = nxt
        else:
            raise ApplierError(
                f"cannot merge into {dotted}: '{segment}' exists but is not an object; "
                "nothing was written"
            )
    node[segments[-1]] = value


def _deep_unset(data: Dict[str, Any], dotted: str) -> None:
    """Remove one leaf and prune now-empty parents along its path (the
    inverse of _deep_set: an apply that CREATED a key is undone by making
    the key absent again, so ConfigObject falls back to the default)."""
    segments = dotted.split(".")
    chain: List[Dict[str, Any]] = [data]
    for segment in segments[:-1]:
        nxt = chain[-1].get(segment)
        if not isinstance(nxt, dict):
            return  # absent already — nothing to remove
        chain.append(nxt)
    chain[-1].pop(segments[-1], None)
    for depth in range(len(chain) - 1, 0, -1):
        parent = chain[depth - 1]
        child = chain[depth]
        if child:
            break
        parent.pop(segments[depth - 1], None)


def apply(plan: Dict[str, Any], file_path: Union[str, Path], write: bool = False,
          label: Optional[str] = None, record_history: bool = True,
          scheme: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Apply a validated plan to the target file (§5.3).

    Dry-run (default): writes NOTHING — no target, no backup, no tmp — and
    returns what would happen. ``write=True`` performs the gated write:
    backup first, merge only whitelisted registry leaf paths into the
    parsed current JSON, serialize with 4-space indent + trailing newline,
    write ``.assistant-tmp`` then atomic ``os.replace`` onto the target.

    Before any byte is written, the POST-apply state runs through the
    sanity simulator (sanity.check): a projected WCAG-contrast refusal
    (only with scheme context) raises ApplierError with nothing written —
    target, backup and history stay byte-identical — and tap-target
    warnings ride the result. This runs for single-setting applies too
    (they skip the multi-change preview, so this is their only invariant
    gate). The write=True gate semantics are unchanged.

    A successful write also appends to the bounded undo history
    at ``<file>.assistant-history.json`` (history.MAX_ENTRIES entries,
    oldest evicted first) unless ``record_history=False`` (used by
    history.undo so an undo is not itself recorded). ``label`` is the
    human-readable request the entry stores. ``scheme`` optionally supplies
    {"role": "#hex"} colors for the projected-contrast check; without it
    the contrast check honestly reports itself skipped.
    """
    target = _resolved(file_path)
    entries: List[Dict[str, Any]] = plan.get("entries", [])
    applicable = [e for e in entries if not e.get("error") and not e.get("no_op")]

    if plan.get("apply_blocked") and write:
        # §4.5: all-or-nothing — a blocked plan is never written.
        raise ApplierError(
            "apply refused: the plan contains rejected or type-mismatched entries; "
            "nothing was written"
        )

    if not applicable:
        # §5.3 step 1: every entry is a no-op -> no pointless backup churn.
        return {
            "written": False,
            "no_changes": True,
            "message": "no changes needed; nothing written",
        }

    if not write:
        return {
            "written": False,
            "message": (
                f"dry-run: {len(applicable)} change(s) would be written to {target} "
                "with --apply; nothing is written now"
            ),
        }

    # The merge target: the parsed CURRENT JSON (fresh read at write time),
    # so every other key survives the merge untouched.
    current = _parse_current(target)

    # Pre-write sanity simulator (issue #120 Phase 1.2): check the POST-apply
    # state BEFORE the backup slot is touched, so a refusal leaves the
    # target, its backup and the undo history byte-identical.
    from . import sanity as _sanity
    projected = _sanity.project(current, applicable)
    verdict = _sanity.check(projected, scheme=scheme)
    if verdict["refusals"]:
        detail = "; ".join(r["message"] for r in verdict["refusals"])
        raise ApplierError(
            f"sanity check refused the apply: {detail}; nothing was written"
        )

    if not target.parent.is_dir():
        # §5: never create parent directories — a missing config directory
        # is an error, not a mkdir.
        raise ApplierError(
            f"target directory {target.parent} does not exist; not creating it"
        )

    # Capture the OLD values for the undo history entry before
    # the merge overwrites them (absent key -> None -> "(unset)").
    olds: List[Dict[str, Any]] = []
    for entry in applicable:
        node: Any = current
        for segment in str(entry.get("path", "")).split("."):
            if not isinstance(node, dict) or segment not in node:
                node = None
                break
            node = node[segment]
        olds.append({"path": entry.get("path"), "old": node,
                     "new": entry.get("new")})

    # §5.3 step 2 — backup (single slot; an existing backup is overwritten).
    backup = _sibling(target, BACKUP_SUFFIX)
    if target.exists():
        backup.write_bytes(target.read_bytes())
    else:
        # Zero-byte marker: the target did not exist before this apply, so
        # --restore returns to the no-file state (sound because the planner
        # aborts on any existing-but-invalid file).
        backup.write_bytes(b"")

    # §5.3 step 3 — merge + serialize.
    try:
        for entry in applicable:
            spec = tool_by_path(str(entry.get("path", "")))
            if spec is None:
                raise ApplierError(
                    f"refusing to write non-registry path {entry.get('path')!r}; "
                    "nothing was written"
                )
            if entry.get("unset"):
                # Undo of a key the apply CREATED: restore absence.
                _deep_unset(current, spec.path)
            else:
                _deep_set(current, spec.path, entry["new"])
        serialized = json.dumps(current, indent=4) + "\n"
    except ApplierError:
        raise
    except (TypeError, ValueError) as exc:
        raise ApplierError(f"serialization failed: {exc}; nothing was written") from exc

    # §5.3 step 4 — atomic replace via the tmp sibling.
    tmp = _sibling(target, TMP_SUFFIX)
    try:
        tmp.write_text(serialized, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ApplierError(
            f"write to {target} failed: {exc}; target and backup left untouched"
        ) from exc

    # Bounded undo history (after the write succeeded, so a
    # failed write leaves no phantom entry).
    history_note = ""
    if record_history:
        from . import history as _history
        entry = _history.record(file_path, label or "", olds)
        history_note = f"; undo entry #{entry['id']}"

    return {
        "written": True,
        "message": f"wrote {len(applicable)} change(s) to {target}; backup at {backup}{history_note}",
        "backup": str(backup),
        "changes": len(applicable),
        "sanity": {"warnings": verdict["warnings"], "notes": verdict["notes"]},
    }


def restore(file_path: Union[str, Path]) -> Dict[str, Any]:
    """§5.4: one-level undo. Requires ``<file>.assistant-backup``; a
    zero-byte backup deletes the target (restores the no-file state);
    otherwise the backup atomically replaces the target and the single
    slot is consumed."""
    target = _resolved(file_path)
    backup = _sibling(target, BACKUP_SUFFIX)
    if not backup.exists():
        raise ApplierError(f"no backup to restore ({backup} does not exist)")

    if backup.stat().st_size == 0:
        removed = target.exists()
        if removed:
            try:
                os.remove(target)
            except OSError as exc:
                raise ApplierError(f"cannot delete {target}: {exc}") from exc
        try:
            backup.unlink()
        except OSError as exc:
            raise ApplierError(f"cannot consume the backup slot {backup}: {exc}") from exc
        message = (
            f"deleted {target} (restored the no-file state); backup slot consumed"
            if removed
            else f"{target} was already absent; backup slot consumed"
        )
        return {"restored": True, "message": message}

    try:
        os.replace(backup, target)
    except OSError as exc:
        raise ApplierError(f"cannot restore {target} from {backup}: {exc}") from exc
    return {
        "restored": True,
        "message": f"restored {target} from its backup; the single backup slot is consumed",
    }

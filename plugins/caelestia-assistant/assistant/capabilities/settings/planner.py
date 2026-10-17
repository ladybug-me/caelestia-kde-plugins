"""Planner for the settings layer (the settings design notes (see module docstrings) §4): ops -> a validated plan.

Guarantees:

- Read-only until the applier runs: the planner READS the target file once
  and never writes anything, anywhere.
- Abort untouched on a broken target: an existing file that is not valid
  JSON, or not a JSON object, raises PlannerError and the CLI exits 1 with
  the file untouched (#120: "it simply isn't applied" — and the running
  shell's own loader would retry-and-toast on it, [C10]).
- No coercion: an existing value whose type disagrees with the registry
  kind (or a non-object intermediate key) aborts that ENTRY with a
  TYPE_MISMATCH error — never a silent cast or structural repair.
- #120's range rule: an absolute out-of-range value becomes a REJECTED
  entry carrying the allowed range; it is listed, not silently dropped.
  Relative ops (multiply/step) are CLAMPED to the registry range with a
  visible ``clamped`` notice (§7d of the settings design notes (see module docstrings)).
- All-or-nothing: any entry error marks the whole plan ``apply_blocked``;
  the applier then refuses to write anything (§4.5).
- Per-monitor overrides are only READ, best-effort, to warn that a global
  change will be shadowed; no monitor file is ever written (§4.6).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from assistant.adapters.caelestia.registry import ToolSpec, tool_by_name


class PlannerError(RuntimeError):
    """Raised when the target file cannot be planned against safely."""


def _read_current(file_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """§4.1: read and parse the target file; missing file -> defaults."""
    if not file_path.exists():
        return {}, ["target file does not exist; defaults assumed (file will be created on apply)"]
    try:
        raw = file_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PlannerError(f"cannot read target file {file_path}: {exc}") from exc
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        raise PlannerError(
            f"target file {file_path} is not valid JSON ({exc}); refusing to touch it"
        ) from exc
    if not isinstance(loaded, dict):
        raise PlannerError(
            f"target file {file_path} is valid JSON but not an object; refusing to touch it"
        )
    # §4.1: deep-copy the parsed object; the plan holds plain values only.
    current = json.loads(json.dumps(loaded))
    return current, []


def _walk(current: Dict[str, Any], spec: ToolSpec) -> Tuple[Optional[Any], Optional[str]]:
    """§4.2: resolve the leaf's current value.

    Returns (value, absent_note). Raises _TypeMismatch when an intermediate
    segment exists but is not an object.
    """
    segments = spec.path.split(".")
    node: Any = current
    absent_note: Optional[str] = None
    for segment in segments[:-1]:
        if segment not in node:
            return spec.default, "key was absent; default assumed"
        node = node[segment]
        if not isinstance(node, dict):
            raise _TypeMismatch(
                f"'{segment}' exists but is not an object; cannot resolve {spec.path}"
            )
    leaf = segments[-1]
    if leaf not in node:
        return spec.default, "key was absent; default assumed"
    return node[leaf], absent_note


class _TypeMismatch(Exception):
    """Internal: the existing structure disagrees with the registry."""


def _existing_type_ok(spec: ToolSpec, value: Any) -> bool:
    """§4.3 type check — bool is checked with ``type() is`` because Python's
    ``json`` parses ``true`` to ``bool`` and ``1 == True``."""
    if spec.kind in ("float", "int"):
        return type(value) in (int, float)  # excludes bool (bool subclasses int)
    if spec.kind == "bool":
        return type(value) is bool
    if spec.kind == "enum":
        return isinstance(value, str)
    return False


def _clamp(spec: ToolSpec, value: float) -> Tuple[float, bool]:
    clamped = False
    if spec.minimum is not None and value < spec.minimum:
        value = spec.minimum
        clamped = True
    if spec.maximum is not None and value > spec.maximum:
        value = spec.maximum
        clamped = True
    return value, clamped


def _finalize_number(spec: ToolSpec, value: float) -> float:
    """§4.4 deterministic rounding: float tools -> round(new, 2); int tools
    -> int(new + 0.5) (all int-tool ranges are non-negative)."""
    if spec.kind == "float":
        return round(float(value), 2)
    return int(value + 0.5)


def _resolve_op(op: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one parser op against the current file state -> one entry."""
    spec = tool_by_name(str(op.get("tool", "")))
    if spec is None:
        return _error_entry(None, op, f"unknown tool {op.get('tool')!r}")

    action = op.get("action")
    raw = op.get("raw")

    try:
        old, absent = _walk(current, spec)
    except _TypeMismatch as exc:
        return _error_entry(spec, op, f"TYPE_MISMATCH: {exc}")

    if not _existing_type_ok(spec, old):
        return _error_entry(
            spec, op,
            "TYPE_MISMATCH: existing value "
            f"{json.dumps(old) if old is not None else 'null'} at {spec.path} does not match "
            f"the registry kind '{spec.kind}'; refusing to coerce",
        )

    value = op.get("value")

    def _compose(detail: Optional[str], extra: Optional[str] = None,
                 clamped_bound: Optional[float] = None) -> Optional[str]:
        """Join the note parts: action detail, absent-key notice, clamped
        notice, the tool's standing inversion note (§2), and the op's own
        honest note from the parser (§3.5)."""
        parts: List[str] = []
        if detail:
            parts.append(detail)
        if absent:
            parts.append(absent)
        if clamped_bound is not None:
            parts.append(f"clamped to {clamped_bound}")
        if spec is not None and spec.name == "setAnimationSpeed":
            parts.append("durations scale: lower = faster")
        if extra:
            parts.append(extra)
        return "; ".join(parts) if parts else None

    if action == "set":
        if spec.kind == "bool":
            if type(value) is not bool:
                # §3.4/§3.5: a number or percent on a bool tool is REJECTED,
                # never coerced.
                return _error_entry(
                    spec, op,
                    "this setting is on/off only; it has no magnitude",
                    old=old,
                )
            new: Any = value
        elif spec.kind == "enum":
            if value not in (spec.enum or ()):
                return _error_entry(
                    spec, op,
                    f"value {value!r} is not one of {'|'.join(spec.enum or ())}",
                    old=old,
                )
            new = value
        else:
            if type(value) not in (int, float) or isinstance(value, bool):
                return _error_entry(spec, op, f"value {value!r} is not a number", old=old)
            if (spec.minimum is not None and value < spec.minimum) or \
               (spec.maximum is not None and value > spec.maximum):
                return _error_entry(
                    spec, op,
                    f"absolute value {value} is outside the allowed range "
                    f"{spec.minimum}-{spec.maximum}; it is not applied",
                    old=old,
                )
            new = _finalize_number(spec, float(value))
        return _entry(
            spec, action, raw, old, new, False, new == old,
            _compose(None, extra=op.get("note")),
        )

    if action in ("multiply", "step"):
        if spec.kind == "bool":
            return _error_entry(
                spec, op, "this setting is on/off only; it has no magnitude", old=old
            )
        if spec.kind == "enum":
            return _error_entry(
                spec, op, "position has no magnitude; use a position word", old=old
            )
        if type(old) not in (int, float) or isinstance(old, bool):
            return _error_entry(
                spec, op, f"existing value at {spec.path} is not a number", old=old
            )
        base = float(old)
        if action == "multiply":
            factor = value if isinstance(value, (int, float)) and not isinstance(value, bool) else 1.0
            computed = base * factor
            detail = f"multiply by {factor} ({raw})"
        else:
            sign = 1 if (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0) else -1
            computed = base + spec.step * sign
            detail = f"step {raw} ({spec.step * sign:+g})"
        clamped_value, clamped = _clamp(spec, computed)
        new = _finalize_number(spec, clamped_value)
        bound: Optional[float] = None
        if clamped:
            bound = spec.minimum if (spec.minimum is not None and computed < spec.minimum) else spec.maximum
        return _entry(
            spec, action, raw, old, new, clamped, new == old,
            _compose(detail, extra=op.get("note"), clamped_bound=bound),
        )

    return _error_entry(spec, op, f"unknown action {action!r}", old=old)


def _entry(spec: ToolSpec, action: str, raw: Optional[str], old: Any, new: Any,
           clamped: bool, no_op: bool, note: Optional[str]) -> Dict[str, Any]:
    entry = {
        "tool": spec.name,
        "path": spec.path,
        "action": action,
        "raw": raw,
        "old": old,
        "new": new,
        "clamped": clamped,
        "no_op": no_op,
        "note": note,
    }
    return entry


def _error_entry(spec: Optional[ToolSpec], op: Dict[str, Any], error: str,
                 old: Any = None) -> Dict[str, Any]:
    entry = {
        "tool": spec.name if spec else str(op.get("tool", "")),
        "path": spec.path if spec else "",
        "action": op.get("action"),
        "raw": op.get("raw"),
        "old": old,
        "new": None,  # an errored entry is never applied
        "clamped": False,
        "no_op": False,
        "note": None,
        "error": error,
    }
    return entry


def _monitor_warnings(entries: List[Dict[str, Any]], target: Path) -> List[str]:
    """§4.6: read-only, best-effort shadow warning for per-monitor overrides.

    No monitor file is ever written; unparseable/unreadable ones are skipped
    silently (they are not ours to judge). global_only tools skip the check —
    the loader quarantines global-only keys in overlay files anyway ([C23]).
    """
    notes: List[str] = []
    monitors_dir = target.parent / "monitors"
    if not monitors_dir.is_dir():
        return notes
    seen: set = set()
    for monitor_file in sorted(glob.glob(str(monitors_dir / "*" / "shell.json"))):
        try:
            data = json.loads(Path(monitor_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        screen = Path(monitor_file).parent.name
        for entry in entries:
            spec = tool_by_name(entry.get("tool", ""))
            if spec is None or spec.global_only:
                continue
            key = (screen, spec.path)
            if key in seen:
                continue
            node: Any = data
            present = True
            for segment in spec.path.split("."):
                if not isinstance(node, dict) or segment not in node:
                    present = False
                    break
                node = node[segment]
            if present:
                seen.add(key)
                notes.append(
                    f"screen '{screen}' has a per-monitor override for this key "
                    f"({spec.path}); the global change will be shadowed there"
                )
    return notes


def plan(ops: List[Dict[str, Any]], file_path: Union[str, Path]) -> Dict[str, Any]:
    """Resolve parser ops against the target file's current values (§4).

    Returns the plan dict: verdict/file/entries/notes/errors/apply_blocked.
    Raises PlannerError when the target file exists but is invalid JSON or
    not an object (the CLI exits 1, the file is never touched).
    """
    target = Path(file_path)
    current, notes = _read_current(target)

    entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for op in ops:
        entry = _resolve_op(op, current)
        entries.append(entry)
        if entry.get("error"):
            errors.append({"tool": entry["tool"], "path": entry["path"], "error": entry["error"]})

    notes.extend(_monitor_warnings(entries, target))

    return {
        "verdict": "INTENT",
        "file": str(target),
        "entries": entries,
        "notes": notes,
        "errors": errors,
        "apply_blocked": bool(errors),
    }

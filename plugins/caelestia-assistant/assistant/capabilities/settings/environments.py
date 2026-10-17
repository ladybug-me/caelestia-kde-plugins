"""assistant.capabilities.settings.environments — named environment snapshots (F17)
and portable export/import (F18) for exponential-build-5.

Issue #120's Phase 3 roadmap: "workspace profiles ... monitor-specific
layouts ... workspace automation". Before any of those can be safe,
there needs to be a way to capture the WHOLE current environment, name
it, get back to it, and move it between machines. That is this module.

An ENVIRONMENT is a named snapshot of:

- the target shell.json content (parsed, canonical-hash pinned),
- the saved macros and profiles lists (captured metadata, so an export
  carries the user's composed bundles; RESTORE does not touch them —
  they are restored through their own explicit commands, F9/C3).

DISCIPLINE (the whole point):

- STORAGE: inside the existing undo-history file under a bounded
  ``"environments"`` key (MAX_ENVIRONMENTS=8, FIFO) — the same file,
  the same ``history._save`` atomic path, the same sibling path as the
  ``undo_log`` (A3), ``macros`` (C3) and ``profiles`` (F9) precedents.
  The documented write scope of the settings layer does not grow.
- RESTORE IS A PLAN, NOT A WRITE: the snapshot's registry-known keys
  are turned into plain planner ops (set for changed/new values, unset
  for keys the snapshot lacks that the live file has), validated by the
  ordinary planner, previewed, and applied only behind the standard
  --apply/--confirm gates. Keys the registry does not manage are
  REPORTED, never touched — the assistant does not know their shape
  and refuses to guess.
- INTEGRITY: every snapshot pins the sha256 of its canonical content;
  ``show`` verifies and names mismatches (a snapshot that fails its own
  hash is an error entry, never a restore source).

F18 EXPORT/IMPORT (same module, the migration half):

- export writes a PORTABLE BUNDLE to a caller-chosen path: schema
  version, the snapshot, and provenance (registry tool count at capture
  time, created_at). Nothing else is written.
- import is REVIEW-ONLY: the bundle is validated (schema, hash,
  registry drift), the shell.json content is diffed against the live
  file, and the environment is stored as a NEW named snapshot marked
  ``imported``; nothing is applied by importing. The user previews and
  restores through the ordinary F17 path if they want it.

Pure module apart from the documented history-file writes behind the
explicit verbs: no execution, no network, no RNG.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import history
from assistant.adapters.caelestia.registry import TOOL_SPECS

__all__ = ["EnvironmentError", "MAX_ENVIRONMENTS", "ENVIRONMENTS_KEY",
           "SCHEMA_VERSION", "NAME_RE", "canonical_hash", "save",
           "list_environments", "delete", "get", "diff_to_ops",
           "export_bundle", "import_bundle"]

MAX_ENVIRONMENTS = 8
ENVIRONMENTS_KEY = "environments"
SCHEMA_VERSION = 1

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,31}")

TOOL_PATHS = {spec.path: spec.name for spec in TOOL_SPECS}

FileTarget = Union[str, Path]


class EnvironmentError(ValueError):
    """Raised for every refusal (unknown environment, bad name, hash
    mismatch, schema drift); the caller renders the reason."""


def canonical_hash(content: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_envs(target) -> List[Dict[str, Any]]:
    return [dict(e) for e in
            history._load(target).get(ENVIRONMENTS_KEY, [])]


def _save_envs(target, envs: List[Dict[str, Any]]) -> None:
    data = history._load(target)
    data[ENVIRONMENTS_KEY] = envs
    history._save(target, data)


def _registry_managed(node: Any, prefix: str = "",
                      out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten a config tree to dotted paths, keeping ONLY the paths the
    registry manages (the planner can validate and write those)."""
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _registry_managed(value, path, out)
            elif path in TOOL_PATHS:
                out[path] = value
    return out


def read_config(target) -> Tuple[Dict[str, Any], List[str]]:
    """The live file parsed ({} when absent) + the list of problems
    (unparseable content is an honest empty-with-problems, never a
    crash — a restore against a corrupt file must still be previewable)."""
    try:
        text = Path(target).read_text(encoding="utf-8")
    except OSError:
        return {}, ["the target file does not exist yet"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, [f"the target file is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return {}, ["the target file is not a JSON object"]
    return data, []


def save(target, name: str) -> Dict[str, Any]:
    """Snapshot the current environment (target content + macros +
    profiles metadata) under a bounded, hash-pinned name. Writes only
    the history file's "environments" key. Duplicate names refuse
    (delete first); the store is FIFO-bounded."""
    if not NAME_RE.fullmatch(str(name or "")):
        raise EnvironmentError(
            f"environment names must match {NAME_RE.pattern!r} — got "
            f"{name!r}")
    content, problems = read_config(target)
    if problems and content == {} and "not exist" not in problems[0]:
        raise EnvironmentError(f"cannot snapshot: {problems[0]}")
    data = history._load(target)
    envs = data.get(ENVIRONMENTS_KEY, [])
    if any(e.get("name") == name for e in envs):
        raise EnvironmentError(
            f"environment {name!r} already exists; delete it first "
            f"(--env-delete {name})")
    entry = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_hash": canonical_hash(content),
        "content": content,
        "registry_tools": len(TOOL_SPECS),
        "macros": [dict(m) for m in data.get("macros", [])],
        "profiles": [dict(p) for p in data.get("profiles", [])],
    }
    envs.append(entry)
    data[ENVIRONMENTS_KEY] = envs[-MAX_ENVIRONMENTS:]  # FIFO bound
    history._save(target, data)
    return {k: v for k, v in entry.items() if k != "content"} | {
        "paths": _count_leaves(content)}


def _count_leaves(node: Any) -> int:
    if isinstance(node, dict):
        return sum(_count_leaves(v) for v in node.values())
    return 1


def list_environments(target) -> List[Dict[str, Any]]:
    """The saved environments, oldest first, WITHOUT their content
    (read-only)."""
    return [{k: v for k, v in env.items() if k != "content"}
            for env in _load_envs(target)]


def get(target, name: str, verify: bool = True) -> Dict[str, Any]:
    """One environment by name. With verify, a hash mismatch raises —
    a snapshot that fails its own integrity is never a restore source."""
    for env in _load_envs(target):
        if env.get("name") == name:
            if verify:
                expected = env.get("content_hash")
                actual = canonical_hash(env.get("content") or {})
                if expected != actual:
                    raise EnvironmentError(
                        f"environment {name!r} FAILED its integrity check "
                        f"(stored hash {expected!r} != actual {actual!r}); "
                        f"it will not be restored or exported")
            return env
    saved = [str(e.get("name")) for e in _load_envs(target)]
    raise EnvironmentError(
        f"unknown environment {name!r}; saved environments: "
        + (", ".join(saved) if saved else "(none yet; save one with "
           "--env-save)"))


def delete(target, name: str) -> Dict[str, Any]:
    """Remove one environment by name (an explicit user command)."""
    envs = _load_envs(target)
    kept = [e for e in envs if e.get("name") != name]
    if len(kept) == len(envs):
        raise EnvironmentError(
            f"unknown environment {name!r}; saved environments: "
            + (", ".join(str(e.get("name")) for e in envs) if envs
               else "(none)"))
    _save_envs(target, kept)
    return {"deleted": name}




def _flatten(node: Any, prefix: str = "",
             out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten a config tree to dotted paths (all leaves)."""
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, path, out)
            else:
                out[path] = value
    return out


def _unknown(d: Dict[str, Any]) -> List[str]:
    all_flat = _flatten(d)
    return sorted(p for p in all_flat if p not in TOOL_PATHS)


def diff_to_ops(target, name: str) -> Dict[str, Any]:
    """The restore plan's RAW MATERIAL (read-only, no planner call):
    registry-known paths that differ between the live file and the
    snapshot become {tool, action, value} set ops (and unset ops for
    live keys the snapshot lacks); unknown keys are REPORTED, split by
    direction (only-in-snapshot, only-in-live), and never planned."""
    env = get(target, name)  # verifies integrity
    snapshot: Dict[str, Any] = env.get("content") or {}
    live, _problems = read_config(target)
    snap_flat = _registry_managed(snapshot)
    live_flat = _registry_managed(live)
    set_ops: List[Dict[str, Any]] = []
    reset_ops: List[Dict[str, Any]] = []
    for path in sorted(set(snap_flat) | set(live_flat)):
        if path in snap_flat and path not in live_flat:
            set_ops.append({"tool": TOOL_PATHS[path], "action": "set",
                            "value": snap_flat[path],
                            "raw": f"env restore {name}: {path}="
                                   f"{snap_flat[path]!r}"})
        elif path in snap_flat and path in live_flat and \
                snap_flat[path] != live_flat[path]:
            set_ops.append({"tool": TOOL_PATHS[path], "action": "set",
                            "value": snap_flat[path],
                            "raw": f"env restore {name}: {path} "
                                   f"{live_flat[path]!r} -> "
                                   f"{snap_flat[path]!r}"})
        elif path not in snap_flat and path in live_flat:
            spec = next((s for s in TOOL_SPECS if s.path == path), None)
            default = getattr(spec, "default", None)
            reset_ops.append({"tool": TOOL_PATHS[path], "action": "set",
                              "value": default,
                              "raw": f"env restore {name}: {path} "
                                     f"{live_flat[path]!r} -> registry "
                                     f"default {default!r} (the snapshot "
                                     f"lacks this key; the applier cannot "
                                     f"remove keys, so restore resets it)"})
    return {
        "name": name,
        "set_ops": set_ops,
        "reset_ops": reset_ops,
        "unknown_only_in_snapshot": _unknown(snapshot),
        "unknown_only_in_live": _unknown(live),
        "identical": not (set_ops or reset_ops),
    }


# ---------------------------------------------------------------------------
# F18: portable export / review-only import.
# ---------------------------------------------------------------------------

def export_bundle(target, name: str) -> Dict[str, Any]:
    """The portable bundle for one verified environment (a plain dict —
    the CLI writes it to the user-chosen path; this module never picks
    paths)."""
    env = get(target, name)  # verifies integrity
    return {
        "schema": SCHEMA_VERSION,
        "kind": "caelestia-assistant-environment",
        "created_at": env.get("created_at"),
        "registry_tools": env.get("registry_tools"),
        "content_hash": env.get("content_hash"),
        "content": env.get("content"),
        "macros": env.get("macros", []),
        "profiles": env.get("profiles", []),
    }


def import_bundle(target, bundle: Dict[str, Any],
                  name: str) -> Dict[str, Any]:
    """Import a bundle as a NEW snapshot, REVIEW-ONLY: schema and hash
    are validated, registry drift is measured and reported, and the
    import lands as a stored environment marked imported=true. NOTHING
    is applied — restoring rides the ordinary F17 diff/plan/consent
    path. The target file itself is never written by the import."""
    if not isinstance(bundle, dict) or \
            bundle.get("kind") != "caelestia-assistant-environment":
        raise EnvironmentError(
            "not a caelestia-assistant environment bundle (missing kind)")
    if bundle.get("schema") != SCHEMA_VERSION:
        raise EnvironmentError(
            f"unsupported bundle schema {bundle.get('schema')!r} "
            f"(expected {SCHEMA_VERSION}) — re-export from the source "
            f"machine")
    content = bundle.get("content")
    if not isinstance(content, dict):
        raise EnvironmentError("bundle content is not a JSON object")
    actual = canonical_hash(content)
    if actual != bundle.get("content_hash"):
        raise EnvironmentError(
            f"bundle FAILED its integrity check (declared "
            f"{bundle.get('content_hash')!r} != actual {actual!r})")
    if not NAME_RE.fullmatch(str(name or "")):
        raise EnvironmentError(
            f"environment names must match {NAME_RE.pattern!r} — got "
            f"{name!r}")
    data = history._load(target)
    envs = data.get(ENVIRONMENTS_KEY, [])
    if any(e.get("name") == name for e in envs):
        raise EnvironmentError(
            f"environment {name!r} already exists locally; import under "
            f"a different name")
    drift = {"bundled_tools": bundle.get("registry_tools"),
             "local_tools": len(TOOL_SPECS)}
    unknown = sorted(p for p in _flatten(content) if p not in TOOL_PATHS)
    entry = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_hash": actual,
        "content": content,
        "registry_tools": len(TOOL_SPECS),
        "macros": bundle.get("macros", []),
        "profiles": bundle.get("profiles", []),
        "imported": True,
        "import_drift": drift,
        "import_unknown_paths": unknown,
    }
    envs.append(entry)
    data[ENVIRONMENTS_KEY] = envs[-MAX_ENVIRONMENTS:]
    history._save(target, data)
    return {k: v for k, v in entry.items() if k != "content"}

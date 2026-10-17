"""assistant.capabilities.settings.monitors — monitor-aware planning (F20,
exponential-build-5; issue #120 b6d, closed from NOT_IMPLEMENTED).

What upstream actually provides (verified against the checkout, cited):

- EVERY config type layers per-screen overrides: ``Root::forScreen``
  (shell/plugin/src/Caelestia/Config/rootnodes.cpp:87-94) reads
  ``monitorConfigDir()`` (shell/plugin/src/Caelestia/Config/common.cpp:15-17,
  i.e. ``<configDir>/monitors``) with the SAME file name as the global
  config — so ``~/.config/caelestia/monitors/<screenName>/shell.json``
  overrides shell.json for that screen, same schema;
- the shipped Nexus DesktopPage writes per-screen values through
  ``GlobalConfig.forScreen`` (shell/modules/nexus/pages/DesktopPage.qml:33);
- a handful of GLOBAL keys select monitor BEHAVIOR (notifs.monitor
  "all"|"focused", bar.workspaces.perMonitor, tabSwitch.allScreens, ...)
  — the registry already carries them with their own citations.

What a stdlib-only assistant CANNOT do (honest): enumerate the
CONNECTED screens — that lives in Quickshell/Wayland. This module
therefore works with the override FILES that exist (or are explicitly
named) and never invents screen names.

Design: ``--monitor NAME`` is a TARGET RESOLVER, not a new writer. It
points the ordinary settings machinery (parser, planner, applier,
backup sibling, history) at
``<target dir>/monitors/<NAME>/shell.json``; everything else — dry-run
default, validation, confirmation, undo — is exactly the machinery
that already exists for shell.json. ``--monitors`` is the read-only
report: discovered override files, their registry-valid/unknown keys,
and the monitor-behavior tools with citations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant.adapters.caelestia.registry import TOOL_SPECS

__all__ = ["MONITOR_NAME_RE", "monitors_dir", "resolve_target",
           "discover", "monitor_behavior_tools"]

MONITOR_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}")

# Registry tools whose SEMANTICS select or scope monitor behavior
# (path-verified; each carries its own upstream citations).
_BEHAVIOR_PATTERNS = (
    "monitor", "screen", "allscreens",
)


def monitors_dir(target: Path) -> Path:
    """The monitors override directory for a config target: siblings of
    the target file (the target's parent IS the config dir)."""
    return Path(target).parent / "monitors"


def resolve_target(target: Path, monitor: str) -> Path:
    """The override file for one screen name. The name is validated
    (no separators, no traversal — a screen name is a single path
    component); the file itself need not exist yet (planning against a
    not-yet-written override is the same as planning against an empty
    config)."""
    if not MONITOR_NAME_RE.fullmatch(str(monitor or "")):
        raise ValueError(
            f"monitor names must match {MONITOR_NAME_RE.pattern!r} "
            f"(a single path component; no '/', no '..') — got "
            f"{monitor!r}")
    return monitors_dir(target) / str(monitor) / Path(target).name


def _flatten(node: Any, prefix: str = "",
             out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


def discover(target: Path) -> List[Dict[str, Any]]:
    """The override files that ALREADY exist under monitors/ (read-only
    discovery of files, not of connected screens — see the module
    docstring). Each entry reports how many of its keys are
    registry-managed and how many are unknown to the registry."""
    import json
    base = monitors_dir(target)
    found: List[Dict[str, Any]] = []
    if not base.is_dir():
        return found
    known_paths = {spec.path for spec in TOOL_SPECS}
    for screen_dir in sorted(base.iterdir()):
        if not screen_dir.is_dir():
            continue
        override = screen_dir / Path(target).name
        if not override.is_file():
            continue
        managed = unknown = 0
        try:
            content = json.loads(override.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            found.append({"screen": screen_dir.name, "path": str(override),
                          "error": "not valid JSON"})
            continue
        for path in _flatten(content if isinstance(content, dict) else {}):
            if path in known_paths:
                managed += 1
            else:
                unknown += 1
        found.append({"screen": screen_dir.name, "path": str(override),
                      "managed_keys": managed, "unknown_keys": unknown})
    return found


def monitor_behavior_tools() -> List[Dict[str, Any]]:
    """The global registry tools that select or scope monitor behavior,
    with their citations (read-only metadata for the --monitors
    report)."""
    rows = []
    for spec in TOOL_SPECS:
        atoms = (spec.name + " " + spec.path).lower().replace(".", " ")
        if any(pattern in atoms for pattern in _BEHAVIOR_PATTERNS):
            rows.append({
                "name": spec.name,
                "path": spec.path,
                "kind": spec.kind,
                "default": spec.default,
                "enum": list(spec.enum) if spec.enum else None,
                "citations": list(spec.citations),
            })
    return rows

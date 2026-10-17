"""Per-monitor-topology config memory (issue #120 Phase 3 roadmap:
"monitor-specific layouts", implemented as a Phase 2.2 memory engine).

What it does, in one sentence: hash the connected-monitor set, and when the
set changes, look up the config deltas that were active the last time that
exact set was connected, and PROPOSE them in the ledger — never auto-apply.

Data sources (verified against the upstream caelestia-kde tree before
writing a line of this module — none are invented):

- The per-monitor override directories the shell's own config loader
  maintains: ``<shell.json's dir>/monitors/<screen-name>/shell.json``
  (see shell/plugin/src/Caelestia/Config/common.cpp::monitorConfigDir and
  assistant/settings/planner.py::_monitor_warnings, which already scans
  the same layout read-only). The set of screen-name directories is the
  persisted "connected-monitor set" this module hashes.
- Resolution/position are deliberately NOT part of the fingerprint: the
  upstream shell persists no such data on disk (the workspace-tracker
  effect broadcasts live state only; KWin's kwinoutputconfig.json is not
  read by the shell), and inventing a schema the shell does not write is
  exactly what this task forbids. The fingerprint is the sorted screen
  name set — count and identity, honestly scoped.

Deltas: for each monitor override file, the entries whose value differs
from the registry default are recorded as {"tool", "path", "value"} rows
(registry paths only; global-only tools are skipped — they cannot live in
a per-monitor overlay, matching planner's own [C6]/§4.6 discipline).

Writes: the learned memory lives in the brain's one state JSON (an
enumerated write path), and proposals go through the existing ledger —
the same two mechanisms every other brain feature uses. Nothing is ever
written to shell.json by this module.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from assistant.adapters.caelestia.registry import tool_by_path

FileTarget = Union[str, Path]

STATE_KEY = "topology"


def fingerprint(monitors_dir: Path) -> Dict[str, Any]:
    """Stable fingerprint of the connected-monitor set.

    The hash is sha256 over the sorted screen names — deterministic across
    runs and machines, as a memory key must be. A missing monitors dir is
    the single-monitor "no overrides" topology: it hashes the empty set,
    it is not an error.
    """
    monitors_dir = Path(monitors_dir)
    # Every monitor the shell knows about gets a directory here, even one
    # carrying no overrides yet — so the fingerprint hashes the DIRECTORY
    # set (the connected set), while collect_deltas still reads only the
    # shell.json files inside them.
    names = sorted(
        p.name for p in monitors_dir.iterdir() if p.is_dir()
    ) if monitors_dir.is_dir() else []
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    return {"monitors": names, "hash": digest}


def _dotted_get(node: Any, dotted: str) -> Any:
    for segment in dotted.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def collect_deltas(target: FileTarget) -> List[Dict[str, Any]]:
    """Non-default settings found in per-monitor override files.

    Read-only. Returns [{"monitor", "tool", "path", "value"}]. Registry
    paths only; global-only tools are skipped (they cannot live in a
    per-monitor overlay — the loader quarantines them, per planner §4.6);
    unparseable override files are skipped silently (not ours to judge).
    """
    target = Path(target)
    monitors_dir = target.parent / "monitors"
    deltas: List[Dict[str, Any]] = []
    for monitor_file in sorted(glob.glob(str(monitors_dir / "*" / "shell.json"))):
        screen = Path(monitor_file).parent.name
        try:
            data = json.loads(Path(monitor_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for path, value in _walk_leaves(data):
            spec = tool_by_path(path)
            if spec is None or spec.global_only:
                continue
            if value == spec.default:
                continue  # default values are not deltas
            deltas.append({
                "monitor": screen,
                "tool": spec.name,
                "path": spec.path,
                "value": value,
            })
    return deltas


# since exponential-build-3 Phase 2 the leaf walker is the ONE
# implementation in settings/lint.py::flatten_leaves (the repo's own
# MinHash scan caught this file's identical private copy)
from assistant.capabilities.settings.lint import flatten_leaves as _walk_leaves


def observe(target: FileTarget, state_path: str,
            ledger=None, propose: bool = False,
            reason: Optional[str] = None) -> Dict[str, Any]:
    """One observe step: fingerprint the topology, remember its deltas,
    and — when the topology CHANGED and we have memory of this exact set
    from before — emit ONE ledger proposal to reapply the remembered
    deltas. Never writes to any config file.

    Returns {"fingerprint", "changed", "remembered": bool, "proposal_id"}.
    """
    from . import state as st  # local import: state owns the one JSON write

    target_path = Path(target)
    fp = fingerprint(target_path.parent / "monitors")
    deltas = collect_deltas(target)

    memory = st.load(state_path)
    topo = memory.setdefault(STATE_KEY, {})
    by_hash = topo.setdefault("by_hash", {})
    last = topo.get("last_fingerprint")

    changed = last is not None and last != fp["hash"]
    remembered = changed and fp["hash"] in by_hash and bool(by_hash[fp["hash"]]["deltas"])

    proposal_id = None
    if propose and remembered and ledger is not None:
        rows = by_hash[fp["hash"]]["deltas"]
        calls = [{"tool": r["tool"], "action": "set", "value": r["value"],
                  "raw": f"topology {fp['hash']}: {r['tool']}={r['value']!r} "
                         f"(was set on {r['monitor']})"}
                 for r in rows]
        proposal_id = ledger.propose(
            "topology_restore", str(target_path),
            {"fingerprint": fp, "calls": calls, "file": str(target_path)},
            reason or (f"monitor set {fp['monitors']} reconnected; last time "
                       f"it had {len(rows)} override(s) active"),
            confidence=0.75)

    # Remember the CURRENT topology's deltas under its own hash and mark it
    # last-seen, whether or not a proposal fired.
    by_hash[fp["hash"]] = {"deltas": deltas,
                           "monitors": fp["monitors"],
                           "recorded_at": _dt.datetime.now(
                               _dt.timezone.utc).isoformat(timespec="seconds")}
    topo["last_fingerprint"] = fp["hash"]
    st.save(memory, state_path)

    return {"fingerprint": fp, "changed": changed,
            "remembered": bool(remembered), "proposal_id": proposal_id,
            "deltas": deltas}

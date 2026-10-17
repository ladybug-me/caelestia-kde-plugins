"""Named presets for the settings layer (issue #120).

Issue #120's Nexus-integration list names the directions verbatim —
"Optimize for gaming", "Optimize for battery life", "Make this look more
like macOS", "Make this look more minimal" — and this module adds
"compact". A preset here is STRICTLY a bundle of already-validated tool
calls: the data lives in tools.json (generated from curations.PRESETS,
where build_registry._preset_value_ok FAILS THE BUILD if any call violates
its tool's validation), and applying a preset is nothing but planning
those calls through the ordinary planner/applier path. No preset ever
introduces an unvalidated code path.

Presets are multi-change by construction, so they always go through the
preview-then-confirm flow (cli.py gates the write behind --confirm / an
interactive y-N; the QML service shows the confirm card).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from assistant.adapters.caelestia.registry import PRESETS

FileTarget = Union[str, Path]


class PresetError(RuntimeError):
    """Raised when a preset name is unknown or a bundle is malformed."""


def presets() -> List[Dict[str, Any]]:
    """Every preset: {name, label, description, calls: [(tool, value)]}."""
    return [dict(p) for p in PRESETS]


def preset_by_name(name: str) -> Optional[Dict[str, Any]]:
    for preset in PRESETS:
        if str(preset["name"]) == name:
            return dict(preset)
    return None


def preset_ops(name: str) -> List[Dict[str, Any]]:
    """A preset's bundle as planner ops (set actions, raw = provenance)."""
    preset = preset_by_name(name)
    if preset is None:
        raise PresetError(
            f"unknown preset {name!r}; known: "
            + ", ".join(str(p["name"]) for p in PRESETS)
        )
    ops: List[Dict[str, Any]] = []
    for tool, value in preset["calls"]:  # type: ignore[misc]
        ops.append({
            "tool": str(tool),
            "action": "set",
            "value": value,
            "raw": f"preset {name}: {tool}={value!r}",
        })
    return ops


def describe_lines() -> List[str]:
    """The --list-presets table."""
    lines = ["caelestia assistant — settings presets "
             f"({len(PRESETS)} named bundles of validated tool calls):", ""]
    for preset in PRESETS:
        calls = ", ".join(f"{tool}={value!r}"
                          for tool, value in preset["calls"])  # type: ignore[misc]
        lines.append(f"  {preset['name']:<14} {preset['label']}")
        lines.append(f"  {'':<14} {preset['description']}")
        lines.append(f"  {'':<14} calls: {calls}")
        lines.append("")
    lines.append("Presets are multi-change: applying one always shows the "
                 "preview and waits for confirmation.")
    return lines


# ---------------------------------------------------------------------------
# Preset state graph + bounded model-check audit (exponential-build-4 B)
# ---------------------------------------------------------------------------


def transition_graph() -> Dict[str, List[str]]:
    """The operational state graph of the preset system.

    States are the named presets plus "custom" (any state the applier
    did not produce from a preset). Transitions are "apply preset X":
    application is currently UNCONDITIONAL — from any state you can
    move to any preset state — so the graph is complete by
    construction, and the audit below proves the obvious. Its value is
    as a GUARD: the moment preset application grows preconditions
    (state prerequisites, ordering constraints), this graph is where
    the change lands and the audit is what catches a preset that
    became unreachable or a dead end — before a user ever sees it.
    """
    names = [str(p["name"]) for p in PRESETS] + ["custom"]
    return {state: [n for n in names if n != state] for state in names}


def audit(node_budget: int = 50_000) -> Dict[str, Any]:
    """Bounded model-check of transition_graph(): deadlock states
    (exact) and per-preset reachability within one step per hop count
    k = number of states (complete graph => SAT proves each in one
    hop). ABSTAINs loudly past the budget; additive, nothing gates on
    it — this is evidence for a human, not a permission."""
    from assistant.capabilities.genius import sat as sat_mod

    graph = transition_graph()
    return sat_mod.graph_audit(graph, "custom", k=len(graph),
                               node_budget=node_budget)

"""settings.consequences — what-if consequence-graph mode (phase 2.7).

Implements the former proposals/2026-09-26-c-whatif-consequences.md
(pruned in exponential-build-3 Phase 2 once implemented — this
docstring carries the design) as specified:
a HAND-CURATED, citation-backed table of KNOWN cross-key interactions in
the shell's own code; a bounded, cycle-safe projection over a candidate
op list; the existing AC-3 conflict check extended to see induced
values; and a read-only what-if surface that renders consequences and
reversibility BEFORE the user consents.

The honesty rule (the proposal's own): a cited-edge list can be WRONG if
upstream changes — so every edge's cited file:line is re-verified
against the checkout by test (the registry's citation-guard pattern,
extended here). The consequence universe is EXACTLY this table:
bounded, auditable, growable by reviewed diffs — NOT a general model,
and NOT "fake open-endedness".

Purely read-only: this module reads op lists and the edge table; it
never writes, never applies, and nothing here is reachable from an NL
request that would apply — the parser/planner/applier spine is
untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["EDGES", "project", "render"]

# ---------------------------------------------------------------------------
# The interaction graph: every edge cites the shipped code that makes it
# true. Line numbers refer to the caelestia-kde checkout this registry is
# built from (verified by assistant/settings/tests/test_consequences.py —
# an edge whose cited line no longer matches its claimed content is a
# test failure, never a silent stale edge).
#
# Edge shape: {trigger_path, trigger_when, effect_path, effect, direction,
#              citation, confidence, note, effect_value}
#   trigger_when:  the value/condition of the trigger that arms the edge
#   effect:        what the trigger does to the effect path (human words)
#   effect_value:  the concrete value the edge induces on effect_path, when
#                  the edge sets one (machine-readable — feeds the AC-3
#                  induced-value check; edges that only annotate an INERT or
#                  CLAMP state carry no effect_value because they change no
#                  shell.json value)
# ---------------------------------------------------------------------------

EDGES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "transparency-off-forces-blur-off",
        "trigger_path": "appearance.transparency.enabled",
        "when": {"op_is": False},
        "effect_path": "appearance.blur",
        "effect": "Nexus toggles it off with you (the toggle handler "
                  "sets blur = false when transparency goes off)",
        "effect_value": False,
        "citation": "shell/modules/nexus/pages/wallandstyle/AppearancePage.qml:132-137",
        "claimed_content": "onToggled: { GlobalConfig.appearance.transparency"
                           ".enabled = checked if (!checked) { "
                           "GlobalConfig.appearance.blur = false } }",
        "confidence": "high",
    },
    {
        "id": "blur-inert-without-transparency",
        "trigger_path": "appearance.blur",
        "when": {"op_is": True, "requires": {"path": "appearance.transparency.enabled",
                                             "equals": False}},
        "effect_path": "appearance.blur",
        "effect": "INERT: the blur regions gate on transparency.enabled "
                  "&& blur together, so blur does nothing until "
                  "transparency is re-enabled",
        "citation": "shell/modules/drawers/blur/BlurOffsets.qml:17",
        "claimed_content": "isActive: target && target.visible && target."
                           "opacity > 0 && GlobalConfig.appearance."
                           "transparency.enabled && GlobalConfig."
                           "appearance.blur",
        "confidence": "high",
    },
    {
        "id": "bar-scale-floor",
        "trigger_path": "bar.scale",
        "when": {"op_below": 0.6},
        "effect_path": "bar.scale (effective)",
        "effect": "CLAMPED at 0.6 at render time: the bar wrapper floors "
                  "the scale with Math.max(0.6, ...) — the shell.json "
                  "value you asked for is not what renders",
        "citation": "shell/modules/bar/BarWrapper.qml:24",
        "claimed_content": "readonly property real barScale: Math.max(0.6, "
                           "!isNaN(Config.bar.scale) ? Config.bar.scale : "
                           "1.0)",
        "confidence": "high",
    },
    {
        "id": "dodge-needs-persistent",
        "trigger_path": "bar.dodgeWindows",
        "when": {"op_is": True, "requires": {"path": "bar.persistent",
                                             "equals": False}},
        "effect_path": "bar.dodgeWindows (effective)",
        "effect": "INERT: dodge requires persistent windows "
                  "(dodgeEnabled = dodgeWindows && persistent && !disabled)",
        "citation": "shell/modules/bar/BarWrapper.qml:27",
        "claimed_content": "readonly property bool dodgeEnabled: Config.bar."
                           "dodgeWindows && Config.bar.persistent && "
                           "!disabled",
        "confidence": "high",
    },
    {
        "id": "border-below-padding-floor",
        "trigger_path": "border.thickness",
        "when": {"op_below": 2.5,
                 "approximate": "the floor is Tokens.padding.small, a QML "
                                "singleton constant; ops at or below ~2 are "
                                "where the floor starts binding"},
        "effect_path": "bar padding (effective)",
        "effect": "the bar's padding floors at Tokens.padding.small — a "
                  "thinner border does not shrink the bar's padding below "
                  "that floor",
        "citation": "shell/modules/bar/BarWrapper.qml:25",
        "claimed_content": "readonly property int padding: Math.max(Tokens."
                           "padding.small, Config.border.thickness)",
        "confidence": "medium",
    },
)


# ---------------------------------------------------------------------------
# Projection.
# ---------------------------------------------------------------------------


def project(ops: Sequence[Dict[str, Any]],
            current: Optional[Dict[str, Any]] = None,
            max_derived: int = 24) -> Dict[str, Any]:
    """Project a candidate op list through the interaction graph.

    ``current``: the live file's values (path -> value) for evaluating
    cross-key conditions (the caller reads it read-only; None means
    cross-key edges that need the live state stay silent — honest).

    Returns (read-only):
    - ``derived``: effects the user did NOT ask for, each with its edge,
      citation, and confidence — bounded at ``max_derived``;
    - ``annotated_ops``: the ops with a ``downstream`` count;
    - ``conflicts``: the EXISTING AC-3 conflict check
      (optimize.check_conflicts) over the candidate list;
    - ``reversibility``: every planned op goes through the journaled
      applier (backup + undo history), stated plainly.
    """
    ops = [dict(op) for op in ops]
    derived: List[Dict[str, Any]] = []

    from assistant.adapters.caelestia.registry import tool_by_name

    def path_of(tool_name: str) -> Optional[str]:
        spec = tool_by_name(str(tool_name))
        return spec.path if spec else None

    def path_value(path: str) -> Any:
        """The value a path will have after the ops: the LAST op touching
        it wins, else the live value."""
        for op in reversed(ops):
            if path_of(str(op.get("tool", ""))) == path:
                return op.get("value")
        if isinstance(current, dict):
            node: Any = current
            for piece in path.split("."):
                if isinstance(node, dict):
                    node = node.get(piece)
                else:
                    return None
            return node
        return None

    def edge_fires(edge: Dict[str, Any], op: Dict[str, Any]) -> bool:
        trigger = edge.get("trigger_path")
        if path_of(str(op.get("tool", ""))) != trigger:
            return False
        when = edge.get("when") or {}
        value = op.get("value")
        if "op_is" in when and value is not when["op_is"]:
            return False
        if "op_below" in when and not (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and value < when["op_below"]):
            return False
        requires = when.get("requires")
        if requires:
            live = path_value(requires["path"])
            if live is None or live != requires["equals"]:
                return False
        return True

    for op in ops:
        if len(derived) >= max_derived:
            break  # the projection is a VIEW, never a wall of effects
        for edge in EDGES:
            if any(row["edge"] == edge["id"] for row in derived):
                continue
            if edge_fires(edge, op):
                derived.append({
                    "from_op": str(op.get("tool", "")),
                    "edge": edge["id"],
                    "effect_path": edge["effect_path"],
                    "effect": edge["effect"],
                    "trigger_when": _describe_when(edge),
                    "citation": edge["citation"],
                    "confidence": edge["confidence"],
                })

    annotated: List[Dict[str, Any]] = []
    for op in ops:
        tool = str(op.get("tool", ""))
        count = sum(1 for row in derived if row["from_op"] == tool)
        row = dict(op)
        row["downstream"] = count
        annotated.append(row)

    conflicts: List[Dict[str, Any]] = []
    # (1) The EXISTING AC-3 check (optimize.check_conflicts, already
    # shipped) over the candidate list: constraint propagation over the
    # registry domains — an out-of-domain request is refused with its
    # surviving domain, exactly as the planner would refuse it. Zero user
    # constraints still validates requested values against the legal
    # domains (the check's own semantics).
    try:
        from . import optimize

        optimize.check_conflicts(ops, ())
    except optimize.ConstraintError as exc:
        conflicts.append({"verdict": "UNSUPPORTED",
                          "reason": str(exc)})
    except Exception:
        pass  # the conflict check must never break the view
    # (2) The proposal's extension: the AC-3 view must also see INDUCED
    # values. An edge that fires sets effect_path := effect_value; if the
    # user ALSO requested a different value for that same path in this
    # very op list, the shell will flip their request off — that is a
    # conflict the flat plan never shows.
    for row in derived:
        edge = next((e for e in EDGES if e["id"] == row["edge"]), None)
        if not edge or "effect_value" not in edge:
            continue
        for op in ops:
            if path_of(str(op.get("tool", ""))) == edge["effect_path"] \
                    and op.get("value") != edge["effect_value"]:
                conflicts.append({
                    "verdict": "INDUCED-CONFLICT",
                    "reason": f"{op.get('tool')}={op.get('value')!r} is "
                              f"requested, but edge {edge['id']} induces "
                              f"{edge['effect_path']}="
                              f"{edge['effect_value']!r} — the shell's own "
                              f"handler wins; the requested value will not "
                              f"hold",
                    "cited": edge["citation"],
                })

    return {
        "n_ops": len(ops),
        "derived": derived,
        "annotated_ops": annotated,
        "conflicts": conflicts,
        "reversibility": {
            "all_journaled": True,
            "note": "every planned change applies through the applier's "
                    "backup + undo history (undo: caelestia-assist "
                    "settings --undo); the what-if view itself writes "
                    "nothing",
        },
        "note": "the consequence universe is exactly the cited edge table "
                f"({len(EDGES)} edges) — bounded, auditable, never a "
                "general model",
    }


def _describe_when(edge: Dict[str, Any]) -> str:
    when = edge.get("when") or {}
    parts = []
    if "op_is" in when:
        parts.append(f"set to {when['op_is']!r}")
    if "op_below" in when:
        parts.append(f"below {when['op_below']}")
    requires = when.get("requires")
    if requires:
        parts.append(f"while {requires['path']} is {requires['equals']!r}")
    return "; ".join(parts) or "always for this path"


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def render(projection: Dict[str, Any], title: str = "what-if") -> List[str]:
    """Text rendering of one projection (nodes, cited edges, conflicts,
    reversibility) — the pre-consent artifact the user actually reads."""
    lines = [f"{title} — projected consequences (read-only view, nothing "
             "is applied):",
             f"  candidate ops: {projection['n_ops']}"]
    derived = projection.get("derived") or []
    if derived:
        lines.append(f"  derived effects you did NOT ask for ({len(derived)}):")
        for row in derived:
            lines.append(f"    - {row['from_op']} ({row['trigger_when']}) "
                         f"-> {row['effect_path']}")
            lines.append(f"        {row['effect']}")
            lines.append(f"        cited: {row['citation']} "
                         f"[{row['confidence']}]")
    else:
        lines.append("  derived effects: none in the known interaction "
                     "table for these ops")
    conflicts = projection.get("conflicts") or []
    if conflicts:
        lines.append(f"  constraint conflicts ({len(conflicts)}):")
        for row in conflicts:
            lines.append(f"    - {row}")
    else:
        lines.append("  constraint check: no conflict in the candidate list")
    reversibility = projection.get("reversibility") or {}
    lines.append(f"  reversibility: {reversibility.get('note', 'n/a')}")
    lines.append(f"  {projection.get('note', '')}")
    return lines

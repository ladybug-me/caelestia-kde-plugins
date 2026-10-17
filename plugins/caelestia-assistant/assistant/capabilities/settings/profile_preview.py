"""assistant.capabilities.settings.profile_preview — composite what-if for profiles
(F11, exponential-build-5).

The single-request what-if (``--what-if``) projects ONE op list through
the interaction graph. A PROFILE is a composition of SOURCES — a preset,
a macro, a call — and the question a user asks before consenting is not
just "what will change?" but "what did each PART of this bundle bring
in, and where do they collide?". This module answers that by composing
engines that already exist and are already tested (no new analysis of
its own):

- ``profiles.compose_ops`` per source: each source's own op contribution;
- ``profiles.profile_ops``: the composed op list and the conflicts the
  source order resolved (F7: nothing dropped silently);
- ``consequences.project`` per source AND for the composite: cited
  derived effects and AC-3 conflicts, per part and overall;
- ``planner.plan``: the real validation (ranges, types, no-ops);
- ``sanity.project`` + ``sanity.check``: the WCAG/projected-state
  invariants the applier itself will enforce, run here read-only so the
  preview shows the refusals BEFORE the consent, not after;
- the registry's group metadata: the blast radius by feature area.

Read-only end to end: it reads the target file, plans, projects, and
renders — nothing is applied and nothing is written anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import consequences, planner, profiles, sanity
from assistant.adapters.caelestia.registry import tool_by_name

__all__ = ["build", "render"]


def _read_current(target: Path) -> Dict[str, Any]:
    """The live file's parsed values (read-only; {} on any absence or
    parse error — the projection then reports honestly that cross-key
    live-state edges were evaluated against an empty state)."""
    try:
        return json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _group_of(tool: str) -> str:
    spec = tool_by_name(tool)
    return getattr(spec, "group", "?") if spec else "?"


def build(target, name: str) -> Dict[str, Any]:
    """Assemble the composite preview for one saved profile. Raises
    ProfileError for an unknown profile; PlannerError passes through
    (the CLI renders it — an out-of-range saved call is exactly the
    kind of finding the preview exists to surface)."""
    target = Path(target)
    ops, conflicts = profiles.profile_ops(target, name)

    sources = next((p.get("sources", []) for p in profiles.list_profiles(target)
                    if p.get("name") == name), [])
    current = _read_current(target)

    per_source: List[Dict[str, Any]] = []
    for src in sources:
        kind = src.get("kind")
        label = (f"{kind}:{src.get('name')}" if kind != "call"
                 else f"call {src.get('tool')}={src.get('value')!r}")
        try:
            src_ops, src_conflicts = profiles.compose_ops(target, [src])
        except profiles.ProfileError as exc:
            per_source.append({"label": label, "error": str(exc),
                               "n_ops": 0, "derived": 0, "conflicts": 0})
            continue
        projection = consequences.project(src_ops, current)
        per_source.append({
            "label": label,
            "n_ops": len(src_ops),
            "derived": projection.get("derived") or [],
            "conflicts": src_conflicts,
        })

    composite = consequences.project(ops, current)
    plan = planner.plan(ops, target)

    projected = sanity.project(current, plan.get("entries", []))
    sanity_report = sanity.check(projected)

    blast: Dict[str, List[str]] = {}
    for op in ops:
        blast.setdefault(_group_of(str(op["tool"])), []).append(str(op["tool"]))

    return {
        "name": name,
        "n_ops": len(ops),
        "ops": ops,
        "conflicts": conflicts,
        "per_source": per_source,
        "composite": composite,
        "plan_blocked": bool(plan.get("apply_blocked")),
        "plan_refusals": [e for e in plan.get("entries", []) if e.get("error")],
        "sanity": {"refusals": sanity_report.get("refusals", []),
                   "warnings": sanity_report.get("warnings", [])},
        "blast_radius": {g: sorted(t) for g, t in sorted(blast.items())},
    }


def render(view: Dict[str, Any]) -> List[str]:
    """Text rendering of the composite preview (the pre-consent artifact)."""
    name = view["name"]
    lines = [f"profile {name!r} — composite what-if (read-only, nothing "
             f"is applied):",
             f"  composed ops: {view['n_ops']} across "
             f"{len(view['blast_radius'])} feature area(s)"]
    for group, tools in view["blast_radius"].items():
        lines.append(f"    {group}: {', '.join(tools)}")

    if view["conflicts"]:
        lines.append(f"  composition conflicts resolved by source order "
                     f"({len(view['conflicts'])}):")
        for c in view["conflicts"]:
            lines.append(f"    - {c['tool']}: {c['dropped']['value']!r} "
                         f"({c['dropped']['from']}) overridden by "
                         f"{c['kept']['value']!r} ({c['kept']['from']})")
    else:
        lines.append("  composition conflicts: none")

    lines.append("  per source:")
    for src in view["per_source"]:
        if src.get("error"):
            lines.append(f"    {src['label']}: UNRESOLVED NOW — {src['error']}")
            continue
        derived = src.get("derived") or []
        conflicts = src.get("conflicts") or []
        lines.append(f"    {src['label']}: {src['n_ops']} op(s), "
                     f"{len(derived)} derived effect(s), "
                     f"{len(conflicts)} internal conflict(s)")
        for row in derived:
            lines.append(f"        derived: {row.get('from_op')} -> "
                         f"{row.get('effect_path')} "
                         f"(cited: {row.get('citation')})")

    composite = view["composite"]
    comp_derived = composite.get("derived") or []
    if comp_derived:
        lines.append(f"  composite derived effects ({len(comp_derived)}):")
        for row in comp_derived:
            lines.append(f"    - {row.get('from_op')} -> "
                         f"{row.get('effect_path')}: {row.get('effect')}")
            lines.append(f"        cited: {row.get('citation')} "
                         f"[{row.get('confidence')}]")
    else:
        lines.append("  composite derived effects: none in the known "
                     "interaction table")

    plan_refusals = view.get("plan_refusals") or []
    if view.get("plan_blocked") or plan_refusals:
        lines.append(f"  PLANNER BLOCKS THIS PROFILE "
                     f"({len(plan_refusals)} rejected op(s)) — the apply "
                     f"would refuse:")
        for e in plan_refusals:
            lines.append(f"    - {e.get('tool')}: {e.get('error')}")
    sanity_ref = view["sanity"]["refusals"]
    sanity_warn = view["sanity"]["warnings"]
    if sanity_ref:
        lines.append(f"  projected-state refusals ({len(sanity_ref)}) — the "
                     f"applier would refuse the write:")
        for row in sanity_ref:
            lines.append(f"    - {row.get('id')}: {row.get('message')}")
    if sanity_warn:
        lines.append(f"  projected-state warnings ({len(sanity_warn)}):")
        for row in sanity_warn:
            lines.append(f"    - {row.get('id')}: {row.get('message')}")
    lines.append("  reversibility: every applied profile goes through the "
                 "journaled applier (backup + bounded undo history)")
    lines.append("  SUGGESTED_NOT_EXECUTED: caelestia-assist settings "
                 "--profile " + name + " --apply --confirm")
    return lines

"""settings.optimize — optimization profiles, Pareto presets, conflict solving.

Issue #120 Phase 3 ("Context-aware recommendations / Desktop optimization
profiles") as a deterministic, model-free module. Three capabilities:

1. PROFILES — each objective profile (gaming, battery, minimal, comfort,
   accessibility) is a scored preference direction over the registry groups.
   `score_plan(profile, ops)` scores a set of tool calls for how well it
   serves the objective; `recommend(profile, k)` returns the registry's
   highest-leverage default-deviations for that objective, with reasons.

2. PARETO — `pareto_profiles(ops_sets)` computes the non-dominated frontier
   across multiple profiles at once (genius.optimize.pareto_frontier), so
   "gaming vs battery" is answered with the actual trade-off set, not a
   single patronising "best".

3. SYNTHESIS — `synthesize(profile, constraints)` finds the setting vector
   closest to the profile's ideal point while satisfying hard constraints,
   via coordinate descent over the continuous axes and a bounded enumeration
   over enum/bool axes. `check_conflicts` runs AC-3 arc-consistency over
   declared constraint pairs and reports UNSAT early with the conflicting
   domains — "compact AND bar scale 1.6" is refused with reasons, never
   half-applied.

Every produced op flows through the existing planner/applier gates exactly
like any other settings request: this module PROPOSES, it never writes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from assistant.adapters.caelestia.registry import ToolSpec, tool_by_name
from assistant.capabilities.genius.optimize import pareto_frontier

__all__ = ["PROFILES", "score_plan", "recommend", "pareto_profiles",
           "synthesize", "check_conflicts", "ConstraintError",
           "preference_drift"]

# ---------------------------------------------------------------------------
# 1. Objective profiles
# ---------------------------------------------------------------------------
# Each profile maps registry GROUPS to a preference direction and weight.
# Groups not listed are neutral. `ideal` names per-tool ideal deviations from
# default for the top tools of that objective (1.0 == "push to the useful
# extreme", validated against the registry range at synthesis time).

PROFILES: Dict[str, Dict[str, Any]] = {
    "gaming": {
        "description": "fewest composited effects, fastest shell, silent osd",
        "groups": {
            "effects": ("decrease", 1.0),
            "animations": ("decrease", 0.9),
            "notifications": ("decrease", 0.7),
            "osd": ("decrease", 0.6),
            "blur": ("decrease", 0.8),
        },
        "target": {
            "setAnimationSpeed": 0.4,
            "setBlurEnabled": False,
            "setTransparencyEnabled": False,
            "setGameModeDisableHyprlandAnimations": True,
            "setGameModeDisableHyprlandBlur": True,
            "setLivePreviews": False,
            "setRoundingScale": 0.6,
        },
    },
    "battery": {
        "description": "less compositing and polling, dimmer luminance effects",
        "groups": {
            "effects": ("decrease", 0.9),
            "blur": ("decrease", 0.9),
            "animations": ("decrease", 0.6),
            "wallpaper-scheme": ("decrease", 0.3),
        },
        "target": {
            "setBlurEnabled": False,
            "setTransparencyEnabled": False,
            "setDisableWallpaperBlur": True,
            "setLivePreviews": False,
            "setAnimationSpeed": 0.6,
        },
    },
    "minimal": {
        "description": "smaller surfaces, tighter spacing, fewer decorations",
        "groups": {
            "bar": ("decrease", 0.7),
            "dock": ("decrease", 0.8),
            "appearance": ("decrease", 0.6),
            "animations": ("decrease", 0.4),
        },
        "target": {
            "setBarScale": 0.7,
            "setDockIconSize": 20,
            "setRoundingScale": 0.5,
            "setSpacingScale": 0.7,
            "setDockBadges": False,
            "setLivePreviews": False,
        },
    },
    "comfort": {
        "description": "larger surfaces, softer motion, easier reading",
        "groups": {
            "bar": ("increase", 0.6),
            "dock": ("increase", 0.6),
            "appearance": ("increase", 0.5),
            "animations": ("increase", 0.3),
        },
        "target": {
            "setBarScale": 1.2,
            "setDockIconSize": 40,
            "setRoundingScale": 1.2,
            "setSpacingScale": 1.15,
        },
    },
    "accessibility": {
        "description": "maximum legibility: large targets, slow motion, clear osd",
        "groups": {
            "bar": ("increase", 0.8),
            "dock": ("increase", 0.8),
            "animations": ("decrease", 0.5),
            "osd": ("increase", 0.4),
            "notifications": ("increase", 0.5),
        },
        "target": {
            "setBarScale": 1.4,
            "setDockIconSize": 48,
            "setAnimationSpeed": 1.4,
            "setSpacingScale": 1.3,
        },
    },
}


def _clamped_ideal(tool: ToolSpec, ideal: Any) -> Any:
    """Ideal deviation validated against the registry — out-of-range ideals
    are pushed to the nearest legal extreme, never clamped silently into an
    invalid op (the planner would reject them anyway)."""
    if tool.kind == "bool":
        return bool(ideal)
    if tool.kind == "enum":
        return ideal if tool.enum and ideal in tool.enum else (
            tool.enum[0] if tool.enum else None)
    if tool.kind in ("int", "float") and isinstance(ideal, (int, float)):
        lo = tool.minimum if tool.minimum is not None else ideal
        hi = tool.maximum if tool.maximum is not None else ideal
        v = min(hi, max(lo, float(ideal)))
        return int(round(v)) if tool.kind == "int" else v
    return None


def score_plan(profile: str, ops: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score a plan (list of {"tool": name, "value": v}) against a profile.

    The score is the weight-averaged agreement between each op's direction
    and the profile's group preferences, normalised to [0, 1]. Unknown tools
    count as neutral; ops the profile disagrees with subtract."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    spec = PROFILES[profile]
    groups = spec["groups"]
    total_w, score = 0.0, 0.0
    per_op: List[Dict[str, Any]] = []
    for op in ops:
        tool = tool_by_name(str(op.get("tool", "")))
        if tool is None:
            per_op.append({"tool": op.get("tool"), "agreement": None,
                           "reason": "unknown tool"})
            continue
        direction = _op_direction(tool, op.get("value"))
        entry = groups.get(tool.group)
        if entry is None:
            agreement, weight = 0.5, 0.2
        else:
            want, weight = entry
            agreement = 1.0 if direction == want else (
                0.5 if direction == "toggle" else 0.0)
        score += agreement * weight
        total_w += weight
        per_op.append({"tool": tool.name, "group": tool.group,
                       "direction": direction,
                       "agreement": round(agreement, 2)})
    norm = round(score / total_w, 3) if total_w else 0.0
    return {
        "profile": profile,
        "score": norm,
        "verdict": ("serves this objective" if norm >= 0.7 else
                    "mixed for this objective" if norm >= 0.45 else
                    "works against this objective"),
        "per_op": per_op,
    }


def _op_direction(tool: ToolSpec, value: Any) -> str:
    if tool.kind == "bool" or tool.kind == "enum":
        return "toggle"
    default = tool.default
    if isinstance(value, (int, float)) and isinstance(default, (int, float)):
        if value > default:
            return "increase"
        if value < default:
            return "decrease"
    return "set"


def recommend(profile: str, k: int = 6) -> Dict[str, Any]:
    """The profile's target ops, validated against the registry, strongest
    first. These are PROPOSALS: the planner still range/type-checks them and
    the applier still gates them behind explicit confirmation."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    target = PROFILES[profile]["target"]
    ops: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name, ideal in target.items():
        tool = tool_by_name(name)
        if tool is None:
            skipped.append(f"{name}: not in this registry build")
            continue
        v = _clamped_ideal(tool, ideal)
        if v is None:
            skipped.append(f"{name}: ideal value not representable")
            continue
        ops.append({"tool": tool.name, "value": v,
                    "path": tool.path,
                    "reason": f"{PROFILES[profile]['description']} — "
                              f"{tool.name} -> {v!r}"})
    return {"profile": profile,
            "description": PROFILES[profile]["description"],
            "ops": ops[:k], "skipped": skipped}


# ---------------------------------------------------------------------------
# 2. Pareto across profiles
# ---------------------------------------------------------------------------

def pareto_profiles(ops_sets: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Given candidate op-sets (e.g. the recommend() output per profile),
    score each against EVERY profile and return the non-dominated frontier
    plus the per-profile score matrix. Axes = profiles, all maximised."""
    axes = sorted(ops_sets)
    points: List[Dict[str, float]] = []
    meta: List[Dict[str, Any]] = []
    for label, ops in ops_sets.items():
        scores = {a: score_plan(a, ops)["score"] for a in axes}
        # axis values must be sortable numbers; flip: use score directly,
        # dominance direction handled below (max on every axis)
        point = {a: scores[a] for a in axes}
        points.append(point)
        meta.append({"label": label, "ops": list(ops), "scores": scores})
    # pareto_frontier is min-oriented per direction; pass directions='max'
    front = pareto_frontier(points, axes, ["max"] * len(axes))
    front_ids = {id(p) for p in front["front"]}
    frontier = [m for m, p in zip(meta, points) if id(p) in front_ids]
    dominated = [m for m, p in zip(meta, points) if id(p) not in front_ids]
    return {"axes": axes, "frontier": frontier, "dominated": dominated,
            "score_matrix": [m["scores"] for m in meta],
            "labels": [m["label"] for m in meta]}


# ---------------------------------------------------------------------------
# 3. Synthesis + constraint solving
# ---------------------------------------------------------------------------

class ConstraintError(ValueError):
    """Raised when constraints are unsatisfiable; `.domains` carries the
    conflicting variables' reduced domains for the error message."""


def _axis(tool: ToolSpec) -> Tuple[str, List[Any]]:
    """A synthesis axis: every value the tool may legally take.
    int/float axes are stepped by tool.step (bounded enumeration, same
    semantics as the Nexus steppers); bool/enum axes are their domains."""
    if tool.kind == "bool":
        return tool.name, [False, True]
    if tool.kind == "enum":
        return tool.name, list(tool.enum or [])
    lo = float(tool.minimum if tool.minimum is not None else tool.default)
    hi = float(tool.maximum if tool.maximum is not None else tool.default)
    step = float(tool.step) if tool.step else (hi - lo) / 8.0 or 1.0
    values: List[float] = []
    v = lo
    guard = 0
    while v < hi - 1e-9 and guard < 200:
        values.append(v)
        v += step
        guard += 1
    values.append(hi)
    if tool.kind == "int":
        return tool.name, sorted({int(round(v)) for v in values})
    return tool.name, sorted({round(v, 6) for v in values})


def check_conflicts(ops: Sequence[Dict[str, Any]],
                    constraints: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    """AC-3 arc consistency over the requested variables.

    constraints: (tool_a, relation, tool_b) where relation is one of
      "!="  values must differ
      "<"   a's value < b's value
      ">"   a's value > b's value
      "with"  if a is set then b must be set too (co-presence)

    Domains START at each tool's full legal registry range (the same values
    the Nexus steppers can reach), AC-3 narrows them, and the REQUESTED
    values are then checked against the narrowed domains: a request that no
    longer has support is the conflict, reported with the surviving domain —
    never clamped, never half-applied.
    """
    domains: Dict[str, List[Any]] = {}
    order: List[str] = []
    for op in ops:
        tool = tool_by_name(str(op.get("tool", "")))
        if tool is None:
            continue
        name, dom = _axis(tool)
        if name not in domains:
            domains[name] = dom
            order.append(name)
    requested: Dict[str, Any] = {
        str(op.get("tool")): op.get("value") for op in ops
        if tool_by_name(str(op.get("tool", ""))) is not None}

    def _supported(rel: str, va: Any, db: List[Any]) -> bool:
        if rel == "!=":
            return any(va != vb for vb in db)
        if rel == "<":
            return any(float(va) < float(vb) for vb in db)
        if rel == ">":
            return any(float(va) > float(vb) for vb in db)
        return True  # "with": co-presence satisfied by b existing

    changed = True
    while changed:
        changed = False
        for a, rel, b in constraints:
            if a not in domains or b not in domains:
                continue
            keep_a = [va for va in domains[a] if _supported(rel, va, domains[b])]
            if not keep_a:
                raise ConstraintError(
                    f"unsatisfiable arc {a} {rel} {b}: domain({a})="
                    f"{domains[a]} has no support in domain({b})={domains[b]}")
            if len(keep_a) != len(domains[a]):
                domains[a] = keep_a
                changed = True
            # reverse arc for symmetry
            rev = {"<": ">", ">": "<", "!=": "!=", "with": "with"}[rel]
            keep_b = [vb for vb in domains[b] if _supported(rev, vb, domains[a])]
            if not keep_b:
                raise ConstraintError(
                    f"unsatisfiable arc {b} {rev} {a}: domain({b})="
                    f"{domains[b]} has no support in domain({a})={domains[a]}")
            if len(keep_b) != len(domains[b]):
                domains[b] = keep_b
                changed = True

    pruned = []
    for name in order:
        want = requested.get(name)
        if want is None:
            continue
        if tool_by_name(name).kind in ("int", "float"):
            survives = any(abs(float(v) - float(want)) <= 1e-9
                           for v in domains[name])
        else:
            survives = want in domains[name]
        if not survives:
            pruned.append({"tool": name, "requested": want,
                           "surviving_domain": domains[name]})
    if pruned:
        first = pruned[0]
        raise ConstraintError(
            f"conflict: {first['tool']}={first['requested']!r} has no support "
            f"after constraint propagation; surviving domain "
            f"{first['surviving_domain']}")
    return {"consistent": True, "domains": {k: domains[k] for k in order},
            "requested": requested}


def synthesize(profile: str,
               constraints: Sequence[Tuple[str, str, str]] = (),
               pinned: Optional[Dict[str, Any]] = None,
               max_ops: int = 8) -> Dict[str, Any]:
    """Closest legal plan to the profile's ideal point under hard constraints.

    Method: bounded coordinate descent. For each axis (registry tool with a
    finite legal domain) pick the value minimising |value - ideal| subject
    to arc-consistency with every other axis's current choice (a small
    joint check over the constraint arcs, retried a few passes because the
    axes interact through the constraints). Deterministic; no RNG."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
    target = dict(PROFILES[profile]["target"])
    if pinned:
        target.update(pinned)

    axes: Dict[str, Tuple[ToolSpec, List[Any], Any]] = {}
    for name, ideal in target.items():
        tool = tool_by_name(name)
        if tool is None:
            continue
        _n, dom = _axis(tool)
        ideal_v = _clamped_ideal(tool, ideal)
        axes[name] = (tool, dom, ideal_v)

    assignment: Dict[str, Any] = {}
    # pass 1: per-axis nearest-to-ideal
    for name, (_tool, dom, ideal_v) in axes.items():
        if ideal_v is None:
            assignment[name] = dom[len(dom) // 2]
        else:
            assignment[name] = min(dom, key=lambda v: abs(float(v) - float(ideal_v)))

    def _arc_ok(rel: str, va: Any, vb: Any) -> bool:
        return {"!=": lambda: va != vb,
                "<": lambda: float(va) < float(vb),
                ">": lambda: float(va) > float(vb),
                "with": lambda: True}[rel]()

    def _satisfying_candidates(rel: str, mover: str, vo: Any,
                               a: str, b: str) -> List[Any]:
        """Values for `mover` that satisfy the arc given the OTHER side is
        already fixed at vo — constraint-directed, not blind."""
        dom = axes[mover][1]
        if rel == "!=":
            return [v for v in dom if v != vo]
        if rel == "<":     # a < b
            return ([v for v in dom if float(v) > float(vo)] if mover == b
                    else [v for v in dom if float(v) < float(vo)])
        if rel == ">":     # a > b
            return ([v for v in dom if float(v) < float(vo)] if mover == b
                    else [v for v in dom if float(v) > float(vo)])
        return []

    # pass 2: repair constraint violations, moving the side that can actually
    # fix the arc (b first, then a), always toward its own ideal
    for _pass in range(6):
        violated = next(((a, rel, b) for a, rel, b in constraints
                         if a in assignment and b in assignment
                         and not _arc_ok(rel, assignment[a], assignment[b])), None)
        if violated is None:
            break
        a, rel, b = violated
        repaired = False
        for mover in (b, a):
            other = a if mover == b else b
            cands = _satisfying_candidates(rel, mover, assignment[other], a, b)
            if not cands:
                continue
            _tool, _dom, ideal_v = axes[mover]
            if ideal_v is None:
                assignment[mover] = cands[len(cands) // 2]
            else:
                assignment[mover] = min(
                    cands, key=lambda v: abs(float(v) - float(ideal_v)))
            if _arc_ok(rel, assignment[a], assignment[b]):
                repaired = True
                break
        if not repaired:
            raise ConstraintError(
                f"cannot satisfy {a} {rel} {b} with "
                f"domains {axes[a][1]} / {axes[b][1]}")
    ops = []
    for name in sorted(assignment):
        tool, _dom, ideal_v = axes[name]
        v = assignment[name]
        ops.append({"tool": name, "value": v, "path": tool.path,
                    "ideal": ideal_v,
                    "delta_from_ideal": (abs(float(v) - float(ideal_v))
                                         if isinstance(ideal_v, (int, float))
                                         and isinstance(v, (int, float)) else 0)})
    ops.sort(key=lambda o: -o["delta_from_ideal"])
    ops = ops[:max_ops]
    return {"profile": profile, "ops": ops,
            "constraints_checked": len(constraints),
            "note": "proposal only — planner validation and the apply gate still apply"}




# ---------------------------------------------------------------------------
# 5. Config drift vs the preference posterior (B2, issue #120 adjacent)
# ---------------------------------------------------------------------------

def preference_drift(config, posterior, specs=None, min_evidence=3,
                     evidence_cap=10, reject_below=0.4):
    """Distance between a LIVE config and a Beta-Binomial preference
    posterior (B2). PURE: plain data in, plain data out — no file reads,
    no writes; the caller (brain side) owns the live file and the model.

    Semantics — "drift" means the config sits in a direction this user's
    own decision history says they REJECT:

    - every ranged numeric tool (float/int with a range) whose live value
      differs from the registry default contributes its normalized
      offset direction (increase/decrease) and magnitude |pos - default|;
    - the posterior row for (group, direction) with at least
      ``min_evidence`` decisions votes: a low acceptance mean (below
      ``reject_below``) is REJECTION of that direction;
    - a tool drifts when its live direction is rejected, weighted by
      offset magnitude x rejection depth x evidence weight
      (min(n, cap)/cap — one decision is one decision, capped at 10);
    - tools with thin evidence (< ``min_evidence`` decisions) are
      reported in ``thin`` and never scored (the honest abstain the
      preference model itself applies).

    Returns {"score": 0..1, "evaluated": k, "flags": [...], "thin": [...]}.
    """
    from assistant.adapters.caelestia.registry import TOOL_SPECS as _SPECS
    if specs is None:
        specs = _SPECS
    flags = []
    thin = []
    total = 0.0
    evaluated = 0
    for spec in specs:
        if spec.kind not in ("float", "int"):
            continue
        if spec.minimum is None or spec.maximum is None:
            continue
        # Resolve the live value along the dotted path (read-only walk).
        node = config
        missing = False
        for segment in spec.path.split("."):
            if isinstance(node, dict) and segment in node:
                node = node[segment]
            else:
                missing = True
                break
        if missing or not isinstance(node, (int, float)) or isinstance(node, bool):
            continue
        span = float(spec.maximum) - float(spec.minimum)
        if span <= 0:
            continue
        default = spec.default
        if not isinstance(default, (int, float)) or isinstance(default, bool):
            continue
        pos = (float(node) - float(spec.minimum)) / span
        dpos = (float(default) - float(spec.minimum)) / span
        offset = pos - dpos
        if abs(offset) < 1e-9:
            continue  # at default: no directional claim to disagree with
        direction = "increase" if offset > 0 else "decrease"
        magnitude = min(1.0, abs(offset))
        row = posterior.get((spec.group or spec.path.split(".")[0], direction))
        evaluated += 1
        if row is None or row.get("n", 0) < min_evidence:
            thin.append({"tool": spec.name, "path": spec.path,
                         "group": spec.group or spec.path.split(".")[0],
                         "direction": direction,
                         "n": 0 if row is None else row.get("n", 0)})
            continue
        mean = float(row.get("p_accept", 0.5))
        if mean < reject_below:
            depth = (reject_below - mean) / reject_below
            weight = min(row.get("n", 0), evidence_cap) / float(evidence_cap)
            total += magnitude * depth * weight
            flags.append({
                "tool": spec.name, "path": spec.path,
                "group": spec.group or spec.path.split(".")[0],
                "direction": direction,
                "live": node, "default": default,
                "p_accept": round(mean, 3), "n": row.get("n", 0),
                "weight": round(magnitude * depth * weight, 4),
            })
    flags.sort(key=lambda f: -f["weight"])
    score = round(min(1.0, total / max(1, evaluated)), 4)
    return {"score": score, "evaluated": evaluated,
            "flags": flags, "thin": thin}

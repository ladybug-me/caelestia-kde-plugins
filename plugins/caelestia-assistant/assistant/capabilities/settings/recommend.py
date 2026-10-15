"""assistant.capabilities.settings.recommend — the value recommender (F29).

For a NUMERIC tool, propose a value from three evidence tiers with
hierarchical partial pooling:

  1. the registry default (the upstream-shipped center),
  2. the preset pool (what each optimization profile sets this tool to),
  3. the user's own APPROVED history (values that were actually applied
     through the consented applier — never rejected or dry-run values).

Model (documented, deliberately simple, stdlib-only): conjugate Gaussian.
The population prior is N(mu0, tau^2) over the pooled evidence
(default + preset values); the user's n approved values are the second
level; the posterior is the precision-weighted blend — the classic
partial-pooling shrinkage: with no history you get the prior, with a
long history you get yourself, in between you get shrunk toward the
pool. The credible interval is the posterior's own 95% band, CLAMPED to
the tool's legal range with the clamp reported (a recommendation
outside the registry's range would be a bug, not a opinion).

Honesty rules: this is a PROPOSAL (inert text, never applied — the
standard dry-run/consent spine owns every write); rejected plans are
learned negatives elsewhere and are NOT part of this estimate (stated);
with zero user history the output says so instead of pretending the
prior is a preference; and the Pareto position against the optimization
profiles is a plain comparison of numbers, not a claim about your
machine's performance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from assistant.adapters.caelestia.registry import ToolSpec, tool_by_name, tool_by_path

__all__ = ["recommend", "render_recommendation"]

_Z_95 = 1.959963985  # two-sided 95% Gaussian quantile (documented approx:
                     # with tiny n a Student-t would be wider; the note
                     # says so instead of faking rigor)


def _preset_pool(spec: ToolSpec) -> List[Tuple[str, float]]:
    """(preset_name, value) for every profile that sets this tool."""
    from pathlib import Path as _P
    tools_json = _P(__file__).resolve().parents[2] / "adapters" / "caelestia" / "tools.json"
    with open(tools_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: List[Tuple[str, float]] = []
    for preset in data.get("presets", ()):
        for call in preset.get("calls", ()):
            if call.get("tool") == spec.name:
                v = call.get("value")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((preset["name"], float(v)))
    return out


def _approved_values(path: str,
                     ledger_path: Optional[str]) -> List[float]:
    """Values the user APPROVED for this path (read-only ledger scan)."""
    if not ledger_path:
        return []
    p = Path(ledger_path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    out: List[float] = []
    for entry in data.get("entries", ()):
        for op in entry.get("ops", ()):
            if op.get("path") == path:
                v = op.get("new")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(float(v))
    return out


def recommend(tool: str,
              ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """One numeric tool's recommended value, pooled and honest."""
    spec = tool_by_name(str(tool)) or tool_by_path(str(tool))
    if spec is None:
        return {"verdict": "UNKNOWN_TOOL", "tool": str(tool)}
    if spec.kind not in ("int", "float"):
        return {"verdict": "UNSUPPORTED_KIND", "tool": spec.name,
                "kind": spec.kind,
                "note": "the recommender is for numeric tools; booleans "
                        "and enums have no meaningful pooling"}

    pool = _preset_pool(spec)
    default = spec.default
    prior_points = [float(default)] + [v for _n, v in pool]
    mu0 = sum(prior_points) / len(prior_points)
    # prior spread: the pooled dispersion, floored so a single-point
    # pool (only the default) does not collapse to a delta
    var_prior = (sum((v - mu0) ** 2 for v in prior_points)
                 / len(prior_points)) if len(prior_points) > 1 else 0.0
    var_prior = max(var_prior, 0.25 * ((spec.maximum or 1) -
                                       (spec.minimum or 0)) ** 2
                    / 16.0)  # quarter of a quarter-range: a weak, honest
                             # prior strength, documented not derived
    k0 = float(len(prior_points))  # pseudo-observation count

    history = _approved_values(spec.path, ledger_path)
    n = len(history)
    if n:
        xbar = sum(history) / n
        # the precision floor: preferences cannot be resolved finer
        # than the tool's own step (a zero-spread history is "n approvals
        # of the same value", not "infinite certainty about it")
        span = ((spec.maximum or 1) - (spec.minimum or 0))
        if spec.step:
            var_floor = float(spec.step) ** 2
        else:
            var_floor = max(span / 8.0, 1e-6) ** 2
        var_data = (sum((v - xbar) ** 2 for v in history) / n) if n > 1 \
            else 0.0
        var_data = max(var_data, var_floor)
        prec0, precd = k0 / var_prior, n / var_data
        post_var = 1.0 / (prec0 + precd)
        post_mean = (prec0 * mu0 + precd * xbar) / (prec0 + precd)
    else:
        post_mean, post_var = mu0, var_prior / max(k0, 1.0)

    lo, hi = spec.minimum, spec.maximum
    clamped = False
    value = post_mean
    if lo is not None and value < lo:
        value, clamped = float(lo), True
    if hi is not None and value > hi:
        value, clamped = float(hi), True
    if spec.kind == "int":
        rounded = int(round(value))
        if abs(rounded - value) > 1e-9:
            clamped = True
        value = rounded

    half = _Z_95 * math.sqrt(post_var)
    ci_lo = max(float(lo), value - half) if lo is not None else value - half
    ci_hi = min(float(hi), value + half) if hi is not None else value + half

    pareto = []
    for name, v in pool:
        if value == v:
            pos = "matches"
        elif value > v:
            pos = "above"
        else:
            pos = "below"
        pareto.append({"profile": name, "value": v, "position": pos})

    return {
        "verdict": "OK",
        "tool": spec.name,
        "path": spec.path,
        "kind": spec.kind,
        "range": [lo, hi],
        "value": value,
        "credible_interval_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "clamped": clamped,
        "evidence": {
            "registry_default": default,
            "preset_pool": [{"profile": n_, "value": v}
                            for n_, v in pool],
            "approved_history": history,
            "n_history": n,
        },
        "pareto_position": pareto,
        "note": ("partial pooling: no approved history, so this is the "
                 "pooled prior — a starting point, not your preference"
                 if not n else
                 f"partial pooling: {n} approved value(s) shrunk toward "
                 "the preset/registry pool by their relative precision")
        + ("; interval clamped to the legal range" if clamped else "")
        + ("; rejections (learned negatives) are NOT part of this "
           "estimate; proposal only — nothing is applied"),
        "proposal_only": True,
    }


def render_recommendation(result: Dict[str, Any]) -> List[str]:
    if result.get("verdict") == "UNKNOWN_TOOL":
        return [f"recommend: unknown tool {result['tool']!r}"]
    if result.get("verdict") == "UNSUPPORTED_KIND":
        return [f"recommend: {result['tool']} is {result['kind']} — "
                "the recommender is for numeric tools"]
    ev = result["evidence"]
    lines = [
        f"recommend: {result['tool']} ({result['path']}) -> "
        f"{result['value']}  [95% interval "
        f"{result['credible_interval_95']}]",
        f"  evidence: registry default {ev['registry_default']}; "
        f"preset pool "
        + (", ".join(f"{p['profile']}={p['value']}"
                     for p in ev["preset_pool"]) or "(no profile sets it)")
        + f"; approved history {ev['n_history']} value(s)",
    ]
    if result["pareto_position"]:
        lines.append("  vs profiles: " + ", ".join(
            f"{p['profile']} ({p['position']} {p['value']})"
            for p in result["pareto_position"]))
    lines.append(f"  {result['note']}")
    lines.append("  proposal only — apply through the standard "
                 "dry-run -> confirm -> journaled apply spine")
    return lines

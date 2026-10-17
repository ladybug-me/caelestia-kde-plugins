"""brain.ope — the off-policy evaluation GATE for capability switches.

The operating rule (exponential-build 3.2): before an agent — this one
or any future one — proposes flipping a capability kill-switch, it must
first replay the historical approve/reject log through the candidate
policy and PRINT the estimated accept rate. This module is that gate:
a library function plus a CLI report. It produces NO autonomy change
by itself — it never writes the capabilities manifest, never flips a
switch, never proposes; it only produces the evidence a human would
need to grant one (the switch stays a file edit, per
capabilities.py's own posture).

The estimator: importance-weighted (IPS, Horvitz & Thompson 1952's
weighting; the off-policy evaluation framing per Dudik, Langford &
Li 2011, "Doubly Robust Policy Evaluation and Learning"). The logged
episodes exist because the BEHAVIOR policy proposed them; a
deterministic CANDIDATE policy either would also have proposed an
episode (weight 1 — its outcome counts) or not (weight 0 — the episode
is outside its support). Where the candidate proposes OUTSIDE the
logged support there is no counterfactual: that gap is REPORTED as
1 - coverage, never silently extrapolated. Support below
``min_support`` is a thin-evidence refusal, not a shrug.

Known kill-switch candidate policies are registered here as explicit
filters over the ledger episode's (kind, target, diff JSON) — no
guessed semantics, and an unknown switch name lists what exists.
Pure functions: the ledger is read, nothing is written anywhere.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Sequence

__all__ = ["evaluate_policy", "switch_candidate", "KNOWN_SWITCHES",
           "ope_report"]

MIN_SUPPORT = 5

# The registered candidate policies: what WOULD have been proposed had
# the switch been ON, expressed as an exact filter over the ledger
# episode's own fields (kind, target substring, diff-JSON substring —
# all case-insensitive where noted). Deliberately explicit: adding a
# switch policy is a reviewable diff to this table.
KNOWN_SWITCHES: Dict[str, Dict[str, str]] = {
    "dbus_surface": {"kind": "settings", "diff_contains": "dbus"},
    "package_audit": {"kind": "agent", "target_contains": "pkgprobe"},
    "notification_observation": {"kind": "settings",
                                 "diff_contains": "notification"},
}


def switch_candidate(spec: Dict[str, str]) -> Callable[[Dict[str, Any]], bool]:
    """Compile one filter spec into a candidate proposal-policy."""
    kind = spec.get("kind")
    target_sub = (spec.get("target_contains") or "").lower()
    diff_sub = (spec.get("diff_contains") or "").lower()

    def candidate(context: Dict[str, Any]) -> bool:
        if kind is not None and context.get("kind") != kind:
            return False
        if target_sub and target_sub not in \
                str(context.get("target", "")).lower():
            return False
        if diff_sub:
            try:
                blob = json.dumps(context.get("diff") or {},
                                  sort_keys=True).lower()
            except (TypeError, ValueError):
                blob = ""
            if diff_sub not in blob:
                return False
        return True
    return candidate


def evaluate_policy(episodes: Sequence[Dict[str, Any]],
                    candidate_fn: Callable[[Dict[str, Any]], bool],
                    min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
    """Replay the historical log through the candidate policy.

    ``episodes``: [{"context": {...}, "approved": bool}, ...] — one row
    per logged proposal the behavior policy actually made. Returns the
    importance-weighted estimated accept rate, the support and coverage
    accounting, and the standing note: EVIDENCE ONLY — the human owns
    the switch. Refusals (empty log, zero support, thin support) are
    explicit ``refused`` reasons, never a shrug of a number."""
    episodes = [e for e in episodes if isinstance(e, dict)]
    n = len(episodes)
    if n == 0:
        return {"refused": "empty log — nothing to replay",
                "estimated_accept_rate": None, "support": 0,
                "coverage": 0.0, "n_episodes": 0,
                "note": _STANDING_NOTE}
    weights: List[int] = []
    rewards: List[float] = []
    for e in episodes:
        approved = e.get("approved")
        approved = approved in (True, 1, "approved", 1.0)
        if candidate_fn(e.get("context") or {}):
            weights.append(1)
            rewards.append(1.0 if approved else 0.0)
        else:
            weights.append(0)
    support = sum(weights)
    coverage = round(support / n, 4)
    if support == 0:
        return {"refused": ("the candidate policy would have proposed "
                            "NOTHING in this log — a point estimate "
                            "would be pure extrapolation"),
                "estimated_accept_rate": None, "support": 0,
                "coverage": coverage, "n_episodes": n,
                "note": _STANDING_NOTE}
    if support < min_support:
        return {"refused": f"thin support ({support} < {min_support}) — "
                           "the estimate exists but is labeled thin and "
                           "NOT reported as a rate",
                "estimated_accept_rate": None, "support": support,
                "coverage": coverage, "n_episodes": n,
                "note": _STANDING_NOTE}
    estimate = round(sum(rewards) / support, 4)
    return {"refused": None, "estimated_accept_rate": estimate,
            "support": support, "coverage": coverage, "n_episodes": n,
            "estimator": ("importance weighting (Horvitz-Thompson "
                          "weights; Dudik/Langford/Li 2011 framing): "
                          "deterministic candidate, weight 1 inside "
                          "its support"),
            "note": _STANDING_NOTE}


_STANDING_NOTE = ("evidence only — this module never writes the "
                  "capabilities manifest; flipping a switch stays a "
                  "file edit the human makes")


def ope_report(episodes: Sequence[Dict[str, Any]],
               switch_name: str) -> Dict[str, Any]:
    """The CLI-shaped report for one KNOWN kill-switch (an unknown name
    is an honest error listing what exists)."""
    spec = KNOWN_SWITCHES.get(switch_name)
    if spec is None:
        raise ValueError(
            f"unknown kill-switch {switch_name!r} — known: "
            f"{', '.join(sorted(KNOWN_SWITCHES))}")
    return evaluate_policy(episodes, switch_candidate(spec))

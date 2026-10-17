"""brain.regret — the periodic regret-vs-best-fixed audit for every
Beta bandit in the repo (exponential-build 3.3).

The question it answers, printed rather than claimed: "did the bandit
actually beat just always doing the single best thing?" The baseline is
the BEST-FIXED-ARM-IN-HINDSIGHT: the arm with the best observed mean
reward, multiplied across the whole horizon — the reward the user would
plausibly have collected had the bandit done nothing clever at all.

Two input shapes, same audit:

- ``audit_from_arms``: the aggregate Beta posteriors every bandit
  already persists ({name: [alpha, beta]}) — the reward sums and play
  counts are derived from the evidence (the prior's own mass is never
  counted; a flat-seeded arm has n = 0);
- ``audit_from_draws``: an explicit per-round draw log
  [(arm, reward), ...] where one exists.

Honesty accounting: the best-fixed total is an ESTIMATE (it assumes
the best arm's observed mean would have held over the whole horizon —
unplayed rounds are unobserved, and the report says exactly that);
the regret figure inherits that caveat and is never described as a
bound; a bandit with fewer than two played arms has no policy
comparison to make and says so; an untouched bandit (no plays at all)
is an abstention, not a zero. Deterministic, pure, no I/O — the CLI
surfaces it, nothing acts on it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["audit_from_arms", "audit_from_draws"]

_FLAT_PRIOR = 1.0


def _audit(per_arm: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shared tail: totals, best-fixed-in-hindsight estimate, regret."""
    horizon = sum(a["plays"] for a in per_arm)
    cumulative = sum(a["reward_sum"] for a in per_arm)
    played = [a for a in per_arm if a["plays"] > 0]
    out: Dict[str, Any] = {
        "per_arm": per_arm,
        "horizon": horizon,
        "cumulative_reward": round(cumulative, 4),
        "method": ("regret vs best-fixed-arm-in-hindsight; the best "
                   "arm's observed mean is assumed to hold over the "
                   "whole horizon — an estimate, not a bound"),
    }
    if horizon == 0:
        out.update({"status": "abstained",
                    "note": "no plays recorded — nothing to compare yet"})
        return out
    if len(played) < 2:
        out.update({
            "status": "single-arm",
            "best_fixed_arm": played[0]["arm"] if played else None,
            "estimated_regret": 0.0,
            "note": ("only one arm was ever played — no policy "
                     "comparison exists; regret is 0 by definition")})
        return out
    best = max(played, key=lambda a: (a["mean_reward"], a["arm"]))
    best_fixed_total = best["mean_reward"] * horizon
    regret = best_fixed_total - cumulative
    out.update({
        "status": "compared",
        "best_fixed_arm": best["arm"],
        "best_fixed_estimate": round(best_fixed_total, 4),
        "estimated_regret": round(regret, 4),
        "note": ("positive estimated regret = the bandit trailed always-"
                 f"-playing-{best['arm']}; an estimate over unobserved "
                 "rounds, printed for the human — nothing acts on it"),
    })
    return out


def audit_from_arms(arms: Dict[str, Sequence[float]],
                    prior: float = _FLAT_PRIOR) -> Dict[str, Any]:
    """The audit from aggregate Beta posteriors {name: [alpha, beta]}.
    Evidence only: plays = (alpha - prior) + (beta - prior), reward sum
    = alpha - prior (the flat prior's own mass is never counted)."""
    per_arm: List[Dict[str, Any]] = []
    for name in sorted(arms):
        alpha, beta = (float(v) for v in arms[name][:2])
        s = max(0.0, alpha - prior)
        f = max(0.0, beta - prior)
        n = s + f
        per_arm.append({"arm": name, "plays": n, "reward_sum": s,
                        "mean_reward": round(s / n, 6) if n else None})
    return _audit(per_arm)


def audit_from_draws(draws: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    """The audit from an explicit per-round draw log [(arm, reward)]."""
    stats: Dict[str, Dict[str, float]] = {}
    for arm, reward in draws:
        row = stats.setdefault(arm, {"plays": 0.0, "reward_sum": 0.0})
        row["plays"] += 1.0
        row["reward_sum"] += 1.0 if reward in (1, True, 1.0) else 0.0
    per_arm: List[Dict[str, Any]] = []
    for name in sorted(stats):
        s = stats[name]["reward_sum"]
        n = stats[name]["plays"]
        per_arm.append({"arm": name, "plays": n, "reward_sum": s,
                        "mean_reward": round(s / n, 6) if n else None})
    return _audit(per_arm)

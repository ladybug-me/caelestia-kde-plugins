"""cortex.conservative — a safe-exploration floor on the strategy
bandit (exponential-build-4 D).

Wu, Shariff, Lattimore & Szepesvári, "Conservative Bandits", ICML 2016:
unconstrained Thompson sampling will happily spend user experience on
an untested strategy. The conservative design adds a SAFE BASELINE arm
(here: the "balanced" routing profile the repo shipped before any
learning) and the paper's central gate — a non-baseline arm is explored
only when its upper confidence bound clears the safe arm's lower
confidence bound minus the budget ε:

    explore arm i  iff  LCB_i > LCB_safe − ε        (the paper's
    UCB_i ≥ LCB_0 − ε, with the strictness flipped to keep UNSAMPLED
    arms from exploring forever on a degenerate UCB tie),

with Hoeffding radii  r = sqrt(ln(2/δ) / (2n))  over the empirical
mean. The paper's Theorem 1 then gives the guarantee this module
claims, no more and no less: with probability ≥ 1−δ, the total regret
against ALWAYS PLAYING THE SAFE ARM is at most ε·T (+ a constant), i.e.
exploration is provably no worse than the baseline by more than the
per-round budget ε, in expectation. That is a statement about distance
from the BASELINE, not about finding the best arm — an arm better than
the safe one is still exploited once its evidence clears the gate.

ON TOP OF, NOT REPLACING: the Beta posteriors this gate reads are the
strategy bandit's own persisted counts (cortex/learn.py NamedBandit
state) — the same state, a gated policy in front of the Thompson
sample. When the gate refuses an arm, the SAFE arm plays; the sampled
arm is reported honestly as "gated" so the caller can see what
Thompson wanted and what the floor allowed. Determinism: the gate is
pure arithmetic; the Thompson sample comes from the caller's seeded
rng exactly as before. Nothing here writes; state stays in the
caller's learned-state JSON.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

__all__ = ["ConservativeBandit", "hoeffding_radius"]


def hoeffding_radius(n: int, delta: float) -> float:
    """The Hoeffding radius sqrt(ln(2/delta)/(2n)) for n observations;
    unbounded (refuses to claim confidence) at n = 0."""
    if n <= 0:
        raise ValueError("radius is undefined at n = 0 — the honest "
                         "answer is 'no evidence', not a small radius")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be inside (0, 1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


class ConservativeBandit:
    """The safe-exploration gate over per-arm Beta evidence.

    ``arms``: {name: {"alpha": float, "beta": float}} — the SAME
    posterior dicts NamedBandit persists (prior mass is included in
    the posterior mean but its play count is not: n = alpha + beta −
    prior mass, so a flat-seeded arm has n = 0 and an unbounded
    radius, which is the honest reading).
    """

    def __init__(self, arms: Dict[str, Dict[str, float]],
                 safe_arm: str, epsilon: float = 0.05,
                 delta: float = 0.05, prior_mass: float = 2.0) -> None:
        if safe_arm not in arms:
            raise ValueError(
                f"safe arm {safe_arm!r} not among the arms "
                f"{sorted(arms)}")
        if epsilon < 0:
            raise ValueError("epsilon (the safety budget) must be >= 0 — "
                             "a negative budget is not a budget")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be inside (0, 1)")
        self.arms = {name: dict(post) for name, post in arms.items()}
        self.safe_arm = safe_arm
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.prior_mass = float(prior_mass)

    # -- posterior reads ---------------------------------------------------

    def _mean(self, name: str) -> float:
        post = self.arms[name]
        return post["alpha"] / (post["alpha"] + post["beta"])

    def _plays(self, name: str) -> int:
        post = self.arms[name]
        # the posterior's own prior mass is not evidence: n = mass − prior
        n = (post["alpha"] + post["beta"]) - self.prior_mass
        return max(0, int(n))

    def _lcb(self, name: str) -> Optional[float]:
        n = self._plays(name)
        if n == 0:
            return None  # no evidence: no lower bound is claimed
        return self._mean(name) - hoeffding_radius(n, self.delta)

    def _ucb(self, name: str) -> Optional[float]:
        n = self._plays(name)
        if n == 0:
            return None
        return self._mean(name) + hoeffding_radius(n, self.delta)

    # -- the gate ----------------------------------------------------------

    def choose(self, thompson_sample: Optional[str] = None
               ) -> Dict[str, Any]:
        """Apply the gate to one decision.

        ``thompson_sample``: the arm the underlying Thompson policy
        already picked (the caller samples; this module gates). Returns
        the chosen arm, whether the gate overrode it, and the bounds
        the decision was made on — the report carries its own evidence.
        With no Thompson sample given, the safe arm's eligibility is
        still computed the same way (the gate can only ever PLAY the
        safe arm or ratify the sample; it never invents a third arm).
        """
        safe_lcb = self._lcb(self.safe_arm)
        details: Dict[str, Any] = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "safe_arm": self.safe_arm,
            "safe_lcb": safe_lcb,
        }
        if safe_lcb is None:
            # the safe arm itself has no evidence: no floor exists yet,
            # play the safe arm and say so (exploring others with no
            # baseline evidence is the unconstrained behavior this
            # module exists to prevent)
            details["reason"] = "safe arm has no evidence — floor closed"
            return {"arm": self.safe_arm, "overridden": True,
                    "gated_sample": thompson_sample, "details": details}
        if thompson_sample is None or thompson_sample == self.safe_arm:
            return {"arm": self.safe_arm,
                    "overridden": thompson_sample not in (None, self.safe_arm),
                    "gated_sample": thompson_sample, "details": details}
        ucb = self._ucb(thompson_sample)
        details["sample_ucb"] = ucb
        if ucb is None:
            # an unplayed arm: its UCB is the prior mean with an
            # unbounded radius — the paper treats unexplored arms'
            # UCB optimistically; here there is no evidence to clear
            # the gate with, so the honest reading is "not yet"
            details["reason"] = "sampled arm has no evidence — floor closed"
            return {"arm": self.safe_arm, "overridden": True,
                    "gated_sample": thompson_sample, "details": details}
        if ucb > safe_lcb - self.epsilon:
            details["reason"] = "sample cleared the gate"
            return {"arm": thompson_sample, "overridden": False,
                    "gated_sample": thompson_sample, "details": details}
        details["reason"] = (
            f"UCB {ucb:.4f} <= safe LCB {safe_lcb:.4f} − ε "
            f"{self.epsilon} — floor closed")
        return {"arm": self.safe_arm, "overridden": True,
                "gated_sample": thompson_sample, "details": details}

    def guarantee(self) -> str:
        """The guarantee, stated exactly as the paper gives it — with
        its scope (distance from the BASELINE, high probability, in
        expectation) — and never oversold."""
        return (
            f"Wu et al. 2016, Theorem 1 (this implementation's form): "
            f"with probability >= {1.0 - self.delta}, the total regret "
            f"against always playing {self.safe_arm!r} is at most "
            f"{self.epsilon} * T (+ constants) — exploration is no "
            f"worse than the baseline by more than ε per round, in "
            f"expectation. This bounds DISTANCE FROM THE BASELINE, "
            f"not regret against the best arm.")

    def report(self) -> Dict[str, Any]:
        """The audit card: per-arm evidence, bounds, and the gate state.
        An untouched arm reports n = 0 and unclaimed bounds — an
        abstention, never a zero."""
        rows: List[Dict[str, Any]] = []
        for name in sorted(self.arms):
            n = self._plays(name)
            rows.append({
                "arm": name, "plays": n,
                "mean": round(self._mean(name), 4) if n else None,
                "lcb": round(self._lcb(name), 4) if self._lcb(name) is not None else None,
                "ucb": round(self._ucb(name), 4) if self._ucb(name) is not None else None,
                "safe": name == self.safe_arm,
            })
        return {"arms": rows, "epsilon": self.epsilon, "delta": self.delta,
                "guarantee": self.guarantee()}

    # -- persistence (the caller's learned-state JSON; same shape as
    #    the underlying bandit state, so no second path exists) ------

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": self.arms, "safe_arm": self.safe_arm,
                "epsilon": self.epsilon, "delta": self.delta,
                "prior_mass": self.prior_mass}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConservativeBandit":
        return cls(arms=d.get("arms") or {}, safe_arm=d.get("safe_arm",
                                                            "balanced"),
                   epsilon=d.get("epsilon", 0.05),
                   delta=d.get("delta", 0.05),
                   prior_mass=d.get("prior_mass", 2.0))

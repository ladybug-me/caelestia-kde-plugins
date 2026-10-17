"""brain.ranking — the shared pairwise ranking primitive (phase 2.4).

Generalizes the already-specified Elo/Bradley-Terry preset ranking
(the former proposals/2026-09-26-c-elo-preset-ranking.md, pruned
in exponential-build-3 Phase 2 once implemented) to ANY pairwise
comparison of named items — presets AND agent-proposed plans both
consume this ONE module:

- **EloLadder**: the online Elo rating (Elo 1978; the Bernoulli(1)
  online special case of Bradley-Terry) — one (winner, loser) pair
  updates both ratings by K * (outcome - expected). Deterministic,
  commutative in history order, O(1) per update, plain dict state.
- **bradley_terry()**: the batch fit — one latent strength theta per
  item, logistic pairwise likelihood P(i beats j) = sigmoid(theta_i -
  theta_j) (Bradley & Terry 1952), fitted by simple gradient ascent
  with adaptive step (the same first-order shape cortex's OnlineLogistic
  uses, reused as an idea, not imported — this fit needs no feature
  model, just per-item free parameters).
- **Uncertainty**: per-item comparison counts render as honesty —
  items with fewer than MIN_COMPARISONS are "not enough data", never a
  confident rank (the proposal's §4 sparsity rule, pinned by test).

Both consumers keep their pair records SEPARATELY (presets: the
settings layer's explicit `prefer` answers; plans: the agent layer's
chosen-vs-rejected consent outcomes) and call the same functions —
state files never merge; the ALGORITHM is the shared thing.

Pure module: no I/O, no randomness (batch fit is deterministic
gradient ascent with a fixed iteration count and order).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

__all__ = ["EloLadder", "bradley_terry", "ladder_report",
           "MIN_COMPARISONS", "DEFAULT_ELO_K", "BASE_RATING"]

MIN_COMPARISONS = 3
DEFAULT_ELO_K = 24.0
BASE_RATING = 1200.0


class EloLadder:
    """Online Elo ratings over named items. Each pair record is
    (winner, loser); ratings start at BASE_RATING (no prior opinion)."""

    def __init__(self, ratings: Dict[str, float] | None = None,
                 counts: Dict[str, int] | None = None,
                 k: float = DEFAULT_ELO_K):
        self.ratings: Dict[str, float] = {
            name: float(value) for name, value in (ratings or {}).items()}
        self.counts: Dict[str, int] = {name: int(value)
                                       for name, value in (counts or {}).items()}
        self.k = float(k)

    def _rating(self, name: str) -> float:
        return self.ratings.setdefault(name, BASE_RATING)

    def _count(self, name: str) -> int:
        return self.counts.setdefault(name, 0)

    def observe(self, winner: str, loser: str) -> None:
        """One pairwise outcome: winner beats loser."""
        if winner == loser:
            raise ValueError("an item cannot play itself")
        rw, rl = self._rating(winner), self._rating(loser)
        expected_w = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        delta = self.k * (1.0 - expected_w)
        self.ratings[winner] = rw + delta
        self.ratings[loser] = rl - delta
        self.counts[winner] = self._count(winner) + 1
        self.counts[loser] = self._count(loser) + 1

    def expected(self, a: str, b: str) -> float:
        """P(a beats b) under the current ratings (the logistic BT link)."""
        ra, rb = self._rating(a), self._rating(b)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def ladder(self, names: Sequence[str] | None = None
               ) -> List[Dict[str, Any]]:
        """Sorted ladder rows: rating, comparisons, honest-confidence."""
        items = list(names) if names else sorted(self.ratings)
        rows = [{
            "item": name,
            "rating": round(self._rating(name), 1),
            "comparisons": self._count(name),
            "enough_data": self._count(name) >= MIN_COMPARISONS,
        } for name in items]
        rows.sort(key=lambda r: (-r["rating"], r["item"]))
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {"ratings": dict(self.ratings),
                "counts": dict(self.counts), "k": self.k}

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "EloLadder":
        data = data or {}
        return cls(data.get("ratings"), data.get("counts"),
                   k=float(data.get("k", DEFAULT_ELO_K)))


def bradley_terry(pairs: Iterable[Tuple[str, str]],
                  iterations: int = 200,
                  step: float = 0.05) -> Dict[str, Any]:
    """Batch Bradley-Terry strengths from (winner, loser) pairs.

    One theta per item, logistic likelihood, deterministic gradient
    ascent (fixed order, fixed iterations, adaptive step decay). The
    strengths are z-like scores (mean 0); the uncertainty column is the
    comparison count — the honest proxy, not a fake posterior.
    """
    pair_list = [(str(w), str(l)) for w, l in pairs]
    counts: Dict[str, int] = {}
    for winner, loser in pair_list:
        if winner == loser:
            raise ValueError(f"item {winner!r} cannot play itself")
        counts[winner] = counts.get(winner, 0) + 1
        counts[loser] = counts.get(loser, 0) + 1
    items = sorted(counts)
    index = {name: i for i, name in enumerate(items)}
    theta = [0.0] * len(items)
    wins = [0.0] * len(items)
    losses = [0.0] * len(items)
    for winner, loser in pair_list:
        wins[index[winner]] += 1.0
        losses[index[loser]] += 1.0
    if not pair_list:
        return {"strengths": {}, "n_pairs": 0, "iterations": 0}
    lr = step
    for it in range(iterations):
        # gradient of sum log sigmoid(theta_w - theta_l)
        grad = [0.0] * len(items)
        for winner, loser in pair_list:
            i, j = index[winner], index[loser]
            p = 1.0 / (1.0 + math.exp(-(theta[i] - theta[j])))
            grad[i] += (1.0 - p)
            grad[j] -= (1.0 - p)
        for i in range(len(items)):
            theta[i] += lr * grad[i]
        lr = step / (1.0 + 0.02 * it)  # decay: stable, deterministic
    # center on mean zero (strengths are relative)
    mean = sum(theta) / len(theta)
    theta = [t - mean for t in theta]
    return {
        "strengths": {name: round(theta[index[name]], 4) for name in items},
        "counts": counts,
        "n_pairs": len(pair_list),
        "iterations": iterations,
        "algorithm": "bradley-terry logistic fit, deterministic gradient "
                     "ascent (Bradley & Terry 1952; the proposal's "
                     "verification plan governs the tests)",
    }


def ladder_report(pairs: Iterable[Tuple[str, str]],
                  elo: EloLadder | None = None) -> Dict[str, Any]:
    """Both views over the SAME pair records — the online ladder and the
    batch fit — plus the agreement between them (Kendall tau, the
    proposal's own recovery metric), reported as numbers, not claims."""
    pair_list = [(str(w), str(l)) for w, l in pairs]
    elo = elo or EloLadder()
    for winner, loser in pair_list:
        elo.observe(winner, loser)
    bt = bradley_terry(pair_list)
    # Kendall tau between the two orderings over items with enough data
    ranked = [r for r in elo.ladder() if r["enough_data"]]
    tau = None
    if len(ranked) >= 2:
        elo_order = [r["item"] for r in ranked]
        bt_order = sorted(ranked, key=lambda r: -bt["strengths"].get(r["item"], 0.0))
        bt_order = [r["item"] for r in bt_order]
        concordant = discordant = 0
        for i in range(len(elo_order)):
            for j in range(i + 1, len(elo_order)):
                same = (elo_order.index(elo_order[i]) - elo_order.index(elo_order[j])) * \
                       (bt_order.index(elo_order[i]) - bt_order.index(elo_order[j]))
                if same > 0:
                    concordant += 1
                elif same < 0:
                    discordant += 1
        denom = concordant + discordant
        tau = round((concordant - discordant) / denom, 3) if denom else None
    return {
        "ladder": elo.ladder(),
        "elo": elo.to_dict(),
        "bradley_terry": bt,
        "agreement_kendall_tau": tau,
        "note": "proposals only — nothing is applied; items under "
                f"{MIN_COMPARISONS} comparisons render 'not enough data'",
    }

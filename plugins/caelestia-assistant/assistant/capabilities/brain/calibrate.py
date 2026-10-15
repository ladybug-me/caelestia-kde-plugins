"""Learn how much to trust each proposal *kind*, and how many proposals to
show per day, from ledger history alone — no model, just Beta-Binomial
bookkeeping over decisions that already happened.

acceptance_rate: a posterior per kind, so a kind the user always rejects
stops being pushed as hard. confidence_calibration: the same bucket-vs-hit-
rate idea as journal.calibration_curve, but sourced from the ledger's own
stated confidence instead of a manually kept journal. DailyBudget: a
two-armed Beta bandit ("more" vs "less" proposals today) driven by whether
the user actually engaged, reusing bandit.HourBandit's Thompson-sampling
pattern for a different decision.
"""
import random
from collections import defaultdict

# Exponential-build 1.2: the weight of an approve-then-quick-undo (the
# reverted apply was still the head of the bounded undo ring). With the
# Beta(1,1) prior, approval + quick undo gives mean 2/(3+w) vs a plain
# reject's 1/3 — strictly further only when w > 3, so 4.0 is the
# smallest integer weight that pins "shifts further than a plain
# reject". A slow undo stays at weight 1.0 (see fold_undo_negatives).
QUICK_UNDO_WEIGHT = 4.0


def acceptance_rate(labeled, prior_alpha=1.0, prior_beta=1.0):
    """labeled: [{"kind", "status"}], status in {"approved", "rejected"}
    (this is exactly Ledger.labeled()'s shape). Returns
    {kind: {"alpha", "beta", "mean", "n"}}.
    """
    stats = defaultdict(lambda: {"alpha": prior_alpha, "beta": prior_beta, "n": 0})
    for item in labeled:
        s = stats[item["kind"]]
        s["n"] += 1
        if item["status"] == "approved":
            s["alpha"] += 1.0
        else:
            s["beta"] += 1.0
    for s in stats.values():
        s["mean"] = round(s["alpha"] / (s["alpha"] + s["beta"]), 3)
    return dict(stats)


def confidence_calibration(labeled, bins=5):
    """labeled: [{"confidence", "status"}]."""
    buckets = defaultdict(list)
    for item in labeled:
        b = min(int(item["confidence"] * bins), bins - 1)
        buckets[b].append(item)
    out = []
    for b in sorted(buckets):
        items = buckets[b]
        avg_conf = sum(i["confidence"] for i in items) / len(items)
        hit = sum(1 for i in items if i["status"] == "approved") / len(items)
        out.append({"bucket": b, "n": len(items),
                    "avg_confidence": round(avg_conf, 3), "approval_rate": round(hit, 3)})
    return out


def fold_undo_negatives(stats, undo_log, weight=1.0,
                        quick_weight=QUICK_UNDO_WEIGHT, kind="settings"):
    """A3 — fold the settings layer's PII-stripped undo log into the
    Beta-Binomial posteriors as EXPLICIT negative signal.

    ``stats`` is an acceptance_rate() result ({kind: {alpha, beta, mean, n}});
    ``undo_log`` is settings.history.undo_log(target) — records of
    {"tool", "magnitude", "direction", "quick"} and nothing else (the
    PII-strip rule; magnitude is retained for inspectability, not used to
    scale the evidence — an undo is one negative observation regardless of
    how big the reverted change was).

    WEIGHTING (exponential-build 1.2 — a weighted update, not a new model):
    a record with ``quick=True`` (the reverted apply was still the head of
    the bounded undo ring — an approve-then-immediately-revert) adds
    ``quick_weight`` to the beta side; any other undo adds the plain
    ``weight`` (1.0, one observation, what a plain reject contributes).
    The arithmetic the default pins: with the Beta(1,1) prior, one
    approval followed by one quick undo gives mean 2/(3+quick_weight);
    one plain reject gives 1/3; the posterior shifts FURTHER than the
    plain reject exactly when quick_weight > 3, so QUICK_UNDO_WEIGHT is
    4.0 — the smallest integer weight that satisfies it (2/7 < 1/3). A
    SLOW undo (weight 1.0) merely cancels the approval it reverts
    (2/4 = 0.5): the previous, conservative behavior, kept.

    Each record adds its weight to the beta (negative) side of BOTH the
    kind-level posterior (``kind``, default "settings": a change the user
    reverted is evidence against settings proposals generally) and a
    per-tool posterior under ``"tool:<name>"`` (evidence against that
    specific knob). Means are recomputed; ``n`` counts folded records so
    the honesty invariant (an effective sample size you can see) holds.
    Mutates and returns ``stats`` (same dict, callers keep ownership).
    """
    for record in undo_log or []:
        tool = record.get("tool")
        if not tool:
            continue
        w = float(quick_weight) if record.get("quick") else float(weight)
        for key in (kind, f"tool:{tool}"):
            entry = stats.setdefault(
                key, {"alpha": 1.0, "beta": 1.0, "mean": 0.5, "n": 0})
            entry["beta"] += w
            entry["n"] += 1
            entry["mean"] = round(
                entry["alpha"] / (entry["alpha"] + entry["beta"]), 3)
    return stats


class DailyBudget:
    """Two arms, "more" and "less" proposals today. Reward = 1 if the user
    engaged (approved or rejected outright) rather than leaving it pending
    long enough to count as ignored."""

    def __init__(self, alpha=None, beta=None):
        self.alpha = dict(alpha) if alpha else {"more": 1.0, "less": 1.0}
        self.beta = dict(beta) if beta else {"more": 1.0, "less": 1.0}

    def choose(self, rng=None):
        rng = rng or random
        return max(self.alpha, key=lambda arm: rng.betavariate(self.alpha[arm], self.beta[arm]))

    def reward(self, arm, engaged):
        if engaged:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0

    def to_dict(self):
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("alpha"), d.get("beta"))

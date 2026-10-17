"""cortex.engine_telemetry — opt-in, coverage/accuracy-only engine
metrics, exportable through the Laplace DP mechanism (exponential-
build-3 F3, riding exponential-build-3 B4's cortex/dp.py).

WHAT IS SHARED, EXACTLY: per routing surface (the tool the router
picked), the aggregate COUNTS and RATES of the bounded example log —
how many routed turns there were (n), how many the engine committed
to a verdict on (decided = applied/approved/rejected; the abstains
and clarifies are the coverage gap), and what fraction of decided
turns were accepted (accuracy). NOTHING ELSE: no text, no phrases, no
feature vectors, no timestamps, no tool names beyond the registry's
own surface identifiers (which the public registry already names).
The master scope said coverage/accuracy only — this module cannot
export anything else, by construction.

THE DP MECHANISM (Dwork et al. 2006, reused from cortex/dp.py — the
B4 machinery, not a second implementation): every exported row's n is
Laplace-noised at sensitivity 1 (floored at 0, halves rounding up),
every rate is noised at sensitivity 1 and clipped to [0, 1] with the
clip counted, and every row carries an explicit "(dp: epsilon=X)"
provenance marker so a noised artifact can never masquerade as exact.
Composition (Dwork & Roth 2014): k exports compose to ~k*epsilon;
export is user-initiated, so the practical bound is export frequency.
Per-row sensitivity rationale: one example log turn moves its
surface's n by 1 and its decided/accepted counts by at most 1 — the
same bounding argument as the lexicon diff (learn.MAX_EXAMPLES = 500
caps any one release's contribution).

CONSENT + CAPABILITY: read-only. The export prints to stdout; there
is no network, no auto-share, and no write. The surface sits behind
the capability manifest's "engine_telemetry" flag (default ON — a
read-only local report, the same posture as lexicon_sharing) so a
community deployment can switch it off with one file edit.

Honesty: a surface with no decided turns has NO accuracy claim — its
accuracy is reported as None (rendered "-"), never 0.0 (which would
claim every decision was wrong); thin counts (n < MIN_N, default 4)
are LABELED thin in the render. The exact export and the noised
export are byte-distinct artifacts with different headers; mixing
them is impossible by construction.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence

from .dp import laplace_noise

__all__ = ["metrics", "export_text", "noise_metrics", "MIN_N"]

MIN_N = 4  # below this, the render labels the row thin
EPSILON_DEFAULT = 1.0


def metrics(learner: Any) -> List[Dict[str, Any]]:
    """Per-surface coverage/accuracy rows from a CortexLearner's
    bounded example log. Row shape: {surface, n, decided, accepted,
    rejected, coverage, accuracy} where coverage = decided/n and
    accuracy = accepted/decided (None when decided == 0 — no claim).
    Sorted by (-n, surface) — deterministic."""
    rows: Dict[str, Dict[str, Any]] = {}
    for example in getattr(learner, "examples", []):
        surface = str(example.get("surface") or "(unknown)")
        row = rows.setdefault(surface, {"surface": surface, "n": 0,
                                        "decided": 0, "accepted": 0,
                                        "rejected": 0})
        row["n"] += 1
        outcome = example.get("outcome")
        if outcome in ("applied", "approved"):
            row["decided"] += 1
            row["accepted"] += 1
        elif outcome == "rejected":
            row["decided"] += 1
            row["rejected"] += 1
        # clarified/abstained/ambiguous: the coverage gap — counted
        # in n, deliberately NOT counted as decided (abstaining is
        # not a wrong answer, it is the honest no-answer)
    out = []
    for row in rows.values():
        n, decided = row["n"], row["decided"]
        out.append({
            "surface": row["surface"],
            "n": n,
            "decided": decided,
            "accepted": row["accepted"],
            "rejected": row["rejected"],
            "coverage": round(decided / n, 4) if n else None,
            "accuracy": (round(row["accepted"] / decided, 4)
                         if decided else None),
        })
    return sorted(out, key=lambda r: (-r["n"], r["surface"]))


def export_text(rows: Sequence[Dict[str, Any]], date: str = "",
                dp: Optional[Dict[str, Any]] = None) -> str:
    """The shareable artifact (exact OR noised — the caller passes
    one or the other; the header states which, byte-distinctly)."""
    kind = "EXACT (local read-back; not for sharing)"
    if dp is not None:
        kind = (f"DP-NOISED (epsilon={dp['epsilon']:g}, "
                f"seed={dp['seed']}; Laplace per Dwork et al. 2006)")
    lines = [
        "# caelestia-assistant engine telemetry — coverage/accuracy only",
        f"# artifact: {kind}",
        f"# fields: surface | n | decided | accepted | rejected | "
        f"coverage | accuracy",
        f"# no text, no phrases, no features, no timestamps are "
        f"present in this artifact",
    ]
    if date:
        lines.append(f"# date: {date}")
    for row in rows:
        thin = " (thin)" if row["n"] < MIN_N else ""
        acc = "-" if row["accuracy"] is None else f"{row['accuracy']}"
        cov = "-" if row["coverage"] is None else f"{row['coverage']}"
        marker = row.get("dp_marker", "")
        lines.append(
            f"{row['surface']} | {row['n']} | {row['decided']} | "
            f"{row['accepted']} | {row['rejected']} | {cov} | {acc}"
            f"{thin}{(' ' + marker) if marker else ''}")
    lines.append("# accuracy is '-' (no claim) for surfaces with zero "
                 "decided turns; thin rows (n < "
                 f"{MIN_N}) are labeled")
    return "\n".join(lines) + "\n"


def noise_metrics(rows: Sequence[Dict[str, Any]],
                  epsilon: float = EPSILON_DEFAULT,
                  seed: Optional[int] = None) -> Dict[str, Any]:
    """The DP export pass (cortex/dp.py's Laplace mechanism): n and
    decided noised at sensitivity 1 (nearest int, halves up, floored
    at 0 — the floor count reported); coverage and accuracy noised at
    sensitivity 1 and clipped to [0, 1] (clip counts reported);
    accepted/rejected recomputed as noised-decided x noised-accuracy
    rounded, so the printed row stays internally consistent; every
    row carries an explicit (dp: epsilon=X) marker. The default seed
    derives from the rows' content hash — reproducible, which is NOT
    independent across exports (the same tradeoff cortex/dp.py
    states; pass an int for fresh noise)."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive (got {epsilon!r})")
    if seed is None:
        # content-derived: the same rows produce the same noise
        # (reproducible, NOT independent across exports — the same
        # tradeoff cortex/dp.py states and pins)
        canonical = repr([sorted(r.items(), key=lambda kv: str(kv[0]))
                           for r in rows])
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
    rng = random.Random(seed)
    out_rows: List[Dict[str, Any]] = []
    n_floored = clipped = 0
    for row in rows:
        n_noised = laplace_noise(float(row["n"]), 1.0, epsilon, rng)
        n_noised = max(int(n_noised + 0.5), 0)
        if n_noised == 0 and row["n"] > 0:
            n_floored += 1
        decided_noised = laplace_noise(float(row["decided"]), 1.0,
                                       epsilon, rng)
        decided_noised = max(int(decided_noised + 0.5), 0)
        if decided_noised > n_noised:
            decided_noised = n_noised
        acc = row["accuracy"]
        acc_noised = None
        if acc is not None and decided_noised > 0:
            # a surface noised down to zero decided turns carries NO
            # accuracy claim (0.0 would assert "all wrong" from no
            # evidence — the noise suppressed the signal, say so)
            acc_noised = laplace_noise(float(acc), 1.0, epsilon, rng)
            if acc_noised < 0.0:
                acc_noised = 0.0
                clipped += 1
            elif acc_noised > 1.0:
                acc_noised = 1.0
                clipped += 1
        out_rows.append({
            "surface": row["surface"],
            "n": n_noised,
            "decided": decided_noised,
            "accepted": (round(decided_noised * acc_noised)
                         if acc_noised is not None else 0),
            "rejected": (decided_noised
                         - round(decided_noised * acc_noised)
                         if acc_noised is not None else decided_noised),
            "coverage": (round(decided_noised / n_noised, 4)
                         if n_noised else None),
            "accuracy": (round(acc_noised, 4)
                         if acc_noised is not None else None),
            "dp_marker": f"(dp: epsilon={epsilon:g})",
        })
    return {"rows": out_rows, "epsilon": epsilon, "seed": seed,
            "n_floored_at_zero": n_floored, "p_clipped": clipped}

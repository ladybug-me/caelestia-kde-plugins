"""brain.prefs — a Beta-posterior preference model over your approve/reject history.

The ledger already records every decision; this module turns that history into
a CALIBRATED preference prior the rest of the assistant acts on:

  observe(group, direction, accepted, hour)  -> updates Beta(alpha, beta) for
      (setting-group, direction, hour-bucket-of-6) with a shared weak prior
  bias(group, direction, hour)               -> posterior mean + 95% interval
      + effective sample size (the honesty part: 1 observation is 1 observation)
  rank(candidates, hour)                     -> reorders proposals by learned
      acceptance probability instead of raw confidence
  explain(...)                               -> "you usually approve X" strings

Why Beta posteriors and not the online logistic the cortex already uses?
Different job: the logistic models ROUTE acceptance from rich features; this
models PER-CATEGORY appetite with a conjugate update that is exact, cheap,
and explainable in one sentence — the right shape for "you keep rejecting
bar changes at night" feedback. Same state file, same atomic-write path.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["PreferenceModel", "hour_bucket"]

_PRIOR_ALPHA = 1.0   # weak Beta(1,1) prior: maximum entropy, no opinions
_PRIOR_BETA = 1.0
_BUCKETS = 4         # 6-hour buckets: night/morning/afternoon/evening
_MIN_SHOWS = 3       # below this n the model refuses to speak strongly


def hour_bucket(hour: Optional[int]) -> int:
    """0=night (0-5), 1=morning (6-11), 2=afternoon (12-17), 3=evening (18-23)."""
    if hour is None:
        return -1
    return int(((hour % 24) // (24 // _BUCKETS)))


class PreferenceModel:
    """Beta posteriors keyed by (group, direction, bucket). JSON round-trips
    through the brain state file via to_dict/from_dict."""

    def __init__(self) -> None:
        self.table: Dict[str, Tuple[float, float]] = {}

    @staticmethod
    def key(group: str, direction: str, bucket: int) -> str:
        return f"{group}|{direction}|{bucket}"

    # ------------------------------------------------------------------
    def observe(self, group: str, direction: str, accepted: bool,
                hour: Optional[int] = None) -> Tuple[float, float]:
        a, b = self.table.get(self.key(group, direction, hour_bucket(hour)),
                              (_PRIOR_ALPHA, _PRIOR_BETA))
        a, b = a + (1.0 if accepted else 0.0), b + (0.0 if accepted else 1.0)
        self.table[self.key(group, direction, hour_bucket(hour))] = (a, b)
        return a, b

    def observe_many(self, decisions: List[Dict[str, Any]]) -> int:
        """Bulk-learn from ledger rows: expects dicts with group/direction/
        approved/hour keys (see from_ledger)."""
        n = 0
        for d in decisions:
            if not d.get("group") or not d.get("direction"):
                continue
            self.observe(d["group"], d["direction"],
                         bool(d.get("approved")), d.get("hour"))
            n += 1
        return n

    def from_ledger(self, ledger_items: List[Dict[str, Any]]) -> int:
        """Adapt ledger proposals: target is a dotted settings path
        (group = first segment), diff direction inferred from the diff value
        sign; approved/rejected status is the label."""
        rows: List[Dict[str, Any]] = []
        for item in ledger_items:
            target = str(item.get("target", ""))
            if not target:
                continue
            group = target.split(".")[0]
            direction = _diff_direction(item.get("diff"))
            status = item.get("status")
            if status not in ("approved", "rejected"):
                continue
            decided = item.get("decided_at") or ""
            hour = None
            try:
                hour = datetime.fromisoformat(decided).hour if decided else None
            except ValueError:
                hour = None
            rows.append({"group": group, "direction": direction,
                         "approved": status == "approved", "hour": hour})
        return self.observe_many(rows)

    # ------------------------------------------------------------------
    def bias(self, group: str, direction: str,
             hour: Optional[int] = None) -> Dict[str, Any]:
        a, b = self.table.get(self.key(group, direction, hour_bucket(hour)),
                              (_PRIOR_ALPHA, _PRIOR_BETA))
        n = a + b - _PRIOR_ALPHA - _PRIOR_BETA
        mean = a / (a + b)
        lo, hi = _beta_ci(a, b)
        return {
            "group": group, "direction": direction,
            "bucket": hour_bucket(hour),
            "p_accept": round(mean, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "n": int(n),
            "verdict": _verdict(mean, int(n)),
        }

    def rank(self, candidates: List[Dict[str, Any]],
             hour: Optional[int] = None) -> List[Dict[str, Any]]:
        """candidates: dicts with 'group', 'direction' and optionally
        'confidence'. Returns them re-ordered by p_accept (n-weighted blend
        with the supplied confidence so rare-but-confident still surfaces)."""
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for c in candidates:
            b = self.bias(str(c.get("group", "")), str(c.get("direction", "")), hour)
            conf = float(c.get("confidence", 0.5))
            # shrink toward the caller's confidence when evidence is thin
            w = min(1.0, b["n"] / 10.0)
            blend = w * b["p_accept"] + (1.0 - w) * conf
            scored.append((blend, c))
        scored.sort(key=lambda t: -t[0])
        out = []
        for score, c in scored:
            row = dict(c)
            row["pref_score"] = round(score, 3)
            out.append(row)
        return out

    def explain(self, group: str, direction: str,
                hour: Optional[int] = None) -> str:
        b = self.bias(group, direction, hour)
        if b["n"] < _MIN_SHOWS:
            return (f"no reliable preference yet for {group} "
                    f"({b['n']} decision{'s' if b['n'] != 1 else ''})")
        when = {0: "at night", 1: "in the morning", 2: "in the afternoon",
                3: "in the evening", -1: "overall"}[b["bucket"]]
        pct = round(b["p_accept"] * 100)
        if pct >= 65:
            return f"you usually APPROVE {group} increases {when} ({pct}% of {b['n']})"
        if pct <= 35:
            return f"you usually REJECT {group} changes {when} ({pct}% of {b['n']})"
        return f"you are mixed on {group} changes {when} ({pct}% of {b['n']})"

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"table": {k: list(v) for k, v in self.table.items()}}

    def from_dict(self, d: Dict[str, Any]) -> "PreferenceModel":
        self.table = {k: tuple(v) for k, v in d.get("table", {}).items()}  # type: ignore
        return self


# ---------------------------------------------------------------------------
def _diff_direction(diff: Any) -> str:
    """'increase' / 'decrease' / 'toggle' from a stored diff value."""
    try:
        if isinstance(diff, dict):
            old, new = diff.get("old"), diff.get("new")
            if isinstance(old, bool) or isinstance(new, bool):
                return "toggle"
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                return "increase" if new > old else "decrease"
        return "set"
    except (TypeError, ValueError):
        return "set"


def _beta_ci(a: float, b: float, conf: float = 0.95) -> Tuple[float, float]:
    """Exact-ish Beta credible interval via the incomplete-beta inverse —
    replaced by a tight, monotone bisection on the CDF (regularised
    incomplete beta with the continued fraction, Lentz's algorithm)."""
    mean = a / (a + b)
    if a + b <= 2.5:
        return (0.0, 1.0)
    lo, hi = 0.0, 1.0
    target = (1.0 - conf) / 2.0

    def cdf(x: float) -> float:
        return _betainc(a, b, x)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    q_lo = (lo + hi) / 2.0
    lo, hi = 0.0, 1.0
    target_hi = 1.0 - target
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf(mid) < target_hi:
            lo = mid
        else:
            hi = mid
    return (q_lo, (lo + hi) / 2.0) if mean > 0 else (mean, mean)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b).

    For INTEGER (a, b) — the only shape the preference model ever produces
    (prior 1 + counts) — the exact finite binomial sum is used:
        I_x(a, b) = P(Bin(a+b-1, x) >= a)
    which cannot converge wrongly. Non-integer shapes fall back to the
    Lentz continued fraction (NR betacf) with a hardened stop.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if float(a).is_integer() and float(b).is_integer() and a >= 1 and b >= 1:
        n = int(a + b - 1)
        total = 0.0
        for j in range(int(a), n + 1):
            total += math.comb(n, j) * (x ** j) * ((1.0 - x) ** (n - j))
        return min(1.0, max(0.0, total))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-12) -> float:
    """Lentz continued fraction for the incomplete beta (NR betacf), with a
    hardened stop: no early exit before m=5 and two consecutive settled
    iterations (integer shapes can hit a spurious delta==1 fixed point)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    settled = 0
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if m >= 5 and abs(delta - 1.0) < eps:
            settled += 1
            if settled >= 2:
                break
        else:
            settled = 0
    return h


def _verdict(mean: float, n: int) -> str:
    if n < _MIN_SHOWS:
        return "unknown"
    if mean >= 0.65:
        return "usually approved"
    if mean <= 0.35:
        return "usually rejected"
    return "mixed"


def config_drift(model: "PreferenceModel", config: Dict[str, Any],
                hour: Optional[int] = None,
                specs=None) -> Dict[str, Any]:
    """B2 — distance between a LIVE config dict and this preference model.

    The adapter half of the drift score: flattens the model's Beta
    posteriors into the plain {(group, direction): {"p_accept", "n"}}
    table optimize.preference_drift consumes, and delegates the scoring to
    the settings layer's own pure function (layering stays
    one-directional: brain imports settings, never the reverse). PURE
    given (model, config): no file reads here either — the caller reads
    the live shell.json with the planner's own read-only helper.

    ``hour=None`` (the daily-brief default) merges the evidence across ALL
    hour buckets — the overall preference. An explicit hour restricts to
    that 6-hour bucket (a night-only rejection stays a night-only signal),
    which is the honest reading of "you usually reject X at this hour".
    """
    from assistant.capabilities.settings.optimize import preference_drift

    merged: Dict[Tuple[str, str], List[float]] = {}
    bucket = hour_bucket(hour)
    for key, (a, b) in model.table.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        group, direction, key_bucket = parts
        if hour is not None and int(key_bucket) != bucket:
            continue
        cell = merged.setdefault((group, direction), [0.0, 0.0])
        cell[0] += a - _PRIOR_ALPHA  # evidence mass only (prior removed)
        cell[1] += b - _PRIOR_BETA
    posterior: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (group, direction), (approves, rejects) in merged.items():
        alpha = _PRIOR_ALPHA + approves
        beta = _PRIOR_BETA + rejects
        posterior[(group, direction)] = {
            "p_accept": round(alpha / (alpha + beta), 3),
            "n": int(approves + rejects)}
    return preference_drift(config, posterior, specs=specs)

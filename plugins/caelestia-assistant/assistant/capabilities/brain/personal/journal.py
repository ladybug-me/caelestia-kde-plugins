"""Decision journal: record a decision with your stated confidence, later
record what actually happened, and see whether your confidence is
calibrated — i.e. whether the things you were "80% sure" about really came
true about 80% of the time.

Brier score is the standard proper scoring rule for a whole journal; the
calibration curve buckets by confidence and compares each bucket's average
confidence to its actual hit rate, using fewer/wider bins than a full
reliability diagram since a personal decision log is small.
"""
from collections import defaultdict


def record(entries, decision_id, statement, confidence):
    """entries: a plain dict (persist it via state.py). Mutates and returns
    entries[decision_id]."""
    entries[decision_id] = {
        "statement": statement, "confidence": float(confidence), "outcome": None,
    }
    return entries[decision_id]


def resolve(entries, decision_id, correct):
    if decision_id not in entries:
        raise KeyError(f"no decision {decision_id}")
    entries[decision_id]["outcome"] = bool(correct)
    return entries[decision_id]


def brier_score(entries):
    resolved = [e for e in entries.values() if e["outcome"] is not None]
    if not resolved:
        return None
    return sum((e["confidence"] - (1.0 if e["outcome"] else 0.0)) ** 2
               for e in resolved) / len(resolved)


def calibration_curve(entries, bins=5):
    resolved = [e for e in entries.values() if e["outcome"] is not None]
    if not resolved:
        return []
    buckets = defaultdict(list)
    for e in resolved:
        b = min(int(e["confidence"] * bins), bins - 1)
        buckets[b].append(e)
    out = []
    for b in sorted(buckets):
        items = buckets[b]
        avg_conf = sum(e["confidence"] for e in items) / len(items)
        hit_rate = sum(1 for e in items if e["outcome"]) / len(items)
        out.append({"bucket": b, "n": len(items),
                    "avg_confidence": round(avg_conf, 3), "hit_rate": round(hit_rate, 3)})
    return out

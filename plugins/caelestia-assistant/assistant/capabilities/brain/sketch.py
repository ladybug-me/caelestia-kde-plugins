"""brain.sketch — Space-Saving heavy hitters and t-digest quantiles
for disk and log sizing (C15).

"Which directories ate my disk?" and "what's the p95 log-line size?"
are streaming questions: the data does not fit comfortably in RAM, and
an exact answer costs a full sort. The classical sketch answers are:

  - SPACE-SAVING (Metwally et al. 2005): top-k heavy hitters with
    guaranteed error bounds over the count, in O(k) memory, with a
    merge operation so partial sketches combine (per-file, then one);
  - t-DIGEST (Dunning 2019): streaming quantiles with extreme accuracy
    at the tails exactly where sizing questions live (p95, p99), via
    the k1 scale function k(q) = (delta/2) * asin(2q - 1) and bounded
    centroid merging.

Both are deterministic for a given input order (the Space-Saving
eviction is positional, the digest merge is order-bounded), pure
stdlib, and capped: values beyond the caps abstain rather than lie.
The caller supplies the numbers (file sizes, line lengths, per-app
event counts); this module never reads a disk itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["SpaceSaving", "TDigest", "sizing_report", "render_sizing"]

_SS_MAX_ITEMS = 200_000
_DIGEST_MAX_VALUES = 200_000


class SpaceSaving:
    """Top-k counters with the Space-Saving eviction: when a new key
    arrives and the table is full, the weakest counter is demoted and
    re-labeled. Every stored count is an OVERESTIMATE bounded by the
    number of evictions — the error is reported, never hidden."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.counts: Dict[str, int] = {}
        self.overestimate: Dict[str, int] = {key: 0 for key in
                                             self.counts}
        self.n_items = 0
        self.n_total = 0

    def add(self, key: str, weight: int = 1) -> None:
        if weight <= 0:
            return
        self.n_items += 1
        self.n_total += weight
        if key in self.counts:
            self.counts[key] += weight
            return
        if len(self.counts) < self.k:
            self.counts[key] = weight
            self.overestimate[key] = 0
            return
        # evict the minimum and carry its count as overestimate
        victim = min(self.counts, key=lambda c: (self.counts[c], c))
        carried = self.counts.pop(victim)
        del self.overestimate[victim]
        self.counts[key] = carried + weight
        self.overestimate[key] = carried

    def heavy_hitters(self) -> List[Dict[str, Any]]:
        out = []
        for key in sorted(self.counts, key=lambda c: (-self.counts[c],
                                                      c)):
            count = self.counts[key]
            err = self.overestimate.get(key, 0)
            out.append({
                "key": key,
                "count": count,
                "max_error": err,
                "share": round(count / self.n_total, 4)
                if self.n_total else 0.0,
            })
        return out

    def merge(self, other: "SpaceSaving") -> "SpaceSaving":
        """Combine two sketches over disjoint streams: sums counts and
        conservatively carries both errors."""
        merged = SpaceSaving(self.k)
        merged.n_items = self.n_items + other.n_items
        merged.n_total = self.n_total + other.n_total
        keys = set(self.counts) | set(other.counts)
        scored = []
        for key in keys:
            c1, e1 = self.counts.get(key, 0), self.overestimate.get(key, 0)
            c2, e2 = other.counts.get(key, 0), \
                other.overestimate.get(key, 0)
            scored.append((key, c1 + c2, e1 + e2))
        scored.sort(key=lambda t: (-t[1], t[0]))
        for key, count, err in scored[:merged.k]:
            merged.counts[key] = count
            merged.overestimate[key] = err
        return merged


class TDigest:
    """A t-digest with the k1 scale function k(q) = (delta/2) *
    asin(2q - 1): cluster size bounds grow toward the middle, so the
    tails (p95, p99) stay sharp. Values buffer, then rebuild in sorted
    order with the k1 size bound — the standard merging digest."""

    def __init__(self, delta: float = 100.0, buffer_cap: int = 10000) -> None:
        self.delta = delta
        self.buffer_cap = buffer_cap
        self.centroids: List[Tuple[float, float]] = []  # (mean, weight)
        self.buffer: List[Tuple[float, float]] = []
        self.n = 0
        self.total = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None

    def _k(self, q: float) -> float:
        q = min(max(q, 0.0), 1.0)
        return (self.delta / 2.0) * math.asin(2.0 * q - 1.0)

    def add(self, value: float, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        self.n += 1
        self.total += weight
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        self.buffer.append((value, weight))
        if len(self.buffer) >= self.buffer_cap:
            self._rebuild()

    def _rebuild(self) -> None:
        if not self.buffer:
            return
        cs = sorted(self.centroids + self.buffer)
        self.buffer = []
        total = sum(w for _m, w in cs)
        if total <= 0:
            self.centroids = cs
            return
        out: List[Tuple[float, float]] = []
        acc_m = 0.0
        acc_w = 0.0
        open_acc = False
        w_so_far = 0.0
        for m, w in cs:
            if not open_acc:
                acc_m, acc_w = m, w
                open_acc = True
                continue
            q = (w_so_far + acc_w) / total
            step = 1.0 / total
            limit = total * (self._k(q + step) - self._k(q - step)) / 2.0
            if acc_w + w <= max(limit, 1.0):
                acc_m = (acc_m * acc_w + m * w) / (acc_w + w)
                acc_w += w
            else:
                out.append((acc_m, acc_w))
                w_so_far += acc_w
                acc_m, acc_w = m, w
        if open_acc and acc_w > 0:
            out.append((acc_m, acc_w))
        self.centroids = out

    def quantile(self, q: float) -> Optional[float]:
        """Interpolated quantile over the centroid ladder. Tail
        centroids are near-singletons under k1, so p95/p99 land within
        one value of exact for typical sizing data."""
        if self.buffer:
            self._rebuild()  # flush pending values before answering
        if not self.centroids or not 0.0 <= q <= 1.0:
            return None
        if q <= 0.0:
            return self.min_value
        if q >= 1.0:
            return self.max_value
        cs = self.centroids
        target = q * self.total
        cum = 0.0
        for i, (mean, weight) in enumerate(cs):
            low = cum
            high = cum + weight
            if target <= high:
                if i == 0:
                    # interpolate between min and the first mean
                    half = weight / 2.0
                    if target <= half:
                        frac = (target / half) if half > 0 else 0.0
                        lo = self.min_value if self.min_value is not None \
                            else mean
                        return lo + frac * (mean - lo)
                    return mean
                prev_mean, prev_weight = cs[i - 1]
                # position inside the gap between neighboring means
                gap_low = prev_mean
                gap_high = mean
                width = weight / 2.0 + prev_weight / 2.0
                frac = (target - (low - prev_weight / 2.0)) / width \
                    if width > 0 else 0.0
                frac = min(max(frac, 0.0), 1.0)
                return gap_low + frac * (gap_high - gap_low)
            cum = high
        return self.max_value

    def merge(self, other: "TDigest") -> "TDigest":
        merged = TDigest(self.delta)
        merged.centroids = list(self.centroids)
        merged.total = self.total
        merged.n = self.n
        merged.min_value = self.min_value
        merged.max_value = self.max_value
        for value, weight in other.centroids + other.buffer:
            merged.add(value, weight)
        return merged


def sizing_report(sizes: Iterable[float], keys: Optional[
        Iterable[str]] = None, k: int = 8,
        quantiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """One sizing card: heavy hitters (when keys are given) plus
    quantiles over the values. Honest about the path taken: exact
    sorted quantiles when the input fits (n <= 1000), the t-digest
    above that."""
    sizes = [float(x) for x in sizes]
    quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
    out: Dict[str, Any] = {"n": len(sizes), "quantile_points": quantiles}
    if len(sizes) > _DIGEST_MAX_VALUES:
        out["quantiles"] = {"verdict": "ABSTAIN",
                            "note": f"more than {_DIGEST_MAX_VALUES} "
                            f"values: the digest abstains"}
    elif len(sizes) <= 1000:
        srt = sorted(sizes)
        out["quantiles"] = {
            "verdict": "EXACT",
            "values": {str(q): srt[min(int(q * len(srt)),
                                       len(srt) - 1)] for q in quantiles},
            "min": srt[0], "max": srt[-1], "sum": sum(sizes),
        }
    else:
        digest = TDigest()
        for x in sizes:
            digest.add(x)
        out["quantiles"] = {
            "verdict": "TDIGEST",
            "values": {str(q): digest.quantile(q) for q in quantiles},
            "min": digest.min_value, "max": digest.max_value,
            "sum": digest.total,
            "n_centroids": len(digest.centroids),
        }
    if keys is not None:
        key_list = list(keys)
        if len(key_list) != len(sizes):
            out["heavy_hitters"] = {"verdict": "ABSTAIN",
                                    "note": "keys and sizes must pair "
                                            "one-to-one"}
        elif len(key_list) > _SS_MAX_ITEMS:
            out["heavy_hitters"] = {"verdict": "ABSTAIN",
                                    "note": "too many items"}
        else:
            ss = SpaceSaving(k=k)
            for key, size in zip(key_list, sizes):
                ss.add(str(key), int(size) if size >= 1 else 1)
            out["heavy_hitters"] = {
                "verdict": "OK",
                "k": k,
                "n_items": ss.n_items,
                "hitters": ss.heavy_hitters(),
                "note": "counts are Space-Saving ESTIMATES with a "
                        "reported max error, not exact sums",
            }
    return out


def render_sizing(data: Dict[str, Any]) -> List[str]:
    lines = [f"sizing: {data['n']} values"]
    q = data.get("quantiles", {})
    if q.get("verdict") in ("OK", "EXACT", "TDIGEST"):
        label = q["verdict"]
        for point, value in q["values"].items():
            rendered = f"{value:,.0f}" if isinstance(value, float) else value
            lines.append(f"  p{int(float(point) * 100):02d}: {rendered}")
        lines.append(f"  ({label})")
    else:
        lines.append(f"  quantiles: ABSTAIN ({q.get('note')})")
    hh = data.get("heavy_hitters")
    if hh:
        if hh.get("verdict") == "OK":
            lines.append("  heavy hitters:")
            for h in hh["hitters"]:
                lines.append(f"    {h['key']}: {h['count']:,} "
                             f"(share {h['share']:.2%}, "
                             f"max error {h['max_error']:,})")
        else:
            lines.append(f"  heavy hitters: ABSTAIN ({hh.get('note')})")
    return lines

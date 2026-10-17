"""genius.data — tabular + time-series + clustering analysis, stdlib-only.

  * CSV/TSV parsing with type inference (bool/int/float/category) and
    full column profiling (nulls, cardinality, ranges, top values)
  * frequency tables, cross-tabulation, group-by aggregation
  * correlation matrix (Pearson + Spearman) across numeric columns
  * time series: moving averages, EWMA, lag-k autocorrelation, PACF-lite,
    Yule-Walker AR(p) fitting with multi-step forecasting, additive
    trend/seasonal decomposition with robustness re-checks, CUSUM
    changepoint detection with a bootstrap significance threshold,
    Bayesian online changepoint detection (BOCPD) as the probabilistic
    complement
  * compression-based similarity: the normalized compression distance
    (NCD) over zlib/bz2 — dependency-free near-duplicate and anomaly
    scoring for any two texts/byte blobs
  * clustering: k-means with k-means++ seeding, agglomerative
    hierarchical clustering (single/complete/average linkage) with a
    cut, and silhouette-style separation scoring
"""
from __future__ import annotations

import csv
import io
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import stats as gs

__all__ = [
    "parse_table", "profile_table", "frequency_table", "crosstab",
    "groupby", "correlation_matrix", "moving_average", "ewma",
    "autocorrelation", "yule_walker_ar", "forecast_ar", "decompose",
    "decompose_robustness", "changepoints", "bocpd", "ncd",
    "kmeans", "agglomerative", "silhouette",
]


# ---------------------------------------------------------------------------
# Parsing + profiling
# ---------------------------------------------------------------------------

def parse_table(text: str, delimiter: str = ",") -> Dict[str, List[Any]]:
    """Parse CSV/TSV text into typed columns. First row = header."""
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("need a header row and at least one data row")
    header = [h.strip() for h in rows[0]]
    cols: Dict[str, List[Any]] = {h: [] for h in header}
    n_bad = 0
    for row in rows[1:]:
        if len(row) != len(header):
            n_bad += 1
            continue
        for h, raw in zip(header, row):
            cols[h].append(_coerce(raw.strip()))
    out = {"rows": len(rows) - 1 - n_bad, "skipped_rows": n_bad, "columns": header}
    out.update(cols)
    return out


def _coerce(v: str) -> Any:
    if v == "":
        return None
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if v.lstrip("+-").isdigit():
            return int(v)
        return float(v)
    except ValueError:
        return v


def profile_table(table: Dict[str, List[Any]]) -> Dict[str, Any]:
    header = table["columns"]
    n = table.get("rows", len(table[header[0]]) if header else 0)
    profile: Dict[str, Any] = {}
    for col in header:
        vals = [v for v in table[col] if v is not None]
        nulls = n - len(vals)
        kinds = {"int": 0, "float": 0, "bool": 0, "str": 0}
        for v in vals:
            kinds[type(v).__name__] = kinds.get(type(v).__name__, 0) + 1
        kind = max(kinds, key=kinds.get) if vals else "empty"
        entry: Dict[str, Any] = {
            "type": kind, "nulls": nulls, "null_rate": round(nulls / n, 4) if n else 0.0,
            "cardinality": len(set(map(str, vals))),
        }
        if kind in ("int", "float") and vals:
            entry.update({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in gs.describe([float(v) for v in vals]).items()
                          if k in ("mean", "sd", "min", "max", "median", "q1", "q3")})
            entry["outliers_iqr"] = gs.detect_outliers([float(v) for v in vals])["n_outliers"]
        elif vals:
            counts: Dict[str, int] = {}
            for v in vals:
                counts[str(v)] = counts.get(str(v), 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            entry["top_values"] = top
        profile[col] = entry
    return {"n_rows": n, "n_columns": len(header), "columns": profile}


def frequency_table(values: Sequence[Any], top: int = 10) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    n = len(values)
    return {"n": n, "distinct": len(counts),
            "top": [{"value": k, "count": c, "share": round(c / n, 4)} for k, c in items[:top]],
            "entropy_bits": round(sum(-c / n * math.log2(c / n) for c in counts.values()), 4)}


def crosstab(row_vals: Sequence[Any], col_vals: Sequence[Any]) -> Dict[str, Any]:
    if len(row_vals) != len(col_vals):
        raise ValueError("paired sequences required")
    rows = sorted({str(v) for v in row_vals})
    cols = sorted({str(v) for v in col_vals})
    table = [[0] * len(cols) for _ in rows]
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    for r, c in zip(row_vals, col_vals):
        table[ri[str(r)]][ci[str(c)]] += 1
    chi = gs.chi2_independence(table)
    return {"rows": rows, "columns": cols, "table": table, "test": chi}


def groupby(table: Dict[str, List[Any]], key: str,
            value: Optional[str] = None, agg: str = "mean") -> Dict[str, Any]:
    header = table["columns"]
    if key not in header:
        raise ValueError(f"key column {key!r} not in table")
    if value is None:
        numeric = [h for h in header if h != key and table[h]
                   and isinstance(table[h][0], (int, float))]
        value = numeric[0] if numeric else header[1] if len(header) > 1 else header[0]
    groups: Dict[str, List[float]] = {}
    for k, v in zip(table[key], table[value]):
        if v is None or k is None:
            continue
        groups.setdefault(str(k), []).append(float(v))
    fns = {
        "mean": lambda g: sum(g) / len(g),
        "sum": sum, "count": len, "min": min, "max": max,
        "sd": lambda g: gs.describe(g)["sd"] if len(g) > 1 else 0.0,
        "median": lambda g: gs.quantile(g, 0.5),
    }
    if agg not in fns:
        raise ValueError(f"unknown agg {agg!r} (mean|sum|count|min|max|sd|median)")
    out = {k: round(fns[agg](g), 6) if isinstance(fns[agg](g), float) else fns[agg](g)
           for k, g in groups.items()}
    return {"by": key, "value": value, "agg": agg, "groups": out,
            "n_groups": len(out)}


def correlation_matrix(table: Dict[str, List[Any]],
                       method: str = "pearson") -> Dict[str, Any]:
    header = table["columns"]
    numeric = [h for h in header if table.get(h) and
               all(v is None or isinstance(v, (int, float)) for v in table[h][:50])]
    numeric = [h for h in numeric if len([v for v in table[h] if v is not None]) >= 3]
    if len(numeric) < 2:
        raise ValueError("need >= 2 numeric columns")
    cols = {h: [float(v) for v in table[h] if v is not None] for h in numeric}
    n = min(len(c) for c in cols.values())
    cols = {h: c[:n] for h, c in cols.items()}
    corr: Dict[str, Dict[str, Any]] = {h: {} for h in numeric}
    for i, a in enumerate(numeric):
        corr[a][a] = 1.0
        for b in numeric[i + 1:]:
            try:
                res = (gs.pearson(cols[a], cols[b]) if method == "pearson"
                       else gs.spearman(cols[a], cols[b]))
                r = res.get("r", res.get("rho"))
            except ValueError:
                r = None
            corr[a][b] = round(r, 4) if r is not None else None
            corr[b][a] = corr[a][b]
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            if corr[a][b] is not None:
                pairs.append({"a": a, "b": b, "r": corr[a][b]})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"method": method, "variables": numeric, "matrix": corr,
            "strongest": pairs[:5],
            "note": "correlation is not causation — a ranked pair is a lead, not a verdict"}


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def moving_average(series: Sequence[float], window: int) -> Dict[str, Any]:
    if window < 1 or window > len(series):
        raise ValueError("window must be in [1, n]")
    s = [float(v) for v in series]
    out = []
    running = sum(s[:window])
    out.append(running / window)
    for i in range(window, len(s)):
        running += s[i] - s[i - window]
        out.append(running / window)
    return {"window": window, "smoothed": [round(v, 6) for v in out],
            "aligned_at": list(range(window - 1, len(s)))}


def ewma(series: Sequence[float], alpha: float = 0.3) -> Dict[str, Any]:
    if not 0 < alpha <= 1:
        raise ValueError("alpha in (0,1]")
    s = [float(v) for v in series]
    out = [s[0]]
    for v in s[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return {"alpha": alpha, "smoothed": [round(v, 6) for v in out]}


def autocorrelation(series: Sequence[float], max_lag: int = 10) -> Dict[str, Any]:
    s = [float(v) for v in series]
    n = len(s)
    if n < 3:
        raise ValueError("need n >= 3")
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    if var == 0:
        return {"acf": [1.0] + [0.0] * min(max_lag, n - 1), "variance_zero": True}
    acf = []
    for lag in range(0, min(max_lag, n - 1) + 1):
        num = sum((s[i] - mean) * (s[i + lag] - mean) for i in range(n - lag))
        acf.append(round(num / (n * var), 6))
    se = 1.96 / math.sqrt(n)
    significant = [i for i, v in enumerate(acf) if i > 0 and abs(v) > se]
    return {"acf": acf, "se_band": round(se, 4), "significant_lags": significant,
            "interpretation": ("strong lag-1 persistence" if len(acf) > 1 and acf[1] > 0.5
                              else "weak persistence" if len(acf) > 1 and acf[1] > 0
                              else "no positive autocorrelation")}


def _levinson_durbin(r: Sequence[float]) -> Tuple[List[float], float]:
    """Levinson-Durbin recursion from autocorrelations r[0..p]."""
    p = len(r) - 1
    if p < 1:
        return [], r[0]
    a = [0.0] * (p + 1)
    e = r[0]
    for k in range(1, p + 1):
        acc = r[k]
        for j in range(1, k):
            acc -= a[j] * r[k - j]
        if e == 0:
            raise ValueError("degenerate autocorrelation")
        ak = acc / e
        for j in range(1, k):
            a[j] = a[j] - ak * a[k - j]
        a[k] = ak
        e = e * (1 - ak * ak)
    return a[1:], e


def yule_walker_ar(series: Sequence[float], order: int = 2) -> Dict[str, Any]:
    """Fit AR(p) by Yule-Walker (via the Levinson-Durbin recursion)."""
    s = [float(v) for v in series]
    n = len(s)
    if n < order + 3:
        raise ValueError(f"AR({order}) needs n >= order+3")
    mean = sum(s) / n
    d = [x - mean for x in s]
    var = sum(x * x for x in d) / n
    r = [1.0]
    for lag in range(1, order + 1):
        r.append(sum(d[i] * d[i + lag] for i in range(n - lag)) / (n * var) if var else 0.0)
    phi, sigma2 = _levinson_durbin(r)
    return {"order": order, "mean": mean, "phi": [round(v, 6) for v in phi],
            "innovation_var": sigma2 * var if var else 0.0,
            "r_squared_proxy": round(1 - sigma2, 4),
            "n": n}


def forecast_ar(series: Sequence[float], horizon: int = 5,
                 order: int = 2) -> Dict[str, Any]:
    fit = yule_walker_ar(series, order)
    phi = fit["phi"]
    mean = fit["mean"]
    s = [float(v) for v in series]
    history = [v - mean for v in s]
    path: List[float] = []
    for _ in range(horizon):
        nxt = sum(phi[k] * history[-(k + 1)] for k in range(len(phi)))
        history.append(nxt)
        path.append(mean + nxt)
    # honest uncertainty grows with horizon
    spread = gs.describe(s)["sd"] if len(s) > 1 else 0.0
    bands = []
    for h in range(1, horizon + 1):
        w = spread * math.sqrt(0.25 * h)  # conservative widening
        bands.append([path[h - 1] - 1.96 * w, path[h - 1] + 1.96 * w])
    return {"history": s, "horizon": horizon, "forecast": [round(v, 6) for v in path],
            "ci95": [[round(lo, 6), round(hi, 6)] for lo, hi in bands],
            "fit": fit, "note": "AR forecast reverts toward the mean; bands widen with horizon"}


def decompose(series: Sequence[float], period: int) -> Dict[str, Any]:
    """Classical additive decomposition: trend (MA) + seasonal means + residual."""
    s = [float(v) for v in series]
    n = len(s)
    if period < 2 or n < 2 * period:
        raise ValueError("need n >= 2*period for a seasonal decomposition")
    half = period // 2
    trend = [None] * n
    for i in range(half, n - half):
        window = s[max(0, i - half):i + half + 1]
        trend[i] = sum(window) / len(window)
    detrended = [(s[i] - trend[i]) if trend[i] is not None else None for i in range(n)]
    seasonal_means: Dict[int, List[float]] = {}
    for i, v in enumerate(detrended):
        if v is not None:
            seasonal_means.setdefault(i % period, []).append(v)
    seasonal_cycle = [sum(v) / len(v) for _, v in sorted(seasonal_means.items())]
    seasonal = [seasonal_cycle[i % period] for i in range(n)]
    resid = [round(s[i] - (trend[i] or _interp(trend, i)) - seasonal[i], 6) for i in range(n)]
    return {"period": period, "trend": [None if t is None else round(t, 6) for t in trend],
            "seasonal": [round(v, 6) for v in seasonal],
            "residual": resid,
            "residual_sd": round(gs.describe([r for r in resid])["sd"], 6)}


def _interp(vals: List[Optional[float]], i: int) -> float:
    for d in range(1, len(vals)):
        lo, hi = i - d, i + d
        if lo >= 0 and vals[lo] is not None:
            return vals[lo]  # type: ignore[index]
        if hi < len(vals) and vals[hi] is not None:
            return vals[hi]  # type: ignore[index]
    return 0.0


def changepoints(series: Sequence[float], drift: float = 0.5, h: float = 5.0,
                 n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Standardized two-sided CUSUM with a bootstrap significance call.

    S+ = max(0, S+ + z_i - k),  S- = max(0, S- - z_i - k)  with z the
    standardized series, k the drift allowance and h the decision interval
    (5 is the classic tabulated choice). A CUSUM crossing h marks a
    changepoint and the accumulators reset.
    """
    s = [float(v) for v in series]
    n = len(s)
    if n < 8:
        raise ValueError("need n >= 8")
    mean = sum(s) / n
    sd = gs.describe(s)["sd"] or 1.0
    cusum_pos = cusum_neg = 0.0
    stat_path: List[float] = []
    detected: List[int] = []
    for i, v in enumerate(s):
        z = (v - mean) / sd
        cusum_pos = max(0.0, cusum_pos + z - drift)
        cusum_neg = max(0.0, cusum_neg - z - drift)
        stat = max(cusum_pos, cusum_neg)
        stat_path.append(round(stat, 4))
        if stat > h:
            detected.append(i)
            cusum_pos = cusum_neg = 0.0
    # bootstrap: max CUSUM of drift-penalized random walks of the same
    # length — how often would noise alone reach the observed maximum?
    rng = random.Random(seed)
    obs_max = max(stat_path) if stat_path else 0.0
    exceed = 0
    for _ in range(n_boot):
        walk_p = walk_n = 0.0
        mx = 0.0
        for _v in range(n):
            z = rng.gauss(0.0, 1.0)
            walk_p = max(0.0, walk_p + z - drift)
            walk_n = max(0.0, walk_n - z - drift)
            mx = max(mx, walk_p, walk_n)
        if mx >= obs_max:
            exceed += 1
    p = (exceed + 1) / (n_boot + 1)
    # The bootstrap is the significance arbiter; a fixed h alone is brittle
    # when mean/sd are computed over a series that contains the very shift
    # we are hunting (which deflates the standardized evidence).
    if p < 0.05 and not detected and obs_max > 2.0:
        peak = max(range(n), key=lambda i: stat_path[i])
        detected.append(peak)
        significant = True
    else:
        significant = p < 0.05 and bool(detected)
    return {"cusum_path": stat_path, "drift": drift, "h": h,
            "changepoints": detected, "n_changepoints": len(detected),
            "max_cusum": obs_max, "bootstrap_p": round(p, 4),
            "verdict": ("real shifts (bootstrap significant)" if significant
                        else "no significant shift beyond noise")}


def startup_regressions(series: Sequence[float], timestamps: Optional[Sequence[str]] = None,
                        labels: Optional[Sequence[str]] = None,
                        drift: float = 0.5, h: float = 5.0,
                        regression_ratio: float = 0.15) -> Dict[str, Any]:
    """Issue #120 Phase 3.3: startup-time regression detection over the
    EXISTING CUSUM changepoint detector (:func:`changepoints`, above —
    no duplicate detector here).

    A changepoint is a REGRESSION when the post-segment mean startup time
    is >= ``regression_ratio`` (default 15%) higher than the pre-segment
    mean, an improvement when it is <= -15%, otherwise a neutral level
    shift. Dates/versions come only from the input: pass ISO
    ``timestamps`` (one per sample) and/or ``labels`` (e.g. shell/package
    versions) and each event reports the stamp/label of the first sample
    AFTER the changepoint — nothing is invented about when or why.

    Pure and read-only. Series are caller-supplied (a boot-time list from
    `systemd-analyze` exports, the user's own logs, or a future telemetry
    feed); this module has no startup-time source of its own — that is
    the honesty rule every Phase 3 feature follows.
    """
    base = changepoints(series, drift=drift, h=h)
    s = [float(v) for v in series]
    n = len(s)
    events: List[Dict[str, Any]] = []
    for cp in base["changepoints"]:
        pre, post = s[:cp], s[cp:]
        pre_mean = sum(pre) / len(pre) if pre else 0.0
        post_mean = sum(post) / len(post) if post else 0.0
        delta = (post_mean - pre_mean) / pre_mean if pre_mean else 0.0
        if delta >= regression_ratio:
            kind = "regression"
        elif delta <= -regression_ratio:
            kind = "improvement"
        else:
            kind = "level shift"
        event: Dict[str, Any] = {
            "index": cp,
            "kind": kind,
            "pre_mean": round(pre_mean, 2),
            "post_mean": round(post_mean, 2),
            "delta_pct": round(100.0 * delta, 1),
        }
        if timestamps and cp < len(timestamps):
            event["at"] = str(timestamps[cp])
        if labels and cp < len(labels):
            event["version"] = str(labels[cp])
        events.append(event)
    return {
        "n": n,
        "verdict": base["verdict"],
        "bootstrap_p": base["bootstrap_p"],
        "regressions": [e for e in events if e["kind"] == "regression"],
        "events": events,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(points: Sequence[Sequence[float]], k: int, iterations: int = 100,
           seed: int = 42) -> Dict[str, Any]:
    pts = [[float(x) for x in p] for p in points]
    n = len(pts)
    if k < 1 or k > n:
        raise ValueError("k must be in [1, n]")
    rng = random.Random(seed)
    # k-means++ seeding (deterministic under the seed)
    centers = [pts[rng.randrange(n)]]
    while len(centers) < k:
        dists = []
        for p in pts:
            d = min(_dist(p, c) for c in centers)
            dists.append(d * d)
        total = sum(dists)
        if total == 0:
            centers.append(pts[rng.randrange(n)])
            continue
        u, acc = rng.random() * total, 0.0
        for i, d in enumerate(dists):
            acc += d
            if acc >= u:
                centers.append(pts[i])
                break
        else:
            centers.append(pts[-1])
    labels = [0] * n
    for _ in range(iterations):
        changed = False
        for i, p in enumerate(pts):
            labels[i] = min(range(k), key=lambda c: _dist(p, centers[c]))
            changed = changed or labels[i] != labels[i]
        new_centers = []
        for c in range(k):
            members = [pts[i] for i in range(n) if labels[i] == c]
            if members:
                new_centers.append([sum(col) / len(members) for col in zip(*members)])
            else:
                new_centers.append(centers[c])
        moved = sum(_dist(a, b) for a, b in zip(centers, new_centers))
        centers = new_centers
        if moved < 1e-12:
            break
    sse = sum(_dist(pts[i], centers[labels[i]]) ** 2 for i in range(n))
    sizes = [0] * k
    for l in labels:
        sizes[l] += 1
    return {"k": k, "centers": [[round(v, 6) for v in c] for c in centers],
            "labels": labels, "sizes": sizes, "sse": round(sse, 6),
            "empty_clusters": sizes.count(0)}


def agglomerative(points: Sequence[Sequence[float]], k: int,
                  linkage: str = "average") -> Dict[str, Any]:
    pts = [[float(x) for x in p] for p in points]
    n = len(pts)
    if k < 1 or k > n:
        raise ValueError("k must be in [1, n]")
    clusters = [[i] for i in range(n)]
    merge_log = []
    while len(clusters) > k:
        best_pair, best_d = (0, 1), float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _linkage_dist(pts, clusters[i], clusters[j], linkage)
                if d < best_d:
                    best_d, best_pair = d, (i, j)
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        merge_log.append({"merged_size": len(clusters[i]), "distance": round(best_d, 6),
                          "clusters_left": len(clusters) - 1})
        del clusters[j]
    labels = [0] * n
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return {"k": k, "linkage": linkage, "labels": labels,
            "sizes": [len(c) for c in clusters],
            "members": [sorted(c) for c in clusters],
            "merge_log": merge_log[-6:]}


def _linkage_dist(pts: List[List[float]], a: List[int], b: List[int],
                   linkage: str) -> float:
    dists = [_dist(pts[i], pts[j]) for i in a for j in b]
    if linkage == "single":
        return min(dists)
    if linkage == "complete":
        return max(dists)
    return sum(dists) / len(dists)


def silhouette(points: Sequence[Sequence[float]], labels: Sequence[int],
               sample_cap: int = 500, seed: int = 42) -> Dict[str, Any]:
    pts = [[float(x) for x in p] for p in points]
    n = len(pts)
    if n != len(labels) or n < 3 or len(set(labels)) < 2:
        raise ValueError("need n>=3 points, >=2 clusters, labels aligned")
    rng = random.Random(seed)
    idx = list(range(n))
    if n > sample_cap:
        idx = rng.sample(idx, sample_cap)
    scores = []
    per_cluster: Dict[int, List[float]] = {}
    for i in idx:
        own = [j for j in idx if labels[j] == labels[i] and j != i]
        if not own:
            s = 0.0
        else:
            a = sum(_dist(pts[i], pts[j]) for j in own) / len(own)
            b = min(sum(_dist(pts[i], pts[j]) for j in idx if labels[j] == c) /
                    max(1, sum(1 for j in idx if labels[j] == c))
                    for c in set(labels) if c != labels[i])
            s = (b - a) / max(a, b) if max(a, b) else 0.0
        scores.append(s)
        per_cluster.setdefault(labels[i], []).append(s)
    mean_s = sum(scores) / len(scores)
    return {"mean_silhouette": round(mean_s, 4),
            "by_cluster": {c: round(sum(v) / len(v), 4) for c, v in per_cluster.items()},
            "interpretation": ("well separated" if mean_s > 0.5 else
                               "reasonable" if mean_s > 0.25 else "weak/no structure"),
            "note": "silhouette near 0 means overlapping clusters — that is honest, not a bug"}


# ---------------------------------------------------------------------------
# Phase 2.2 additions: NCD, BOCPD, decomposition robustness.
# ---------------------------------------------------------------------------


def ncd(a: bytes, b: bytes, compressor: str = "zlib") -> Dict[str, Any]:
    """Normalized Compression Distance (Li, Chen, Li, Ma & Vitanyi 2004,
    "The similarity metric", IEEE Trans. Inform. Theory 50(12)):

        NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    with C = compressed size. Two identical inputs give ~0; unrelated
    inputs ~1 (bounded with small clamps because real compressors are
    not ideal). zlib (level 6) is the default; bz2 is available for a
    second opinion. Str inputs are encoded utf-8. Pure, deterministic,
    dependency-free — the compressor IS the similarity model, which is
    exactly why it belongs in a no-ML assistant.
    """
    import zlib
    import bz2

    if isinstance(a, str):
        a = a.encode("utf-8", "replace")
    if isinstance(b, str):
        b = b.encode("utf-8", "replace")

    def c(blob: bytes) -> int:
        if compressor == "zlib":
            return len(zlib.compress(blob, 6))
        if compressor == "bz2":
            return len(bz2.compress(blob, 5))
        raise ValueError(f"unknown compressor {compressor!r} (zlib|bz2)")

    ca, cb, cab = c(a), c(b), c(a + b)
    denom = max(ca, cb)
    if denom == 0:  # two empty inputs: identical by definition
        return {"ncd": 0.0, "compressor": compressor,
                "sizes": {"a": 0, "b": 0, "ab": 0},
                "note": "both inputs empty"}
    value = (cab - min(ca, cb)) / denom
    value = min(1.0, max(0.0, value))  # clamped: real compressors overshoot
    verdict = ("near-identical" if value < 0.10 else
               "similar" if value < 0.35 else
               "dissimilar" if value < 0.65 else "unrelated")
    return {"ncd": round(value, 4), "compressor": compressor,
            "sizes": {"a": ca, "b": cb, "ab": cab},
            "verdict": verdict,
            "algorithm": "NCD (Li et al. 2004), C = zlib/bz2 compressed size"}


def bocpd(series: Sequence[float], hazard: float = 100.0,
          variance: Optional[float] = None,
          prior_mean: Optional[float] = None) -> Dict[str, Any]:
    """Bayesian online changepoint detection (Adams & MacKay 2007,
    "Bayesian online changepoint detection", arXiv:0710.3742v2 [stat.ML]).

    The run-length posterior recursion with a Gaussian observation model
    (known variance, conjugate normal prior over the segment mean):

        growth:   p(r_t = r_{t-1}+1, x_t) = p(x_t | D_{t-1}^r) * (1-H)
        changept: p(r_t = 0, x_t)        = p(x_t | prior)   * H
        H(r) = 1/hazard   (the constant-hazard choice from the paper)

    Output per step: the MAP run length, the changepoint probability
    p(r_t = 0 | x_1..t), and the posterior mean of the current segment.
    This is the PROBABILISTIC complement to the CUSUM detector above —
    CUSUM answers "did the mean shift" with a bootstrap significance
    call; BOCPD answers "how likely is THIS step a boundary, under a
    generative model". Deterministic, O(n^2) worst case in the run
    length (n small in practice), no RNG.
    """
    import math as _math

    s = [float(v) for v in series]
    n = len(s)
    if n < 4:
        raise ValueError("need n >= 4 observations")
    if hazard <= 0:
        raise ValueError("hazard (expected run length) must be > 0")
    sample_mean = sum(s) / n
    if variance is None:
        # robust noise-scale default: the variance of successive first
        # differences / 2 — insensitive to the level shifts we are hunting
        # (a shift contributes ONE large diff, not a shifted distribution),
        # the standard changepoint-layer trick
        diffs = [s[i + 1] - s[i] for i in range(n - 1)]
        diff_var = gs.describe(diffs).get("variance") or 0.0
        variance = max(diff_var / 2.0, 1e-6)
    else:
        variance = float(variance)
    mu0 = float(prior_mean) if prior_mean is not None else sample_mean
    # weak prior precision: half an observation of the prior mean
    prior_tau = 0.5 / variance

    H = 1.0 / hazard
    # run-length posterior at t: list index = run length r
    probs: List[float] = [1.0]  # r=0 with probability 1 at t=0
    means: List[float] = [mu0]  # posterior mean of the segment for each r
    taus: List[float] = [prior_tau]

    def predictive(x: float, mu: float, tau: float) -> float:
        # Student-t-ish normal with posterior variance 1/tau; the simple
        # normal approximation keeps the recursion transparent and the
        # MAP structure identical to the paper's Gaussian example
        var = variance + 1.0 / max(tau, 1e-12)
        return _math.exp(-0.5 * (x - mu) ** 2 / var) / _math.sqrt(
            2.0 * _math.pi * var)

    map_runlengths: List[int] = []
    cp_probs: List[float] = []
    segment_means: List[float] = []
    for t in range(1, n):
        x = s[t]
        # evaluate the predictive under every current run length
        likelihoods = [predictive(x, means[r], taus[r]) for r in range(len(probs))]
        # growth probabilities (r -> r+1): the segment survives
        growth = [(1.0 - H) * probs[r] * likelihoods[r]
                  for r in range(len(probs))]
        # changepoint arm (r -> 0): the OBSERVATION is drawn from the
        # PRIOR predictive, not from any surviving segment — the segment
        # before a break is discarded, which is the whole point of the
        # Adams & MacKay recursion
        prior_predictive = predictive(x, mu0, prior_tau)
        cp_mass = H * sum(probs) * prior_predictive
        new_probs = [cp_mass] + growth
        evidence = sum(new_probs) or 1e-300
        probs = [p / evidence for p in new_probs]
        # update the observation model per run length (conjugate normal)
        new_means = [mu0]
        new_taus = [prior_tau]
        for r in range(len(means)):
            tau_old = taus[r]
            mu_old = means[r]
            tau_new = tau_old + 1.0 / variance
            mu_new = (mu_old * tau_old + x / variance) / tau_new
            new_means.append(mu_new)
            new_taus.append(tau_new)
        means, taus = new_means, new_taus
        # report: MAP run length, p(r=0), current segment mean
        best_r = max(range(len(probs)), key=lambda r: (probs[r], -r))
        map_runlengths.append(best_r)
        cp_probs.append(round(probs[0], 4))
        segment_means.append(round(means[best_r], 4))

    changepoints = [t for t in range(1, n)
                    if map_runlengths[t - 1] <= 1 and cp_probs[t - 1] >= 0.3]
    return {"n": n, "hazard": hazard,
            "map_run_length": map_runlengths,
            "changepoint_prob": cp_probs,
            "segment_mean": segment_means,
            "changepoints": changepoints,
            "algorithm": ("BOCPD (Adams & MacKay 2007), constant hazard, "
                          "conjugate normal segment model; the "
                          "probabilistic complement to CUSUM")}


def decompose_robustness(series: Sequence[float], period: int) -> Dict[str, Any]:
    """Robustness checks on the additive STL-lite decomposition.

    The shipped ``decompose`` is a moving-average trend + seasonal means.
    The honest question for any decomposition is whether its pieces
    survive reasonable perturbation of its assumptions, so this runs
    the SAME estimator under neighbor periods (period-1, period,
    period+1) and under a lightly trimmed detrended pass, then reports
    agreement:

    - ``trend_agreement``: fraction of sampled points where the variant
      trends move in the same direction as the base trend between
      consecutive samples (sign agreement);
    - ``seasonal_amplitude_stability``: the spread (relative sd) of the
      seasonal swing (max-min of the cycle) across variants — an
      unstable swing means the seasonal component is mostly noise;
    - ``residual_sd_stability``: same spread for the residual sd;
    - a plain-language ``verdict`` ("robust" / "period-sensitive" /
      "fragile") with the numbers behind it.

    Read-only analysis of the caller's series; deterministic.
    """
    s = [float(v) for v in series]
    variants: Dict[str, Dict[str, Any]] = {}
    for name, p in (("base", period), ("period_minus_1", period - 1),
                    ("period_plus_1", period + 1)):
        if p < 2 or len(s) < 2 * p:
            continue
        variants[name] = decompose(s, p)

    if len(variants) < 2:
        raise ValueError("series too short for a robustness re-check "
                         "(need 2*(period+1) points)")

    base = variants.get("base") or list(variants.values())[0]

    def trimmed_pass() -> Optional[Dict[str, Any]]:
        # re-run the seasonal estimate with the top/bottom 10% of
        # detrended values per phase dropped (a light robustness trim)
        if "base" not in variants:
            return None
        n = len(s)
        period_b = period
        detrended = base["residual"]  # s - trend - seasonal, but we want
        # s - trend: recompute honestly from the base trend
        trend = base["trend"]
        detrended = [s[i] - (trend[i] if trend[i] is not None
                             else 0.0) for i in range(n)]
        by_phase: Dict[int, List[float]] = {}
        for i, v in enumerate(detrended):
            by_phase.setdefault(i % period_b, []).append(v)
        trimmed_cycle: List[float] = []
        for phase in range(period_b):
            values = sorted(by_phase.get(phase, []))
            if not values:
                trimmed_cycle.append(0.0)
                continue
            cut = max(1, len(values) // 10)
            kept = values[cut:-cut] or values
            trimmed_cycle.append(sum(kept) / len(kept))
        seasonal = [trimmed_cycle[i % period_b] for i in range(n)]
        resid = [s[i] - (trend[i] if trend[i] is not None else 0.0)
                 - seasonal[i] for i in range(n)]
        return {"seasonal": seasonal,
                "residual_sd": gs.describe(resid)["sd"]}

    trimmed = trimmed_pass()

    # trend sign agreement between the base and each period variant,
    # with a flatness epsilon: a step whose trend movement is under
    # 5% of the series sd is "flat", and flat-vs-anything-flat agrees —
    # comparing floating noise around a constant trend is not evidence.
    series_sd = gs.describe(s)["sd"] or 1.0
    eps = 0.05 * series_sd
    trend = [t for t in base["trend"] if t is not None]
    agreements: Dict[str, float] = {}
    for name, variant in variants.items():
        if name == "base":
            continue
        other = [t for t in variant["trend"] if t is not None]
        m = min(len(trend), len(other))
        steps = [(trend[i + 1] - trend[i], other[i + 1] - other[i])
                 for i in range(m - 1)]
        def _agrees(a: float, b: float) -> bool:
            a_flat, b_flat = abs(a) <= eps, abs(b) <= eps
            if a_flat and b_flat:
                return True
            if a_flat or b_flat:
                return False
            return (a > 0) == (b > 0)
        same = sum(1 for a, b in steps if _agrees(a, b))
        agreements[name] = round(same / max(1, len(steps)), 3)

    def swing(vals: Sequence[float]) -> float:
        return max(vals) - min(vals) if vals else 0.0

    swings = [swing(v["seasonal"]) for v in variants.values()]
    if trimmed is not None:
        swings.append(swing(trimmed["seasonal"]))
    mean_swing = sum(swings) / len(swings) if swings else 0.0
    amp_stability = round(
        (gs.describe(swings)["sd"] / mean_swing) if mean_swing > 1e-12 else 0.0, 3)

    sds = [v["residual_sd"] for v in variants.values()]
    if trimmed is not None and trimmed.get("residual_sd") is not None:
        sds.append(float(trimmed["residual_sd"]))
    sd_stability = round(
        (gs.describe(sds)["sd"] / abs(gs.describe(sds)["mean"]))
        if any(sds) else 0.0, 3)

    weakest = min(agreements.values(), default=1.0)
    if weakest >= 0.8 and amp_stability < 0.25:
        verdict = "robust"
    elif weakest >= 0.45 and amp_stability < 0.6:
        verdict = "period-sensitive"
    else:
        verdict = "fragile"
    return {"period": period,
            "trend_agreement": agreements,
            "seasonal_amplitude_stability": amp_stability,
            "residual_sd_stability": sd_stability,
            "trimmed_pass_included": trimmed is not None,
            "verdict": verdict,
            "algorithm": ("STL-lite robustness: same estimator under "
                          "period-1/period+1 and a 10%-trimmed seasonal "
                          "pass; sign-agreement + relative-sd reporting"),
            "note": "read-only analysis; verdicts carry their numbers"}

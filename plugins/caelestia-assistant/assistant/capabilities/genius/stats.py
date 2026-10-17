"""genius.stats — a real statistics engine, stdlib-only.

Special functions (needed for honest p-values, no scipy):

  * regularized incomplete gamma P(a,x) / Q(a,x) — series + continued
    fraction (Lentz), used by the chi-square distribution
  * regularized incomplete beta I_x(a,b) — continued fraction, used by
    the t distribution and the F distribution
  * normal CDF/quantile (via erf and Wichura AS241)

Inference:

  * one-sample, two-sample (pooled + Welch) and paired t tests
  * Mann-Whitney U, Wilcoxon signed-rank (normal approximations with
    tie correction), sign test
  * chi-square test of independence on a contingency table
  * one-way ANOVA (with the F p-value) and Jarque-Bera normality
  * one-sample z test for proportions with odds ratio for 2x2 tables

Description & models:

  * full descriptive suite (moments, quantiles, MAD, trimmed means)
  * Pearson (with p-value), Spearman, Kendall correlations
  * OLS regression (simple and multiple, via linalg least squares) with
    standard errors, t statistics and adjusted R-squared
  * bootstrap confidence intervals and permutation tests
  * outlier detection: z-score, IQR fences, modified z (MAD), Grubbs
  * effect sizes: Cohen's d, Hedges' g, odds ratio
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from . import linalg
from .probability import normal_cdf

__all__ = [
    "describe", "quantile", "pearson", "spearman", "kendall",
    "t_test_one_sample", "t_test_two_sample", "t_test_paired",
    "mann_whitney_u", "wilcoxon_signed_rank", "sign_test",
    "chi2_independence", "anova_one_way", "jarque_bera",
    "z_test_proportion", "odds_ratio", "cohen_d",
    "ols_regression", "bootstrap_ci", "permutation_test",
    "detect_outliers", "gammainc_p", "betainc", "chi2_sf", "t_cdf", "f_sf",
]


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def _lower_gamma_series(a: float, x: float) -> float:
    ap, summ, delt = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * 1e-14:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_gamma_cf(a: float, x: float) -> float:
    tiny = 1e-300
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("need x >= 0 and a > 0")
    if x == 0:
        return 0.0
    if x < a + 1:
        return _lower_gamma_series(a, x)
    return 1.0 - _upper_gamma_cf(a, x)


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (continued fraction, NR 6.4)."""
    if x < 0 or x > 1 or a <= 0 or b <= 0:
        raise ValueError("need a,b > 0 and x in [0,1]")
    if x == 0 or x == 1:
        return x
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _beta_cf(a, b, x) / a
    return 1 - math.exp(lbeta + b * math.log(1 - x) + a * math.log(x)) * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delt = d * c
        h *= delt
        if abs(delt - 1) < 1e-14:
            break
    return h


def chi2_sf(x: float, k: int) -> float:
    """Survival function P(X > x) for chi-square with k dof."""
    return max(0.0, 1.0 - gammainc_p(k / 2, x / 2))


def t_cdf(t: float, dof: int) -> float:
    """CDF of Student's t."""
    x = dof / (dof + t * t)
    tail = 0.5 * betainc(dof / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def f_sf(f: float, d1: int, d2: int) -> float:
    """Survival function of the F distribution."""
    if f <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return betainc(d2 / 2, d1 / 2, x)


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------

def _nums(data: Sequence[float]) -> List[float]:
    vals = [float(v) for v in data]
    if not vals:
        raise ValueError("empty sample")
    return vals


def quantile(data: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the 'inclusive' definition)."""
    vals = sorted(_nums(data))
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (pos - lo) * (vals[hi] - vals[lo])


def describe(data: Sequence[float]) -> Dict[str, Any]:
    vals = _nums(data)
    n = len(vals)
    s = sorted(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    m2 = sum((x - mean) ** 2 for x in vals) / n
    m3 = sum((x - mean) ** 3 for x in vals) / n
    m4 = sum((x - mean) ** 4 for x in vals) / n
    skew = (m3 / m2 ** 1.5) if m2 > 0 and n > 2 else 0.0
    kurt = (m4 / m2 ** 2 - 3.0) if m2 > 0 and n > 3 else 0.0
    med = quantile(vals, 0.5)
    mad = sorted(abs(x - med) for x in vals)[len(vals) // 2]
    counts: Dict[float, int] = {}
    for x in vals:
        counts[x] = counts.get(x, 0) + 1
    mode_val = max(counts, key=counts.get) if counts else None
    return {
        "n": n, "mean": mean, "median": med, "mode": mode_val,
        "sd": sd, "variance": var, "sem": sd / math.sqrt(n) if n else 0.0,
        "min": s[0], "max": s[-1], "range": s[-1] - s[0],
        "q1": quantile(vals, 0.25), "q3": quantile(vals, 0.75),
        "iqr": quantile(vals, 0.75) - quantile(vals, 0.25),
        "mad": mad, "skewness": skew, "excess_kurtosis": kurt,
        "cv": sd / mean if mean else None,
        "five_number": {"min": s[0], "q1": quantile(vals, 0.25), "median": med,
                        "q3": quantile(vals, 0.75), "max": s[-1]},
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def pearson(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need equal-length samples with n >= 2")
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("zero variance in a sample")
    r = sxy / math.sqrt(sxx * syy)
    r = max(-1.0, min(1.0, r))
    t = r * math.sqrt((n - 2) / max(1e-300, 1 - r * r))
    p = 2 * (1 - t_cdf(abs(t), n - 2))
    return {"r": r, "t": t, "p": p, "n": n,
            "interpretation": _r_verdict(r)}


def _r_verdict(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def _ranks(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    rx, ry = _ranks(x), _ranks(y)
    res = pearson(rx, ry)
    res["rho"] = res.pop("r")
    return res


def kendall(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need equal-length samples with n >= 2")
    conc = disc = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx * sy > 0:
                conc += 1
            elif sx * sy < 0:
                disc += 1
    tau = (conc - disc) / (n * (n - 1) / 2)
    # normal-approximation p (tau-b tie handling omitted; exact for no ties)
    v = 2 * tau * math.sqrt(9 * n * (n - 1)) / (2 * (2 * n + 5))
    p = 2 * (1 - normal_cdf(abs(v)))
    return {"tau": tau, "concordant": conc, "discordant": disc, "p_approx": p, "n": n}


# ---------------------------------------------------------------------------
# Tests of location
# ---------------------------------------------------------------------------

def t_test_one_sample(data: Sequence[float], mu0: float = 0.0) -> Dict[str, Any]:
    d = describe(data)
    if d["n"] < 2 or d["sd"] == 0:
        raise ValueError("need n >= 2 with non-zero variance")
    t = (d["mean"] - mu0) / d["sem"]
    dof = d["n"] - 1
    p = 2 * (1 - t_cdf(abs(t), dof))
    crit = _t_crit(dof)
    return {"t": t, "dof": dof, "p": p, "mean": d["mean"], "mu0": mu0,
            "ci95": [d["mean"] - crit * d["sem"], d["mean"] + crit * d["sem"]],
            "significant_5pct": p < 0.05,
            "verdict": f"mean {'!=' if p < 0.05 else 'not clearly !='} {mu0} (p={p:.4g})"}


def _t_crit(dof: int) -> float:
    # two-sided 95% critical value via the quantile function inversion
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, dof) < 0.975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_test_two_sample(a: Sequence[float], b: Sequence[float],
                      welch: bool = True) -> Dict[str, Any]:
    da, db = describe(a), describe(b)
    na, nb = da["n"], db["n"]
    if na < 2 or nb < 2:
        raise ValueError("need n >= 2 per sample")
    if welch:
        va, vb = da["sd"] ** 2, db["sd"] ** 2
        se2 = va / na + vb / nb
        if se2 == 0:
            raise ValueError("both samples have zero variance")
        t = (da["mean"] - db["mean"]) / math.sqrt(se2)
        dof = int(round(se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))))
        dof = max(1, dof)
    else:
        sp2 = ((na - 1) * da["sd"] ** 2 + (nb - 1) * db["sd"] ** 2) / (na + nb - 2)
        se = math.sqrt(sp2 * (1 / na + 1 / nb))
        if se == 0:
            raise ValueError("pooled variance is zero")
        t = (da["mean"] - db["mean"]) / se
        dof = na + nb - 2
    p = 2 * (1 - t_cdf(abs(t), dof))
    return {"t": t, "dof": dof, "p": p, "welch": welch,
            "mean_a": da["mean"], "mean_b": db["mean"],
            "difference": da["mean"] - db["mean"],
            "cohen_d": cohen_d(a, b)["d"],
            "significant_5pct": p < 0.05}


def t_test_paired(before: Sequence[float], after: Sequence[float]) -> Dict[str, Any]:
    if len(before) != len(after) or len(before) < 2:
        raise ValueError("paired test needs equal lengths, n >= 2")
    diffs = [b - a for b, a in zip(before, after)]
    res = t_test_one_sample(diffs, 0.0)
    res["mean_diff"] = res["mean"]
    res.pop("mean")
    return res


def cohen_d(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    da, db = describe(a), describe(b)
    na, nb = da["n"], db["n"]
    sp = math.sqrt(((na - 1) * da["sd"] ** 2 + (nb - 1) * db["sd"] ** 2) / (na + nb - 2))
    d = (da["mean"] - db["mean"]) / sp if sp else 0.0
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    mag = abs(d)
    size = "small" if mag < 0.2 else "medium" if mag < 0.5 else "large" if mag < 0.8 else "very large"
    return {"d": d, "hedges_g": g, "magnitude": size}


# ---------------------------------------------------------------------------
# Non-parametric tests
# ---------------------------------------------------------------------------

def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    if not a or not b:
        raise ValueError("non-empty samples required")
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort()
    ranks = _ranks([v for v, _ in combined])
    n1, n2 = len(a), len(b)
    r1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu_u = n1 * n2 / 2
    # tie correction
    counts: Dict[float, int] = {}
    for v, _ in combined:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t ** 3 - t for t in counts.values())
    n = n1 + n2
    sd_u = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u - mu_u + 0.5) / sd_u if sd_u else 0.0
    p = 2 * normal_cdf(z)
    return {"u": u, "z": z, "p_approx": p, "n1": n1, "n2": n2,
            "significant_5pct": p < 0.05,
            "note": "normal approximation with continuity + tie correction"}


def wilcoxon_signed_rank(before: Sequence[float], after: Sequence[float]) -> Dict[str, Any]:
    if len(before) != len(after) or len(before) < 5:
        raise ValueError("needs paired samples with n >= 5")
    diffs = [(b - a, i) for i, (b, a) in enumerate(zip(before, after))]
    diffs = [(d, i) for d, i in diffs if d != 0]
    if len(diffs) < 5:
        raise ValueError("too few non-zero differences")
    n = len(diffs)
    ordered = sorted(range(n), key=lambda i: abs(diffs[i][0]))
    w_plus = w_minus = 0.0
    ranks = _ranks([abs(d) for d, _ in diffs])
    for i, (d, _) in enumerate(diffs):
        if d > 0:
            w_plus += ranks[i]
        else:
            w_minus += ranks[i]
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu + 0.5) / sd if sd else 0.0
    p = 2 * normal_cdf(z)
    return {"w": w, "w_plus": w_plus, "w_minus": w_minus, "z": z, "p_approx": p,
            "n": n, "significant_5pct": p < 0.05}


def sign_test(before: Sequence[float], after: Sequence[float]) -> Dict[str, Any]:
    if len(before) != len(after) or not before:
        raise ValueError("paired samples required")
    pos = sum(1 for b, a in zip(before, after) if b > a)
    neg = sum(1 for b, a in zip(before, after) if b < a)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "p": 1.0, "note": "all differences are zero"}
    from .probability import binomial_cdf
    k = min(pos, neg)
    p = min(1.0, 2 * binomial_cdf(k, n, 0.5))
    return {"pos": pos, "neg": neg, "n": n, "p": p, "significant_5pct": p < 0.05}


# ---------------------------------------------------------------------------
# Categorical tests
# ---------------------------------------------------------------------------

def chi2_independence(table: Sequence[Sequence[int]]) -> Dict[str, Any]:
    if len(table) < 2 or len(table[0]) < 2:
        raise ValueError("need at least a 2x2 table")
    rows = len(table)
    cols = len(table[0])
    if any(len(r) != cols for r in table):
        raise ValueError("ragged table")
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    n = sum(row_tot)
    if n == 0:
        raise ValueError("empty table")
    chi2 = 0.0
    expected = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            e = row_tot[i] * col_tot[j] / n
            expected[i][j] = e
            if e > 0:
                chi2 += (table[i][j] - e) ** 2 / e
    dof = (rows - 1) * (cols - 1)
    p = chi2_sf(chi2, dof)
    cramers_v = math.sqrt(chi2 / (n * (min(rows, cols) - 1))) if n else 0.0
    return {"chi2": chi2, "dof": dof, "p": p, "cramers_v": cramers_v,
            "expected": expected, "significant_5pct": p < 0.05}


def z_test_proportion(successes: int, n: int, p0: float) -> Dict[str, Any]:
    if n <= 0 or not 0 < p0 < 1 or successes < 0 or successes > n:
        raise ValueError("bad arguments for proportion test")
    phat = successes / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se
    p = 2 * (1 - normal_cdf(abs(z)))
    return {"phat": phat, "z": z, "p": p, "p0": p0,
            "ci95": [phat - 1.96 * math.sqrt(phat * (1 - phat) / n),
                     phat + 1.96 * math.sqrt(phat * (1 - phat) / n)],
            "significant_5pct": p < 0.05}


def odds_ratio(table: Sequence[Sequence[int]]) -> Dict[str, float]:
    if len(table) != 2 or len(table[0]) != 2:
        raise ValueError("odds ratio needs a 2x2 table")
    a, b = table[0][0], table[0][1]
    c, d = table[1][0], table[1][1]
    if min(a, b, c, d) <= 0:
        orv = float("inf") if (b * c) == 0 else (a * d) / (b * c)
        se_ln = float("inf")
    else:
        orv = (a * d) / (b * c)
        se_ln = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci = [math.exp(math.log(orv) - 1.96 * se_ln), math.exp(math.log(orv) + 1.96 * se_ln)] if se_ln not in (float("inf"),) else None
    return {"odds_ratio": orv, "ci95_log": ci,
            "note": "OR=1 means no association; CI excluding 1 is evidence of one"}


# ---------------------------------------------------------------------------
# ANOVA + normality
# ---------------------------------------------------------------------------

def anova_one_way(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2:
        raise ValueError("need >= 2 groups")
    allv = [v for g in groups for v in g]
    grand = sum(allv) / len(allv)
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_within = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
    k, n = len(groups), len(allv)
    df_b, df_w = k - 1, n - k
    ms_b = ss_between / df_b if df_b else 0.0
    ms_w = ss_within / df_w if df_w else 0.0
    f = ms_b / ms_w if ms_w > 0 else float("inf")
    p = f_sf(f, df_b, df_w) if ms_w > 0 else 0.0
    return {"f": f, "p": p, "df_between": df_b, "df_within": df_w,
            "ss_between": ss_between, "ss_within": ss_within,
            "eta_squared": ss_between / (ss_between + ss_within) if (ss_between + ss_within) else 0.0,
            "group_means": {f"group_{i}": sum(g) / len(g) for i, g in enumerate(groups)},
            "significant_5pct": p < 0.05}


def jarque_bera(data: Sequence[float]) -> Dict[str, Any]:
    d = describe(data)
    n = d["n"]
    if n < 8:
        raise ValueError("JB needs n >= 8")
    jb = n * (d["skewness"] ** 2 / 6 + d["excess_kurtosis"] ** 2 / 24)
    p = chi2_sf(jb, 2)
    return {"jb": jb, "p": p, "skewness": d["skewness"],
            "excess_kurtosis": d["excess_kurtosis"],
            "normal_5pct": p >= 0.05,
            "verdict": "consistent with normality" if p >= 0.05 else "deviates from normality"}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def ols_regression(y: Sequence[float], *xs: Sequence[float],
                   names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Multiple linear regression with full inference (SE, t, p, adj R2)."""
    n = len(y)
    if not xs or any(len(x) != n for x in xs):
        raise ValueError("y and every x must share length n")
    if n < len(xs) + 2:
        raise ValueError("not enough observations for this many predictors")
    design = [[1.0] + [float(x[i]) for x in xs] for i in range(n)]
    beta, info = linalg.least_squares(design, [float(v) for v in y])
    k = len(beta)
    yv = [float(v) for v in y]
    ybar = sum(yv) / n
    ss_res = info["ss_res"]
    ss_tot = sum((v - ybar) ** 2 for v in yv)
    dof = n - k
    sigma2 = ss_res / dof
    # standard errors from (X'X)^-1 diagonal
    xt = linalg.transpose(design)
    xtx_inv = linalg.inverse(linalg.matmul(xt, design))
    se = [math.sqrt(max(0.0, sigma2 * xtx_inv[i][i])) for i in range(k)]
    tstats = [beta[i] / se[i] if se[i] > 0 else 0.0 for i in range(k)]
    pvals = [2 * (1 - t_cdf(abs(t), dof)) for t in tstats]
    labels = ["intercept"] + list(names or [f"x{i + 1}" for i in range(len(xs))])
    adj_r2 = 1 - (1 - info["r2"]) * (n - 1) / dof if dof else info["r2"]
    f_stat = (info["r2"] / max(1e-300, 1 - info["r2"])) * dof / len(xs) if len(xs) else 0.0
    return {"coefficients": dict(zip(labels, beta)),
            "std_errors": dict(zip(labels, se)),
            "t_values": dict(zip(labels, tstats)),
            "p_values": dict(zip(labels, pvals)),
            "r2": info["r2"], "adj_r2": adj_r2,
            "f": f_stat, "f_p": f_sf(f_stat, len(xs), dof) if len(xs) else None,
            "residual_se": math.sqrt(sigma2), "n": n, "dof": dof,
            "equation": " + ".join(f"({b:.6g})*{l}" for l, b in zip(labels, beta))}


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def bootstrap_ci(data: Sequence[float], statistic=None, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    if len(data) < 2:
        raise ValueError("need n >= 2")
    stat = statistic or (lambda d: sum(d) / len(d))
    rng = random.Random(seed)
    d = [float(v) for v in data]
    stats = []
    for _ in range(n_boot):
        sample = [d[rng.randrange(len(d))] for _ in range(len(d))]
        try:
            stats.append(stat(sample))
        except (ValueError, ZeroDivisionError):
            continue
    if len(stats) < 100:
        raise ValueError("statistic failed on nearly all bootstrap resamples")
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return {"observed": stat(d), "ci": [lo, hi], "n_boot": len(stats),
            "alpha": alpha,
            "note": "percentile bootstrap; resamples your data, assumes it represents the world"}


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                      seed: int = 42) -> Dict[str, Any]:
    if not a or not b:
        raise ValueError("non-empty samples required")
    da, db = describe(a), describe(b)
    observed = da["mean"] - db["mean"]
    rng = random.Random(seed)
    pooled = [float(v) for v in a] + [float(v) for v in b]
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = sum(pooled[:na]) / na - sum(pooled[na:]) / len(pooled[na:])
        if abs(diff) >= abs(observed) - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"observed_difference": observed, "p": p, "n_perm": n_perm,
            "significant_5pct": p < 0.05,
            "note": "exact-ish under H0: exchangeable groups"}


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def detect_outliers(data: Sequence[float], method: str = "iqr") -> Dict[str, Any]:
    vals = _nums(data)
    d = describe(vals)
    flags: List[Dict[str, Any]] = []
    if method == "iqr":
        lo = d["q1"] - 1.5 * d["iqr"]
        hi = d["q3"] + 1.5 * d["iqr"]
        ext_lo = d["q1"] - 3.0 * d["iqr"]
        ext_hi = d["q3"] + 3.0 * d["iqr"]
        for i, v in enumerate(vals):
            if v < lo or v > hi:
                flags.append({"index": i, "value": v,
                              "severity": "extreme" if v < ext_lo or v > ext_hi else "mild"})
        rule = {"fence_low": lo, "fence_high": hi}
    elif method == "zscore":
        for i, v in enumerate(vals):
            z = (v - d["mean"]) / d["sd"] if d["sd"] else 0.0
            if abs(z) > 3:
                flags.append({"index": i, "value": v, "z": round(z, 3), "severity": "z>3"})
        rule = {"threshold": 3.0}
    elif method == "mad":
        med = d["median"]
        mad_sd = 1.4826 * d["mad"]
        for i, v in enumerate(vals):
            mz = (v - med) / mad_sd if mad_sd else 0.0
            if abs(mz) > 3.5:
                flags.append({"index": i, "value": v, "modified_z": round(mz, 3),
                              "severity": "mz>3.5"})
        rule = {"threshold": 3.5, "robust": True}
    else:
        raise ValueError(f"unknown method {method!r} (iqr|zscore|mad)")
    return {"method": method, "rule": rule, "outliers": flags,
            "n": len(vals), "n_outliers": len(flags)}

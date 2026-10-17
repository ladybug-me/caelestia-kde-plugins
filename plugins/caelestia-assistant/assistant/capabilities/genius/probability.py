"""genius.probability — probability, Bayes, Monte Carlo, Markov chains.

  * combinatorics: nCr/nPr with repetition, stars & bars, Catalan, derangements
  * Bayes chains: posterior from prior x likelihood, multi-hypothesis form,
    sequential evidence updating
  * distributions: binomial, Poisson, geometric, hypergeometric, exponential,
    normal (pdf/cdf/quantile via the complementary error function + Newton)
  * expected value / variance for arbitrary discrete random variables
  * Monte Carlo engine: seeded, with antithetic-variates variance reduction,
    mean/std/CI reporting, and expression-level simulation
  * Markov chains: stationary distribution by power iteration, multi-step
    transition, absorbing-chain analysis, trajectory simulation
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence

from . import linalg

__all__ = [
    "nCr", "nPr", "stars_and_bars", "catalan", "derangements",
    "bayes", "bayes_multi", "evidence_update",
    "binomial_pmf", "binomial_cdf", "poisson_pmf", "poisson_cdf",
    "geometric_pmf", "hypergeometric_pmf", "exponential_pdf", "exponential_cdf",
    "normal_pdf", "normal_cdf", "normal_quantile",
    "discrete_stats", "monte_carlo", "simulate_expression",
    "markov_stationary", "markov_step", "markov_absorbing", "markov_simulate",
]


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def nCr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def nPr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.perm(n, r)


def stars_and_bars(n: int, k: int) -> int:
    """Number of ways to put n identical items into k boxes."""
    if k <= 0:
        return 1 if n == 0 else 0
    return math.comb(n + k - 1, k - 1)


def catalan(n: int) -> int:
    return math.comb(2 * n, n) // (n + 1)


def derangements(n: int) -> int:
    d = [1, 0]
    for i in range(2, n + 1):
        d.append((i - 1) * (d[i - 1] + d[i - 2]))
    return d[n] if n >= 0 else 0


# ---------------------------------------------------------------------------
# Bayes
# ---------------------------------------------------------------------------

def bayes(prior: float, likelihood_given_h: float, likelihood_given_not_h: float) -> Dict[str, Any]:
    """Single-hypothesis Bayes: P(H|E) = P(E|H)P(H) / P(E)."""
    if not 0 <= prior <= 1:
        raise ValueError("prior must be in [0,1]")
    num = likelihood_given_h * prior
    den = num + likelihood_given_not_h * (1 - prior)
    if den == 0:
        raise ValueError("evidence has zero probability under every hypothesis")
    post = num / den
    return {"prior": prior, "posterior": post, "bayes_factor": (likelihood_given_h / likelihood_given_not_h
                                                                if likelihood_given_not_h else float("inf")),
            "note": f"P(H|E) rose from {_p(prior)} to {_p(post)}"}


def bayes_multi(hypotheses: Dict[str, float], likelihoods: Dict[str, float]) -> Dict[str, Any]:
    """P(H_i|E) for mutually exclusive exhaustive hypotheses."""
    if set(hypotheses) != set(likelihoods):
        raise ValueError("hypothesis and likelihood keys must match")
    total = sum(hypotheses[h] * likelihoods[h] for h in hypotheses)
    if total == 0:
        raise ValueError("evidence impossible under all hypotheses")
    post = {h: hypotheses[h] * likelihoods[h] / total for h in hypotheses}
    best = max(post, key=post.get)
    return {"prior": hypotheses, "posterior": post, "best": best,
            "best_p": post[best],
            "ranked": sorted(post.items(), key=lambda kv: -kv[1])}


def evidence_update(prior: Dict[str, float], evidence: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Sequential Bayes: apply a list of (per-hypothesis likelihood) observations."""
    current = dict(prior)
    trail: List[Dict[str, float]] = []
    for i, ev in enumerate(evidence):
        total = sum(current[h] * ev.get(h, 0.0) for h in current)
        if total == 0:
            raise ValueError(f"evidence #{i + 1} impossible under all hypotheses")
        current = {h: current[h] * ev.get(h, 0.0) / total for h in current}
        trail.append(dict(current))
    ranked = sorted(current.items(), key=lambda kv: -kv[1])
    return {"prior": prior, "posterior": current, "trail": trail, "ranked": ranked}


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def binomial_pmf(k: int, n: int, p: float) -> float:
    if not 0 <= p <= 1 or k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(binomial_pmf(i, n, p) for i in range(k + 1))


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0 or lam < 0:
        return 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def geometric_pmf(k: int, p: float) -> float:
    """P(first success on trial k)."""
    if k < 1 or not 0 < p <= 1:
        return 0.0
    return (1 - p) ** (k - 1) * p


def hypergeometric_pmf(k: int, n_pop: int, n_success: int, n_draw: int) -> float:
    if k < 0 or k > min(n_success, n_draw):
        return 0.0
    return math.comb(n_success, k) * math.comb(n_pop - n_success, n_draw - k) / math.comb(n_pop, n_draw)


def exponential_pdf(x: float, lam: float) -> float:
    return lam * math.exp(-lam * x) if x >= 0 and lam > 0 else 0.0


def exponential_cdf(x: float, lam: float) -> float:
    return 1 - math.exp(-lam * x) if x >= 0 and lam > 0 else 0.0


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    z = (x - mu) / sigma
    return math.exp(-z * z / 2) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def normal_quantile(q: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF by the Wichura AS241 rational approximation."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        z = math.sqrt(-2 * math.log(q))
        z = (((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) / \
            ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1)
    elif q <= ph:
        z = q - 0.5
        r = z * z
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * z / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        z = math.sqrt(-2 * math.log(1 - q))
        z = -(((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) / \
            ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1)
    return mu + sigma * z


def discrete_stats(values: Sequence[float], probs: Sequence[float]) -> Dict[str, float]:
    """E[X], Var[X], sd from an explicit discrete distribution."""
    if len(values) != len(probs) or not probs:
        raise ValueError("values and probs must be the same non-empty length")
    if abs(sum(probs) - 1) > 1e-9:
        raise ValueError(f"probs sum to {sum(probs):.6f}, not 1")
    mu = sum(v * p for v, p in zip(values, probs))
    var = sum((v - mu) ** 2 * p for v, p in zip(values, probs))
    return {"mean": mu, "variance": var, "sd": math.sqrt(var)}


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(fn, n: int = 100000, seed: int = 42,
                antithetic: bool = False) -> Dict[str, Any]:
    """Seeded Monte Carlo over a fn(rng)->float, with mean/sd/95% CI."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(seed)
    draws: List[float] = []
    if antithetic:
        for _ in range((n + 1) // 2):
            u = rng.random()
            draws.extend((u, 1.0 - u))
        draws = draws[:n]
    else:
        draws = [fn(rng) for _ in range(n)]
    mean = sum(draws) / n
    var = sum((x - mean) ** 2 for x in draws) / max(1, n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    z95 = normal_quantile(0.975)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci95": [mean - z95 * se, mean + z95 * se],
            "antithetic": antithetic, "seed": seed,
            "note": "the CI is the sampling error of the simulation, not of the world"}


def simulate_expression(expr: str, n: int = 100000, seed: int = 42) -> Dict[str, Any]:
    """Monte Carlo of an expression over UNIF(0,1) draws u1..u9 (mathengine)."""
    from . import mathengine
    node = mathengine.parse(expr)
    syms = sorted({nd.name for nd in mathengine._walk(node) if nd.kind == "var"})
    if len(syms) > 9:
        raise ValueError("at most 9 random variables u1..u9 supported")

    def draw(rng):
        env = {f"u{i + 1}": rng.random() for i in range(len(syms))}
        return mathengine.evaluate(node, env)

    result = monte_carlo(draw, n=n, seed=seed)
    result["expr"] = mathengine.to_str(node)
    result["variables"] = syms
    return result


# ---------------------------------------------------------------------------
# Markov chains
# ---------------------------------------------------------------------------

def _validate_transition(t: Sequence[Sequence[float]]) -> List[List[float]]:
    m = [[float(v) for v in row] for row in t]
    n = len(m)
    if n == 0 or any(len(row) != n for row in m):
        raise ValueError("transition matrix must be square")
    for row in m:
        s = sum(row)
        if abs(s - 1) > 1e-6:
            raise ValueError(f"row sums to {s:.6f}, not 1")
    return m


def markov_stationary(t: Sequence[Sequence[float]], iterations: int = 5000,
                       tol: float = 1e-12) -> Dict[str, Any]:
    """Stationary distribution by iterating pi P (power iteration)."""
    m = _validate_transition(t)
    n = len(m)
    pi = [1.0 / n] * n
    for _ in range(iterations):
        nxt = [sum(pi[i] * m[i][j] for i in range(n)) for j in range(n)]
        if max(abs(nxt[j] - pi[j]) for j in range(n)) < tol:
            pi = nxt
            break
        pi = nxt
    top = max(range(n), key=lambda j: pi[j])
    return {"stationary": pi, "top_state": top, "top_p": pi[top],
            "converged": abs(sum(pi) - 1) < 1e-9,
            "ranked": sorted(enumerate(pi), key=lambda kv: -kv[1])}


def markov_step(t: Sequence[Sequence[float]], state: int, steps: int) -> Dict[str, Any]:
    m = _validate_transition(t)
    n = len(m)
    if not 0 <= state < n:
        raise ValueError("state index out of range")
    v = [1.0 if i == state else 0.0 for i in range(n)]
    for _ in range(steps):
        v = [sum(v[i] * m[i][j] for i in range(n)) for j in range(n)]
    best = max(range(n), key=lambda j: v[j])
    return {"from_state": state, "steps": steps, "distribution": v,
            "most_likely": best, "most_likely_p": v[best]}


def markov_absorbing(t: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Absorbing-chain analysis: expected steps + absorption probabilities."""
    m = _validate_transition(t)
    n = len(m)
    absorbing = [i for i in range(n) if m[i][i] > 1 - 1e-9]
    transient = [i for i in range(n) if i not in absorbing]
    if not absorbing or not transient:
        return {"absorbing_states": absorbing, "transient_states": transient,
                "note": "not an absorbing chain (need both absorbing and transient states)"}
    idx = {s: k for k, s in enumerate(transient)}
    k = len(transient)
    q = [[m[i][j] for j in transient] for i in transient]
    # fundamental matrix N = (I - Q)^-1
    imq = [[(1.0 if i == j else 0.0) - q[i][j] for j in range(k)] for i in range(k)]
    ninv = linalg.inverse(imq)
    steps_to_absorb = [sum(ninv[i]) for i in range(k)]
    r = [[m[i][j] for j in absorbing] for i in transient]
    b = linalg.matmul(ninv, r)  # absorption probabilities
    return {"absorbing_states": absorbing, "transient_states": transient,
            "expected_steps_to_absorption": {transient[i]: steps_to_absorb[i] for i in range(k)},
            "absorption_probabilities": {
                transient[i]: {absorbing[j]: b[i][j] for j in range(len(absorbing))}
                for i in range(k)}}


def markov_simulate(t: Sequence[Sequence[float]], start: int, steps: int,
                    seed: int = 7) -> Dict[str, Any]:
    m = _validate_transition(t)
    rng = random.Random(seed)
    state, path, counts = start, [start], [0] * len(m)
    counts[start] = 1
    for _ in range(steps):
        u, acc, state = rng.random(), 0.0, state
        nxt = state
        for j, p in enumerate(m[state]):
            acc += p
            if u < acc:
                nxt = j
                break
        state = nxt
        counts[state] += 1
        path.append(state)
    return {"path": path, "counts": counts,
            "frequencies": [c / (steps + 1) for c in counts]}


def _p(v: float) -> str:
    return f"{v:.4f}"

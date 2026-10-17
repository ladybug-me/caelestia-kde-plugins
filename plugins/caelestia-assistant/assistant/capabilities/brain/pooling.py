"""brain.pooling — hierarchical partial pooling for Beta posteriors
(Efron & Morris 1975, "Data analysis using Stein's estimator and its
simple generalizations", JASA; the empirical-Bayes shrinkage idea).

The problem it solves: every preference posterior in this repo starts
from a flat Beta(1, 1) prior — "no opinion". But a NEW arm (a new tool
proposal, a new strategy, a new timing bucket) is NOT independent of
the arms that came before: it belongs to a POPULATION of arms whose
observed acceptance rates carry information ("this user approves
~70% of settings proposals, with modest spread"). Starting the new
arm at the population's hierarchical prior instead of the flat prior
is partial pooling — the new arm borrows strength from the pool and
only escapes it as its own evidence accumulates.

The method (deterministic method of moments, no iteration, no RNG):

- per arm, the OBSERVED evidence is n_i = (alpha_i - a0) + (beta_i - b0)
  and the observed rate p_i = s_i / n_i (s_i = alpha_i - a0), where
  (a0, b0) is the flat prior the arms were seeded with — the prior's
  own mass never counts as evidence;
- pooled mean mu0 = sum(s_i) / sum(n_i) over arms WITH evidence;
- between-arm variance tau^2 = max(0, Var(p_i) - mean(mu0(1-mu0)/n_i))
  (the standard moments decomposition: observed spread minus expected
  sampling noise);
- prior concentration k0 = mu0(1-mu0)/tau^2, capped at the pool's total
  evidence when tau^2 ~ 0 (the arms agree completely: full pooling) and
  floored at 1.0 (never worse than a one-pseudocount opinion);
- the hierarchical prior for a NEW arm is
  Beta(1 + k0*mu0, 1 + k0*(1-mu0)) — flat prior mass PLUS pooled
  pseudo-counts, so the new arm starts AT the pool's mean and only
  moves as its own decisions arrive.

Honesty rules: arms with no observed evidence contribute nothing to
the pool (they are listed, not fabricated); a pool with NO evidence
anywhere yields the flat prior unchanged and says so; the shrinkage
weight of every arm is visible in the output. Pure functions, JSON-
round-trippable dicts, deterministic, no I/O — the caller persists
through the learned-state path it already owns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

__all__ = ["pool_arms", "hierarchical_prior", "shrunk_mean"]

_FLAT_PRIOR_ALPHA = 1.0
_FLAT_PRIOR_BETA = 1.0
_MIN_K0 = 1.0


def pool_arms(arms: Dict[str, Sequence[float]],
              prior_alpha: float = _FLAT_PRIOR_ALPHA,
              prior_beta: float = _FLAT_PRIOR_BETA) -> Dict[str, Any]:
    """Method-of-moments hyperparameters over {name: [alpha, beta]}.

    Returns {"mu0", "k0", "tau2", "n_arms", "evidence_arms",
    "no_evidence_arms", "total_evidence", "pooled": bool} — ``pooled``
    is False when NO arm had evidence, in which case the honest
    hierarchical prior degenerates to the flat one."""
    if not arms:
        raise ValueError("no arms to pool")
    rates: List[float] = []
    evidence: List[float] = []
    s_total = n_total = 0.0
    no_evidence: List[str] = []
    for name in sorted(arms):
        alpha, beta = (float(v) for v in arms[name][:2])
        s = max(0.0, alpha - prior_alpha)
        n = max(0.0, (alpha - prior_alpha) + (beta - prior_beta))
        if n <= 0:
            no_evidence.append(name)
            continue
        s_total += s
        n_total += n
        rates.append(s / n)
        evidence.append(n)
    if not rates:
        return {"mu0": 0.5, "k0": 0.0, "tau2": 0.0,
                "n_arms": len(arms), "evidence_arms": 0,
                "no_evidence_arms": no_evidence, "total_evidence": 0.0,
                "pooled": False,
                "note": "no arm has observed evidence — nothing to "
                        "borrow from; the flat prior stands"}
    mu0 = s_total / n_total
    m = len(rates)
    mean_rate = sum(rates) / m
    var_observed = sum((r - mean_rate) ** 2 for r in rates) / m
    # expected sampling noise around the POOLED mean (the moments
    # decomposition: Var(p_i) = tau^2 + mu0(1-mu0)/n_i)
    mean_sampling_var = sum(mu0 * (1.0 - mu0) / n for n in evidence) / m
    tau2 = max(0.0, var_observed - mean_sampling_var)
    if mu0 <= 0.0 or mu0 >= 1.0 or tau2 <= 1e-12:
        # every arm agreed (or the pool sits at an extreme): FULL
        # pooling — the concentration caps at the pool's own evidence
        k0 = n_total
    else:
        k0 = mu0 * (1.0 - mu0) / tau2
    k0 = max(_MIN_K0, min(k0, n_total))
    return {"mu0": round(mu0, 6), "k0": round(k0, 6),
            "tau2": round(tau2, 6), "n_arms": len(arms),
            "evidence_arms": m, "no_evidence_arms": no_evidence,
            "total_evidence": round(n_total, 6), "pooled": True}


def hierarchical_prior(pooled: Dict[str, Any]) -> Dict[str, Any]:
    """The prior a NEW arm should start from, given the pool:
    Beta(k0*mu0, k0*(1-mu0)) — centered exactly at the pooled mean with
    concentration k0. A 0.5 Jeffreys-style floor keeps an extreme pool
    (mu0 at 0 or 1) from producing an improper prior — that floor is
    part of the stated rule, not a silent clamp of user data. When the
    pool is unpooled (no evidence anywhere) the FLAT prior stands, as
    the honesty rule promises."""
    mu0, k0 = pooled["mu0"], pooled["k0"]
    if not pooled.get("pooled", False):
        return {"alpha": 1.0, "beta": 1.0, "prior_mean": 0.5,
                "pseudo_counts": 0.0, "is_flat": True,
                "note": str(pooled.get("note", ""))}
    alpha = max(k0 * mu0, 0.5)
    beta = max(k0 * (1.0 - mu0), 0.5)
    return {"alpha": round(alpha, 6), "beta": round(beta, 6),
            "prior_mean": round(alpha / (alpha + beta), 6),
            "pseudo_counts": round(k0, 6),
            "is_flat": False}


def shrunk_mean(alpha: float, beta: float, pooled: Dict[str, Any],
                prior_alpha: float = _FLAT_PRIOR_ALPHA,
                prior_beta: float = _FLAT_PRIOR_BETA) -> Dict[str, Any]:
    """One EXISTING arm's shrunk acceptance estimate: its observed
    evidence pooled with the hierarchical prior — the Efron-Morris
    estimator (s_i + k0*mu0) / (n_i + k0). The shrinkage weight
    (how much of the estimate is the pool's) is reported, visible, and
    shrinks toward 0 as the arm's own evidence grows."""
    s = max(0.0, alpha - prior_alpha)
    n = max(0.0, (alpha - prior_alpha) + (beta - prior_beta))
    mu0, k0 = pooled["mu0"], pooled["k0"]
    estimate = (s + k0 * mu0) / (n + k0)
    return {"estimate": round(estimate, 6),
            "own_evidence": round(n, 6),
            "shrinkage_weight": round(k0 / (n + k0), 6)}

"""brain.gp_prefs — preference learning from "A or B?" with a Gaussian
process, Cholesky all the way (C13).

Users answer pairwise questions far more reliably than absolute ones
("which of these two bars looks right?" beats "what opacity, 0-100?").
This module learns a latent utility over candidate settings from
pairwise choices, using a GP prior over item features and a probit
likelihood on the utility DIFFERENCE — the classical Gaussian-process
preference model (Chu & Ghahramani 2005), fit by Newton iterations on
the Laplace approximation:

    log p(f) = -1/2 f' K^-1 f  +  sum_i log Phi((f_a - f_b)/sqrt(2))

All linear algebra is a hand-rolled Cholesky decomposition and
triangular solves — n is bounded at 50 items, so a dense 50x50 kernel
is the whole world and a back-substitution costs microseconds.

Deliverables beyond the ranking:

  - the NEXT QUESTION: the unasked pair whose utility-difference
    posterior variance is highest — the question that, answered, is
    expected to remove the most uncertainty (deterministic tie-breaks);
  - VALIDATED SETTER PROPOSALS: the top-ranked items that name a real
    registry tool get their value checked against the tool's own spec
    (kind, minimum, maximum, enum) BEFORE surfacing — a proposal that
    would be rejected by the planner never leaves this module wearing
    the word 'validated'. Proposals are returned, never applied.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["GPPreferenceModel", "validate_item"]

_MAX_ITEMS = 50
_NEWTON_STEPS = 30
_PREFER_SCALE = math.sqrt(2.0)  # probit scale on utility differences


# ---------------------------------------------------------------------------
# the linear algebra: Cholesky, nothing else
# ---------------------------------------------------------------------------

def _chol(matrix: List[List[float]]) -> List[List[float]]:
    """Lower-triangular Cholesky of a symmetric positive-definite
    matrix; a small jitter ladder rescues near-singular kernels."""
    n = len(matrix)
    jitter = 0.0
    for attempt in range(6):
        try:
            L = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    s = matrix[i][j] - sum(
                        L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        if s - jitter <= 0:
                            raise ValueError("not positive definite")
                        L[i][i] = math.sqrt(s - jitter)
                    else:
                        L[i][j] = s / L[j][j]
            return L
        except (ValueError, OverflowError):
            jitter = max(jitter * 10.0, 1e-10)
    raise ValueError("kernel could not be factorized even with jitter")


def _chol_solve(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b given A = L L'."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) \
            / L[i][i]
    return x


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _softplus(x: float) -> float:
    """log(1 + e^x), stable both sides."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _log_phi(z: float) -> float:
    """log Phi(z) via -softplus(-z): never underflows, unlike Phi."""
    return -_softplus(-z)


def _probit_score(z: float) -> float:
    """phi(z)/Phi(z) in log space — the probit gradient term."""
    log_pdf = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    return math.exp(log_pdf - _log_phi(z))


def _probit_curvature(z: float) -> float:
    """d2/dz2 log Phi(z) = -r (z + r) with r = phi/Phi. Strongly
    negative near the decision region (where the pairwise likelihood
    is concave and Newton behaves); tiny float noise in the far tails
    is harmless because the GP prior term dominates the Hessian."""
    r = _probit_score(z)
    return -r * (z + r)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class GPPreferenceModel:
    """GP preference model over <= 50 items with numeric features."""

    def __init__(self, items: List[Dict[str, Any]],
                 length_scale: float = 1.0, signal: float = 1.0,
                 noise: float = 0.25) -> None:
        if not items:
            raise ValueError("need at least one item")
        if len(items) > _MAX_ITEMS:
            raise ValueError(f"bounded at {_MAX_ITEMS} items "
                             f"(got {len(items)}) — the dense kernel "
                             f"is deliberately small")
        self.items = list(items)
        self.index = {str(it["id"]): i for i, it in enumerate(items)}
        self.length_scale = float(length_scale)
        self.signal = float(signal)
        self.noise = float(noise)
        self.features = [list(map(float, it.get("features", [0.0])))
                         for it in items]
        # comparisons: (i, j, y) with y=+1 if i won, -1 if j won
        self.pairs: List[Tuple[int, int, int]] = []
        self._kernel = self._build_kernel()

    # -- kernel ------------------------------------------------------------
    def _sqdist(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("item feature vectors must share a length")
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _build_kernel(self) -> List[List[float]]:
        n = len(self.features)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                r2 = self._sqdist(self.features[i], self.features[j])
                value = self.signal ** 2 * math.exp(
                    -r2 / (2.0 * self.length_scale ** 2))
                K[i][j] = K[j][i] = value
        for i in range(n):
            K[i][i] += self.noise ** 2
        return K

    # -- likelihood plumbing -------------------------------------------------
    def _pair_delta(self, f: List[float], i: int, j: int) -> float:
        return (f[i] - f[j]) / _PREFER_SCALE

    def _grad_hess(self, f: List[float]) -> Tuple[List[float],
                                                  List[List[float]]]:
        """Gradient and Hessian of log p(pairs | f) (+ prior handled by
        the caller's K term)."""
        n = len(f)
        g = [0.0] * n
        H = [[0.0] * n for _ in range(n)]
        for i, j, y in self.pairs:
            z = y * self._pair_delta(f, i, j)
            # stable probit terms (log-space; Phi never underflows)
            coef = _probit_score(z) * y / _PREFER_SCALE
            g[i] += coef
            g[j] -= coef
            d2 = _probit_curvature(z)
            e = [0.0] * n
            e[i] = 1.0 / _PREFER_SCALE * y
            e[j] = -1.0 / _PREFER_SCALE * y
            for a in range(n):
                if e[a] == 0.0:
                    continue
                for b in range(n):
                    if e[b] != 0.0:
                        H[a][b] -= d2 * e[a] * e[b]
        return g, H

    def fit(self) -> Dict[str, Any]:
        """Laplace approximation: Newton to the posterior mode of
        log p(pairs|f) - 1/2 f'K^-1 f, i.e. the Newton system
        (K^-1 + Lambda) df = K^-1 f - g with Lambda = -d2(loglik) —
        symmetric positive definite, so every solve is a Cholesky."""
        n = len(self.items)
        Lk = _chol(self._kernel)
        # explicit K^-1 (n <= 50: column solves are microseconds)
        Kinv = [[0.0] * n for _ in range(n)]
        for col in range(n):
            e = [1.0 if i == col else 0.0 for i in range(n)]
            sol = _chol_solve(Lk, e)
            for row in range(n):
                Kinv[row][col] = sol[row]
        f = [0.0] * n
        Ln = None
        steps = 0
        for _ in range(_NEWTON_STEPS):
            steps += 1
            g, lam = self._grad_hess(f)
            Kinv_f = _chol_solve(Lk, f)
            neg_grad = [Kinv_f[i] - g[i] for i in range(n)]
            M = [[Kinv[i][j] + lam[i][j] for j in range(n)]
                 for i in range(n)]
            Ln = _chol(M)
            # Newton on the negative objective: f <- f - M^-1 (K^-1 f - g)
            step = _chol_solve(Ln, neg_grad)
            f = [f[i] - step[i] for i in range(n)]
            if max(abs(s) for s in step) < 1e-8:
                break
        # posterior variances: diag of (K^-1 + Lambda)^-1 =
        # ||Ln^-1 e_i||^2 per column
        var = [0.0] * n
        if Ln is not None:
            for col in range(n):
                e = [1.0 if i == col else 0.0 for i in range(n)]
                z = _chol_solve(Ln, e)
                var[col] = sum(v * v for v in z)
        ranking = sorted(range(n), key=lambda i: (-f[i], i))
        return {
            "utilities": {str(self.items[i]["id"]): round(f[i], 6)
                          for i in range(n)},
            "variances": {str(self.items[i]["id"]): round(var[i], 6)
                          for i in range(n)},
            "ranking": [str(self.items[i]["id"]) for i in ranking],
            "n_pairs": len(self.pairs),
            "n_newton_steps": steps,
        }

    # -- interaction -------------------------------------------------------
    def record(self, winner: str, loser: str) -> None:
        if winner == loser:
            return
        if winner not in self.index or loser not in self.index:
            raise ValueError("unknown item id in comparison")
        i, j = self.index[winner], self.index[loser]
        self.pairs.append((i, j, +1))

    def next_question(self) -> Optional[Tuple[str, str]]:
        """The unasked pair with the largest utility-difference
        variance — the most informative single question."""
        asked = {(min(i, j), max(i, j)) for i, j, _ in self.pairs}
        result = self.fit()
        var = result["variances"]
        best = None
        best_score = -1.0
        ids = [str(it["id"]) for it in self.items]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if (i, j) in asked:
                    continue
                score = var[ids[i]] + var[ids[j]]
                if score > best_score + 1e-12:
                    best_score = score
                    best = (ids[i], ids[j])
        return best


def validate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Check one candidate proposal against the registry's own spec:
    tool exists, kind matches, value within range / enum. This is the
    same discipline the planner applies before anything is offered."""
    from assistant.adapters.caelestia.registry import tool_by_name
    tool_name = str(item.get("tool", ""))
    spec = tool_by_name(tool_name)
    if spec is None:
        return {"item": item, "validated": False,
                "reason": f"unknown tool {tool_name!r}"}
    if "value" not in item:
        return {"item": item, "validated": False,
                "reason": "item carries no value"}
    value = item["value"]
    if spec.kind == "bool" and not isinstance(value, bool):
        return {"item": item, "validated": False,
                "reason": f"{tool_name} wants a bool, got {value!r}"}
    if spec.kind in ("int", "float"):
        if not isinstance(value, (int, float)) or \
                isinstance(value, bool):
            return {"item": item, "validated": False,
                    "reason": f"{tool_name} wants a number, got "
                              f"{value!r}"}
        if spec.minimum is not None and value < spec.minimum:
            return {"item": item, "validated": False,
                    "reason": f"{value} below minimum {spec.minimum} "
                              f"for {tool_name} — rejected, never "
                              f"clamped"}
        if spec.maximum is not None and value > spec.maximum:
            return {"item": item, "validated": False,
                    "reason": f"{value} above maximum {spec.maximum} "
                              f"for {tool_name} — rejected, never "
                              f"clamped"}
    if spec.kind == "enum":
        allowed = set(spec.enum or ())
        if value not in allowed:
            return {"item": item, "validated": False,
                    "reason": f"{value!r} is not one of "
                              f"{sorted(allowed)[:8]}"}
    return {"item": item, "validated": True, "reason": "within spec"}

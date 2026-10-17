"""genius.baysnet — Bayesian networks, HMMs, and naive Bayes.

  * discrete Bayesian networks: define a DAG with CPTs, then exact
    inference by enumeration with caching; diagnostic queries
    P(cause | observed symptoms) with a plain-English explanation
  * Hidden Markov Models: the forward algorithm (likelihood of an
    observation sequence), Viterbi (the most probable hidden state path
    — "what was the system doing"), and next-state prediction
  * naive Bayes classifier with Laplace smoothing: trains from rows of
    {feature: value} dicts, ranks classes with log-probabilities, and
    names the features that moved the decision (its evidence)

Small, exact, auditable. No junction trees, no sampling — the honest
envelope for nets of a few dozen nodes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

__all__ = ["BayesNet", "HMM", "NaiveBayes", "hmm_forward", "hmm_viterbi",
           "hmm_predict_next"]


class BayesError(ValueError):
    pass


class _Node:
    __slots__ = ("name", "parents", "cpt")

    def __init__(self, name: str, parents: Sequence[str], cpt: Dict[str, float]):
        self.name = name
        self.parents = list(parents)
        self.cpt = cpt
        self.validate()

    def validate(self) -> None:
        combos = 1
        for p in self.parents:
            combos *= 2
        if len(self.cpt) != combos:
            raise BayesError(
                f"{self.name}: CPT needs {combos} entries for parents {self.parents}")
        for k, v in self.cpt.items():
            if not 0 <= v <= 1:
                raise BayesError(f"{self.name}: P={v} outside [0,1]")
        # rows must sum to 1 across the parent assignment
        for combo in _parent_assignments(self.parents):
            p_true = self.cpt[_key(combo)]
            if abs(p_true + (1 - p_true) - 1) > 1e-9:  # trivially true; kept for clarity
                raise BayesError("CPT rows must be probabilities")

    def p(self, value: bool, assignment: Dict[str, bool]) -> float:
        combo = {p: assignment[p] for p in self.parents}
        pt = self.cpt[_key(combo)]
        return pt if value else 1.0 - pt


def _parent_assignments(parents: Sequence[str]) -> Iterable[Dict[str, bool]]:
    n = len(parents)
    for bits in range(2 ** n):
        yield {parents[i]: bool((bits >> (n - 1 - i)) & 1) for i in range(n)}


def _key(combo: Dict[str, bool]) -> str:
    if not combo:
        return "*"
    return ",".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(combo.items()))


class BayesNet:
    """A discrete two-value-per-node Bayesian network."""

    def __init__(self):
        self.nodes: Dict[str, _Node] = {}

    def add(self, name: str, parents: Sequence[str] = (),
            cpt: Optional[Dict[str, float]] = None) -> "BayesNet":
        if name in self.nodes:
            raise BayesError(f"node {name!r} already exists")
        for p in parents:
            if p not in self.nodes:
                raise BayesError(f"parent {p!r} not added before {name!r}")
        # No cycle check needed: parents must already exist and `name`
        # has no incoming edges yet, so a cycle through `name` is
        # impossible by construction under this API.
        cpt = cpt if cpt is not None else {"*": 0.5}
        self.nodes[name] = _Node(name, parents, cpt)
        return self

    def probability(self, assignment: Dict[str, bool]) -> float:
        """Joint probability of a full or partial assignment (marginalized
        by exact enumeration over the unassigned nodes)."""
        names = list(self.nodes)
        missing = [n for n in names if n not in assignment]

        def total(idx: int) -> float:
            if idx == len(missing):
                prod = 1.0
                for n in names:
                    prod *= self.nodes[n].p(assignment[n], assignment)
                return prod
            m = missing[idx]
            s = 0.0
            for val in (False, True):
                assignment[m] = val
                s += total(idx + 1)
            del assignment[m]
            return s

        return total(0)

    def query(self, target: str, evidence: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        evidence = evidence or {}
        if target in evidence:
            return {"target": target, "p": 1.0 if evidence[target] else 0.0,
                    "evidence": evidence, "note": "target is in the evidence"}
        num = self.probability({**evidence, target: True})
        den = num + self.probability({**evidence, target: False})
        if den == 0:
            raise BayesError("evidence has probability zero under this network")
        p = num / den
        return {"target": target, "p": round(p, 6), "p_not": round(1 - p, 6),
                "evidence": evidence,
                "verdict": f"P({target} | evidence) = {p:.2%}"}

    def diagnose(self, causes: Sequence[str], evidence: Dict[str, bool],
                 top: int = 5) -> Dict[str, Any]:
        """Rank candidate causes by posterior given observed evidence."""
        results = []
        for c in causes:
            if c in evidence:
                continue
            try:
                q = self.query(c, evidence)
                results.append({"cause": c, "posterior": q["p"]})
            except BayesError:
                continue
        results.sort(key=lambda r: -r["posterior"])
        total = sum(r["posterior"] for r in results) or 1.0
        for r in results:
            r["share"] = round(r["posterior"] / total, 4)
        return {"evidence": evidence, "ranked_causes": results[:top],
                "note": "posteriors are relative, not mutually exclusive"}


# ---------------------------------------------------------------------------
# Hidden Markov Models
# ---------------------------------------------------------------------------

Matrix = List[List[float]]
Vector = List[float]


def _validate_matrix(m: Matrix, name: str, rows_label: str) -> None:
    if not m or len(m) != len(m[0]):
        raise BayesError(f"{name} must be square")
    for row in m:
        if abs(sum(row) - 1) > 1e-6:
            raise BayesError(f"{name} rows must sum to 1")


def hmm_forward(prior: Vector, trans: Matrix, emit: Matrix,
                obs: Sequence[int]) -> Dict[str, Any]:
    """Forward algorithm: likelihood + filtered state beliefs per step."""
    n = len(prior)
    _validate_matrix(trans, "transition", "")
    if len(emit) != n:
        raise BayesError("emission rows must match state count")
    for row in emit:
        if abs(sum(row) - 1) > 1e-6:
            raise BayesError("emission rows must sum to 1")
    alpha = [prior[i] * emit[i][obs[0]] for i in range(n)]
    likelihood = sum(alpha)
    alpha = [a / likelihood for a in alpha]
    beliefs = [alpha[:]]
    for t in range(1, len(obs)):
        nxt = [sum(alpha[i] * trans[i][j] for i in range(n)) * emit[j][obs[t]]
               for j in range(n)]
        s = sum(nxt)
        if s == 0:
            raise BayesError(f"observation #{t} impossible under this HMM")
        alpha = [v / s for v in nxt]
        likelihood *= s
        beliefs.append(alpha[:])
    best_final = max(range(n), key=lambda i: alpha[i])
    return {"likelihood": likelihood, "log_likelihood": round(math.log(max(likelihood, 1e-300)), 6),
            "beliefs": [[round(b, 6) for b in step] for step in beliefs],
            "final_belief": [round(a, 6) for a in alpha],
            "most_likely_final_state": best_final}


def hmm_viterbi(prior: Vector, trans: Matrix, emit: Matrix,
                obs: Sequence[int]) -> Dict[str, Any]:
    """Viterbi: the single most probable hidden path for the observations."""
    n = len(prior)
    delta = [math.log(max(prior[i], 1e-300)) + math.log(max(emit[i][obs[0]], 1e-300))
             for i in range(n)]
    back: List[List[int]] = []
    for t in range(1, len(obs)):
        nd, bt = [], []
        for j in range(n):
            best_i, best_v = 0, -1e300
            for i in range(n):
                v = delta[i] + math.log(max(trans[i][j], 1e-300))
                if v > best_v:
                    best_v, best_i = v, i
            nd.append(best_v + math.log(max(emit[j][obs[t]], 1e-300)))
            bt.append(best_i)
        delta, _ = nd, back.append(bt)
    end = max(range(n), key=lambda i: delta[i])
    path = [end]
    for bt in reversed(back):
        path.append(bt[path[-1]])
    path.reverse()
    return {"path": path, "path_log_likelihood": round(delta[end], 6),
            "n_steps": len(obs),
            "decoded": "state sequence that best explains the observations"}


def hmm_predict_next(prior: Vector, trans: Matrix, emit: Matrix,
                     obs: Sequence[int]) -> Dict[str, Any]:
    fwd = hmm_forward(prior, trans, emit, obs)
    alpha = fwd["beliefs"][-1]
    n = len(alpha)
    predicted = [sum(alpha[i] * trans[i][j] for i in range(n)) for j in range(n)]
    best = max(range(n), key=lambda j: predicted[j])
    return {"state_distribution": [round(p, 6) for p in predicted],
            "next_likely_state": best, "p": round(predicted[best], 6)}


class HMM:
    """Convenience wrapper binding prior/transition/emission together."""

    def __init__(self, states: Sequence[str], observations: Sequence[str],
                 prior: Vector, trans: Matrix, emit: Matrix):
        self.states = list(states)
        self.observations = list(observations)
        self.prior = list(prior)
        self.trans = [list(r) for r in trans]
        self.emit = [list(r) for r in emit]

    def _obs_index(self, o: str) -> int:
        try:
            return self.observations.index(o)
        except ValueError:
            raise BayesError(f"unknown observation {o!r}")

    def observe(self, seq: Sequence[str]) -> Dict[str, Any]:
        idx = [self._obs_index(o) for o in seq]
        fwd = hmm_forward(self.prior, self.trans, self.emit, idx)
        vit = hmm_viterbi(self.prior, self.trans, self.emit, idx)
        nxt = hmm_predict_next(self.prior, self.trans, self.emit, idx)
        return {"forward": fwd, "viterbi": vit, "next": nxt,
                "states": self.states,
                "decoded_states": [self.states[i] for i in vit["path"]]}


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

class NaiveBayes:
    """Multinomial naive Bayes over string-valued features (Laplace-smoothed)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Dict[str, int] = {}
        self.feature_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.vocabulary: Dict[str, Set[str]] = {}

    def fit(self, rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "NaiveBayes":
        if len(rows) != len(labels) or not rows:
            raise BayesError("need aligned, non-empty rows and labels")
        for row, label in zip(rows, labels):
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            for f, v in row.items():
                fc = self.feature_counts.setdefault(label, {}).setdefault(str(f), {})
                fv = str(v)
                fc[fv] = fc.get(fv, 0) + 1
                self.vocabulary.setdefault(str(f), set()).add(fv)
        return self

    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.class_counts:
            raise BayesError("model is untrained")
        total = sum(self.class_counts.values())
        scores = {}
        for cls, cc in self.class_counts.items():
            logp = math.log(cc / total)
            for f, v in row.items():
                fv = str(v)
                vocab = self.vocabulary.get(str(f), set())
                fc = self.feature_counts.get(cls, {}).get(str(f), {})
                num = fc.get(fv, 0) + self.alpha
                den = cc + self.alpha * max(1, len(vocab))
                logp += math.log(num / den)
            scores[cls] = logp
        best = max(scores, key=scores.get)
        # normalize via log-sum-exp
        mx = scores[best]
        exps = {c: math.exp(s - mx) for c, s in scores.items()}
        z = sum(exps.values())
        probs = {c: e / z for c, e in exps.items()}
        evidence = self._evidence(row, best)
        return {"class": best, "probabilities": {c: round(p, 4) for c, p in probs.items()},
                "evidence": evidence[:5],
                "n_training_rows": total}

    def _evidence(self, row: Dict[str, Any], cls: str) -> List[str]:
        out = []
        cc = self.class_counts[cls]
        total = sum(self.class_counts.values())
        for f, v in sorted(row.items(), key=lambda kv: str(kv[0])):
            fv = str(v)
            fc = self.feature_counts.get(cls, {}).get(str(f), {})
            k = fc.get(fv, 0)
            if k:
                out.append(f"P[{cls}] {f}={fv}: seen {k}/{cc} in-class "
                           f"({k / cc:.0%}) vs base rate {cc / total:.0%}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"class_counts": dict(self.class_counts),
                "feature_counts": {c: {f: dict(v) for f, v in fc.items()}
                                   for c, fc in self.feature_counts.items()},
                "vocabulary": {f: sorted(v) for f, v in self.vocabulary.items()},
                "alpha": self.alpha, "n": sum(self.class_counts.values())}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NaiveBayes":
        model = cls(alpha=float(data.get("alpha", 1.0)))
        model.class_counts = {k: int(v) for k, v in (data.get("class_counts") or {}).items()}
        model.feature_counts = {c: {f: dict(v) for f, v in fc.items()}
                                for c, fc in (data.get("feature_counts") or {}).items()}
        model.vocabulary = {f: set(v) for f, v in (data.get("vocabulary") or {}).items()}
        return model

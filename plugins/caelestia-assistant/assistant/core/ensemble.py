"""cortex.ensemble — Learn++.NSE, the smooth-drift alternative to the
ADWIN consensus (exponential-build-4 D).

Elwell & Polikar, "Incremental Learning of Concept Drift in
Nonstationary Environments", IEEE Trans. Neural Networks 22(10), 2011:
where the ADWIN drift consensus detects a shift and (in this repo's
consensus-following design) marks the learned state as changed, the
NSE family instead keeps an ENSEMBLE of experts born at different
times and re-weights them by their error on a RECENT window: an expert
who was right lately gains say-so; an expert who was right only in the
old regime keeps its knowledge but loses its vote. Adaptation is
continuous re-weighting instead of a binary forget-everything event.

Members: each is a frozen weight-vector over the SAME feature space the
router's logistic model uses (a snapshot of the online model taken
every ``block`` examples — the paper's data-batch notion, adapted to a
stream of labeled routing decisions). Weighting is the paper's
dynamic-weight rule over the recent window of the last ``window``
labeled examples, floored so no expert is ever weighted below
``min_weight`` (the paper's ε̃ floor: old knowledge never fully dies —
that is the difference from a reset):

    e_k(S) = windowed error of expert k
    w_k  ∝  1 / (e_k + ε_floor)          (then L1-normalized)

Prediction: the weighted average of member outputs (logistic models
emit probabilities; the average is a probability — no invented
certainty). Members are capped: when the ensemble exceeds ``max_members``
the OLDEST (never the least accurate — age, deterministic) is dropped.

SELECTABLE, NOT A REPLACEMENT: the ADWIN consensus stays the default
drift machinery; this ships next to it so the two philosophies —
detect-and-reset vs re-weight-continuously — can be compared on the
same stream through the regret-vs-best-fixed audit discipline
(brain/regret.py) via ``compare_with_adwin``. The comparison reports
both verdicts with their own honesty notes; it never declares a winner
from one stream. Deterministic (no RNG); state is plain dicts for the
learned-state JSON path; nothing here writes or routes anything.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["LearnPPNSE", "compare_with_adwin"]

_EPS_FLOOR = 0.01
_DEFAULT_BLOCK = 25
_DEFAULT_WINDOW = 40
_DEFAULT_MAX_MEMBERS = 8


class LearnPPNSE:
    """The time-windowed expert ensemble over one labeled stream."""

    def __init__(self, n_features: int,
                 block: int = _DEFAULT_BLOCK,
                 window: int = _DEFAULT_WINDOW,
                 max_members: int = _DEFAULT_MAX_MEMBERS,
                 min_weight: float = 0.02) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if block < 1 or window < 1 or max_members < 1:
            raise ValueError("block/window/max_members must be >= 1")
        if not (0.0 <= min_weight < 1.0):
            raise ValueError("min_weight must be in [0, 1)")
        self.n_features = n_features
        self.block = block
        self.window = window
        self.max_members = max_members
        self.min_weight = min_weight
        # members: [{"weights": [...], "bias": float, "born_at": int}]
        self.members: List[Dict[str, Any]] = []
        # the labeled stream tail: (features, label)
        self.recent: List[Tuple[List[float], int]] = []
        self.n_seen = 0
        self._pending: List[Tuple[List[float], int]] = []

    # -- online model pieces (the same logistic the router uses) -------

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math_exp(-z))
        ez = math_exp(z)
        return ez / (1.0 + ez)

    def predict_member(self, member: Dict[str, Any],
                       features: Sequence[float]) -> float:
        z = member["bias"] + sum(w * x for w, x in
                                 zip(member["weights"], features))
        return self._sigmoid(z)

    def predict(self, features: Sequence[float]) -> Optional[float]:
        """The ensemble's weighted-average probability; None before the
        first block completes (no experts, no invented prediction)."""
        if not self.members:
            return None
        weights = self.member_weights()
        total = sum(weights)
        if total <= 0:
            return None
        acc = 0.0
        for member, w in zip(self.members, weights):
            acc += w * self.predict_member(member, features)
        return acc / total

    # -- the stream ------------------------------------------------------

    def observe(self, features: Sequence[float], label: int) -> Dict[str, Any]:
        """One labeled example. Every ``block`` examples the CURRENT
        online model's frozen snapshot is born as a new expert and the
        whole ensemble is re-weighted on the recent window."""
        row = ([float(x) for x in features], int(label))
        self.recent.append(row)
        self._pending.append(row)
        self.n_seen += 1
        born = None
        if len(self._pending) >= self.block:
            born = len(self.members)
            weights, bias = self._train_member(self._pending)
            self.members.append({
                "weights": weights, "bias": bias,
                "born_at": self.n_seen,
            })
            self._pending = []
            if len(self.members) > self.max_members:
                # age out the oldest — deterministic, never accuracy-based
                self.members = self.members[1:]
            self.reweight()
        return {"n_seen": self.n_seen, "members": len(self.members),
                "born": born is not None}

    @staticmethod
    def _train_member(rows: Sequence[Tuple[List[float], int]]
                      ) -> Tuple[List[float], float]:
        """A fresh binary logistic per block — the paper's own design
        ("a new classifier is trained on each batch of data"): fixed-
        iteration gradient descent, no RNG, no library. This is a
        snapshot expert, frozen at birth and never updated again."""
        width = len(rows[0][0])
        w = [0.0] * width
        b = 0.0
        rate = 0.3
        for _ in range(150):
            gw = [0.0] * width
            gb = 0.0
            for feats, label in rows:
                p = LearnPPNSE._sigmoid(
                    b + sum(wi * x for wi, x in zip(w, feats)))
                err = p - label
                for j, x in enumerate(feats):
                    gw[j] += err * x
                gb += err
            n = len(rows)
            w = [wi - rate * g / n for wi, g in zip(w, gw)]
            b -= rate * gb / n
        return [round(wi, 6) for wi in w], round(b, 6)

    # -- the paper's dynamic weighting -------------------------------------

    def member_weights(self) -> List[float]:
        """w_k ∝ 1/(windowed error + ε_floor), floored at min_weight,
        L1-normalized. The recent window is the paper's dynamic
        dataset S^(t); errors are recomputed EVERY call from the kept
        tail — no stale cached weights."""
        if not self.members:
            return []
        tail = self.recent[-self.window:]
        raw: List[float] = []
        for member in self.members:
            if not tail:
                raw.append(1.0)
                continue
            err = 0.0
            for feats, label in tail:
                p = self.predict_member(member, feats)
                err += abs(p - label)  # L1 loss on probabilities
            err /= len(tail)
            raw.append(1.0 / (err + _EPS_FLOOR))
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        weights = [max(self.min_weight, r / total) for r in raw]
        norm = sum(weights)
        return [w / norm for w in weights]

    def reweight(self) -> None:
        self.member_weights()

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"n_features": self.n_features, "block": self.block,
                "window": self.window, "max_members": self.max_members,
                "min_weight": self.min_weight, "members": self.members,
                "recent": self.recent[-self.window:],
                "n_seen": self.n_seen}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnPPNSE":
        ens = cls(n_features=int(d.get("n_features", 1)),
                  block=int(d.get("block", _DEFAULT_BLOCK)),
                  window=int(d.get("window", _DEFAULT_WINDOW)),
                  max_members=int(d.get("max_members",
                                        _DEFAULT_MAX_MEMBERS)),
                  min_weight=float(d.get("min_weight", 0.02)))
        ens.members = [dict(m) for m in d.get("members", [])]
        ens.recent = [tuple(r) for r in d.get("recent", [])]
        ens.n_seen = int(d.get("n_seen", 0))
        return ens


def math_exp(z: float) -> float:
    import math
    return math.exp(max(-700.0, min(700.0, z)))


def compare_with_adwin(stream: Sequence[int],
                       block: int = 10) -> Dict[str, Any]:
    """The honest side-by-side the selectable-mode contract asks for:
    the same stream through (a) the ADWIN drift consensus and (b) the
    NSE ensemble's weight trajectory. Features are the trivial 1-D
    [x] stream itself; "accuracy" is agreement with the stream's
    current label. The comparison reports both verdicts and the weight
    trajectory; it does NOT declare a winner — one stream is one
    stream, and the regret-vs-best-fixed audit is the deeper tool."""
    from .adwin import DriftConsensus

    consensus = DriftConsensus()
    for value in stream:
        consensus.update(bool(value))
    adwin_status = consensus.status()

    ens = LearnPPNSE(n_features=1, block=block,
                     window=max(10, len(stream) // 4))
    trajectory: List[Dict[str, Any]] = []
    for i, value in enumerate(stream):
        row = ens.observe([float(value)], int(value))
        if row["born"]:
            trajectory.append({"at": ens.n_seen,
                               "members": len(ens.members),
                               "weights": [round(w, 4) for w in
                                           ens.member_weights()]})
    return {
        "n": len(stream),
        "adwin": {"flagged": adwin_status.get("flag_drift"),
                  "summary": adwin_status.get("summary")},
        "nse": {"members": len(ens.members),
                "weights": [round(w, 4) for w in ens.member_weights()],
                "trajectory": trajectory},
        "note": "two philosophies on one stream: detect-and-reset "
                "(ADWIN consensus) vs continuous re-weighting (NSE). "
                "No winner is declared from one stream — run the "
                "regret-vs-best-fixed audit (brain/regret.py) for the "
                "policy comparison",
    }

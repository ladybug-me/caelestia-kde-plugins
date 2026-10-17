"""The self-learning loop: routing weights fitted from user decisions.

This is the module that makes the cortex IMPROVE with use, with three
classical mechanisms and zero neural networks:

1. ONLINE LOGISTIC REGRESSION (AdaGrad) — every routed request whose
   outcome is observed (applied/approved = positive; rejected =
   negative; clarified = weak negative) becomes one training example
   over the router's own per-candidate features (lex/sem/fuzz/noun/cue
   agreement). AdaGrad (Duchi et al. 2011) per-coordinate adaptive
   learning rates suit the sparse, differently-scaled features. The
   fitted weights map back onto ``RouterState`` (normalized so the
   router's score scale stays comparable), which means the NEXT route
   is literally scored by what this user accepted and rejected before.

2. CALIBRATION — Beta-Binomial posterior over the router's softmax
   confidence vs. actual acceptance (the brain's calibrate.py idea,
   applied to routing). ``calibrated_confidence`` reports the posterior
   mean; the pipeline surfaces it as the honest probability instead of
   the raw softmax p.

3. STRATEGY BANDIT — a Thompson-sampling bandit (brain/preset_bandit's
   NamedBandit, reused verbatim) over three fixed RouterState profiles
   (lexical-heavy / semantic-heavy / balanced). Each query is routed
   with the sampled profile's weights; the outcome rewards the arm.
   The router learns not just how much to trust each signal overall,
   but which SIGNAL MIX this user's phrasing style responds to.

Drift: acceptance-rate changes across time buckets reuse the brain's
MinHash-free set-diff idea — here a plain rate comparison with a
threshold, reported (not acted on silently) so the user can decide to
reset learning.

Persistence: plain dict inside the brain state (same atomic-write
path); this module does no file I/O. Deterministic: seeded RNG only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from assistant.capabilities.brain.preset_bandit import NamedBandit
from .router import RouterState

LEARN_VERSION = 1
MAX_EXAMPLES = 500

# Feature order (must match the router's feature dict keys).
# The six router features the student models see. "struct" (added in
# exponential-build-5 F2) is the structural lift: how much the deterministic
# layers (noun floor, pattern floors, coverage floor, Enabled prior, name
# bigram) raised a candidate above its pure weighted-feature score. Without
# it, structurally-floored candidates are invisible to the logistic student
# and the CART/logistic agreement gate degrades — the feature rows must
# describe what actually drove the score.
_FEATURES = ("lex", "sem", "fuzz", "noun", "cue", "struct")

# Fixed routing-strategy profiles for the bandit.
STRATEGY_PROFILES: Dict[str, RouterState] = {
    "lexical": RouterState(w_lex=0.55, w_sem=0.12, w_fuzz=0.10, w_noun=0.21, bias=0.02),
    "semantic": RouterState(w_lex=0.26, w_sem=0.42, w_fuzz=0.18, w_noun=0.12, bias=0.02),
    "balanced": RouterState(),  # the hand-set prior
}


# ---------------------------------------------------------------------------
# Online logistic regression (AdaGrad).
# ---------------------------------------------------------------------------


@dataclass
class OnlineLogistic:
    """5-feature logistic model with AdaGrad updates. Weights start at
    the identity prior (all 1.0 on the raw feature scale, bias 0) —
    i.e. "trust every signal equally until the user teaches otherwise"."""

    weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    bias: float = 0.0
    acc_sq: Tuple[float, ...] = (1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
    bias_acc_sq: float = 1e-6
    lr: float = 0.15
    examples: int = 0

    def predict(self, features: Dict[str, float]) -> float:
        z = self.bias
        for i, name in enumerate(_FEATURES):
            z += self.weights[i] * float(features.get(name, 0.0))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def update(self, features: Dict[str, float], label: int) -> None:
        """One AdaGrad SGD step: per-coordinate adaptive step size
        ``lr / sqrt(accumulated_squared_gradients)``."""
        p = self.predict(features)
        error = (label - p)  # d/dz of log loss
        weights = list(self.weights)
        acc = list(self.acc_sq)
        for i, name in enumerate(_FEATURES):
            x = float(features.get(name, 0.0))
            grad = -error * x
            acc[i] += grad * grad
            weights[i] -= self.lr * grad / math.sqrt(acc[i])
        self.bias_acc_sq += (error * error)
        self.bias += self.lr * error / math.sqrt(self.bias_acc_sq)
        self.weights = tuple(weights)
        self.acc_sq = tuple(acc)
        self.examples += 1

    def to_dict(self) -> Dict[str, object]:
        return {"weights": list(self.weights), "bias": self.bias,
                "acc_sq": list(self.acc_sq), "bias_acc_sq": self.bias_acc_sq,
                "lr": self.lr, "examples": self.examples}

    @staticmethod
    def from_dict(data: Optional[Dict[str, object]]) -> "OnlineLogistic":
        model = OnlineLogistic()
        if not isinstance(data, dict):
            return model
        try:
            model.weights = OnlineLogistic._fit_len(
                tuple(float(w) for w in data.get("weights", model.weights)))
            model.bias = float(data.get("bias", 0.0))
            model.acc_sq = OnlineLogistic._fit_len(
                tuple(float(a) for a in data.get("acc_sq", model.acc_sq)), 1e-6)
            model.bias_acc_sq = float(data.get("bias_acc_sq", 1e-6))
            model.lr = float(data.get("lr", 0.15))
            model.examples = int(data.get("examples", 0))
        except (TypeError, ValueError):
            return OnlineLogistic()
        return model

    @staticmethod
    def _fit_len(values: Tuple[float, ...], pad: float = 1.0) -> Tuple[float, ...]:
        """Pad/truncate a persisted weight vector to the current feature
        count — pre-struct learner states (5 weights) load with the new
        feature's weight at its identity prior instead of crashing."""
        n = len(_FEATURES)
        values = values[:n]
        return values + (pad,) * (n - len(values))


# ---------------------------------------------------------------------------
# Calibration (Beta-Binomial over confidence buckets).
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Beta(a, b) posterior over P(accept | router said ROUTED). A flat
    prior (1, 1) plus observed outcomes; ``mean`` is the posterior mean
    a + b / (alpha + beta) — the honest confidence after evidence."""

    alpha: float = 1.0
    beta: float = 1.0
    buckets: Dict[str, Dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.buckets is None:
            self.buckets = {}

    def observe(self, accepted: bool, bucket: str = "all") -> None:
        self.alpha += 1.0 if accepted else 0.0
        self.beta += 0.0 if accepted else 1.0
        row = self.buckets.setdefault(bucket, {"alpha": 1.0, "beta": 1.0})
        row["alpha"] += 1.0 if accepted else 0.0
        row["beta"] += 0.0 if accepted else 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def observations(self, bucket: str) -> int:
        """How many real outcomes landed in this bucket (the honesty
        gate for surfacing: small samples say nothing)."""
        row = self.buckets.get(bucket)
        if not row:
            return 0
        return int(row.get("alpha", 1.0) + row.get("beta", 1.0) - 2.0)

    def bucket_mean(self, bucket: str) -> float:
        row = self.buckets.get(bucket) or {"alpha": 1.0, "beta": 1.0}
        return row["alpha"] / (row["alpha"] + row["beta"])

    def to_dict(self) -> Dict[str, object]:
        return {"alpha": self.alpha, "beta": self.beta, "buckets": self.buckets}

    @staticmethod
    def from_dict(data: Optional[Dict[str, object]]) -> "Calibration":
        cal = Calibration()
        if not isinstance(data, dict):
            return cal
        try:
            cal.alpha = float(data.get("alpha", 1.0))
            cal.beta = float(data.get("beta", 1.0))
            cal.buckets = {str(k): {kk: float(vv) for kk, vv in v.items()}
                           for k, v in (data.get("buckets") or {}).items()}
        except (TypeError, ValueError):
            return Calibration()
        return cal


def confidence_bucket(p: float) -> str:
    """Softmax-probability bucket used as the calibration key."""
    if p >= 0.75:
        return "p75+"
    if p >= 0.5:
        return "p50-75"
    if p >= 0.3:
        return "p30-50"
    return "p30-"


def rolling_hit_rate(labels: Sequence[int], window: int = 5) -> List[float]:
    """The calibration bucket's rolling hit-rate: acceptance rate over
    each ``window`` consecutive routed outcomes, in arrival order — the
    same series shape the numeric telemetry feeds its changepoint
    detectors. Overlapping windows smooth single-outcome noise while
    preserving the level shift a real change in routing quality
    produces. Deterministic; a pure function of the label sequence."""
    if window < 1:
        raise ValueError("window must be >= 1")
    series: List[float] = []
    for t in range(window, len(labels) + 1):
        chunk = labels[t - window:t]
        series.append(sum(1 for v in chunk if v == 1) / float(window))
    return series


# ---------------------------------------------------------------------------
# The learner (facade over model + calibration + bandit + example log).
# ---------------------------------------------------------------------------


class CortexLearner:
    """Stateful learner, persisted as ONE plain dict by the caller:

    ``{"version": 1, "model": {...}, "calibration": {...},
       "bandit": {...}, "examples": [...], "drift": {...}}``

    ``examples`` is a bounded append-log of (features, label, surface)
    rows — kept so learning is auditable and replayable, the same
    honesty rule as the brain ledger."""

    def __init__(self, data: Optional[Dict[str, object]] = None) -> None:
        data = data if isinstance(data, dict) else {}
        self.model = OnlineLogistic.from_dict(data.get("model"))  # type: ignore[arg-type]
        self.calibration = Calibration.from_dict(data.get("calibration"))  # type: ignore[arg-type]
        self.bandit = NamedBandit.from_dict(data.get("bandit"))  # type: ignore[arg-type]
        raw_examples = data.get("examples") or []
        self.examples: List[Dict[str, object]] = [
            dict(row) for row in raw_examples if isinstance(row, dict)
        ][:MAX_EXAMPLES]
        self.accepts = int(data.get("accepts", 0)) or sum(  # type: ignore[arg-type]
            1 for row in self.examples if row.get("label") == 1
        )
        self.rejects = int(data.get("rejects", 0)) or sum(
            1 for row in self.examples if row.get("label") == 0
        )
        self._rng = random.Random(0xC0E7E)  # seeded: strategy choice is reproducible
        self.conservative_exploration = bool(
            data.get("conservative_exploration", False))
        self._epsilon = float(data.get("conservative_epsilon", 0.05))
        self.last_gate_report: Optional[Dict[str, object]] = None
        self.drift_mode = str(data.get("drift_mode", "consensus"))
        self._drift_state = data.get("drift_state") or {}

    # -- learning ----------------------------------------------------------

    def choose_strategy(self, conservative: Optional[bool] = None
                        ) -> Tuple[str, RouterState]:
        """Thompson-sample a routing strategy. Returns (name, state).

        exponential-build-4 D adds the SELECTABLE conservative mode
        (Wu et al. 2016): the same Thompson sample, gated by the safe-
        exploration floor in conservative.py — a non-baseline strategy
        plays only when its evidence clears the "balanced" floor's
        lower bound minus ε. The mode is a user choice (the data's
        ``conservative_exploration`` key), never a silent replacement:
        with the mode off this function is byte-for-byte the old
        Thompson path."""
        ranked = self.bandit.rank(list(STRATEGY_PROFILES), rng=self._rng)
        name = ranked[0][0] if ranked else "balanced"
        if conservative is None:
            conservative = bool(getattr(self, "conservative_exploration",
                                        False))
        if conservative:
            from . import conservative as cons_mod
            arms = {n: {"alpha": self.bandit.arms.get(n, [1.0, 1.0])[0],
                        "beta": self.bandit.arms.get(n, [1.0, 1.0])[1]}
                    for n in STRATEGY_PROFILES}
            gate = cons_mod.ConservativeBandit(
                arms=arms, safe_arm="balanced",
                epsilon=self._epsilon, delta=0.05)
            verdict = gate.choose(thompson_sample=name)
            self.last_gate_report = verdict
            name = verdict["arm"]
        return name, STRATEGY_PROFILES.get(name, RouterState())

    def observe(self, text: str, surface: str, features: Dict[str, float],
                p: float, outcome: str) -> None:
        """Record one routed outcome and learn from it. Outcome mapping:
        applied/approved -> label 1; rejected -> label 0; clarified and
        abstain -> weak negative 0 (with weight reflected only in the
        calibration, not the model — ambiguous phrasings are not the
        model's fault)."""
        if outcome in ("applied", "approved"):
            label = 1
        elif outcome == "rejected":
            label = 0
        elif outcome in ("clarified", "abstain"):
            label = 0
        elif outcome == "ambiguous":
            self.calibration.observe(False, confidence_bucket(p))
            return
        else:
            return
        self.model.update(features, label)
        self.calibration.observe(label == 1, confidence_bucket(p))
        if label == 1:
            self.accepts += 1
        else:
            self.rejects += 1
        self.examples.append({"text": text[:120], "surface": surface,
                              "features": {k: round(float(v), 4) for k, v in features.items()},
                              "p": round(float(p), 4), "label": label,
                              "outcome": outcome})
        if len(self.examples) > MAX_EXAMPLES:
            self.examples = self.examples[-MAX_EXAMPLES:]

    def reward_strategy(self, name: str, accepted: bool) -> None:
        self.bandit.reward(name, accepted)

    def calibration_note(self, p: float,
                         min_observations: int = 5) -> Optional[str]:
        """The USER-VISIBLE calibration sentence for a route scored p:
        'routes scored like this one were right ~92% of the time (25
        decisions)' — the observed acceptance rate of this confidence
        bucket, surfaced instead of kept internal. None when the bucket
        has too few observations to say anything honest."""
        bucket = confidence_bucket(p)
        n = self.calibration.observations(bucket)
        if n < min_observations:
            return None
        rate = self.calibration.bucket_mean(bucket)
        return (f"routes scored like this one were right "
                f"~{round(rate * 100)}% of the time ({n} decisions)")

    def calibrated_confidence(self, p: float) -> float:
        """Posterior-mean acceptance probability for this softmax band —
        the honest confidence the pipeline reports."""
        return round(self.calibration.bucket_mean(confidence_bucket(p)), 3)

    # -- router-state adaptation -------------------------------------------

    def router_state(self) -> RouterState:
        """The learned RouterState: the sampled strategy profile's
        weights rescaled by the fitted logistic weights (normalized to
        keep the score scale stable). Falls back to the strategy profile
        alone until the model has seen enough evidence to mean anything
        (>= 12 examples)."""
        _name, base = self.choose_strategy()
        if self.model.examples < 12:
            return base
        fitted = [max(0.0, w) for w in self.model.weights]  # negative weights are uninterpretable here
        total = sum(fitted)
        if total <= 0:
            return base
        fitted = [w / total for w in fitted]
        # map [lex, sem, fuzz, noun, cue-kind, struct] onto the
        # RouterState weight slots; 'cue' and 'struct' fold into w_noun
        # (all three are the structural/agreement signals).
        w_lex = fitted[0]
        w_sem = fitted[1]
        w_fuzz = fitted[2]
        w_noun = fitted[3] + fitted[4]
        if len(fitted) > 5:
            w_noun += fitted[5]
        # renormalize the four slots to the strategy profile's total mass
        mass = base.w_lex + base.w_sem + base.w_fuzz + base.w_noun
        slot_total = w_lex + w_sem + w_fuzz + w_noun
        if slot_total <= 0:
            return base
        scale = mass / slot_total
        return RouterState(
            w_lex=round(w_lex * scale, 4), w_sem=round(w_sem * scale, 4),
            w_fuzz=round(w_fuzz * scale, 4), w_noun=round(w_noun * scale, 4),
            bias=base.bias, min_score=base.min_score,
            min_margin=base.min_margin, temperature=base.temperature,
        )

    # -- drift ---------------------------------------------------------------

    def drift_check(self) -> Dict[str, object]:
        """Acceptance-rate drift across example halves (old vs new). A
        report, not an action: the CLI surfaces it; only the user resets."""
        if len(self.examples) < 10:
            return {"status": "insufficient-data", "examples": len(self.examples)}
        half = len(self.examples) // 2
        old = self.examples[:half]
        new = self.examples[half:]
        rate_old = sum(1 for r in old if r.get("label") == 1) / len(old)
        rate_new = sum(1 for r in new if r.get("label") == 1) / len(new)
        delta = rate_new - rate_old
        status = "stable"
        if abs(delta) >= 0.25:
            status = "drift-detected"
        return {"status": status, "old_rate": round(rate_old, 3),
                "new_rate": round(rate_new, 3), "delta": round(delta, 3),
                "examples": len(self.examples)}

    def bocpd_drift_check(self, window: int = 5,
                          cp_threshold: float = 0.5) -> Dict[str, object]:
        """The probabilistic complement to :meth:`drift_check`: the
        example log's rolling hit-rate, fed through
        ``genius.data.bocpd`` (Adams & MacKay 2007) — the SAME BOCPD
        primitive the numeric telemetry already uses, lazily imported
        the same way dispatch.py lazily imports the k-means — so a real
        shift in routing accuracy is flagged the same honest way a
        shift in CPU load is. A report, not an action: the CLI surfaces
        it, only the user resets. Thin data says nothing (labelled
        ``insufficient-data`` — with fewer than window+3 outcomes the
        series is shorter than BOCPD's own n >= 4 floor), and the
        changepoint probability is REPORTED with its threshold, never
        claimed as certainty. Deterministic: no RNG anywhere on the
        path."""
        labels = [int(row.get("label", 0)) for row in self.examples
                  if isinstance(row, dict)]
        series = rolling_hit_rate(labels, window)
        if len(series) < 4:
            return {"status": "insufficient-data",
                    "examples": len(self.examples),
                    "series_points": len(series)}
        from assistant.capabilities.genius.data import bocpd  # the ONE BOCPD implementation
        result = bocpd(series)
        probs = result["changepoint_prob"]
        best_t = max(range(len(probs)), key=lambda t: (probs[t], -t))
        best_p = probs[best_t]
        means = result["segment_mean"]
        return {
            "status": ("drift-detected" if best_p >= cp_threshold
                       else "stable"),
            "examples": len(self.examples),
            "window": window,
            "series_points": len(series),
            "max_changepoint_prob": best_p,
            "threshold": cp_threshold,
            "at_series_step": best_t,
            "segment_mean_before": (means[best_t - 1] if best_t > 0
                                    else None),
            "segment_mean_after": means[best_t],
            "algorithm": result["algorithm"],
        }

    def regret_audit(self) -> Dict[str, object]:
        """Exponential-build 3.3: the strategy bandit's cumulative
        reward against the best-fixed-arm-in-hindsight baseline — a
        printed estimate over unobserved rounds, never acted on."""
        from assistant.capabilities.brain.regret import audit_from_arms
        return audit_from_arms(self.bandit.arms)

    def ph_adwin_consensus(self) -> Dict[str, object]:
        """The SECOND drift opinion (exponential-build 3, item B2): the
        example log's accept/reject stream replayed through BOTH
        incremental detectors — PageHinkleyDrift (cortex/conformal.py,
        untouched) and ADWIN (cortex/adwin.py, Bifet & Gavaldà 2007) —
        under the consensus rule: drift is FLAGGED to the user only
        when BOTH detectors have alarmed (dual agreement suppresses
        single-detector false alarms); each detector's own state is
        always reported individually. Read-only replay exactly like
        bocpd_drift_check: nothing is persisted, nothing acts, and
        Page-Hinkley's own behaviour is unchanged. Deterministic: pure
        arithmetic over the examples in arrival order."""
        from .adwin import DriftConsensus  # lazy, like bocpd above
        labels = [row.get("label") for row in self.examples
                  if isinstance(row, dict)]
        pair = DriftConsensus()
        for label in labels:
            pair.update(label == 1)
        report = pair.status()
        report["examples"] = len(labels)
        # exponential-build-4 D: the SELECTABLE smooth-drift alternative
        # runs on the same stream when the data asks for it — the NSE
        # ensemble re-weights experts by recent accuracy instead of the
        # consensus' detect-and-reset. Both verdicts are reported side
        # by side with the honest no-winner note; the consensus stays
        # the default and nothing acts on either.
        if getattr(self, "drift_mode", "consensus") == "nse":
            from .ensemble import LearnPPNSE
            nse = LearnPPNSE.from_dict(
                (self._drift_state or {}).get("nse", {}))
            if nse.n_features < 1:
                nse = LearnPPNSE(n_features=1)
            for row in (r for r in self.examples
                        if isinstance(r, dict)):
                feats = (row.get("features") or {})
                nse.observe([float(len(feats))], int(row.get("label") or 0))
            report["nse_alternative"] = {
                "members": len(nse.members),
                "weights": [round(w, 4) for w in nse.member_weights()],
                "note": "Learn++.NSE (Elwell & Polikar 2011): continuous "
                        "re-weighting, not reset; no winner is declared "
                        "here — compare via the regret audit",
            }
        return report

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": LEARN_VERSION,
            "model": self.model.to_dict(),
            "calibration": self.calibration.to_dict(),
            "bandit": self.bandit.to_dict(),
            "examples": self.examples,
            "accepts": self.accepts,
            "rejects": self.rejects,
        }

    # -- reporting ----------------------------------------------------------

    def report(self) -> Dict[str, object]:
        return {
            "examples": len(self.examples),
            "accepts": self.accepts,
            "rejects": self.rejects,
            "acceptance_rate": round(self.accepts / max(1, self.accepts + self.rejects), 3),
            "calibration": {
                "overall": round(self.calibration.mean, 3),
                "buckets": {b: round(self.calibration.bucket_mean(b), 3)
                            for b in ("p75+", "p50-75", "p30-50", "p30-")},
            },
            "strategy_arms": {
                name: {
                    "mean": round(alpha / (alpha + beta), 3),
                    "draws": int(alpha + beta - 2),  # observations beyond the flat prior
                }
                for name, (alpha, beta) in sorted(self.bandit.arms.items())
            },
            "drift": self.drift_check(),
            "drift_bocpd": self.bocpd_drift_check(),
            # exponential-build 3 item B2: the dual-detector consensus
            # (Page-Hinkley + ADWIN) — read-only replay, flag only on
            # dual agreement.
            "drift_ph_adwin": self.ph_adwin_consensus(),
            # exponential-build 3.3: did the strategy bandit beat
            # always-playing its single best arm? printed, not acted on.
            "regret": self.regret_audit(),
            "fitted_weights": {
                name: round(w, 4) for name, w in zip(_FEATURES, self.model.weights)
            },
        }


# ---------------------------------------------------------------------------
# Batch-curated active learning (issue #120 Phase 4.3): near-threshold
# phrases are LOGGED for batch review instead of being learned silently.
#
# The router's own cutoffs live in RouterState (router.py): min_score
# (default 0.30) gates ABSTAIN and min_margin (default 0.06) gates
# AMBIGUOUS. A phrase that falls below either is exactly the evidence an
# active learner wants — but learning from it online would silently bend
# the router toward phrases nobody confirmed. So the candidate is parked
# in the state dict (key "cortex_review", bounded), and `cortex review`
# surfaces the batch for the user to label (teaching outcome "applied"
# for the named surface) or dismiss. Nothing here routes, writes, or
# learns by itself: pure list/dict transforms over caller-owned state.
# ---------------------------------------------------------------------------

REVIEW_KEY = "cortex_review"
MAX_CANDIDATES = 50


def log_review_candidate(bucket: List[Dict[str, object]], text: str,
                         verdict: str, candidates: List[Dict[str, object]],
                         at: str) -> List[Dict[str, object]]:
    """Append one near-threshold phrase to the review bucket (pure).

    ``bucket`` is the caller's list (state[REVIEW_KEY]); identical texts
    are deduplicated (the newest occurrence wins). Bounded at
    MAX_CANDIDATES, oldest evicted.
    """
    if verdict not in ("ABSTAIN", "AMBIGUOUS", "OUT_OF_ONTOLOGY"):
        return bucket
    # ``at`` is declared str, but two call sites (cli.py, dispatch.py)
    # pass the live ``_now()`` datetime — coerce once, here, so the
    # bucket never carries a non-JSON object into brain_state.save
    # (the pre-existing chat-exit crash this pins: datetime is not
    # JSON serializable).
    if not isinstance(at, str):
        at = at.isoformat() if hasattr(at, "isoformat") else str(at)
    top = candidates[0] if candidates else None
    entry: Dict[str, object] = {
        "text": text[:120],
        "verdict": verdict,
        "surface": top.get("surface") if top else None,
        "p": top.get("p") if top else None,
        "at": at,
    }
    kept = [c for c in bucket if c.get("text") != entry["text"]]
    kept.insert(0, entry)
    return kept[:MAX_CANDIDATES]


def is_near_threshold(verdict: str) -> bool:
    """The verdicts that mark a near-threshold phrase (router.py's own
    ABSTAIN/AMBIGUOUS gates: below min_score or below min_margin)."""
    return verdict in ("ABSTAIN", "AMBIGUOUS")


def label_candidate(state: Dict[str, object], index: int, surface: str,
                    learner: "CortexLearner") -> Dict[str, object]:
    """Teach the router one reviewed phrase: text -> surface.

    The features come from a fresh route of the text (deterministic); if
    the router currently maps the phrase to a different surface, the
    labeled surface gets an empty feature vector — the update still moves
    the bias honestly instead of inventing agreement. The labeled
    candidate is REMOVED from the bucket (it is no longer outstanding).
    """
    from .router import route as _route

    bucket = list(state.get(REVIEW_KEY, []))
    if index < 0 or index >= len(bucket):
        raise ValueError(f"no review candidate {index} (0..{len(bucket) - 1})")
    entry = bucket.pop(index)
    text = str(entry["text"])
    fresh = _route(text)
    top = fresh.top
    if top is not None and top.surface == surface and top.p is not None:
        p = float(top.p)
    else:
        p = float(entry.get("p") or 0.0)
    features = fresh.features.get(surface, {})
    learner.observe(text, surface, features, p, "applied")
    state[REVIEW_KEY] = bucket
    return {"labeled": True, "text": text, "surface": surface,
            "remaining": len(bucket)}


def dismiss_candidate(state: Dict[str, object], index: int) -> Dict[str, object]:
    """Drop one candidate from the bucket without teaching anything."""
    bucket = list(state.get(REVIEW_KEY, []))
    if index < 0 or index >= len(bucket):
        raise ValueError(f"no review candidate {index} (0..{len(bucket) - 1})")
    removed = bucket.pop(index)
    state[REVIEW_KEY] = bucket
    return {"dismissed": True, "text": str(removed["text"]),
            "remaining": len(bucket)}

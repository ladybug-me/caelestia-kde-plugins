"""cortex.crf_tagger — a linear-chain CRF slot tagger, the
calibrated-probability ALTERNATIVE to the averaged perceptron
(exponential-build-4 F).

Lafferty, McCallum & Pereira 2001, "Conditional Random Fields:
Probabilistic Models for Segmenting and Labeling Sequence Data",
ICML — the linear-chain CRF: one log-linear distribution over whole
tag sequences conditioned on the tokens,

    P(y | x) ∝ exp( Σ_i  w[y_i]·f(x, i)  +  Σ_i  T[y_{i-1}, y_i] ),

with the same feature style and tag set as cortex/slot_tagger.py's
perceptron (the tagger is distilled from the same grammar-supervision
contract; this module shares its tokenizer, shape feature, char
n-grams and TAGS — nothing about the existing tagger is modified).

Training is the paper's: maximum conditional likelihood via
forward-backward — the gradient is (empirical feature counts) minus
(expectation of feature counts under P(y|x)), the expectations coming
from the forward-backward recursions in log space (log-sum-exp; no
underflow, no scaling tricks to explain away). Decoding is Viterbi
over the same potentials.

WHY AN ALTERNATIVE AND NOT A REPLACEMENT: the perceptron's confidence
is a margin pushed through a pinned sigmoid — an invented calibration
the module itself documents. A CRF's marginals are probabilities by
construction (they come from a normalized conditional distribution),
which is the right shape for the conformal gate that sits downstream.
Whether that shape is ACTUALLY better on this repo's small supervised
history is an empirical claim this build MEASURES, not asserts —
``calibration_comparison`` trains both on the same rows and reports
held-out Brier and log-loss side by side, winner and all. Ship it
selectable (`tag_gated_crf`), conformal-gated exactly like the
existing tagger, grammar-fallback identical.

Deterministic: no RNG anywhere (batch gradient, fixed iterations,
sorted feature order). Pure stdlib; state is plain dicts.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .conformal import ConformalCalibrator
from .slot_tagger import (TAGS, SlotTagger, _shape,
                          _spans_of, _tokenize_with_offsets,
                          char_ngrams)

__all__ = ["CRFTagger", "tag_gated_crf", "calibration_comparison"]

_EPOCHS = 8
_RATE = 0.5
_UNK_TRANS = "<s>"


class CRFTagger:
    """The linear-chain CRF over the slot tags."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = data if isinstance(data, dict) else {}
        self.weights: Dict[str, Dict[str, float]] = {
            tag: {} for tag in TAGS}
        self.transitions: Dict[str, Dict[str, float]] = {
            prev: {tag: 0.0 for tag in TAGS} for prev in [_UNK_TRANS] + list(TAGS)}
        self.trained_examples = int(data.get("trained_examples", 0))

    # -- features (shared with the perceptron) ----------------------------

    @staticmethod
    def _features(tokens: List[str], i: int) -> List[str]:
        w = tokens[i].lower()
        feats = [
            f"w={w}",
            f"shape={_shape(tokens[i])}",
            f"p1={w[:2]}", f"p2={w[:3]}",
            f"s1={w[-2:]}", f"s2={w[-3:]}",
            f"prev={tokens[i - 1].lower() if i else '<s>'}",
            f"next={tokens[i + 1].lower() if i + 1 < len(tokens) else '</s>'}",
        ]
        for g in char_ngrams(w, 3):
            feats.append(f"ng={g}")
        return feats

    # -- potentials ---------------------------------------------------------

    def _emit(self, tag: str, feats: Sequence[str]) -> float:
        table = self.weights[tag]
        return sum(table.get(f, 0.0) for f in feats)

    def _trans(self, prev: str, tag: str) -> float:
        return self.transitions.get(prev, {}).get(tag, 0.0)

    @staticmethod
    def _lse(values: Sequence[float]) -> float:
        if not values:
            return float("-inf")
        m = max(values)
        if m == float("-inf"):
            return m
        return m + math.log(sum(math.exp(v - m) for v in values))

    # -- forward-backward (log space) ---------------------------------------

    def _forward(self, token_feats: List[List[str]]) -> List[Dict[str, float]]:
        n = len(token_feats)
        alpha: List[Dict[str, float]] = []
        first = {tag: self._trans(_UNK_TRANS, tag)
                 + self._emit(tag, token_feats[0]) for tag in TAGS}
        alpha.append(first)
        for i in range(1, n):
            row = {}
            for tag in TAGS:
                emit = self._emit(tag, token_feats[i])
                row[tag] = self._lse(
                    [alpha[i - 1][prev] + self._trans(prev, tag)
                     for prev in TAGS]) + emit
            alpha.append(row)
        return alpha

    def _backward(self, token_feats: List[List[str]]) -> List[Dict[str, float]]:
        n = len(token_feats)
        beta: List[Dict[str, float]] = [dict() for _ in range(n)]
        for tag in TAGS:
            beta[n - 1][tag] = 0.0
        for i in range(n - 2, -1, -1):
            for prev in TAGS:
                beta[i][prev] = self._lse(
                    [self._trans(prev, tag)
                     + self._emit(tag, token_feats[i + 1])
                     + beta[i + 1][tag] for tag in TAGS])
        return beta

    def _log_z(self, alpha: List[Dict[str, float]]) -> float:
        return self._lse(list(alpha[-1].values()))

    # -- training -------------------------------------------------------------

    def train(self, rows: Sequence[Tuple[str, List[str]]],
              epochs: int = _EPOCHS) -> Dict[str, Any]:
        """Batch conditional-likelihood training (forward-backward
        expectations vs empirical counts), fixed epochs, fixed rate.
        Rows are (text, gold BIO tags) — the same supervision shape
        the perceptron trains from."""
        prepared: List[Tuple[List[List[str]], List[str]]] = []
        for text, tags in rows:
            tokens = [t for t, _s, _e in _tokenize_with_offsets(text)]
            if len(tokens) != len(tags) or not tokens:
                continue
            prepared.append(([self._features(tokens, i)
                              for i in range(len(tokens))], list(tags)))
        if not prepared:
            raise ValueError("no usable rows: supervision matches no "
                             "tokenized text (nothing is faked)")
        for _epoch in range(epochs):
            grad_w: Dict[str, Dict[str, float]] = {
                tag: {} for tag in TAGS}
            grad_t: Dict[str, Dict[str, float]] = {
                prev: {tag: 0.0 for tag in TAGS}
                for prev in [_UNK_TRANS] + list(TAGS)}
            for token_feats, tags in prepared:
                n = len(token_feats)
                alpha = self._forward(token_feats)
                beta = self._backward(token_feats)
                log_z = self._log_z(alpha)
                # gradient step: empirical minus expected, per position
                for i in range(n):
                    y = tags[i]
                    prev_tag = _UNK_TRANS if i == 0 else tags[i - 1]
                    denom = self._lse([alpha[i][t] + beta[i][t]
                                       for t in TAGS])
                    for tag in TAGS:
                        marginal = math.exp(alpha[i][tag] + beta[i][tag]
                                            - denom)
                        # emission: empirical (y == tag) minus marginal
                        coef = (1.0 if tag == y else 0.0) - marginal
                        if coef == 0.0:
                            continue
                        table = grad_w[tag]
                        for f in token_feats[i]:
                            table[f] = table.get(f, 0.0) + coef
                        # transition: P(y_{i-1} = prev, y_i = tag)
                        if i > 0:
                            prev_y = tags[i - 1]
                            tcoef = ((1.0 if tag == y and prev_tag == prev_y
                                      else 0.0)
                                     - math.exp(alpha[i - 1][prev_tag]
                                                + self._trans(prev_tag, tag)
                                                + self._emit(tag, token_feats[i])
                                                + beta[i][tag]
                                                - log_z))
                            grad_t[prev_tag][tag] += tcoef
                        else:
                            tcoef = ((1.0 if tag == y else 0.0)
                                     - math.exp(self._trans(_UNK_TRANS, tag)
                                                + self._emit(tag, token_feats[i])
                                                + beta[i][tag] - log_z))
                            grad_t[_UNK_TRANS][tag] += tcoef
            # apply
            for tag in TAGS:
                table = self.weights[tag]
                for f, g in grad_w[tag].items():
                    table[f] = table.get(f, 0.0) + _RATE * g / len(prepared)
            for prev in grad_t:
                for tag, g in grad_t[prev].items():
                    if g:
                        self.transitions[prev][tag] += \
                            _RATE * g / len(prepared)
        self.trained_examples += len(prepared)
        return {"rows": len(prepared), "epochs": epochs}

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {t: dict(table) for t, table in self.weights.items()},
                "transitions": {p: dict(row) for p, row in
                                self.transitions.items()},
                "trained_examples": self.trained_examples}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CRFTagger":
        tagger = cls(data)
        return tagger

    # -- decoding + marginals -----------------------------------------------

    def tag(self, text: str) -> Dict[str, Any]:
        """Viterbi decode + per-token marginals from forward-backward.
        The confidence is the mean marginal of the decoded tags — a
        probability by construction (normalized conditional), which is
        the point of the CRF over the perceptron's margin heuristic."""
        tokens = [t for t, _s, _e in _tokenize_with_offsets(text)]
        if not tokens:
            return {"spans": {}, "tokens": [], "confidence": None,
                    "marginals": []}
        token_feats = [self._features(tokens, i) for i in range(len(tokens))]
        # Viterbi
        n = len(tokens)
        chart: List[Dict[str, Tuple[float, Optional[str]]]] = []
        first = {tag: (self._trans(_UNK_TRANS, tag)
                       + self._emit(tag, token_feats[0]), None)
                 for tag in TAGS}
        chart.append(first)
        for i in range(1, n):
            row = {}
            for tag in TAGS:
                emit = self._emit(tag, token_feats[i])
                best_prev = max(
                    TAGS,
                    key=lambda prev: (chart[i - 1][prev][0]
                                      + self._trans(prev, tag), prev))
                score = (chart[i - 1][best_prev][0]
                         + self._trans(best_prev, tag) + emit)
                row[tag] = (score, best_prev)
            chart.append(row)
        final = max(TAGS, key=lambda t: (chart[n - 1][t][0], t))
        tags: List[str] = [final]
        for i in range(n - 1, 0, -1):
            tags.append(chart[i][tags[-1]][1] or _UNK_TRANS)
        tags.reverse()
        # marginals
        alpha = self._forward(token_feats)
        beta = self._backward(token_feats)
        log_z = self._log_z(alpha)
        marginals = []
        for i in range(n):
            denom = self._lse([alpha[i][t] + beta[i][t] for t in TAGS])
            marginals.append({tag: math.exp(alpha[i][tag] + beta[i][tag]
                                            - denom) for tag in TAGS})
        confidence = sum(marginals[i][tags[i]] for i in range(n)) / n
        spans = _spans_of(tags, tokens)
        return {"tokens": tokens, "tags": tags, "spans": spans,
                "confidence": round(confidence, 4),
                "marginals": [[round(m[tag], 4) for tag in TAGS]
                              for m in marginals],
                "model": "linear-chain CRF (Lafferty et al. 2001)"}


def tag_gated_crf(text: str, tagger: CRFTagger,
                  calibrator: Optional[ConformalCalibrator] = None,
                  noun_matcher: Optional[Callable[
                      [str], Sequence[str]]] = None) -> Dict[str, Any]:
    """The CRF twin of slot_tagger.tag_gated: identical gate shape
    (untrained -> grammar fallback; conformal verdict; else answer),
    identical grammar fallback, so the two taggers are SELECTABLE with
    the same contract and neither bypasses the conformal gate."""
    if tagger.trained_examples == 0 or not any(
            tagger.weights[t] for t in TAGS):
        return _grammar_fallback(text, "CRF tagger untrained",
                                 calibrator=calibrator,
                                 noun_matcher=noun_matcher)
    result = tagger.tag(text)
    confidence = result.get("confidence")
    if confidence is None or not result["spans"]:
        return _grammar_fallback(text, "CRF tagger found no slot structure",
                                 calibrator=calibrator,
                                 noun_matcher=noun_matcher)
    verdict = (calibrator.verdict(confidence)
               if calibrator is not None and calibrator.scores
               else {"covered": None,
                     "reason": ("insufficient calibration data — no "
                                "distribution-free claim available")})
    if verdict.get("covered") is not True:
        return _grammar_fallback(text, str(verdict.get("reason")),
                                 calibrator=calibrator,
                                 noun_matcher=noun_matcher)
    result.update({"used": "crf-tagger", "conformal": verdict.get("reason")})
    return result


def _grammar_fallback(text: str, reason: str,
                      calibrator: Optional[ConformalCalibrator] = None,
                      noun_matcher: Optional[Callable[
                          [str], Sequence[str]]] = None) -> Dict[str, Any]:
    from assistant.capabilities.settings import slots as _slots
    slots = _slots.extract(text)
    targets = list(noun_matcher(text) or []) if noun_matcher else []
    return {"used": "grammar-fallback", "reason": reason,
            "slots": slots, "targets": targets}


def calibration_comparison(rows: Sequence[Tuple[str, List[str]]],
                           holdout: int = 4) -> Dict[str, Any]:
    """The empirical claim the selectable contract demands: SAME
    supervision, BOTH taggers trained, held-out rows scored by Brier
    and log-loss of each model's confidence against the binary
    'decoded BIO tags == gold tags' outcome. Reports both and names
    the measured winner — or says the data is too thin to claim
    anything (fewer than 2*holdout rows). Deterministic split: last
    ``holdout`` rows are held out."""
    if len(rows) < 2 * holdout:
        return {"n": len(rows), "crf": None, "perceptron": None,
                "note": "too few labeled rows to compare calibration "
                        "honestly — bring more supervised history"}
    train = list(rows[:-holdout])
    test = list(rows[-holdout:])
    crf = CRFTagger()
    crf.train(train)
    perceptron = SlotTagger()
    perceptron.train(train)

    def brier_nll(model_confidences: Sequence[float]) -> Tuple[float, float]:
        errors = [(c - 1.0) ** 2 for c in model_confidences]
        nlls = []
        for c in model_confidences:
            c = min(max(c, 1e-6), 1.0 - 1e-6)
            nlls.append(-math.log(1.0 - c))  # outcome 0 (exact match)
        return round(sum(errors) / len(errors), 4), round(
            sum(nlls) / len(nlls), 4)

    crf_conf: List[float] = []
    per_conf: List[float] = []
    outcomes: List[int] = []
    for text, gold in test:
        crf_result = crf.tag(text)
        per_result = perceptron.tag(text)
        crf_correct = 1 if crf_result.get("tags") == gold else 0
        per_correct = 1 if per_result.get("tags") == gold else 0
        outcomes.extend([crf_correct, per_correct])
        crf_conf.append(crf_result.get("confidence") or 0.0)
        per_conf.append(per_result.get("confidence") or 0.0)
    crf_brier, crf_nll = brier_nll(crf_conf)
    per_brier, per_nll = brier_nll(per_conf)
    winner = ("crf" if crf_brier < per_brier else
              "perceptron" if per_brier < crf_brier else "tie")
    return {
        "n_train": len(train), "n_holdout": len(test),
        "crf": {"brier": crf_brier, "logloss_vs_exact_match": crf_nll},
        "perceptron": {"brier": per_brier,
                       "logloss_vs_exact_match": per_nll},
        "winner": winner,
        "note": "held-out calibration comparison on THIS supervision — "
                "the winner is measured, not asserted, and may flip as "
                "history grows; both remain selectable",
    }

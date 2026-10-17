"""agent.clarify — dialogue by information gain, no language model.

When the goal split is AMBIGUOUS, the cheapest intelligent move is one good
question. This module picks it the 20-questions way:

  1. candidate hypotheses carry priors (from the goal classifier's scores);
  2. every question partitions the hypothesis set (its answers have known
     likelihoods per hypothesis);
  3. pick the question with the highest EXPECTED information gain
     I(Q) = H(prior) - E_answer[H(posterior | answer)];
  4. stop asking when the top posterior clears `stop_threshold` (0.8) or the
     best gain is below `min_gain` — a question that cannot change anything
     is noise, not intelligence.

Deterministic, pure, and honest: with no discriminating questions left it
says so instead of interrogating the user for theatre.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["EntropyPicker", "Question"]


class Question:
    """A question whose answers partition the hypothesis space.

    `answers`: {answer_label: {hypothesis_id: likelihood}} — likelihoods are
    relative per answer and renormalised; hypotheses absent from an answer's
    map get likelihood 0 for that answer (the answer rules them out)."""

    def __init__(self, qid: str, text: str, answers: Dict[str, Dict[str, float]],
                 resolves: str = "") -> None:
        self.qid = qid
        self.text = text
        self.answers = answers
        self.resolves = resolves  # which classifier distinction this refines


def _entropy(dist: Dict[str, float]) -> float:
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for p in dist.values():
        pr = p / total
        if pr > 0:
            h -= pr * math.log2(pr)
    return h


class EntropyPicker:
    def __init__(self, hypotheses: Dict[str, float],
                 stop_threshold: float = 0.8, min_gain: float = 0.25) -> None:
        """hypotheses: {id: prior weight} (unnormalised is fine)."""
        total = sum(hypotheses.values()) or 1.0
        self.belief: Dict[str, float] = {
            k: v / total for k, v in hypotheses.items()}
        self.stop_threshold = stop_threshold
        self.min_gain = min_gain
        self.asked: List[str] = []

    # ------------------------------------------------------------------
    def top(self) -> Tuple[str, float]:
        h, p = max(self.belief.items(), key=lambda kv: kv[1])
        return h, p

    def settled(self) -> bool:
        _h, p = self.top()
        return p >= self.stop_threshold

    # ------------------------------------------------------------------
    def expected_gain(self, q: Question) -> float:
        h0 = _entropy(self.belief)
        if h0 == 0.0:
            return 0.0
        expected = 0.0
        for label, likes in q.answers.items():
            post = {}
            for hyp, prior in self.belief.items():
                like = likes.get(hyp, 0.0)
                post[hyp] = prior * like
            z = sum(post.values())
            if z <= 0:
                continue  # impossible answer contributes nothing
            prob = z  # P(answer) under current belief
            norm = {k: v / z for k, v in post.items()}
            expected += prob * _entropy(norm)
        return h0 - expected

    def best_question(self, questions: Sequence[Question]) -> Optional[Question]:
        """The highest-gain unasked question, or None when asking cannot
        change the belief enough to be worth the user's time."""
        best: Optional[Question] = None
        best_gain = self.min_gain
        for q in questions:
            if q.qid in self.asked:
                continue
            gain = self.expected_gain(q)
            if gain > best_gain:
                best, best_gain = q, gain
        return best

    # ------------------------------------------------------------------
    def observe(self, q: Question, answer: str) -> Dict[str, float]:
        """Bayesian update with the observed answer; returns the posterior."""
        likes = q.answers.get(answer, {})
        post = {hyp: prior * likes.get(hyp, 0.0)
                for hyp, prior in self.belief.items()}
        z = sum(post.values())
        if z > 0:
            self.belief = {k: v / z for k, v in post.items()}
        self.asked.append(q.qid)
        return dict(self.belief)

    def state(self) -> Dict[str, Any]:
        h, p = self.top()
        return {
            "belief": {k: round(v, 4) for k, v in
                       sorted(self.belief.items(), key=lambda kv: -kv[1])},
            "top": h,
            "confidence": round(p, 4),
            "settled": self.settled(),
            "asked": list(self.asked),
        }

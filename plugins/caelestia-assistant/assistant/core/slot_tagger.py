"""cortex.slot_tagger — a LEARNED slot tagger layered on the existing
grammar, never replacing it.

What exists already (grep-first, none of it replaced):
- ``settings/slots.py`` — the compositional slot grammar
  ``{intensifier, target, direction, dimension}`` (regex-compositional,
  pure, honest declines);
- ``cortex/lexicon.py::char_ngrams`` — the char 3-gram features the
  router already uses for unknown words;
- ``cortex/conformal.py::ConformalCalibrator`` — the split-conformal
  gate over scores ("routes at or above t were accepted >= 1-alpha of
  the time"), the repo's one uncertainty mechanism.

The gap this closes: the grammar matches PHRASE LISTS; a phrasing it
never saw ("bar a smidge shorter, please") falls through. This module
trains a structured perceptron (Collins 2002, "Discriminative Training
Methods for Hidden Markov Models: Theory and Experiments with
Perceptron Algorithms", EMNLP) with Viterbi decoding and averaged
weights to tag token-level BIO slots, using ONLY local approve/reject
history — the same source cortex/learn.py trains from. No external
corpus exists anywhere in this module:

- APPROVED requests supervise: the gold BIO sequence is read off the
  EXISTING grammar's own output (slots.extract()'s matched raws; an
  optional caller-supplied noun matcher supplies TARGET spans) — the
  tagger distills the grammar the user actually accepted;
- REJECTED requests supervise NOTHING here (a request the user refused
  is not evidence of correct slot structure; it is skipped, never
  fabricated into a negative sequence).

The gate (the safety property): a 2-best Viterbi margin (the standard
N-best extension, Forney 1973) becomes the tagger's confidence, and the
conformal calibrator decides whether that confidence is covered. Not
covered — or NO calibration data at all — and the request falls back to
the existing grammar/char-ngram path with the calibrator's own reason
string attached. The tagger adds coverage; it never guesses.

Pure: no I/O, no clock, no randomness (fixed update order, averaged
weights); identical input bytes -> identical output. Persistence is a
plain dict for the caller to store in learned-state JSON (an existing
write path); this module writes nothing.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .conformal import ConformalCalibrator
from .lexicon import char_ngrams

__all__ = ["TAGS", "SlotTagger", "supervision_from_history", "MAX_TOKENS"]

# BIO tag set over the slots the existing grammar reports raw spans for
# (the intensifier kind has no raw span in slots.extract()'s output, so
# it stays outside the tag set rather than being guessed).
_TAGS = ["O"]
for _kind in ("CUE", "GENERIC", "TARGET"):
    _TAGS.append(f"B-{_kind}")
    _TAGS.append(f"I-{_kind}")
TAGS: Tuple[str, ...] = tuple(_TAGS)

MAX_TOKENS = 64  # a settings request is a sentence, not a paragraph

_WORD_RE = re.compile(r"[^\s]+")


def _tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        out.append((raw.strip(".,!?:;()\"'"), m.start(), m.end()))
    return [(w, a, b) for (w, a, b) in out if w]


def _shape(word: str) -> str:
    if word.isdigit():
        return "<num>"
    if any(c.isupper() for c in word):
        return "<cap>"
    return "<low>"


# ---------------------------------------------------------------------------
# The structured perceptron (Collins 2002; 2-best Viterbi, Forney 1973).
# ---------------------------------------------------------------------------

class SlotTagger:
    """Averaged structured perceptron over BIO slot tags.

    Features per position i: the word (lowercased), its shape, 2
    prefixes/2 suffixes, its char 3-grams (lexicon.char_ngrams — the
    router's own unknown-word feature), the previous and next words, and
    the (previous tag, tag) transition. Deterministic end to end."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data if isinstance(data, dict) else {}
        self.weights: Dict[str, Dict[str, float]] = {
            tag: {} for tag in TAGS}
        self._totals: Dict[str, Dict[str, float]] = {
            tag: {} for tag in TAGS}
        self._steps = 0
        self.updates = int(data.get("updates", 0))
        self.trained_examples = int(data.get("trained_examples", 0))
        saved = data.get("weights") or {}
        for tag, feats in saved.items():
            if tag in self.weights and isinstance(feats, dict):
                self.weights[tag] = {str(k): float(v)
                                     for k, v in feats.items()}
        self._averaged_cache: Optional[Dict[str, Dict[str, float]]] = None

    # -- features -------------------------------------------------------------

    def _features(self, tokens: List[str], i: int,
                  prev_tag: str) -> List[str]:
        w = tokens[i].lower()
        feats = [
            f"w={w}",
            f"shape={_shape(tokens[i])}",
            f"p1={w[:2]}", f"p2={w[:3]}",
            f"s1={w[-2:]}", f"s2={w[-3:]}",
            f"prev={tokens[i - 1].lower() if i else '<s>'}",
            f"next={tokens[i + 1].lower() if i + 1 < len(tokens) else '</s>'}",
            f"pt={prev_tag}",
        ]
        for g in char_ngrams(w, 3):
            feats.append(f"ng={g}")
        return feats

    def _score_tag(self, feats: List[str], tag: str,
                   use_averaged: bool) -> float:
        table = (self._averaged() if use_averaged else self.weights)[tag]
        return sum(table.get(f, 0.0) for f in feats)

    # -- decoding (2-best Viterbi) --------------------------------------------

    def _viterbi_2best(self, tokens: List[str], use_averaged: bool = True
                       ) -> Tuple[float, List[str], float, List[str]]:
        """Return (best score, best tags, second-best score,
        second-best tags). Standard 2-best Viterbi with a deterministic
        tie-break (tag order in TAGS)."""
        n = len(tokens)
        # chart[node] = list of (score, tags, tag_at_node) best-first
        chart: List[Dict[str, List[Tuple[float, List[str], str]]]] = [
            {} for _ in range(n)]
        for tag in TAGS:
            feats = self._features(tokens, 0, "O")
            score = self._score_tag(feats, tag, use_averaged=use_averaged)
            chart[0][tag] = [(score, [tag], tag)]
        for i in range(1, n):
            for tag in TAGS:
                scored: List[Tuple[float, List[str], str]] = []
                for prev in TAGS:
                    feats = self._features(tokens, i, prev)
                    base = self._score_tag(feats, tag,
                                           use_averaged=use_averaged)
                    for (pscore, ptags, _p) in chart[i - 1][prev]:
                        scored.append((pscore + base, ptags + [tag], tag))
                scored.sort(key=lambda t: (-t[0], t[1]))
                chart[i][tag] = scored[:2]
        finals: List[Tuple[float, List[str], str]] = []
        for tag in TAGS:
            finals.extend(chart[n - 1][tag])
        finals.sort(key=lambda t: (-t[0], t[1]))
        # merge duplicate tag sequences (2-best per node can collide)
        seen = set()
        uniq: List[Tuple[float, List[str], str]] = []
        for score, tags, _ in finals:
            key = tuple(tags)
            if key not in seen:
                seen.add(key)
                uniq.append((score, tags, tags[-1] if tags else "O"))
            if len(uniq) == 2:
                break
        while len(uniq) < 2:
            uniq.append((-math.inf, ["O"] * n, "O"))
        s1, t1, _ = uniq[0]
        s2, t2, _ = uniq[1]
        return s1, t1, s2, t2

    # -- public API -----------------------------------------------------------

    def tag(self, text: str, use_averaged: bool = True) -> Dict[str, Any]:
        """Tag one request; returns tags, spans and the 2-best margin.
        Inference decodes with the AVERAGED weights; training updates
        decode with the current weights (Collins 2002's own split)."""
        tokens = [w for w, _a, _b in _tokenize_with_offsets(text)]
        if not tokens:
            return {"tags": [], "spans": {}, "confidence": None,
                    "margin": None}
        if len(tokens) > MAX_TOKENS:
            raise ValueError(
                f"request has {len(tokens)} tokens; the tagger accepts at "
                f"most {MAX_TOKENS} (rejected, never truncated)")
        s1, best, s2, _second = self._viterbi_2best(tokens, use_averaged)
        margin = s1 - s2
        confidence = _margin_confidence(margin, len(tokens))
        return {"tags": best, "spans": _spans_of(best, tokens),
                "confidence": confidence, "margin": round(margin, 4)}

    def update(self, text: str, gold_tags: List[str]) -> bool:
        """One perceptron update against a gold sequence (Collins 2002:
        w += phi(x, gold) - phi(x, predicted)). Returns True when the
        prediction was wrong (the update changed something)."""
        tokens = [w for w, _a, _b in _tokenize_with_offsets(text)]
        if len(tokens) != len(gold_tags):
            raise ValueError("gold tag sequence must align with tokens")
        if any(t not in TAGS for t in gold_tags):
            raise ValueError(f"unknown tag in {gold_tags!r}")
        pred = self.tag(text, use_averaged=False)["tags"]
        wrong = pred != gold_tags
        self._steps += 1
        prev_g = prev_p = "O"
        for i in range(len(tokens)):
            fg = self._features(tokens, i, prev_g)
            fp = self._features(tokens, i, prev_p)
            if gold_tags[i] != pred[i] or fg != fp:
                for f in fg:
                    self._bump(gold_tags[i], f, +1.0)
                for f in fp:
                    self._bump(pred[i], f, -1.0)
            prev_g, prev_p = gold_tags[i], pred[i]
        self.updates += 1
        self.trained_examples += 1
        self._averaged_cache = None
        return wrong

    def train(self, rows: Sequence[Tuple[str, List[str]]],
              epochs: int = 12) -> int:
        """Deterministic multi-epoch training over (text, gold) rows:
        fixed row order, full pass each epoch, early stop when an epoch
        makes no mistakes (the perceptron convergence guarantee — the
        data is separable by construction, being the grammar's own
        labels). Returns the number of epochs actually run."""
        rows = list(rows)
        if not rows:
            return 0
        for epoch in range(1, max(1, epochs) + 1):
            mistakes = sum(1 for text, gold in rows if self.update(text, gold))
            if mistakes == 0:
                return epoch
        return max(1, epochs)

    def _bump(self, tag: str, feature: str, delta: float) -> None:
        w = self.weights[tag].get(feature, 0.0) + delta
        self.weights[tag][feature] = w
        t = self._totals[tag].get(feature, 0.0) + self._steps * delta
        self._totals[tag][feature] = t

    def _averaged(self) -> Dict[str, Dict[str, float]]:
        """Averaged weights (Collins 2002 section 3.3): w_avg = w_total/t.
        Cached because decoding reads every feature."""
        if self._averaged_cache is not None:
            return self._averaged_cache
        if self._steps == 0:
            self._averaged_cache = self.weights
            return self._averaged_cache
        avg: Dict[str, Dict[str, float]] = {}
        for tag in TAGS:
            table = dict(self.weights[tag])
            for f, total in self._totals[tag].items():
                table[f] = total / self._steps
            avg[tag] = table
        self._averaged_cache = avg
        return self._averaged_cache

    # -- persistence (learned-state JSON; the caller writes it) ---------------

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {t: dict(self.weights[t]) for t in self.weights},
                "totals": {t: dict(self._totals[t]) for t in self._totals},
                "steps": self._steps,
                "updates": self.updates,
                "trained_examples": self.trained_examples}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SlotTagger":
        tagger = cls()
        data = data if isinstance(data, dict) else {}
        saved = data.get("weights") or {}
        for tag, feats in saved.items():
            if tag in tagger.weights and isinstance(feats, dict):
                tagger.weights[tag] = {str(k): float(v)
                                       for k, v in feats.items()}
        saved_totals = data.get("totals") or {}
        for tag, feats in saved_totals.items():
            if tag in tagger._totals and isinstance(feats, dict):
                tagger._totals[tag] = {str(k): float(v)
                                       for k, v in feats.items()}
        tagger._steps = int(data.get("steps", 0))
        tagger.updates = int(data.get("updates", 0))
        tagger.trained_examples = int(data.get("trained_examples", 0))
        tagger._averaged_cache = None
        return tagger


def _margin_confidence(margin: float, n_tokens: int) -> float:
    """2-best margin -> (0, 1) confidence through a fixed-scale logistic:
    monotone, deterministic, never claims certainty (the same transform
    class fusion.py documents; gamma pinned so a per-token gap of ~1.0
    is already confident)."""
    gamma = 2.0
    scaled = gamma * margin / max(1, n_tokens)
    return _sigmoid(scaled)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _spans_of(tags: List[str], tokens: List[str]) -> Dict[str, Any]:
    spans: Dict[str, List[Dict[str, Any]]] = {}
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O":
            i += 1
            continue
        kind = tag.split("-", 1)[1]
        j = i
        while (j + 1 < len(tags)
               and tags[j + 1] == f"I-{kind}"):
            j += 1
        spans.setdefault(kind, []).append({
            "tokens": [i, j], "raw": " ".join(tokens[i:j + 1])})
        i = j + 1
    return spans


# ---------------------------------------------------------------------------
# Supervision from LOCAL history only (the cortex/learn.py data source).
# ---------------------------------------------------------------------------

def supervision_from_history(examples: Sequence[Dict[str, Any]],
                             noun_matcher: Optional[Callable[
                                 [str], Sequence[str]]] = None,
                             grammar_extract=None) -> List[Dict[str, Any]]:
    """Turn approved history rows into gold BIO sequences.

    ``examples``: cortex/learn.py's own example rows ({text, label,
    outcome, ...}). ONLY approved rows (label 1 / outcome applied or
    approved) supervise; rejected or undecided rows are skipped — a
    refused request says nothing about slot structure. Gold spans are
    read off the EXISTING grammar: settings/slots.extract()'s own
    matched raws (cue + generic), plus TARGET spans when the caller
    supplies a noun matcher (e.g. the parser's noun matching). Requests
    the grammar cannot label at all produce no supervision — nothing is
    guessed to fill the gap."""
    if grammar_extract is None:
        from assistant.capabilities.settings import slots as _slots
        grammar_extract = _slots.extract

    out: List[Dict[str, Any]] = []
    for row in examples:
        if not isinstance(row, dict):
            continue
        approved = (row.get("label") == 1
                    or row.get("outcome") in ("applied", "approved"))
        if not approved:
            continue
        text = str(row.get("text", ""))
        tokens = _tokenize_with_offsets(text)
        if not tokens:
            continue
        slots = grammar_extract(text)
        tags = ["O"] * len(tokens)
        tagged = False
        for slot_key, kind in (("cue", "CUE"), ("generic", "GENERIC")):
            raw = (slots.get(slot_key) or {}).get("raw")
            if raw:
                if _tag_raw(raw, text, tokens, tags, kind):
                    tagged = True
        if noun_matcher is not None:
            for noun in (noun_matcher(text) or []):
                if _tag_raw(str(noun), text, tokens, tags, "TARGET"):
                    tagged = True
        if tagged:
            out.append({"text": text,
                        "tokens": [w for w, _a, _b in tokens],
                        "tags": tags})
    return out


def _tag_raw(raw: str, text: str, tokens, tags: List[str], kind: str
             ) -> bool:
    """Span one grammar raw string onto the token index list (first
    match). Returns True when something was tagged."""
    needle = raw.lower()
    words = [w.lower() for w, _a, _b in tokens]
    n = len(needle.split())
    for start in range(len(words) - n + 1):
        if " ".join(words[start:start + n]) == needle:
            tags[start] = f"B-{kind}"
            for k in range(start + 1, start + n):
                tags[k] = f"I-{kind}"
            return True
    return False


# ---------------------------------------------------------------------------
# The gated entry point: tagger first, grammar fallback on low confidence.
# ---------------------------------------------------------------------------

def tag_gated(text: str,
              tagger: SlotTagger,
              calibrator: Optional[ConformalCalibrator] = None,
              noun_matcher: Optional[Callable[[str], Sequence[str]]] = None
              ) -> Dict[str, Any]:
    """One request, one answer, one honest path label.

    - the tagger answers when it is trained AND the conformal calibrator
      (with actual calibration data) covers its confidence;
    - everything else — untrained tagger, confidence below the conformal
      threshold, or NO calibration data at all — falls back to the
      EXISTING grammar path (settings/slots.extract, the char-ngram
      layer's own machinery), carrying the calibrator's own reason.
    """
    if tagger.trained_examples == 0 or not any(tagger.weights[t]
                                               for t in TAGS):
        return _grammar_fallback(text, "tagger untrained",
                                 calibrator=calibrator,
                                 noun_matcher=noun_matcher)
    result = tagger.tag(text)
    confidence = result.get("confidence")
    if confidence is None or not result["spans"]:
        return _grammar_fallback(text, "tagger found no slot structure",
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
    result.update({"used": "tagger", "conformal": verdict.get("reason")})
    return result


def _grammar_fallback(text: str, reason: str,
                      calibrator: Optional[ConformalCalibrator] = None,
                      noun_matcher: Optional[Callable[
                          [str], Sequence[str]]] = None) -> Dict[str, Any]:
    """The EXISTING grammar/char-ngram path, unchanged and verbatim."""
    from assistant.capabilities.settings import slots as _slots

    slots = _slots.extract(text)
    targets = list(noun_matcher(text) or []) if noun_matcher else []
    return {"used": "grammar-fallback", "reason": reason,
            "slots": slots, "targets": targets}

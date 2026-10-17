"""brain.kneser_ney — Kneser-Ney smoothed n-gram language models over
CALLER-SUPPLIED LOCAL TEXT: an additional signal beside the vault's
SymSpell delete-index (brain/spellfix.py) for typo correction, and a
next-token completion signal — both gated by evidence strength so they
only speak when the corpus supports them.

Kneser & Ney 1995, "Improved backing-off for m-gram language modeling",
ICASSP 1995 — the Kneser-Ney family. Its distinguishing feature (the
reason this is NOT a reimplementation of the Katz-style escape in
genius/markov.py): the lowest-order distribution is the CONTINUATION
count — in how many DISTINCT contexts a token appeared — not its raw
frequency. A word seen once in three different contexts is better
backoff evidence than a word seen five times always after the same
word, and KN encodes exactly that.

Chen & Goodman 1999, "An Empirical Study of Smoothing Techniques for
Language Modeling", Computer Speech & Language 13(4) — the source of
the INTERPOLATED form implemented here:

    P_KN(w | h) = max(c(hw) - D, 0) / c(h) + lambda(h) * P_KN(w | h')
    lambda(h)   = D * N+(h) / c(h)      (N+(h) = distinct continuations
                                         of h; c(h) = sum of c(hw);
                                         h' = h minus its oldest token)

and of the leaving-one-out discount estimate D = n1 / (n1 + 2*n2),
computed per order from the corpus counts themselves (n1 / n2 = number
of n-grams of that order with count exactly 1 / 2). Fixed and
deterministic: there is no hyperparameter search, no RNG, nothing
fitted iteratively. "Training" here means counting n-grams and
estimating the closed-form discounts, nothing more.

Two flavors from ONE core:
- ``WordKN`` (default n=3): word n-grams — the COMPLETION signal
  (``complete`` ranks next words over the observed vocabulary);
- ``CharKN`` (default n=4): character n-grams — the CORRECTION signal
  (``suggest`` ranks 1-2-edit candidates by char-model probability,
  reusing the delete-variant helper of brain/spellfix.py as the
  candidate prefilter and verifying true bounded Damerau-Levenshtein
  distance afterwards).

At the unigram level the KN structure is exact:
P_continuation(w) = |distinct contexts of w| / |distinct bigram types|,
absolutely discounted by D_1 and renormalized so the ENTIRE leftover
mass lambda_0 = D_1 * N+ / |bigram types| is the probability of the
never-seen class (an unseen token is worth exactly the carved-out
discount mass — no floor is invented, and a token the model literally
cannot see scores 0, reported as such).

Evidence gating (the integration honesty): ``gate(model, query)``
returns "confident" | "thin" | "abstain" against DOCUMENTED thresholds
(min total training tokens; min distinct continuations of the query's
longest observed prefix), and ``spellfix_fallback`` yields to the
existing SymSpell answer unless SymSpell abstained AND KN is
"confident" — the spellfix path stays primary; this module is only the
additional signal.

Determinism: same texts -> byte-identical model dict. The module
writes NOTHING: ``train`` returns a plain JSON-serializable dict the
caller persists through the learned-state path it already owns.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .nlp import words
# The delete-variant helper of the vault's SymSpell index (same package,
# one shared implementation — the IDEA is reused, not duplicated; the
# bounded edit-distance verifier below is honest extra work).
from .spellfix import _deletes

__all__ = ["WordKN", "CharKN", "train", "score", "complete", "suggest",
           "gate", "spellfix_fallback"]

_BOS = "<s>"    # begin-of-sequence pad: never a text token (word tokens
                # are [a-z0-9]+; char tokens are single characters)
_SEP = "\x1f"   # unit separator for context keys (JSON-safe, never part
                # of a lowercased token)

# Documented gate defaults (see gate()):
DEFAULT_MIN_TOKENS = 50     # corpus must have >= this many tokens at all
DEFAULT_MIN_CONTEXTS = 2    # query's observed prefix must have >= this
                            # many distinct observed continuations


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(flavor: str, text: str) -> List[str]:
    """word flavor: the shared nlp tokenizer (lowercase [a-z0-9]+);
    char flavor: every lowercased character (spaces included — they are
    real correction signal: 'hwre' vs 'where')."""
    if flavor == "word":
        return words(text)
    return list(text.lower())


# ---------------------------------------------------------------------------
# Training = counting + closed-form discounts (nothing is fitted)
# ---------------------------------------------------------------------------

def _loo_discount(histogram: Counter) -> float:
    """Chen & Goodman 1999's leaving-one-out estimate of the absolute
    discount: D = n1 / (n1 + 2*n2) over the count histogram (n1 / n2 =
    number of n-grams counted exactly once / twice). 0.0 in the
    degenerate case where the histogram holds neither (nothing to
    discount — no smoothing mass exists, honestly)."""
    n1 = histogram.get(1, 0)
    n2 = histogram.get(2, 0)
    if n1 + 2 * n2 <= 0:
        return 0.0
    return n1 / (n1 + 2 * n2)


def train(flavor: str, texts: Sequence[str],
          n: Optional[int] = None) -> Dict[str, Any]:
    """Count n-grams and estimate the per-order discounts. ``flavor`` is
    "word" or "char"; ``n`` defaults to 3 (word) / 4 (char). Returns the
    JSON-serializable model dict. Pure: no file I/O, no RNG."""
    if flavor not in ("word", "char"):
        raise ValueError(f"unknown flavor {flavor!r} (use 'word' or 'char')")
    if n is None:
        n = 3 if flavor == "word" else 4
    n = int(n)
    if n < 2:
        raise ValueError("n >= 2 required (KN needs bigrams for the "
                         "continuation counts)")

    orders: Dict[int, Dict[str, Dict[str, int]]] = {k: {} for k in range(2, n + 1)}
    vocab: Counter = Counter()          # observed token counts (no pads)
    word_vocab: Counter = Counter()     # WORD vocabulary — suggest()'s
                                        # candidate pool, kept for both
                                        # flavors (a char model still
                                        # corrects words)
    total_tokens = 0

    for text in texts:
        toks = _tokenize(flavor, text)
        if not toks:
            continue
        total_tokens += len(toks)
        vocab.update(toks)
        word_vocab.update(words(text) if flavor == "char" else toks)
        padded = [_BOS] * (n - 1) + toks
        for k in range(2, n + 1):
            table = orders[k]
            # count every (real token, its full available left context
            # including BOS pads) — BOS is never itself a predicted token
            for i in range(n - k, len(padded) - k + 1):
                key = _SEP.join(padded[i:i + k - 1])
                nxt = padded[i + k - 1]
                row = table.setdefault(key, {})
                row[nxt] = row.get(nxt, 0) + 1

    # continuation counts from the bigram table: c_cont(w) = number of
    # DISTINCT left contexts in which w appeared (sentence-initial <s>
    # counts as one context) — the KN-vs-Katz distinction
    contexts_of: Dict[str, set] = {}
    for ctx, row in orders[2].items():
        for w in row:
            contexts_of.setdefault(w, set()).add(ctx)
    continuations = {w: len(seen) for w, seen in contexts_of.items()}
    bigram_types = sum(len(row) for row in orders[2].values())

    # per-order leaving-one-out discounts: index 0 = unigram level
    # (over continuation counts), index k-1 = order-k n-gram level
    discounts = [_loo_discount(Counter(continuations.values()))]
    for k in range(2, n + 1):
        discounts.append(_loo_discount(
            Counter(c for row in orders[k].values() for c in row.values())))

    return {
        "algorithm": "kneser-ney-interpolated",
        "citations": [
            "Kneser & Ney 1995, 'Improved backing-off for m-gram language "
            "modeling', ICASSP",
            "Chen & Goodman 1999, 'An Empirical Study of Smoothing "
            "Techniques for Language Modeling', Computer Speech & Language "
            "13(4) — interpolated form + leaving-one-out discount",
        ],
        "flavor": flavor,
        "n": n,
        "discounts": discounts,
        "total_tokens": total_tokens,
        "bigram_types": bigram_types,
        "continuations": dict(sorted(continuations.items())),
        "vocab": dict(sorted(vocab.items())),
        "word_vocab": dict(sorted(word_vocab.items())),
        "orders": {str(k): {ctx: dict(sorted(row.items()))
                            for ctx, row in sorted(orders[k].items())}
                   for k in range(2, n + 1)},
        "note": ("counting + closed-form discounts only; nothing fitted "
                 "iteratively; no I/O — the caller persists this dict"),
    }


class _KNFlavor:
    """The one core; subclasses only fix the tokenization flavor."""

    flavor = ""
    default_n = 3

    def __init__(self, n: Optional[int] = None) -> None:
        self.n = int(n) if n is not None else self.default_n

    def train(self, texts: Sequence[str]) -> Dict[str, Any]:
        return train(self.flavor, texts, self.n)


class WordKN(_KNFlavor):
    """Word n-grams (default n=3) — the COMPLETION signal: complete()
    ranks the observed vocabulary by P(next | prefix)."""

    flavor = "word"
    default_n = 3


class CharKN(_KNFlavor):
    """Character n-grams (default n=4) — the CORRECTION signal:
    suggest() ranks 1-2-edit word candidates by char-model
    probability."""

    flavor = "char"
    default_n = 4


# ---------------------------------------------------------------------------
# The probability core (interpolated KN, exact normalization)
# ---------------------------------------------------------------------------

def _p(model: Dict[str, Any], token: str, hist: List[str]) -> float:
    """P_KN(token | hist), hist = at most n-1 tokens, OLDEST first (the
    recursion drops the oldest). Unseen context backs off with weight 1
    to the shorter context (the natural interpolated-KN behavior —
    there is no c(h) to divide by). At the unigram level the ENTIRE
    leftover discount mass lambda_0 is the probability of any token not
    in the continuation vocabulary."""
    orders: Dict[str, Dict[str, Dict[str, int]]] = model["orders"]
    discounts: List[float] = model["discounts"]
    j = len(hist)
    if j > 0:
        key = _SEP.join(hist)
        table = orders.get(str(j + 1), {}).get(key)
        if table:
            c_h = sum(table.values())
            d = discounts[j]                     # level j uses order j+1
            first = max(table.get(token, 0) - d, 0.0) / c_h
            lam = d * len(table) / c_h            # len(table) = N+(h)
            return first + lam * _p(model, token, hist[1:])
        return _p(model, token, hist[1:])
    # unigram continuation level
    bigram_types = model["bigram_types"]
    if bigram_types == 0:
        return 0.0    # degenerate model: no continuation distribution
    d1 = discounts[0]
    cont: Dict[str, int] = model["continuations"]
    if token in cont:
        return max(cont[token] - d1, 0.0) / bigram_types
    return d1 * len(cont) / bigram_types        # lambda_0 -> the unseen class


def score(model: Dict[str, Any], text: str) -> Optional[float]:
    """Average per-token natural-log probability of ``text`` under the
    model (BOS-padded contexts, tokens scored only). None when the
    model has no continuation distribution to score with (empty /
    degenerate corpus — no invented probabilities); float('-inf') when
    some token has literally zero probability (the model assigns no
    mass to it — no floor is invented, the impossibility is reported)."""
    if not isinstance(model, dict) or not model.get("bigram_types"):
        return None
    toks = _tokenize(model.get("flavor", "word"), text)
    if not toks:
        return None
    n = int(model["n"])
    padded = [_BOS] * (n - 1) + toks
    logs: List[float] = []
    for i, tok in enumerate(toks):
        p = _p(model, tok, padded[i:i + n - 1])
        if p <= 0.0:
            return float("-inf")
        logs.append(math.log(p))
    return sum(logs) / len(logs)


def complete(model: Dict[str, Any], prefix: str, k: int = 5) -> List[Dict[str, Any]]:
    """Ranked next-token completions over the observed vocabulary:
    probability descending, ties broken lexicographically (no sampling).
    The prefix's last n-1 tokens are the context; a shorter prefix uses
    its shorter context (the recursion handles it). [] when the model
    has nothing to offer. Tokens the model gives zero probability are
    excluded — they cannot honestly be completions."""
    if not isinstance(model, dict) or not model.get("bigram_types"):
        return []
    n = int(model["n"])
    ctx = _tokenize(model["flavor"], prefix)[-(n - 1):] if n > 1 else []
    ranked = []
    for tok in model["vocab"]:
        p = _p(model, tok, ctx)
        if p > 0.0:
            ranked.append((-p, tok, p))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [{"token": tok, "p": round(p, 6)} for _neg, tok, p in ranked[:k]]


# ---------------------------------------------------------------------------
# Correction (char flavor): bounded-edit candidates ranked by the model
# ---------------------------------------------------------------------------

def _osa_distance(a: str, b: str, cap: int = 2) -> Optional[int]:
    """Bounded optimal-string-alignment distance (Damerau-Levenshtein
    with adjacent transpositions — the edit model SymSpell uses), or
    None when the distance exceeds ``cap``. Pure arithmetic."""
    if abs(len(a) - len(b)) > cap:
        return None
    prev2: Optional[List[int]] = None
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(prev[j] + 1,          # deletion
                      cur[j - 1] + 1,       # insertion
                      prev[j - 1] + cost)   # (mis)match
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                val = min(val, prev2[j - 2] + 1)   # adjacent transposition
            cur[j] = val
            row_min = min(row_min, val)
        if row_min > cap:      # every path through this row is too far
            return None
        prev2, prev = prev, cur
    return prev[len(b)] if prev[len(b)] <= cap else None


def suggest(model: Dict[str, Any], word: str, k: int = 5,
            min_tokens: int = DEFAULT_MIN_TOKENS,
            min_contexts: int = DEFAULT_MIN_CONTEXTS) -> Dict[str, Any]:
    """Correction candidates for ``word`` from the CHAR model's word
    vocabulary, ranked by char-model probability (score(), average
    per-char log-probability) among true 1-2-edit candidates. The
    candidate prefilter reuses brain/spellfix.py's ``_deletes``
    (SymSpell's delete-index idea: a word and its typo share a
    delete-variant); the bounded Damerau-Levenshtein distance is then
    verified honestly, and the query itself is always a candidate (the
    model may confirm the word as-is). Abstains (empty candidates) when
    the evidence gate does not clear — no correction is invented on
    thin evidence."""
    if not isinstance(model, dict):
        raise ValueError("model dict required")
    if model.get("flavor") != "char":
        raise ValueError("suggest() needs the CHAR flavor (the correction "
                         f"signal); this model is {model.get('flavor')!r}")
    out: Dict[str, Any] = {
        "query": word,
        "algorithm": ("kneser-ney char model over 1-2-edit candidates "
                      "(delete-variant prefilter reused from "
                      "brain/spellfix.py; OSA distance verified)"),
        "gate": gate(model, word, min_tokens, min_contexts),
    }
    if out["gate"] == "abstain":
        out["candidates"] = []
        out["note"] = ("the char model has no evidence for this word "
                       "(gate thresholds) — no correction invented")
        return out
    word = word.lower()
    vocab_words = model.get("word_vocab", {})
    if not vocab_words:
        out["candidates"] = []
        out["note"] = "no word vocabulary in the model"
        return out
    query_deletes = _deletes(word, 2)
    pool = {word}
    for cand in vocab_words:
        if abs(len(cand) - len(word)) <= 2 and _deletes(cand, 2) & query_deletes:
            pool.add(cand)
    rows: List[Dict[str, Any]] = []
    zero_probability: List[str] = []
    for cand in sorted(pool):
        dist = _osa_distance(word, cand, 2)
        if dist is None:
            continue                      # prefilter superset: verify exactly
        s = score(model, cand)
        if s is None or s == float("-inf"):
            zero_probability.append(cand)
            continue
        rows.append({"word": cand, "edit_distance": dist,
                     "logp_per_char": round(s, 4)})
    rows.sort(key=lambda r: (-r["logp_per_char"], r["word"]))
    out["candidates"] = rows[:k]
    out["checked"] = len(pool)
    if zero_probability:
        out["zero_probability"] = zero_probability
    return out


# ---------------------------------------------------------------------------
# Evidence gating + the integration contract
# ---------------------------------------------------------------------------

def gate(model: Dict[str, Any], query: str,
         min_tokens: int = DEFAULT_MIN_TOKENS,
         min_contexts: int = DEFAULT_MIN_CONTEXTS) -> str:
    """One of "confident" | "thin" | "abstain" — documented thresholds:

    - "abstain": the model has no continuation distribution at all, or
      fewer than ``min_tokens`` (default 50) total training tokens —
      the corpus as a whole is too thin to say anything — or the
      query's token sequence has no observed suffix-context (nothing
      of it was ever seen as a context; for completion that means
      zero evidence about what follows it);
    - "thin": the longest observed suffix-context of the query has
      fewer than ``min_contexts`` (default 2) distinct observed
      continuations — the prefix was attested, but with a single
      continuation the "distribution" is one anecdote;
    - "confident": at least ``min_contexts`` distinct continuations
      were observed after the query's longest observed suffix-context.
    """
    if not isinstance(model, dict) or not model.get("bigram_types"):
        return "abstain"
    if int(model.get("total_tokens", 0)) < min_tokens:
        return "abstain"
    toks = _tokenize(model.get("flavor", "word"), query)
    if not toks:
        return "abstain"
    n = int(model["n"])
    orders = model["orders"]
    for j in range(min(len(toks), n - 1), 0, -1):
        key = _SEP.join(toks[-j:])
        table = orders.get(str(j + 1), {}).get(key)
        if table:
            return "confident" if len(table) >= min_contexts else "thin"
    return "abstain"


def spellfix_fallback(current_suggestion: Any, kn_result: Any,
                      model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The integration contract: the existing SymSpell spellfix path is
    PRIMARY; this Kneser-Ney model is only an additional signal. The
    existing answer ALWAYS wins while it exists; KN's suggestion is
    used ONLY when SymSpell abstained (empty suggestion) AND the KN
    result cleared its evidence gate ("confident"). ``kn_result`` is
    expected to be suggest()'s dict; if it lacks a gate verdict, the
    gate is re-derived from ``model``. Returns a plain dict — nothing
    is mutated, nothing is written."""
    if current_suggestion:
        return {"use": "spellfix", "suggestion": current_suggestion,
                "why": ("the existing spellfix answer is primary; "
                        "KN never overrides it")}
    kn = kn_result if isinstance(kn_result, dict) else {}
    kn_gate = kn.get("gate")
    if kn_gate is None and isinstance(model, dict) and kn.get("query"):
        kn_gate = gate(model, str(kn["query"]))
    candidates = kn.get("candidates") or []
    if kn_gate == "confident" and candidates:
        top = candidates[0]
        best = top.get("word") if isinstance(top, dict) else top
        return {"use": "kneser_ney", "suggestion": best,
                "why": ("spellfix abstained AND the KN char model "
                        "cleared its evidence gate (confident)")}
    return {"use": "none", "suggestion": None,
            "why": (f"spellfix abstained and KN did not clear its gate "
                    f"(gate={kn_gate!r}) — no correction invented")}

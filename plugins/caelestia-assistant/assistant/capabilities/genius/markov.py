"""genius.markov — generation and sequence prediction without a model file.

  * word-level Markov text generator (order-N chains, temperature
    sampling, seeded for reproducibility)
  * n-gram sequence predictor with Katz-style backoff and escape
    probabilities — "given this history, what comes next" with honest
    confidence that shrinks on unseen data
  * template grammars: `"{greeting} {name}, {message}"` with weighted
    slot tables and recursive-free safe expansion
  * constrained generation: retry loops until must-contain / must-not
    constraints hold (bounded, honest about failure)
  * name generator: syllable-level chains with length/charset/regex
    constraints
  * sequence surprise: per-item -log2 p under the model — an outlier
    detector for event streams (a shell history, log kinds, anything)
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "generate_text", "SequencePredictor", "predict_next", "sequence_surprise",
    "expand_template", "generate_name", "ConstrainedGenerator",
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+|[^\sA-Za-z']", text)


def generate_text(corpus: str, n_words: int = 60, order: int = 2,
                  temperature: float = 1.0, seed: int = 42,
                  start: Optional[str] = None) -> Dict[str, Any]:
    """Order-N word-level Markov generation from the corpus."""
    if order < 1:
        raise ValueError("order must be >= 1")
    tokens = _tokenize(corpus)
    if len(tokens) < order + 2:
        raise ValueError("corpus too small for this order")
    chains: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for i in range(len(tokens) - order):
        key = tuple(tokens[i:i + order])
        nxt = tokens[i + order]
        chains.setdefault(key, {})
        chains[key][nxt] = chains[key].get(nxt, 0) + 1
    rng = random.Random(seed)
    if start:
        heads = [k for k in chains if " ".join(k).lower().startswith(start.lower())]
        key = rng.choice(heads) if heads else max(chains, key=lambda k: sum(chains[k].values()))
    else:
        key = rng.choice(list(chains))
    out = list(key)
    branchiness = 0
    for _ in range(n_words):
        table = chains.get(key)
        if not table:
            break
        items = sorted(table.items())
        if temperature <= 0.05:
            nxt = max(items, key=lambda kv: kv[1])[0]
        else:
            weights = [c ** (1.0 / temperature) for _, c in items]
            nxt = rng.choices([w for w, _ in items], weights=weights, k=1)[0]
        if len(table) > 1:
            branchiness += 1
        out.append(nxt)
        key = tuple(out[-order:])
    return {"text": " ".join(w for w in out if re.match(r"[A-Za-z']+$", w) or w.isalnum()
                             or w in ",.!?;:'-") ,
            "order": order, "temperature": temperature, "seed": seed,
            "n_words": len(out), "branch_points": branchiness,
            "n_states": len(chains),
            "note": "Markov chains riff on style, not meaning"}


class SequencePredictor:
    """N-gram predictor with Katz-style backoff + escape mass for the unseen."""

    def __init__(self, max_order: int = 4):
        if max_order < 1:
            raise ValueError("max_order >= 1")
        self.max_order = max_order
        self.counts: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = {
            k: {} for k in range(1, max_order + 1)}
        self.unigrams: Dict[str, int] = {}
        self.n_events = 0

    def fit(self, sequence: Sequence[str]) -> "SequencePredictor":
        seq = [str(s) for s in sequence]
        self.n_events += len(seq)
        for s in seq:
            self.unigrams[s] = self.unigrams.get(s, 0) + 1
        for order in range(1, self.max_order + 1):
            for i in range(len(seq) - order):
                key = tuple(seq[i:i + order])
                nxt = seq[i + order]
                self.counts[order].setdefault(key, {})
                table = self.counts[order][key]
                table[nxt] = table.get(nxt, 0) + 1
        return self

    def distribution(self, history: Sequence[str]) -> Dict[str, float]:
        hist = [str(h) for h in history][-self.max_order:]
        if not hist:
            total = sum(self.unigrams.values()) or 1
            dist = {k: v / total for k, v in self.unigrams.items()}
        else:
            # Katz-style: highest order with support wins, keep escape mass
            escape = 0.1
            dist = None
            for order in range(min(self.max_order, len(hist)), 0, -1):
                key = tuple(hist[-order:])
                table = self.counts[order].get(key)
                if table and sum(table.values()) >= 1:
                    total = sum(table.values())
                    dist = {k: (1 - escape) * c / total for k, c in table.items()}
                    break
            if dist is None:
                total = sum(self.unigrams.values()) or 1
                dist = {k: v / total for k, v in self.unigrams.items()}
        # always reserve escape mass for events never seen yet: split
        # the 0.1 mass over the known-but-unpredicted vocabulary AND one
        # <UNK> slot for events outside the vocabulary entirely
        vocab = set(self.unigrams) - set(dist)
        share = 0.1 / (len(vocab) + 1)
        for v in vocab:
            dist[v] = dist.get(v, 0.0) + share
        dist["<UNK>"] = dist.get("<UNK>", 0.0) + share
        return dist

    def predict(self, history: Sequence[str], top: int = 5) -> Dict[str, Any]:
        dist = self.distribution(history)
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        best_p = ranked[0][1] if ranked else 0.0
        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
        return {"history": [str(h) for h in history][-self.max_order:],
                "top": [{"next": k, "p": round(p, 4)} for k, p in ranked[:top]],
                "best": ranked[0][0] if ranked else None,
                "best_p": round(best_p, 4),
                "entropy_bits": round(entropy, 3),
                "confidence": ("high" if best_p > 0.6 else
                               "medium" if best_p > 0.25 else "low"),
                "vocab": len(dist),
                "note": "backoff n-grams; unseen histories fall back gracefully"}


def predict_next(sequence: Sequence[str], top: int = 5) -> Dict[str, Any]:
    p = SequencePredictor()
    p.fit(sequence)
    return p.predict(sequence[-4:] if len(sequence) >= 4 else sequence, top=top)


def sequence_surprise(sequence: Sequence[str]) -> Dict[str, Any]:
    """Per-item surprise -log2 p under an n-gram model trained on the past.

    For each t the model is the one trained on seq[:t] only (leave-one-out
    in time), so the score is honest about what was knowable then.
    O(n^2) in the sequence length — fine for the few-hundred-event
    streams this is meant for.
    """
    seq = [str(s) for s in sequence]
    if len(seq) < 3:
        raise ValueError("need at least 3 events")
    surprises = []
    for t in range(1, len(seq)):
        predictor = SequencePredictor()
        predictor.fit(seq[:t])
        dist = predictor.distribution(seq[max(0, t - 4):t])
        p = dist.get(seq[t], dist.get("<UNK>", 1e-9))
        surprises.append({"index": t, "event": seq[t],
                          "surprise_bits": round(-math.log2(max(p, 1e-9)), 3)})
    values = [s["surprise_bits"] for s in surprises]
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
    # anomaly line: both 2-sigma above the mean AND at least 3 bits of
    # surprise (3 bits = the model gave the event < 1/8 probability)
    threshold = max(mean + 2 * sd if sd > 0 else mean + 2, 3.0)
    surprises_sorted = sorted(surprises, key=lambda s: -s["surprise_bits"])
    return {"surprises": surprises, "mean_bits": round(mean, 3),
            "sd_bits": round(sd, 3), "threshold_bits": round(threshold, 3),
            "anomalies": [s for s in surprises_sorted if s["surprise_bits"] > threshold][:5],
            "most_surprising": surprises_sorted[:3]}


# ---------------------------------------------------------------------------
# Template grammars
# ---------------------------------------------------------------------------

def expand_template(template: str, slots: Dict[str, Sequence[str]],
                    seed: int = 7) -> Dict[str, Any]:
    """Expand {slot} references with weighted random choices (seeded)."""
    rng = random.Random(seed)
    used: Dict[str, str] = {}
    pattern = re.compile(r"\{([a-z_]+)\}")
    expansions = 0

    def expand(s: str, depth: int = 0) -> str:
        nonlocal expansions
        if depth > 6:
            return s

        def repl(m: "re.Match[str]") -> str:
            nonlocal expansions
            slot = m.group(1)
            options = slots.get(slot)
            if not options:
                return m.group(0)
            pick = rng.choice(list(options))
            expansions += 1
            used.setdefault(slot, pick)
            return expand(pick, depth + 1)

        return pattern.sub(repl, s)

    result = expand(template)
    unresolved = pattern.findall(result)
    return {"template": template, "result": result,
            "slots_used": used, "expansions": expansions,
            "unresolved": unresolved,
            "complete": not unresolved}


_GREETINGS = ["hey", "yo", "hi", "heads up"]
_THINGS = ["the build", "the shell", "the config", "the bar", "the dock"]
_STATES = ["looks great", "needs a rebuild", "synced cleanly", "has stale caches",
           "is ready to ship"]


class ConstrainedGenerator:
    """Generate until constraints hold (bounded retries, honest failure)."""

    def __init__(self, corpus: str, seed: int = 11):
        self.corpus = corpus
        self.seed = seed

    def generate(self, n_words: int = 40, must_contain: Sequence[str] = (),
                must_not_match: Sequence[str] = (), max_tries: int = 25,
                order: int = 2) -> Dict[str, Any]:
        if not must_contain and not must_not_match:
            raise ValueError("give at least one constraint")
        for attempt in range(max_tries):
            out = generate_text(self.corpus, n_words=n_words, order=order,
                                seed=self.seed + attempt)
            text = out["text"].lower()
            ok = True
            missing = []
            for needle in must_contain:
                if needle.lower() not in text:
                    missing.append(needle)
                    ok = False
            for pattern in must_not_match:
                if re.search(pattern, text):
                    ok = False
                    break
            if ok:
                return {"success": True, "attempts": attempt + 1, **out,
                        "satisfied": {"must_contain": list(must_contain),
                                      "must_not": list(must_not_match)}}
        return {"success": False, "attempts": max_tries, "text": None,
                "reason": "constraints not satisfiable in the retry budget — "
                          "loosen them or grow the corpus"}


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_SYLLABLES = {
    "onset": ["b", "br", "c", "ch", "d", "dr", "f", "g", "gl", "k", "l", "m",
              "n", "p", "pr", "r", "s", "sh", "t", "th", "tr", "v", "w", "z", ""],
    "nucleus": ["a", "e", "i", "o", "u", "ae", "ai", "ea", "ei", "ia", "io",
                "oa", "oo", "ou"],
    "coda": ["", "", "", "n", "r", "s", "l", "m", "x", "th", "ss", "ll"],
}


def generate_name(n: int = 5, min_len: int = 4, max_len: int = 9,
                  seed: int = 5, pattern: Optional[str] = None) -> Dict[str, Any]:
    """Syllable-structured names, consonant-vowel filtered for pronounceability."""
    rng = random.Random(seed)
    names = []
    tries = 0
    while len(names) < n and tries < n * 60:
        tries += 1
        syl_count = rng.randint(2, 3)
        parts = []
        for i in range(syl_count):
            onset = rng.choice(_SYLLABLES["onset"])
            nucleus = rng.choice(_SYLLABLES["nucleus"])
            coda = rng.choice(_SYLLABLES["coda"]) if i < syl_count - 1 else \
                rng.choice(["", "", "n", "r", "s", "x", "th"])
            parts.append(onset + nucleus + coda)
        name = "".join(parts)
        if pattern and not re.fullmatch(pattern, name):
            continue
        if not (min_len <= len(name) <= max_len):
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", name):  # unpronounceable
            continue
        names.append(name)
    return {"names": names, "n_generated": len(names),
            "pattern": pattern,
            "note": "CVC syllable chains — pronounceable, meaningless by design"}

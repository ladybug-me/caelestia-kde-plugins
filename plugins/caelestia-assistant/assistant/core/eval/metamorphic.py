"""The metamorphic arena (Group A1): variants AUTO-GENERATED from
registry metadata, lexicon direction pairs and deterministic typo
operators — run against the router with metamorphic invariants.

A metamorphic relation is a property that must hold across a
transformation when there is no single "right answer" to assert:

- ``paraphrase``     wrapping the request in politeness/intent frames
                     ("could you ...", "... please") must not change the
                     top surface;
- ``synonym``        swapping ONE content word for a same-meaning word
                     drawn from the shipped lexicon's own equivalence
                     classes must not change the top surface;
- ``unit_change``    re-spelling a value's unit ("15 pixels" -> "15px",
                     "50 percent" -> "50%") must not change the top
                     surface;
- ``typo``           one bounded edit (swap/delete/double, distance 1)
                     inside the request's longest word must still reach
                     the accepted set among the top-3 (the typo corrector
                     recovers it; ranking may legitimately shuffle);
- ``polarity_flip``  replacing a direction word with its antonym
                     (thinner <-> thicker) must NOT change the routed
                     tool (the value side flips, the OBJECT does not) and
                     must flip the extracted direction cue sign.

Every variant is generated DETERMINISTICALLY from the committed
registry/lexicon (no RNG): same inputs, same variants, same verdicts.
Consistency rates are reported with the same seeded bootstrap intervals
as the other suites, and the ratchet floors live in the test
(``assistant/eval/tests/test_metamorphic.py``), not in prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .engine import BOOT_SEED, _ci, load_set

__all__ = ["TRANSFORMS", "run_metamorphic", "generate_variants"]

TRANSFORMS = ("paraphrase", "synonym", "unit_change", "typo", "polarity_flip")

# -- paraphrase frames -------------------------------------------------------
# Deterministic frames that add intent/politeness without adding a value
# cue or naming another object. Order matters (output order is fixed).
_PARAPHRASE_FRAMES = (
    "could you {t}",
    "please {t}",
    "i want to {t}",
    "{t} please",
    "can you {t}",
)

# -- unit re-spellings -------------------------------------------------------
_UNIT_VARANTS = [
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*pixels?\b"), r"\1px"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*px\b"), r"\1 pixels"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*percent\b"), r"\1%"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*%\b"), r"\1 percent"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*seconds?\b"), r"\1s"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*s\b(?![-\w])"), r"\1 seconds"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*minutes?\b"), r"\1m"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*m\b(?![-\wa-z])"), r"\1 minutes"),
]

# -- typo operators (distance 1, deterministic, bounded) ---------------------
_WORD_RE = re.compile(r"[a-z]{4,}")


def _typo_of(text: str) -> Optional[str]:
    """One distance-1 edit in the longest (first-max) word: swap, then
    delete, then double — the first applicable operator wins, so the
    same text always yields the same typo."""
    words = _WORD_RE.findall(text)
    if not words:
        return None
    word = max(words, key=lambda w: (len(w), -text.index(w)))
    start = text.index(word)
    if len(word) >= 5:
        i = len(word) // 2
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if swapped != word:
            return text[:start] + swapped + text[start + len(word):]
    i = len(word) // 2
    deleted = word[:i] + word[i + 1:]
    if deleted != word:
        return text[:start] + deleted + text[start + len(word):]
    doubled = word[:i + 1] + word[i] + word[i + 1:]
    return text[:start] + doubled + text[start + len(word):]


def _synonym_variant(text: str) -> Optional[str]:
    """Swap ONE content token for a lexicon sibling (same equivalence
    class, never itself). Deterministic: first key (sorted) with a
    usable alternative, first alternative (sorted)."""
    from assistant.core.lexicon import SYNONYMS
    tokens = re.findall(r"[a-z][a-z-]+", text)
    for tok in sorted(set(tokens)):
        key = tok.lower()
        alts = SYNONYMS.get(key)
        if not alts:
            # hyphen join: "see-through" is one token; also try the plain word
            alts = SYNONYMS.get(key.replace("-", ""))
        if not alts:
            continue
        for alt in sorted(alts):
            if alt != key and f" {alt} " not in f" {text} ":
                return text.replace(tok, alt, 1)
    return None


def _polarity_flip_variant(text: str) -> Optional[str]:
    """Replace the first direction word with its antonym (both sides are
    shipped DIRECTION_WORDS with opposite sign). Deterministic."""
    from assistant.core.lexicon import DIRECTION_WORDS
    tokens = re.findall(r"[a-z][a-z-]+", text)
    for tok in tokens:
        sign = DIRECTION_WORDS.get(tok.lower())
        if sign is None:
            continue
        opposite = _FLIP.get(tok.lower())
        if opposite and DIRECTION_WORDS.get(opposite) == -sign:
            return text.replace(tok, opposite, 1)
    return None


# Antonym pairs drawn ONLY from words already in DIRECTION_WORDS (both
# sides present, opposite signs) — the flip is always a real direction
# word the router understands, never invented vocabulary.
_FLIP = {
    "thinner": "thicker", "thicker": "thinner",
    "smaller": "bigger", "bigger": "smaller",
    "slimmer": "thicker",
    "larger": "smaller",
    "wider": "thinner",
    "reduce": "increase", "increase": "reduce",
    "decrease": "increase",
    "less": "more", "more": "less",
    "fewer": "more",
    "lower": "raise", "raise": "lower",
    "shrink": "expand", "expand": "shrink",
    "enlarge": "reduce",
    "brighter": "darker", "darker": "brighter",
    "faster": "slower", "slower": "faster",
    "quicker": "slower",
    "softer": "harder", "harder": "softer",
    "smoother": "harder",
    "later": "earlier", "earlier": "later",
    "sooner": "later",
    "transparent": "opaque", "translucent": "opaque",
    "opaque": "transparent",
    "see-through": "opaque",
    "rounder": "sharper", "sharper": "rounder",
    "lighter": "heavier", "heavier": "lighter",
}


def generate_variants(text: str) -> Dict[str, List[str]]:
    """All deterministic variants of one request, per transform. Pure
    function — the same text always yields the same variant list."""
    out: Dict[str, List[str]] = {t: [] for t in TRANSFORMS}
    stripped = text.strip().rstrip(".").strip()
    lowered = stripped.lower()
    frames = []
    for f in _PARAPHRASE_FRAMES:
        v = f.format(t=lowered)
        # no-op guard: never re-emit the original, never stack "please"
        if v == lowered or v.startswith("please please"):
            continue
        frames.append(v)
    out["paraphrase"] = frames
    syn = _synonym_variant(stripped)
    if syn and syn.lower() != lowered:
        out["synonym"] = [syn]
    for pattern, repl in _UNIT_VARANTS:
        if pattern.search(stripped):
            out["unit_change"] = [pattern.sub(repl, stripped)]
            break
    typo = _typo_of(stripped)
    if typo and typo.lower() != lowered:
        out["typo"] = [typo]
    flip = _polarity_flip_variant(stripped)
    if flip and flip.lower() != lowered:
        out["polarity_flip"] = [flip]
    return out


def run_metamorphic(split: str = "dev") -> Dict[str, Any]:
    """Run every generated variant of every dev routing item through the
    cold-start router and score the metamorphic invariants.

    Skipped items (no variants, or a pre-declared expect_verdict — those
    are honesty checks whose "right answer" is a verdict, not a surface,
    and wrapping them in politeness frames measures nothing) are counted
    and reported, never silently dropped.
    """
    from assistant.core.lexicon import DIRECTION_WORDS, direction_of
    from assistant.core.router import DEFAULT_STATE, Router

    data = load_set("routing", split)
    router = Router()
    consistency: Dict[str, List[int]] = {t: [] for t in TRANSFORMS}
    breaks: List[Dict[str, Any]] = []
    n_variants = 0
    n_skipped = 0

    for item in data["items"]:
        text = item["text"]
        accept = set(item.get("accept") or ())
        if item.get("expect_verdict") or not accept:
            n_skipped += 1
            continue
        base = router.route(text, state=DEFAULT_STATE, k=5)
        base_top = base.top.surface if base.top else None
        base_direction = direction_of(text)
        variants = generate_variants(text)
        for transform, vlist in variants.items():
            for v in vlist:
                if transform == "polarity_flip" and base_direction == 0:
                    # no net direction to flip: the relation cannot even
                    # be stated — n/a, not a failure
                    n_skipped += 1
                    continue
                n_variants += 1
                res = router.route(v, state=DEFAULT_STATE, k=5)
                top = res.top.surface if res.top else None
                if transform == "typo":
                    ok = bool(accept & {c.surface for c in res.candidates[:3]})
                elif transform == "polarity_flip":
                    # the OBJECT is invariant; the DIRECTION must flip
                    tool_stable = (top == base_top) or (
                        res.top is not None and res.top.surface in accept)
                    ok = bool(tool_stable
                              and base_direction * direction_of(v) < 0)
                else:
                    ok = (top == base_top) or (top in accept)
                consistency[transform].append(int(ok))
                if not ok:
                    breaks.append({
                        "item": item["id"], "transform": transform,
                        "base": base_top, "variant": v, "top": top,
                        "verdict": res.verdict,
                    })

    metrics = {}
    for t in TRANSFORMS:
        metrics[f"{t}_consistency"] = _ci(consistency[t]) if consistency[t] \
            else ("n/a", "n/a", "n/a")
    return {
        "suite": "metamorphic", "split": split,
        "n_items": len(data["items"]),
        "n_variants": n_variants,
        "n_skipped": n_skipped,
        "metrics": metrics,
        "breaks": breaks[:40],
        "n_breaks": len(breaks),
        "seed": BOOT_SEED,
    }

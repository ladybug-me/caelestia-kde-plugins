"""cortex.guard — the post-ranking evidence guard (F3, exponential-build-5).

The router's four signals (lexical, semantic, fuzzy, noun) are blended
into one score, and the F2 type gate constrains candidate KINDS. Both
are score-level mechanisms. This module adds the two STRUCTURAL checks
that a blended score cannot express, each citing its evidence and each
leaving a note on the routed candidate. Pure functions, no I/O, fully
deterministic — the same ranking in, the same ranking out.

G1 — specific addressing displaces a non-tool top.

    Full name-atom coverage (every atom of a tool's own name present in
    the request) is SPECIFIC addressing by the router's own definition;
    the coverage floor comment already states it "outranks any generic
    lexical overlap a sibling tool can muster". When the top candidate
    is a PRESET or a coarse SURFACE (kinds whose documents legitimately
    collect many topics) and a tool with full coverage sits just below
    it, the blended score has let generic document overlap outrank the
    request's own explicit naming — "show the battery in the performance
    panel" names setPerformanceShowBattery; the battery-saver preset
    merely contains the word "battery". G1 swaps them. Margin-bound:
    the tool must be within ``g1_margin`` of the top, so a preset the
    user clearly invoked (a much larger gap) is never displaced.

G2 — a coarse surface matched only as a prepositional object yields to
a tool matching the rest of the request.

    "keep more notifications in history" is a request ABOUT
    notifications; "history" appears only inside the prepositional
    phrase "in history". The coarse history surface wins on that single
    token. When EVERY occurrence of the top surface's own name token(s)
    sits immediately after a preposition, the surface's evidence is
    syntactically incidental, and the best tool that matches the REST
    of the request (within ``g2_margin``) is promoted. Preference among
    qualifying tools is by (matched distinct content tokens, score) —
    the tool that actually addresses more of the request wins, score
    breaks ties. A surface named outside a prepositional phrase ("show
    history") is never touched.

Both rules only REORDER the already-scored candidate list; they never
invent scores, never touch candidates below the promoted one, and never
fire together (G1 wins; G2 refuses to stack on a G1 swap).
"""

from __future__ import annotations

import re
from typing import FrozenSet, List, Mapping, Sequence, Tuple

__all__ = ["apply", "PREPOSITIONS", "STOPWORD_FLOOR"]

# The prepositions that mark an incidental (object) mention. Deliberately
# small: a preposition is only evidence when it RELATES two things, and
# these are the relating prepositions of customization requests
# ("notifications in history", "battery in the panel").
PREPOSITIONS: FrozenSet[str] = frozenset(
    {"in", "inside", "within", "on", "at", "from"}
)

# Content tokens shorter than this are not evidence of addressing
# ("in", "on", "up" are noise; a 2-char atom like "id" is already
# handled upstream by the atom/stopword machinery).
STOPWORD_FLOOR = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def _surface_name_tokens(surface: str) -> List[str]:
    return [w for w in _WORD_RE.findall(surface.lower()) if w]


def _pp_only(
    token: str, words: Sequence[str]
) -> bool:
    """True when EVERY occurrence of ``token`` in ``words`` is preceded
    by a preposition. A token that does not appear at all is NOT
    pp-only (the surface matched some other way; G2 refuses to guess)."""
    hits = [i for i, w in enumerate(words) if w == token]
    if not hits:
        return False
    return all(i > 0 and words[i - 1] in PREPOSITIONS for i in hits)


def _content_stems(
    words: Sequence[str], stems: Mapping[str, str], exclude: FrozenSet[str]
) -> FrozenSet[str]:
    out = set()
    for w in words:
        if len(w) < STOPWORD_FLOOR or w in exclude:
            continue
        s = stems.get(w, w)
        if len(s) >= STOPWORD_FLOOR:
            out.add(s)
    return frozenset(out)


def _doc_overlap(key: str, query_stems: FrozenSet[str], doc_stems: Mapping[str, FrozenSet[str]]) -> int:
    return len(query_stems & doc_stems.get(key, frozenset()))


def apply(
    scored: List[Tuple[float, str]],
    *,
    documents: Mapping[str, Tuple[str, str]],
    coverage: Mapping[str, float],
    raw_words: Sequence[str],
    stems: Mapping[str, str],
    doc_stems: Mapping[str, FrozenSet[str]],
    min_margin: float = 0.06,
    g1_margin: float = 0.10,
    g2_margin: float = 0.20,
) -> Tuple[List[Tuple[float, str]], List[str]]:
    """Adjust a sorted (score desc) candidate list. Returns the (new
    list, evidence notes). Never raises on odd input; an unfiresable
    context returns the list unchanged.

    A promoted candidate's score is floored to ``top + min_margin`` —
    the same discipline as the router's noun/coverage floors: the
    structural rule asserts the winner, and the score list must stay
    descending (rank 1 = max score) for the margin/softmax verdicts to
    stay coherent. The note on the top candidate records that a guard,
    not the blended score, produced the ordering."""
    notes: List[str] = []
    if not scored:
        return scored, notes
    top_score, top_key = scored[0]
    top_kind = documents.get(top_key, ("", ""))[1]

    def _promote(order: List[Tuple[float, str]], key: str) -> List[Tuple[float, str]]:
        pair = next(p for p in order if p[1] == key)
        rest = [p for p in order if p[1] != key]
        # +1e-4: one display-ulp of headroom — scores are stored rounded
        # to 4 decimals, so an exact min_margin bump can land a hair
        # BELOW the threshold after rounding (0.86 - 0.8 = 0.05999...).
        bumped = (round(order[0][0] + min_margin, 4) + 1e-4, pair[1])
        return [bumped] + rest

    # G1 — specific addressing displaces a non-tool top.
    if top_kind != "tool":
        best: Tuple[float, str] | None = None
        for score, key in scored[1:6]:
            if documents.get(key, ("", ""))[1] != "tool":
                continue
            if coverage.get(key, 0.0) >= 0.999:
                if best is None or score > best[0]:
                    best = (score, key)
        if best is not None and top_score - best[0] <= g1_margin:
            scored = _promote(scored, best[1])
            notes.append(
                f"guard G1: '{best[1]}' is specifically addressed (full "
                f"name-atom coverage); displaces the {top_kind} top "
                f"'{top_key}', which matched only by document overlap"
            )
            top_key, top_score, top_kind = scored[0][1], scored[0][0], "tool"

    # G2 — a coarse surface matched only as a prepositional object.
    if top_kind == "surface" and not notes:
        name_toks = _surface_name_tokens(top_key)
        if name_toks and all(_pp_only(t, raw_words) for t in name_toks):
            pp_tokens = frozenset(t for t in name_toks if t in raw_words)
            q_stems = _content_stems(raw_words, stems, pp_tokens)
            best: Tuple[Tuple[int, float, str], str] | None = None
            for score, key in scored[1:]:
                if documents.get(key, ("", ""))[1] != "tool":
                    continue
                if top_score - score > g2_margin:
                    continue
                overlap = _doc_overlap(key, q_stems, doc_stems)
                if overlap == 0:
                    continue
                rank_key = (overlap, score, key)
                if best is None or rank_key > best[0]:
                    best = (rank_key, key)
            if best is not None:
                promoted = best[1]
                scored = _promote(scored, promoted)
                notes.append(
                    f"guard G2: '{top_key}' matched only as the object of "
                    f"a preposition ({', '.join(sorted(pp_tokens))}); "
                    f"promoted '{promoted}', which matches the rest of "
                    f"the request"
                )
    return scored, notes

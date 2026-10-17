"""personal.cloze — cloze-deletion flashcard DRAFTS from note text
(exponential-build-4 C).

The gap: notes exist, review cards don't — writing cloze deletions by
hand is the chore that never happens. This module drafts candidates
mechanically and hands them to the SAME approve/reject review gate
every other proposal goes through. NOTHING here schedules anything:
a draft becomes a card (and enters the FSRS scheduler in srs.py)
only after the user approves it, exactly like rule packs are review
data until a human moves them.

The selector is a HEURISTIC, and the docstring says so plainly:
without a language model, blank selection is mechanical — capitalized
noun phrases (never sentence-initial: that's a name, not necessarily
a key term — still sometimes wrong), standalone numbers, and dates.
It will sometimes produce a bad question ("the [Mitochondria] is the
powerhouse of the cell" is fine; "[Tuesday] I met Dana" is not).
This is a quantity-over-quality tool that only earns its keep
BECAUSE review is mandatory — the draft queue is the quality control,
not the heuristic.

Honest envelope: no sentence ranking beyond position/frequency
signals, no reading of note structure beyond the markdown text the
caller passes, no cloze of math-notation-heavy lines (they blank
badly — refused, listed as skipped). State moves through plain dicts
(the caller's learned-state JSON path); this module writes NOTHING.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

__all__ = ["STATE_KEY", "draft_cards", "approve", "reject", "open_drafts",
           "promote"]

STATE_KEY = "personal_cloze_drafts"

# numbers and dates worth blanking (standalone — a number inside a
# word, a version fragment, is noise); a trailing sentence period is
# punctuation, not part of the number. dates: YYYY-MM-DD, "12 March
# 2026"-style is left to the noun-phrase pass
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d+)?)(?![\w]|\.\d)")
_DATE_RE = re.compile(r"(?<![\w])\d{4}-\d{2}-\d{2}(?![\w])")
# capitalized multi-word phrases: 2-4 consecutive capitalized words,
# NOT at sentence start (the first word of a sentence is capitalized
# for punctuation reasons, not because it is a proper noun)
_CAP_PHRASE_RE = re.compile(
    r"(?<![.!?\n] )(?<![.!?\n])(([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}))")
_MATH_RE = re.compile(r"[=\\^$_]|\b(?:int|sum|frac)\b")

MAX_DRAFTS_PER_NOTE = 6
_MIN_SENTENCE_TOKENS = 6
_MIN_TERM_LEN = 3


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _candidates(sentence: str) -> List[Dict[str, str]]:
    """Blank candidates in one sentence, in discovery order: dates,
    numbers, capitalized phrases (sentence-initial excluded by the
    regex's lookbehind)."""
    out: List[Dict[str, str]] = []
    taken: List[str] = []
    for pattern, kind in ((_DATE_RE, "date"), (_NUM_RE, "number"),
                          (_CAP_PHRASE_RE, "phrase")):
        for m in pattern.finditer(sentence):
            term = m.group(1) if m.groups() else m.group(0)
            if kind == "phrase" and m.start() == 0:
                # sentence-initial capitalization is ambiguous (it may
                # be punctuation, not a proper noun): the FIRST word of
                # a sentence-initial phrase is dropped, the confident
                # tail is kept. "Albert Einstein published..." drafts
                # "Einstein"; "The Mitochondria produces..." drafts
                # "Mitochondria". A single-word phrase at position 0 is
                # ambiguous through and through — refused.
                words = term.split()
                if len(words) < 2:
                    continue
                term = " ".join(words[1:])
            if len(term) < _MIN_TERM_LEN:
                continue
            if any(term in t or t in term for t in taken):
                continue  # one blank per overlapping span family
            taken.append(term)
            out.append({"term": term, "kind": kind})
    return out


def draft_cards(text: str, note_id: str, max_cards: int = MAX_DRAFTS_PER_NOTE
                ) -> Dict[str, Any]:
    """Draft cloze candidates from one note's markdown text. Returns
    {"drafts": [{question, answer, kind, source}], "skipped_math": n}
    — drafts ONLY: no scheduler state is touched, no card exists until
    approve() moves it. Math-heavy sentences are skipped and counted
    (they blank badly — an honest refusal per sentence, not per note).
    """
    drafts: List[Dict[str, Any]] = []
    skipped_math = 0
    for sentence in _sentences(text):
        if len(sentence.split()) < _MIN_SENTENCE_TOKENS:
            continue
        if _MATH_RE.search(sentence):
            skipped_math += 1
            continue
        for cand in _candidates(sentence):
            if len(drafts) >= max_cards:
                break
            question = sentence.replace(cand["term"], "[...]", 1)
            drafts.append({
                "question": question,
                "answer": cand["term"],
                "kind": cand["kind"],
                "source": note_id,
            })
    return {"drafts": drafts, "skipped_math": skipped_math,
            "note": "mechanical blank selection without a language "
                    "model — sometimes a bad question; review is "
                    "mandatory, scheduling happens only after approval"}


def open_drafts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The pending review queue (oldest first, deterministic)."""
    return [dict(d, id=k) for k, d in sorted(state.get(STATE_KEY, {}).items())
            if d.get("status") == "draft"]


def approve(state: Dict[str, Any], draft_id: str) -> Dict[str, Any]:
    """Approve one draft: it becomes a REAL card via srs.new_card() —
    the FSRS scheduler's own entry point, the same one every human-
    authored card uses. This is the ONLY path from draft to card."""
    from . import srs

    drafts = state.setdefault(STATE_KEY, {})
    if draft_id not in drafts:
        raise KeyError(f"no draft {draft_id}")
    draft = drafts[draft_id]
    if draft.get("status") != "draft":
        raise ValueError(
            f"draft {draft_id} is already {draft.get('status')!r} — "
            "a decided draft is not re-decidable (the queue keeps its "
            "first decision)")
    draft["status"] = "approved"
    draft["card"] = srs.new_card()
    return dict(draft)


def reject(state: Dict[str, Any], draft_id: str,
           reason: Optional[str] = None) -> Dict[str, Any]:
    """Reject one draft: recorded (with the optional reason — a
    rejected draft is evidence about the heuristic, worth keeping),
    never scheduled, never re-surfaced as pending."""
    drafts = state.setdefault(STATE_KEY, {})
    if draft_id not in drafts:
        raise KeyError(f"no draft {draft_id}")
    draft = drafts[draft_id]
    if draft.get("status") != "draft":
        raise ValueError(
            f"draft {draft_id} is already {draft.get('status')!r} — "
            "a decided draft is not re-decidable (the queue keeps its "
            "first decision)")
    draft["status"] = "rejected"
    if reason:
        draft["reason"] = reason
    return dict(draft)


def promote(state: Dict[str, Any], draft_id: str) -> Dict[str, Any]:
    """The approved card, ready for the caller's review queue / scheduler
    bookkeeping (the caller persists; nothing writes here)."""
    draft = state.get(STATE_KEY, {}).get(draft_id)
    if not draft:
        raise KeyError(f"no draft {draft_id}")
    if draft.get("status") != "approved":
        raise ValueError(
            f"draft {draft_id} is {draft.get('status')!r} — only an "
            "approved draft promotes; there is no bypass of review")
    return {"id": draft_id, "question": draft["question"],
            "answer": draft["answer"], "card": draft["card"],
            "source": draft.get("source")}

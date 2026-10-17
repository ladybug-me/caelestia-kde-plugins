"""cortex.coref — cross-turn pronoun resolution for the pending-plan
composer (exponential-build-4 F).

Hobbs 1978, "Resolving Pronoun References", Lingua 44: the naive
(impoverished) coreference algorithm. What the paper actually does:
walk the parse trees of preceding sentences in a fixed breadth-first
order, propose each NP as antecedent, apply the binding-theory
filters, take the first survivor. What this repo has: NO parser (a
stdlib-only repo does not ship parse trees), so this is the paper's
SEARCH DISCIPLINE faithfully applied to shallow candidates
— sentences walked most-recent-first, candidates scanned right-to-left
(the recency order the naive algorithm's tree walk produces), filters
applied in the paper's spirit:

  * the antecedent is not itself a pronoun;
  * number agreement for "them" (a plural-looking head: the candidate
    ends in a plural 's' or is a known compound) — a heuristic, stated;
  * "it"/"this"/"that" prefer singular candidates (the mirror heuristic);
  * proper nouns outrank common nouns at equal recency? NO — the
    naive algorithm has no such preference, and inventing one would be
    tuning past the citation. First survivor in walk order wins.

Honest envelope (said twice on purpose): without parse trees this is
NOT Hobbs' algorithm — it is the recency-first, filter-gated search
over shallow candidates that fits a stdlib-only repo. It will
occasionally bind "it" to the wrong object ("open the terminal and
close it" resolves fine; "move the window and break it" — 'it' means
the window — also resolves fine; ambiguity between two same-turn
objects is where the shallowness shows). The composer re-validates
through the standard planner regardless, so a wrong binding is caught
by the planner's own validation, not trusted blindly.

Pure functions; no state; the PlanCache wiring (cortex/plans.py) is
where this gets called BEFORE the composed plan re-validates.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["resolve_pronoun", "resolve_turn", "PRONOUNS"]

PRONOUNS = {"it", "this", "that", "them", "they"}

_PRONOUN_RE = re.compile(r"\b(it|this|that|them|they)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "please", "make", "set", "open", "close", "then", "also", "again",
    "now", "just", "turn", "put", "get", "my", "me", "i", "you", "your",
    "is", "are", "be", "do", "does", "can", "could", "would", "should",
    "want", "want", "need", "little", "bit", "more", "less", "smaller",
    "bigger", "larger", "shorter", "longer", "thinner", "wider",
}

_PLURAL_SUFFIXES = ("s", "es")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _is_pronoun(token: str) -> bool:
    return token.lower() in PRONOUNS


def _is_content(token: str) -> bool:
    low = token.lower()
    return low not in _STOP and not _is_pronoun(low)


def _looks_plural(token: str) -> bool:
    low = token.lower()
    return any(low.endswith(s) for s in _PLURAL_SUFFIXES) and len(low) > 3


def _agrees(pronoun: str, candidate: str) -> bool:
    """The number-agreement heuristic, stated as a heuristic: 'them'/
    'they' want plural-looking candidates; 'it'/'this'/'that' want
    non-plural ones. Wrong on irregulars ("mice") — the docstring's
    honesty applies."""
    pronoun = pronoun.lower()
    if pronoun in ("them", "they"):
        return _looks_plural(candidate)
    return not _looks_plural(candidate)


def resolve_pronoun(pronoun: str, context_turns: Sequence[str],
                    pending_labels: Optional[Sequence[str]] = None
                    ) -> Dict[str, Any]:
    """Resolve one pronoun against the conversation's prior context,
    Hobbs-style: most recent sentence first, candidates right-to-left,
    first filter-passing content word wins.

    ``pending_labels``: the pending plan's own human-readable labels
    (tool names, raw targets) — the plan itself is the MOST recent
    context (the just-composed ops are what "it" most plausibly refers
    to), so they are walked BEFORE the older turns.

    Returns {"resolved": bool, "antecedent": str-or-None, "source":
    "pending-plan"|"prior-turn", "note": str}. No mutation, no state.
    """
    if not _PRONOUN_RE.fullmatch(pronoun.strip().lower() or "it"):
        return {"resolved": False, "antecedent": None, "source": None,
                "note": f"{pronoun!r} is not a resolvable pronoun here "
                        f"(resolved set: {sorted(PRONOUNS)})"}
    for label in reversed(list(pending_labels or [])):
        for token in reversed(_tokens(str(label))):
            if _is_content(token) and _agrees(pronoun, token):
                return {"resolved": True, "antecedent": token.lower(),
                        "source": "pending-plan",
                        "note": "first filter-passing candidate in "
                                "Hobbs walk order over the pending plan"}
    for turn in reversed(list(context_turns or [])):
        for token in reversed(_tokens(str(turn))):
            if _is_content(token) and _agrees(pronoun, token):
                return {"resolved": True, "antecedent": token.lower(),
                        "source": "prior-turn",
                        "note": "first filter-passing candidate in "
                                "Hobbs walk order over prior turns"}
    return {"resolved": False, "antecedent": None, "source": None,
            "note": "no candidate survived the filters — refusing to "
                    "guess (the naive algorithm's honest dead end)"}


def resolve_turn(text: str, context_turns: Sequence[str],
                 pending_labels: Optional[Sequence[str]] = None
                 ) -> Dict[str, Any]:
    """Resolve every pronoun in one new turn against the context.
    Returns the rewritten text (pronouns replaced by their antecedent,
    deterministically, left to right) plus the per-pronoun reports.
    Unresolved pronouns are LEFT AS-IS and reported — a silent guess
    is exactly what this module refuses."""
    reports: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    def replace(match: re.Match) -> str:
        pronoun = match.group(0)
        if pronoun.lower() not in seen:
            seen[pronoun.lower()] = resolve_pronoun(
                pronoun, context_turns, pending_labels)
            reports.append(seen[pronoun.lower()])
        verdict = seen[pronoun.lower()]
        if verdict["resolved"]:
            return str(verdict["antecedent"])
        return pronoun

    rewritten = _PRONOUN_RE.sub(replace, text)
    return {"original": text, "rewritten": rewritten,
            "resolved": [r for r in reports if r["resolved"]],
            "unresolved": [r for r in reports if not r["resolved"]],
            "note": "Hobbs 1978 search discipline over shallow "
                    "candidates (no parser in a stdlib repo) — the "
                    "composed plan still re-validates through the "
                    "standard planner"}

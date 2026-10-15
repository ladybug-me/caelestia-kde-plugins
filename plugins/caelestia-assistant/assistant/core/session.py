"""Conversational session state: anaphora, ellipsis, clarification.

The one-shot CLI answers each request in isolation; issue #120's
clarification examples ("I found several changes that match your
request... Apply these changes?") need a session that remembers:

- what was just discussed (``last_surfaces``) so "make it a bit
  smaller" resolves "it";
- what was just asked (``pending_question`` +
  ``pending_candidates``) so a bare "the dock one" / "the second" /
  "yes" answers the open question instead of starting a new request;
- the running transcript (bounded, serializable) for ``--json``
  machine mode and for memory.py's episodic store.

Resolution rules (all deterministic, all pure):

1. ANSWER MODE — when the previous turn ended AMBIGUOUS and the input
   is a short reply (yes/no, an ordinal, or a noun that matches one of
   the pending candidates' vocabulary), resolve the pending question
   instead of routing afresh.
2. ANAPHORA — a leading pronoun ("it", "that", "them", "the same
   thing") is replaced by the last-discussed surface's user-facing
   name atoms.
3. ELLIPSIS — an input that carries cues (direction/bool/position/
   number) but no addressable noun re-attaches to the last surface
   ("a bit smaller" after discussing the dock -> "dock smaller").

The session NEVER writes; it produces resolved text + bookkeeping.
The pipeline and the CLI gates (``--apply``, ``--confirm``, ledger
approval) stay the only write paths, unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .router import RouteResult, RouterState, DEFAULT_STATE, route

# ---------------------------------------------------------------------------
# Grammar.
# ---------------------------------------------------------------------------

_PRONOUN_RE = re.compile(
    r"^\s*(?:it|that|them|those|the same(?:\s+thing|\s+one)?|both|all of them)\b",
    re.IGNORECASE,
)
# Object-position pronoun: "actually make IT bigger", "set THAT to 0.8" —
# the pronoun need not lead the sentence, it just has to be the object of
# a settings verb (which keeps "it" in unrelated positions untouched).
_PRONOUN_OBJECT_RE = re.compile(
    r"\b(?:make|making|turn|turning|set|setting|move|moving|put|putting|shrink|shrinking|"
    r"grow|growing|increase|increasing|decrease|decreasing|raise|raising|lower|lowering|"
    r"show|hide|enable|disable)\s+(?:it|them|that|those)\b",
    re.IGNORECASE,
)
_ORDINALS: Dict[str, int] = {
    "first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2, "fourth": 3, "4th": 3, "four": 3,
    "fifth": 4, "5th": 4, "five": 4,
}
_ORDINAL_RE = re.compile(r"\b(?:the\s+)?(\d+|first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\b(?:\s+one)?\b")
_YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|y|go ahead|do it|apply)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(?:no|nope|n|nah|cancel|stop|never\s+mind)\b", re.IGNORECASE)

_MAX_TURNS = 64  # bounded transcript; sessions are cheap, memory is not


# ---------------------------------------------------------------------------
# State shapes.
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One recorded exchange (bounded to the last _MAX_TURNS)."""

    text: str                      # exactly what the user said
    resolved: str                  # after anaphora/ellipsis resolution
    verdict: str = ""              # router verdict for the resolved text
    surfaces: List[str] = field(default_factory=list)  # top candidates
    note: str = ""                 # how the turn was resolved, if at all


@dataclass
class SessionState:
    """The whole conversational context. Serializable via to_dict."""

    turns: List[Turn] = field(default_factory=list)
    last_surfaces: List[str] = field(default_factory=list)
    pending_question: Optional[str] = None
    pending_candidates: List[str] = field(default_factory=list)
    pending_resolved_text: str = ""
    # The session-scoped PENDING plan cache's serialized form (owned by
    # cortex/plans.py PlanCache — the what-if iteration primitive). A raw
    # passthrough slot: this class carries it across bridge round-trips
    # without interpreting it.
    pending_plan: Optional[Dict[str, object]] = None

    # -- bookkeeping ---------------------------------------------------------

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > _MAX_TURNS:
            self.turns = self.turns[-_MAX_TURNS:]

    def to_dict(self) -> Dict[str, object]:
        return {
            "turns": [
                {"text": t.text, "resolved": t.resolved, "verdict": t.verdict,
                 "surfaces": t.surfaces, "note": t.note}
                for t in self.turns
            ],
            "last_surfaces": self.last_surfaces,
            "pending_question": self.pending_question,
            "pending_candidates": self.pending_candidates,
            "pending_plan": self.pending_plan,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, object]]) -> "SessionState":
        state = SessionState()
        if not data:
            return state
        for row in data.get("turns", []) or []:
            if isinstance(row, dict):
                state.turns.append(Turn(
                    text=str(row.get("text", "")),
                    resolved=str(row.get("resolved", "")),
                    verdict=str(row.get("verdict", "")),
                    surfaces=[str(s) for s in row.get("surfaces", []) or []],
                    note=str(row.get("note", "")),
                ))
        state.last_surfaces = [str(s) for s in data.get("last_surfaces", []) or []]
        state.pending_question = (
            str(data["pending_question"]) if data.get("pending_question") else None
        )
        state.pending_candidates = [str(c) for c in data.get("pending_candidates", []) or []]
        state.pending_plan = data.get("pending_plan") if isinstance(
            data.get("pending_plan"), dict) else None
        return state


# ---------------------------------------------------------------------------
# Reference resolution.
# ---------------------------------------------------------------------------


def _surface_words(surface: str) -> str:
    """User-facing words for one surface ("setDockIconSize" -> "dock icon size")."""
    from .corpus import tool_atoms
    from assistant.adapters.caelestia.registry import tool_by_name

    spec = tool_by_name(surface)
    if spec is not None:
        return " ".join(tool_atoms(spec)) or surface
    from .lexicon import camel_split

    return " ".join(camel_split(surface.replace("preset:", "")))


def resolve_reference(text: str, state: SessionState) -> Tuple[str, str]:
    """Resolve anaphora/ellipsis against the session. Returns
    ``(resolved_text, note)`` — the note explains what was substituted
    (shown to the user; nobody should wonder what "it" became)."""
    raw = text.strip()
    if not raw:
        return raw, ""

    pronoun = _PRONOUN_RE.match(raw) or _PRONOUN_OBJECT_RE.search(raw)
    cue_only = _has_cues_only(raw)

    if not state.last_surfaces:
        if pronoun:
            return raw, "no earlier setting in this session to resolve the pronoun against"
        return raw, ""

    if pronoun:
        target = " ".join(_surface_words(s) for s in state.last_surfaces[:2])
        if _PRONOUN_RE.match(raw):
            resolved = _PRONOUN_RE.sub(target, raw, count=1)
        else:
            resolved = _PRONOUN_OBJECT_RE.sub(target, raw, count=1)
        return resolved, f"pronoun resolved to: {', '.join(state.last_surfaces[:2])}"

    if cue_only:
        target = " ".join(_surface_words(s) for s in state.last_surfaces[:1])
        resolved = f"{target} {raw}"
        return resolved, f"ellipsis resolved against: {state.last_surfaces[0]}"

    return raw, ""


def _has_cues_only(text: str) -> bool:
    """Direction/bool/position/number words with no addressable noun:
    "a bit smaller", "more", "to the left", "0.8". Compares STEMS — the
    tokenizer stems both sides, so the cue vocabulary must be stemmed
    too ("smaller" -> "small")."""
    from .lexicon import BOOL_OFF_WORDS, BOOL_ON_WORDS, DIRECTION_WORDS, stem
    from .router import extract_cues
    from .vectorize import tokenize

    cues = extract_cues(text)
    if not cues:
        return False
    tokens = tokenize(text)
    cue_vocabulary = (
        {stem(w) for w in DIRECTION_WORDS}
        | {stem(w) for w in BOOL_ON_WORDS}
        | {stem(w) for w in BOOL_OFF_WORDS}
        | {"top", "bottom", "left", "right", "reset", "default",
           "toggle", "bit", "tad", "much", "on", "off"}
    )
    non_cue = [t for t in tokens if t not in cue_vocabulary]
    return len(non_cue) == 0 and len(tokens) > 0


# ---------------------------------------------------------------------------
# Clarification-answer resolution.
# ---------------------------------------------------------------------------


def _answer_index(text: str) -> Optional[int]:
    match = _ORDINAL_RE.search(text.lower())
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        idx = int(token) - 1
    else:
        # "second" -> 1; word-number to index via the first-of-list rule
        word = token.rstrip("st")
        idx = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}.get(
            word, {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}.get(token)
        )
    return idx if idx is not None and idx >= 0 else None


def resolve_answer(text: str, state: SessionState) -> Tuple[Optional[str], str]:
    """When the previous turn was an open AMBIGUOUS question, try to read
    this input as the ANSWER. Returns ``(chosen_surface, note)``; None
    when the input is not recognizable as an answer to the pending
    question (the caller then treats it as a fresh request)."""
    if not state.pending_candidates or not state.pending_question:
        return None, ""
    lowered = text.strip().lower()

    if _YES_RE.match(lowered):
        return state.pending_candidates[0], "answered yes — taking the first candidate"
    if _NO_RE.match(lowered):
        return None, "answered no — pending question dropped"

    idx = _answer_index(lowered)
    if idx is not None and idx < len(state.pending_candidates):
        return state.pending_candidates[idx], f"answered by ordinal — candidate {idx + 1}"

    # A noun reply: does it match exactly ONE pending candidate's vocab?
    from .vectorize import tokenize

    reply_tokens = set(tokenize(lowered))
    if not reply_tokens:
        return None, ""
    hits: List[str] = []
    for surface in state.pending_candidates:
        cand_tokens = set(tokenize(_surface_words(surface)))
        if reply_tokens & cand_tokens:
            hits.append(surface)
    if len(hits) == 1:
        return hits[0], "answered by naming the setting"
    return None, ""


# ---------------------------------------------------------------------------
# The turn engine.
# ---------------------------------------------------------------------------


@dataclass
class SessionTurnResult:
    """What one session turn produced (machine-readable surface)."""

    mode: str                     # "answer" | "request" | "empty"
    resolved_text: str
    route: Optional[RouteResult] = None
    chosen_surface: Optional[str] = None
    note: str = ""
    question: Optional[str] = None
    verdict: str = ""


def turn(text: str, state: SessionState,
         router_state: RouterState = DEFAULT_STATE) -> SessionTurnResult:
    """Process one conversational turn, mutating ``state`` (the session
    IS the state; this is the one deliberately stateful cortex module —
    everything below it stays pure)."""
    raw = (text or "").strip()
    if not raw:
        return SessionTurnResult(mode="empty", resolved_text="", note="empty turn")

    # 1. Answer mode: does this reply resolve the pending question?
    chosen, answer_note = resolve_answer(raw, state)
    if chosen is not None:
        state.pending_question = None
        state.pending_candidates = []
        resolved = f"{_surface_words(chosen)} {state.pending_resolved_text}".strip()
        state.last_surfaces = [chosen]
        result = route(resolved, state=router_state)
        state.record(Turn(text=raw, resolved=resolved, verdict=result.verdict,
                          surfaces=[chosen], note=answer_note))
        return SessionTurnResult(
            mode="answer", resolved_text=resolved, route=result,
            chosen_surface=chosen, note=answer_note, verdict=result.verdict,
        )

    # 2. Fresh request: resolve references, route, update context.
    resolved, note = resolve_reference(raw, state)
    result = route(resolved, state=router_state)
    surfaces = [c.surface for c in result.candidates[:3]]

    if result.verdict == "AMBIGUOUS":
        state.pending_question = result.question
        state.pending_candidates = surfaces
        state.pending_resolved_text = resolved
    else:
        state.pending_question = None
        state.pending_candidates = []
        state.pending_resolved_text = ""

    if surfaces:
        state.last_surfaces = surfaces
    state.record(Turn(text=raw, resolved=resolved, verdict=result.verdict,
                      surfaces=surfaces, note=note))
    return SessionTurnResult(
        mode="request", resolved_text=resolved, route=result,
        note=note, question=result.question, verdict=result.verdict,
    )

"""cortex.plans — the session-scoped PENDING plan cache (phase 2.5).

The gap this closes: a follow-up request composed against the LAST
APPLIED state loses the PENDING plan. With what-if iteration ("make the
bar thinner", "and the dock smaller too", "actually spacing tighter as
well") the user amasses changes BEFORE committing; each turn must
compose against the ACCUMULATED pending ops, not just the live file.

This is the SHARED PRIMITIVE the what-if mode (settings/consequences.py
+ `settings --what-if`) consumes: one plan cache per session, carried in
the session dict (bridge round-trips it), read by the chat loop, the
what-if surface, and available to any future consumer — never
duplicated.

Composition semantics (mirroring the compound planner's own rules):
- ops compose by TOOL NAME, LATER WINS (the spoken order the compound
  layer already uses);
- the composed list re-VALIDATES through the standard planner before
  anything is proposed — the cache never bypasses validation;
- COMMIT happens only when an apply actually went through (the gate
  closes the loop); a refused apply leaves the ops pending for the next
  turn (that is the point);
- DISCARD is explicit ("never mind", "discard", "start over" in the
  chat loop) or implicit when a new topic produces a plan whose ops
  fully replace... never — replacement only happens per-tool via the
  later-wins rule, so nothing is silently dropped.

Serialization: a plain dict in the session payload
(``session["pending_plan"]``); bounded to the last MAX_PENDING ops.
Pure module: no I/O — the caller persists (the session dict already
round-trips through the bridge/chat state).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["PlanCache", "DISCARD_RE", "MAX_PENDING_OPS"]

MAX_PENDING_OPS = 12  # a session composes a handful of changes, not a wall

# Explicit discard phrases in the chat loop (the honest way to drop a
# pending plan instead of refusing it forever).
DISCARD_RE = re.compile(
    r"^\s*(?:never\s?mind|nevermind|discard|drop\s+(?:it|that|the\s+plan)|"
    r"start\s+over|forget\s+it|cancel\s+that)\b",
    re.IGNORECASE,
)


class PlanCache:
    """The session's pending ops (tool-name keyed, later wins)."""

    def __init__(self, ops: Optional[List[Dict[str, Any]]] = None,
                 turns: Optional[List[str]] = None):
        self.ops: List[Dict[str, Any]] = [dict(op) for op in (ops or [])]
        self.turns: List[str] = list(turns or [])

    # -- composition ---------------------------------------------------------

    def pending_labels(self) -> List[str]:
        """Human-readable labels of the pending ops (tool names and raw
        notes) — the candidate antecedents pronoun resolution walks
        first, since the pending plan is the most recent context."""
        labels: List[str] = []
        for op in self.ops:
            for key in ("raw", "note"):
                # "tool" is deliberately excluded: a camelCase identifier
                # is not an English antecedent a user could have meant
                value = op.get(key)
                if value:
                    labels.append(str(value))
        return labels

    def resolve_turn(self, text: str) -> Dict[str, Any]:
        """Cross-turn pronoun resolution (exponential-build-4 F):
        'open firefox' then 'pin it' — the 'it' resolves against the
        pending plan's labels first, then prior turns, Hobbs-style,
        BEFORE the rewritten turn goes anywhere near a parser or the
        planner. The rewritten text is what compose() should see; the
        report is carried so the caller can show its work."""
        from . import coref
        return coref.resolve_turn(text, self.turns, self.pending_labels())

    def compose(self, new_ops: Sequence[Dict[str, Any]],
                text: str = "") -> List[Dict[str, Any]]:
        """Merge the pending ops with a new turn's ops: same-tool ops are
        REPLACED by the later one (spoken order wins), new tools append.
        The merged list becomes the new pending state and is returned for
        re-validation by the standard planner. Bounded at MAX_PENDING_OPS
        (oldest dropped — a session is not a hoarder)."""
        merged = {str(op.get("tool", "")): dict(op) for op in self.ops}
        for op in new_ops:
            merged[str(op.get("tool", ""))] = dict(op)
        ordered = list(merged.values())[-MAX_PENDING_OPS:]
        self.ops = ordered
        if text and (not self.turns or self.turns[-1] != text):
            self.turns.append(text[:80])
            self.turns = self.turns[-MAX_PENDING_OPS:]
        return [dict(op) for op in self.ops]

    def pending(self) -> List[Dict[str, Any]]:
        """The current pending ops (copy — callers must not mutate)."""
        return [dict(op) for op in self.ops]

    def pending_count(self) -> int:
        return len(self.ops)

    def summary(self) -> str:
        """One line the chat card can show: what is still pending."""
        if not self.ops:
            return ""
        tools = ", ".join(str(op.get("tool", "?")) for op in self.ops)
        return (f"{len(self.ops)} pending change(s) not applied yet "
                f"({tools})")

    # -- lifecycle -------------------------------------------------------------

    def commit(self) -> List[Dict[str, Any]]:
        """An apply went through: hand over the ops and clear. Returns the
        committed ops (for the record/episode)."""
        ops = self.ops
        self.ops = []
        self.turns = []
        return ops

    def discard(self) -> None:
        """Explicit drop (never mind / start over)."""
        self.ops = []
        self.turns = []

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"ops": [dict(op) for op in self.ops],
                "turns": list(self.turns)}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlanCache":
        data = data or {}
        ops = [dict(op) for op in (data.get("ops") or [])
               if isinstance(op, dict)]
        turns = [str(t) for t in (data.get("turns") or [])]
        return cls(ops, turns)

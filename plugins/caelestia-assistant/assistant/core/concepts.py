"""Taught concepts (exp-build-5 F15): user-named bundles of validated
tool calls — "make it glassy" becomes a reusable, learnable command.

Contract:
- A concept is created ONLY from calls the user explicitly named and
  consented to (CLI teach, or a plan the user approved and then named).
  Nothing is ever auto-created from utterances.
- Every call is registry-checked at teach time and re-validated by the
  standard planner at apply time. Teaching validates, it never applies.
- Recall is char-n-gram Dice + lexicon overlap over the concept's name
  and taught example utterances. Below threshold: ABSTAIN (honest miss).
  Two concepts within the ambiguity gap: AMBIGUOUS. A strong built-in
  preset match alongside: CONFLICT_ASK — both options are shown, the
  user picks. Recall proposes; only the consent gate applies.
- Outcomes learn: per-(tool, value) Beta posteriors update on
  accept/reject, surfaced in `list` (never silent replacement of the
  built-in presets).
- Sharing: export/import is a reviewable, capped JSON diff (recipes
  only — posteriors are personal and never exported).

Bounded: MAX_CONCEPTS concepts, MAX_EXAMPLES examples, MAX_CALLS calls
per concept; import refuses overflow. Stdlib only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from assistant.adapters.caelestia.registry import tool_by_name

__all__ = ["ConceptError", "teach", "forget", "recall", "record_outcome",
           "list_concepts", "export_diff", "import_diff",
           "MAX_CONCEPTS", "MAX_CALLS", "MAX_EXAMPLES"]

MAX_CONCEPTS = 64
MAX_CALLS = 12
MAX_EXAMPLES = 4
RECALL_THRESHOLD = 0.62     # Dice similarity to answer at all
AMBIGUOUS_GAP = 0.08        # two concepts closer than this ask instead
PRESET_MATCH = 0.55         # preset similarity that triggers CONFLICT_ASK
_STATE_KEY = "taught_concepts"


class ConceptError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# storage shape: state[_STATE_KEY] = {name: {calls, examples, alpha, beta}}
# calls: [[tool, value]], alpha/beta: {"tool=<json value>": [a, b]}
# ---------------------------------------------------------------------------


def _table(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.setdefault(_STATE_KEY, {})


def _call_key(tool: str, value: Any) -> str:
    return f"{tool}={value!r}"


def _parse_call(call: Any) -> Tuple[str, Any]:
    if isinstance(call, str):
        name, sep, value_text = call.partition("=")
        if not sep:
            raise ConceptError(f"--call expects TOOL=VALUE (got {call!r})")
        import json
        try:
            value = json.loads(value_text)
        except ValueError:
            value = value_text
        return name.strip(), value
    if isinstance(call, (list, tuple)) and len(call) == 2:
        return str(call[0]), call[1]
    raise ConceptError(f"bad call {call!r}")


def _ngrams(text: str, ns: Tuple[int, ...] = (2, 3, 4)) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    grams: set = set()
    for n in ns:
        if len(t) < n:
            grams.add(t)
            continue
        for i in range(len(t) - n + 1):
            grams.add(t[i:i + n])
    return grams


def _dice(a: str, b: str) -> float:
    ga, gb = _ngrams(a), _ngrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3]


# ---------------------------------------------------------------------------
# teach / forget / list
# ---------------------------------------------------------------------------


def teach(state: Dict[str, Any], name: str, calls: List[Any],
          examples: Optional[List[str]] = None,
          source: str = "explicit") -> Dict[str, Any]:
    name = (name or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9 _-]{1,39}", name):
        raise ConceptError(
            f"concept name must be 2-40 chars of letters/digits/space/-_: {name!r}")
    if not calls:
        raise ConceptError("a concept needs at least one call")
    if len(calls) > MAX_CALLS:
        raise ConceptError(f"a concept takes at most {MAX_CALLS} calls")
    parsed: List[Tuple[str, Any]] = [_parse_call(c) for c in calls]
    for tool, _value in parsed:
        if tool_by_name(tool) is None:
            raise ConceptError(f"{tool} is not a registry tool")
    table = _table(state)
    if name not in table and len(table) >= MAX_CONCEPTS:
        raise ConceptError(f"concept store full ({MAX_CONCEPTS}); forget one first")
    exs = [e.strip() for e in (examples or []) if e and e.strip()]
    if len(exs) > MAX_EXAMPLES:
        raise ConceptError(f"at most {MAX_EXAMPLES} example phrases")
    old = table.get(name) or {}
    table[name] = {
        "calls": [[t, v] for t, v in parsed],
        "examples": exs,
        "alpha": old.get("alpha", {}),
        "beta": old.get("beta", {}),
        "source": source,
    }
    return {"name": name, "calls": table[name]["calls"],
            "examples": exs, "n_concepts": len(table)}


def forget(state: Dict[str, Any], name: str) -> bool:
    name = (name or "").strip().lower()
    return bool(_table(state).pop(name, None))


def list_concepts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for name in sorted(_table(state)):
        c = _table(state)[name]
        posterior = []
        for tool, value in c["calls"]:
            key = _call_key(tool, value)
            a = c["alpha"].get(key, 0.0)
            b = c["beta"].get(key, 0.0)
            posterior.append({"tool": tool, "value": value,
                              "p": (round(a / (a + b), 3) if a + b else None),
                              "n": int(a + b)})
        out.append({"name": name, "calls": c["calls"],
                    "examples": c["examples"], "posterior": posterior,
                    "source": c.get("source", "explicit")})
    return out


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def _concept_score(state: Dict[str, Any], text: str,
                   name: str) -> float:
    c = _table(state)[name]
    scores = [_dice(text, name.replace("_", " "))]
    scores.extend(_dice(text, ex) for ex in c["examples"])
    best = max(scores)
    words = _content_words(text)
    named = _content_words(name)
    if named and all(w in words for w in named):
        best = min(1.0, best + 0.15)   # all name words literally present
    return best


def recall(state: Dict[str, Any], text: str,
           presets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The recall verdict for one utterance:
    ABSTAIN (no concept close), AMBIGUOUS (two concepts too close),
    CONFLICT_ASK (a built-in preset matches too), or RECALL (propose the
    concept's calls — the caller still gates the apply)."""
    text = (text or "").strip()
    table = _table(state)
    if not table or not text:
        return {"verdict": "ABSTAIN", "reason": "no concepts taught"
                if not table else "empty utterance"}
    ranked = sorted(((_concept_score(state, text, n), n) for n in table),
                    key=lambda kv: (-kv[0], kv[1]))
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else (0.0, None))
    if top[0] < RECALL_THRESHOLD:
        return {"verdict": "ABSTAIN", "reason": "no concept close enough",
                "best": {"name": top[1], "score": round(top[0], 3)}}
    if second[1] is not None and top[0] - second[0] < AMBIGUOUS_GAP \
            and second[0] >= RECALL_THRESHOLD:
        return {"verdict": "AMBIGUOUS",
                "candidates": [{"name": top[1], "score": round(top[0], 3)},
                               {"name": second[1], "score": round(second[0], 3)}]}
    preset_hit = None
    text_words = set(_content_words(text))
    for p in (presets or []):
        s = max(_dice(text, str(p.get("name", ""))),
                _dice(text, str(p.get("label", ""))),
                _dice(text, str(p.get("description", ""))))
        # short preset names lose at raw Dice against a long utterance;
        # if every distinctive name/label word appears literally, that is
        # a real conflict signal, not a similarity artifact
        name_words = [w for w in _content_words(
            f"{p.get('name', '')} {p.get('label', '')}") if len(w) >= 4]
        if name_words and all(w in text_words for w in name_words):
            s = max(s, 0.85)
        if s >= PRESET_MATCH:
            preset_hit = {"name": str(p.get("name")), "score": round(s, 3)}
            break
    if preset_hit:
        return {"verdict": "CONFLICT_ASK", "concept": {"name": top[1],
                "score": round(top[0], 3)}, "preset": preset_hit,
                "question": (f"use your taught concept {top[1]!r} or the "
                             f"built-in preset {preset_hit['name']!r}?")}
    c = table[top[1]]
    posterior = []
    for tool, value in c["calls"]:
        k = _call_key(tool, value)
        a, b = c["alpha"].get(k, 0.0), c["beta"].get(k, 0.0)
        posterior.append({"tool": tool, "value": value,
                          "p": (round(a / (a + b), 3) if a + b else None),
                          "n": int(a + b)})
    return {"verdict": "RECALL", "name": top[1], "score": round(top[0], 3),
            "calls": [{"name": t, "value": v} for t, v in c["calls"]],
            "posterior": posterior}


def record_outcome(state: Dict[str, Any], name: str,
                   accepted: bool) -> Dict[str, Any]:
    """Beta update per (tool, value) after a consented apply/reject."""
    name = (name or "").strip().lower()
    c = _table(state).get(name)
    if c is None:
        raise ConceptError(f"unknown concept {name!r}")
    for tool, value in c["calls"]:
        key = _call_key(tool, value)
        a, b = c["alpha"].get(key, 0.0), c["beta"].get(key, 0.0)
        c["alpha"][key], c["beta"][key] = (
            (a + 1.0, b) if accepted else (a, b + 1.0))
    return {"name": name, "accepted": accepted}


# ---------------------------------------------------------------------------
# sharing: reviewable recipe diffs (no posteriors)
# ---------------------------------------------------------------------------


def export_diff(state: Dict[str, Any],
                names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    table = _table(state)
    want = [n.lower() for n in (names or table)] if names else sorted(table)
    out = []
    for n in want:
        c = table.get(n)
        if c is None:
            raise ConceptError(f"unknown concept {n!r}")
        out.append({"name": n, "calls": c["calls"], "examples": c["examples"]})
    return out


def import_diff(state: Dict[str, Any], entries: List[Dict[str, Any]],
                force: bool = False) -> Dict[str, int]:
    if not isinstance(entries, list) or len(entries) > MAX_CONCEPTS:
        raise ConceptError(f"import expects a list of at most {MAX_CONCEPTS} entries")
    added = updated = skipped = 0
    for e in entries:
        name = str(e.get("name", "")).strip().lower()
        exists = name in _table(state)
        if exists and not force:
            skipped += 1
            continue
        teach(state, name, e.get("calls") or [],
              examples=e.get("examples"), source="imported")
        if exists:
            updated += 1
        else:
            added += 1
    return {"added": added, "updated": updated, "skipped": skipped}

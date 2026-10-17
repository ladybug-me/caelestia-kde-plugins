"""retrieval.cbr — the Case-Based Reasoning cycle over caller-supplied cases.

Aamodt & Plaza 1994, "Case-Based Reasoning: Foundational Issues,
Methodological Variations, and System Approaches", AI Communications
7(1), 39-59 — the RETRIEVE -> REUSE -> REVISE -> RETAIN cycle this
module formalizes, layered on the BM25 retrieval this package already
ships. What was adapted: the cycle is instantiated for THIS assistant's
honesty rules (caller-supplied data, ledger-gated retention, explicit
abstention) rather than copied as a generic framework.

A CASE is a plain dict:

    {"id": str, "problem": str, "resolution": str,
     "accepted": bool, "source": str}

- problem:    the user-visible description text (what was wrong);
- resolution: what fixed it;
- accepted:   the recorded outcome (did the resolution work);
- source:     where the case came from ("issue #6", "docs/…", …).

THE CASE BASE IS CALLER-SUPPLIED (a plain list) — the same
caller-supplied-data pattern as brain/workspace.py, stated plainly:
this checkout persists no case store of its own, so the caller owns
the list and persists it through the learned-state path it already
owns. This module writes NOTHING except the one ledger proposal that
retain() files through the Ledger it is handed.

The four steps:

- retrieve(case_base, query, k): BM25 over the cases' problem texts.
  NOT a reimplementation: an in-memory index dict in exactly the shape
  indexer.build_index() produces, handed to the SHIPPED Searcher
  (search.py — Okapi BM25 with indexer's K1/B constants and the same
  indexer.tokenize vocabulary). Duplicate case ids are rejected with
  ValueError, never silently absorbed.
- reuse(case, query, ...): the ADAPTATION-RULE TABLE below — explicit,
  cited, data-driven; every fired rule is reported with its id and its
  changes, nothing is adapted silently. A resolution naming a settings
  tool that is absent from the registry (with no substitution rule to
  repair it) ABSTAINS instead of transferring a fix that cannot run.
- revise(adapted, validate_fn): the CALLER's validator marks steps that
  fail, with the validator's own reason; this module never applies
  anything (the settings planner's validation is the real validator in
  production; tests inject a fake).
- retain(case_base, new_case, ledger): the retain step is ITSELF a
  ledger proposal (brain/ledger.py, kind="cbr_case") — never a silent
  append. The caller appends the case to its own list only after the
  ledger's approve/reject flow says so.

Deterministic and pure apart from the ledger call: no RNG, no
timestamps, no filesystem access. Two runs over the same inputs produce
identical bytes (pinned by test).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .indexer import B, K1, VERSION, split_sentences, tokenize
from .search import Searcher

__all__ = ["ADAPTATION_RULES", "TOOL_RENAMES", "cbr_cycle", "retain",
           "retrieve", "reuse", "revise"]

_REQUIRED_CASE_KEYS = ("id", "problem", "resolution", "accepted", "source")

# Settings tool names are setXxx-shaped; this is the same convention as
# diagnostics/settings_join.py's _TOOL_NAME_RE (the repo's own rule for
# finding settings-tool mentions in free text — cited reuse of that
# convention, not a second opinion about what a tool name looks like).
_TOOL_MENTION_RE = re.compile(r"\b(set[A-Z][A-Za-z0-9]+)\b")

# A bare numeric literal in a resolution line, scaled by the numeric
# baseline rule below. Glued to letters/underscores/commas it is NOT a
# bare literal (versions, tool args, 1,000-grouped numbers are left
# alone rather than corrupted); a trailing sentence period is allowed.
_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z_.,])(\d+(?:\.\d+)?)(?![0-9A-Za-z_,])")

# Confidence defaults for the retain proposal — deliberately
# conservative FIXED numbers (0.8 for a user-confirmed outcome, 0.3 for
# a case retained for its problem-side signal only), not calibrated
# values; the caller can always pass its own.
_ACCEPTED_CONFIDENCE = 0.8
_REJECTED_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# The adaptation-rule table (Aamodt & Plaza 1994, the REUSE step)
# ---------------------------------------------------------------------------

# Tool renames this module is allowed to substitute, old name -> current
# registry name. The shipped table is INTENTIONALLY EMPTY, said plainly:
# a full read of assistant/settings/tools.json (the generated, citation-
# verified registry of every then-exposed setXxx tool) found NO rename-shaped pair
# — every row is a current name with its citations, and the registry
# carries no history of old names (nothing in tools.json, the settings design notes (see module docstrings) or
# the settings sources records a tool that was renamed). Manufacturing
# an old->new pair without a source would be a fake claim, so the table
# ships empty (pinned by test) and the SUBSTITUTION MECHANISM is still
# exercised through caller-supplied mappings.
TOOL_RENAMES: Tuple[Tuple[str, str], ...] = ()

# The adaptation rules themselves. Each entry documents when it applies,
# what it does, and why (the citation). The ids are what a fired rule
# reports back; a rule that changes nothing does not fire — adaptations
# are never silent, and unfired rules are never claimed.
ADAPTATION_RULES: Tuple[Dict[str, str], ...] = (
    {
        "id": "adapt-tool-substitution",
        "when": "the resolution names a settings tool that TOOL_RENAMES "
                "(or a caller-supplied mapping) lists under an old name",
        "action": "replace each old tool name with its current name",
        "rationale": "Aamodt & Plaza 1994 (the REUSE step): a retrieved "
                     "solution is adapted to the current problem's "
                     "context, never copied blindly; tool renames are "
                     "exactly the surface drift adaptation exists to "
                     "repair, but only through verified mappings",
    },
    {
        "id": "adapt-drop-stale-step",
        "when": "the CALLER asserts a premise no longer holds "
                "(stale_premises) and a resolution step cites that premise",
        "action": "drop the step whose premise is asserted stale, and "
                  "record the dropped line and the premise that killed it",
        "rationale": "Aamodt & Plaza 1994: adaptation removes parts of "
                     "the old solution whose justifying problem features "
                     "are gone; the premise assertion is the caller's "
                     "to make, never this module's to guess",
    },
    {
        "id": "adapt-numeric-rescale",
        "when": "the CALLER supplies a baseline shift "
                "numeric_baseline=(was, now) — the assertion that the "
                "resolution's bare numeric literals were measured "
                "against `was` and must read against `now`",
        "action": "scale every bare numeric literal by the explicit "
                  "ratio now/was, recording each replacement",
        "rationale": "parameter adjustment, the simplest of the "
                     "adaptation methods Aamodt & Plaza 1994 survey "
                     "under methodological variations; the ratio is the "
                     "caller's assertion, every replacement is listed, "
                     "and nothing is rounded away silently",
    },
)

_RULES_BY_ID = {entry["id"]: entry for entry in ADAPTATION_RULES}


# ---------------------------------------------------------------------------
# Case and case-base validation
# ---------------------------------------------------------------------------

def _validate_case(case: Any, where: str = "case") -> Dict[str, Any]:
    """Reject malformed cases with ValueError (never clamp, never guess)."""
    if not isinstance(case, dict):
        raise ValueError(f"{where}: a case must be a dict, got {type(case).__name__}")
    for key in _REQUIRED_CASE_KEYS:
        if key not in case:
            raise ValueError(f"{where}: case missing required key {key!r}")
    for key in ("id", "problem", "resolution", "source"):
        if not isinstance(case[key], str) or not case[key].strip():
            raise ValueError(f"{where}: case key {key!r} must be a non-empty string")
    if not isinstance(case["accepted"], bool):
        raise ValueError(f"{where}: case key 'accepted' must be a bool")
    return case


def _validate_case_base(case_base: Any) -> List[Dict[str, Any]]:
    if not isinstance(case_base, (list, tuple)):
        raise ValueError(f"case base must be a list of cases, got {type(case_base).__name__}")
    seen: Dict[str, Dict[str, Any]] = {}
    for position, case in enumerate(case_base):
        _validate_case(case, where=f"case base[{position}]")
        if case["id"] in seen:
            raise ValueError(
                f"duplicate case id {case['id']!r} — duplicate cases are "
                "rejected, never silently absorbed")
        seen[case["id"]] = case
    return list(case_base)


def _validate_renames(renames: Any) -> Dict[str, str]:
    """Validate an old->new tool mapping (shipped or caller-supplied)."""
    if not isinstance(renames, dict):
        raise ValueError("tool renames must be a dict of {old name: new name}")
    for old, new in renames.items():
        if not isinstance(old, str) or not old or not isinstance(new, str) or not new:
            raise ValueError(f"tool rename {old!r} -> {new!r}: both sides must be non-empty strings")
        if old == new:
            raise ValueError(f"tool rename {old!r} -> itself is a no-op; remove it")
    return dict(renames)


# ---------------------------------------------------------------------------
# RETRIEVE — BM25 over case problem texts, via the shipped Searcher
# ---------------------------------------------------------------------------

def _case_index(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """An in-memory index dict in exactly the shape indexer.build_index()
    produces, so the SHIPPED Searcher (search.py) does the BM25 ranking —
    same tokenizer (indexer.tokenize), same k1/b constants, same scoring
    formula; nothing about BM25 is reimplemented here."""
    docs: List[Dict[str, Any]] = []
    postings: Dict[str, Dict[str, int]] = {}
    df: Dict[str, int] = {}
    total_tokens = 0
    for doc_index, case in enumerate(cases):
        stream = tokenize(case["problem"])
        tf = Counter(stream)
        for term, count in tf.items():
            postings.setdefault(term, {})[doc_index] = count
        for term in tf:
            df[term] = df.get(term, 0) + 1
        total_tokens += len(stream)
        docs.append({
            "id": case["id"],
            "title": case["problem"].split("\n")[0][:80],
            "source": case["source"],
            "dl": len(stream),
            "sentences": split_sentences(case["problem"]),
        })
    n_docs = len(cases)
    return {
        "version": VERSION,
        "params": {"k1": K1, "b": B},
        "n_docs": n_docs,
        "avgdl": (total_tokens / n_docs) if n_docs else 0.0,
        "docs": docs,
        "df": df,
        "postings": {term: {str(i): count for i, count in pairs.items()}
                     for term, pairs in postings.items()},
    }


def retrieve(case_base: Sequence[Dict[str, Any]], query: str, k: int = 3) -> List[Dict[str, Any]]:
    """RETRIEVE: BM25-rank the case base by problem-text overlap.

    Returns up to k hits, each {"case_id", "score", "case", "snippet"},
    in the Searcher's deterministic order (score desc, then case id).
    An empty case base is an honest empty answer ([]) — no case is
    invented; malformed cases or duplicate ids raise ValueError.
    """
    cases = _validate_case_base(case_base)
    if not cases:
        return []
    hits = Searcher(_case_index(cases)).search(query, k=k)
    by_id = {case["id"]: case for case in cases}
    return [{"case_id": hit["doc_id"],
             "score": hit["score"],
             "case": by_id[hit["doc_id"]],
             "snippet": hit["snippet"]} for hit in hits]


def _default_tool_registry():
    """The real tool registry: the generated settings tools.json (issue
    #120), resolved through the settings layer's own loader — the same
    artifact diagnostics/settings_join.py resolves tool mentions
    through. Lazy import so this module itself stays I/O-free."""
    from assistant.adapters.caelestia.registry import tool_by_name

    def _registered(name: str) -> bool:
        return tool_by_name(name) is not None

    return _registered


def _fmt_number(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


# ---------------------------------------------------------------------------
# REUSE — the adaptation-rule table
# ---------------------------------------------------------------------------

def reuse(case: Dict[str, Any], query: str, *,
          stale_premises: Sequence[str] = (),
          numeric_baseline: Optional[Sequence[float]] = None,
          tool_renames: Optional[Dict[str, str]] = None,
          tool_registry=None) -> Dict[str, Any]:
    """REUSE: adapt one retrieved case's resolution for the query.

    The caller-asserted adaptation context (all optional):

    - stale_premises: substrings asserting premises that no longer hold;
      a resolution line containing one is DROPPED (rule
      adapt-drop-stale-step), with the line and premise recorded.
    - numeric_baseline: a (was, now) pair asserting that the
      resolution's bare numeric literals were measured against `was`;
      each is scaled by now/was (rule adapt-numeric-rescale), every
      replacement recorded. Deliberately literal: EVERY bare numeric
      literal scales (a "GCC 10" version or an issue number scales
      too) — which is exactly why every replacement is listed for the
      human to reject, and why the rule only fires when the caller
      explicitly asserts the baseline.
    - tool_renames: {old: new} tool-name substitutions beyond the
      shipped (empty) TOOL_RENAMES table (rule
      adapt-tool-substitution).
    - tool_registry: callable name -> bool deciding whether a settings
      tool name is registered. Default: the real settings registry
      (tools.json). A resolution naming an UNREGISTERED tool that no
      rename repairs makes the whole case untransferable -> ABSTAINED.

    The query itself is carried through into the result for provenance;
    adaptation pressure is NEVER guessed from raw query text — only the
    caller-asserted context above adapts anything (each entry cites its
    rule id; unfired rules are not claimed).

    Returns {"status": "ADAPTED"|"VERBATIM"|"ABSTAINED", "case_id",
    "resolution", "rules_fired", "abstain_reason", "note"}.
    """
    _validate_case(case, where=f"case {case.get('id', '<no id>')!r}")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    renames = _validate_renames(
        dict(TOOL_RENAMES) if tool_renames is None else tool_renames)
    if not isinstance(stale_premises, (list, tuple)):
        raise ValueError("stale_premises must be a sequence of substrings")
    for premise in stale_premises:
        if not isinstance(premise, str) or not premise:
            raise ValueError("each stale premise must be a non-empty substring")
    ratio: Optional[float] = None
    if numeric_baseline is not None:
        if (not isinstance(numeric_baseline, (list, tuple)) or len(numeric_baseline) != 2):
            raise ValueError("numeric_baseline must be a (was, now) pair of numbers")
        was, now = numeric_baseline
        for value in (was, now):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("numeric_baseline entries must be numbers")
        if float(was) == 0.0:
            raise ValueError("numeric_baseline 'was' must be non-zero (no ratio is defined)")
        ratio = float(now) / float(was)

    lines = case["resolution"].split("\n")

    # --- rule adapt-tool-substitution: rename old tool names ----------
    substituted: List[str] = []
    per_line_changes: List[Dict[str, Any]] = []
    for lineno, line in enumerate(lines):
        line_changes: List[Dict[str, Any]] = []

        def _substitute(match: "re.Match[str]") -> str:
            old = match.group(0)
            new = renames.get(old)
            if new is None:
                return old
            line_changes.append({"before": old, "after": new})
            return new

        substituted.append(_TOOL_MENTION_RE.sub(_substitute, line))
        for change in line_changes:
            per_line_changes.append({"line": lineno, **change})
    fired: List[Dict[str, Any]] = []
    if per_line_changes:
        fired.append(_fired("adapt-tool-substitution", per_line_changes))

    # --- transferability: no unregistered, unrepaired tool names ------
    mentions = _TOOL_MENTION_RE.findall("\n".join(substituted))
    check = tool_registry if tool_registry is not None else _default_tool_registry()
    if not callable(check):
        raise ValueError("tool_registry must be a callable name -> bool")
    unregistered = sorted({name for name in mentions if not check(name)})
    if unregistered:
        return {
            "status": "ABSTAINED",
            "case_id": case["id"],
            "accepted": case["accepted"],
            "resolution": None,
            "rules_fired": [],
            "abstain_reason": (
                "the resolution references settings tool(s) absent from the "
                f"registry ({', '.join(unregistered)}) and no substitution "
                "rule maps them onto a registered name — the fix cannot "
                "transfer honestly, so nothing is offered"),
            "note": None,
        }

    # --- rule adapt-drop-stale-step: drop stale-premise steps ----------
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for lineno, line in enumerate(substituted):
        premise = next((p for p in stale_premises if p in line), None)
        if premise is not None:
            dropped.append({"line": lineno, "premise": premise, "text": line})
        else:
            kept.append(line)
    if dropped:
        fired.append(_fired("adapt-drop-stale-step", dropped))
    if not any(line.strip() for line in kept):
        return {
            "status": "ABSTAINED",
            "case_id": case["id"],
            "accepted": case["accepted"],
            "resolution": None,
            "rules_fired": fired,
            "abstain_reason": (
                "every resolution step rests on a premise the caller "
                "asserts is stale — nothing of the case transfers"),
            "note": None,
        }

    # --- rule adapt-numeric-rescale: scale to the current baseline -----
    if ratio is not None:
        scaled: List[str] = []
        rescales: List[Dict[str, Any]] = []
        for line in kept:
            line_changes: List[Dict[str, Any]] = []

            def _scale(match: "re.Match[str]") -> str:
                before = match.group(0)
                after = _fmt_number(float(before) * ratio)
                if after == before:
                    return before  # a no-op replacement is not an adaptation
                line_changes.append({"before": before, "after": after})
                return after

            scaled.append(_NUMBER_RE.sub(_scale, line))
            for change in line_changes:
                rescales.append(change)
        if rescales:
            fired.append(_fired("adapt-numeric-rescale", rescales,
                                extra={"ratio": round(ratio, 6)}))
            kept = scaled

    return {
        "status": "ADAPTED" if fired else "VERBATIM",
        "case_id": case["id"],
        "accepted": case["accepted"],
        "resolution": "\n".join(kept),
        "rules_fired": fired,
        "abstain_reason": None,
        "note": (None if case["accepted"] else
                 "the recorded outcome was NOT accepted — the resolution is "
                 "offered as prior art, not as a verified fix"),
    }


def _fired(rule_id: str, changes: List[Dict[str, Any]],
           extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One fired adaptation, self-describing: the rule table's own
    when/action/rationale travel with the change list."""
    entry = dict(_RULES_BY_ID[rule_id])
    entry["changes"] = changes
    if extra:
        entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# REVISE — the caller's validator marks failing steps
# ---------------------------------------------------------------------------

def revise(adapted: Dict[str, Any], validate_fn) -> Dict[str, Any]:
    """REVISE: run the caller's validator over the adapted resolution's
    steps (its non-empty lines) and mark the failures.

    validate_fn(step_text) -> (passed: bool, reason: str | None) — the
    settings planner's validation is the real validator in production;
    tests inject a fake. Failing steps are MARKED with the validator's
    own reason; this module never applies, repairs, or drops anything
    (applying is the user's call through the surfaces that already
    exist).

    Returns {"status": "CONFIRMED"|"REVISE", "steps": [...], "failed_steps",
    "applied": False}; an abstained reuse result has nothing to revise
    and is rejected with ValueError.
    """
    if (not isinstance(adapted, dict)
            or adapted.get("status") not in ("ADAPTED", "VERBATIM")
            or not isinstance(adapted.get("resolution"), str)):
        raise ValueError("revise needs a reuse result with a resolution "
                         "(status ADAPTED or VERBATIM); an abstained reuse "
                         "has nothing to revise")
    if not callable(validate_fn):
        raise ValueError("validate_fn must be callable: step text -> "
                         "(passed, reason)")
    steps = [line.strip() for line in adapted["resolution"].split("\n") if line.strip()]
    if not steps:
        raise ValueError("the adapted resolution has no non-empty steps to validate")
    out_steps: List[Dict[str, Any]] = []
    failed: List[int] = []
    for index, step in enumerate(steps, start=1):
        verdict = validate_fn(step)
        if (not isinstance(verdict, tuple) or len(verdict) != 2
                or not isinstance(verdict[0], bool)
                or not (verdict[1] is None or isinstance(verdict[1], str))):
            raise ValueError("validate_fn must return (passed: bool, "
                             f"reason: str | None), got {verdict!r}")
        passed, reason = verdict
        out_steps.append({"i": index, "text": step, "passed": passed,
                          "reason": None if passed else reason})
        if not passed:
            failed.append(index)
    return {
        "case_id": adapted.get("case_id"),
        "status": "CONFIRMED" if not failed else "REVISE",
        "n_steps": len(steps),
        "steps": out_steps,
        "failed_steps": failed,
        "applied": False,  # this module never applies anything
    }


# ---------------------------------------------------------------------------
# RETAIN — a ledger proposal, never a silent append
# ---------------------------------------------------------------------------

def retain(case_base: Sequence[Dict[str, Any]], new_case: Dict[str, Any],
           ledger, confidence: Optional[float] = None) -> Dict[str, Any]:
    """RETAIN: file the new case as a ledger proposal (kind="cbr_case").

    The proposal's diff is the CANONICAL case text (json.dumps with
    sort_keys). Nothing is appended to the case base here — the case
    base is caller-owned, and the ledger's approve/reject flow is the
    caller's only signal to append (the same doctrine as every other
    brain proposal). Confidence defaults to fixed conservative numbers
    (0.8 user-confirmed outcome, 0.3 otherwise — documented constants,
    not calibrated values) and may be overridden; out-of-range values
    are rejected, never clamped.
    """
    cases = _validate_case_base(case_base)
    _validate_case(new_case, where="new case")
    if any(case["id"] == new_case["id"] for case in cases):
        raise ValueError(f"case id {new_case['id']!r} is already in the case "
                         "base — duplicate cases are rejected, never silently absorbed")
    propose = getattr(ledger, "propose", None)
    if not callable(propose):
        raise ValueError("retain needs a brain.ledger.Ledger (an object with "
                         ".propose) — the retain step is a ledger proposal, "
                         "never a silent append")
    if confidence is None:
        confidence = _ACCEPTED_CONFIDENCE if new_case["accepted"] else _REJECTED_CONFIDENCE
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number in [0, 1]")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence {confidence!r} is out of range [0, 1] — "
                         "rejected, never clamped")
    canonical = json.dumps(new_case, sort_keys=True, ensure_ascii=False)
    pid = propose(kind="cbr_case", target=new_case["id"], diff=canonical,
                  reason=("CBR RETAIN: file the resolved case for future "
                          "retrieval (Aamodt & Plaza 1994); the case base is "
                          "caller-owned — approving this proposal is the "
                          "caller's signal to append the case"),
                  confidence=float(confidence))
    return {
        "status": "PROPOSED",
        "proposal_id": pid,
        "kind": "cbr_case",
        "target": new_case["id"],
        "confidence": float(confidence),
        "appended": False,
        "note": ("nothing was appended to the case base; the ledger's "
                 "approve/reject flow is the only signal"),
    }


# ---------------------------------------------------------------------------
# The whole cycle as one call
# ---------------------------------------------------------------------------

def cbr_cycle(case_base: Sequence[Dict[str, Any]], query: str, k: int = 1,
              validate_fn=None, *, stale_premises: Sequence[str] = (),
              numeric_baseline: Optional[Sequence[float]] = None,
              tool_renames: Optional[Dict[str, str]] = None,
              tool_registry=None, new_case: Optional[Dict[str, Any]] = None,
              ledger=None) -> Dict[str, Any]:
    """RETRIEVE -> REUSE -> REVISE (-> RETAIN when a new case is given).

    Only the BEST retrieved case is reused (k defaults to 1); k=0
    retrieved cases is an honest ABSTENTION — no case is invented. The
    revise stage runs only when a validator is supplied (the settings
    planner's validation is the real one); the retain stage runs only
    when BOTH a new case and a ledger are supplied (retain is a ledger
    proposal, never a silent append). Every stage's output is in the
    returned dict under its own key; skipped stages say why.
    """
    cases = _validate_case_base(case_base)
    hits = retrieve(cases, query, k=k)
    if not hits:
        return {
            "query": query,
            "status": "ABSTAINED",
            "abstain_reason": ("no case in the base matches the query "
                               "(0 retrieved) — no case is invented"),
            "retrieve": [],
            "reuse": None,
            "revise": None,
            "retain": None,
        }
    best = hits[0]
    adapted = reuse(best["case"], query, stale_premises=stale_premises,
                    numeric_baseline=numeric_baseline,
                    tool_renames=tool_renames, tool_registry=tool_registry)
    out: Dict[str, Any] = {
        "query": query,
        "status": adapted["status"],
        "abstain_reason": adapted["abstain_reason"],
        "retrieve": hits,
        "case_id": best["case_id"],
        "reuse": adapted,
    }
    if adapted["status"] == "ABSTAINED":
        out["revise"] = None
        out["retain"] = None
        return out
    if validate_fn is not None:
        out["revise"] = revise(adapted, validate_fn)
    else:
        out["revise"] = {"status": "NOT_RUN",
                         "reason": ("no validator supplied — revision is the "
                                    "caller's step (the settings planner's "
                                    "validation is the real one)")}
    if new_case is not None or ledger is not None:
        if new_case is None or ledger is None:
            raise ValueError("retain needs BOTH a new case and a ledger")
        out["retain"] = retain(cases, new_case, ledger)
    else:
        out["retain"] = {"status": "NOT_RUN",
                         "reason": ("retain files a ledger proposal for a NEW "
                                    "case once the outcome is known; supply "
                                    "new_case + ledger to run it — nothing is "
                                    "invented here")}
    return out

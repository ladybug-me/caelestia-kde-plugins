"""cortex.dispatch — the unified local-vs-cloud decision point.

One dispatcher, two backends, one existing confidence gate deciding between
them. The decision to serve a request from the local ontology or hand it
off to the (opt-in, user-keyed) cloud sidebar lives HERE — in the cortex
layer's own machinery — never in the sidebar's QML. The sidebar asks the
bridge (op ``dispatch``); this module answers.

THE HAND-OFF RULE (existing gates only — no second threshold system):

- verdict ``ABSTAIN``, or ``QUESTION`` with no candidates at all (nothing
  scored: the router's own min_score gate in effect)  -> cloud,
  reason ``router-abstain``;
- verdict ``DELEGATE`` to a surface WITHOUT an inline runner -> cloud,
  reason ``delegate:<surface>``. A DELEGATE surface WITH an inline
  runner (genius / diagnose / search / brain / issue / agent — the
  routing fix's generalized inline execution, ``cortex/delegate.py``)
  answers LOCALLY: the target layer runs read-only in the same turn
  (the agent always in --simulate) and its answer becomes the sidebar
  bubble. Only a FAILED inline run hands off, reason
  ``delegate:<surface>-inline-failed``;
- verdict ``PLAN`` whose top-route score is NOT covered by the conformal
  calibrator's verdict, when calibration data exists (>= 5 accepted
  routes) -> cloud, reason ``below-conformal-threshold``. No calibration
  data means no distribution-free claim either way — the request stays
  local rather than being punted on a guess;
- everything else stays LOCAL: PLAN (validated ops), QUESTION with
  candidates (the local layer owns the clarification, with its session
  machinery), EXPLAIN / UNDO / LIST / INERT, and runnable DELEGATEs.

GAP LOGGING (every hand-off, §5.2): the state's ``cortex_gaps`` bucket —
the same bounded-bucket discipline as ``cortex_review`` (dedup by shape,
newest-first, bounded) — records the query SHAPE (stemmed, stopword-
filtered, distinctive-token signature) and the intent category, NOT the
raw text: enough to cluster, count and describe the gap later ("N requests
like X fell through"), with no reason to retain user prose. Near-threshold
hand-offs also feed the existing ``cortex_review`` batch surface.

GAP CLUSTERING (§5.3): ``cluster_gaps`` reuses genius' k-means (the same
deterministic implementation workspace.py uses) over the gap vectors, and
applies the SAME minimum-support (3) + cluster-purity (0.6) floor
workspace.py established — a handful of scattered gaps produces ZERO
candidates, by design and by test.

SURFACING (§5.4): ``propose_gap_clusters`` turns a qualifying cluster into
a LEDGER PROPOSAL (kind ``ontology_gap``) — never an automatically-added
tool. The ledger's approve/reject is the whole interaction, exactly like
every other proposal in the system.

TOOL-TEMPLATE STUBS (§5.5, exponential-build 1.1): a dense, UNADDRESSED
gap cluster can additionally be drafted into a reviewable stub file
(name, cue words, a TODO body) by ``select_stub_candidates`` + the CLI's
explicit ``--draft-stubs`` flag — a candidate scaffold for a future
local tool that is never registered in the router, never wired into
this dispatcher, never executed, and never clobbered once a human has
edited it. The functions here stay PURE (no I/O): the CLI owns writing
the files, exactly as it owns writing state.

Read-only except the state buckets and the ledger proposals it is given;
this module executes nothing and writes no files itself.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .learn import REVIEW_KEY, CortexLearner, log_review_candidate
from .delegate import run_delegate, runner_names
from .pipeline import process
from .session import SessionState
from .vectorize import tokenize

__all__ = ["GAPS_KEY", "MAX_GAPS", "query_shape", "log_gap", "dispatch",
           "cluster_gaps", "propose_gap_cluster", "propose_gap_clusters",
           "tool_template_slug", "tool_template_stub_text",
           "select_stub_candidates"]

# The gap bucket: same discipline as cortex_review (bounded, newest-first,
# deduped by shape). Gaps are lower-frequency events than review candidates,
# so the bound is larger, but it stays a bound.
GAPS_KEY = "cortex_gaps"
MAX_GAPS = 120

# How many distinctive tokens make up a gap's shape (enough to describe and
# cluster; deliberately not the full request).
SHAPE_TOKENS = 8


# ---------------------------------------------------------------------------
# Query shape (the privacy-preserving signature of a request).
# ---------------------------------------------------------------------------

def query_shape(text: str) -> List[str]:
    """Distinctive-token signature: stemmed, stopword-filtered tokens,
    longest-first (content words beat function words), capped at
    SHAPE_TOKENS, deduplicated. Two phrasings of the same request shape
    collide far more often than two unrelated ones."""
    tokens = tokenize(text)
    unique = sorted(set(tokens), key=lambda w: (-len(w), w))
    return unique[:SHAPE_TOKENS]


def log_gap(state: Dict[str, Any], text: str, category: str,
            now: str) -> Dict[str, Any]:
    """Record one hand-off in the bounded gap bucket (mutates ``state``;
    the caller owns persistence). Same-shape requests are counted, not
    re-stored; the newest categorization wins. Returns the updated entry."""
    shape = query_shape(text) or ["(unparsed)"]
    bucket = [g for g in (state.get(GAPS_KEY) or [])
              if isinstance(g, dict)]
    entry: Optional[Dict[str, Any]] = None
    for existing in bucket:
        if existing.get("shape") == shape:
            entry = existing
            bucket.remove(existing)
            break
    if entry is not None:
        entry["n"] = int(entry.get("n", 1)) + 1
        entry["category"] = category
        entry["last_at"] = now
    else:
        entry = {"shape": shape, "category": category, "n": 1,
                 "first_at": now, "last_at": now}
    bucket.insert(0, entry)
    state[GAPS_KEY] = bucket[:MAX_GAPS]
    return entry


# ---------------------------------------------------------------------------
# The hand-off decision.
# ---------------------------------------------------------------------------

def _hand_off_reason(result, state: Dict[str, Any]) -> Optional[str]:
    """Why this request must go to the cloud tier, or None if it stays
    local. Uses the router's own verdicts and the conformal calibrator's
    own verdict only — no new thresholds are introduced anywhere."""
    if result.verdict == "ABSTAIN":
        return "router-abstain"
    if result.verdict == "OUT_OF_ONTOLOGY":
        # F6: the request names nothing in the local ontology — the
        # cloud tier is exactly the escape hatch for this, and the gap
        # bucket records the shape for later clustering.
        return "out-of-ontology"
    if result.verdict == "DELEGATE":
        return f"delegate:{result.delegate or 'unknown'}"
    if result.verdict == "QUESTION" and not result.candidates:
        # every clause came back "nothing matched" — the router's min_score
        # gate in effect; the local ontology has nothing to say.
        return "router-abstain"
    if result.verdict == "PLAN" and result.plan and not result.plan.get("apply_blocked"):
        from .conformal import ConformalCalibrator
        data = state.get("conformal")
        if isinstance(data, dict) and data.get("scores"):
            cal = ConformalCalibrator()
            cal.from_dict(data)
            verdict = cal.verdict(result.confidence)
            if verdict.get("covered") is False:
                return "below-conformal-threshold"
    return None


def render_answer(result) -> List[str]:
    """The local tier's chat answer (sidebar-flavoured rendering; the CLI
    keeps its own chat card in cli.py — same result object, different
    surface, no shared presentation code to drift)."""
    lines: List[str] = []
    if result.session_note:
        lines.append(result.session_note)
    if result.verdict == "PLAN" and result.plan is not None:
        entries = result.plan.get("entries", [])
        if result.plan.get("apply_blocked"):
            lines.append("I parsed this, but the plan has validation errors:")
            for err in result.plan.get("errors", [])[:5]:
                lines.append(f"  {err.get('tool', '?')}: {err.get('error', '?')}")
        else:
            applicable = [e for e in entries if e.get("new") is not None]
            if len(applicable) == 1:
                e = applicable[0]
                lines.append(f"Applied: {e.get('path', '?')} "
                             f"{e.get('old', '(unset)')} -> {e.get('new')}.")
                lines.append("Undo is available — just say 'undo the last change'.")
            else:
                lines.append("I found several changes that match your request — "
                             "approve them above to apply.")
                for e in applicable[:8]:
                    lines.append(f"  {e.get('path', '?')}: {e.get('old', '(unset)')} "
                                 f"-> {e.get('new')}")
    elif result.verdict == "QUESTION":
        for q in result.questions[:4]:
            lines.append(q)
        # an apply-blocked plan's validation errors ARE the answer (the
        # issue's own rule: out-of-range simply isn't applied)
        if result.plan and result.plan.get("errors"):
            for err in result.plan.get("errors", [])[:5]:
                lines.append(f"  {err.get('tool', '?')}: {err.get('error', '?')}")
    elif result.verdict == "EXPLAIN" and result.explain_answer:
        lines.append(str(result.explain_answer.get("answer", "")))
        cites = result.explain_answer.get("cites") or []
        if cites:
            lines.append(f"grounded in: {'; '.join(str(c) for c in cites[:3])}")
    elif result.verdict in ("UNDO", "LIST") and result.history_plan:
        hp = result.history_plan
        if hp.get("action"):
            lines.append(f"Undoing: {hp.get('reason', '')}")
        else:
            lines.append(f"{hp.get('reason', '')}")
        for entry in hp.get("entries", [])[:5]:
            label = entry.get("label") or f"#{entry.get('id')}"
            lines.append(f"  [{entry.get('id')}] {label} at {entry.get('at')}")
    elif result.verdict == "INERT":
        lines.extend(result.suggestions)
    if getattr(result, "calibration_note", None):
        lines.append(str(result.calibration_note))
    if result.evidence:
        shown = "; ".join(dict.fromkeys(result.evidence))[:220]
        lines.append(f"(why: {shown})")
    for note in result.notes[:3]:
        if note:
            lines.append(f"note: {note}")
    if not lines:
        lines.append("(no local answer for this verdict — say more and I'll try again)")
    return lines


def apply_calls(result) -> Optional[Dict[str, Any]]:
    """SettingsTools-shaped apply payload for a local PLAN: the validated
    entries as [{name, value}] plus a label. The sidebar applies them
    through the EXISTING SettingsTools gate (single entry applies
    immediately and is undoable; multiple entries show the preview card);
    this function only extracts data, it writes nothing. None when the plan
    is blocked or the verdict is not PLAN."""
    if result.verdict != "PLAN" or result.plan is None:
        return None
    if result.plan.get("apply_blocked"):
        return None
    entries = result.plan.get("entries", [])
    calls = [{"name": e.get("tool"), "value": e.get("new")}
             for e in entries if e.get("new") is not None]
    if not calls:
        return None
    label = (result.candidates[0].get("clause") if result.candidates
             else "cortex dispatch")
    return {"calls": calls, "label": label}


def undo_action(result) -> Optional[Dict[str, Any]]:
    """The gated undo the sidebar should run for a local UNDO verdict
    (SettingsTools.undo / undoById — the existing surfaces). None when
    there is nothing to undo."""
    if result.verdict != "UNDO" or not result.history_plan:
        return None
    hp = result.history_plan
    if not hp.get("action"):
        return None
    if hp.get("entry_id") is not None:
        return {"id": hp.get("entry_id")}
    return {"steps": hp.get("steps") or 1}


def dispatch(text: str, *, state: Optional[Dict[str, Any]] = None,
             session: Optional[SessionState] = None,
             file_path: Optional[str] = None,
             now: Optional[str] = None) -> Dict[str, Any]:
    """Decide one request: local answer or cloud hand-off.

    ``state`` is the caller's brain-state dict; on a hand-off the gap
    bucket (and, for near-threshold verdicts, the review bucket) are
    updated in it — the caller persists. The returned outcome dict:

    ``{"action": "local"|"cloud", "reason": str|None, "confidence": float,
       "session": {...}, "result": {...},
       "answer": [lines]            # local only
       "apply": {calls, label}       # local PLAN only (gated by the caller)
       "undo": {steps}|{id}          # local UNDO only (gated by the caller)}``
    """
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = state if isinstance(state, dict) else {}
    session = session if session is not None else SessionState()
    learner = CortexLearner(state.get("cortex_learn")) \
        if state.get("cortex_learn") else None
    result = process(text, session=session, learner=learner, file_path=file_path)

    # Calibration surfacing (phase 2.4): the honest accuracy line for
    # THIS confidence bucket, when the sample supports it — the same
    # sentence the CLI chat card shows.
    if learner is not None:
        try:
            # raw route probability is the bucket key (learn_hook), not
            # the calibrated confidence — a different quantity
            raw_p = ((result.learn_hook or {}).get("p", result.confidence)
                     if result.learn_hook is not None else result.confidence)
            result.calibration_note = learner.calibration_note(raw_p)
        except Exception:
            result.calibration_note = None

    reason = _hand_off_reason(result, state)

    # Inline-runnable delegates answer LOCALLY (the routing fix's decision
    # point, kept in the cortex layer — never the sidebar's QML): genius /
    # diagnose / search / brain / issue / agent requests run their target
    # layer inline, read-only, and the sidebar shows that answer instead of
    # handing the request to the cloud tier. A failed inline run falls back
    # to the cloud tier with a distinct, countable reason.
    inline: Optional[tuple] = None
    if (reason and result.verdict == "DELEGATE"
            and result.delegate in runner_names()):
        inline = run_delegate(result.delegate, text, state)
        if inline is not None:
            reason = None
        else:
            reason = f"delegate:{result.delegate}-inline-failed"

    # Taught concepts (F15): when the cortex would hand this off, a
    # user-taught concept may own the phrase. Recall only PROPOSES — the
    # apply payload goes through the same gated surface as every plan,
    # and a built-in preset match asks instead of replacing.
    if reason:
        from assistant.core import concepts as _concepts
        from assistant.capabilities.settings.curations import PRESETS as _PRESETS
        try:
            hit = _concepts.recall(state, text, presets=_PRESETS)
        except Exception:
            hit = None
        if hit and hit.get("verdict") == "RECALL":
            lines = [
                f"taught concept '{hit['name']}' (similarity {hit['score']}): "
                f"propose {len(hit['calls'])} call(s)",
            ]
            for c in hit["calls"]:
                lines.append(f"  {c['name']} = {c['value']!r}")
            for p in hit["posterior"]:
                if p["p"] is not None:
                    lines.append(f"  ({p['tool']} accepted {p['p']} over "
                                 f"{p['n']} past uses)")
            return {
                "action": "local", "reason": None,
                "confidence": hit["score"],
                "session": session.to_dict(),
                "result": result.to_dict(),
                "answer": lines,
                "apply": {"calls": hit["calls"],
                          "label": f"taught concept: {hit['name']}"},
                "concept": hit,
            }
        if hit and hit.get("verdict") in ("CONFLICT_ASK", "AMBIGUOUS"):
            if hit["verdict"] == "CONFLICT_ASK":
                lines = [hit["question"],
                         f"  concept: {hit['concept']['name']} "
                         f"({hit['concept']['score']})",
                         f"  preset:  {hit['preset']['name']} "
                         f"({hit['preset']['score']})"]
            else:
                lines = ["several taught concepts match:"]
                for cand in hit["candidates"]:
                    lines.append(f"  {cand['name']} ({cand['score']})")
            return {
                "action": "local", "reason": None,
                "confidence": 0.5, "session": session.to_dict(),
                "result": result.to_dict(), "answer": lines,
                "concept": hit,
            }

    outcome: Dict[str, Any] = {
        "action": "cloud" if reason else "local",
        "reason": reason,
        "confidence": result.confidence,
        "session": session.to_dict(),
        "result": result.to_dict(),
    }
    if reason:
        log_gap(state, text, reason, now)
        if result.verdict in ("ABSTAIN", "QUESTION", "OUT_OF_ONTOLOGY"):
            # the same near-threshold phrases the CLI chat loop parks for
            # batch review — sidebar traffic feeds the same surface.
            # OUT_OF_ONTOLOGY phrases are exactly the review material for
            # "should a local capability cover this?" (F6).
            state[REVIEW_KEY] = log_review_candidate(
                list(state.get(REVIEW_KEY, [])), text, result.verdict,
                result.candidates or [], at=now)
        return outcome

    if inline is not None:
        payload, inline_lines = inline
        answer = [line.strip() for line in inline_lines if line.strip()]
        outcome["answer"] = answer or render_answer(result)
        outcome["delegate_payload"] = payload
    else:
        outcome["answer"] = render_answer(result)
    apply_info = apply_calls(result)
    if apply_info is not None:
        outcome["apply"] = apply_info
    undo = undo_action(result)
    if undo is not None:
        outcome["undo"] = undo
    return outcome


# ---------------------------------------------------------------------------
# Gap clustering (the workspace.py discipline, applied to gap vectors).
# ---------------------------------------------------------------------------

def cluster_gaps(state: Dict[str, Any], *, k: Optional[int] = None,
                 min_support: int = 3, purity: float = 0.6,
                 seed: int = 42) -> Dict[str, Any]:
    """Cluster the logged gap shapes; only consistent clusters surface.

    Vectors are token-count bags over the gaps' own vocabulary; clustering
    reuses genius' deterministic k-means (k-means++ seeding under a fixed
    seed — the same implementation workspace.py uses, lazily imported to
    keep the layer boundary explicit). A cluster is a candidate only when
    its total request count (sum of the member shapes' ``n``) reaches
    ``min_support`` AND its modal shape covers at least ``purity`` of
    those requests — the exact floor workspace.py established for session
    profiles, because a handful of scattered gaps is noise, not a need."""
    gaps = [g for g in (state.get(GAPS_KEY) or []) if isinstance(g, dict)
            and g.get("shape")]
    n = len(gaps)
    if n == 0:
        return {"n_gaps": 0, "k": 0, "candidates": [], "rejected_clusters": 0}

    vocab: Dict[str, int] = {}
    for gap in gaps:
        for token in gap["shape"]:
            vocab.setdefault(token, len(vocab))
    points = [[0.0] * len(vocab) for _ in gaps]
    for i, gap in enumerate(gaps):
        for token in gap["shape"]:
            points[i][vocab[token]] += 1.0

    if k is None:
        k = min(8, max(2, round(math.sqrt(n / 2))))
    k = max(1, min(k, n))
    if n >= 2 and k >= 2:
        from assistant.capabilities.genius.data import kmeans
        labels = kmeans(points, k, seed=seed)["labels"]
    else:
        labels = [0] * n  # one cluster; the support/purity floor still applies

    candidates: List[Dict[str, Any]] = []
    rejected = 0
    for cid in sorted(set(labels)):
        members = [g for g, label in zip(gaps, labels) if label == cid]
        support = sum(int(m.get("n", 1)) for m in members)
        if support < min_support:
            rejected += 1
            continue
        modal = max(members, key=lambda m: (int(m.get("n", 1)),
                                            " ".join(m.get("shape", []))))
        p = int(modal.get("n", 1)) / support
        if p < purity:
            rejected += 1
            continue
        candidates.append({
            "tokens": modal["shape"],
            "category": modal.get("category"),
            "support": support,
            "purity": round(p, 3),
            "shapes": len(members),
            "label": " ".join(modal["shape"][:5]),
        })
    candidates.sort(key=lambda c: (-c["support"], c["label"]))
    return {"n_gaps": n, "k": k, "candidates": candidates,
            "rejected_clusters": rejected}


def _propose_candidate(candidate: Dict[str, Any], ledger) -> Optional[int]:
    """File ONE qualifying cluster as a ledger proposal (kind
    ``ontology_gap``), or None when a pending duplicate exists. The
    single proposal step both surfacing functions share."""
    target = f"gap-cluster:{candidate['label']}"
    if target in {p.get("target") for p in ledger.pending()}:
        return None
    why = (f"{candidate['support']} request(s) shaped like "
           f"\"{candidate['label']}\" fell through to the cloud tier "
           f"(category {candidate['category']}, purity "
           f"{candidate['purity']}, {candidate['shapes']} distinct "
           f"phrasing(s)) — want a local tool for this?")
    return ledger.propose("ontology_gap", target,
                          {"cluster": candidate}, why,
                          candidate["purity"])


def propose_gap_cluster(state: Dict[str, Any], ledger, label: str,
                        **kwargs: Any) -> Dict[str, Any]:
    """Surface ONE gap cluster (by its modal-shape label) as a ledger
    proposal — the single-cluster entry point the unified inbox
    (cortex/inbox.py) dispatches ``approve`` through. Same floors, same
    proposal shape, same never-auto-absorb rule as
    ``propose_gap_clusters``; a cluster already pending is skipped, not
    stacked. Returns ``{"proposed": pid|None, "candidate": dict|None}`` —
    ``candidate`` None means no qualifying cluster carries that label
    right now."""
    summary = cluster_gaps(state, **kwargs)
    for candidate in summary["candidates"]:
        if candidate["label"] == label:
            pid = _propose_candidate(candidate, ledger)
            return {"proposed": pid, "candidate": candidate}
    return {"proposed": None, "candidate": None}


def propose_gap_clusters(state: Dict[str, Any], ledger,
                         **kwargs: Any) -> Dict[str, Any]:
    """Turn qualifying gap clusters into LEDGER PROPOSALS (kind
    ``ontology_gap``). Never auto-absorbs anything: the ledger's
    approve/reject flow is the whole interaction, same as every other
    proposal. One live proposal per cluster target (a pending duplicate is
    skipped, not stacked)."""
    summary = cluster_gaps(state, **kwargs)
    pids: List[int] = []
    for candidate in summary["candidates"]:
        pid = _propose_candidate(candidate, ledger)
        if pid is not None:
            pids.append(pid)
    return {"proposals": pids, "summary": summary}


# ---------------------------------------------------------------------------
# Tool-template stub drafting (exponential-build 1.1). Pure helpers only:
# the CLI's explicit --draft-stubs flag turns the selected candidates into
# reviewable .md files; nothing here registers, wires, or executes anything.
# ---------------------------------------------------------------------------

def tool_template_slug(label: str) -> str:
    """A filesystem-safe slug for a cluster label: lowercase, non-
    alphanumerics collapsed to underscores, trimmed. Deterministic (a
    pure function of the label, which is itself the cluster's modal
    shape), so the same cluster always drafts the same stub filename."""
    stem = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return stem or "unnamed_cluster"


def tool_template_stub_text(candidate: Dict[str, Any],
                            drafted_at: str) -> str:
    """The full markdown text of one tool-template stub (pure). The stub
    names itself DRAFT, cites the evidence that produced it, lists the
    cue words, and leaves a TODO body — it becomes a tool only if a
    human writes the named classical algorithm behind it (author/year
    citation required, the same discipline every existing module
    follows). Until then the requests keep falling through honestly."""
    cues = ", ".join(candidate["tokens"][:SHAPE_TOKENS])
    name = tool_template_slug(candidate["label"])
    return "\n".join([
        f"# tool-template: {name}",
        "",
        f"status: DRAFT — auto-drafted from a cortex gap cluster on "
        f"{drafted_at};",
        "  NOT registered in the router, NOT wired into the dispatcher,",
        "  NOT executable by anything. It becomes a tool only if a human",
        "  writes the named classical algorithm behind it (author, year",
        "  citation required, same discipline as every existing module).",
        "",
        "## Evidence (why this cluster qualifies)",
        "",
        f"- {candidate['support']} request(s) shaped like "
        f"\"{candidate['label']}\" fell through to the cloud tier",
        f"- category: {candidate['category']}; purity "
        f"{candidate['purity']}; {candidate['shapes']} distinct phrasing(s)",
        "",
        "## Proposed surface",
        "",
        f"- name: {name}",
        f"- cue words: {cues}",
        "",
        "## Body",
        "",
        "TODO: implement this tool as a named classical algorithm with a",
        "citation in its docstring (author, year). Until that module",
        "exists, these requests keep falling through honestly instead of",
        "being guessed at.",
        "",
    ])


def select_stub_candidates(state: Dict[str, Any], ledger=None,
                           **kwargs: Any) -> Dict[str, Any]:
    """Qualifying clusters that are still UNADDRESSED — the ledger holds
    NO item at all (pending, approved, or rejected) for the cluster's
    ``gap-cluster:<label>`` target. Once the ledger has the cluster, the
    ledger flow IS the address, and a stub alongside it would stack a
    second review surface on the same need. Reuses ``cluster_gaps``
    unchanged (same support/purity floors, same clustering); only the
    candidate filtering is new. Never mutates state, never writes,
    never registers — the caller decides what happens with the list."""
    summary = cluster_gaps(state, **kwargs)
    targets: set = set()
    if ledger is not None:
        targets = {p.get("target") for p in ledger.items
                   if isinstance(p, dict)}
    addressed: List[Dict[str, Any]] = []
    unaddressed: List[Dict[str, Any]] = []
    for cand in summary["candidates"]:
        target = f"gap-cluster:{cand['label']}"
        (addressed if target in targets else unaddressed).append(cand)
    return {"summary": summary, "unaddressed": unaddressed,
            "addressed": addressed}

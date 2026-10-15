"""CLI surface for the cortex layer: ``chat``, ``route``, ``cortex``.

Commands (routed by hub.py):

  caelestia-assist chat [--json] [--file PATH] [--apply-confirmed]
      Interactive conversational REPL. Every routed plan is shown with
      evidence and a calibrated confidence; NOTHING writes until the
      y/N prompt answers y (the multi-change confirmation gate issue
      #120 asks for, in conversational form). ``--json`` switches to a
      line-delimited JSON protocol for the QML bridge / scripts — pure
      information unless ``--apply-confirmed`` ALSO asserts the
      caller's own confirmation UI was used.

  caelestia-assist route "text" [--file PATH] [--json] [-k N]
      One-shot: route + plan + evidence, no interaction, never writes.

  caelestia-assist cortex report|recall|reset-learning|suggest
      The self-learning and memory dashboard: calibration, strategy
      bandit, drift, episodic recall, and the proactive co-change
      suggestion (as a ledger proposal — ``suggest`` never applies).

State lives in the brain's state.json (keys ``cortex_learn`` and
``cortex_memory``), written through the same atomic-save path as every
other learned model. The chat/session state itself is per-process.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant.capabilities.brain import state as brain_state
from assistant.capabilities.settings import applier as settings_applier
from assistant.capabilities.settings import consequences
from assistant.capabilities.settings import history as settings_history
from assistant.capabilities.settings import planner as settings_planner
from assistant.capabilities.settings.cli import default_target, render_plan
from . import learn as cortex_learn
from .delegate import run_delegate, runner_names
from .learn import CortexLearner
from .memory import (
    followup_suggestion,
    record as memory_record,
    recall as memory_recall,
    resolve_halflife,
    summarize as memory_summarize,
)
from .pipeline import episode_for, process as cortex_process
from .plans import DISCARD_RE, PlanCache
from .session import SessionState

LEARN_KEY = "cortex_learn"
MEMORY_KEY = "cortex_memory"


# ---------------------------------------------------------------------------
# Shared state helpers.
# ---------------------------------------------------------------------------


def _load_learner(state: Optional[Dict[str, Any]] = None) -> CortexLearner:
    data = state if state is not None else brain_state.load()
    return CortexLearner(data.get(LEARN_KEY))  # type: ignore[arg-type]


def _load_memory(state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    data = state if state is not None else brain_state.load()
    raw = data.get(MEMORY_KEY) or []
    return [dict(row) for row in raw]  # type: ignore[union-attr]


def _now() -> datetime:
    return datetime.now()


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _fmt_candidates(result) -> List[str]:
    lines = []
    for cand in result.candidates[:5]:
        marker = "->" if cand is result.candidates[0] else "  "
        lines.append(f"{marker} {cand['surface']}  (score {cand['score']:.3f}, "
                     f"kind {cand['kind']})")
    return lines


def _render_turn(result, *, applied: bool = False,
                 inline_delegate: bool = False) -> List[str]:
    """Human rendering of one cortex result (the chat card).

    ``inline_delegate``: the DELEGATE target already ran inline this turn
    (its answer follows this card), so the "try:" hint is suppressed —
    it stays the fallback for the rare case the inline run itself fails.
    """
    lines: List[str] = []
    verdict_label = {
        "PLAN": "FOUND A CHANGE" if applied else "PROPOSED CHANGE",
        "QUESTION": "NEED A DETAIL",
        "ABSTAIN": "NO MATCH",
        "DELEGATE": "OTHER LAYER",
        "EXPLAIN": "EXPLANATION",
        "UNDO": "UNDO",
        "LIST": "HISTORY",
        "INERT": "OUTSIDE SHELL.JSON",
    }.get(result.verdict, result.verdict)
    lines.append("[cortex] " + verdict_label + "  (confidence "
                 # D8: a cold-start confidence is the flat Beta prior —
                 # a fake number. Say so instead.
                 + ("uncalibrated (no history)"
                    if getattr(result, "calibrated", None) is False
                    else f"{result.confidence:.2f}")
                 + (f", strategy {result.strategy}" if result.strategy else "") + ")")
    if result.session_note:
        lines.append(f"  {result.session_note}")
    lines.extend(f"  {line}" for line in _fmt_candidates(result))
    if result.evidence:
        shown = "; ".join(dict.fromkeys(result.evidence))[:220]
        lines.append(f"  evidence: {shown}")
    if result.verdict == "PLAN" and result.plan is not None:
        lines.append("")
        lines.extend("  " + line for line in render_plan(
            result.plan, [], "cortex chat", None, applied, None, None))
    if result.verdict == "QUESTION":
        lines.extend(f"  ? {q}" for q in result.questions)
    if result.verdict == "EXPLAIN" and result.explain_answer:
        lines.append(f"  {result.explain_answer.get('answer', '')}")
        cites = result.explain_answer.get("cites") or []
        if cites:
            lines.append(f"  grounded in: {'; '.join(str(c) for c in cites)}")
    if result.history_plan:
        hp = result.history_plan
        lines.append(f"  {hp.get('reason', '')}")
        for entry in hp.get("entries", [])[:5]:
            label = entry.get("label") or f"#{entry.get('id')}"
            lines.append(f"    - [{entry.get('id')}] {label} at {entry.get('at')}")
    if result.delegate and not inline_delegate:
        hint = {
            "diagnose": "caelestia-assist diagnose <file>",
            "search": "caelestia-assist search \"...\"",
            "brain": "caelestia-assist brain --help",
            "issue": "caelestia-assist issue draft --title ...",
            "agent": "caelestia-assist agent \"...\" --simulate",
        }.get(result.delegate, result.delegate)
        lines.append(f"  try: {hint}")
    if result.suggestions:
        lines.extend(f"  {s}" for s in result.suggestions)
    for note in result.notes[:6]:
        if note:
            lines.append(f"  note: {note}")
    return lines


def _render_report(learner: CortexLearner, episodes: List[Dict[str, object]]) -> List[str]:
    report = learner.report()
    memory = memory_summarize(episodes)
    lines = [
        "cortex — learned intelligence report",
        f"  routing examples observed : {report['examples']} "
        f"({report['accepts']} accepted / {report['rejects']} rejected)",
        f"  acceptance rate           : {report['acceptance_rate']}",
        f"  calibration (overall)     : {report['calibration']['overall']}",
    ]
    for bucket, mean in report["calibration"]["buckets"].items():
        lines.append(f"    {bucket:8s}: {mean}")
    lines.append("  strategy arms (Thompson):")
    for name, row in report["strategy_arms"].items():
        lines.append(f"    {name:10s}: mean {row['mean']} over {row['draws']} draws")
    lines.append(f"  fitted signal weights     : {report['fitted_weights']}")
    lines.append(f"  drift                     : {report['drift']}")
    lines.append(f"  drift (BOCPD, hit-rate)   : {report['drift_bocpd']}")
    # exponential-build 3 item B2: dual-detector consensus — drift is
    # FLAGGED only when BOTH Page-Hinkley and ADWIN have alarmed.
    lines.append(f"  drift (PH+ADWIN consensus) : "
                 f"{report['drift_ph_adwin']['summary']}")
    lines.append(f"  bandit regret vs best arm : {report['regret'].get('status')}"
                 f" (est. {report['regret'].get('estimated_regret')})")
    lines.append(f"  memory episodes           : {memory['episodes']} "
                 f"({memory['applied_episodes']} applied)")
    if memory["top_surfaces"]:
        tops = ", ".join(f"{row['surface']}({row['count']})" for row in memory["top_surfaces"][:5])
        lines.append(f"  most-changed settings     : {tops}")
    if memory["top_cochanges"]:
        pairs = ", ".join(f"{row['a']}+{row['b']} (lift {row['lift']})" for row in memory["top_cochanges"][:3])
        lines.append(f"  strongest co-changes      : {pairs}")
    if memory["peak_hour"] is not None:
        lines.append(f"  customization peak hour   : {memory['peak_hour']}:00")
    return lines


# ---------------------------------------------------------------------------
# The chat REPL.
# ---------------------------------------------------------------------------


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _apply_plan(result, target: Path, text: str) -> bool:
    """The one write path of the chat — the existing applier, gated by
    the interactive confirmation (issue #120's multi-change gate)."""
    if result.plan is None:
        return False
    if not _confirm("apply these changes?"):
        return False
    try:
        settings_applier.apply(result.plan, target, write=True, label=f"chat: {text}")
    except settings_applier.ApplierError as exc:
        print(f"  error: {exc}", file=sys.stderr)
        return False
    return True


def _run_undo(result, target: Path) -> bool:
    if not result.history_plan:
        return False
    action = result.history_plan.get("action")
    try:
        if action == "undo":
            steps = int(result.history_plan.get("steps") or 1)
            if not _confirm(f"undo the last {steps} change(s)?"):
                return False
            outcome = settings_history.undo(target, steps=steps)
        elif action == "undo_by_id":
            entry_id = int(result.history_plan.get("entry_id") or 0)
            if not _confirm(f"revert history entry #{entry_id}?"):
                return False
            outcome = settings_history.undo_by_id(target, entry_id=entry_id)
        else:
            return False
    except settings_history.HistoryError as exc:
        print(f"  error: {exc}", file=sys.stderr)
        return False
    print(f"  restored: {bool(outcome.get('restored') or outcome.get('written'))}"
          + (f" — {outcome.get('message', '')}" if outcome.get("message") else ""))
    return bool(outcome.get("restored") or outcome.get("written"))


def cmd_chat(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="caelestia-assist chat",
        description="conversational settings assistant (learned routing, "
                    "confirmation-gated applies, never writes without y)",
    )
    parser.add_argument("--file", default=None, help="target shell.json override")
    parser.add_argument("--json", action="store_true",
                        help="line-delimited JSON protocol per turn (machine mode)")
    parser.add_argument("--apply-confirmed", action="store_true",
                        help="JSON mode only: the caller asserts its own confirmation "
                             "UI was used; single-op plans apply without the y/N prompt")
    parser.add_argument("--no-learn", action="store_true",
                        help="do not read or update learned state this session")
    args = parser.parse_args(argv)

    target = Path(args.file) if args.file else default_target()
    state = brain_state.load()
    learner = None if args.no_learn else _load_learner(state)
    episodes = _load_memory(state)
    review_bucket = list(state.get("cortex_review", []))
    session = SessionState()
    plan_cache = PlanCache.from_dict(session.pending_plan)

    print(f"cortex chat — natural-language shell.json assistant")
    print(f"target: {target}")
    print("commands: /exit /history /undo N | everything else is a request")
    if args.json:
        print(json.dumps({"type": "session", "target": str(target), "apply_allowed": bool(args.apply_confirmed)}))

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text in ("/exit", "/quit", "/q"):
            break
        if text == "/history":
            for turn in session.to_dict()["turns"]:
                print(f"  you: {turn['text']}  ->  {', '.join(turn['surfaces']) or turn['verdict']}")
            continue
        if text.startswith("/undo"):
            parts = text.split()
            steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            try:
                outcome = settings_history.undo(target, steps=steps)
            except settings_history.HistoryError as exc:
                print(f"  error: {exc}", file=sys.stderr)
                continue
            print(f"  restored: {bool(outcome.get('restored'))}")
            continue

        # Phase 2.5: explicit discard drops the pending plan (the honest
        # way out of an iteration, instead of refusing it forever).
        if DISCARD_RE.match(text):
            dropped = plan_cache.pending_count()
            plan_cache.discard()
            session.pending_plan = None
            print(f"  discarded {dropped} pending change(s)")
            continue

        # Phase 2.5: an explicit APPLY of the accumulated pending plan —
        # recognized BEFORE the session's yes-answer machinery can eat it
        # ("apply" is also a yes-word there; here it is an instruction).
        if (re.match(r"^\s*(?:apply|apply these|apply these changes|"
                     r"apply them|apply it|go ahead|commit)\s*[.!]?\s*$",
                     text, re.IGNORECASE)
                and plan_cache.pending_count() > 0):
            pending_ops = plan_cache.pending()
            class _ApplyResult:  # minimal shape _apply_plan needs
                plan = None
            try:
                apply_result = _ApplyResult()
                apply_result.plan = settings_planner.plan(pending_ops,
                                                          target)
            except settings_planner.PlannerError as exc:
                print(f"  error: composed plan refused: {exc}",
                      file=sys.stderr)
                continue
            if _apply_plan(apply_result, target, "pending plan"):
                plan_cache.commit()
                session.pending_plan = None
                print(f"  applied {len(pending_ops)} composed change(s) "
                      "(backup written; 'undo the last change' reverts it)")
            else:
                session.pending_plan = plan_cache.to_dict()
                pending_line = plan_cache.summary()
                print(f"  {pending_line} — still pending (refused)")
            continue

        # Phase 2.7: what-if turns render the CONSEQUENCE view for the
        # composed plan (pending + this request) and never gate an apply —
        # iterate first, commit later.
        if re.match(r"^\s*what\s+if\b", text, re.IGNORECASE):
            whatif_text = re.sub(r"^\s*what\s+if\b", "", text,
                                 flags=re.IGNORECASE).strip()
            if whatif_text:
                # "what if X" — plan X (read-only), compose it with the
                # pending plan, and project the CONSEQUENCES.
                whatif_result = cortex_process(
                    whatif_text, session=session, learner=learner,
                    file_path=target, now=_now())
                new_ops = whatif_result.ops or []
            else:
                # A BARE "what if" projects the pending plan itself — it
                # must NOT route a placeholder request (the router's
                # fuzzy match on "show my pending changes" once resolved
                # to a toast toggle and polluted the pending plan).
                new_ops = []
            ops = new_ops or plan_cache.pending()
            if ops:
                if new_ops:
                    ops = plan_cache.compose(new_ops, whatif_text)
                current: Dict[str, Any] = {}
                if target.exists():
                    try:
                        current, _notes = settings_planner._read_current(
                            target)
                    except Exception:
                        current = {}
                # The composed RAW ops (later-wins per tool, mirroring
                # the compound layer) re-validate through the standard
                # planner; the PROJECTION then runs over the planner's
                # RESOLVED entries — step/multiply ops become absolute
                # values, never raw deltas (projecting a step delta of
                # -1 as an absolute scale was a real bug this pins).
                resolved_ops: List[Dict[str, Any]] = []
                try:
                    composed_plan = settings_planner.plan(ops, target)
                    resolved_ops = [
                        {"tool": e.get("tool"), "value": e.get("new")}
                        for e in composed_plan.get("entries", [])
                        if not e.get("error")]
                except settings_planner.PlannerError:
                    pass  # the consequence view renders regardless
                projection = consequences.project(
                    resolved_ops or ops, current=current)
                if args.json:
                    print(json.dumps({"type": "whatif", "projection":
                                      projection}, default=str))
                else:
                    print("\n".join(consequences.render(
                        projection, title="what-if")))
                    pending_line = plan_cache.summary()
                    if pending_line:
                        print(f"  {pending_line}")
                        print("  say the next change, or 'apply these "
                              "changes' when ready")
                session.pending_plan = plan_cache.to_dict()
            else:
                print("  nothing to project: no ops in this request and "
                      "no pending plan")
            continue

        result = cortex_process(text, session=session, learner=learner, file_path=target, now=_now())

        # Phase 2.5: compose this turn's ops with the PENDING plan (later
        # wins per tool — the compound layer's own rule) and re-validate
        # through the standard planner before proposing anything.
        if result.verdict == "PLAN" and result.ops:
            if plan_cache.pending_count() > 0:
                composed_ops = plan_cache.compose(result.ops, text)
                try:
                    composed_plan = settings_planner.plan(composed_ops,
                                                          target)
                    result.plan = composed_plan
                    result.ops = composed_ops
                    result.session_note = (
                        (result.session_note + " " if result.session_note
                         else "") + f"composed with your pending plan "
                        f"({plan_cache.pending_count()} ops total)")
                except settings_planner.PlannerError as exc:
                    result.notes.append(f"composed plan refused: {exc}")
            else:
                plan_cache.compose(result.ops, text)

        # Issue #120 phase 4.3: near-threshold phrases are parked for batch
        # review instead of being silently absorbed by the online learner.
        if cortex_learn.is_near_threshold(result.verdict):
            review_bucket = cortex_learn.log_review_candidate(
                review_bucket, text, result.verdict,
                result.candidates or [], at=_now())

        # Inline delegation (routing fix, phase 1.2/1.3): the DELEGATE
        # target runs in this turn — genius (as before, now generalized),
        # diagnose, search, brain, issue, and the agent's simulated task
        # graph. A failed inline run degrades to the hint card below.
        if result.verdict == "DELEGATE" and result.delegate in runner_names():
            inline = run_delegate(result.delegate, text, state)
            if inline is not None:
                payload, inline_lines = inline
                if args.json:
                    print(json.dumps(
                        {"type": "turn", "verdict": "DELEGATE",
                         "delegate": result.delegate,
                         result.delegate: payload}, default=str))
                else:
                    print("\n".join(_render_turn(result, inline_delegate=True)))
                    print("\n".join(inline_lines))
                if learner is not None:
                    episodes = memory_record(episodes,
                                             episode_for(result, text, "routed", now=_now()))
                continue
            result.notes.append(
                f"inline {result.delegate} run failed — the fallback hint stands")

        # Calibration surfacing (phase 2.4): the honest line users can
        # hold the assistant to — observed accuracy for THIS confidence
        # bucket, only when the sample can support a claim. The bucket
        # key is the RAW route probability (learn_hook carries it); the
        # calibrated confidence is a different quantity.
        raw_p = ((result.learn_hook or {}).get("p", result.confidence)
                 if result.learn_hook is not None else result.confidence)
        calibration_note = (learner.calibration_note(raw_p)
                            if learner is not None else None)

        applied = False
        undone = False
        if result.verdict == "PLAN":
            if args.json:
                if args.apply_confirmed and result.plan and len(result.plan.get("entries", [])) == 1:
                    try:
                        settings_applier.apply(result.plan, target, write=True, label=f"chat: {text}")
                        applied = True
                    except settings_applier.ApplierError as exc:
                        result.notes.append(f"apply failed: {exc}")
            else:
                applied = _apply_plan(result, target, text)
        elif result.verdict == "UNDO":
            if not args.json:
                undone = _run_undo(result, target)

        if args.json:
            payload = result.to_dict()
            payload["type"] = "turn"
            payload["applied"] = applied
            payload["undone"] = undone
            print(json.dumps(payload))
        else:
            card = _render_turn(result, applied=applied)
            if calibration_note:
                card.append(f"  {calibration_note}")
            print("\n".join(card))
            if applied:
                plan_cache.commit()
                session.pending_plan = None
                print("  applied (backup written; 'undo the last change' "
                      "reverts it)")
            elif result.verdict == "PLAN" and plan_cache.pending_count():
                session.pending_plan = plan_cache.to_dict()
                pending_line = plan_cache.summary()
                if pending_line:
                    print(f"  {pending_line} — compose the next change or "
                          "say 'apply these changes'")

        # Learning + memory (persisted through the brain state).
        if learner is not None:
            outcome = ("applied" if applied else
                       "undone" if undone else
                       "rejected" if result.verdict == "PLAN" else
                       "clarified" if result.verdict == "QUESTION" else
                       "routed" if result.verdict == "PLAN" else result.verdict.lower())
            if result.learn_hook is not None:
                hook = result.learn_hook
                learner.observe(hook["text"], hook["surface"], hook["features"],
                                hook["p"], outcome)
            if applied and result.learn_hook is not None:
                learner.reward_strategy(result.strategy or "balanced", True)
            elif result.verdict == "PLAN" and not applied:
                learner.reward_strategy(result.strategy or "balanced", False)
            episodes = memory_record(episodes, episode_for(result, text, outcome, now=_now()))
            # Proactive follow-up (second-brain suggestion, proposal only).
            if applied and result.candidates:
                suggestions = followup_suggestion(
                    episodes, result.candidates[0]["surface"], now=_now(),
                    halflife_days=resolve_halflife(state))
                for suggestion in suggestions[:1]:
                    print(f"  suggestion: you often also change {suggestion['surface']} "
                          f"({suggestion['count']}x together) — say '{suggestion['surface'].replace('set', 'change ').strip()}' "
                          f"if you want it proposed")

    if learner is not None or episodes or review_bucket:
        state = brain_state.load()
        state[LEARN_KEY] = learner.to_dict() if learner is not None else state.get(LEARN_KEY)
        state[MEMORY_KEY] = episodes
        state["cortex_review"] = review_bucket
        brain_state.save(state)
    if not args.json:
        print("session ended; learned routing and memory updated")
    return 0


# ---------------------------------------------------------------------------
# One-shot route.
# ---------------------------------------------------------------------------


def cmd_route(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="caelestia-assist route",
        description="one-shot universal routing: any phrase -> ranked surfaces + "
                    "validated plan (read-only, never writes)",
    )
    parser.add_argument("text", help="the request phrase")
    parser.add_argument("--file", default=None, help="target shell.json override")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-k", type=int, default=5, help="top-k candidates")
    args = parser.parse_args(argv)

    target = Path(args.file) if args.file else default_target()
    state = brain_state.load()
    learner = _load_learner(state)
    result = cortex_process(args.text, learner=learner, file_path=target, now=_now())
    # Inline delegation (routing fix, phase 1.2/1.3): one-shot answers the
    # same way the chat REPL does — run the target layer inline, render in
    # the same turn; a failed run falls through to the hint card.
    if result.verdict == "DELEGATE" and result.delegate in runner_names():
        inline = run_delegate(result.delegate, args.text, state)
        if inline is not None:
            payload, inline_lines = inline
            if args.json:
                print(json.dumps({"verdict": "DELEGATE", "delegate": result.delegate,
                                  result.delegate: payload}, default=str))
            else:
                print("\n".join(_render_turn(result, inline_delegate=True)))
                print("\n".join(inline_lines))
            return 0
        result.notes.append(
            f"inline {result.delegate} run failed — the fallback hint stands")
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print("\n".join(_render_turn(result)))
    return 0


# ---------------------------------------------------------------------------
# Cortex management (report / recall / reset-learning / suggest).
# ---------------------------------------------------------------------------


def cmd_cortex(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="caelestia-assist cortex",
        description="the learned-intelligence dashboard: calibration, strategies, "
                    "drift, episodic memory, co-change suggestions",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="learning + memory report")
    recall_p = sub.add_parser("recall", help="recall past interactions")
    recall_p.add_argument("query", nargs="?", default="", help="surface or text filter")
    recall_p.add_argument("-k", type=int, default=10)
    sub.add_parser("reset-learning", help="wipe learned routing weights (memory kept)")
    suggest_p = sub.add_parser("suggest", help="co-change follow-up suggestions")
    suggest_p.add_argument("surface", help="tool name just applied (e.g. setBarScale)")
    suggest_p.add_argument("--apply", action="store_true",
                           help="record the suggestion as a pending LEDGER proposal "
                                "(still needs ledger approve to write)")
    review_p = sub.add_parser("review", help="batch-review near-threshold phrases "
                                              "(logged, never silently "
                                              "learned)")
    review_p.add_argument("action", nargs="?", default="list",
                          choices=["list", "label", "dismiss"])
    review_p.add_argument("arg1", nargs="?", default="",
                          help="label/dismiss: the candidate index (see list)")
    review_p.add_argument("arg2", nargs="?", default="",
                          help="label: the correct surface (e.g. setBarScale)")
    hl = sub.add_parser("halflife", help="read or set the memory decay "
                                         "half-life in days (user-editable; "
                                         "default 14, bounds 0.5-365)")
    hl.add_argument("days", nargs="?", type=float, default=None,
                    help="new half-life in days (omit to read)")
    conf_p = sub.add_parser("confusables", help="list the mined confusable "
                            "tool pairs and their clarifying questions "
                            "(A4: the ambiguity ask, made answerable)")
    conf_p.add_argument("--mine", action="store_true",
                        help="rebuild the committed pairs artifact "
                             "(assistant/cortex/confusable_pairs.json) — a "
                             "reviewed change like a registry re-pin")
    gaps_p = sub.add_parser("gaps", help="cluster the logged local-ontology gaps "
                                         "(cloud hand-offs) and surface candidates")
    gaps_p.add_argument("--propose", action="store_true",
                        help="record qualifying clusters as pending LEDGER "
                             "proposals (kind ontology_gap; approve/reject "
                             "decides — never auto-applied)")
    gaps_p.add_argument("--min-support", type=int, default=3,
                        help="minimum requests per cluster (default 3, the "
                             "workspace.py floor)")
    gaps_p.add_argument("--purity", type=float, default=0.6,
                        help="minimum modal-shape coverage (default 0.6, the "
                             "workspace.py floor)")
    gaps_p.add_argument("--draft-stubs", action="store_true",
                        help="draft a reviewable tool-template stub file "
                             "(name, cue words, TODO body) for each dense, "
                             "unaddressed cluster — never registered, never "
                             "wired into the dispatcher, never clobbered "
                             "once a human edits it")
    gaps_p.add_argument("--stubs-dir", default=None,
                        help="where stub drafts are written (default: "
                             "~/.local/state/caelestia-brain/tool_templates)")
    lex_p = sub.add_parser("lexicon", help="federated, opt-in, signed "
                                            "lexicon-diff sharing "
                                            "(export/import/forget; no network, "
                                            "no auto-merge — verify signatures "
                                            "with YOUR external tool)")
    lex_p.add_argument("action", choices=["export", "import", "forget",
                                           "list", "trust"],
                       help="export: print the reviewable diff; import: apply a "
                            "diff from stdin as supervised pairs; forget: drop "
                            "one imported set by id; list: imported sets; "
                            "trust: the ADVISORY signer trust view (never "
                            "auto-decides anything)")
    lex_p.add_argument("diff_id", nargs="?", default="",
                       help="forget: the diff id to drop (see list)")
    lex_p.add_argument("--signer", default=None,
                       help="import: the identity YOU verified with your own "
                            "tool (minisign/sq/gpg) — recorded for the "
                            "advisory trust history only", )
    lex_p.add_argument("--dp", nargs="?", const=1.0, default=None,
                       type=float, metavar="EPSILON",
                       help="export: pass the export through the Laplace-"
                            "mechanism noising pass (cortex/dp.py, Dwork "
                            "et al. 2006) — per-row n/p are noised at "
                            "epsilon per export (default 1.0 when the "
                            "value is omitted) and every row carries a "
                            "(dp: epsilon=X) provenance marker; absent "
                            "flag = the exact export, byte-identical to "
                            "before")
    lex_p.add_argument("--dp-seed", type=int, default=None, metavar="SEED",
                       help="export --dp: the noise seed (default: derived "
                            "deterministically from the diff's content id, "
                            "so the same diff noised twice is reproducible "
                            "— reproducible noise is NOT independent "
                            "across exports; pass any int for fresh noise)")
    cn = sub.add_parser("concept", help="taught concepts (F15): user-named "
                                         "bundles of validated tool calls, "
                                         "recalled by phrase, Beta-posterior "
                                         "learning, reviewable share diffs")
    cn.add_argument("action", choices=["teach", "list", "forget", "recall",
                                        "outcome", "export", "import"])
    cn.add_argument("name", nargs="?", default="",
                    help="teach/forget/outcome/recall: the concept name "
                         "(or free text for recall)")
    cn.add_argument("--call", action="append", dest="calls", default=None,
                    metavar="TOOL=VALUE",
                    help="teach: a validated tool call (repeatable; every "
                         "tool must exist in the registry)")
    cn.add_argument("--example", action="append", dest="examples", default=None,
                    help="teach: an example phrase that should recall this "
                         "concept (repeatable, up to 4)")
    cn.add_argument("--accept", action="store_true",
                    help="outcome: the concept's plan was accepted (default "
                         "with --accept; omit for reject)")
    cn.add_argument("--force", action="store_true",
                    help="import: overwrite existing concepts with the "
                         "same name (default: skip them)")
    cn.set_defaults(fn=None)

    pw = sub.add_parser("power", help="predictive battery/thermal advice "
                                      "(read-only; caller-supplied series, "
                                      "inert suggestions)")
    pw.add_argument("--battery", default=None, metavar="P,P,...",
                    help="battery capacity series in percent, uniformly "
                         "spaced (e.g. 80,78,76,74)")
    pw.add_argument("--thermal", default=None, metavar="M,M,...",
                    help="thermal zone series in millidegrees C, uniformly "
                         "spaced")
    pw.add_argument("--battery-minutes", type=float, default=30.0,
                    metavar="MIN",
                    help="minutes between battery samples (default 30)")
    pw.add_argument("--thermal-minutes", type=float, default=5.0,
                    metavar="MIN",
                    help="minutes between thermal samples (default 5)")
    pw.add_argument("--low", type=float, default=20.0, metavar="PCT",
                    help="battery low threshold in percent (default 20)")
    pw.add_argument("--high", type=float, default=85000.0, metavar="MC",
                    help="thermal high threshold in millidegrees C "
                         "(default 85000 = 85 C)")
    pz = sub.add_parser("personalize", help="F16 personalized "
                                            "suggestions mined from your "
                                            "approved history")
    pz.add_argument("action", nargs="?", default="list",
                    choices=["list", "file"],
                    help="list = read-only mining view; file = file "
                         "qualifying suggestions into the brain ledger "
                         "(the #120 proposal surface; you approve/reject "
                         "there)")
    pz.add_argument("--file", metavar="PATH", default=None,
                    help="settings target whose history to mine (default: "
                         "the settings layer's default shell.json)")
    pz.add_argument("--ledger", metavar="PATH", default=None,
                    help="brain ledger path for file (default: the brain "
                         "CLI's DEFAULT_LEDGER)")
    cx = sub.add_parser("context", help="F30 context-aware "
                                        "recommendations (read-only probes, "
                                        "inert commands)")
    cx.add_argument("--probe", action="store_true",
                    help="read the real power state from "
                         "/sys/class/power_supply (read-only; without "
                         "this, no battery advice is made)")
    cx.add_argument("--now", metavar="ISO", default=None,
                    help="anchor for hour-bucket selection (default: the "
                         "real clock — this is a pull-based verb)")
    cx.add_argument("--ledger", metavar="PATH", default=None,
                    help="fit the preference model from this brain ledger "
                         "instead of the DEFAULT_LEDGER")
    sc = sub.add_parser("schedule", help="F19 workspace schedules "
                                         "(pull-based; nothing runs by "
                                         "itself)")
    sc.add_argument("action", choices=["save", "list", "delete", "eval",
                                        "file"],
                    help="save = define (needs --name and --profile or "
                         "--preset, condition via --window/--on-battery); "
                         "list/delete/eval are read-only; file = file the "
                         "FIRING schedules into the brain ledger")
    sc.add_argument("--name", default=None, help="schedule name")
    sc.add_argument("--profile", default=None,
                    help="action: a saved profile (F9)")
    sc.add_argument("--preset", default=None, help="action: a preset name")
    sc.add_argument("--window", default=None,
                    help="condition: daily HH:MM-HH:MM (may wrap midnight)")
    sc.add_argument("--on-battery", default=None,
                    choices=["true", "false"],
                    help="condition: fire when discharging (true) or "
                         "charging/on AC (false)")
    sc.add_argument("--now", dest="sched_now", metavar="ISO", default=None,
                    help="eval/file: anchor time (default: the real clock "
                         "— eval is a pull-based verb)")
    sc.add_argument("--probe", action="store_true",
                    help="eval/file: read the real power state (read-only)")
    sc.add_argument("--file", metavar="PATH", default=None,
                    help="settings target whose schedules/history to use")
    sc.add_argument("--ledger", metavar="PATH", default=None,
                    help="brain ledger for file (default: DEFAULT_LEDGER)")
    rf = sub.add_parser("refit", help="F28 guarded weight re-fit: replay "
                                      "the example log through a fresh "
                                      "logistic, measure candidate vs "
                                      "current on the dev arena, adopt "
                                      "only on no regression")
    rf.add_argument("--apply", action="store_true",
                    help="persist the candidate weights when the ratchet "
                         "adopts (default: measure and report only)")
    tm = sub.add_parser("telemetry", help="engine coverage/accuracy "
                                         "report; opt-in Laplace-DP export "
                                         "(read-only, stdout only)")
    tm.add_argument("--export", action="store_true",
                    help="print the shareable artifact instead of the "
                         "local report")
    tm.add_argument("--dp", nargs="?", const=1.0, default=None,
                    type=float, metavar="EPSILON",
                    help="--export: pass the artifact through the Laplace "
                         "noising pass (cortex/dp.py's mechanism; default "
                         "epsilon 1.0 when the value is omitted)")
    tm.add_argument("--dp-seed", type=int, default=None, metavar="SEED",
                    help="--export --dp: noise seed (default: derived from "
                         "the rows' content hash — reproducible, NOT "
                         "independent across exports)")
    args = parser.parse_args(argv)

    state = brain_state.load()
    learner = _load_learner(state)
    episodes = _load_memory(state)

    if args.cmd == "halflife":
        from .memory import (HALFLIFE_BOUNDS, HALFLIFE_DAYS, MEMORY_HALFLIFE_KEY,
                             resolve_halflife)
        if args.days is None:
            print(f"memory decay half-life: {resolve_halflife(state)} days "
                  f"(default {HALFLIFE_DAYS}; bounds "
                  f"{HALFLIFE_BOUNDS[0]}-{HALFLIFE_BOUNDS[1]})")
            return 0
        lo, hi = HALFLIFE_BOUNDS
        if not (lo <= args.days <= hi):
            print(f"error: half-life must be within {lo}-{hi} days",
                  file=sys.stderr)
            return 1
        state[MEMORY_HALFLIFE_KEY] = args.days
        brain_state.save(state)
        print(f"memory decay half-life set to {args.days} days "
              "(affects recall weighting from now on; history untouched)")
        return 0

    if args.cmd == "power":
        # exponential-build-3 E2: read-only advisory over caller-
        # supplied series. Suggestions are INERT SUGGESTED_NOT_EXECUTED
        # strings; nothing writes, nothing executes.
        from . import power_advisor
        battery = thermal = None
        try:
            if args.battery is not None:
                battery = [float(x) for x in args.battery.split(",")
                           if x.strip()]
            if args.thermal is not None:
                thermal = [float(x) for x in args.thermal.split(",")
                           if x.strip()]
        except ValueError:
            print("error: --battery/--thermal need comma-separated "
                  "numbers", file=sys.stderr)
            return 2
        if battery is None and thermal is None:
            print("error: pass --battery P,P,... and/or --thermal M,M,... "
                  "(telemetry series are caller-supplied; no persistent "
                  "sampler ships yet — sample with the on-demand probes "
                  "and pass the series)", file=sys.stderr)
            return 2
        try:
            if battery is not None:
                report_b = power_advisor.advise_battery(
                    battery, sample_minutes=args.battery_minutes,
                    low_pct=args.low)
            if thermal is not None:
                report_t = power_advisor.advise_thermal(
                    thermal, sample_minutes=args.thermal_minutes,
                    high_mc=args.high)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        halves = []
        if battery is not None:
            halves.append(("battery", report_b))
        if thermal is not None:
            halves.append(("thermal", report_t))
        for title, r in halves:
            print(f"{title}:")
            if not r.get("available"):
                print(f"  {r['reason']}")
                continue
            if title == "battery":
                print(f"  level {r['level']:g}%, trend "
                      f"{r['trend_pct_per_hour']:+.2f}%/h"
                      + (" (charging)" if r["charging"] else ""))
                if r["minutes_to_low"] is not None:
                    print(f"  projected to reach "
                          f"{r['low_threshold_pct']:g}% in ~"
                          f"{r['minutes_to_low'] / 60:.1f} h")
            else:
                print(f"  level {r['level_mc'] / 1000:.1f} C, trend "
                      f"{r['trend_mc_per_sample'] / 1000:+.2f} C/sample")
                if r["minutes_to_high"] is not None:
                    print(f"  projected to reach "
                          f"{r['high_threshold_mc'] / 1000:.0f} C in ~"
                          f"{r['minutes_to_high']:.0f} min")
            path = "; ".join(
                f"{f['point']} [{f['low']}, {f['high']}]"
                for f in r["forecast"][:3])
            print(f"  forecast: {path} ... ({r['interval_note']})")
            cp = r["changepoint"]
            if cp.get("recent"):
                print("  changepoint: RECENT regime change — "
                      "trend extrapolation is built on mixed history")
            elif cp.get("checked"):
                print(f"  changepoint: {cp['note']}")
            else:
                print(f"  changepoint: {cp['note']}")
            if r.get("suggestion"):
                print(f"  {r['suggestion']}")
        return 0

    if args.cmd == "personalize":
        # F16: personalized suggestions mined from CONSENTED evidence
        # (the settings history ring holds only approved applies).
        # list = read-only; file = writes ONLY the brain ledger, the
        # same #120 proposal surface every other proposal rides — the
        # approve/reject decision is the user's, through the existing
        # flow. Filing is an explicit verb, never a background loop.
        from . import personalize
        from assistant.capabilities.settings.history import entries as history_entries
        from assistant.capabilities.settings.cli import default_target
        target = Path(args.file) if args.file else default_target()
        try:
            entries = history_entries(target)
        except Exception as exc:
            print(f"error: cannot read the settings history at {target}: "
                  f"{exc}", file=sys.stderr)
            return 1
        mined = personalize.mine(entries)
        if args.action == "list":
            print("\n".join(personalize.render_lines(mined)))
            return 0
        from assistant.capabilities.brain.cli import DEFAULT_LEDGER
        from assistant.capabilities.brain.ledger import Ledger
        ledger = Ledger(Path(args.ledger) if args.ledger else DEFAULT_LEDGER)
        stats = personalize.file_suggestions(mined, ledger)
        print(f"filed {stats['filed']} suggestion(s) into "
              f"{ledger.path} "
              f"(pending duplicates skipped: "
              f"{stats['skipped_pending']}, recently decided skipped: "
              f"{stats['skipped_cooldown']})")
        if stats["filed"] == 0:
            print("nothing new to file (already pending, recently "
                  "decided, or below the evidence floors)")
        print("decide via: caelestia-assist inbox (the unified "
              "pending-decisions view; or brain ledger approve|reject)")
        return 0

    if args.cmd == "context":
        # F30: context-aware recommendations. Read-only probes; the
        # command strings are SUGGESTED_NOT_EXECUTED; the clock selects,
        # it never supplies evidence.
        from . import context as context_mod
        from datetime import datetime as _dt
        now = None
        if args.now:
            try:
                now = _dt.fromisoformat(args.now)
            except ValueError:
                print("error: --now needs an ISO-8601 datetime",
                      file=sys.stderr)
                return 2
        else:
            now = _dt.now()
        power = context_mod.read_power_state() if args.probe else None
        model = None
        from assistant.capabilities.brain.cli import DEFAULT_LEDGER
        from assistant.capabilities.brain.ledger import Ledger
        from assistant.capabilities.brain.prefs import PreferenceModel
        ledger_path = args.ledger or str(DEFAULT_LEDGER)
        if Path(ledger_path).exists():
            model = PreferenceModel()
            model.from_ledger(Ledger(ledger_path).items)
            if not model.table:
                model = None
        view = context_mod.recommend_contextual(now=now, power=power,
                                                model=model)
        print("\n".join(context_mod.render_lines(view)))
        return 0

    if args.cmd == "schedule":
        # F19: pull-based workspace schedules. eval/file are explicit
        # verbs; nothing runs by itself, ever.
        from . import schedules as sched_mod
        from assistant.capabilities.settings.cli import default_target
        from assistant.capabilities.settings.history import HistoryError
        target = Path(args.file) if args.file else default_target()
        try:
            if args.action == "save":
                if not args.name:
                    print("error: save needs --name", file=sys.stderr)
                    return 2
                on_battery = (None if args.on_battery is None
                              else args.on_battery == "true")
                entry = sched_mod.save(
                    target, args.name, profile=args.profile,
                    preset=args.preset, window=args.window,
                    on_battery=on_battery)
                print(f"saved schedule {entry['name']!r} "
                      f"(window {entry.get('window')}, "
                      f"on-battery {entry.get('on_battery')})")
                print("check it with: cortex schedule eval")
                return 0
            if args.action == "list":
                saved = sched_mod.list_schedules(target)
                if not saved:
                    print("no schedules defined; define one with "
                          "schedule save --name N --profile P "
                          "--window HH:MM-HH:MM")
                for s in saved:
                    print(f"  {s['name']:<20} action={s['action']} "
                          f"window={s.get('window')} "
                          f"on-battery={s.get('on_battery')}")
                return 0
            if args.action == "delete":
                if not args.name:
                    print("error: delete needs --name", file=sys.stderr)
                    return 2
                result = sched_mod.delete(target, args.name)
                print(f"deleted schedule {result['deleted']!r}")
                return 0
            # eval / file
            from datetime import datetime as _dt
            if args.sched_now:
                try:
                    now = _dt.fromisoformat(args.sched_now)
                except ValueError:
                    print("error: --now needs an ISO-8601 datetime",
                          file=sys.stderr)
                    return 2
            else:
                now = _dt.now()
            power = (context_probe_power() if args.probe else None)
            result = sched_mod.evaluate(target, now=now, power=power)
            print("\n".join(sched_mod.render_eval(result)))
            if args.action == "file":
                from assistant.capabilities.brain.cli import DEFAULT_LEDGER
                from assistant.capabilities.brain.ledger import Ledger
                ledger = Ledger(Path(args.ledger) if args.ledger
                                else DEFAULT_LEDGER)
                stats = sched_mod.file_firing(result, ledger, now=now)
                print(f"filed {stats['filed']} firing schedule(s) into "
                      f"the ledger (pending duplicates skipped: "
                      f"{stats['skipped_pending']}, recently decided "
                      f"skipped: {stats['skipped_cooldown']})")
            return 0
        except sched_mod.ScheduleError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except HistoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.cmd == "refit":
        # F28: the guarded re-fit. The arena measurement decides; the
        # audit trail records every decision; --apply persists the
        # candidate model ONLY when the ratchet adopted it.
        from . import refit as refit_mod
        from .learn import LEARN_KEY
        from .learn import CortexLearner
        state = brain_state.load()
        learner_data = dict(state.get(LEARN_KEY) or {})
        result = refit_mod.refit_ratchet(learner_data, state)
        if result["adopted"] and args.apply:
            learner_data["model"] = result["candidate_model"]
            state[LEARN_KEY] = learner_data
        brain_state.save(state)
        print("\n".join(refit_mod.render_lines(result)))
        if result["adopted"] and not args.apply:
            print("(dry-run: re-run with --apply to persist the "
                  "candidate weights)")
        return 0

    if args.cmd == "telemetry":
        # exponential-build-3 F3: coverage/accuracy-only engine metrics,
        # opt-in Laplace-DP export via the B4 mechanism. Read-only.
        from . import features as capabilities
        if not capabilities.enabled("engine_telemetry"):
            print("error: engine_telemetry is disabled in the capability "
                  "manifest (edit capabilities.json to enable; it cannot "
                  "be enabled by a request)", file=sys.stderr)
            return 1
        from . import engine_telemetry
        rows = engine_telemetry.metrics(learner)
        if not rows:
            print("no routed examples logged yet — nothing to report "
                  "(the engine logs its own routed turns locally)")
            return 0
        if not args.export:
            print("engine telemetry (coverage/accuracy only — no text, "
                  "no phrases, no features leave this machine unless you "
                  "--export)")
            for row in rows:
                thin = "  (thin)" if row["n"] < engine_telemetry.MIN_N \
                    else ""
                acc = "-" if row["accuracy"] is None \
                    else f"{row['accuracy']:.2f}"
                cov = "-" if row["coverage"] is None \
                    else f"{row['coverage']:.2f}"
                print(f"  {row['surface']:<28} n={row['n']:>3}  "
                      f"decided={row['decided']:>3}  coverage={cov:<5} "
                      f"accuracy={acc}{thin}")
            print("export opt-in: telemetry --export [--dp EPSILON] "
                  "(exact artifact is local read-back; --dp applies the "
                  "Laplace mechanism per Dwork et al. 2006)")
            return 0
        date = datetime.now().strftime("%Y-%m-%d")
        if args.dp is None:
            sys.stdout.write(engine_telemetry.export_text(rows, date=date))
            print("# (exact artifact — local read-back; use --dp for the "
                  "shareable noised form)", file=sys.stderr)
            return 0
        try:
            noised = engine_telemetry.noise_metrics(
                rows, epsilon=args.dp, seed=args.dp_seed)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        sys.stdout.write(engine_telemetry.export_text(
            noised["rows"], date=date, dp=noised))
        print(f"# (dp) epsilon={noised['epsilon']:g}, "
              f"seed={noised['seed']}: "
              f"n floored at zero: {noised['n_floored_at_zero']}; "
              f"rates clipped to [0,1]: {noised['p_clipped']}",
              file=sys.stderr)
        print("# (dp) composition: k sequential exports compose to roughly "
              "k*epsilon (Dwork & Roth 2014) — the practical bound is how "
              "often you export", file=sys.stderr)
        return 0

    if args.cmd == "concept":
        from . import concepts
        state = brain_state.load()
        try:
            if args.action == "teach":
                if not args.name or not args.calls:
                    print("error: teach needs a NAME and at least one "
                          "--call TOOL=VALUE", file=sys.stderr)
                    return 2
                hit = concepts.teach(state, args.name, args.calls,
                                     examples=args.examples,
                                     source="cli-explicit")
                brain_state.save(state)
                print(f"taught concept '{hit['name']}' with "
                      f"{len(hit['calls'])} call(s) "
                      f"({hit['n_concepts']} concepts stored); nothing "
                      "applied — recall only proposes, the consent gate "
                      "applies")
                return 0
            if args.action == "list":
                rows = concepts.list_concepts(state)
                if not rows:
                    print("no taught concepts yet — teach one: cortex "
                          "concept teach glassy --call setBlurEnabled=true")
                    return 0
                for r in rows:
                    p = ", ".join(f"{x['tool']}={x['value']!r}"
                                  + (f" p={x['p']} n={x['n']}"
                                     if x["p"] is not None else "")
                                  for x in r["posterior"])
                    print(f"{r['name']}: {p}")
                    if r["examples"]:
                        print(f"  phrases: {'; '.join(r['examples'])}")
                return 0
            if args.action == "forget":
                if concepts.forget(state, args.name):
                    brain_state.save(state)
                    print(f"forgot concept '{args.name}'")
                    return 0
                print(f"error: unknown concept '{args.name}'",
                      file=sys.stderr)
                return 2
            if args.action == "recall":
                from assistant.capabilities.settings.curations import PRESETS
                hit = concepts.recall(state, args.name, presets=PRESETS)
                print(f"verdict: {hit['verdict']}")
                if hit["verdict"] == "RECALL":
                    for c in hit["calls"]:
                        print(f"  {c['name']} = {c['value']!r}")
                    print("  (proposal only — the consent gate applies)")
                return 0
            if args.action == "outcome":
                hit = concepts.record_outcome(state, args.name,
                                              accepted=bool(args.accept))
                brain_state.save(state)
                print(f"recorded {'accept' if hit['accepted'] else 'reject'} "
                      f"for '{hit['name']}'")
                return 0
            if args.action == "export":
                names = [args.name] if args.name else None
                print(json.dumps(concepts.export_diff(state, names),
                                 indent=1, ensure_ascii=False))
                return 0
            if args.action == "import":
                entries = json.load(sys.stdin)
                res = concepts.import_diff(state, entries,
                                           force=bool(args.force))
                brain_state.save(state)
                print(f"import: {res['added']} added, {res['updated']} "
                      f"updated, {res['skipped']} skipped (existing; "
                      "use --force to overwrite)")
                return 0
        except concepts.ConceptError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.cmd == "lexicon":
        from . import features as capabilities
        if not capabilities.enabled("lexicon_sharing"):
            print("error: lexicon_sharing is disabled in the capability "
                  "manifest (edit capabilities.json to enable; it cannot "
                  "be enabled by a request)", file=sys.stderr)
            return 1
        from . import lexicon_diff
        if args.action == "export":
            rows = lexicon_diff.export_rows(state)
            if not rows:
                print("nothing learned to share yet (the learner's example "
                      "log is empty)")
                return 0
            if args.dp is not None:
                # opt-in (exponential-build 3 B4): the Laplace-mechanism
                # noising pass over the export artifact; the default
                # path below stays byte-identical when the flag is absent
                from . import dp
                try:
                    report = dp.noise_diff(
                        rows, epsilon=args.dp, seed=args.dp_seed,
                        date=datetime.now().strftime("%Y-%m-%d"))
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                sys.stdout.write(report["text"])
                print(f"# (dp) epsilon={report['epsilon']:g}, "
                      f"seed={report['seed']}: "
                      f"{report['rows_out']}/{report['rows_in']} rows "
                      f"kept; n floored at zero: "
                      f"{report['n_floored_at_zero']}; p clipped to "
                      f"[0,1]: {report['p_clipped']}; "
                      f"{report['guarantee']}", file=sys.stderr)
                print("# (dp) composition: k sequential exports compose to "
                      "roughly k*epsilon (Dwork & Roth 2014) — the "
                      "practical bound is how often you export",
                      file=sys.stderr)
                print("# export is read-only: sign it with YOUR external tool "
                      "(minisign/sq/gpg) before sharing", file=sys.stderr)
                return 0
            sys.stdout.write(lexicon_diff.render(
                rows, date=datetime.now().strftime("%Y-%m-%d")))
            print("# export is read-only: sign it with YOUR external tool "
                  "(minisign/sq/gpg) before sharing", file=sys.stderr)
            return 0
        if args.action == "import":
            text = sys.stdin.read()
            report = lexicon_diff.import_diff(state, text, signer=args.signer)
            if "error" in report:
                print(f"error: {report['error']}", file=sys.stderr)
                for warn in report.get("warnings", []):
                    print(f"  warning: {warn}", file=sys.stderr)
                return 1
            brain_state.save(state)
            print(f"imported diff {report['diff_id']}: "
                  f"{report['imported']} pair(s)")
            print("  tools this diff boosts (review them):")
            for tool in report["boosted_tools"]:
                print(f"    - {tool}")
            for warn in report.get("warnings", []):
                print(f"  warning: {warn}")
            print(f"  {report['note']}")
            if report.get("signer_trust"):
                print(f"  {report['signer_trust']}")
            print(f"  {report['verify']}")
            print(f"  rollback: {report['rollback']}")
            return 0
        if args.action == "forget":
            report = lexicon_diff.forget(state, args.diff_id)
            if "error" in report:
                print(f"error: {report['error']}", file=sys.stderr)
                return 1
            brain_state.save(state)
            print(f"forgot {report['forgot']} "
                  f"({report['rows_dropped']} row(s)); {report['note']}")
            return 0
        if args.action == "trust":
            scores = lexicon_diff.signer_trust(state)
            if not scores:
                print("no signer history — imports without --signer "
                      "record nothing; trust starts at the flat prior")
                return 0
            print(lexicon_diff.render_advisory(scores))
            return 0
        # list
        ids = lexicon_diff.imported_ids(state)
        if not ids:
            print("no imported lexicon diffs")
            return 0
        imports = state.get(lexicon_diff.IMPORTS_KEY) or {}
        for did in ids:
            entry = imports.get(did) or {}
            print(f"  {did}  {entry.get('n', '?')} pair(s)")
        print("forget with: cortex lexicon forget <id>")
        return 0

    if args.cmd == "report":
        print("\n".join(_render_report(learner, episodes)))
        return 0

    if args.cmd == "recall":
        rows = memory_recall(episodes, args.query, now=_now(), k=args.k,
                             halflife_days=resolve_halflife(state))
        if not rows:
            print("no matching episodes in memory")
            return 0
        for row in rows:
            surfaces = ", ".join(str(s) for s in row.get("surfaces", []))
            print(f"  [{row.get('at')}] {row.get('outcome')}: {row.get('text')}"
                  + (f"  -> {surfaces}" if surfaces else ""))
        return 0

    if args.cmd == "reset-learning":
        state[LEARN_KEY] = CortexLearner().to_dict()
        brain_state.save(state)
        print("learned routing weights reset to priors (memory kept)")
        return 0

    if args.cmd == "review":
        from . import learn as cortex_learn
        bucket = list(state.get(cortex_learn.REVIEW_KEY, []))
        if args.action == "list":
            if not bucket:
                print("no near-threshold candidates logged yet (ambiguous or "
                      "abstained chat phrases land here)")
                return 0
            print(f"{len(bucket)} candidate(s), oldest last:")
            for i, c in enumerate(bucket):
                print(f"  [{i}] {c['verdict']:<8} p={c.get('p')} :: {c['text']}")
            print("label with: cortex review label INDEX SURFACE   "
                  "(teaches the router, nothing applies)")
            print("drop with:  cortex review dismiss INDEX")
            return 0
        if learner is None:
            print("learning is disabled (--no-learn or no state); cannot label",
                  file=sys.stderr)
            return 1
        try:
            if args.action == "label":
                if not args.arg1 or not args.arg2:
                    print("review label needs INDEX and SURFACE", file=sys.stderr)
                    return 2
                res = cortex_learn.label_candidate(state, int(args.arg1),
                                                   args.arg2, learner)
                state[LEARN_KEY] = learner.to_dict()
                brain_state.save(state)
                print(f"taught: {res['text']!r} -> {res['surface']} "
                      f"({res['remaining']} candidate(s) left)")
            else:
                res = cortex_learn.dismiss_candidate(state, int(args.arg1))
                brain_state.save(state)
                print(f"dismissed: {res['text']!r} ({res['remaining']} left)")
            return 0
        except (ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.cmd == "confusables":
        from assistant.core.features import enabled
        if not enabled("confusable_clarifier"):
            print("confusable_clarifier is disabled in the capability "
                  "manifest (edit capabilities.json to enable; it cannot "
                  "be enabled from a request)", file=sys.stderr)
            return 1
        from . import confusables
        if args.mine:
            data = confusables.mine_pairs()
            confusables.PAIRS_PATH.write_text(
                json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print(f"wrote {confusables.PAIRS_PATH.name}: "
                  f"{data['n_pairs']} pairs")
            return 0
        for line in confusables.render_lines():
            print(line)
        return 0

    if args.cmd == "gaps":
        from .dispatch import (cluster_gaps, propose_gap_clusters,
                               select_stub_candidates, tool_template_slug,
                               tool_template_stub_text)
        from assistant.capabilities.brain.cli import DEFAULT_LEDGER
        from assistant.capabilities.brain.ledger import Ledger

        # Stub selection runs BEFORE any proposal is filed, so one run can
        # both file the ledger proposal and leave the human a reviewable
        # scaffold to look at while that proposal is still pending. A
        # cluster the ledger already knows about (any status) is addressed:
        # the ledger flow IS the address, no second review surface.
        draft_report = None
        if args.draft_stubs:
            sel = select_stub_candidates(state, Ledger(DEFAULT_LEDGER),
                                         min_support=args.min_support,
                                         purity=args.purity)
            out_dir = (Path(args.stubs_dir) if args.stubs_dir else
                       Path.home() /
                       ".local/state/caelestia-brain/tool_templates")
            drafted, skipped_existing = [], []
            for cand in sel["unaddressed"]:
                slug = tool_template_slug(cand["label"])
                path = out_dir / f"{slug}.md"
                if path.exists():
                    # A stub that already exists may carry human edits —
                    # never clobbered, reported instead.
                    skipped_existing.append(slug)
                    continue
                if not out_dir.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    tool_template_stub_text(
                        cand, datetime.now().isoformat(timespec="seconds")),
                    encoding="utf-8")
                drafted.append(slug)
            draft_report = {"dir": out_dir, "drafted": drafted,
                            "skipped_existing": skipped_existing,
                            "addressed": [c["label"]
                                          for c in sel["addressed"]]}

        if args.propose:
            res = propose_gap_clusters(state, Ledger(DEFAULT_LEDGER),
                                      min_support=args.min_support,
                                      purity=args.purity)
            if res["proposals"]:
                print(f"{len(res['proposals'])} pending proposal(s) recorded "
                      f"(kind ontology_gap) — `caelestia-assist brain ledger` "
                      f"decides; nothing is applied automatically")
            else:
                print("no qualifying cluster reached the support/purity floor "
                      "— nothing proposed")
            summary = res["summary"]
        else:
            summary = cluster_gaps(state, min_support=args.min_support,
                                   purity=args.purity)
        print(f"local-ontology gaps: {summary['n_gaps']} logged shape(s); "
              f"k={summary['k']}, {summary['rejected_clusters']} cluster(s) "
              f"below the support/purity floor")
        if not summary["candidates"]:
            print("no candidate local tools — a handful of scattered gaps is "
                  "noise, not a need (by design)")
        for c in summary["candidates"]:
            print(f"  {c['support']}x  \"{c['label']}\"  "
                  f"(category {c['category']}, purity {c['purity']}, "
                  f"{c['shapes']} phrasing(s))")
        if summary["candidates"] and not args.propose:
            print("surface with: cortex gaps --propose  (ledger proposals, "
                  "never auto-applied)")
        if draft_report is not None:
            d = draft_report
            if d["drafted"]:
                print(f"drafted {len(d['drafted'])} tool-template stub(s) "
                      f"in {d['dir']}: {', '.join(d['drafted'])}")
                print("  stubs are DRAFT files only — never registered, "
                      "never wired into the dispatcher; a human writes the "
                      "real module")
            else:
                print("no stub drafted — every qualifying cluster is "
                      "already addressed in the ledger or already has its "
                      "stub file")
            if d["skipped_existing"]:
                print(f"  skipped {len(d['skipped_existing'])} existing "
                      f"stub file(s) (never clobbered)")
            if d["addressed"]:
                print(f"  {len(d['addressed'])} cluster(s) already "
                      f"addressed in the ledger (no second review "
                      f"surface): {'; '.join(d['addressed'])}")
        return 0

    if args.cmd == "suggest":
        suggestions = followup_suggestion(episodes, args.surface, now=_now(),
                                           halflife_days=resolve_halflife(state))
        if not suggestions:
            print(f"no co-change pattern with {args.surface} yet "
                  f"(needs repeated applied history)")
            return 0
        for suggestion in suggestions:
            print(f"  {suggestion['surface']}: {suggestion['count']}x together, "
                  f"decay-weighted score {suggestion['weight']}")
        if args.apply:
            from assistant.capabilities.brain import ledger as brain_ledger
            from assistant.capabilities.brain import settings_bridge
            top = suggestions[0]["surface"]
            # brain/ledger.py exposes the Ledger class (no module-level
            # load/save helpers); use DEFAULT_LEDGER for continuity with the
            # brain CLI's proposal store.
            from assistant.capabilities.brain.cli import DEFAULT_LEDGER
            ledger = brain_ledger.Ledger(DEFAULT_LEDGER)
            pid = settings_bridge.propose(
                ledger, str(default_target()), None, [], 
                reason=f"cortex co-change suggestion after {args.surface} "
                       f"({top} changed together {suggestions[0]['count']}x)",
                confidence=round(min(0.9, 0.4 + 0.1 * suggestions[0]["count"]), 2),
            )
            if pid is None:
                print("nothing to propose (the dry-run plan was blocked or a no-op)")
            else:
                print(f"ledger proposal {pid} created (approve with: "
                      f"caelestia-assist brain ledger approve {pid})")
        return 0

    return 2


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else [])
    if not argv:
        argv = ["--help"]
    if argv[0] == "chat":
        return cmd_chat(argv[1:])
    if argv[0] == "route":
        return cmd_route(argv[1:])
    if argv[0] in ("report", "recall", "reset-learning", "suggest", "review"):
        return cmd_cortex(argv)
    return cmd_cortex(argv)


def context_probe_power():
    """One read-only power probe (F30's reader), shared by the schedule
    verbs."""
    from . import context as context_mod
    return context_mod.read_power_state()


if __name__ == "__main__":
    raise SystemExit(main())

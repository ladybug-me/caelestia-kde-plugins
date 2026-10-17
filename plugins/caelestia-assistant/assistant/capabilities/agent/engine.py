"""agent.engine — the agent loop: plan -> simulate -> consent -> act -> learn.

    caelestia-assist agent "clean my downloads then make the shell minimal"
    caelestia-assist agent "why does vesktop freeze" --simulate     # no prompts

This is the "human employee on the shell" seam, built strictly on the
existing safety rails rather than beside them:

  1. goals.decompose() turns the request into a dependency-ordered task
     graph whose leaves are ACTIONS bound to real layer dispatchers;
  2. simulate() walks the graph and reports what WOULD run, at what risk,
     without executing anything — the projection IS the deliverable;
  3. execute() runs read-only nodes directly; every consent_required node
     goes through the caller's consent_fn (CLI: a y/N prompt; bridge: an
     explicit round-trip; tests: a scripted lambda). A refused node marks
     its downstream skipped — never partially applied;
  4. every outcome (accepted/rejected/failed) is observed into the learning
     loop (cortex learn hook + brain preference model) so the NEXT graph is
     shaped by what you actually approve.

Hard rules enforced here, in one place:
  - STATE_CHANGING nodes never run without a True from consent_fn;
  - PRIVILEGED/DESTRUCTIVE work is never implemented as a node at all — it
    surfaces as an inert SUGGESTED_NOT_EXECUTED string in a result;
  - a failed node does not abort the graph: independent branches continue,
    dependents are skipped with the reason.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from assistant.capabilities.brain import brief as brain_brief
from assistant.capabilities.brain import tidy
from assistant.capabilities.diagnostics import engine
from assistant.capabilities.genius import meta as genius_meta
from assistant.capabilities.retrieval import search as retrieval_search
from . import goals

__all__ = ["Agent", "DISPATCHERS", "default_dispatchers"]

ConsentFn = Callable[[Dict[str, Any]], bool]
Dispatcher = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]

_INERT_PREFIX = "SUGGESTED_NOT_EXECUTED"


# ---------------------------------------------------------------------------
# Dispatchers: action name -> (ctx, params) -> plain result dict.
# Every dispatcher is offline and read-only EXCEPT tidy_apply/apply_plan,
# which only run after the engine's consent gate anyway.
# ---------------------------------------------------------------------------

def _d_diagnose(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    d = engine.diagnose(ctx.get("text", ""))
    return {"verdict": d.get("verdict"),
            "candidates": [{"id": c["rule"]["id"],
                            "title": c["rule"].get("title", ""),
                            "score": c["score"]}
                           for c in d.get("candidates", [])[:3]],
            "margin": d.get("margin")}


def _d_retrieve(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    hits = retrieval_search.search(ctx.get("text", ""), k=int(params.get("k", 3)))
    return {"hits": [{"id": h.get("id"), "title": h.get("title"),
                      "score": round(float(h.get("score", 0.0)), 3)}
                     for h in hits[:params.get("k", 3)]]}


def _d_scan(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from assistant.capabilities.scan import Automaton, scan_text
    auto = Automaton(["quickshell", "kwin", "dbus", "error", "critical",
                      "segfault", "failed to", "timeout"])
    text = ctx.get("stream_text", "")
    if not text:
        return {"skipped": "no stream text provided (pass stream_text in ctx)"}
    s = scan_text(text, auto)
    return {"lines_scanned": s["lines_scanned"],
            "lines_matching": s["lines_matching"],
            "rate_alarm": s["rate"]["page_hinkley_change_at_chunk"] is not None,
            "top_hits": s["pattern_hits"][:3]}


def _d_fix_plan(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Compose the fix plan from earlier nodes' results stored in ctx."""
    diag = ctx.get("results", {}).get("diagnose", {})
    retr = ctx.get("results", {}).get("retrieve_similar", {})
    lines: List[str] = []
    if diag.get("verdict") and diag["verdict"] != "NO_MATCH":
        for c in diag.get("candidates", []):
            lines.append(f"known signature {c['id']}: {c['title']}")
    for h in retr.get("hits", [])[:3]:
        lines.append(f"closest past resolution: {h.get('id')} — {h.get('title')}")
    if not lines:
        lines.append("no verified match; enable Debug Mode and capture "
                     "journalctl --user -u caelestia-shell output")
    lines.append(f"{_INERT_PREFIX}: [commands the cited resolution used — "
                 f"copy them yourself after reading]")
    return {"plan": lines}


def _d_explain(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    retr = ctx.get("results", {}).get("retrieve_similar", {})
    out = []
    for h in retr.get("hits", []):
        out.append(f"{h.get('id')} ({h.get('score')}): {h.get('title')}")
    return {"explanation": out or ["nothing found in the local corpus"]}


def _d_route_request(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from assistant.core import pipeline as cortex_pipeline
    res = cortex_pipeline.process(ctx.get("text", ""))
    ops = [{"tool": op.get("tool"), "action": op.get("action"),
            "value": op.get("value")}
           for op in (getattr(res, "ops", None) or [])]
    # The planner's RESOLVED entries (absolute values): step/multiply
    # ops become the values they actually produce. The consequence
    # projection (validate_plan) must run over these — projecting a raw
    # step delta (-1) as an absolute scale fires false edges.
    resolved = [{"tool": e.get("tool"), "value": e.get("new")}
                for e in ((getattr(res, "plan", None) or {}).get("entries") or [])
                if not e.get("error")]
    return {"verdict": res.verdict, "action": getattr(res, "action", ""),
            "ops_count": len(ops),
            "ops": ops,
            "resolved_ops": resolved,
            "detail": str(getattr(res, "detail", ""))[:400]}


def _d_validate_plan(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    route = ctx.get("results", {}).get("route_request", {})
    verdict = route.get("verdict")
    if verdict == "PLAN":
        # Phase 2.7: the consent card shows CONSEQUENCES, not just
        # actions — the what-if projection rides the validation node so
        # --simulate and the execution report both carry it (read-only
        # view; the proposal's agent seam).
        consequences = {}
        # Prefer the RESOLVED ops (absolute values, steps resolved by
        # the planner); fall back to raw ops only when the plan is
        # absent.
        project_ops = route.get("resolved_ops") or route.get("ops")
        if project_ops:
            try:
                from assistant.capabilities.settings import consequences as consequences_mod

                consequences = consequences_mod.project(project_ops)
            except Exception:
                consequences = {}
        return {"validated": True, "note": "cortex produced a validated "
                "plan", "consequences": consequences}
    if verdict in ("QUESTION", "AMBIGUOUS"):
        return {"validated": False, "needs_input": True,
                "note": route.get("detail", "the request needs one answer")}
    return {"validated": False, "needs_input": verdict == "ABSTAIN",
            "note": route.get("detail", "no settings intent found")}


def _d_propose_plan(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """The consent presentation itself: the engine gates the node; this
    dispatcher only formats what the user is deciding on."""
    route = ctx.get("results", {}).get("route_request", {})
    return {"proposal": route.get("detail", "settings plan pending"),
            "risk": "STATE_CHANGING",
            "note": "nothing is written until you approve"}


def _d_apply_plan(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs ONLY behind the engine consent gate. Delegates to the cortex
    chat turn machinery by re-processing with an explicit approval answer —
    the applier, history, and undo stay the cortex's, unchanged."""
    from assistant.core import pipeline as cortex_pipeline
    res = cortex_pipeline.process("yes")
    return {"applied": getattr(res, "verdict", "") == "PLAN",
            "note": "applied through the cortex's own gated path; "
                    "undo: caelestia-assist settings --undo"}


def _d_tidy_survey(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    root = ctx.get("tidy_root", params.get("root") or "~/Downloads")
    plan = tidy.survey(root)
    ctx["tidy_plan"] = plan
    return {"scanned": plan["scanned"],
            "moves": len(plan["type_moves"]) + len(plan["duplicates"]),
            "space_recoverable_mb": round(plan["space_recoverable"] / 1048576, 1),
            "plan": {k: plan[k] for k in ("root", "scanned", "big_files",
                                          "empty_dirs")}}


def _d_tidy_propose(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    plan = ctx.get("tidy_plan") or tidy.survey(ctx.get("tidy_root", "~/Downloads"))
    ctx["tidy_plan"] = plan
    return {"rendered": tidy.render_plan(plan),
            "risk": "STATE_CHANGING",
            "note": "moves happen only after approval; rollback is journaled"}


def _d_tidy_apply(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    plan = ctx.get("tidy_plan")
    if not plan:
        return {"applied": 0, "note": "no surveyed plan in context"}
    moves = plan["type_moves"] + plan["duplicates"]
    res = tidy.apply_moves(moves, plan["root"])
    return {"applied": len(res["applied"]), "skipped": len(res["skipped"]),
            "journal": res["journal"]}


def _d_brief(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    b = brain_brief.compose(
        pending=ctx.get("pending"),
        stuck=ctx.get("stuck"),
        today=ctx.get("today"),
        forecast=ctx.get("forecast"),
        pref_lines=ctx.get("pref_lines"))
    return {"brief": b, "rendered": brain_brief.render(b)}


def _d_genius(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return genius_meta.route_and_do(ctx.get("text", ""))


# -- phase 2.3 archetype dispatchers (assistant/agent/archetypes.py) ------


def _d_lint_config(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    result = archetypes.lint_config(ctx.get("config_target"))
    ctx["lint_findings"] = result.get("findings", [])
    return result


def _d_config_drift(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    return archetypes.config_drift(ctx.get("config_target"))


def _d_reconcile_propose(ctx: Dict[str, Any],
                         params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    proposal = archetypes.reconcile_proposal(ctx.get("lint_findings"),
                                             ctx.get("config_target"))
    ctx["reconcile_ops"] = proposal.get("ops", [])
    return proposal


def _d_reconcile_apply(ctx: Dict[str, Any],
                       params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs ONLY behind the engine consent gate (the node is
    consent_required=True): the standard planner+applier path."""
    from . import archetypes
    return archetypes.reconcile_apply(ctx.get("reconcile_ops") or [],
                                      ctx.get("config_target"))


def _d_package_report(ctx: Dict[str, Any],
                      params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    return archetypes.package_report()


def _d_log_triage(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    text = ctx.get("stream_text") or ctx.get("text", "")
    lines = text.splitlines() if text else []
    if not lines:
        return {"skipped": "no log lines (pass stream_text in ctx)"}
    return archetypes.triage_logs(lines)


def _d_triage_draft(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """sysintel triage -> issues draft PREVIEW (no --confirm, no file
    write — the issues layer's own contract)."""
    import contextlib
    import io

    from assistant.capabilities.issues import cli as issues_cli
    from . import archetypes

    triage = ctx.get("results", {}).get("log_triage", {})
    if not triage or triage.get("skipped"):
        return {"skipped": "no triage result to draft from"}
    body = archetypes.triage_draft_description(triage)
    title = ("log triage: " + (ctx.get("text", "")[:60] or "template mining"))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = issues_cli.main(["draft", "--title", title],
                               stdin_text=body)
    if code != 0:
        return {"error": f"issue draft exited {code}"}
    return {"draft": buffer.getvalue(),
            "note": "preview only — nothing written; pass --confirm "
                    "yourself to save the draft file"}


def _d_notification_triage(ctx: Dict[str, Any],
                           params: Dict[str, Any]) -> Dict[str, Any]:
    from . import archetypes
    events = ctx.get("notification_events") or []
    result = archetypes.triage_notifications(events)
    if not events:
        result["skipped"] = ("no event records provided (pass "
                             "notification_events in ctx); the live DBus "
                             "observation surface is a separate, "
                             "capability-gated concern")
    return result


def _d_screenshot_diff(ctx: Dict[str, Any],
                       params: Dict[str, Any]) -> Dict[str, Any]:
    import re as _re

    from . import archetypes

    before = ctx.get("screenshot_before") or params.get("before")
    after = ctx.get("screenshot_after") or params.get("after")
    if not before or not after:
        # last resort: .png paths mentioned in the request itself
        paths = _re.findall(r"[\w./~:-]+\.(?:png|PNG)", ctx.get("text", ""))
        if len(paths) >= 2 and not before and not after:
            before, after = paths[0], paths[1]
    if not before or not after:
        return {"skipped": "two PNG paths needed (screenshot_before / "
                           "screenshot_after in ctx, or both named in the "
                           "request)"}
    return archetypes.diff_screenshots(str(before), str(after))


def _d_screenshot_draft(ctx: Dict[str, Any],
                        params: Dict[str, Any]) -> Dict[str, Any]:
    import contextlib
    import io

    from assistant.capabilities.issues import cli as issues_cli
    from . import archetypes

    diff = ctx.get("results", {}).get("screenshot_diff", {})
    if not diff or diff.get("skipped"):
        return {"skipped": "no diff result to draft from"}
    body = archetypes.screenshot_draft_description(diff)
    title = ("visual regression: " + (ctx.get("text", "")[:56]
                                       or "structural screenshot diff"))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = issues_cli.main(["draft", "--title", title],
                               stdin_text=body)
    if code != 0:
        return {"error": f"issue draft exited {code}"}
    return {"draft": buffer.getvalue(),
            "note": "preview only — nothing written"}


DISPATCHERS: Dict[str, Dispatcher] = {
    "diagnose": _d_diagnose,
    "retrieve_similar": _d_retrieve,
    "scan_stream": _d_scan,
    "fix_plan": _d_fix_plan,
    "explain_result": _d_explain,
    "route_request": _d_route_request,
    "validate_plan": _d_validate_plan,
    "propose_plan": _d_propose_plan,
    "apply_plan": _d_apply_plan,
    "tidy_survey": _d_tidy_survey,
    "tidy_propose": _d_tidy_propose,
    "tidy_apply": _d_tidy_apply,
    "brief": _d_brief,
    "genius_dispatch": _d_genius,
    # phase 2.3 archetypes
    "lint_config": _d_lint_config,
    "config_drift": _d_config_drift,
    "reconcile_propose": _d_reconcile_propose,
    "reconcile_apply": _d_reconcile_apply,
    "package_report": _d_package_report,
    "log_triage": _d_log_triage,
    "triage_draft": _d_triage_draft,
    "notification_triage": _d_notification_triage,
    "screenshot_diff": _d_screenshot_diff,
    "screenshot_draft": _d_screenshot_draft,
}

READ_ONLY_RESULT_KEYS = ("plan", "explanation", "hits", "brief", "rendered",
                         "proposal", "applied")


def default_dispatchers() -> Dict[str, Dispatcher]:
    return dict(DISPATCHERS)


# ---------------------------------------------------------------------------
class Agent:
    """One agent run: plan -> (clarify) -> simulate -> execute -> report."""

    def __init__(self,
                 dispatchers: Optional[Dict[str, Dispatcher]] = None,
                 consent_fn: Optional[ConsentFn] = None,
                 observer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 clarify_questions: Optional[List[Any]] = None,
                 base_ctx: Optional[Dict[str, Any]] = None) -> None:
        self.dispatchers = dispatchers or default_dispatchers()
        self.consent_fn = consent_fn or (lambda node: False)
        self.observer = observer  # learning hook: (node_action, outcome_dict)
        self.clarify_questions = clarify_questions or []
        self.base_ctx = dict(base_ctx or {})
        self._graph: Dict[str, Any] = {}
        self._ctx: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def plan(self, text: str) -> Dict[str, Any]:
        graph = goals.decompose(text)
        if "error" in graph:
            return graph
        self._graph = graph
        self._ctx = {**self.base_ctx, "text": text, "results": {}}
        return graph

    # ------------------------------------------------------------------
    def simulate(self, text: str) -> Dict[str, Any]:
        """Projection only: what would run, in what order, at what risk.
        Nothing executes — this is the honest 'dry run' the safety contract
        requires, and it is also the plan a human reads before consenting."""
        graph = self.plan(text)
        if "error" in graph:
            return graph
        projection = []
        for node in graph["nodes"]:
            projection.append({
                "id": node["id"], "action": node["action"],
                "title": node["title"], "risk": node["risk"],
                "consent_required": node["consent_required"],
                "depends": node["depends"],
                "would": (f"execute {node['action']} (read-only)"
                          if node["risk"] == "READ_ONLY"
                          else f"ASK you before {node['action']}"),
            })
        return {"projection": projection,
                "order": graph["order"],
                "critical_path": graph["critical_path"],
                "consent_required": graph["consent_required"],
                "ambiguous": graph["ambiguous"],
                "note": ("the request was ambiguous — consider answering the "
                         "clarifying question first" if graph["ambiguous"]
                         else "graph is fully determined")}

    # ------------------------------------------------------------------
    def execute(self, text: str) -> Dict[str, Any]:
        graph = self.plan(text)
        if "error" in graph:
            return graph
        done: Dict[str, str] = {}
        for node in graph["nodes"]:
            deps_ok = all(done.get(d) == "done" for d in node["depends"])
            if not deps_ok:
                failed = [d for d in node["depends"] if done.get(d) != "done"]
                node["status"] = "skipped"
                node["skip_reason"] = f"dependencies not completed: {failed}"
                done[node["id"]] = "skipped"
                self._observe(node, "skipped")
                continue
            if node["consent_required"]:
                allowed = bool(self.consent_fn(node))
                if not allowed:
                    node["status"] = "refused"
                    node["result"] = {"note": "you refused; nothing was changed"}
                    done[node["id"]] = "refused"
                    self._observe(node, "refused")
                    continue
                node["consented"] = True
            dispatcher = self.dispatchers.get(node["action"])
            if dispatcher is None:
                node["status"] = "failed"
                node["result"] = {"error": f"no dispatcher for {node['action']}"}
                done[node["id"]] = "failed"
                self._observe(node, "failed")
                continue
            try:
                result = dispatcher(self._ctx, node.get("params", {}))
                node["result"] = result
                node["status"] = "done"
                done[node["id"]] = "done"
                # Results are addressable BOTH by node id and by action
                # name: downstream dispatchers compose from earlier nodes
                # by action ("diff", "diagnose", ...), which is what the
                # original method ids promised ("match", "retrieve") but
                # never delivered — node ids are n1, n2, ... so the
                # action-name key is the one composition actually uses.
                self._ctx["results"][node["id"]] = result
                self._ctx["results"][node["action"]] = result
                self._observe(node, "accepted")
            except Exception as exc:  # honest failure, graph continues
                node["status"] = "failed"
                node["result"] = {"error": f"{type(exc).__name__}: {exc}"}
                done[node["id"]] = "failed"
                self._observe(node, "failed")
        return {
            "nodes": graph["nodes"],
            "order": graph["order"],
            "summary": self._summary(graph),
            "ctx_results": self._ctx["results"],
        }

    # ------------------------------------------------------------------
    def _observe(self, node: Dict[str, Any], outcome: str) -> None:
        if self.observer is not None:
            try:
                self.observer(node["action"], {
                    "outcome": outcome,
                    "goal": node.get("goal"),
                    "consent_required": node.get("consent_required"),
                    "risk": node.get("risk"),
                })
            except Exception:
                pass  # learning must never break execution

    @staticmethod
    def _summary(graph: Dict[str, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in graph["nodes"]:
            counts[n["status"]] = counts.get(n["status"], 0) + 1
        return counts

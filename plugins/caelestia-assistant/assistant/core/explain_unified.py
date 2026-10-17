"""cortex.explain_unified — ONE ``why`` over every engine that surfaces a
decision.

The assistant explains itself in five different places, each in its own
module: diagnostics rules cite their file:line sources; the cortex router
shows its clause->surface path plus the conformal interval; the settings
layer answers "why does it look like this" against the live config
(settings/explain.py); the brain exposes Beta-Binomial acceptance
posteriors (brain/calibrate.py); the setup wizard renders its AHP/TOPSIS
weights (settings/wizard.py). This module walks back through WHICHEVER
engine produced the last surfaced item and renders the SAME structured
shape for all of them:

    {"engine": str,       # which module's machinery produced the answer
     "headline": str,     # one line, built from the module's own values
     "lines": [str],      # the module's OWN explanation output, verbatim
     "citations": [str],  # the module's own file:line citations, verbatim
     "confidence": float|None}   # the module's own confidence value

TEMPLATING, NOT SYNTHESIZING: every string rendered here is either lifted
from the source module's own output (``settings.explain``'s answer,
``wizard.render``'s lines, ``dispatch.render_answer``'s chat card, the
conformal verdict's own reason/guarantee sentences, a ledger item's own
reason) or a fixed template around the module's own numbers. No engine's
text is paraphrased, no new explanation is invented.

Name-collision note (grep-first): settings/explain.py owns the settings
layer's read-only explanations; genius/metacog.py owns rule induction and
request clustering. Neither is touched or re-implemented here — this
module only CALLS them and re-renders what they already say.

Walk-back (`why` with no id): the newest ledger proposal IS the last
surfaced action with a durable record; its kind names the engine that
produced it (settings -> the settings layer, ontology_gap -> the cortex
gap machinery, drift_* -> the brain's drift differ), and the item's own
reason/confidence ride along verbatim. `why <id>` accepts the unified
inbox's ids (ledger:<pid>, gap:<pid|label>, plan:<tool>, agent:<node>).

Read-only: this module writes nothing anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant.capabilities.brain import calibrate
from assistant.capabilities.brain import state as brain_state
from assistant.capabilities.brain.ledger import Ledger
from . import dispatch as dispatch_mod

__all__ = ["explain_diagnostics", "explain_cortex", "explain_settings",
           "explain_brain", "explain_wizard", "explain_ledger_item",
           "last_ledger_item", "explain_last", "render", "main"]

ENGINES = ("diagnose", "route", "settings", "brain", "wizard")


# ---------------------------------------------------------------------------
# Per-engine adapters (each reuses that module's own explanation output).
# ---------------------------------------------------------------------------

def explain_diagnostics(text: str) -> Dict[str, Any]:
    """The diagnostics rule engine's own verdict: top rule, its cited
    fixes (verbatim SUGGESTED_NOT_EXECUTED-safe strings), references."""
    from assistant.capabilities.diagnostics import engine as diagnostics_engine

    d = diagnostics_engine.diagnose(text)
    top = d.get("top")
    if not top:
        lines = ["no rule matched — the honest answer is "
                 "\"I don't know\" (diagnostics engine's own verdict)"]
        unmatched = d.get("unmatched_templates")
        if unmatched:
            lines.append("recurring shapes among the unmatched lines "
                         f"({unmatched.get('n_lines', 0)} line(s) templated):")
            for row in unmatched.get("rows", [])[:5]:
                lines.append(f"  - [{row['count']}x] {row['template'][:140]}")
            lines.append(str(unmatched.get(
                "note", "Drain clusters shapes, not causes"))
                + " — a recurring template is a candidate for a "
                "human-written rule, not a diagnosis")
        return {"engine": "diagnostics", "headline": "NO_MATCH",
                "lines": lines,
                "citations": [], "confidence": None}
    rule = top["rule"]
    lines = [f"{fix.get('text', '')}" for fix in (rule.get("fix") or [])]
    lines.append(f"evidence: {top.get('evidence')}")
    if d.get("margin") is not None:
        lines.append(f"margin over runner-up: {d['margin']} points")
    tools = top.get("settings_tools") or []
    if tools:
        lines.append("settings tools addressing this root cause: "
                     + ", ".join(str(t) for t in tools))
    return {"engine": "diagnostics",
            "headline": f"{d['verdict']}: {rule.get('id')} — "
                        f"{rule.get('title', '')}",
            "lines": lines,
            "citations": [str(r) for r in (rule.get("references") or [])],
            "confidence": rule.get("confidence")}


def explain_cortex(text: str, state: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """The router's own path for one request: pipeline result rendered by
    dispatch's own render_answer (the same chat card the sidebar shows),
    plus the conformal calibrator's own verdict sentence for this score."""
    from .pipeline import process
    from .session import SessionState
    from .learn import CortexLearner
    from .conformal import ConformalCalibrator

    state = state if isinstance(state, dict) else {}
    learner = CortexLearner(state.get("cortex_learn")) \
        if state.get("cortex_learn") else None
    result = process(text, session=SessionState(), learner=learner)
    lines = dispatch_mod.render_answer(result)
    citations: List[str] = list(dict.fromkeys(result.evidence or []))
    conformal_note: Optional[str] = None
    data = state.get("conformal")
    if isinstance(data, dict) and data.get("scores"):
        cal = ConformalCalibrator()
        cal.from_dict(data)
        verdict = cal.verdict(result.confidence)
        conformal_note = str(verdict.get("reason") or "")
        if verdict.get("guarantee"):
            lines.append(str(verdict["guarantee"]))
        lines.append(conformal_note)
    else:
        cal = ConformalCalibrator()
        verdict = cal.verdict(result.confidence)
        lines.append(str(verdict.get("reason")))
    top = result.candidates[0] if result.candidates else None
    surface = top.get("surface") if top else None
    headline = f"{result.verdict}"
    if surface:
        headline += f" via {surface} (p={top.get('p')})"
    # exponential-build-3 G3: on ABSTAIN, report the top-2 score gap —
    # "no setting clearly matches" is more useful with "here is how
    # close the best guesses were" (the router's own ranking, its own
    # numbers; nothing new is computed here)
    gap = getattr(result, "abstain_gap", None)
    if result.verdict in ("ABSTAIN", "OUT_OF_ONTOLOGY") and gap:
        lines.append(
            f"top-2 gap at abstain: '{gap['top']}' scored "
            f"{gap['top_score']:.3f} vs '{gap['runner_up']}' at "
            f"{gap['runner_up_score']:.3f} (gap {gap['gap']:.3f}) — "
            "neither cleared the routing bar")
    return {"engine": "cortex", "headline": headline, "lines": lines,
            "citations": citations, "confidence": result.confidence}


def explain_settings(query: str, target, ledger_path=None) -> Dict[str, Any]:
    """The settings layer's own explain answer (verbatim), plus its
    backward provenance chain (settings/explain.py's own hop strings)."""
    from assistant.capabilities.settings import explain as settings_explain

    result = settings_explain.explain(query, target)
    lines = [str(result["answer"])]
    spec = result.get("spec") or {}
    if spec.get("path"):
        lines.append(f"registry key: {spec['path']} (tool {spec['name']}, "
                     f"kind {spec.get('kind')})")
    if ledger_path is not None:
        try:
            prov = settings_explain.provenance(query, target, ledger_path)
            for hop in prov.get("chain", []):
                lines.append(f"provenance: {hop.get('detail', '')}")
        except settings_explain.ExplainError:
            pass  # a question that is not a registry setting stays as-is
    # knowledge-graph enrichment (exponential-build-5 F12): the curated
    # interaction chain around this key, when one exists — read-only,
    # citation-carrying, and ABSTAINING (no line) when the key has no
    # curated upstream. `why` is not the cold path, so the graph build
    # cost sits behind this engine import, not the CLI's.
    if spec.get("path"):
        try:
            from assistant.capabilities.graph import build as graph_build
            from assistant.capabilities.graph import queries as graph_queries
            g = graph_build.build_graph()
            aff = graph_queries.what_affects(g, str(spec["path"]))
            if aff.get("verdict") == "OK" and aff.get("upstream"):
                ups = ", ".join(
                    f"{u['from']} ({u['confidence']}, {u['citation']})"
                    for u in aff["upstream"][:3])
                lines.append(f"graph: {len(aff['upstream'])} curated "
                             f"interaction(s) act on this key — {ups} "
                             f"(graph affects {spec['path']} for the "
                             f"cited chain)")
            elif aff.get("verdict") == "OK":
                lines.append("graph: no curated interaction acts on this "
                             "key (the table is bounded — absence is not "
                             "a claim of independence)")
        except (ValueError, KeyError):
            # a graph build problem must not take the whole `why` down;
            # the failure is surfaced as the missing enrichment, loudly
            # enough in tests, and never silently fabricated
            lines.append("graph: enrichment unavailable (build error)")
    # causal attribution (exponential-build-5 F14): the backward chain
    # of ARMED, cited interactions explaining the key's CURRENT value,
    # read live from the target file. No armed chain -> the honest
    # "set directly" line; unknown needed values -> abstention lines.
    if spec.get("path"):
        try:
            from pathlib import Path as _Path
            from assistant.capabilities.settings import causal as settings_causal
            from assistant.capabilities.settings import planner as settings_planner
            live_target = (_Path(target) if isinstance(target, str)
                           else target)
            current, _notes = settings_planner._read_current(live_target)
            chain = settings_causal.why_chain(str(spec["path"]), current)
            lines.extend(settings_causal.render_chain(chain))
        except (ValueError, KeyError, OSError):
            lines.append("causal: chain unavailable (state read error)")
    return {"engine": "settings",
            "headline": str(result["answer"]),
            "lines": lines,
            "citations": [str(c) for c in (result.get("cites") or [])],
            "confidence": None}


def explain_brain(ledger_path=None, kind: Optional[str] = None
                  ) -> Dict[str, Any]:
    """The brain's own Beta-Binomial acceptance posterior per proposal
    kind (brain/calibrate.acceptance_rate over the ledger's labeled
    history) — the numbers the ledger surfaces are composed from."""
    lines: List[str] = []
    stats: Dict[str, Dict[str, Any]] = {}
    if ledger_path is not None and Path(ledger_path).exists():
        labeled = Ledger(ledger_path).labeled()
        stats = calibrate.acceptance_rate(labeled)
    if not stats:
        return {"engine": "brain",
                "headline": "no labeled proposal history yet — the "
                            "Beta(1,1) prior (0.5) is all anyone can "
                            "honestly claim",
                "lines": lines, "citations": [], "confidence": None}
    order = sorted(stats.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
    for name, s in order:
        lines.append(f"{name}: approved ~{round(s['mean'] * 100)}% of the "
                     f"time ({s['n']} decisions; alpha={s['alpha']}, "
                     f"beta={s['beta']})")
    chosen = stats.get(kind) if kind else None
    if chosen is None:
        chosen = order[0][1]
        kind = order[0][0]
    headline = (f"proposals of kind {kind} were approved "
                f"~{round(chosen['mean'] * 100)}% of the time "
                f"({chosen['n']} decisions)")
    return {"engine": "brain", "headline": headline, "lines": lines,
            "citations": [], "confidence": chosen["mean"]}


def explain_wizard(answers: List[int]) -> Dict[str, Any]:
    """The wizard's own AHP/TOPSIS recommendation and its own render()
    output (settings/wizard.py), verbatim."""
    from assistant.capabilities.settings import wizard as settings_wizard

    result = settings_wizard.run(answers)
    lines = settings_wizard.render(result)
    closeness = float(result["closeness"][result["winner"]])
    return {"engine": "wizard",
            "headline": f"preset '{result['winner']}' — TOPSIS closeness "
                        f"{round(closeness, 3)}",
            "lines": lines,
            "citations": [],
            "confidence": round(closeness, 3)}


# ---------------------------------------------------------------------------
# Walk-back: which engine produced the last surfaced item?
# ---------------------------------------------------------------------------

def explain_ledger_item(item: Dict[str, Any], target=None,
                        ledger_path=None) -> Dict[str, Any]:
    """Walk one ledger proposal back to its producing engine and render
    that engine's own explanation, with the item's own reason/confidence
    riding along verbatim."""
    kind = item.get("kind")
    diff = item.get("diff") or {}
    if kind == "settings":
        calls = diff.get("calls") or []
        query = None
        if calls and isinstance(calls, list):
            query = (calls[0] or {}).get("name")
        query = query or diff.get("preset")
        item_target = diff.get("file") or target
        if query and item_target:
            result = explain_settings(str(query), item_target,
                                      ledger_path=ledger_path)
        else:
            result = explain_brain(ledger_path, kind="settings")
        result["lines"] = [f"ledger reason: {item.get('reason', '')}"] \
            + result["lines"]
        result["confidence"] = item.get("confidence")
        return result
    if kind == "ontology_gap":
        return {"engine": "cortex",
                "headline": f"gap cluster {item.get('target', '')}",
                "lines": [str(item.get("reason", ""))],
                "citations": [], "confidence": item.get("confidence")}
    if kind and str(kind).startswith("drift"):
        return {"engine": "brain",
                "headline": f"{kind}: {item.get('target', '')}",
                "lines": [str(item.get("reason", ""))],
                "citations": [], "confidence": item.get("confidence")}
    return {"engine": "ledger",
            "headline": f"proposal #{item.get('id')} ({kind}): "
                        f"{item.get('target', '')}",
            "lines": [str(item.get("reason", "")),
                      f"stated confidence {item.get('confidence')}"],
            "citations": [], "confidence": item.get("confidence")}


def last_ledger_item(ledger_path) -> Optional[Dict[str, Any]]:
    """The newest ledger record — 'the last action' with a durable
    record (pending or decided)."""
    ledger = Ledger(ledger_path)
    if not ledger.items:
        return None
    return max(ledger.items, key=lambda i: i.get("id", 0))


def explain_last(ledger_path, target=None) -> Dict[str, Any]:
    item = last_ledger_item(ledger_path)
    if item is None:
        return {"engine": "ledger",
                "headline": "no recorded action yet",
                "lines": ["the ledger is empty — nothing to walk back "
                          "through"],
                "citations": [], "confidence": None}
    result = explain_ledger_item(item, target=target,
                                 ledger_path=ledger_path)
    result["lines"] = [f"last action: ledger proposal "
                       f"#{item['id']} ({item.get('kind')}, "
                       f"{item.get('status')})"] + result["lines"]
    return result


# ---------------------------------------------------------------------------
# Inbox-id walk-back (the same id space `inbox list` prints).
# ---------------------------------------------------------------------------

def explain_inbox_id(inbox, item_id: str, goal: Optional[str] = None,
                     target=None) -> Dict[str, Any]:
    source = item_id.split(":", 1)[0]
    suffix = item_id.split(":", 1)[1] if ":" in item_id else ""
    if source in ("ledger", "gap") and suffix.isdigit():
        ledger = Ledger(inbox.ledger_path)
        for raw in ledger.items:
            if raw.get("id") == int(suffix):
                result = explain_ledger_item(raw, target=target,
                                             ledger_path=inbox.ledger_path)
                result["lines"] = [f"item {item_id} "
                                   f"({raw.get('kind')}, "
                                   f"{raw.get('status')})"] + result["lines"]
                return result
        return {"engine": "ledger", "headline": f"no proposal {suffix}",
                "lines": [], "citations": [], "confidence": None}
    if source == "gap":
        return {"engine": "cortex", "headline": f"gap cluster {suffix}",
                "lines": ["an unproposed qualifying cluster — approving it "
                          "files a ledger proposal; nothing has been "
                          "surfaced yet"],
                "citations": [], "confidence": None}
    if source == "plan":
        cache = inbox._pending_plan()
        ops = [op for op in cache.pending()
               if str(op.get("tool", "?")) == suffix]
        if not ops:
            return {"engine": "cortex",
                    "headline": f"no pending plan op for {suffix}",
                    "lines": [], "citations": [], "confidence": None}
        return {"engine": "cortex",
                "headline": f"pending plan op {suffix} = "
                            f"{ops[0].get('value', '?')}",
                "lines": [cache.summary() or "pending plan (no ops)",
                          "the composed list re-validates through the "
                          "standard planner before anything is proposed "
                          "(cortex/plans.py's own contract)"],
                "citations": [], "confidence": None}
    if source == "agent":
        if not goal:
            return {"engine": "agent",
                    "headline": "agent nodes need the goal text (--goal)",
                    "lines": ["the consent queue lives inside one agent "
                              "run — pass the same goal to see it"],
                    "citations": [], "confidence": None}
        from assistant.capabilities.agent.engine import Agent
        projection = Agent().simulate(goal)
        lines = [f"{step.get('id')}: {step.get('would', '')} — "
                 f"{step.get('title', '')}"
                 for step in projection.get("projection", [])]
        return {"engine": "agent",
                "headline": f"agent plan for: {goal}"[:80],
                "lines": lines, "citations": [], "confidence": None}
    return {"engine": "ledger", "headline": f"unknown id {item_id}",
            "lines": [], "citations": [], "confidence": None}


# ---------------------------------------------------------------------------
# The one consistent rendering.
# ---------------------------------------------------------------------------

def render(result: Dict[str, Any]) -> List[str]:
    out = [f"why: [{result['engine']}] {result['headline']}"]
    for line in result.get("lines", []):
        out.append(f"  {line}" if line else "")
    cites = result.get("citations") or []
    if cites:
        out.append(f"  cites: {'; '.join(cites[:4])}"
                   + (f" (+{len(cites) - 4} more)" if len(cites) > 4 else ""))
    if result.get("confidence") is not None:
        out.append(f"  confidence: {result['confidence']} "
                   "(the producing engine's own value)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ledger", default=None, help="proposal ledger JSON")
    p.add_argument("--state", default=None, help="brain state JSON")
    p.add_argument("--file", default=None, help="target shell.json")
    p.add_argument("--json", action="store_true")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="caelestia-assist why",
        description="explain the last action (or a specific item) through "
                    "the engine that produced it — each engine's own "
                    "explanation output, one consistent shape")
    ap.add_argument("target_id", nargs="?", default=None,
                    help="inbox item id (ledger:PID, gap:PID, plan:TOOL, "
                         "agent:NODE); omit to explain the last action")
    ap.add_argument("words", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--engine", choices=ENGINES, default=None,
                    help="explain a live engine answer instead of an item")
    ap.add_argument("--query", default=None,
                    help="text for --engine diagnose/route/settings")
    ap.add_argument("--answers", default=None,
                    help="five wizard answers, e.g. 1,3,5,4,2")
    _add_common(ap)
    args = ap.parse_args(argv)

    from assistant.capabilities.brain.cli import DEFAULT_LEDGER  # the canonical path
    ledger_path = Path(args.ledger) if args.ledger else DEFAULT_LEDGER
    state = brain_state.load(args.state) if args.state else {}

    if args.engine:
        if args.engine == "diagnose":
            result = explain_diagnostics(args.query or " ".join(args.words))
        elif args.engine == "route":
            result = explain_cortex(args.query or " ".join(args.words),
                                    state=state)
        elif args.engine == "settings":
            if not args.file:
                print("why: --engine settings needs --file (the target "
                      "shell.json)", file=sys.stderr)
                return 2
            result = explain_settings(args.query or " ".join(args.words),
                                      args.file, ledger_path=ledger_path)
        elif args.engine == "wizard":
            if not args.answers:
                print("why: --engine wizard needs --answers N,N,... (the "
                      "pairwise answers settings --wizard asks for)",
                      file=sys.stderr)
                return 2
            result = explain_wizard(
                [int(a) for a in args.answers.split(",")])
        else:  # brain
            result = explain_brain(ledger_path)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n".join(render(result)))
        return 0

    if args.target_id is None:
        result = explain_last(ledger_path, target=args.file)
    else:
        from .inbox import Inbox
        inbox = Inbox(ledger_path,
                      Path(args.state) if args.state else None, None)
        result = explain_inbox_id(inbox, args.target_id, goal=args.query,
                                  target=args.file)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("\n".join(render(result)))
    return 0

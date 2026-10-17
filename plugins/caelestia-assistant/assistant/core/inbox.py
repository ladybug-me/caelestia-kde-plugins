"""cortex.inbox — the unified pending-decisions inbox (one view, four sources).

Every surface that can park a decision for the human used to be queried
separately: the brain proposal ledger, the cortex gap clusters that
dispatch.py surfaces as ``ontology_gap`` proposals, the session's pending
settings plan (cortex/plans.py), and the agent's per-node consent queue
(agent/engine.py). This module aggregates them into ONE ranked view and one
dispatch pair — ``inbox approve <id>`` / ``inbox reject <id>`` — without
implementing any approval logic or write path of its own.

READ + DISPATCH, not a fifth decider:

- ``ledger:<pid>`` items dispatch to ``brain.ledger.Ledger.decide`` —
  the same entry point ``brain ledger approve|reject`` has always called;
- ``gap:<...>`` items: an already-surfaced ``ontology_gap`` proposal is a
  ledger item (dispatched through ``Ledger.decide``, exactly like every
  other proposal — dispatch.py's own rule); an unproposed qualifying
  cluster dispatches to ``cortex.dispatch.propose_gap_cluster`` — the
  surfacing module's own entry point, which files it into the ledger for
  the human's real decision. Rejecting an unproposed cluster writes
  nothing anywhere (there is nothing to reject yet) — the honest answer;
- ``plan:<tool>`` items: the pending-plan cache deliberately has no
  per-op approve (it composes by tool, later wins, and re-validates
  through the standard planner before anything is proposed). Approve
  therefore re-validates through ``settings.planner.plan`` and hands the
  plan to ``settings.applier.apply`` — dry-run preview by default, the
  real (backed-up, undo-bounded) write only behind the CLI's explicit
  ``--write``, mirroring the chat loop's confirmation gate — and calls
  ``PlanCache.commit`` only after an apply actually went through.
  Reject dispatches to ``PlanCache.discard`` — the cache's real, explicit,
  TOTAL drop ("never mind"); the output says so, never silently;
- ``agent:<node>`` items come from ``Agent.simulate(goal)``'s projection
  (read-only). Deciding runs ``Agent.execute(goal)`` with a scripted
  ``consent_fn`` — the engine's own per-node consent gate, the same way
  tests and the bridge script it; approve without ``--write`` only shows
  the projection.

RANKING — composed from values that already exist, nothing new computed
(the operating rule: reuse, don't invent a scoring formula):

- ``stated``: the source's own confidence (ledger item ``confidence``,
  gap cluster ``purity``); 1.0 (neutral) when the source states none;
- ``kind_mean``: the Beta-Binomial acceptance posterior for the item's
  kind from ``brain.calibrate.acceptance_rate`` over the ledger's
  labeled history — the module's own Beta(1,1) prior (0.5) for a kind
  with no history;
- ``arm_mean``: the Thompson-sampling bandit arm's mean estimate
  (``NamedBandit.rank`` returns ``(name, draw, mean_estimate)``; the mean
  is the stable, non-random summary the class itself shows humans) —
  keyed by the same arms the settings decisions already reward
  (``tool:<name>`` via ``brain.settings_bridge.TOOL_ARM_PREFIX``, or the
  preset name); the class's own flat Beta(1,1) prior elsewhere. This is
  the bandit cortex/learn.py uses for its strategy arms
  (``brain.preset_bandit.NamedBandit``) — same class, no second
  implementation.

``score = stated * kind_mean * arm_mean`` (rounded 3) — a product of
posterior means, tie-broken by (source, id) so the order is deterministic.

Pure read + dispatch: the module writes only through the source modules'
own entry points (ledger JSON, applier, plan-cache payload the caller
owns). No new write path, no new network, no new imports beyond the
allow-list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from assistant.capabilities.brain import calibrate
from assistant.capabilities.brain import state as brain_state
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.preset_bandit import NamedBandit
from . import dispatch as dispatch_mod
from .plans import PlanCache

__all__ = ["SOURCE_ORDER", "Inbox", "collect_items", "main"]

# Stable tie-break order for items whose composed score is identical:
# ledger decisions first (they gate real writes), then gap clusters,
# then the pending plan, then agent consents.
SOURCE_ORDER = {"ledger": 0, "gap": 1, "plan": 2, "agent": 3}

# The settings-layer posterior the plan ops live under: a plan-cache
# approval is a settings decision (the same kind the applier writes).
_PLAN_KIND = "settings"
_AGENT_KIND = "agent"


def _arm_for(item: Dict[str, Any]) -> Optional[str]:
    """The bandit arm this item already has decisions under, if any.

    Reads the SAME keys settings_bridge.decide rewards: the proposal
    diff's ``preset`` or ``tool:<name>`` arm for settings ledger items;
    the ``tool:<name>`` arm for a plan-cache op. No arm is guessed for
    gap clusters or agent nodes (their decisions have never fed this
    bandit, so no honest value exists)."""
    kind = item.get("kind")
    if item["source"] == "plan" or (item["source"] == "ledger"
                                    and kind == "settings"):
        diff = item.get("_diff") or {}
        preset = diff.get("preset")
        if preset:
            return str(preset)
        tool = diff.get("tool")
        if tool:
            return "tool:" + str(tool)
        op = item.get("_op") or {}
        if op.get("tool"):
            return "tool:" + str(op.get("tool"))
    return None


def _bandit_mean(bandit: NamedBandit, arm: Optional[str]) -> float:
    """The class's own stable mean estimate for one arm (0.5 = its flat
    Beta(1,1) prior for an arm with no history — ``_arm``'s default)."""
    if arm is None:
        return 0.5
    ranked = bandit.rank([arm])
    return float(ranked[0][2])


class Inbox:
    """Aggregates the four sources and dispatches decisions to them.

    Construct with explicit paths (tests pass tmp files; the CLI wires
    the same defaults brain/cli.py uses). Nothing is read until ``items``
    is called; nothing is written except through the source modules.
    """

    def __init__(self, ledger_path, state_path=None,
                 pending_plan_path: Optional[Path] = None):
        self.ledger_path = ledger_path
        self.state_path = state_path
        self.pending_plan_path = pending_plan_path

    # -- sources ------------------------------------------------------------

    def _ledger(self) -> Ledger:
        return Ledger(self.ledger_path)

    def _state(self) -> Dict[str, Any]:
        if self.state_path is None:
            return {}
        return brain_state.load(self.state_path)

    def _bandit(self, state: Dict[str, Any]) -> NamedBandit:
        return NamedBandit.from_dict(state.get("preset_bandit", {}))

    def _pending_plan(self) -> PlanCache:
        if self.pending_plan_path is None:
            return PlanCache()
        p = Path(self.pending_plan_path)
        if not p.exists():
            return PlanCache()
        data = json.loads(p.read_text(encoding="utf-8"))
        return PlanCache.from_dict(data)

    # -- aggregation ----------------------------------------------------------

    def items(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """The ranked unified view (see module docstring for the score)."""
        ledger = self._ledger()
        state = self._state()
        labeled = ledger.labeled()
        kind_stats = calibrate.acceptance_rate(labeled)
        bandit = self._bandit(state)
        items: List[Dict[str, Any]] = []

        for raw in ledger.pending():
            if raw.get("kind") == "ontology_gap":
                source = "gap"
            else:
                source = "ledger"
            items.append({
                "id": f"{source}:{raw['id']}", "source": source,
                "kind": raw.get("kind"), "title": str(raw.get("target", "")),
                "why": raw.get("reason", ""), "proposed": True,
                "pid": raw["id"], "_diff": raw.get("diff"),
                "confidence": raw.get("confidence"),
            })

        # Qualifying gap clusters dispatch has NOT surfaced yet (read-only
        # preview of dispatch's own cluster_gaps; approving one files it
        # through dispatch's own entry point).
        if state:
            summary = dispatch_mod.cluster_gaps(state)
            pending_targets = {p.get("target") for p in ledger.pending()}
            for cand in summary.get("candidates", []):
                if f"gap-cluster:{cand['label']}" in pending_targets:
                    continue  # already a ledger proposal, listed above
                items.append({
                    "id": f"gap:{cand['label']}", "source": "gap",
                    "kind": "ontology_gap", "title": cand["label"],
                    "why": (f"{cand['support']} request(s) fell through; "
                            "not proposed yet"),
                    "proposed": False, "confidence": cand["purity"],
                })

        for op in self._pending_plan().pending():
            tool = str(op.get("tool", "?"))
            items.append({
                "id": f"plan:{tool}", "source": "plan", "kind": _PLAN_KIND,
                "title": f"{tool} = {op.get('value', '?')}",
                "why": "pending plan op (re-validated through the standard "
                       "planner before any apply)",
                "confidence": None, "_op": op,
            })

        if goal:
            items.extend(self._agent_items(goal))

        for item in items:
            stated = item.get("confidence")
            stated_f = 1.0 if stated is None else float(stated)
            kind_mean = float(kind_stats.get(
                item.get("kind") or "",
                {"mean": 0.5})["mean"])  # calibrate's own Beta(1,1) prior
            arm = _arm_for(item)
            arm_mean = _bandit_mean(bandit, arm)
            item["rank_inputs"] = {"stated": round(stated_f, 3),
                                   "kind_mean": round(kind_mean, 3),
                                   "arm_mean": round(arm_mean, 3),
                                   "arm": arm}
            item["score"] = round(stated_f * kind_mean * arm_mean, 3)
            item.pop("_diff", None)
            item.pop("_op", None)
        items.sort(key=lambda i: (-i["score"], SOURCE_ORDER[i["source"]],
                                  i["id"]))
        return items

    # -- dispatch -------------------------------------------------------------

    def decide(self, item_id: str, approve: bool, goal: Optional[str] = None,
               write: bool = False, target=None) -> Dict[str, Any]:
        """Dispatch one decision to the source module's own entry point.
        Returns a plain result dict; never raises for a wrong prefix —
        reports and refuses."""
        if item_id.startswith("ledger:"):
            return self._decide_ledger(int(item_id.split(":", 1)[1]), approve)
        if item_id.startswith("gap:"):
            return self._decide_gap(item_id.split(":", 1)[1], approve)
        if item_id.startswith("plan:"):
            return self._decide_plan(item_id.split(":", 1)[1], approve,
                                     write=write, target=target)
        if item_id.startswith("agent:"):
            return self._decide_agent(item_id.split(":", 1)[1], approve,
                                      goal=goal, write=write)
        return {"id": item_id, "dispatched": False,
                "note": "unknown id prefix (ledger:|gap:|plan:|agent:)"}

    def _decide_ledger(self, pid: int, approve: bool) -> Dict[str, Any]:
        try:
            item = self._ledger().decide(pid, approve)
        except ValueError as exc:
            return {"id": f"ledger:{pid}", "dispatched": False,
                    "note": str(exc)}
        return {"id": f"ledger:{pid}", "dispatched": True,
                "via": "brain.ledger.Ledger.decide",
                "status": item["status"]}

    def _decide_gap(self, suffix: str, approve: bool) -> Dict[str, Any]:
        # Two gap shapes live in the inbox: a SURFACED cluster is a ledger
        # proposal (its id is gap:<pid> — the ledger decides it, exactly
        # like every other proposal); an UNSURFACED qualifying cluster's
        # id is gap:<label> and approve files it through dispatch.py's own
        # single-cluster entry point. A numeric suffix is a pid ONLY when
        # a pending ontology_gap proposal actually carries it — otherwise
        # it is a label (shape tokens can be numeric; pids cannot lie).
        ledger = self._ledger()
        if suffix.isdigit():
            pid = int(suffix)
            for raw in ledger.pending():
                if raw.get("id") == pid and raw.get("kind") == "ontology_gap":
                    result = self._decide_ledger(pid, approve)
                    result["id"] = f"gap:{pid}"
                    return result
        return self._decide_gap_unproposed(ledger, suffix, approve)

    def _decide_gap_unproposed(self, ledger, label: str, approve: bool
                               ) -> Dict[str, Any]:
        if not approve:
            # An unproposed cluster was never filed anywhere: rejecting it
            # writes nothing (an already-surfaced one is a gap:<pid>
            # item and goes through Ledger.decide instead).
            return {"id": f"gap:{label}", "dispatched": True,
                    "via": "(no write surface — cluster was never proposed)",
                    "status": "not-proposed"}
        # Already a pending ledger proposal under its label? Then a
        # duplicate would be stacked — surface that honestly instead.
        target = f"gap-cluster:{label}"
        for raw in ledger.pending():
            if raw.get("target") == target:
                item = ledger.decide(raw["id"], True)
                return {"id": f"gap:{label}", "dispatched": True,
                        "via": "brain.ledger.Ledger.decide",
                        "status": item["status"]}
        result = dispatch_mod.propose_gap_cluster(self._state(), ledger, label)
        if result["candidate"] is None:
            return {"id": f"gap:{label}", "dispatched": False,
                    "note": "no qualifying cluster with that label right now"}
        if result["proposed"] is None:
            return {"id": f"gap:{label}", "dispatched": False,
                    "note": "a proposal for this cluster is already pending"}
        return {"id": f"gap:{label}", "dispatched": True,
                "via": "cortex.dispatch.propose_gap_cluster",
                "ledger_pid": result["proposed"],
                "note": f"filed as ledger proposal #{result['proposed']} — "
                        "decide it there (gap proposals never auto-apply)"}

    def _decide_plan(self, tool: str, approve: bool, write: bool = False,
                     target=None) -> Dict[str, Any]:
        cache = self._pending_plan()
        ops = [op for op in cache.pending()
               if str(op.get("tool", "?")) == tool]
        if not ops:
            return {"id": f"plan:{tool}", "dispatched": False,
                    "note": "no pending plan op for that tool"}
        if not approve:
            # PlanCache.discard is explicit and TOTAL (the cache composes
            # by tool and never drops a single op silently) — say so.
            n = cache.pending_count()
            cache.discard()
            self._persist_plan(cache)
            return {"id": f"plan:{tool}", "dispatched": True,
                    "via": "cortex.plans.PlanCache.discard",
                    "status": "discarded",
                    "note": f"the WHOLE pending plan was discarded "
                            f"({n} op(s)) — the cache has no per-op drop"}
        return self._approve_plan(ops, tool, cache, write=write, target=target)

    def _approve_plan(self, ops: List[Dict[str, Any]], tool: str,
                      cache: PlanCache, write: bool = False,
                      target=None) -> Dict[str, Any]:
        # The cache never bypasses validation: the standard planner sees
        # the ops first, exactly as the chat loop re-validates them.
        if target is None:
            return {"id": f"plan:{tool}", "dispatched": False,
                    "note": "no target file given (--file PATH); nothing "
                            "validated or applied"}
        from assistant.capabilities.settings import applier as settings_applier
        from assistant.capabilities.settings import planner as settings_planner
        try:
            plan = settings_planner.plan(ops, target)
        except settings_planner.PlannerError as exc:
            return {"id": f"plan:{tool}", "dispatched": False,
                    "note": f"planner refused: {exc}"}
        if not write:
            preview = settings_applier.apply(plan, target, write=False)
            return {"id": f"plan:{tool}", "dispatched": True,
                    "via": "settings.planner.plan + settings.applier.apply "
                           "(dry-run)",
                    "status": "preview",
                    "plan": preview,
                    "note": "dry-run only — pass --write to apply through "
                            "the backed-up, undo-bounded applier"}
        applied = settings_applier.apply(plan, target, write=True,
                                         label=f"inbox: plan:{tool}")
        cache.commit()
        self._persist_plan(cache)
        return {"id": f"plan:{tool}", "dispatched": True,
                "via": "settings.applier.apply (write=True, backed up)",
                "status": "applied", "plan": applied,
                "note": "undo: caelestia-assist settings --undo"}

    def _persist_plan(self, cache: PlanCache) -> None:
        if self.pending_plan_path is None:
            return
        p = Path(self.pending_plan_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(cache.to_dict(), indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(p)

    # -- agent source -----------------------------------------------------------

    def _agent_items(self, goal: str) -> List[Dict[str, Any]]:
        from assistant.capabilities.agent.engine import Agent
        projection = Agent().simulate(goal)
        if "error" in projection:
            return [{"id": "agent:error", "source": "agent",
                     "kind": _AGENT_KIND, "title": projection["error"],
                     "why": "agent could not decompose the goal",
                     "confidence": None}]
        items = []
        for step in projection.get("projection", []):
            if not step.get("consent_required"):
                continue
            items.append({
                "id": f"agent:{step['id']}", "source": "agent",
                "kind": _AGENT_KIND, "title": step.get("title", ""),
                "why": f"would {step.get('would', '')} "
                       f"(risk {step.get('risk', '?')})",
                "confidence": None, "_node": step["id"],
            })
        return items

    def _decide_agent(self, node_id: str, approve: bool,
                      goal: Optional[str] = None, write: bool = False
                      ) -> Dict[str, Any]:
        if not goal:
            return {"id": f"agent:{node_id}", "dispatched": False,
                    "note": "agent decisions need the goal text (--goal) — "
                            "the consent queue lives inside one agent run"}
        from assistant.capabilities.agent.engine import Agent
        if not approve or not write:
            # Read-only: show what the run would do around this node.
            projection = Agent().simulate(goal)
            return {"id": f"agent:{node_id}", "dispatched": True,
                    "via": "agent.engine.Agent.simulate (read-only)",
                    "status": "approved-pending-write" if approve
                    else "refused-pending-run",
                    "projection": projection.get("projection", []),
                    "note": "consent is per node and per run; pass --goal "
                            "with --write to run the engine's own gate"}
        # The engine's own consent gate, scripted to THIS node exactly the
        # way the bridge and the tests script it: every other consent node
        # is refused, so nothing beyond it can ride along.
        def consent_fn(node: Dict[str, Any]) -> bool:
            return bool(node.get("id") == node_id)

        agent = Agent(consent_fn=consent_fn)
        result = agent.execute(goal)
        status = "consented-and-run" if approve else "refused"
        return {"id": f"agent:{node_id}", "dispatched": True,
                "via": "agent.engine.Agent.execute (consent_fn gate)",
                "status": status, "summary": result.get("summary", {})}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ledger", default=None,
                   help="proposal ledger JSON (default: the brain CLI's)")
    p.add_argument("--state", default=None,
                   help="brain state JSON (default: brain.state's resolve)")
    p.add_argument("--pending-plan", default=None, dest="pending_plan",
                   help="session payload JSON holding the pending plan "
                        "(the chat round-trips it; the inbox only reads it, "
                        "and rewrites it after discard/commit)")
    p.add_argument("--goal", default=None,
                   help="agent goal text (shows the per-node consent queue)")
    p.add_argument("--file", default=None,
                   help="target shell.json for plan-item applies")


def _build(ledger: Optional[str], state: Optional[str],
           pending_plan: Optional[str]) -> Inbox:
    state_path = Path(state) if state else (
        brain_state.resolve_path() if state is None else None)
    from assistant.capabilities.brain.cli import DEFAULT_LEDGER  # the canonical path
    return Inbox(ledger or DEFAULT_LEDGER,
                 state_path,
                 Path(pending_plan) if pending_plan else None)


def _render(items: List[Dict[str, Any]]) -> List[str]:
    if not items:
        return ["inbox: nothing pending — every source is clear."]
    out = [f"{len(items)} pending decision(s), ranked "
           f"(score = stated x kind posterior x bandit arm mean):"]
    for it in items:
        out.append(f"  [{it['id']}] ({it['score']}) {it['title']}")
        if it.get("why"):
            out.append(f"      why: {it['why']}")
        ri = it.get("rank_inputs") or {}
        out.append(f"      inputs: stated={ri.get('stated')} "
                   f"kind={ri.get('kind_mean')} arm={ri.get('arm_mean')} "
                   f"({ri.get('arm') or 'no arm'})")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="caelestia-assist inbox",
        description="the unified pending-decisions inbox: ledger proposals, "
                    "gap clusters, the pending plan and agent consents in "
                    "one ranked view; approve/reject dispatch to each "
                    "source's own entry point")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "ranked"):
        sp = sub.add_parser(name)
        _add_common(sp)
        sp.add_argument("--json", action="store_true")
    for name in ("approve", "reject"):
        sp = sub.add_parser(name)
        sp.add_argument("item_id")
        _add_common(sp)
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--write", action="store_true",
                        help="plan/agent approvals: allow the real "
                             "(backed-up, gated) write; default is the "
                             "dry-run preview")
    args = ap.parse_args(argv)
    inbox = _build(args.ledger, args.state, args.pending_plan)
    if args.cmd in ("list", "ranked"):
        items = inbox.items(goal=args.goal)
        lines = _render(items)
        if args.json:
            print(json.dumps({"items": items}, indent=2, default=str))
        else:
            print("\n".join(lines))
        return 0
    approve = args.cmd == "approve"
    result = inbox.decide(args.item_id, approve, goal=args.goal,
                          write=args.write, target=args.file)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("dispatched") else 1
    print(f"{args.cmd} {args.item_id}: "
          f"{'dispatched' if result.get('dispatched') else 'NOT dispatched'}")
    for key in ("via", "status", "ledger_pid", "note"):
        if result.get(key):
            print(f"  {key}: {result[key]}")
    return 0 if result.get("dispatched") else 1

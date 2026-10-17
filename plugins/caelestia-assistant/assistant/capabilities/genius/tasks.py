"""genius.tasks — decompose any goal into a schedulable, evidence-carrying plan.

A human employee handed "fix the flaky tests" does not start coding:
they decompose, sequence, estimate, and flag blockers. This module does
the same, deterministically:

  * HTN-lite decomposition: 12 goal archetypes (build/fix/learn/
    research/write/migrate/clean/organize/plan/deploy/audit/design),
    each with a method — an ordered subtask template with dependencies,
    default effort estimates, and a `why` for every step
  * effort priors per step kind, with a caller-overridable table so the
    brain's Bayesian duration model can feed real numbers in later
  * topological scheduling: cycles are refused loudly, not silently
    re-ordered
  * critical path (CPM) + parallel-early start computation
  * today-fit packing: which subset fits N minutes, highest priority
    density first (a 0/1 knapsack by value density)
  * progress + staleness: which tasks are blocked, which have been
    sitting too long (exponential age urgency, like the memory decay)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["decompose", "schedule", "critical_path", "fit_today",
           "progress_report", "estimate_goal", "GOAL_ARCHETYPES"]


# ---------------------------------------------------------------------------
# HTN-lite: goal archetypes and their methods
# ---------------------------------------------------------------------------

_GOAL_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "build": {
        "triggers": ["build", "create", "make", "implement", "develop", "write app", "prototype"],
        "steps": [
            ("clarify", "define the done condition and constraints", None),
            ("design", "sketch the structure and choose the approach", ["clarify"]),
            ("scaffold", "set up the skeleton with tests failing red", ["design"]),
            ("implement", "fill in the core path end to end", ["scaffold"]),
            ("harden", "edge cases, errors, and the unhappy paths", ["implement"]),
            ("verify", "test suite green + manual smoke test", ["harden"]),
            ("document", "README / usage notes while it is fresh", ["verify"]),
        ],
        "effort_min": {"clarify": 20, "design": 45, "scaffold": 60, "implement": 150,
                        "harden": 90, "verify": 40, "document": 25},
    },
    "fix": {
        "triggers": ["fix", "bug", "debug", "broken", "crash", "repair", "not working"],
        "steps": [
            ("reproduce", "make it fail on demand — the first real step", None),
            ("isolate", "bisect: shrink the failing case to essentials", ["reproduce"]),
            ("diagnose", "name the root cause; write it down in one sentence", ["isolate"]),
            ("patch", "smallest change that fixes the cause, not the symptom", ["diagnose"]),
            ("regression", "add the failing case as a passing test", ["patch"]),
            ("verify", "run the surrounding suite; check for cousins", ["regression"]),
        ],
        "effort_min": {"reproduce": 25, "isolate": 35, "diagnose": 30, "patch": 45,
                        "regression": 20, "verify": 20},
    },
    "learn": {
        "triggers": ["learn", "study", "understand", "explore", "get into"],
        "steps": [
            ("survey", "map the field: vocabulary, landmarks, good sources", None),
            ("define_goal", "write the specific thing you want to be able to DO", None),
            ("core", "learn the 20% that explains 80% — deliberately small", ["survey", "define_goal"]),
            ("practice", "spaced problems, not marathons — 3 sessions beat 1", ["core"]),
            ("build", "make one tiny real thing with it", ["practice"]),
            ("review", "schedule spaced reviews; knowledge decays without them", ["build"]),
        ],
        "effort_min": {"survey": 45, "define_goal": 15, "core": 90, "practice": 120,
                        "build": 90, "review": 15},
    },
    "research": {
        "triggers": ["research", "investigate", "compare", "evaluate", "find out", "which"],
        "steps": [
            ("question", "write the question so it can be answered in principle", None),
            ("sources", "gather candidates; note credibility per source", None),
            ("extract", "pull claims with their evidence into one table", ["question", "sources"]),
            ("synthesize", "answer the question; mark what remains unknown", ["extract"]),
            ("decide", "convert the answer into a next action", ["synthesize"]),
        ],
        "effort_min": {"question": 15, "sources": 45, "extract": 60, "synthesize": 45, "decide": 15},
    },
    "write": {
        "triggers": ["write", "draft", "document", "blog", "report", "notes on"],
        "steps": [
            ("audience", "one sentence: who reads this and what changes for them", None),
            ("outline", "skeleton of claims, in order of persuasion", ["audience"]),
            ("draft", "ugly first draft — momentum over polish", ["outline"]),
            ("revise", "cut 20%, fix the flow, one idea per paragraph", ["draft"]),
            ("proof", "read aloud once; fix what trips", ["revise"]),
        ],
        "effort_min": {"audience": 10, "outline": 30, "draft": 90, "revise": 45, "proof": 20},
    },
    "migrate": {
        "triggers": ["migrate", "port", "move", "upgrade", "transition"],
        "steps": [
            ("inventory", "list everything that moves; nothing undocumented ships", None),
            ("compatibility", "what breaks, what is already fine, what improves", ["inventory"]),
            ("pilot", "move one small real piece first and live with it", ["compatibility"]),
            ("execute", "move the rest in reversible chunks", ["pilot"]),
            ("verify", "old references gone, behavior equivalent", ["execute"]),
            ("rollback_plan", "write the un-do before you need it", None),
        ],
        "effort_min": {"inventory": 40, "compatibility": 50, "pilot": 60, "execute": 120,
                        "verify": 40, "rollback_plan": 15},
    },
    "clean": {
        "triggers": ["clean", "cleanup", "tidy", "prune", "declutter"],
        "steps": [
            ("define_keep", "decide what keep means — before looking at anything", None),
            ("sort", "bucket everything: keep / toss / unsure (no deciding yet)", ["define_keep"]),
            ("decide_unsure", "resolve the unsure bucket item by item", ["sort"]),
            ("remove", "remove the toss bucket — recycle where possible", ["decide_unsure"]),
            ("maintain", "one habit that prevents the pile from returning", ["remove"]),
        ],
        "effort_min": {"define_keep": 10, "sort": 60, "decide_unsure": 45, "remove": 30, "maintain": 5},
    },
    "organize": {
        "triggers": ["organize", "structure", "categorize", "systematize", "sort out"],
        "steps": [
            ("take_inventory", "what exists, where, in what quantity", None),
            ("choose_axes", "pick the 1-2 dimensions you actually retrieve by", ["take_inventory"]),
            ("place", "every item gets one obvious home", ["choose_axes"]),
            ("label", "name things for future-you, not for taxonomy", ["place"]),
            ("prune", "delete the categories nothing ever lands in", ["label"]),
        ],
        "effort_min": {"take_inventory": 30, "choose_axes": 20, "place": 60, "label": 30, "prune": 10},
    },
    "plan": {
        "triggers": ["plan", "roadmap", "strategy", "figure out how"],
        "steps": [
            ("outcome", "the measurable end state, in one sentence", None),
            ("constraints", "time, money, skills, dependencies you cannot change", None),
            ("options", "three genuinely different routes, not one route thrice", ["outcome", "constraints"]),
            ("sequence", "order the chosen route; find the critical path", ["options"]),
            ("commit", "put the first three steps on a calendar", ["sequence"]),
        ],
        "effort_min": {"outcome": 15, "constraints": 20, "options": 45, "sequence": 30, "commit": 10},
    },
    "deploy": {
        "triggers": ["deploy", "release", "ship", "publish", "roll out"],
        "steps": [
            ("preflight", "checklist: tests, secrets, rollback path", None),
            ("stage", "deploy to a staging target identical where it matters", ["preflight"]),
            ("verify_stage", "smoke the staged thing like a user would", ["stage"]),
            ("prod", "deploy to production during a calm window", ["verify_stage"]),
            ("watch", "monitor the first 30 minutes; keep hands on keyboard", ["prod"]),
        ],
        "effort_min": {"preflight": 20, "stage": 30, "verify_stage": 25, "prod": 20, "watch": 30},
    },
    "audit": {
        "triggers": ["audit", "review", "assess", "checkup", "health check"],
        "steps": [
            ("scope", "what is in and out of the audit, and for whom", None),
            ("gather", "collect the evidence — logs, metrics, files", ["scope"]),
            ("findings", "facts with severity, each with its evidence attached", ["gather"]),
            ("recommend", "actions ordered by impact-over-effort", ["findings"]),
        ],
        "effort_min": {"scope": 15, "gather": 45, "findings": 60, "recommend": 30},
    },
    "design": {
        "triggers": ["design", "redesign", "mock", "layout", "ux"],
        "steps": [
            ("users", "who, doing what, in which context", None),
            ("constraints", "platform, tokens, performance budget", ["users"]),
            ("explore", "3 directions at low fidelity before polishing any", ["constraints"]),
            ("refine", "detail the chosen direction; name the trade-offs", ["explore"]),
            ("validate", "put it in front of one real user; watch silently", ["refine"]),
        ],
        "effort_min": {"users": 20, "constraints": 20, "explore": 60, "refine": 75, "validate": 40},
    },
}

GOAL_ARCHETYPES = {k: [t for t in v["triggers"]] for k, v in _GOAL_ARCHETYPES.items()}


def _match_archetype(goal: str) -> Tuple[Optional[str], List[str]]:
    g = goal.lower()
    best, hits = None, []
    for name, spec in _GOAL_ARCHETYPES.items():
        for trig in spec["triggers"]:
            if re_search_word(trig, g):
                if name not in [b for b in hits]:
                    hits.append(name)
                    if best is None:
                        best = name
    return best, hits


def re_search_word(needle: str, haystack: str) -> bool:
    import re as _re
    return bool(_re.search(r"(?<![a-z])" + _re.escape(needle) + r"(?![a-z])",
                           haystack))


def decompose(goal: str, effort_overrides: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Break a goal into an ordered, dependency-linked subtask list."""
    best, alternatives = _match_archetype(goal)
    if best is None:
        return {"goal": goal, "archetype": None,
                "verdict": "NO_DECOMPOSITION",
                "question": f"which kind of work is this: {', '.join(sorted(GOAL_ARCHETYPES))}?",
                "alternatives": []}
    spec = _GOAL_ARCHETYPES[best]
    effort = dict(spec["effort_min"])
    effort.update({k: int(v) for k, v in (effort_overrides or {}).items()})
    steps = []
    for name, description, deps in spec["steps"]:
        steps.append({"id": name, "title": f"{name}: {description}",
                      "depends_on": deps or [], "effort_min": effort[name],
                      "why": f"{best} work reliably needs an explicit {name} step"})
    total = sum(s["effort_min"] for s in steps)
    return {"goal": goal, "archetype": best,
            "matched_on": spec["triggers"][:3],
            "alternatives": [a for a in alternatives if a != best],
            "verdict": "DECOMPOSED",
            "steps": steps,
            "total_effort_min": total,
            "total_effort_human": _hm(total),
            "critical_path": critical_path(steps)["critical_path"],
            "parallelizable": critical_path(steps)["slack_total_min"],
            "note": "a template decomposition — edit the steps, do not worship them"}


def schedule(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Topological order + earliest starts. Refuses cycles loudly."""
    by_id = {s["id"]: s for s in steps}
    ids = list(by_id)
    order: List[str] = []
    state: Dict[str, int] = {}  # 0=unvisited 1=visiting 2=done

    def visit(node: str, stack: List[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = " -> ".join(stack + [node])
            raise ValueError(f"dependency cycle detected: {cycle}")
        state[node] = 1
        for dep in by_id[node].get("depends_on", []):
            if dep not in by_id:
                raise ValueError(f"{node!r} depends on unknown step {dep!r}")
            visit(dep, stack + [node])
        state[node] = 2
        order.append(node)

    for i in ids:
        visit(i, [])
    starts: Dict[str, int] = {}
    for node in order:
        deps = by_id[node].get("depends_on", [])
        starts[node] = max([starts[d] + by_id[d]["effort_min"] for d in deps], default=0)
    return {"order": order,
            "earliest_start_min": {k: starts[k] for k in order},
            "makespan_min": max((starts[k] + by_id[k]["effort_min"] for k in order),
                               default=0)}


def critical_path(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """CPM: longest dependency chain (effort-summed) and total slack."""
    by_id = {s["id"]: s for s in steps}
    sched = schedule(list(steps))
    ends = {k: sched["earliest_start_min"][k] + by_id[k]["effort_min"] for k in by_id}
    makespan = sched["makespan_min"]
    # walk backward greedily from the latest end
    end_node = max(ends, key=lambda k: ends[k]) if ends else None
    path: List[str] = []
    node = end_node
    while node is not None:
        path.append(node)
        deps = by_id[node].get("depends_on", [])
        node = max(deps, key=lambda d: ends[d]) if deps else None
    path.reverse()
    path_min = sum(by_id[p]["effort_min"] for p in path)
    return {"critical_path": path, "critical_path_min": path_min,
            "makespan_min": makespan,
            "slack_total_min": makespan - path_min,
            "note": "the critical path is what delays the goal; everything else has slack"}


def fit_today(steps: Sequence[Dict[str, Any]], minutes: int,
              priorities: Optional[Dict[str, float]] = None,
              done_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
    """0/1 knapsack by priority density that respects the dependency DAG:
    only steps whose dependencies are already done (or also chosen now,
    in order) can be scheduled; each pick may unblock the next."""
    if minutes <= 0:
        raise ValueError("minutes must be > 0")
    priorities = priorities or {}
    done = set(done_ids or [])
    by_id = {s["id"]: s for s in steps}

    def unblocked(sid: str, chosen: List[str]) -> bool:
        deps = by_id[sid].get("depends_on", [])
        return all(d in done or d in chosen for d in deps)

    items = {s["id"]: {"id": s["id"], "effort_min": int(s.get("effort_min", 30)),
                       "priority": float(priorities.get(s["id"], 0.5))}
             for s in steps}
    for it in items.values():
        it["density"] = it["priority"] / max(1, it["effort_min"])
    chosen: List[str] = []
    left = minutes
    while True:
        ready = [it for sid, it in items.items()
                 if sid not in chosen and unblocked(sid, chosen)
                 and it["effort_min"] <= left]
        if not ready:
            break
        ready.sort(key=lambda it: -it["density"])
        pick = ready[0]
        chosen.append(pick["id"])
        left -= pick["effort_min"]
    total_pr = sum(items[c]["priority"] for c in chosen)
    return {"budget_min": minutes,
            "chosen": chosen,
            "chosen_effort_min": minutes - left,
            "leftover_min": left,
            "priority_captured": round(total_pr, 3),
            "note": "dependency-aware greedy density knapsack; a human sanity-check beats the optimum"}


def progress_report(tasks: Sequence[Dict[str, Any]], now_day: float = 0.0) -> Dict[str, Any]:
    """Blocked / stale / urgent triage over a task list.

    Task shape: {"id", "depends_on" (list), "created_day", "done" (bool)}
    """
    by_id = {t["id"]: t for t in tasks}
    blocked = []
    for t in tasks:
        if t.get("done"):
            continue
        missing = [d for d in t.get("depends_on", []) if d in by_id and not by_id[d].get("done")]
        if missing:
            blocked.append({"id": t["id"], "waiting_on": missing})
    stale = []
    for t in tasks:
        if t.get("done"):
            continue
        age = now_day - float(t.get("created_day", now_day))
        if age > 3:
            urgency = 1 - math.exp(-age / 7)  # saturating urgency
            stale.append({"id": t["id"], "age_days": round(age, 1),
                          "urgency": round(urgency, 3)})
    stale.sort(key=lambda s: -s["urgency"])
    done_n = sum(1 for t in tasks if t.get("done"))
    return {"total": len(tasks), "done": done_n,
            "open": len(tasks) - done_n,
            "blocked": blocked,
            "stale": stale[:8],
            "next_action": (stale[0]["id"] if stale else
                            blocked[0]["id"] if blocked else
                            (next((t["id"] for t in tasks if not t.get("done")), None))),
            "note": "stale tasks decay toward urgent, never silently dropped"}


def estimate_goal(goal: str) -> Dict[str, Any]:
    d = decompose(goal)
    if d["verdict"] != "DECOMPOSED":
        return d
    steps = d["steps"]
    sched = schedule(steps)
    cp = critical_path(steps)
    return {"goal": goal, "archetype": d["archetype"],
            "total_effort_min": d["total_effort_min"],
            "total_effort_human": d["total_effort_human"],
            "calendar_min_if_serial": sched["makespan_min"],
            "critical_path_min": cp["critical_path_min"],
            "steps": len(steps)}


def _hm(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"

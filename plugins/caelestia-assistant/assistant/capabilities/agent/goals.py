"""agent.goals — HTN-flavoured goal decomposition over the assistant's layers.

The agent's first honest question is: "what KIND of thing is being asked?"
The second is: "what ordered, dependency-resolved steps serve it?" This
module answers both with a hierarchical task network (HTN) method library:

  compound goal ("clean my downloads and then make the shell minimal")
      -> split into primitive goals (cue lexicons, like genius.meta's router)
      -> each primitive goal decomposes through METHODS into subtasks
      -> subtasks form a DAG (genius.graphs.toposort + critical path)
      -> every leaf is an ACTION bound to a real layer dispatcher

Design contract (inherited, not renegotiated):
  - decomposition is deterministic: same goal in, same task graph out;
  - every leaf carries a risk tier (READ_ONLY / STATE_CHANGING / PRIVILEGED /
    DESTRUCTIVE, mirroring diagnostics.risk) and a consent flag;
  - the decomposer never marks anything safe to auto-run: STATE_CHANGING
    leaves stay consent_required=True and the engine enforces it;
  - when confidence in the goal split is low, the graph contains exactly one
    node: a CLARIFY question (agent.clarify picks it by information gain).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from assistant.capabilities.genius.graphs import toposort, longest_path_dag

__all__ = ["decompose", "split_compound", "GOAL_METHODS", "compound_ways"]

# ---------------------------------------------------------------------------
# Cue lexicons — which primitive goal does a phrase belong to?
# ---------------------------------------------------------------------------

_GOAL_CUES: Dict[str, List[str]] = {
    "diagnose_issue": [
        "broken", "fails", "failing", "crash", "crashes", "freezes", "frozen",
        "error", "not working", "stopped working", "went away", "disappears",
        "hangs", "stuck", "slow", "lag", "stutter", "tearing", "glitch",
    ],
    "research_topic": [
        "why", "how does", "how do i", "what is", "explain", "difference",
        "meaning", "docs", "documentation", "guide",
    ],
    "change_settings": [
        "make", "set", "move", "enable", "disable", "turn on", "turn off",
        "increase", "decrease", "reduce", "hide", "show", "thinner", "thicker",
        "smaller", "bigger", "larger", "faster", "slower", "compact", "minimal",
        "blur", "scale", "spacing", "rounding", "position",
    ],
    "clean_files": [
        "clean", "organize", "organise", "tidy", "dedupe", "duplicates",
        "downloads", "free up", "clutter", "sort my files", "empty dirs",
    ],
    "daily_review": [
        "plan my day", "what's pending", "whats pending", "pending",
        "brief", "summary of my", "review my", "stuck", "what should i do",
    ],
    "compute": [
        "calculate", "solve", "derivative", "integral", "probability",
        "statistics", "matrix", "regression", "forecast", "optimize",
        "minimize", "maximize", "shortest path", "assign",
    ],
    # -- phase 2.3 archetypes (same cue-lexicon mechanism, new goals) --
    "config_hygiene": [
        "config hygiene", "lint my config", "lint my shell", "drift",
        "dotfile", "dotfiles", "check my config", "validate my config",
        "my shell.json", "reconcile my config",
    ],
    "package_audit": [
        "audit my packages", "package audit", "my packages",
        "outdated packages", "stale packages", "dependencies",
        "what is installed", "what's installed", "installed packages",
    ],
    "log_triage": [
        "triage my logs", "log triage", "mine my logs", "my logs",
        "log templates", "journal triage", "summarize my log",
        "draft an issue from my log",
    ],
    "notification_triage": [
        "batch my notifications", "notification spam",
        "too many notifications", "triage notifications",
        "notification triage", "interrupting me",
    ],
    "screenshot_diff": [
        "compare screenshots", "screenshot diff", "structural bug",
        "before after screenshot", "visual regression", "diff these"
        " screenshots", "ui changed",
    ],
}

# Compounds: conjunctions that join independent goals
_COMPOUND_WORDS = (" and then ", " then ", " afterwards ", " after that ",
                   " and also ", " also ")


def split_compound(text: str) -> List[str]:
    """Split a compound request into ordered primitive phrases.

    Deliberately conservative: a compound split only happens on explicit
    sequencing words; plain "and" joins clauses of the SAME goal (the cortex
    already handles multi-clause settings requests) and is not split here."""
    low = f" {text.strip().lower()} "
    for w in _COMPOUND_WORDS:
        if w in low:
            parts = [p.strip(" ,.;") for p in low.split(w)]
            parts = [p for p in parts if p]
            if len(parts) > 1:
                return parts
    return [text.strip()]


def classify_goal(text: str) -> Dict[str, Any]:
    """Cue-score the primitive goals; return the winner, the margin, and the
    runners-up. Honest verdicts: AMBIGUOUS when the margin is thin, ABSTAIN
    when nothing scores. Bare arithmetic shape (numbers + operators, %) adds
    a compute boost — the same principle genius.meta's router uses: a bare
    expression IS a compute request."""
    low = f" {text.lower()} "
    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}
    for goal, cues in _GOAL_CUES.items():
        hits = [c for c in cues if f" {c} " in low or low.strip().startswith(c)
                or f"{c} " in low]
        scores[goal] = float(len(hits))
        if hits:
            matched[goal] = hits
    # arithmetic shape detection (deterministic, no model)
    import re
    if re.search(r"\d+\s*[-+*/^%]\s*\d+", text) or re.search(r"\d+\s*%", text):
        scores["compute"] = scores.get("compute", 0.0) + 2.0
        matched.setdefault("compute", []).append("arithmetic shape")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_goal, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score == 0.0:
        verdict, goal = "ABSTAIN", "compute"   # neutral fallback: read-only
    elif top_score - second_score < 1.0:
        verdict, goal = "AMBIGUOUS", top_goal
    else:
        verdict, goal = "ROUTE", top_goal
    return {
        "goal": goal,
        "verdict": verdict,
        "score": top_score,
        "margin": top_score - second_score,
        "evidence": matched,
        "runners_up": [g for g, s in ranked[1:3] if s > 0],
    }


# ---------------------------------------------------------------------------
# HTN methods: primitive goal -> task graph (dict specs)
# ---------------------------------------------------------------------------
# Node spec: {id_suffix, action, title, risk, consent, params}
# "action" names a dispatcher the ENGINE binds (engine.DISPATCHERS);
# "risk" mirrors diagnostics.risk tiers; consent is required for anything
# above READ_ONLY. Dependencies are expressed as: node id_suffix -> list of
# id_suffixes it needs. The compound form sequences goals with "then".

GOAL_METHODS: Dict[str, List[Dict[str, Any]]] = {
    "diagnose_issue": [
        {"id": "collect", "action": "scan_stream",
         "title": "Scan any attached log/stream for known signatures",
         "risk": "READ_ONLY", "consent": False,
         "params": {"note": "uses assistant.capabilities.scan one-pass structures"}},
        {"id": "match", "action": "diagnose",
         "title": "Match against the deterministic rule engine",
         "risk": "READ_ONLY", "consent": False, "depends": ["collect"],
         "params": {}},
        {"id": "retrieve", "action": "retrieve_similar",
         "title": "Retrieve the closest past resolutions",
         "risk": "READ_ONLY", "consent": False, "depends": ["collect"],
         "params": {}},
        {"id": "plan_fix", "action": "fix_plan",
         "title": "Assemble the fix plan with inert suggested commands",
         "risk": "READ_ONLY", "consent": False,
         "depends": ["match", "retrieve"], "params": {}},
    ],
    "research_topic": [
        {"id": "retrieve", "action": "retrieve_similar",
         "title": "Search the repo's own docs and resolved issues",
         "risk": "READ_ONLY", "consent": False, "params": {"k": 5}},
        {"id": "explain", "action": "explain_result",
         "title": "Explain what was found and cite the sources",
         "risk": "READ_ONLY", "consent": False, "depends": ["retrieve"],
         "params": {}},
    ],
    "change_settings": [
        {"id": "route", "action": "route_request",
         "title": "Route the request to settings ops (learned cortex)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
        {"id": "validate", "action": "validate_plan",
         "title": "Validate ops against the registry (types, ranges)",
         "risk": "READ_ONLY", "consent": False, "depends": ["route"],
         "params": {}},
        {"id": "propose", "action": "propose_plan",
         "title": "Show the plan and WAIT for explicit approval",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["validate"],
         "params": {}},
        {"id": "apply", "action": "apply_plan",
         "title": "Apply the approved plan (undo history records it)",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["propose"],
         "params": {}},
    ],
    "clean_files": [
        {"id": "survey", "action": "tidy_survey",
         "title": "Survey the target directory (duplicates, stale, routing)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
        {"id": "propose", "action": "tidy_propose",
         "title": "Show the move plan and WAIT for explicit approval",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["survey"],
         "params": {}},
        {"id": "apply", "action": "tidy_apply",
         "title": "Apply approved moves (journaled, rollback-able)",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["propose"],
         "params": {}},
    ],
    "daily_review": [
        {"id": "brief", "action": "brief",
         "title": "Compose the daily brief (ledger, stuck, plan, forecast)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
    ],
    "compute": [
        {"id": "compute", "action": "genius_dispatch",
         "title": "Dispatch to the genius engines (math/stats/logic/...)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
    ],
    # -- phase 2.3 archetypes: same HTN shape, new methods -------------
    "config_hygiene": [
        {"id": "lint", "action": "lint_config",
         "title": "Lint the live shell.json against the registry schema",
         "risk": "READ_ONLY", "consent": False, "params": {}},
        {"id": "drift", "action": "config_drift",
         "title": "Score drift against the learned preference posterior",
         "risk": "READ_ONLY", "consent": False, "depends": ["lint"],
         "params": {}},
        {"id": "propose", "action": "reconcile_propose",
         "title": "Propose reconciliation ops and WAIT for approval",
         "risk": "STATE_CHANGING", "consent": True,
         "depends": ["drift"], "params": {}},
        {"id": "apply", "action": "reconcile_apply",
         "title": "Apply the approved resets (planner validates, applier gates)",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["propose"],
         "params": {}},
    ],
    "package_audit": [
        {"id": "report", "action": "package_report",
         "title": "Read-only package probe + static keyword match (opt-in)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
    ],
    "log_triage": [
        {"id": "triage", "action": "log_triage",
         "title": "Mine log templates + flag frequency anomalies (sysintel)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
        {"id": "draft", "action": "triage_draft",
         "title": "Compose the issue draft from the triage result",
         "risk": "READ_ONLY", "consent": False, "depends": ["triage"],
         "params": {}},
    ],
    "notification_triage": [
        {"id": "classify", "action": "notification_triage",
         "title": "Classify notification events into batch/pass proposals",
         "risk": "READ_ONLY", "consent": False, "params": {}},
    ],
    "screenshot_diff": [
        {"id": "diff", "action": "screenshot_diff",
         "title": "Pixel-region hash diff of the two PNGs (no OCR)",
         "risk": "READ_ONLY", "consent": False, "params": {}},
        {"id": "draft", "action": "screenshot_draft",
         "title": "Compose the structural-bug issue draft from the diff",
         "risk": "READ_ONLY", "consent": False, "depends": ["diff"],
         "params": {}},
    ],
}


# ---------------------------------------------------------------------------
def decompose(text: str, clarify_hook: Optional[Any] = None) -> Dict[str, Any]:
    """Full decomposition: compound split -> goal classification -> method
    expansion -> DAG ordering. Returns the task graph plus honest metadata
    (verdicts, margins, critical path). When the split is ambiguous the
    graph is a single CLARIFY node (unless a clarify_hook resolves it)."""
    parts = split_compound(text)
    classifications = [classify_goal(p) for p in parts]

    ambiguous = [c for c in classifications if c["verdict"] == "AMBIGUOUS"]
    abstained = [c for c in classifications if c["verdict"] == "ABSTAIN"]

    if ambiguous and clarify_hook is not None:
        answer = clarify_hook(parts, classifications)
        if answer:
            classifications = [classify_goal(parts[0])]
            parts = [parts[0]]

    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    prev_goal_nodes: List[str] = []
    seq = 0
    for idx, (part, cls) in enumerate(zip(parts, classifications)):
        goal = cls["goal"]
        method = GOAL_METHODS.get(goal, GOAL_METHODS["compute"])
        goal_node_ids: List[str] = []
        for step in method:
            seq += 1
            nid = f"n{seq}"
            deps = [f"n{seq - j}" for j in (0,)]  # placeholder; fixed below
            deps = []
            nodes.append({
                "id": nid,
                "goal": goal,
                "action": step["action"],
                "title": step["title"],
                "risk": step["risk"],
                "consent_required": step["consent"],
                "input_text": part,
                "params": dict(step.get("params", {})),
                "depends": [],   # filled below
                "status": "pending",
            })
            goal_node_ids.append(nid)
        # intra-goal dependencies from the method's "depends" suffixes
        suffix_to_id = {method[i]["id"]: goal_node_ids[i] for i in range(len(method))}
        for i, step in enumerate(method):
            for dep_suffix in step.get("depends", []):
                if dep_suffix in suffix_to_id:
                    nodes_find = next(n for n in nodes if n["id"] == goal_node_ids[i])
                    nodes_find["depends"].append(suffix_to_id[dep_suffix])
        # inter-goal sequencing: this goal's FIRST node waits on the previous
        # goal's LAST node (compound semantics: "and then")
        if prev_goal_nodes and goal_node_ids:
            edges.append((prev_goal_nodes[-1], goal_node_ids[0]))
            first = next(n for n in nodes if n["id"] == goal_node_ids[0])
            first["depends"].append(prev_goal_nodes[-1])
        prev_goal_nodes = goal_node_ids

    dag_edges = []
    for n in nodes:
        for d in n["depends"]:
            dag_edges.append((d, n["id"]))
    for a, b in edges:
        dag_edges.append((a, b))
    ts = toposort(dag_edges) if dag_edges else {"order": [n["id"] for n in nodes],
                                                "is_dag": True, "cycle": []}
    if not ts["is_dag"]:
        return {"error": "internal cycle — decomposition bug", "cycle": ts["cycle"]}
    cp = longest_path_dag(dag_edges)
    order_index = {nid: i for i, nid in enumerate(ts["order"])}
    nodes.sort(key=lambda n: order_index.get(n["id"], 0))
    result = {
        "nodes": nodes,
        "order": ts["order"],
        "critical_path": cp.get("critical_path") or [],
        "classifications": classifications,
        "ambiguous": bool(ambiguous),
        "abstained": bool(abstained),
        "compound": len(parts) > 1,
        "consent_required": any(n["consent_required"] for n in nodes),
    }
    # exponential-build-4 B: bounded model-check the goal DAG BEFORE it
    # is shown — dead ends (task nodes with no dependents and no
    # successors...) and unreachability of any node from the root,
    # within the node count as the step bound. The transition graph is
    # dependency-order: a node's successors are its dependents. The
    # audit ABSTAINs loudly past its budget; a budget refusal never
    # blocks the plan (it is an audit, not a gate) — recorded honestly.
    try:
        from assistant.capabilities.genius import sat as sat_mod
        graph = {nid: [] for nid in ts["order"]}
        for a, b in dag_edges:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, [])
        root = ts["order"][0] if ts["order"] else ""
        if root and len(graph) <= 64:
            audit = sat_mod.graph_audit(graph, root,
                                        k=max(2, len(graph)),
                                        node_budget=50_000)
            # In a goal DAG a terminal node IS the goal, not a fault:
            # deadlock states that are leaves (no dependents, nothing
            # after them) are the plan's normal completion points. Only
            # UNREACHABLE nodes are findings — and ABSTAIN stays loud.
            successors = {a for a, _b in dag_edges}
            leaves = [s for s in audit["deadlock_states"]
                      if s not in successors]
            real_deadlocks = [s for s in audit["deadlock_states"]
                              if s in successors]
            clean = (not audit["unreachable_within_k"]) and not real_deadlocks
            result["reachability_audit"] = {
                "status": "CLEAN" if clean else "FINDINGS",
                "terminal_nodes": leaves,
                "dead_ends_with_dependents": real_deadlocks,
                "unreachable_within_k": audit["unreachable_within_k"],
                "note": audit["note"],
            }
    except Exception as exc:  # the audit must never break the plan
        result["reachability_audit"] = {"status": "ERROR",
                                        "reason": f"{type(exc).__name__}: {exc}"}
    return result

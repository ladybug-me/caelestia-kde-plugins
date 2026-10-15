"""assistant.capabilities.settings.causal — causal attribution over the curated DAG
(exponential-build-5 F14).

Two questions, both answered ONLY from the shipped, citation-backed
consequences table plus the live state the caller supplies read-only:

  * ``why_chain(path, current)`` — BACKWARD: which armed interactions
    explain the current value of ``path``? Multi-hop (bounded, cycle-
    safe), every hop carrying the edge's citation, confidence and
    arming condition. A trigger value no curated edge explains
    terminates the chain as DIRECT ("set directly — see undo history"),
    never as a guess. An edge whose ``requires`` live value is missing
    is reported UNEVALUATED — the abstention the design demands.

  * ``counterfactual(tool, value, current)`` — FORWARD: "if I set X,
    what happens?" — the existing ``consequences.project`` projection
    (reuse, not re-implementation) EXTENDED with the honesty the what-if
    view never had: every curated edge that COULD have fired but could
    not be evaluated (its ``requires`` path missing from the live
    state) is listed as UNEVALUATED instead of silently staying quiet.

The honesty rules are the table's own: the causal universe is EXACTLY
the curated edges (bounded, growable by reviewed diffs); chains are
explanations OF CITED CODE, not causes in nature; and every unknown
needed value is an abstention, never a default.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import consequences as _cons
from assistant.adapters.caelestia.registry import tool_by_name, tool_by_path

__all__ = ["why_chain", "counterfactual", "unevaluated_edges",
           "render_chain", "render_counterfactual", "MAX_CHAIN_DEPTH"]

MAX_CHAIN_DEPTH = 4  # bounded walk; deeper chains are refused, not faked


def _live(current: Optional[Dict[str, Any]], path: str) -> Any:
    """Read a dotted path from the NESTED live dict (None = unknown)."""
    if not isinstance(current, dict):
        return None
    node: Any = current
    for piece in path.split("."):
        if isinstance(node, dict):
            node = node.get(piece)
        else:
            return None
    return node


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _edge_state(edge: Dict[str, Any], current: Optional[Dict[str, Any]]
                ) -> Dict[str, Any]:
    """Classify one edge against live state for BACKWARD explanation.

    Returns {"status": ARMED | NOT_ARMED | UNEVALUATED, ...}:
      ARMED       — the trigger's live value satisfies the when-condition
                    AND any requires precondition holds AND (when the
                    edge sets a value) the effect's live value matches
                    what the edge induces — this edge genuinely explains
                    the effect's current value.
      NOT_ARMED   — a known value contradicts the condition.
      UNEVALUATED — a needed live value is missing (the abstention).
    """
    when = edge.get("when") or {}
    trigger = edge["trigger_path"]
    live_trigger = _live(current, trigger)
    reason_unknown: Optional[str] = None
    if "op_is" in when:
        if live_trigger is None:
            reason_unknown = f"live value of {trigger} unknown"
        elif live_trigger is not when["op_is"]:
            return {"status": "NOT_ARMED"}
    if "op_below" in when and reason_unknown is None:
        if live_trigger is None:
            reason_unknown = f"live value of {trigger} unknown"
        elif not (_is_number(live_trigger) and live_trigger < when["op_below"]):
            return {"status": "NOT_ARMED"}
    requires = when.get("requires")
    if requires:
        live_req = _live(current, requires["path"])
        if live_req is None:
            reason_unknown = (reason_unknown or
                              f"live value of {requires['path']} unknown")
        elif live_req != requires.get("equals"):
            return {"status": "NOT_ARMED"}
    if reason_unknown is not None:
        return {"status": "UNEVALUATED", "reason": reason_unknown}
    # the edge's induced value must match the effect's live value, else
    # it explains a PAST state, not the current one
    effect_value = edge.get("effect_value")
    if effect_value is not None:
        live_effect = _live(current, edge["effect_path"])
        if live_effect is None:
            return {"status": "UNEVALUATED",
                    "reason": f"live value of {edge['effect_path']} unknown"}
        if live_effect != effect_value:
            return {"status": "NOT_ARMED"}
    return {"status": "ARMED"}


def _hop(edge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "from": edge["trigger_path"],
        "edge": edge.get("id"),
        "effect_path": edge["effect_path"],
        "effect": edge.get("effect"),
        "confidence": edge.get("confidence"),
        "citation": edge.get("citation"),
        "claimed_content": edge.get("claimed_content"),
        "when": _describe(edge),
    }


def _describe(edge: Dict[str, Any]) -> str:
    when = edge.get("when") or {}
    parts: List[str] = []
    if "op_is" in when:
        parts.append(f"{edge['trigger_path']} is {when['op_is']!r}")
    if "op_below" in when:
        parts.append(f"{edge['trigger_path']} < {when['op_below']!r}")
    if "requires" in when:
        r = when["requires"]
        parts.append(f"{r['path']} == {r.get('equals')!r}")
    return " and ".join(parts) if parts else "unconditional"


def why_chain(path: str, current: Optional[Dict[str, Any]] = None,
              max_depth: int = MAX_CHAIN_DEPTH) -> Dict[str, Any]:
    """Backward causal chain for the current value of ``path``.

    ``current`` is the NESTED live state (planner._read_current's shape).
    Verdicts: OK (>= 1 chain) / NO_CURATED_CAUSE (the value is what it
    is without any armed curated interaction — set directly) /
    UNKNOWN_PATH / DEPTH_EXCEEDED refuses rather than truncating a live
    chain silently."""
    spec = tool_by_path(path) or tool_by_name(path)
    if spec is not None:
        path = spec.path
    if not any(e["effect_path"] == path or e["trigger_path"] == path
               for e in _cons.EDGES):
        if spec is None:
            return {"verdict": "UNKNOWN_PATH", "path": path}
        return {"verdict": "NO_CURATED_CAUSE", "path": path,
                "note": "the key is in the registry but no curated "
                        "interaction touches it — its value is set "
                        "directly (see undo history)"}

    chains: List[Dict[str, Any]] = []
    unevaluated: List[Dict[str, Any]] = []
    _walk(path, current, [], chains, unevaluated, set(), max_depth)
    if not chains:
        return {"verdict": "NO_CURATED_CAUSE", "path": path,
                "unevaluated": unevaluated,
                "note": "no armed curated interaction explains this "
                        "value; it was set directly (see undo history "
                        "for who set it and when)"}
    return {"verdict": "OK", "path": path, "chains": chains,
            "unevaluated": unevaluated,
            "note": "chains are explanations OF CITED CODE in the "
                    "curated table (bounded, growable by reviewed "
                    "diffs), not causes in nature"}


def _walk(path: str, current: Optional[Dict[str, Any]],
          hops: List[Dict[str, Any]], chains: List[Dict[str, Any]],
          unevaluated: List[Dict[str, Any]], visited: set,
          depth: int) -> None:
    if depth == 0:
        if hops:
            chains.append({"hops": list(hops),
                           "terminal": "DEPTH_LIMIT",
                           "note": "chain stopped at the bounded depth "
                                   f"({MAX_CHAIN_DEPTH}); refusing to "
                                   "guess further"})
        return
    progressed = False
    for edge in _cons.EDGES:
        if edge["effect_path"] != path:
            continue
        if edge["trigger_path"] == path:
            continue  # self-annotation edges explain nothing backward
        state = _edge_state(edge, current)
        if state["status"] == "UNEVALUATED":
            entry = dict(_hop(edge))
            entry["status"] = "UNEVALUATED"
            entry["reason"] = state["reason"]
            if entry not in unevaluated:
                unevaluated.append(entry)
            continue
        if state["status"] != "ARMED":
            continue
        key = (edge["trigger_path"], edge.get("id"))
        if key in visited:
            continue  # cycle in the curated table itself: stop, no loop
        progressed = True
        hop = _hop(edge)
        _walk(edge["trigger_path"], current, hops + [hop], chains,
              unevaluated, visited | {key}, depth - 1)
    if not progressed and hops:
        live = _live(current, path)
        detail = (f"{path} is currently {live!r} — no curated "
                  "interaction explains it (set directly; see undo "
                  "history)" if live is not None else
                  f"live value of {path} unknown — cannot attribute "
                  "further (abstaining)")
        chains.append({"hops": list(hops), "terminal": "DIRECT",
                       "terminal_detail": detail})


def unevaluated_edges(current: Optional[Dict[str, Any]] = None
                      ) -> List[Dict[str, Any]]:
    """Curated edges whose ``requires`` live value is unknown — the ones
    a projection against this state could not evaluate. Public so the
    what-if CLI view lists them instead of silently dropping them."""
    out: List[Dict[str, Any]] = []
    for edge in _cons.EDGES:
        when = edge.get("when") or {}
        requires = when.get("requires")
        if not requires:
            continue
        if _live(current, requires["path"]) is None:
            out.append({
                "edge": edge.get("id"),
                "requires": requires["path"],
                "reason": f"live value of {requires['path']} unknown — "
                          "this edge could fire or not; abstaining",
            })
    return out


def counterfactual(tool: str, value: Any,
                   current: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """Forward counterfactual: "if I set ``tool`` to ``value``…".

    Reuses ``consequences.project`` verbatim for the fired effects and
    ADDS the honesty layer: curated edges that could not be evaluated
    because their ``requires`` live value is unknown are listed as
    UNEVALUATED — silence about an unevaluable edge is a silent drop,
    and this module's contract is the opposite."""
    spec = tool_by_name(str(tool)) or tool_by_path(str(tool))
    if spec is None:
        return {"verdict": "UNKNOWN_TOOL", "tool": str(tool)}
    ops = [{"tool": spec.name, "value": value}]
    projection = _cons.project(ops, current=current)
    out = {"verdict": "OK", "tool": spec.name, "value": value,
           "derived": projection.get("derived", []),
           "conflicts": projection.get("conflicts", []),
           "unevaluated": unevaluated_edges(current),
           "note": "single pass over the curated table; the transitive "
                   "view is `graph breaks <path>` (F12), this is the "
                   "cited immediate projection"}
    return out


def render_chain(result: Dict[str, Any]) -> List[str]:
    """Human lines for why_chain (the CLI and the why engine reuse)."""
    lines: List[str] = []
    if result.get("verdict") == "UNKNOWN_PATH":
        lines.append(f"causal: {result['path']} is not in the registry — "
                     "no causal answer exists")
        return lines
    if result.get("verdict") == "NO_CURATED_CAUSE":
        lines.append(f"causal: no armed curated interaction explains "
                     f"{result['path']}")
        note = result.get("note")
        if note:
            lines.append(f"  ({note})")
    else:
        for i, chain in enumerate(result.get("chains", ()), 1):
            for hop in chain.get("hops", ()):
                lines.append(
                    f"causal: {hop['effect_path']} <- {hop['from']} "
                    f"({hop['confidence']}, {hop['citation']})")
                lines.append(f"  because {hop['when']}: {hop['effect']}")
            term = chain.get("terminal_detail") or chain.get("terminal")
            if term:
                lines.append(f"  terminal: {term}")
    for u in result.get("unevaluated", ()):
        lines.append(f"causal: unevaluated edge {u.get('edge')}: "
                     f"{u.get('reason')}")
    return lines


def render_counterfactual(result: Dict[str, Any]) -> List[str]:
    """Human lines for counterfactual."""
    if result.get("verdict") == "UNKNOWN_TOOL":
        return [f"counterfactual: unknown tool {result['tool']!r}"]
    lines = [f"counterfactual: if {result['tool']} were "
             f"{result['value']!r}…"]
    for d in result.get("derived", ()):
        lines.append(f"  derived: {d.get('effect')} "
                     f"[{d.get('effect_path')}] ({d.get('confidence')}, "
                     f"{d.get('citation')})")
    if not result.get("derived"):
        lines.append("  derived: nothing in the curated table fires "
                     "(absence is not a guarantee)")
    for u in result.get("unevaluated", ()):
        lines.append(f"  unevaluated: edge {u.get('edge')}: {u.get('reason')}")
    return lines

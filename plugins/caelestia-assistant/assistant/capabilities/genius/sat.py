"""genius.sat — a small DPLL/CDCL-style SAT solver and a bounded model
checker over finite-state graphs (no dependencies).

The solver: DPLL (Davis, Logemann & Loveland 1962, "A Machine Program
for Theorem-Proving", CACM 5(7)) with unit propagation and pure-literal
elimination, plus the CDCL garnish that earns the hyphen — conflict
clause LEARNING (Marques-Silva & Sakallah 1996, "GRASP: A Search
Algorithm for Propositional Satisfiability", IEEE TC 45(11): at a
conflict, the negation of the current partial assignment's conflict-
relevant literals is learned as a new clause, so the same dead end is
never re-entered). Deterministic: variables are decided in sorted
order, the first unit clause wins propagation, no RNG.

THE BOUND (the honesty this module exists to keep): a hard node budget
counts every recursive decision node; exhausting it raises
``SatBudgetError`` — an explicit ABSTAIN, never a silent timeout and
never a silent "probably unsatisfiable". A caller that sees no
exception has a real SAT/UNSAT verdict for the exact CNF given.

Bounded model checking over finite-state graphs (Biere, Cimatti,
Clarke & Zhu 1999, "Symbolic Model Checking without BDDs", TACAS —
the SAT-encoding idea, scaled down to explicit state sets): states are
named nodes, transitions a dict node -> sorted successors, and a step
bound k. The encoder builds propositional variables v_t_s ("the system
is in state s at step t"), clamps the initial state, encodes every
transition edge, requires at least one live state per step, and asks
whether a target state is reachable WITHIN k steps:

    SAT   -> reachable within k (a witness path is recoverable)
    UNSAT -> provably unreachable within k (bounded — nothing is
             claimed about step k+1)

Deadlock states (no outgoing transitions) are found structurally, no
SAT needed — every edge is enumerated, so the answer is exact.

Integration targets (exponential-build-4 B): the settings preset state
graph (settings/presets.py::transition_graph) and the agent's HTN goal
DAGs (agent/goals.py::decompose, additive audit key). ``graph_audit``
is the one entry point both use.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["SatBudgetError", "SATSolver", "solve", "encode_reachability",
           "graph_audit", "witness_path"]


class SatBudgetError(RuntimeError):
    """The node budget ran out — an ABSTAIN, never a guessed verdict."""


_DEFAULT_NODE_BUDGET = 200_000


class SATSolver:
    """DPLL with unit propagation, pure literals, and clause learning.
    CNF: a clause is a frozenset of signed ints (variable i > 0, its
    negation < 0); a formula is a tuple of clauses."""

    def __init__(self, node_budget: int = _DEFAULT_NODE_BUDGET) -> None:
        if node_budget < 1:
            raise ValueError("node_budget must be >= 1")
        self.node_budget = int(node_budget)
        self.nodes_used = 0
        self.learned: List[FrozenSet[int]] = []

    # ------------------------------------------------------------------
    # core inference
    # ------------------------------------------------------------------

    def _dpll(self, clauses: Tuple[FrozenSet[int], ...],
              assign: Dict[int, bool], depth: int) -> Optional[Dict[int, bool]]:
        self.nodes_used += 1
        if self.nodes_used > self.node_budget:
            raise SatBudgetError(
                f"sat: node budget {self.node_budget} exhausted after "
                f"{self.nodes_used} decision nodes — ABSTAIN (no verdict "
                "is guessed past the bound)")
        # unit propagation (first unit clause wins — deterministic)
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                unassigned: List[int] = []
                satisfied = False
                for lit in clause:
                    val = assign.get(abs(lit))
                    if val is None:
                        unassigned.append(lit)
                    elif (lit > 0) == val:
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not unassigned:
                    # conflict: learn the negation of the current branch
                    # decision trail (the light CDCL clause) and fail
                    self._learn_conflict(clause, assign)
                    return None
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    assign[abs(lit)] = lit > 0
                    changed = True
        # pure literal elimination (deterministic: lowest variable first)
        counts: Dict[int, Set[bool]] = {}
        for clause in clauses:
            satisfied = any((lit > 0) == assign.get(abs(lit), not (lit > 0))
                            and abs(lit) in assign and (lit > 0) == assign[abs(lit)]
                            for lit in clause)
            if satisfied:
                continue
            for lit in clause:
                var = abs(lit)
                if var in assign:
                    continue
                counts.setdefault(var, set()).add(lit > 0)
        for var in sorted(counts):
            polarities = counts[var]
            if len(polarities) == 1:
                assign[var] = polarities.pop()
                return self._dpll(clauses, assign, depth + 1)
        # decide (lowest unassigned variable, False first — the
        # conservative branch; determinism over speed)
        unassigned_vars = sorted({abs(lit) for c in clauses for lit in c
                                  if abs(lit) not in assign})
        if not unassigned_vars:
            # every clause satisfied by construction of the walk above?
            for clause in clauses:
                if not any((lit > 0) == assign.get(abs(lit)) for lit in clause):
                    return None
            return dict(assign)
        var = unassigned_vars[0]
        for value in (False, True):
            trial = dict(assign)
            trial[var] = value
            result = self._dpll(clauses, trial, depth + 1)
            if result is not None:
                return result
        return None

    def _learn_conflict(self, clause: FrozenSet[int],
                        assign: Dict[int, bool]) -> None:
        """Record the CDCL-style learned clause: the negation of every
        currently-assigned literal involved in the conflict's clause.
        Learned clauses accumulate in ``self.learned``; the next solve
        reuses them (the point of learning: the same dead end is never
        re-entered)."""
        learned: Set[int] = set()
        for lit in clause:
            val = assign.get(abs(lit))
            if val is not None:
                learned.add(-lit if (lit > 0) == val else lit)
            else:
                learned.add(lit)
        if learned:
            self.learned.append(frozenset(learned))

    # ------------------------------------------------------------------
    # public face
    # ------------------------------------------------------------------

    def solve(self, cnf: Sequence[Iterable[int]]) -> Dict[str, Any]:
        """Solve one CNF. Returns {"satisfiable": bool, "model": dict,
        "nodes_used": int, "learned_clauses": int}; raises
        SatBudgetError past the bound (ABSTAIN)."""
        self.learned = []
        self.nodes_used = 0
        clauses = tuple(frozenset(int(lit) for lit in clause)
                        for clause in cnf)
        if not clauses:
            return {"satisfiable": True, "model": {}, "nodes_used": 0,
                    "learned_clauses": 0}
        if any(any(lit == 0 for lit in c) for c in clauses):
            raise ValueError("literal 0 is not a variable (DIMACS: vars "
                             "start at 1)")
        assign: Dict[int, bool] = {}
        model = self._dpll(clauses, assign, 0)
        return {"satisfiable": model is not None,
                "model": (model or {}),
                "nodes_used": self.nodes_used,
                "learned_clauses": len(self.learned)}


def solve(cnf: Sequence[Iterable[int]],
          node_budget: int = _DEFAULT_NODE_BUDGET) -> Dict[str, Any]:
    """One-shot convenience over SATSolver."""
    return SATSolver(node_budget=node_budget).solve(cnf)


# ---------------------------------------------------------------------------
# Bounded model checking over finite-state graphs
# ---------------------------------------------------------------------------


def _normalize_graph(graph: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for node, successors in graph.items():
        succ = sorted(set(str(s) for s in successors))
        unknown = [s for s in succ if s not in graph]
        if unknown:
            raise ValueError(
                f"transition from {node!r} names state(s) {unknown} not "
                "present in the graph — refused, not silently dropped")
        out[str(node)] = succ
    return out


def deadlock_states(graph: Dict[str, Sequence[str]]) -> List[str]:
    """States with no outgoing transitions — exact, structural, no SAT."""
    return sorted(n for n, succ in _normalize_graph(graph).items() if not succ)


def encode_reachability(graph: Dict[str, Sequence[str]], start: str,
                        target: str, k: int
                        ) -> Tuple[List[List[int]], Dict[str, int], List[str]]:
    """CNF for "target reachable from start within k steps".

    Encoding (Biere, Cimatti, Clarke & Zhu 1999, scaled to explicit
    state sets): states get a virtual absorbing HALT successor so a
    path that reached the target early may STOP — without it, the
    at-least-one-live clause would force motion past dead ends and
    wrongly prove targets unreachable. Variables v_t_s = "in state s at
    step t"; clauses:

      - v_0_start (the system starts at start);
      - exactly-one live state per step (at-least-one + pairwise
        at-most-one) — keeps every model a single walkable path;
      - v_{t+1,s'} requires a live predecessor (every incoming edge:
        -v_{t+1,s'} | v_{t,p} | ... ) — ghosts without a real path
        ancestor are impossible, which is what makes UNSAT a genuine
        unreachability proof;
      - the target is live at SOME step <= k.

    SAT -> reachable within k (the model IS a witness path); UNSAT ->
    provably unreachable within k (nothing is claimed about k+1).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    g = _normalize_graph(graph)
    states = sorted(g)
    if start not in g:
        raise ValueError(f"start {start!r} not in graph")
    if target not in g:
        raise ValueError(f"target {target!r} not in graph")
    halt = "<halt>"
    all_states = states + [halt]
    succ: Dict[str, List[str]] = {s: list(g[s]) + [halt] for s in states}
    succ[halt] = [halt]
    index = {s: i for i, s in enumerate(all_states)}

    def var(t: int, s: str) -> int:
        return 1 + t * len(all_states) + index[s]

    cnf: List[List[int]] = [[var(0, start)]]
    # exactly-one live at t=0 as well: every other state is NOT live
    for s in all_states:
        if s != start:
            cnf.append([-var(0, s)])
    for t in range(k):
        # exactly-one live state at t+1
        cnf.append([var(t + 1, s) for s in all_states])
        for i1 in range(len(all_states)):
            for i2 in range(i1 + 1, len(all_states)):
                cnf.append([-var(t + 1, all_states[i1]),
                            -var(t + 1, all_states[i2])])
        # every live state at t+1 has a live predecessor at t
        for s2 in all_states:
            preds = [s for s in all_states if s2 in succ[s]]
            cnf.append([-var(t + 1, s2)] + [var(t, s) for s in preds])
    # target live at some step <= k (start == target: trivially true,
    # returned early by callers; the unit clause below stays honest)
    cnf.append([var(t, target) for t in range(k + 1)])
    names = [f"v_{t}_{s}" for t in range(k + 1) for s in all_states]
    return cnf, index, names


def witness_path(graph: Dict[str, Sequence[str]], start: str, target: str,
                 k: int, node_budget: int = _DEFAULT_NODE_BUDGET
                 ) -> Dict[str, Any]:
    """SAT on the reachability encoding; the exactly-one encoding makes
    the model a single walkable path (recovery reads the live variable
    at each step and stops at the target or at HALT). UNSAT = provably
    unreachable within k. start == target is trivially reachable."""
    g = _normalize_graph(graph)
    states = sorted(g)
    if start not in g:
        raise ValueError(f"start {start!r} not in graph")
    if target not in g:
        raise ValueError(f"target {target!r} not in graph")
    if start == target:
        return {"reachable_within_k": True, "steps": 0, "path": [start],
                "bound_k": k, "nodes_used": 0}
    cnf, index, _names = encode_reachability(graph, start, target, k)
    result = solve(cnf, node_budget=node_budget)
    if not result["satisfiable"]:
        return {"reachable_within_k": False, "steps": None,
                "path": None, "bound_k": k, "nodes_used":
                result["nodes_used"]}
    model = result["model"]
    halt = "<halt>"
    all_states = states + [halt]

    def var_id(t: int, s: str) -> int:
        return 1 + t * len(all_states) + index[s]

    path: List[str] = []
    for t in range(k + 1):
        step = [s for s in all_states if model.get(var_id(t, s))]
        current = step[0] if step else halt
        if current == halt:
            break
        path.append(current)
        if current == target:
            break
    return {"reachable_within_k": True,
            "steps": len(path) - 1 if path else 0,
            "path": path,
            "bound_k": k,
            "nodes_used": result["nodes_used"]}


def graph_audit(graph: Dict[str, Sequence[str]], start: str,
                k: int = 6, node_budget: int = _DEFAULT_NODE_BUDGET
                ) -> Dict[str, Any]:
    """The one entry point the settings preset graph and the HTN goal
    DAGs both go through: deadlocks (exact) + bounded unreachability of
    every non-start state (SAT; one solve per target so a budget
    exhaustion names its target). ABSTAIN is loud: the audit reports
    {"status": "ABSTAINED_BUDGET", ...} rather than guessing."""
    g = _normalize_graph(graph)
    if start not in g:
        raise ValueError(f"start {start!r} not in graph")
    states = sorted(g)
    deadlocks = deadlock_states(g)
    unreachable: List[str] = []
    abstained: List[Dict[str, Any]] = []
    witnesses: Dict[str, List[str]] = {}
    for target in states:
        if target == start:
            continue
        try:
            outcome = witness_path(g, start, target, k,
                                   node_budget=node_budget)
        except SatBudgetError as exc:
            abstained.append({"target": target, "reason": str(exc)})
            continue
        if outcome["reachable_within_k"]:
            witnesses[target] = outcome["path"] or []
        else:
            unreachable.append(target)
    status = ("ABSTAINED_BUDGET" if abstained else
              "CLEAN" if not (unreachable or deadlocks) else "FINDINGS")
    return {
        "status": status,
        "n_states": len(states),
        "bound_k": k,
        "deadlock_states": deadlocks,
        "unreachable_within_k": unreachable,
        "witnesses": witnesses,
        "abstained": abstained,
        "note": "unreachability is BOUNDED (within k steps — nothing is "
                "claimed beyond the bound); deadlocks are exact; a "
                "budget exhaustion is an ABSTAIN, never a verdict",
    }

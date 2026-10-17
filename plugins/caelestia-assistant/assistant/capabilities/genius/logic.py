"""genius.logic — deduction, satisfiability, and constraint reasoning.

  * propositional parser for `p and not q -> r` style formulas
  * truth tables; tautology / contradiction / contingency classification
  * logical equivalence and entailment checking
  * DPLL SAT solver with unit propagation and pure-literal elimination
  * forward-chaining inference over `IF ... THEN` rules with proof traces
  * constraint satisfaction: backtracking with MRV + forward checking,
    and AC-3 arc-consistency preprocessing

All deterministic, all stdlib, all evidence-carrying.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "LogicError", "Formula", "parse_formula", "evaluate_formula", "truth_table",
    "classify_formula", "equivalent", "entails", "sat_solve", "rule_infer",
    "CSP", "solve_csp", "schedule_resources", "Statement", "parse_statement",
    "syllogism_check",
]

_TOKENS = {
    "and": "&", "or": "|", "not": "~", "->": "->", "=>": "->",
    "iff": "<->", "<=>": "<->", "xor": "^",
}


class LogicError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Formula AST: tuple-based ('var', p) ('not', f) ('and', l, r) ...
# ---------------------------------------------------------------------------

Formula = Tuple


def _lex(text: str) -> List[str]:
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        two = text[i:i + 2]
        if two in ("->", "=>"):
            out.append("->")
            i += 2
            continue
        if text[i:i + 3] in ("<->", "<=>"):
            out.append("<->")
            i += 3
            continue
        if ch in "()~^&|":
            out.append(ch)
            i += 1
            continue
        if ch.isalnum() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            out.append(_TOKENS.get(word.lower(), word))
            i = j
            continue
        raise LogicError(f"unexpected character {ch!r}")
    return out


def parse_formula(text: str) -> Formula:
    tokens = _lex(text)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        tok = peek()
        pos[0] += 1
        return tok

    def parse_iff() -> Formula:
        left = parse_or()
        while peek() == "<->":
            take()
            left = ("iff", left, parse_or())
        return left

    def parse_or() -> Formula:
        left = parse_and()
        while peek() in ("|", "^"):
            op = take()
            left = ("xor", left, parse_and()) if op == "^" else ("or", left, parse_and())
        return left

    def parse_and() -> Formula:
        left = parse_imp()
        while peek() == "&":
            take()
            left = ("and", left, parse_imp())
        return left

    def parse_imp() -> Formula:
        left = parse_unary()
        while peek() == "->":
            take()
            left = ("imp", left, parse_unary())
        return left

    def parse_unary() -> Formula:
        tok = peek()
        if tok == "~":
            take()
            return ("not", parse_unary())
        if tok == "(":
            take()
            inner = parse_iff()
            if peek() != ")":
                raise LogicError("missing ')'")
            take()
            return inner
        if tok is None or tok in ("&", "|", ")", "->", "<->", "^"):
            raise LogicError(f"unexpected token {tok!r}")
        return ("var", take())

    f = parse_iff()
    if pos[0] != len(tokens):
        raise LogicError(f"trailing tokens: {tokens[pos[0]:]}")
    return f


def formula_vars(f: Formula) -> Set[str]:
    kind = f[0]
    if kind == "var":
        return {f[1]}
    out: Set[str] = set()
    for part in f[1:]:
        if isinstance(part, tuple):
            out |= formula_vars(part)
    return out


def formula_str(f: Formula) -> str:
    kind = f[0]
    if kind == "var":
        return f[1]
    if kind == "not":
        return f"~{formula_str(f[1])}"
    op = {"and": "&", "or": "|", "imp": "->", "iff": "<->", "xor": "^"}[kind]
    return f"({formula_str(f[1])} {op} {formula_str(f[2])})"


def evaluate_formula(f: Formula, env: Dict[str, bool]) -> bool:
    kind = f[0]
    if kind == "var":
        if f[1] not in env:
            raise LogicError(f"no value for {f[1]!r}")
        return env[f[1]]
    if kind == "not":
        return not evaluate_formula(f[1], env)
    a = evaluate_formula(f[1], env)
    b = evaluate_formula(f[2], env)
    return {"and": a and b, "or": a or b, "imp": (not a) or b,
            "iff": a == b, "xor": a != b}[kind]


def truth_table(text: str, limit_vars: int = 8) -> Dict[str, Any]:
    f = parse_formula(text) if isinstance(text, str) else text
    vs = sorted(formula_vars(f))
    if len(vs) > limit_vars:
        raise LogicError(f"too many variables ({len(vs)}) for a full table; use sat_solve")
    rows = []
    for combo in itertools.product([False, True], repeat=len(vs)):
        env = dict(zip(vs, combo))
        rows.append({**{v: env[v] for v in vs}, "result": evaluate_formula(f, env)})
    return {"formula": formula_str(f), "variables": vs, "rows": rows,
            "n_true": sum(1 for r in rows if r["result"]),
            "n_rows": len(rows)}


def classify_formula(text: str) -> Dict[str, Any]:
    tt = truth_table(text)
    n_true = tt["n_true"]
    kind = ("tautology" if n_true == tt["n_rows"]
            else "contradiction" if n_true == 0 else "contingency")
    return {"formula": tt["formula"], "classification": kind,
            "true_rows": n_true, "total_rows": tt["n_rows"]}


def equivalent(a: str, b: str) -> Dict[str, Any]:
    fa, fb = parse_formula(a), parse_formula(b)
    vs = sorted(formula_vars(fa) | formula_vars(fb))
    witness = None
    for combo in itertools.product([False, True], repeat=len(vs)):
        env = dict(zip(vs, combo))
        va = evaluate_formula(fa, env)
        vb = evaluate_formula(fb, env)
        if va != vb:
            witness = {**env, f"{a!r}": va, f"{b!r}": vb}
            break
    return {"equivalent": witness is None, "counterexample": witness}


def entails(a: str, b: str) -> Dict[str, Any]:
    """Does a |- b?  Iff (a AND not-b) is unsatisfiable."""
    f = ("and", parse_formula(a), ("not", parse_formula(b)))
    sat = sat_solve(f)
    return {"entails": not sat["satisfiable"],
            "because": "a AND not-b is unsatisfiable" if not sat["satisfiable"]
            else f"counterexample: {sat['model']}"}


def sat_solve(text: str) -> Dict[str, Any]:
    """DPLL with unit propagation and pure-literal elimination."""
    f = parse_formula(text) if isinstance(text, str) else text

    def clauses_of(f: Formula, polarity: bool = True) -> List[FrozenSet[Tuple[str, bool]]]:
        if f[0] == "var":
            return [frozenset([(f[1], polarity)])]
        if f[0] == "not":
            return clauses_of(f[1], not polarity)
        if f[0] == "and":
            return clauses_of(f[1], polarity) + clauses_of(f[2], polarity)
        if f[0] == "or":
            a, b = clauses_of(f[1], polarity), clauses_of(f[2], polarity)
            return [x | y for x in a for y in b]
        if f[0] == "imp":
            return clauses_of(("or", ("not", f[1]), f[2]), polarity)
        if f[0] == "iff":
            return (clauses_of(("and", ("imp", f[1], f[2]), ("imp", f[2], f[1])), polarity)
                    if polarity else
                    clauses_of(("or", f[1], f[2]), False) + clauses_of(("or", f[1], f[2]), True))
        if f[0] == "xor":
            # a xor b == (a or b) and not (a and b)
            cnf = ("and", ("or", f[1], f[2]), ("or", ("not", f[1]), ("not", f[2])))
            return clauses_of(cnf, polarity)
        raise LogicError(f"cannot handle {f[0]}")

    try:
        clauses = clauses_of(f)
    except RecursionError:
        raise LogicError("formula too large for CNF conversion")

    def dpll(clauses: List[FrozenSet[Tuple[str, bool]]],
             assign: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        # unit propagation
        changed = True
        while changed:
            changed = False
            for c in clauses:
                if not c:
                    return None
                if len(c) == 1:
                    (var, val) = next(iter(c))
                    if var in assign and assign[var] != val:
                        return None
                    if var not in assign:
                        assign = {**assign, var: val}
                        clauses = [cl for cl in clauses if (var, val) not in cl]
                        clauses = [cl - {(var, not val)} for cl in clauses]
                        changed = True
                        break
        if not clauses:
            return assign
        # pure literal elimination
        lits = {lit for c in clauses for lit in c}
        for var in {v for v, _ in lits}:
            if (var, True) in lits and (var, False) not in lits:
                return dpll([c for c in clauses if (var, True) not in c], {**assign, var: True})
            if (var, False) in lits and (var, True) not in lits:
                return dpll([c for c in clauses if (var, False) not in c], {**assign, var: False})
        # branch on the first variable of the shortest clause (MRV-ish)
        shortest = min(clauses, key=len)
        var = next(iter(shortest))[0]
        for val in (True, False):
            sub = [c for c in clauses if (var, val) not in c]
            sub = [c - {(var, not val)} for c in sub]
            result = dpll(sub, {**assign, var: val})
            if result is not None:
                return result
        return None

    model = dpll([frozenset(c) for c in clauses], {})
    vs = sorted(formula_vars(f))
    return {"formula": formula_str(f), "satisfiable": model is not None,
            "model": {k: model[k] for k in sorted(model)} if model else None,
            "n_clauses": len(clauses),
            "variables": vs}


# ---------------------------------------------------------------------------
# Rule-based forward chaining (with proof traces)
# ---------------------------------------------------------------------------

def rule_infer(rules: Sequence[Dict[str, Any]], facts: Iterable[str],
              max_steps: int = 200) -> Dict[str, Any]:
    """Forward-chaining over IF-THEN rules.

    Rules: {"if": ["hot", "humid"], "then": "storm"} — facts are strings.
    Every derivation is recorded so the proof can be shown.
    """
    known: Set[str] = set(facts)
    proofs: Dict[str, Any] = {f: {"from": "given", "via": None} for f in known}
    steps: List[Dict[str, Any]] = []
    rule_list = list(rules)
    changed = True
    guard = 0
    while changed and guard < max_steps:
        changed = False
        guard += 1
        for idx, rule in enumerate(rule_list):
            ants = [str(a) for a in rule["if"]]
            missing = [a for a in ants if a not in known]
            if missing:
                continue
            head = str(rule["then"])
            if head in known:
                continue
            known.add(head)
            proofs[head] = {"from": ants, "via": f"rule #{idx + 1}"}
            steps.append({"step": len(steps) + 1, "rule": idx + 1,
                          "fired": head, "because": ants})
            changed = True
    return {"facts": sorted(known), "derived": sorted(known - set(facts)),
            "steps": steps, "proofs": {k: v for k, v in proofs.items()
                                        if k not in set(facts)},
            "iterations": guard}


# ---------------------------------------------------------------------------
# Constraint satisfaction (backtracking + MRV + forward checking + AC-3)
# ---------------------------------------------------------------------------

class CSP:
    """A tiny constraint problem: variables, domains, binary + all-different."""

    def __init__(self, variables: Sequence[str],
                 domains: Dict[str, Sequence[Any]],
                 constraints: Optional[Sequence[Tuple[str, str, str]]] = None,
                 all_different: Optional[Sequence[Sequence[str]]] = None):
        self.variables = list(variables)
        self.domains: Dict[str, list] = {v: list(domains[v]) for v in self.variables}
        # constraints as (var_a, var_b, "op") with op in <,<=,>,>=,!=,==
        self.constraints = [(a, b, op) for a, b, op in (constraints or [])]
        self.all_different = [list(g) for g in (all_different or [])]

    def _ok(self, a: Any, b: Any, op: str) -> bool:
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                "!=": a != b, "==": a == b, "=": a == b}.get(op, True)

    def _consistent(self, var: str, value: Any, assign: Dict[str, Any]) -> bool:
        for a, b, op in self.constraints:
            if a == var and b in assign and not self._ok(value, assign[b], op):
                return False
            if b == var and a in assign and not self._ok(assign[a], value, op):
                return False
        for group in self.all_different:
            if var in group:
                for other in group:
                    if other != var and other in assign and assign[other] == value:
                        return False
        return True

    def ac3(self) -> List[str]:
        """Arc-consistency preprocessing (returns pruned domains info)."""
        def revise(xi: str, xj: str) -> bool:
            removed = []
            for x in list(self.domains[xi]):
                if not any(self._ok(x, y, op) for y in self.domains[xj]
                           for a, b, op in self.constraints
                           if (a == xi and b == xj)):
                    # also handle reversed orientation
                    if not any(self._ok(y, x, op) for y in self.domains[xj]
                               for a, b, op in self.constraints
                               if (a == xj and b == xi)):
                        removed.append(x)
            for x in removed:
                self.domains[xi].remove(x)
            return bool(removed)

        queue = [(a, b) for a, b, _ in self.constraints] + \
                [(b, a) for a, b, _ in self.constraints]
        pruned: List[str] = []
        while queue:
            xi, xj = queue.pop()
            before = len(self.domains[xi])
            if revise(xi, xj):
                pruned.append(f"{xi}: {before} -> {len(self.domains[xi])} values")
                if not self.domains[xi]:
                    break
                for a, b, _ in self.constraints:
                    if b == xi and a != xj:
                        queue.append((a, b))
        return pruned


def solve_csp(csp: CSP, max_solutions: int = 10) -> Dict[str, Any]:
    """Backtracking search with MRV + forward checking."""
    solutions: List[Dict[str, Any]] = []
    nodes = [0]

    def forward_check(var: str, value: Any, domains: Dict[str, list]) -> bool:
        saved = {v: list(d) for v, d in domains.items()}
        for a, b, op in csp.constraints:
            if a == var and b not in saved_in_progress[0]:
                domains[b] = [y for y in domains[b] if csp._ok(value, y, op)]
            elif b == var and a not in saved_in_progress[0]:
                domains[a] = [y for y in domains[a] if csp._ok(y, value, op)]
        for group in csp.all_different:
            if var in group:
                for other in group:
                    if other != var:
                        domains[other] = [y for y in domains[other] if y != value]
        if any(not d for d in domains.values()):
            domains.clear()
            domains.update(saved)
            return False
        return True

    saved_in_progress: List[set] = [set()]
    domains = {v: list(d) for v, d in csp.domains.items()}

    def backtrack(assign: Dict[str, Any]):
        if len(assign) == len(csp.variables):
            solutions.append(dict(assign))
            return len(solutions) >= max_solutions
        unassigned = [v for v in csp.variables if v not in assign]
        var = min(unassigned, key=lambda v: len(domains[v]))  # MRV
        nodes[0] += 1
        for value in list(domains[var]):
            if not csp._consistent(var, value, assign):
                continue
            snapshot = {v: list(d) for v, d in domains.items()}
            assign[var] = value
            saved_in_progress[0] = set(assign)
            ok = forward_check(var, value, domains)
            if ok:
                if backtrack(assign):
                    return True
            assign.pop(var)
            saved_in_progress[0] = set(assign)
            domains.clear()
            domains.update(snapshot)
        return False

    ac3_report = csp.ac3()
    if any(not csp.domains[v] for v in csp.variables):
        return {"solutions": [], "n_solutions": 0, "nodes_explored": 0,
                "ac3_pruned": ac3_report, "satisfiable": False}
    domains = {v: list(d) for v, d in csp.domains.items()}
    backtrack({})
    return {"solutions": solutions, "n_solutions": len(solutions),
            "nodes_explored": nodes[0], "ac3_pruned": ac3_report,
            "satisfiable": bool(solutions),
            "first": solutions[0] if solutions else None}


# ---------------------------------------------------------------------------
# Resource-contention scheduling (phase 2.2): the general primitive.
#
# settings/optimize.py runs AC-3 arc-consistency over SETTINGS KEYS (which
# bar supports which spacing). This is the SAME constraint machinery,
# lifted to the general shape the agent layer needs: tasks contend for
# limited resources across discrete slots. No settings knowledge lives
# here and no settings caller changes — this is an addition, not a
# migration; both surfaces sit on the one CSP implementation above.
# ---------------------------------------------------------------------------


def schedule_resources(tasks: Sequence[Any],
                       n_slots: int,
                       precedence: Sequence[Tuple[str, str]] = (),
                       windows: Optional[Dict[str, Tuple[int, int]]] = None,
                       max_solutions: int = 5) -> Dict[str, Any]:
    """Schedule tasks that contend for named resources over ``n_slots``.

    The general resource-contention primitive (usable by the agent layer
    as a planning step, and by any caller with the same shape):

    - ``tasks``: sequence of ``(task_id, resource)`` pairs or dicts with
      ``{"id", "resource"}`` — two tasks sharing a resource can never
      occupy the same slot (modeled as all-different per resource, the
      classic timetabling formulation);
    - ``precedence``: ``(before_id, after_id)`` pairs — ``before``
      must land strictly earlier than ``after`` (binary ``<`` arcs);
    - ``windows``: optional per-task ``{task: (earliest, latest)}`` slot
      bounds (domain pruning before AC-3 runs);
    - ``n_slots``: the discrete horizon (slots are integers 0..n-1;
      durations are NOT modeled — one task, one slot, the unit-capacity
      case; anything coarser composes over multiple calls).

    Solving = the CSP machinery above: domain windows, then AC-3 arc
    consistency, then backtracking with MRV + forward checking. The
    FIRST solution is the schedule; ``ac3_pruned`` reports which domains
    the arc-consistency pass narrowed before search (the same honest
    evidence settings/optimize reports); an unsatisfiable request
    returns the pruning trail, never a silent empty answer.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    task_ids: List[str] = []
    resource_of: Dict[str, str] = {}
    for task in tasks:
        if isinstance(task, dict):
            task_id = str(task.get("id", ""))
            resource = str(task.get("resource", ""))
        else:
            task_id, resource = str(task[0]), str(task[1])
        if not task_id or not resource:
            raise ValueError("each task needs an id and a resource")
        if task_id in resource_of:
            raise ValueError(f"duplicate task id {task_id!r}")
        task_ids.append(task_id)
        resource_of[task_id] = resource

    known = set(task_ids)
    for before, after in precedence:
        if before not in known or after not in known:
            raise ValueError(f"precedence names unknown task: "
                             f"{before!r} -> {after!r}")

    domains: Dict[str, Sequence[int]] = {}
    windows = windows or {}
    for task_id in task_ids:
        if task_id in windows:
            lo, hi = windows[task_id]
            lo, hi = max(0, int(lo)), min(n_slots - 1, int(hi))
            if lo > hi:
                raise ValueError(f"window for {task_id!r} is empty")
            domains[task_id] = list(range(lo, hi + 1))
        else:
            domains[task_id] = list(range(n_slots))

    # all-different per resource: tasks on the same resource never share
    # a slot (unit capacity — the timetabling classic).
    by_resource: Dict[str, List[str]] = {}
    for task_id in task_ids:
        by_resource.setdefault(resource_of[task_id], []).append(task_id)
    all_different = [group for group in by_resource.values() if len(group) > 1]

    # precedence as strictly-before binary arcs on the slot integers
    constraints = [(before, after, "<") for before, after in precedence]

    problem = CSP(task_ids, domains, constraints, all_different)
    result = solve_csp(problem, max_solutions=max_solutions)
    out: Dict[str, Any] = {
        "algorithm": ("resource-contention scheduling over the CSP above: "
                      "per-resource all-different + precedence arcs, AC-3 "
                      "then backtracking (MRV + forward checking)"),
        "satisfiable": result["satisfiable"],
        "ac3_pruned": result["ac3_pruned"],
        "nodes_explored": result["nodes_explored"],
        "resources": {r: sorted(g) for r, g in by_resource.items()},
    }
    first = result.get("first")
    if first is not None:
        out["schedule"] = dict(sorted(first.items(), key=lambda kv: kv[1]))
        # the load view: slot -> tasks (readable plan for the agent layer)
        load: Dict[int, List[str]] = {}
        for task_id, slot in first.items():
            load.setdefault(int(slot), []).append(task_id)
        out["slot_load"] = {str(slot): sorted(tasks_)
                            for slot, tasks_ in sorted(load.items())}
    else:
        out["schedule"] = None
        out["reason"] = ("no conflict-free assignment exists under these "
                         "slots/precedence/windows (see ac3_pruned)")
    return out


# ---------------------------------------------------------------------------
# First-order resolution for categorical syllogisms (exponential-build 2.4).
#
# Method: binary resolution on the clause set premises + NEGATED
# conclusion (Robinson 1965, "A Machine-Oriented Logic Based on the
# Resolution Principle", JACM 12(1)). The fragment is deliberately tiny:
# unary predicates over one variable plus ground Skolem constants for
# existentials — every clause is Horn (Horn 1951), so binary resolution
# needs no factoring here and saturation is DECIDABLE (the derivable
# clause set is bounded by predicates x constants). Valid <=> the empty
# clause is derived; saturation without it is the honest "invalid".
# ---------------------------------------------------------------------------

@dataclass
class Statement:
    """One parsed categorical/singular statement."""
    kind: str            # all | no | some | some_not | is | is_not
    subject: str         # class word for all/no/some*, constant for is*
    cls: str             # the predicate/class word


_STMT_PATTERNS = (
    (re.compile(r"^(?:all|every)\s+([a-z]+)\s+(?:are|is)\s+([a-z]+)$", re.I),
     lambda m: Statement("all", m.group(1).lower(), m.group(2).lower())),
    (re.compile(r"^no\s+([a-z]+)\s+(?:are|is)\s+([a-z]+)$", re.I),
     lambda m: Statement("no", m.group(1).lower(), m.group(2).lower())),
    (re.compile(r"^([a-z]+)\s+is\s+not\s+(?:a\s+|an\s+)?([a-z]+)$", re.I),
     lambda m: Statement("is_not", m.group(1).lower(), m.group(2).lower())),
    (re.compile(r"^some\s+([a-z]+)\s+(?:are|is)\s+not\s+(?:a\s+|an\s+)?([a-z]+)$", re.I),
     lambda m: Statement("some_not", m.group(1).lower(), m.group(2).lower())),
    (re.compile(r"^some\s+([a-z]+)\s+(?:are|is)\s+(?:a\s+|an\s+)?([a-z]+)$", re.I),
     lambda m: Statement("some", m.group(1).lower(), m.group(2).lower())),
    (re.compile(r"^([a-z]+)\s+(?:is|are)\s+(?:a\s+|an\s+)?([a-z]+)$", re.I),
     lambda m: Statement("is", m.group(1).lower(), m.group(2).lower())),
)
_RESERVED = {"all", "every", "no", "some", "not", "is", "are",
             "the", "therefore", "then"}

_VAR = ("var",)  # the single universally quantified term


def parse_statement(text: str) -> Statement:
    """Parse one statement of the well-formed subset (all/no/some/some-
    not A are B, and singular 'x is (not) Y'). Anything else is an
    honest LogicError listing what the fragment accepts."""
    s = " ".join(str(text).strip().split())
    for pattern, build in _STMT_PATTERNS:
        m = pattern.match(s)
        if m:
            stmt = build(m)
            if stmt.subject in _RESERVED or stmt.cls in _RESERVED:
                continue
            return stmt
    raise LogicError(
        f"not in the syllogism fragment: {text!r} — accepted forms: "
        "'all A are B', 'no A are B', 'some A are B', 'some A are not B', "
        "'x is Y', 'x is not Y'")


def _negate(stmt: Statement) -> Statement:
    """The classical categorical negation (drives the reductio)."""
    table = {"all": "some_not", "some_not": "all", "no": "some",
             "some": "no", "is": "is_not", "is_not": "is"}
    return Statement(table[stmt.kind], stmt.subject, stmt.cls)


def _canon(word: str) -> str:
    """Canonical class-word form: one trailing 's' stripped (regular
    plurals), so 'cats' and 'cat' are the same predicate. Deliberately
    NOT a morphology engine: irregulars ('men'/'man') stay distinct and
    the docstring says so — no guessing beyond the stated rule."""
    return word[:-1] if word.endswith("s") and len(word) > 1 else word


def _skolem_clauses(stmt: Statement, tag: int) -> Set[FrozenSet[Tuple]]:
    """Clause translation. Literals: (sign, pred, term) with term
    ('var',) or ('const', name). Class words are canonicalized through
    the regular-plural rule; constants are never touched."""
    A, B = _canon(stmt.subject), _canon(stmt.cls)
    if stmt.kind == "all":
        return {frozenset({(False, A, _VAR), (True, B, _VAR)})}
    if stmt.kind == "no":
        return {frozenset({(False, A, _VAR), (False, B, _VAR)})}
    c = ("const", f"_sk{tag}")
    if stmt.kind == "some":
        return {frozenset({(True, A, c)}), frozenset({(True, B, c)})}
    if stmt.kind == "some_not":
        return {frozenset({(True, A, c)}), frozenset({(False, B, c)})}
    c = ("const", stmt.subject)
    if stmt.kind == "is":
        return {frozenset({(True, B, c)})}
    return {frozenset({(False, B, c)})}  # is_not


def _substitute(lit: Tuple, var_as: Tuple) -> Tuple:
    sign, pred, term = lit
    return (sign, pred, var_as if term == _VAR else term)


def _resolvents(c1: FrozenSet[Tuple], c2: FrozenSet[Tuple]
                ) -> List[FrozenSet[Tuple]]:
    """All binary resolvents of two clauses (unification on the single
    variable: var unifies with anything; const only with itself)."""
    out: List[FrozenSet[Tuple]] = []
    for l1 in c1:
        for l2 in c2:
            if l1[0] == l2[0] or l1[1] != l2[1]:
                continue  # need opposite signs on the same predicate
            t1, t2 = l1[2], l2[2]
            if t1 != _VAR and t2 != _VAR and t1 != t2:
                continue  # distinct constants cannot unify
            if t1 == _VAR and t2 == _VAR:
                sigma = _VAR  # identity
            elif t1 == _VAR:
                sigma = t2
            else:
                sigma = t1
            r = frozenset({_substitute(l, sigma) for l in c1 - {l1}} |
                          {_substitute(l, sigma) for l in c2 - {l2}})
            out.append(r)
    return out


def _clause_key(clause: FrozenSet[Tuple]) -> Tuple:
    return tuple(sorted(clause))


def syllogism_check(premises: Sequence[str], conclusion: str,
                    max_steps: int = 2000) -> Dict[str, Any]:
    """Is the argument valid? Premises + negated conclusion are refuted
    by resolution (Robinson 1965): the empty clause proves VALID;
    saturation WITHOUT it proves INVALID (decidable in this Horn,
    unary fragment — the saturated set is satisfiable, the premises can
    hold while the conclusion fails). Parse failures are honest
    LogicErrors; the derivation is reported, never just a verdict."""
    stmts = [parse_statement(p) for p in premises]
    concl = parse_statement(conclusion)
    negated = _negate(concl)
    clauses: Set[FrozenSet[Tuple]] = set()
    kinds: List[Tuple[str, str]] = []
    for k, (stmt, label) in enumerate(
            [(s, "premise") for s in stmts] + [(negated, "negated_conclusion")]):
        tag = k + 1
        clauses |= _skolem_clauses(stmt, tag)
        kinds.append((label, stmt.kind))
    saturated = set(clauses)
    steps = 0
    derivation: List[Dict[str, Any]] = []
    valid = False
    while steps < max_steps:
        current = sorted(saturated, key=_clause_key)
        new: Set[FrozenSet[Tuple]] = set()
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                for r in _resolvents(current[i], current[j]):
                    if r not in saturated:
                        new.add(r)
                        if len(derivation) < 25:
                            derivation.append({
                                "parents": [_clause_key(current[i])[:2],
                                            _clause_key(current[j])[:2]],
                                "resolvent": sorted(
                                    [[l[0], l[1], list(l[2])]
                                     for l in r])})
        steps += 1
        if frozenset() in new:
            valid = True
            break
        if not new or new <= saturated:
            break
        saturated |= new
    return {
        "method": "first-order resolution (Robinson 1965); Horn unary "
                  "fragment, decidable saturation",
        "valid": valid,
        "premises": [f"{s.kind} {s.subject} {s.cls}" for s in stmts],
        "negated_conclusion": f"{negated.kind} {negated.subject} "
                              f"{negated.cls}",
        "initial_clauses": len(clauses),
        "saturated_clauses": len(saturated),
        "iterations": steps,
        "derivation_prefix": derivation,
        "note": ("the empty clause was derived: no model makes the "
                 "premises true and the conclusion false"
                 if valid else
                 "saturation found no empty clause: the premises CAN "
                 "hold while the conclusion fails — invalid"),
    }

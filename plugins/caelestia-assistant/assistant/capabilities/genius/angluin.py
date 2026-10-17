"""genius.angluin — L* automaton learning for regex induction.

Angluin 1987, "Learning Regular Sets from Queries and Counterexamples",
Information and Control 75(2), 87-106 — the L* algorithm: an observation
table (S, E, T) refined by MEMBERSHIP queries, repaired to closedness and
consistency, conjectured as a DFA, and corrected by EQUIVALENCE queries
whose counterexamples re-enter the table. This module implements the
table machinery (closed/consistent repair, conjecture construction,
deterministic throughout) exactly as the paper defines it, with ONE
deliberate adaptation stated plainly below.

THE ORACLE HONESTY (the crux — read this before trusting any output):
L* is defined for a TOTAL membership oracle and a REAL equivalence
oracle. Neither exists when a user hands over labeled examples. So the
module ships two modes, and every result names its mode and its claim:

- EXAMPLES mode (induce_from_examples): the oracle is the labeled set,
  which is PARTIAL — most queries a total table would ask are
  unanswerable. This mode therefore does not pretend to run the full
  loop: it derives the CANONICAL CONSISTENT QUOTIENT over the labeled
  space (states = the distinct rows the labels actually determine,
  refined Moore-style until acceptance and per-symbol transitions are
  uniform within every class — the same object L*'s conjecture step
  builds when the oracle happens to be total on the table's cells).
  The result is ONE DFA consistent with every labeled example, and it
  says so: many other languages also fit the examples; no equivalence
  oracle exists to rule them out; nothing is claimed beyond the labels.
  Behavior in unlabeled territory routes to a designated UNKNOWN sink
  that accepts nothing and is dropped from the rendered regex — the
  rendered artifact separates the labeled space, full stop.

- TARGET mode (learn_from_target): the caller supplies a regex the
  machine may query; the membership oracle is the target's own
  fullmatch (stdlib re — a total oracle, honestly), and the
  equivalence oracle is APPROXIMATED by a deterministic conformance
  set: every string over the alphabet up to a length bound, plus the
  caller's examples, plus the table's own suffixes. Angluin's
  convergence guarantee needs a true equivalence oracle; a conformance
  set is a sound-but-incomplete stand-in (a mismatch is a genuine
  counterexample; a clean pass is NOT a proof of equality — the
  result reports exactly which set was checked). Counterexamples are
  processed with the classic all-suffixes-to-S augmentation (the
  treatment standard presentations of L* use; Rivest & Schapire 1993,
  "Inference of Finite Automata Using Homing Sequences", Inf. Comput.
  119(2), analyze the prefix/suffix variants and shrink the update —
  not done here, stated rather than hidden).

The regex RENDERING is by state elimination (the standard textbook
construction — eliminate non-start, non-final states in ascending
index order, alternation parts in sorted order, every literal escaped
via re.escape): deterministic, and CROSS-VALIDATED before return — the
rendered regex is re-compiled and checked against the DFA on the whole
conformance set; a rendering bug is a refusal, never a silent
mismatch.

Abstain/refuse paths everywhere: contradictory labels (the same string
labeled both ways) raise ValueError — never a guess; no positives AND
no negatives abstains (nothing to learn); a table that cannot close
within the state bound (default 64) refuses with the reason; an
alphabet too large for the conformance set refuses rather than
quietly shrinking the set; expression-size blowup in rendering refuses
rather than emitting megabyte regexes.

Pure module: no I/O, no execution, no RNG; same inputs -> same output
bytes (pinned by test).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

__all__ = ["AngluinError", "induce_from_examples", "learn_from_target",
           "dfa_to_regex", "conformance_strings"]

UNKNOWN = None  # a cell the partial oracle cannot answer (EXAMPLES mode)

DEFAULT_MAX_STATES = 64
_MAX_CONFORMANCE_STRINGS = 600
_MAX_RENDER_CHARS = 100_000


class AngluinError(ValueError):
    """Raised for abstentions, contradictions and budget refusals (the
    caller renders the reason; nothing is guessed)."""


# ---------------------------------------------------------------------------
# Shared DFA representation + rendering
# ---------------------------------------------------------------------------

class _DFA:
    """Deterministic finite automaton: states are 0..n-1 (0 = start),
    `accepting` is a frozenset of state ids, `delta[state][symbol]` is a
    state id (a total function — the UNKNOWN sink is an explicit state)."""

    __slots__ = ("alphabet", "delta", "accepting")

    def __init__(self, alphabet: Sequence[str], delta: List[Dict[str, int]],
                 accepting: FrozenSet[int]) -> None:
        self.alphabet = tuple(alphabet)
        self.delta = delta
        self.accepting = accepting

    def accepts(self, s: str) -> bool:
        state = 0
        for ch in s:
            nxt = self.delta[state].get(ch)
            if nxt is None:
                return False  # no transition: reject (sink semantics)
            state = nxt
        return state in self.accepting

    def size(self) -> int:
        return len(self.delta)


def _quote(ch: str) -> str:
    return re.escape(ch)


def _alt(parts: List[str]) -> str:
    """Deterministic alternation: sorted, deduplicated parts. The EMPTY
    STRING is a legitimate part (an epsilon path through the eliminated
    state) and is KEPT — filtering it would silently drop the empty
    word from the language, which the cross-validation exists to catch."""
    unique = sorted(set(parts))
    if not unique:
        raise AngluinError("alternation over an empty part set — internal "
                           "invariant violation, refusing to render")
    if len(unique) == 1:
        return unique[0]
    return "(?:" + "|".join(unique) + ")"


def _star(expr: str) -> str:
    if expr.startswith("(?:") and expr.endswith(")"):
        return expr + "*"
    return "(?:" + expr + ")*"


def _live_subgraph(dfa: _DFA) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """States that can still reach an accepting state (plus start).
    Dead states accept nothing, so dropping them preserves the language;
    transitions into dropped states are dropped with them."""
    n = dfa.size()
    live: Set[int] = set(dfa.accepting)
    changed = True
    while changed:
        changed = False
        for state in range(n):
            if state in live:
                continue
            for ch in dfa.alphabet:
                nxt = dfa.delta[state].get(ch)
                if nxt is not None and nxt in live:
                    live.add(state)
                    changed = True
                    break
    live.add(0)
    return dfa.delta, live


def dfa_to_regex(dfa: _DFA) -> str:
    """Render a DFA's language as a regex via state elimination.

    Construction (the standard textbook one, made deterministic):
    a fresh start S* with an epsilon edge to state 0, a fresh final F*
    with epsilon edges from every accepting state; eliminate every
    original state in ascending index order (smallest-state-number
    first); edge labels are concatenated in symbol order; alternation
    parts sorted; the empty word renders as the empty string. The
    result is cross-validated by the CALLERS of this module (and by
    tests) against the DFA — this function refuses on blowup rather
    than emitting an unusable expression.
    """
    delta, live = _live_subgraph(dfa)
    n = dfa.size()
    if not dfa.accepting:
        # The empty language: a regex that matches nothing. (?!) is a
        # zero-width negative lookahead — it fails everywhere, which is
        # exactly the claim; documented rather than hidden.
        return "(?!)"

    # edge[(u, v)] = regex label (None = no edge). States: 0..n-1 real,
    # n = fresh start, n+1 = fresh final.
    start, final = n, n + 1
    edges: Dict[Tuple[int, int], str] = {}
    edges[(start, 0)] = ""
    for a in dfa.accepting:
        edges[(a, final)] = ""
    for u in live:
        for ch in sorted(dfa.alphabet):
            v = delta[u].get(ch)
            if v is None or v not in live:
                continue
            key = (u, v)
            addition = _quote(ch)
            if key in edges:
                edges[key] = _alt([edges[key], addition])
            else:
                edges[key] = addition

    order = [s for s in range(n) if s in live]  # ascending, determinist.

    def self_loop_label(state: int) -> str:
        return edges.get((state, state), "")

    for state in order:
        loop = self_loop_label(state)
        loop_star = _star(loop) if loop else ""
        incoming = [(u, lab) for (u, v), lab in edges.items()
                    if v == state and u != state]
        outgoing = [(v, lab) for (u, v), lab in edges.items()
                    if u == state and v != state]
        for u, in_lab in incoming:
            for v, out_lab in outgoing:
                combined = in_lab + loop_star + out_lab
                key = (u, v)
                if key in edges:
                    edges[key] = _alt([edges[key], combined])
                else:
                    edges[key] = combined
        for key in [k for k in edges if k[0] == state or k[1] == state]:
            del edges[key]
        if sum(len(lab) for lab in edges.values()) > _MAX_RENDER_CHARS:
            raise AngluinError(
                "the state-elimination rendering exceeded the expression-"
                "size budget — this DFA's language has no compact regex "
                "under the deterministic construction; refusing instead "
                "of emitting an unusable expression")

    result = edges.get((start, final))
    if result is None:
        # No path start -> final survived elimination: empty language
        # after dead-state dropping (cannot happen when accepting is
        # non-empty and live, but refusing honestly beats guessing).
        raise AngluinError(
            "internal inconsistency: no start-to-final path survived "
            "state elimination — refusing to render")
    return result or "(?:)"


# ---------------------------------------------------------------------------
# Conformance sets (the equivalence-oracle approximation, deterministic)
# ---------------------------------------------------------------------------

def conformance_strings(alphabet: Sequence[str], max_len: int,
                        extras: Sequence[str] = ()) -> List[str]:
    """Every string over `alphabet` up to length `max_len`, plus extras,
    in deterministic order (length, then lexicographic), bounded: an
    alphabet whose full enumeration would exceed the string budget
    REFUSES with the reason (the honest alternative — silently shrinking
    the set — would overstate the equivalence check)."""
    sigma = sorted(set(alphabet))
    if not sigma:
        raise AngluinError("conformance needs a non-empty alphabet")
    total = sum(len(sigma) ** k for k in range(max_len + 1))
    if total > _MAX_CONFORMANCE_STRINGS:
        raise AngluinError(
            f"the conformance set over alphabet {sigma!r} up to length "
            f"{max_len} would need {total} strings (budget "
            f"{_MAX_CONFORMANCE_STRINGS}) — refusing rather than silently "
            "shrinking the equivalence check; pass a smaller alphabet or "
            "a shorter bound")
    out: List[str] = [""]
    frontier = [""]
    for _ in range(max_len):
        nxt = [s + ch for s in frontier for ch in sigma]
        out.extend(sorted(nxt))
        frontier = nxt
    for extra in extras:
        if extra not in out:
            out.append(extra)
    return out


# ---------------------------------------------------------------------------
# The observation table (target mode: total oracle, classic L*)
# ---------------------------------------------------------------------------

class _Table:
    def __init__(self, alphabet: Sequence[str],
                 membership: Callable[[str], bool]) -> None:
        self.alphabet = sorted(set(alphabet))
        self.membership = membership
        self.S: List[str] = [""]
        self.E: List[str] = [""]
        self._cache: Dict[str, bool] = {}

    def value(self, s: str) -> bool:
        if s not in self._cache:
            self._cache[s] = bool(self.membership(s))
        return self._cache[s]

    def row(self, s: str) -> Tuple[bool, ...]:
        return tuple(self.value(s + e) for e in self.E)

    def _row_of_some_s(self, target_row: Tuple[bool, ...]) -> Optional[str]:
        for s in self.S:
            if self.row(s) == target_row:
                return s
        return None

    def make_closed(self) -> None:
        while True:
            for s in list(self.S):
                for ch in self.alphabet:
                    r = self.row(s + ch)
                    if self._row_of_some_s(r) is None:
                        self.S.append(s + ch)
                        break
                else:
                    continue
                break
            else:
                return

    def make_consistent(self) -> None:
        while True:
            for i, s1 in enumerate(self.S):
                for s2 in self.S[i + 1:]:
                    if self.row(s1) != self.row(s2):
                        continue
                    for ch in self.alphabet:
                        r1, r2 = self.row(s1 + ch), self.row(s2 + ch)
                        if r1 == r2:
                            continue
                        # find the distinguishing column, add ch+e to E
                        for k, e in enumerate(self.E):
                            if r1[k] != r2[k]:
                                self.E.append(ch + e)
                                break
                        else:  # pragma: no cover - rows differ, loop found k
                            raise AngluinError("consistency repair failed "
                                               "to find a distinguishing "
                                               "column — internal error")
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                return

    def conjecture(self) -> _DFA:
        """The DFA the table defines: one state per distinct row over S,
        start = row(""), accepting = rows whose empty-suffix cell is 1
        (Dung-of-tables determinism: first-seen row order = state ids)."""
        row_index: Dict[Tuple[bool, ...], int] = {}
        state_of: Dict[str, int] = {}
        for s in self.S:
            r = self.row(s)
            if r not in row_index:
                row_index[r] = len(row_index)
            state_of[s] = row_index[r]
        n = len(row_index)
        delta = [{ch: 0 for ch in self.alphabet} for _ in range(n)]
        for s in self.S:
            for ch in self.alphabet:
                delta[state_of[s]][ch] = state_of[self._row_of_some_s(
                    self.row(s + ch)) or ""]
                # closed guarantees a matching S row; the `or ""` never
                # fires post-closure (kept only to satisfy the type).
        accepting = frozenset(state_of[s] for s in self.S
                               if self.value(s))
        return _DFA(self.alphabet, delta, accepting)


def learn_from_target(target: str, alphabet: Sequence[str],
                      examples: Sequence[str] = (),
                      max_states: int = DEFAULT_MAX_STATES,
                      max_len: int = 3,
                      max_rounds: int = 200) -> Dict[str, Any]:
    """TARGET mode: re-learn a regex from its own behavior.

    Membership oracle = the target's fullmatch (total, honest);
    equivalence oracle = the deterministic conformance set (all strings
    over `alphabet` up to `max_len` + `examples` + the table's suffixes)
    — a sound-but-INCOMPLETE stand-in: every reported counterexample is
    real, and a clean pass proves nothing beyond the set checked (the
    result reports the set size, never claiming target equality)."""
    try:
        pattern = re.compile(target)
    except re.error as exc:
        raise AngluinError(f"the target is not a valid regex: {exc}") from exc
    oracle = lambda s: pattern.fullmatch(s) is not None
    table = _Table(alphabet, oracle)
    conformance = conformance_strings(alphabet, max_len, examples)
    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            raise AngluinError(
                f"L* did not converge within {max_rounds} counterexample "
                "rounds for this target — refusing rather than looping")
        table.make_consistent()
        table.make_closed()
        dfa = table.conjecture()
        if dfa.size() > max_states:
            raise AngluinError(
                f"the conjectured DFA exceeded the state bound "
                f"({dfa.size()} > {max_states}) — this target is outside "
                "the module's stated capacity; refusing, not truncating")
        counterexample = None
        for s in conformance:
            if dfa.accepts(s) != oracle(s):
                counterexample = s
                break
        if counterexample is None:
            regex = dfa_to_regex(dfa)
            compiled = re.compile(regex)
            for s in conformance:  # rendering cross-validation
                if (compiled.fullmatch(s) is not None) != dfa.accepts(s):
                    raise AngluinError(
                        "the rendered regex disagrees with the learned DFA "
                        f"on {s!r} — a rendering bug, refusing to ship it")
            return {"mode": "target", "dfa": _dfa_payload(dfa),
                    "regex": regex, "rounds": rounds,
                    "states": dfa.size(),
                    "membership_queries": len(table._cache),
                    "conformance": {"checked": len(conformance),
                                    "max_len": max_len,
                                    "note": "sound but incomplete — "
                                            "equality is NOT claimed"},
                    "target": target}
        # classic counterexample processing: all suffixes of the
        # counterexample enter S (see the module docstring for the
        # variant discussion and citation)
        w = counterexample
        while w:
            if w not in table.S:
                table.S.append(w)
            w = w[:-1]


# ---------------------------------------------------------------------------
# EXAMPLES mode: the canonical consistent quotient over a partial oracle
# ---------------------------------------------------------------------------

def induce_from_examples(positives: Sequence[str],
                         negatives: Sequence[str],
                         max_states: int = DEFAULT_MAX_STATES
                         ) -> Dict[str, Any]:
    """EXAMPLES mode: ONE DFA consistent with the labeled examples.

    The labels are a PARTIAL oracle: the states are the distinct rows
    the labels actually determine (over the prefixes of the labeled
    strings as S, their suffixes as E, "" always included), refined
    Moore-style until acceptance and per-symbol transition targets are
    uniform inside every class — exactly the quotient L*'s conjecture
    step builds when the oracle is total on the table's cells. Strings
    the labels do not determine route to an UNKNOWN sink that accepts
    nothing and is dropped from the rendered regex. The claim carried
    by the result is PRECISELY: this DFA separates the labeled space —
    many other languages also fit the examples, no equivalence oracle
    exists to rule them out, and the result says so in its note."""
    pos = [str(s) for s in positives]
    neg = [str(s) for s in negatives]
    if not pos and not neg:
        raise AngluinError(
            "no positives and no negatives — nothing to learn; abstaining")
    labels: Dict[str, bool] = {}
    for s in pos:
        if s in labels and labels[s] is not True:
            raise AngluinError(
                f"contradictory labels: {s!r} appears in both positives "
                "and negatives — refusing to guess which is right")
        labels[s] = True
    for s in neg:
        if s in labels and labels[s] is not False:
            raise AngluinError(
                f"contradictory labels: {s!r} appears in both positives "
                "and negatives — refusing to guess which is right")
        labels[s] = False

    labeled = sorted(labels)
    alphabet = sorted({ch for s in labeled for ch in s})
    # S: every prefix of every labeled string, plus "" (sorted — the
    # deterministic row order); E is implicit: the signature below IS
    # the row the labels determine (own label + successor structure).
    S: Set[str] = {""}
    for s in labeled:
        for k in range(1, len(s) + 1):
            S.add(s[:k])
    S = sorted(S)

    SINK = -1
    klass: Dict[str, int] = {s: 0 for s in S}
    while True:
        # signature: (own label or None, successor class per symbol,
        # where successors inside S carry their class and everything
        # else carries SINK). One pass; split until stable.
        signatures: Dict[str, Tuple] = {}
        for s in S:
            parts: List[Any] = [labels.get(s)]
            for ch in alphabet:
                succ = s + ch
                parts.append(klass[succ] if succ in klass else SINK)
            signatures[s] = tuple(parts)
        groups: Dict[Tuple, List[str]] = {}
        for s in S:
            groups.setdefault(signatures[s], []).append(s)
        new_klass: Dict[str, int] = {}
        for cid, (_sig, members) in enumerate(
                sorted(groups.items(), key=lambda kv: min(kv[1]))):
            for m in members:
                new_klass[m] = cid
        stable = len(set(new_klass.values())) == len(set(klass.values()))
        klass = new_klass
        if stable:
            break

    n = len(set(klass.values()))
    if n > max_states:
        raise AngluinError(
            f"the consistent quotient needs {n} states (bound "
            f"{max_states}) — too many for this module's stated "
            "capacity; add fewer or cleaner examples, or raise the bound")

    # Acceptance per class: uniform by refinement (the own label is in
    # the signature); classes whose members are ALL unlabeled reject —
    # the conservative choice, stated in the docstring.
    class_accept: Dict[int, bool] = {}
    for s in S:
        lab = labels.get(s)
        if lab is not None:
            if klass[s] in class_accept and class_accept[klass[s]] != lab:
                raise AngluinError(
                    "internal inconsistency: class acceptance is not "
                    "uniform after refinement — refusing")
            class_accept[klass[s]] = lab
    accepting = frozenset(c for c, a in class_accept.items() if a)

    sink = n  # the UNKNOWN territory: total, accepting nothing
    delta: List[Dict[str, int]] = [dict() for _ in range(n + 1)]
    # Transitions are well-defined per class (the successor class is in
    # the signature, so every member agrees); read them off the
    # canonical (smallest) member.
    canonical: Dict[int, str] = {}
    for s in S:
        if klass[s] not in canonical or s < canonical[klass[s]]:
            canonical[klass[s]] = s
    for cid, rep in sorted(canonical.items()):
        for ch in alphabet:
            succ = rep + ch
            delta[cid][ch] = klass[succ] if succ in klass else sink
    for ch in alphabet:
        delta[sink][ch] = sink

    dfa = _DFA(alphabet, delta, accepting)
    # The construction guarantee, verified rather than assumed:
    for s in labeled:
        if dfa.accepts(s) != labels[s]:
            raise AngluinError(
                f"internal inconsistency: the quotient DFA disagrees "
                f"with the label of {s!r} — refusing to ship it")
    regex = dfa_to_regex(dfa)
    compiled = re.compile(regex)
    for s in labeled:  # rendering cross-validation
        if (compiled.fullmatch(s) is not None) != labels[s]:
            raise AngluinError(
                f"the rendered regex disagrees with the label of {s!r} "
                "— a rendering bug, refusing to ship it")
    return {"mode": "examples", "dfa": _dfa_payload(dfa),
            "regex": regex, "states": n + 1,  # + the UNKNOWN sink
            "labeled": len(labeled),
            "note": "ONE separator consistent with the labeled examples — "
                    "not the unique language; unlabeled behavior carries "
                    "no claim (UNKNOWN sink, dropped from the regex)"}


def _dfa_payload(dfa: _DFA) -> Dict[str, Any]:
    return {"alphabet": list(dfa.alphabet),
            "start": 0,
            "accepting": sorted(dfa.accepting),
            "delta": [{ch: tgt for ch, tgt in sorted(delta.items())}
                      for delta in dfa.delta]}

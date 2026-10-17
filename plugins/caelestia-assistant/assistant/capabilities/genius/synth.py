"""genius.synth — inductive program synthesis over string examples, scaled
past the 2-3-example regime by version-space algebra.

Gulwani 2011, "Automating String Processing in Spreadsheets Using
Input-Output Examples", POPL 2011 — FlashFill scales past candidate
enumeration by representing the programs consistent with each example
as a VERSION SPACE and intersecting those spaces across examples
instead of testing each program against each example. This module is
that idea over the miniature explainability-first DSL below; nothing
is invented here, the representation is just the classic one.

REPRESENTATION (stated plainly, because the scaling claim depends on
it): the concrete cross product of stage expressions is NEVER
materialized. Instead —
- the HYPOTHESIS MENU is the anchor piece set: every DSL expression
  whose value on the FIRST example is a non-empty substring of that
  example's output (this prefilter is unchanged from the original
  2-3-example implementation; it defines which constructs are even
  candidates);
- each menu piece's TRACE is its tuple of values, one per example;
  pieces with identical traces form one TRACE CLASS — they are
  interchangeable in every consistent program, which is the
  version-space sharing that collapses coinciding constructs
  (1st-vs-last delimiter, head-vs-prefix, ...) into one object;
- each example's VERSION SPACE is its set of ABSTRACT SPLITS: the
  ways to cut that example's output into m stage values joined by
  m-1 constant joiners, restricted to menu-realized values and
  deduplicated by VALUE. It is built once per example and per stage
  count, bounded by _MAX_SPLITS, and kept as a membership set. Its
  size is bounded by the OUTPUT LENGTH (split positions), not by the
  number of expressions — enumerating it costs O(len(output)^2) for
  2 stages and O(len(output)^2 * distinct values) for 3, versus the
  naive #pieces**stages cross product;
- INTERSECTION: an abstract program (trace-class tuple + joiners) is
  consistent with the examples exactly when, for every example, its
  value-slice is a member of that example's pre-built space. The
  search expands the ANCHOR example's splits depth-first over trace
  classes with per-example prefix pruning (a partial program's
  concatenation so far must prefix every remaining output), and each
  surviving abstract program counts its concrete programs
  ARITHMETICALLY — the product of its trace-class sizes — so the
  ambiguity count is exact without enumeration.
This is why 8-12 examples are comfortable: the work is
O(examples x splits x small class products), linear in the example
count, with hard budgets (_MAX_SPLITS, _MAX_VS_NODES) that turn the
old implementation's combinatorial hang into a fast honest refusal.

The DSL (explainability over coverage, on purpose — every construct
has a plain-English rendering):
- a program is 1..3 STAGES joined by CONSTANT strings (possibly empty);
- each stage extracts from the before-string of the example with one of:
    prefix(i, d, k)  text before the k-th occurrence of delimiter d
    suffix(i, d, k)  text after the k-th occurrence of delimiter d
    between(i, d1, k1, d2, k2)  the text after the k1-th d1 and before
                     the k2-th d2 (occurrence indices 1, 2, or -1 = last)
    head(i, n) / tail(i, n)     the first / last n characters
- delimiters are the non-alphanumeric characters already present in the
  examples (ranked by frequency, capped at 5) — nothing is invented.

The inductive check (the honesty core): a candidate program must
reproduce EVERY example EXACTLY, and the joiner constants must be
identical across all examples; anything the DSL cannot explain is an
explicit ABSTAIN ("not guessable from these examples in this DSL") —
the same refuse-don't-clamp discipline as everywhere else. Multiple
consistent programs are reported as ambiguous (count included, never
hidden); the returned one is the canonical smallest under a stated
rule: FEWEST STAGES first, then the lexicographically smallest tuple
of stage expressions in the canonical construct order (delimiter-
ranked prefix/suffix with occurrence 1, 2, last, then between, then
head/tail by length), then the LEFTMOST joiner split. That rule is
the original implementation's enumeration order, kept byte-for-byte
in behavior — it is at least as honest as any other fixed tie-break
because it prefers the SIMPLEST program first and states its rule.

Output safety: the transformation's application to new strings is pure
text computation. When the user supplies real target names, the module
renders `SUGGESTED_NOT_EXECUTED:` mv lines (risk tier STATE_CHANGING)
— inert strings the user copies and runs themselves, exactly the
troubleshooting layers' convention. Nothing here touches the
filesystem, spawns, or executes: pure string algebra, deterministic,
no RNG, no I/O.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["SynthError", "synthesize", "apply_program", "render",
           "suggest_renames"]

MAX_STAGES = 3
MIN_EXAMPLES = 2
MAX_EXAMPLES = 32
_MAX_DELIMS = 5
_MAX_OCC = 3          # occurrence indices 1, 2, -1
_MAX_HEADTAIL = 8
_MAX_SPLITS = 20_000      # per-example abstract-split bound, per stage count
_MAX_VS_NODES = 120_000   # search-visit bound, per stage count


class SynthError(ValueError):
    """Raised for abstentions and malformed requests (the caller renders
    the reason; nothing is guessed)."""


# ---------------------------------------------------------------------------
# Expression enumeration and evaluation (the DSL itself is unchanged).
# ---------------------------------------------------------------------------

def _delims(samples: Sequence[str]) -> List[str]:
    """Non-alphanumeric characters present in the examples — whitespace
    included, it separates fields like any other character — ranked by
    total frequency (deterministic: frequency desc, then codepoint)."""
    counts: Dict[str, int] = {}
    for s in samples:
        for ch in s:
            if not ch.isalnum():
                counts[ch] = counts.get(ch, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ch for ch, _n in ranked[:_MAX_DELIMS]]


def _occurrences(s: str, d: str) -> List[int]:
    out: List[int] = []
    start = 0
    while len(out) < _MAX_OCC:
        idx = s.find(d, start)
        if idx == -1:
            break
        out.append(idx)
        start = idx + len(d)
    return out


def _nth_occ(s: str, d: str, k: int) -> Optional[int]:
    """Position where the k-th occurrence of d starts (k 1-based; -1 =
    last). None when the occurrence does not exist."""
    occs = _occurrences(s, d)
    if k == -1:
        return occs[-1] if occs else None
    if 1 <= k <= len(occs):
        return occs[k - 1]
    return None


def _eval_expr(expr: Dict[str, Any], inputs: Sequence[str]) -> str:
    """Evaluate one stage expression against the inputs. A missing
    delimiter/occurrence yields "" — the candidate simply fails to
    explain anything non-empty (verification does the rest)."""
    kind = expr["kind"]
    i = int(expr.get("input", 1)) - 1
    if not (0 <= i < len(inputs)):
        return ""
    s = inputs[i]
    if kind == "head":
        return s[:max(0, int(expr["n"]))]
    if kind == "tail":
        n = max(0, int(expr["n"]))
        return s[len(s) - n:] if n <= len(s) else s
    d1 = str(expr.get("d1", ""))
    k1 = int(expr.get("k1", 1))
    p1 = _nth_occ(s, d1, k1)
    if p1 is None:
        return ""
    if kind == "prefix":
        return s[:p1]
    if kind == "suffix":
        return s[p1 + len(d1):]
    if kind == "between":
        d2 = str(expr.get("d2", ""))
        k2 = int(expr.get("k2", 1))
        p2 = _nth_occ(s, d2, k2)
        if p2 is None or p2 <= p1:
            return ""
        return s[p1 + len(d1):p2]
    return ""


def _exprs_for_input(idx: int, sample: str,
                     delims: Sequence[str]) -> List[Dict[str, Any]]:
    """Canonical enumeration order for one input's expressions."""
    out: List[Dict[str, Any]] = []
    for d in delims:
        for k in (1, 2, -1):
            out.append({"kind": "prefix", "input": idx, "d1": d, "k1": k})
            out.append({"kind": "suffix", "input": idx, "d1": d, "k1": k})
    for d1 in delims:
        for k1 in (1, 2, -1):
            for d2 in delims:
                for k2 in (1, 2, -1):
                    if d1 == d2 and k1 == k2:
                        continue
                    out.append({"kind": "between", "input": idx,
                                "d1": d1, "k1": k1, "d2": d2, "k2": k2})
    for n in range(1, min(_MAX_HEADTAIL, len(sample)) + 1):
        out.append({"kind": "head", "input": idx, "n": n})
        out.append({"kind": "tail", "input": idx, "n": n})
    return out


def _explainable_pieces(inputs_per_example: Sequence[Sequence[str]],
                        out: str,
                        delims: Sequence[str]
                        ) -> List[Tuple[Dict[str, Any], int]]:
    """(expr, length) candidates whose value for example 1 is a NON-EMPTY
    substring of example 1's output — the anchor prefilter that defines
    the hypothesis menu (unchanged from the original implementation)."""
    pieces: List[Tuple[Dict[str, Any], int]] = []
    inputs = inputs_per_example[0]
    for idx, s in enumerate(inputs, start=1):
        for expr in _exprs_for_input(idx, s, delims):
            v = _eval_expr(expr, inputs)
            if v and v in out:
                pieces.append((expr, len(v)))
    return pieces


# ---------------------------------------------------------------------------
# Version-space algebra (Gulwani 2011): per-example abstract split spaces
# + trace-class intersection. The concrete expression cross product is
# never enumerated; program counts are computed arithmetically.
# ---------------------------------------------------------------------------

# An abstract split: (stage value tuple, joiner tuple).
_AbstractKey = Tuple[Tuple[str, ...], Tuple[str, ...]]
# A surviving abstract program: (trace-class id tuple, joiner tuple).
_Node = Tuple[Tuple[int, ...], Tuple[str, ...]]


def _split_space(out: str, menu: Set[str], m: int,
                 bound: int) -> Tuple[List[_AbstractKey], Set[_AbstractKey]]:
    """One example's abstract version space for m-stage programs: every
    way to cut `out` into m stage values joined by m-1 constant joiners
    such that each stage value is REALIZED by some menu piece on this
    example. The menu carries "" only when a menu piece legitimately
    evaluates to "" here (a delimiter absent from this example's
    before-string). Keys are unique per cut; the returned list preserves
    deterministic construction order and the set gives O(1) membership —
    the space is built ONCE per example and reused for every
    intersection test."""
    keys: List[_AbstractKey] = []
    seen: Set[_AbstractKey] = set()
    L = len(out)
    if m == 1:
        if out in menu:
            keys.append(((out,), ()))
        return keys, set(keys)
    if m == 2:
        for p in range(L + 1):
            w1 = out[:p]
            if w1 not in menu:
                continue
            for q in range(p, L + 1):
                w2 = out[q:]
                if w2 not in menu:
                    continue
                key = ((w1, w2), (out[p:q],))
                if key not in seen:
                    if len(keys) >= bound:
                        raise SynthError(
                            f"the abstract split space of one example "
                            f"exceeds the version-space bound ({bound}) — "
                            "the output is too long or too rich for this "
                            "DSL's representation; abstaining rather than "
                            "enumerating past the bound")
                    seen.add(key)
                    keys.append(key)
        return keys, seen
    # m == 3: prefix cut p, suffix cut q, and a placement (r, s) of the
    # middle value strictly between them; j1 = out[p:r], j2 = out[s:q].
    sorted_menu = sorted(menu)
    for p in range(L + 1):
        w1 = out[:p]
        if w1 not in menu:
            continue
        for q in range(p, L + 1):
            w3 = out[q:]
            if w3 not in menu:
                continue
            for w2 in sorted_menu:
                if w2 == "":
                    for r in range(p, q + 1):
                        key = ((w1, w2, w3),
                               (out[p:r], out[r:q]))
                        if key not in seen:
                            if len(keys) >= bound:
                                raise SynthError(
                                    f"the abstract split space of one "
                                    f"example exceeds the version-space "
                                    f"bound ({bound}) — the output is too "
                                    "long or too rich for this DSL's "
                                    "representation; abstaining rather "
                                    "than enumerating past the bound")
                            seen.add(key)
                            keys.append(key)
                    continue
                pos = out.find(w2, p)
                while pos != -1 and pos + len(w2) <= q:
                    key = ((w1, w2, w3),
                           (out[p:pos], out[pos + len(w2):q]))
                    if key not in seen:
                        if len(keys) >= bound:
                            raise SynthError(
                                f"the abstract split space of one "
                                f"example exceeds the version-space "
                                f"bound ({bound}) — the output is too "
                                "long or too rich for this DSL's "
                                "representation; abstaining rather "
                                "than enumerating past the bound")
                        seen.add(key)
                        keys.append(key)
                    pos = out.find(w2, pos + 1)
    return keys, seen


def _search(m: int, anchor_keys: Sequence[_AbstractKey],
            by_value: Dict[str, List[int]],
            classes: Sequence[Tuple[Tuple[str, ...], List[int]]],
            spaces: Sequence[Set[_AbstractKey]],
            vals: Sequence[Tuple[str, ...]],
            outs: Sequence[str], n_examples: int,
            bound: int) -> Tuple[List[_Node], int]:
    """Intersect the anchor example's abstract splits with every other
    example's pre-built space. Depth-first over trace classes (a class
    is a group of menu pieces with identical value traces — the
    version-space sharing), pruned per example: the concatenation so
    far must be a PREFIX of every remaining output; a full assignment
    survives exactly when its value-slice is a member of each
    example's space. Returns the surviving abstract programs and the
    number of search visits (budget accounting — the concrete cross
    product is never built)."""
    nodes: List[_Node] = []
    visits = 0

    def visit(t: int, cur: Tuple[int, ...], prefix: List[str]) -> None:
        nonlocal visits
        visits += 1
        if visits > bound:
            raise SynthError(
                f"the version-space search for a {m}-stage program "
                f"exceeded the node budget ({bound}) — these examples "
                "underconstrain the program space for this DSL; add "
                "distinguishing examples rather than enumerate past "
                "the bound")
        if t == m:
            # full program: intersection = membership in EVERY
            # example's pre-built space. cur holds CLASS ids, so the
            # value slice reads each class's own trace (classes[c][0]),
            # never the piece that happens to sit at index c.
            for j in range(1, n_examples):
                slice_key = (tuple(classes[c][0][j] for c in cur), joiners)
                if slice_key not in spaces[j]:
                    return
            nodes.append((cur, joiners))
            return
        for cid in stage_lists[t]:
            tr = classes[cid][0]
            nxt = cur + (cid,)
            # partial: the concatenation of the stages chosen so far
            # (with their joiners, and the joiner that precedes this
            # stage) must be a PREFIX of every remaining output — a
            # necessary condition on any consistent completion, so it
            # prunes the class expansion without losing programs
            new_prefix: List[str]
            if t == 0:
                ok = m == 1
                if not ok:
                    ok = all(outs[j].startswith(tr[j])
                             for j in range(1, n_examples))
                if not ok:
                    continue
                new_prefix = list(tr)
            else:
                ok = True
                for j in range(1, n_examples):
                    cand = prefix[j] + joiners[t - 1] + tr[j]
                    if not outs[j].startswith(cand):
                        ok = False
                        break
                if not ok:
                    continue
                new_prefix = [prefix[j] + joiners[t - 1] + tr[j]
                              for j in range(n_examples)]
            visit(t + 1, nxt, new_prefix)

    for w, joiners in anchor_keys:
        stage_lists = [by_value.get(x, []) for x in w]
        if any(not lst for lst in stage_lists):
            continue
        visit(0, (), ["" for _ in range(n_examples)])
    return nodes, visits


def _node_order_key(node: _Node,
                    classes: Sequence[Tuple[Tuple[str, ...], List[int]]]
                    ) -> Tuple[int, ...]:
    """The canonical tie-break key: the lexicographically smallest
    concrete expansion (each trace class's first member, in canonical
    piece order), then the leftmost joiner split (shortest first
    joiner) — exactly the original enumeration's candidate order."""
    firsts = [classes[c][1][0] for c in node[0]]
    return tuple(firsts) + tuple(len(j) for j in node[1])


def synthesize(examples: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """Induce ONE transformation from 2..MAX_EXAMPLES before/after
    examples via version-space intersection (see the module docstring
    for the representation).

    Returns {"program": ..., "rendered": ..., "checked": n,
             "alternatives": k, "ambiguous": bool, "stages": m,
             "version_space": {...}}. Raises SynthError (an honest
    ABSTAIN) when the request is malformed, the DSL cannot explain the
    examples, or a version-space budget is exceeded (the old
    implementation would have hung there instead)."""
    pairs = [(str(a), str(b)) for a, b in examples]
    if not (MIN_EXAMPLES <= len(pairs) <= MAX_EXAMPLES):
        raise SynthError(
            f"synthesis needs exactly {MIN_EXAMPLES}-{MAX_EXAMPLES} "
            f"before/after examples (got {len(pairs)}); fewer cannot "
            "induce, more is outside this tool's contract")
    if any(not a or not b for a, b in pairs):
        raise SynthError("empty before/after strings carry no signal")
    ins = [a for a, _b in pairs]
    outs = [b for _a, b in pairs]
    delims = _delims(ins + outs)
    # one BEFORE-string per example (the expr schema keeps an input
    # index for forward compatibility, but this API supervises exactly
    # one input column — v1)
    inputs_per_example = [[pairs[j][0]] for j in range(len(pairs))]
    n_examples = len(pairs)

    # hypothesis menu: the anchor pieces (example 1 prefilter — the
    # same candidate set the original implementation enumerated)
    pieces = _explainable_pieces(inputs_per_example, outs[0], delims)
    exprs = [e for e, _ln in pieces]
    # per-piece traces and trace classes (the version-space sharing)
    vals = [tuple(_eval_expr(e, inputs_per_example[j])
                  for j in range(n_examples))
            for e in exprs]
    class_index: Dict[Tuple[str, ...], int] = {}
    classes: List[Tuple[Tuple[str, ...], List[int]]] = []
    for idx, tr in enumerate(vals):
        cid = class_index.get(tr)
        if cid is None:
            cid = len(classes)
            class_index[tr] = cid
            classes.append((tr, []))
        classes[cid][1].append(idx)
    menus: List[Set[str]] = [
        {vals[i][j] for i in range(len(exprs))} for j in range(n_examples)]
    by_value: Dict[str, List[int]] = {}
    for cid, (tr, _members) in enumerate(classes):
        by_value.setdefault(tr[0], []).append(cid)

    chosen: Optional[_Node] = None
    chosen_m = 0
    total = 0
    space_sizes: List[int] = []
    for m in range(1, MAX_STAGES + 1):
        # per-example spaces, constructed once for this stage count
        spaces: List[Set[_AbstractKey]] = []
        anchor_keys: Optional[List[_AbstractKey]] = None
        for j in range(n_examples):
            keys, keyset = _split_space(outs[j], menus[j], m, _MAX_SPLITS)
            spaces.append(keyset)
            if j == 0:
                anchor_keys = keys
        space_sizes = [len(s) for s in spaces]
        if not anchor_keys:
            continue
        nodes, _visits = _search(m, anchor_keys, by_value, classes,
                                 spaces, vals, outs, n_examples,
                                 _MAX_VS_NODES)
        if nodes:
            chosen = min(nodes, key=lambda n: _node_order_key(n, classes))
            chosen_m = m
            total = 0
            for cids, _joiners in nodes:
                prod = 1
                for c in cids:
                    prod *= len(classes[c][1])
                total += prod
            break

    if chosen is None:
        raise SynthError(
            "no transformation in this DSL reproduces every example — "
            "abstaining, not guessing (try examples that share a "
            "delimiter pattern)")
    stages = [exprs[classes[c][1][0]] for c in chosen[0]]
    program = {"stages": stages, "joiners": list(chosen[1])}
    return {"program": program,
            "rendered": render(program),
            "checked": len(pairs),
            "alternatives": max(0, total - 1),
            "ambiguous": total > 1,
            "stages": chosen_m,
            "version_space": {"pieces": len(exprs),
                              "trace_classes": len(classes),
                              "programs": total,
                              "example_spaces": space_sizes}}


def apply_program(program: Dict[str, Any], inputs: Sequence[str]) -> str:
    """Apply the learned transformation to new input strings (pure text
    computation; the same expression evaluator training used). Refuses
    (never silently empties) when the program reads an input that was
    not supplied — an honest error, never a fabricated string."""
    stages = program.get("stages") or []
    joiners = program.get("joiners") or []
    if not stages:
        raise SynthError("no program to apply")
    supplied = len(inputs)
    needed = max(int(e.get("input", 1)) for e in stages)
    if needed > supplied:
        raise SynthError(
            f"the learned program reads input {needed} but only "
            f"{supplied} input string(s) were supplied — refusing to "
            "invent the missing part")
    vals = [_eval_expr(e, [str(s) for s in inputs]) for e in stages]
    acc = vals[0]
    for m in range(1, len(vals)):
        acc += (joiners[m - 1] if m - 1 < len(joiners) else "") + vals[m]
    return acc


# ---------------------------------------------------------------------------
# Rendering (every construct keeps its plain-English form).
# ---------------------------------------------------------------------------

_OCC_WORD = {1: "1st", 2: "2nd", -1: "last"}


def _render_expr(expr: Dict[str, Any]) -> str:
    i = int(expr.get("input", 1))
    kind = expr["kind"]
    d1 = expr.get("d1", "")
    k1 = int(expr.get("k1", 1))
    pos = _OCC_WORD.get(k1, f"{k1}th")
    if kind == "prefix":
        return f"text before the {pos} {d1!r} of input {i}"
    if kind == "suffix":
        return f"text after the {pos} {d1!r} of input {i}"
    if kind == "between":
        d2 = expr.get("d2", "")
        k2 = int(expr.get("k2", 1))
        pos2 = _OCC_WORD.get(k2, f"{k2}th")
        return (f"text between the {pos} {d1!r} and the {pos2} {d2!r} "
                f"of input {i}")
    if kind == "head":
        return f"the first {expr['n']} character(s) of input {i}"
    return f"the last {expr['n']} character(s) of input {i}"


def render(program: Dict[str, Any]) -> str:
    stages = program.get("stages") or []
    joiners = program.get("joiners") or []
    parts = [_render_expr(e) for e in stages]
    out = parts[0]
    for m in range(1, len(parts)):
        out += f"  +  {joiners[m - 1]!r}  +  " + parts[m]
    return out


def _shell_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def suggest_renames(program: Dict[str, Any],
                    names: Sequence[str]) -> List[str]:
    """Inert, copy-paste-ready mv lines (STATE_CHANGING tier) pairing
    each given name with the program's output for it. NEVER executed —
    the prefix is the whole contract; `mv -n` never clobbers."""
    lines: List[str] = []
    for name in names:
        new = apply_program(program, [name])
        if new == name:
            continue  # a no-op rename is not a suggestion
        lines.append(f"SUGGESTED_NOT_EXECUTED: [STATE_CHANGING] "
                     f"mv -n -- {_shell_quote(name)} {_shell_quote(new)}")
    return lines

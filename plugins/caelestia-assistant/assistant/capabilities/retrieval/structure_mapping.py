"""retrieval.structure_mapping — structure-mapping analogical retrieval.

Gentner 1983, "Structure-Mapping: A Theoretical Framework for Analogy",
Cognitive Science 7(2), 155-170 — the two claims this module
instantiates: (1) analogy is driven by RELATIONAL STRUCTURE, not
surface features — object attributes and entity identity are discarded
in favor of shared relations; (2) the SYSTEMATICITY PRINCIPLE —
connected systems of relations governed by higher-order relations
(AND/OR combinators, here) are preferred over isolated, unconnected
matches.

What was adapted: Gentner's theory is a cognitive-science account of
analogy, not an algorithm. The correspondence search below is a
deliberately tiny deterministic RELATIVE of the Structure-Mapping
Engine — Falkenhainer, Forbus & Gentner 1989, "The Structure-Mapping
Engine: Algorithm and Examples", Artificial Intelligence 41(1), 1-63 —
which likewise constructs matches greedily (local match hypotheses
combined into global matches) instead of exhaustively optimizing over
the space of entity correspondences, a space that is combinatorial
(worst-case factorial) in the problem size. This module is smaller
still: two deterministic greedy passes with fixed tie-breaks, and NO
optimality claim — the correspondence it returns is the greedy one,
not the proven-maximal one. Honesty about that bound is why problems
are capped at MAX_RELATIONS relations (above that: ValueError, refuse
rather than degrade).

Representation: each problem is a small PREDICATE GRAPH

    {"entities": ["input", "~/.config/foo", ...],
     "relations": [{"name": str, "args": [entity-or-literal, ...]}]}

- an arg that is a STRING listed in "entities" is an ENTITY (an object
  in the domain — the pasted text, a file path, a command);
- a string NOT in "entities" is a LITERAL (surface content: patterns,
  expected values — DISCARDED by the matcher, that is the theory);
- an arg that is an INT is a REFERENCE to relations[i] — the shape of
  a SECOND-ORDER relation (a relation over relations), which is how
  allOf/anyOf combinator structure becomes matchable structure that
  the systematicity principle can prefer. Membership in "entities" is
  the ONLY thing that distinguishes an entity from a literal — a
  typo'd entity name therefore reads as a literal; the module does not
  guess which was meant.

The builder rule_match_graph(rule) turns a diagnostics rule dict
(engine.py's matcher-tree shape over rules.d) into an honest graph:
each matched leaf becomes a predicate over the rule's matched
text/file/cmd — text_regex(pattern, input), text_substring(value,
input), file_present(file)/file_absent(file) (the expected flag
changes what is asserted, so it belongs to the predicate NAME, not to
a discarded literal), cmd_output(pattern, command) — and each
allOf/anyOf combinator becomes an AND/OR second-order relation whose
args are indices of the relations it governs (emitted post-order, so
references always point backwards from the builder).

Scoring (weights are module constants, fixed and documented):

- RELATION_MATCH_WEIGHT per matched relation pair (same name, same
  arity, position-wise same argument ROLES — entity/entity,
  literal/literal, ref/ref; argument IDENTITY is never required);
- CONNECTEDNESS_WEIGHT per shared corresponding entity between two
  matched relations (matched relations that share mapped objects are a
  connected system, not a coincidence pile);
- SYSTEMATICITY_WEIGHT per matched relation pair nested under a
  matched second-order relation (a matched AND/OR system beats the
  same matches isolated).

Entity or attribute matches contribute ZERO by themselves — that is
the theory's whole point, pinned by test: two problems with zero token
overlap but identical relational structure outscore two problems with
high token overlap but different structure.

This is a SIGNAL, not a ranker: surface_candidates() returns analogical
candidates to be read ALONGSIDE BM25/NCD retrieval — never instead of
it (every payload says so). No filesystem access: caller-supplied
problem graphs plus the rule-dict convenience. Deterministic: no RNG,
no I/O, fixed tie-breaks (second-order systems first, then smallest
relation name, then argument signature, then indices).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["MAX_RELATIONS", "MIN_CANDIDATE_SCORE", "match",
           "rule_match_graph", "surface_candidates"]

# Honesty over completeness (see module docstring): exact structure-
# mapping is combinatorial (SME itself is greedy for this reason —
# Falkenhainer, Forbus & Gentner 1989); this module caps the problem
# size instead of pretending to scale.
MAX_RELATIONS = 64

# A single isolated first-order match (score 1.0) carries no connected
# structure; ranking it as an "analogical candidate" would be noise.
# Below this score surface_candidates abstains (empty list + reason).
MIN_CANDIDATE_SCORE = 1.5

RELATION_MATCH_WEIGHT = 1.0
CONNECTEDNESS_WEIGHT = 0.5
SYSTEMATICITY_WEIGHT = 1.0

_LEAF_ARG_HELP = "relation args must be strings (entities or literals) or ints (relation indices)"


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------

def _normalize(problem: Any, label: str) -> Dict[str, Any]:
    """Validate one problem graph and return its normalized form:
    {"entities": set[str], "relations": [{"name", "args": [(kind, value)],
    "second_order": bool}]} where kind is "entity" | "literal" | "ref"."""
    if not isinstance(problem, dict):
        raise ValueError(f"{label}: a problem graph must be a dict, got {type(problem).__name__}")
    for key in ("entities", "relations"):
        if key not in problem:
            raise ValueError(f"{label}: missing required key {key!r}")
    entities = problem["entities"]
    relations = problem["relations"]
    if not isinstance(entities, list) or not all(isinstance(e, str) and e for e in entities):
        raise ValueError(f"{label}: 'entities' must be a list of non-empty strings")
    if len(set(entities)) != len(entities):
        raise ValueError(f"{label}: duplicate entity names — rejected, never silently merged")
    if not isinstance(relations, list):
        raise ValueError(f"{label}: 'relations' must be a list")
    if len(relations) > MAX_RELATIONS:
        raise ValueError(
            f"{label}: {len(relations)} relations exceeds the {MAX_RELATIONS}-relation "
            "cap — exact structure-mapping over the correspondence space is "
            "combinatorial (Falkenhainer, Forbus & Gentner 1989 build SME "
            "greedily for exactly this reason); this module refuses to "
            "degrade silently instead of guessing")
    entity_set = set(entities)
    normalized: List[Dict[str, Any]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict) or "name" not in relation or "args" not in relation:
            raise ValueError(f"{label}: relation {index} must be a dict with 'name' and 'args'")
        name = relation["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}: relation {index} needs a non-empty string name")
        if not isinstance(relation["args"], list):
            raise ValueError(f"{label}: relation {index} args must be a list")
        args: List[Tuple[str, Any]] = []
        for position, value in enumerate(relation["args"]):
            if isinstance(value, bool):
                raise ValueError(f"{label}: relation {index} arg {position}: {_LEAF_ARG_HELP}")
            if isinstance(value, int):
                if not 0 <= value < len(relations):
                    raise ValueError(
                        f"{label}: relation {index} arg {position} references relation "
                        f"{value}, which is outside this graph's {len(relations)} relation(s)")
                args.append(("ref", value))
            elif isinstance(value, str):
                args.append(("entity" if value in entity_set else "literal", value))
            else:
                raise ValueError(f"{label}: relation {index} arg {position}: {_LEAF_ARG_HELP}")
        normalized.append({"name": name, "args": args,
                           "second_order": any(kind == "ref" for kind, _ in args)})
    return {"entities": entity_set, "relations": normalized}


def _is_candidate(rel_a: Dict[str, Any], rel_b: Dict[str, Any]) -> bool:
    """Same relation name, same arity, position-wise same argument ROLES
    (structural role — never argument identity or literal content)."""
    if rel_a["name"] != rel_b["name"] or len(rel_a["args"]) != len(rel_b["args"]):
        return False
    return all(kind_a == kind_b
               for (kind_a, _), (kind_b, _) in zip(rel_a["args"], rel_b["args"]))


def _entity_pairs(rel_a: Dict[str, Any], rel_b: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Position-wise (entity_a, entity_b) pairs a candidate match asserts."""
    return [(value_a, value_b)
            for (kind_a, value_a), (kind_b, value_b) in zip(rel_a["args"], rel_b["args"])
            if kind_a == "entity" and kind_b == "entity"]


def _signature(relation: Dict[str, Any]) -> Tuple[str, ...]:
    """A deterministic, comparable rendering of a relation's args."""
    return tuple(f"{kind}:{value}" for kind, value in relation["args"])


# ---------------------------------------------------------------------------
# match — greedy best-first correspondence search
# ---------------------------------------------------------------------------

def match(problem_a: Dict[str, Any], problem_b: Dict[str, Any]) -> Dict[str, Any]:
    """The structural correspondence between two problems, with Gentner's
    score. Deterministic greedy best-first, TWO passes:

    - pass A (systematicity): second-order candidate pairs (AND/OR
      systems), smallest name first; a combinator match COMMITS its
      whole connected system — the pair, plus transitively every child
      pair its ref args name — and is addable only when every pair in
      that closure is itself a candidate and the entity correspondence
      stays one-to-one. A combinator never matches "hollow": a matched
      AND/OR IS its matched children.
    - pass B (relational): first-order candidate pairs, one at a time,
      same tie-breaks (smallest relation name, then argument signature,
      then indices), each addable only if the entity correspondence
      stays one-to-one.

    The result is the GREEDY correspondence, not a proven maximum (see
    module docstring). Returns {"structural_score", "matched_relations",
    "entity_correspondence", "n_relations_a", "n_relations_b", "why"}.
    """
    norm_a = _normalize(problem_a, "problem_a")
    norm_b = _normalize(problem_b, "problem_b")
    rel_a, rel_b = norm_a["relations"], norm_b["relations"]

    candidates = [(i, j)
                  for i, ra in enumerate(rel_a)
                  for j, rb in enumerate(rel_b)
                  if _is_candidate(ra, rb)]
    candidate_set = set(candidates)

    def _priority(pair: Tuple[int, int]) -> Tuple[Any, ...]:
        i, j = pair
        order = 0 if rel_a[i]["second_order"] else 1
        return (order, rel_a[i]["name"], _signature(rel_a[i]), i,
                _signature(rel_b[j]), j)

    matched_a: Dict[int, int] = {}   # relation index a -> b
    matched_b: Dict[int, int] = {}   # relation index b -> a
    correspondence: Dict[str, str] = {}       # entity a -> entity b
    inverse: Dict[str, str] = {}             # entity b -> entity a

    def _entity_ok(pairs: Sequence[Tuple[int, int]]) -> Optional[Dict[str, str]]:
        """The entity mappings the pairs would add, or None on conflict
        (an entity mapped to two entities, or two entities to one)."""
        additions: Dict[str, str] = {}
        additions_inverse: Dict[str, str] = {}
        for i, j in pairs:
            for entity_a, entity_b in _entity_pairs(rel_a[i], rel_b[j]):
                if correspondence.get(entity_a, entity_b) != entity_b:
                    return None
                if inverse.get(entity_b, entity_a) != entity_a:
                    return None
                if additions.get(entity_a, entity_b) != entity_b:
                    return None
                if additions_inverse.get(entity_b, entity_a) != entity_a:
                    return None
                additions[entity_a] = entity_b
                additions_inverse[entity_b] = entity_a
        return additions

    def _closure(i: int, j: int) -> List[Tuple[int, int]]:
        """The connected system a second-order match commits: (i, j)
        plus, transitively, every child pair its ref args name."""
        system: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        stack = [(i, j)]
        while stack:
            pair = stack.pop()
            if pair in seen:
                continue
            seen.add(pair)
            system.append(pair)
            for (kind_a, value_a), (kind_b, value_b) in zip(rel_a[pair[0]]["args"],
                                                            rel_b[pair[1]]["args"]):
                if kind_a == "ref" and kind_b == "ref":
                    stack.append((value_a, value_b))
        return sorted(system)

    # --- pass A: second-order systems, systematicity first -----------
    for i, j in sorted(candidates, key=_priority):
        if not rel_a[i]["second_order"] or i in matched_a or j in matched_b:
            continue
        system = _closure(i, j)
        ok = all((pi, pj) in candidate_set for pi, pj in system)
        if ok:
            for pi, pj in system:
                if (pi in matched_a and matched_a[pi] != pj) or \
                        (pj in matched_b and matched_b[pj] != pi):
                    ok = False
                    break
        if not ok:
            continue
        additions = _entity_ok(system)
        if additions is None:
            continue
        for pi, pj in system:
            matched_a[pi] = pj
            matched_b[pj] = pi
        correspondence.update(additions)
        inverse.update({entity_b: entity_a for entity_a, entity_b in additions.items()})

    # --- pass B: first-order relations, one at a time -----------------
    for i, j in sorted(candidates, key=_priority):
        if rel_a[i]["second_order"] or i in matched_a or j in matched_b:
            continue
        additions = _entity_ok([(i, j)])
        if additions is None:
            continue
        matched_a[i] = j
        matched_b[j] = i
        correspondence.update(additions)
        inverse.update({entity_b: entity_a for entity_a, entity_b in additions.items()})

    # --- score the established correspondence -------------------------
    matched_pairs = sorted(matched_a.items())
    score = RELATION_MATCH_WEIGHT * len(matched_pairs)

    systematicity = 0
    for i, j in matched_pairs:
        if not rel_a[i]["second_order"]:
            continue
        for (kind_a, value_a), (kind_b, value_b) in zip(rel_a[i]["args"], rel_b[j]["args"]):
            if kind_a == "ref" and kind_b == "ref" and matched_a.get(value_a) == value_b:
                systematicity += 1
    score += SYSTEMATICITY_WEIGHT * systematicity

    def _entities_of(relation: Dict[str, Any]) -> Set[str]:
        return {value for kind, value in relation["args"] if kind == "entity"}

    connectedness = 0
    for x in range(len(matched_pairs)):
        for y in range(x + 1, len(matched_pairs)):
            connectedness += len(_entities_of(rel_a[matched_pairs[x][0]])
                                 & _entities_of(rel_a[matched_pairs[y][0]]))
    score += CONNECTEDNESS_WEIGHT * connectedness

    matched_relations = [
        {"name": rel_a[i]["name"], "a": i, "b": j,
         "second_order": rel_a[i]["second_order"]}
        for i, j in matched_pairs
    ]
    return {
        "structural_score": round(score, 6),
        "matched_relations": matched_relations,
        "entity_correspondence": dict(sorted(correspondence.items())),
        "n_relations_a": len(rel_a),
        "n_relations_b": len(rel_b),
        "why": _explain(matched_pairs, rel_a, systematicity, connectedness),
    }


def _explain(matched_pairs: Sequence[Tuple[int, int]], rel_a: Sequence[Dict[str, Any]],
             systematicity: int, connectedness: int) -> str:
    """The deterministic human-readable match summary."""
    if not matched_pairs:
        return ("no relational overlap — shared surface features alone do "
                "not match (Gentner 1983)")
    counts: Dict[str, int] = {}
    for i, _j in matched_pairs:
        counts[rel_a[i]["name"]] = counts.get(rel_a[i]["name"], 0) + 1
    listing = ", ".join(name if count == 1 else f"{name} x{count}"
                        for name, count in sorted(counts.items()))
    parts = [f"{len(matched_pairs)} matched relation(s): {listing}"]
    if systematicity:
        parts.append(f"{systematicity} of them nested under matched "
                     "second-order AND/OR structure (systematicity)")
    if connectedness:
        parts.append(f"{connectedness} shared-entity bridge(s) between "
                     "matched relations (connectedness)")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# rule_match_graph — honest predicates from a diagnostics rule
# ---------------------------------------------------------------------------

def rule_match_graph(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Build a predicate graph from a diagnostics rule dict (the
    engine.py matcher-tree shape over rules.d):

    - text_regex(pattern, input), text_substring(value, input) —
      predicates over the user's pasted text (entity "input");
    - file_present(file) / file_absent(file) — the file_exists leaf,
      with its expected flag deciding the predicate NAME (a file that
      must exist and a file that must be absent are different
      assertions, not a discarded literal); the path is an ENTITY, so
      two leaves probing the same file share an object;
    - cmd_output(pattern, command) — a predicate over a pasted
      command's output; the command is an ENTITY;
    - allOf/anyOf become AND/OR second-order relations whose args are
      the indices of the relations they govern (emitted post-order, so
      references point backwards);
    - case_insensitive flags are deliberately NOT predicates: they
      refine HOW a pattern matches, not WHAT is related.

    No filesystem access and no evaluation — the rule dict is
    caller-supplied (engine.load_rules() in production), and this
    builder only reshapes it.
    """
    if not isinstance(rule, dict) or "matchers" not in rule or "id" not in rule:
        raise ValueError("rule must be a diagnostics rule dict with 'id' and 'matchers'")
    relations: List[Dict[str, Any]] = []
    entities: Set[str] = set()

    def _leaf_relation(leaf: Any) -> Dict[str, Any]:
        if not isinstance(leaf, dict):
            raise ValueError(f"{rule['id']}: matcher leaf must be a dict")
        kind = leaf.get("type")
        if kind == "text_regex":
            pattern = leaf.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"{rule['id']}: text_regex leaf needs a pattern")
            entities.add("input")
            return {"name": "text_regex", "args": [pattern, "input"]}
        if kind == "text_substring":
            value = leaf.get("value")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{rule['id']}: text_substring leaf needs a value")
            entities.add("input")
            return {"name": "text_substring", "args": [value, "input"]}
        if kind == "file_exists":
            path = leaf.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{rule['id']}: file_exists leaf needs a path")
            expected = leaf.get("expected", "present")
            if expected not in ("present", "absent"):
                raise ValueError(f"{rule['id']}: file_exists expected must be "
                                 f"'present' or 'absent', got {expected!r}")
            entities.add(path)
            return {"name": f"file_{expected}", "args": [path]}
        if kind == "cmd_output":
            command = leaf.get("command")
            pattern = leaf.get("pattern")
            if not isinstance(command, str) or not command:
                raise ValueError(f"{rule['id']}: cmd_output leaf needs a command")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"{rule['id']}: cmd_output leaf needs a pattern")
            entities.add(command)
            return {"name": "cmd_output", "args": [pattern, command]}
        raise ValueError(f"{rule['id']}: cannot build an honest predicate for "
                         f"matcher type {kind!r}")

    def _build(node: Any) -> int:
        """Emit relations for a matcher node; return its relation index
        (combinators are appended after their members — post-order)."""
        if not isinstance(node, dict):
            raise ValueError(f"{rule['id']}: matcher nodes must be dicts")
        if "allOf" in node or "anyOf" in node:
            name = "AND" if "allOf" in node else "OR"
            members = node["allOf"] if "allOf" in node else node["anyOf"]
            if not isinstance(members, list) or not members:
                raise ValueError(f"{rule['id']}: an empty {name} combinator "
                                 "asserts nothing — refusing to build it")
            child_indices = [_build(member) for member in members]
            relations.append({"name": name, "args": child_indices})
            return len(relations) - 1
        relations.append(_leaf_relation(node))
        return len(relations) - 1

    _build(rule["matchers"])
    if len(relations) > MAX_RELATIONS:
        raise ValueError(
            f"{rule['id']}: {len(relations)} relations exceeds the "
            f"{MAX_RELATIONS}-relation cap — refused rather than degraded")
    return {"id": rule["id"], "entities": sorted(entities), "relations": relations}


# ---------------------------------------------------------------------------
# surface_candidates — the analogical signal, alongside (never replacing)
# ---------------------------------------------------------------------------

def surface_candidates(problems: Sequence[Dict[str, Any]],
                       query_problem: Dict[str, Any], top: int = 3) -> Dict[str, Any]:
    """Rank analogical candidates for one query problem.

    problems: a sequence of {"id": str, "problem": graph} — the ids are
    the caller's join keys back to whatever else it knows (rule ids,
    case ids); duplicate ids are rejected. Each result carries
    {"id", "structural_score", "matched_relations", "why"}.

    Candidates below MIN_CANDIDATE_SCORE are dropped; when none reach
    the bar the module ABSTAINS (empty results + reason) — a weak
    structural overlap is an honest nothing, not a noise ranking. The
    payload always says this signal rides ALONGSIDE BM25/NCD, never
    replacing them.
    """
    if not isinstance(top, int) or isinstance(top, bool) or top < 1:
        raise ValueError("top must be an integer >= 1")
    _normalize(query_problem, "query_problem")
    seen: Set[str] = set()
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for position, item in enumerate(problems):
        if not isinstance(item, dict) or "id" not in item or "problem" not in item:
            raise ValueError('each problem must be {"id": str, "problem": '
                             f'graph}} — problems[{position}] is not')
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError(f"problems[{position}]: id must be a non-empty string")
        if item["id"] in seen:
            raise ValueError(f"duplicate problem id {item['id']!r} — rejected, "
                             "never silently merged")
        seen.add(item["id"])
        result = match(query_problem, item["problem"])
        if result["structural_score"] >= MIN_CANDIDATE_SCORE:
            entries.append((item["id"], result))
    ranked = sorted(entries, key=lambda pair: (-pair[1]["structural_score"], pair[0]))
    results = [
        {
            "id": problem_id,
            "structural_score": result["structural_score"],
            "matched_relations": result["matched_relations"],
            "why": result["why"],
        }
        for problem_id, result in ranked[:top]
    ]
    if not results:
        return {
            "status": "abstained",
            "results": [],
            "reason": (f"no candidate reaches the structural bar "
                       f"(>= {MIN_CANDIDATE_SCORE}): relational overlap is "
                       "too thin to rank — an honest empty answer, not a "
                       "noise ranking"),
            "note": "analogical signal only — read it ALONGSIDE BM25/NCD, "
                    "never instead of them",
        }
    return {
        "status": "ok",
        "results": results,
        "reason": None,
        "note": "analogical signal only — read it ALONGSIDE BM25/NCD, "
                "never instead of them",
    }

"""diagnostics.rete — a forward-chaining Rete network beside the rule engine.

Forgy 1982, "Rete: A Fast Algorithm for the Two-Variable Pattern Matching
Problem", Artificial Intelligence 19(1), pp. 17-37 — the classic Rete design,
as adapted here:

- FACTS are JSON-serializable triples {"type": str, "key": str,
  "value": str}; fact identity IS the triple (the first payload seen for a
  triple wins).
- A PATTERN is the same shape with value "*" allowed as a WILDCARD matching
  any value for that key (only the value component may be wildcarded).
- The ALPHA network keys one node per distinct pattern (type, key,
  value-or-"*"), with an alpha MEMORY shared by every rule and premise
  position using that pattern — the alpha-memory reuse property is pinned
  by test (one assert feeds every consumer exactly once).
- The BETA network is a per-rule chain of JOIN nodes, each with a LEFT
  memory (partial tokens) and a RIGHT memory (facts received from its
  alpha node). Our patterns are constant triples with no variables that
  bind across premises, so a join performs an unfiltered cross product of
  its left tokens and right facts (stated plainly: no Forgy-style
  variable-binding tests); the two memories still give the incremental,
  assert-once propagation Rete exists for.
- A TERMINAL node per rule fires on each complete token (one fact consumed
  per premise, in premise order) and asserts the rule's conclusion fact
  carrying its derivation: the rule id plus the exact facts consumed.

Derived-vs-given: a terminal's conclusion is recorded with source
"derived" and re-traverses the network exactly like an asserted fact
(source "given"). Re-deriving an already-present fact is a NO-OP — the
fact is not re-added and its first derivation is kept — which, together
with each distinct fact entering each alpha memory exactly once,
guarantees fixpoint termination for cyclic rule sets (pinned by test with
a deliberately cyclic rule pair).

Honesty: a rule whose premises reference facts no producer supplies simply
never fires; zero premises matched means zero derived facts and no
invented conclusions. Rules with an EMPTY premise list are rejected at
construction — an unconditional "conclusion" is an invention, not a
derivation. Dormant rules are LISTED with their first unsatisfied premise
(dormant_rules()), never hidden.

The shipped COMPOSITE_RULES derive composite conclusions from REAL rules.d
rule ids only: each premise is a match fact the CLI layer derives from the
engine's own gate-passing candidates, each conclusion is a new "derived"
fact, and each rationale/citation composes the underlying rules' own
claims and references (docs/TROUBLESHOOTING.md sections + issue numbers
exactly as the underlying rules carry them) — no new factual claim about
the shell is made here.

Pure module: no I/O, no execution, nothing written; deterministic by
construction (every iteration follows insertion order).
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["COMPOSITE_RULES", "ReteNetwork", "WILDCARD", "derive",
           "derived_evidence", "facts_from_diagnosis", "render_derived_lines"]

WILDCARD = "*"

_FACT_KEYS = frozenset({"type", "key", "value"})
_RULE_KEYS = frozenset({"id", "premises", "conclusion", "rationale", "citation"})

# Defensive cap: a rule set that can mint unboundedly many NEW distinct
# facts (e.g. a counter-incrementing conclusion) has no fixpoint; run()
# refuses to loop forever and says so instead of truncating silently.
_MAX_DERIVED = 10_000


# ---------------------------------------------------------------------------
# Validation (strict: out-of-range is REJECTED with ValueError, never fixed)
# ---------------------------------------------------------------------------

def _validate_triple(node: Any, role: str, allow_wildcard: bool) -> Tuple[str, str, str]:
    """Validate one fact/pattern dict; returns its (type, key, value) triple."""
    if not isinstance(node, dict):
        raise ValueError(f"{role} must be a dict, got {type(node).__name__}")
    if set(node) != _FACT_KEYS:
        raise ValueError(
            f"{role} must have exactly the keys {sorted(_FACT_KEYS)}, "
            f"got {sorted(node)}")
    for part in ("type", "key", "value"):
        value = node[part]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{role} field {part!r} must be a non-empty string, got {value!r}")
    if node["type"] == WILDCARD or node["key"] == WILDCARD:
        raise ValueError(
            f"{role}: only the VALUE component may be wildcarded ({WILDCARD!r}); "
            f"got type={node['type']!r} key={node['key']!r}")
    if node["value"] == WILDCARD and not allow_wildcard:
        raise ValueError(f"{role} cannot carry the wildcard {WILDCARD!r} as its value")
    return (node["type"], node["key"], node["value"])


def _validate_rule(rule: Any, index: int) -> Dict[str, Any]:
    """Validate one composite rule; returns it unchanged."""
    where = f"rules[{index}]"
    if not isinstance(rule, dict):
        raise ValueError(f"{where} must be a dict, got {type(rule).__name__}")
    if set(rule) != _RULE_KEYS:
        raise ValueError(
            f"{where} must have exactly the keys {sorted(_RULE_KEYS)}, "
            f"got {sorted(rule)}")
    if not isinstance(rule["id"], str) or not rule["id"]:
        raise ValueError(f"{where}: 'id' must be a non-empty string")
    for text_field in ("rationale", "citation"):
        if not isinstance(rule[text_field], str) or not rule[text_field]:
            raise ValueError(f"{where} ({rule['id']}): {text_field!r} must be a non-empty string")
    premises = rule["premises"]
    if not isinstance(premises, list) or not premises:
        raise ValueError(
            f"{where} ({rule['id']}): 'premises' must be a NON-EMPTY list — a rule "
            "with zero premises would fire unconditionally, and an unconditional "
            "conclusion is an invention, not a derivation")
    for i, premise in enumerate(premises):
        _validate_triple(premise, f"{where} premise {i}", allow_wildcard=True)
    _validate_triple(rule["conclusion"], f"{where} conclusion", allow_wildcard=False)
    return rule


# ---------------------------------------------------------------------------
# Network nodes (private; the public surface is ReteNetwork below)
# ---------------------------------------------------------------------------

class _AlphaNode:
    """One distinct pattern; its memory holds every fact that ever matched."""

    __slots__ = ("pattern", "memory", "successors")

    def __init__(self, pattern: Tuple[str, str, str]) -> None:
        self.pattern = pattern
        self.memory: List[Tuple[str, str, str]] = []  # arrival order
        self.successors: List["_JoinNode"] = []


class _JoinNode:
    """One premise position of one rule; joins left tokens with right facts."""

    __slots__ = ("premise", "alpha", "parent", "child", "terminal",
                 "left_memory", "right_memory")

    def __init__(self, premise: Tuple[str, str, str], alpha: _AlphaNode,
                 parent: Optional["_JoinNode"]) -> None:
        self.premise = premise
        self.alpha = alpha
        self.parent = parent          # None = this join reads the beta root
        self.child: Optional["_JoinNode"] = None
        self.terminal: Optional[_TerminalNode] = None
        self.left_memory: List[Tuple[Tuple[str, str, str], ...]] = []
        self.right_memory: List[Tuple[str, str, str]] = []


class _TerminalNode:
    """End of one rule's beta chain; fires on each complete token."""

    __slots__ = ("rule", "network")

    def __init__(self, rule: Dict[str, Any], network: "ReteNetwork") -> None:
        self.rule = rule
        self.network = network


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class ReteNetwork:
    """A Rete network over one rule set; facts in, derived facts out.

    Typical use: assert the given facts, run() to fixpoint, then read the
    derived facts (in derivation order), explain() any of them back to the
    given facts, and list which rules stayed dormant.
    """

    def __init__(self, rules: Sequence[Dict[str, Any]]) -> None:
        self._rules: List[Dict[str, Any]] = []
        self._rule_by_id: Dict[str, Dict[str, Any]] = {}
        for index, raw in enumerate(rules):
            rule = _validate_rule(raw, index)
            if rule["id"] in self._rule_by_id:
                raise ValueError(f"duplicate rule id {rule['id']!r}")
            self._rules.append(rule)
            self._rule_by_id[rule["id"]] = rule

        # Alpha nodes shared by pattern; beta chains per rule.
        self._alpha: Dict[Tuple[str, str, str], _AlphaNode] = {}
        self._terminals: List[_TerminalNode] = []
        for rule in self._rules:
            joins: List[_JoinNode] = []
            for premise in rule["premises"]:
                pattern = (premise["type"], premise["key"], premise["value"])
                alpha = self._alpha.get(pattern)
                if alpha is None:
                    alpha = _AlphaNode(pattern)
                    self._alpha[pattern] = alpha
                join = _JoinNode(pattern, alpha, joins[-1] if joins else None)
                alpha.successors.append(join)
                if joins:
                    joins[-1].child = join
                joins.append(join)
            terminal = _TerminalNode(rule, self)
            joins[-1].terminal = terminal
            self._terminals.append(terminal)

        # Working memory: triple -> record, in arrival order (given first,
        # then derived in derivation order).
        self._wm: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._agenda: List[Tuple[str, str, str]] = []
        self._agenda_cursor = 0
        self._derived_order: List[Tuple[str, str, str]] = []
        self._terminal_log: List[Dict[str, Any]] = []
        self._fired: Dict[str, int] = {}
        self._fired_order: List[str] = []

    # -- assertion -----------------------------------------------------

    def assert_fact(self, fact: Dict[str, str]) -> bool:
        """Record one fact as GIVEN (source="given").

        Returns True if the fact was new, False if an identical triple was
        already present (a no-op: the fact does not re-enter any alpha
        memory, so no terminal re-fires — pinned by test). The only other
        source, "derived", is recorded by the network itself when a
        terminal fires.
        """
        triple = _validate_triple(fact, role="fact", allow_wildcard=False)
        if triple in self._wm:
            return False
        self._wm[triple] = {"fact": {"type": triple[0], "key": triple[1],
                                     "value": triple[2]},
                            "source": "given", "derivation": None}
        self._agenda.append(triple)
        return True

    # -- inference -----------------------------------------------------

    def run(self) -> List[Dict[str, str]]:
        """Propagate the agenda to fixpoint; returns facts derived by THIS
        call, in derivation order.

        Termination: each distinct fact enters each alpha memory at most
        once, so each (token, fact) combination is emitted at most once;
        a terminal whose conclusion is already present records the firing
        and derives nothing. A rule set that mints unboundedly many NEW
        distinct facts has no fixpoint; run() refuses at _MAX_DERIVED with
        an explicit error rather than looping (or truncating) silently.
        """
        start = len(self._derived_order)
        while self._agenda_cursor < len(self._agenda):
            triple = self._agenda[self._agenda_cursor]
            self._agenda_cursor += 1
            self._route(triple)
            if len(self._derived_order) > _MAX_DERIVED:
                raise RuntimeError(
                    f"rete: refused to continue past {_MAX_DERIVED} derived "
                    "facts — this rule set mints unboundedly many new facts "
                    "and has no fixpoint; refusing rather than looping or "
                    "truncating silently (rule-authoring bug)")
        return [self._wm[t]["fact"] for t in self._derived_order[start:]]

    def _route(self, triple: Tuple[str, str, str]) -> None:
        """Send one fact to every alpha node whose pattern it matches."""
        for (ptype, pkey, pvalue), alpha in self._alpha.items():
            if triple[0] == ptype and triple[1] == pkey and (
                    pvalue == WILDCARD or triple[2] == pvalue):
                alpha.memory.append(triple)
                for join in alpha.successors:
                    join.right_memory.append(triple)
                    if join.parent is None:
                        # Root join: the beta root contributes the empty
                        # token, so this fact alone completes premise 0.
                        self._pass_on(join, (triple,))
                    else:
                        for token in list(join.left_memory):
                            self._pass_on(join, token + (triple,))

    def _pass_on(self, join: _JoinNode,
                 token: Tuple[Tuple[str, str, str], ...]) -> None:
        """A token has grown at `join`; send it downstream."""
        if join.child is not None:
            child = join.child
            child.left_memory.append(token)
            for right in list(child.right_memory):
                self._pass_on(child, token + (right,))
        elif join.terminal is not None:
            self._conclude(join.terminal, token)

    def _conclude(self, terminal: _TerminalNode,
                  token: Tuple[Tuple[str, str, str], ...]) -> None:
        """One rule's premises are all satisfied by `token`; fire."""
        rule = terminal.rule
        conclusion = rule["conclusion"]
        triple = (conclusion["type"], conclusion["key"], conclusion["value"])
        if triple in self._wm:
            outcome = "already-present"  # no-op: first derivation is kept
        else:
            self._wm[triple] = {
                "fact": dict(conclusion), "source": "derived",
                "derivation": {"rule": rule["id"], "premises": list(token)},
            }
            self._derived_order.append(triple)
            self._agenda.append(triple)
            outcome = "derived"
        self._terminal_log.append({
            "rule": rule["id"], "premises": list(token),
            "conclusion": {"type": triple[0], "key": triple[1],
                           "value": triple[2]},
            "outcome": outcome,
        })
        if rule["id"] not in self._fired:
            self._fired_order.append(rule["id"])
        self._fired[rule["id"]] = self._fired.get(rule["id"], 0) + 1

    # -- inspection ------------------------------------------------------

    def facts(self) -> List[Dict[str, str]]:
        """Every fact in working memory, in arrival order (given first)."""
        return [record["fact"] for record in self._wm.values()]

    def explain(self, fact: Dict[str, str]) -> Dict[str, Any]:
        """The full derivation tree of one fact, chained back to given facts.

        Returns {"fact", "source", "rule", "rationale", "citation",
        "premises": [...]}. Given facts are leaves (rule None); derived
        facts carry the rule that produced them and the trees of the exact
        facts its premises consumed. Only the FIRST derivation of a fact is
        recorded (re-derivations are no-ops), so the tree is a DAG by
        construction; a cycle would be an internal invariant violation and
        raises. Unknown facts are REJECTED with ValueError.
        """
        triple = _validate_triple(fact, role="fact", allow_wildcard=False)
        if triple not in self._wm:
            raise ValueError(
                f"cannot explain unknown fact {dict(fact)!r}: it was never "
                "asserted or derived in this network")
        return self._explain_triple(triple, ())

    def _explain_triple(self, triple: Tuple[str, str, str],
                        seen: Tuple[Tuple[str, str, str], ...]) -> Dict[str, Any]:
        if triple in seen:
            raise RuntimeError(
                f"derivation cycle detected at {triple!r} — internal "
                "invariant violation (a derivation may only consume facts "
                "that predate it); please report")
        record = self._wm[triple]
        node: Dict[str, Any] = {
            "fact": record["fact"], "source": record["source"],
            "rule": None, "rationale": None, "citation": None,
            "premises": [],
        }
        if record["source"] == "derived":
            derivation = record["derivation"]
            rule = self._rule_by_id[derivation["rule"]]
            node["rule"] = rule["id"]
            node["rationale"] = rule["rationale"]
            node["citation"] = rule["citation"]
            node["premises"] = [
                self._explain_triple(p, seen + (triple,))
                for p in derivation["premises"]
            ]
        return node

    def alpha_memory(self, pattern: Dict[str, str]) -> List[Dict[str, str]]:
        """The facts held by one pattern's alpha memory (shared across every
        rule using that pattern — the Rete alpha-memory reuse surface)."""
        triple = _validate_triple(pattern, role="pattern", allow_wildcard=True)
        alpha = self._alpha.get(triple)
        if alpha is None:
            return []
        return [self._wm[t]["fact"] for t in alpha.memory]

    def terminal_log(self) -> List[Dict[str, Any]]:
        """Every terminal firing, in firing order, with its outcome."""
        return [dict(entry) for entry in self._terminal_log]

    def fired_rules(self) -> List[str]:
        """Rule ids that fired at least once, in first-fire order."""
        return list(self._fired_order)

    def dormant_rules(self) -> List[Dict[str, Any]]:
        """Rules that never fired, each with its first unsatisfied premise.

        A rule is dormant when some premise pattern has no matching fact in
        working memory (with no cross-premise variables this is exactly the
        firing condition, so a dormant rule is a rule whose premises
        reference facts no producer supplied — listed, not hidden).
        """
        dormant: List[Dict[str, Any]] = []
        for rule in self._rules:
            if rule["id"] in self._fired:
                continue
            missing: Optional[Dict[str, str]] = None
            for pattern in rule["premises"]:
                if not self._pattern_matched(pattern):
                    missing = {"type": pattern["type"], "key": pattern["key"],
                               "value": pattern["value"]}
                    break
            if missing is None:
                # Unreachable after a complete run (all premises matched
                # implies the rule fired); reported rather than hidden.
                dormant.append({"rule": rule["id"], "unsatisfied_premise": None,
                                "note": "premises satisfied but rule never "
                                        "fired (internal inconsistency)"})
            else:
                dormant.append({"rule": rule["id"], "unsatisfied_premise": missing})
        return dormant

    def _pattern_matched(self, pattern: Dict[str, str]) -> bool:
        for (ftype, fkey, fvalue) in self._wm:
            if (ftype == pattern["type"] and fkey == pattern["key"]
                    and (pattern["value"] == WILDCARD
                         or fvalue == pattern["value"])):
                return True
        return False


# ---------------------------------------------------------------------------
# The shipped composite rules (real rules.d ids only; citations compose the
# underlying rules' own references — no new factual claim about the shell)
# ---------------------------------------------------------------------------

COMPOSITE_RULES: List[Dict[str, Any]] = [
    {
        "id": "CR-color-war-001",
        "premises": [
            {"type": "match", "key": "rule", "value": "CL-config-kmycolors-001"},
            {"type": "match", "key": "rule", "value": "CL-config-schemereset-001"},
        ],
        "conclusion": {"type": "derived", "key": "combined-diagnosis",
                       "value": "legacy-color-service-plus-missing-scheme-handoff"},
        "rationale": (
            "Both documented color problems matched at once. "
            "CL-config-kmycolors-001: the legacy kde-material-you-colors service "
            "fights the shell's scheme pipeline (its fix: stop and disable that "
            "service, confirm the apply-loop stopped, re-run the autostart step). "
            "CL-config-schemereset-001: the color variant falls back to Tonal Spot "
            "after reboot because the automatic scheme hand-off is off (its fix: "
            "enable 'Automatic Color scheme', update the shell). The two fixes do "
            "not conflict — one disables a legacy user service, the other enables "
            "a shell setting — so both apply."),
        "citation": (
            "docs/TROUBLESHOOTING.md §3.7 (issue #14) and §3.4 (issue #763) — "
            "the references carried by CL-config-kmycolors-001 and "
            "CL-config-schemereset-001"),
    },
    {
        "id": "CR-legacy-color-cleanup-001",
        "premises": [
            {"type": "match", "key": "rule", "value": "CL-config-kmycolors-001"},
            {"type": "match", "key": "rule", "value": "CL-folder-cleanup-legacy-001"},
        ],
        "conclusion": {"type": "derived", "key": "combined-fix",
                       "value": "disable-legacy-service-and-remove-stale-materialyou-schemes"},
        "rationale": (
            "Two matched rules converge on the same legacy migration. "
            "CL-config-kmycolors-001 disables the kde-material-you-colors user "
            "service; CL-folder-cleanup-legacy-001 also disables it and marks the "
            "generated MaterialYou*.colors scheme files safe_to_delete: yes. One "
            "disable satisfies both rules' service step; removing the stale "
            "MaterialYou color files is the cleanup rule's own addition."),
        "citation": (
            "docs/TROUBLESHOOTING.md §3.7 (issue #14) — the shared reference of "
            "CL-config-kmycolors-001 and CL-folder-cleanup-legacy-001"),
    },
    {
        "id": "CR-color-war-002",
        "premises": [
            {"type": "derived", "key": "combined-diagnosis",
             "value": "legacy-color-service-plus-missing-scheme-handoff"},
            {"type": "match", "key": "rule", "value": "CL-folder-cleanup-legacy-001"},
        ],
        "conclusion": {"type": "derived", "key": "migration-plan",
                       "value": "full-legacy-material-you-exit"},
        "rationale": (
            "With both color faults derived (CR-color-war-001) and the "
            "legacy-cleanup rule also matched, the full documented migration is "
            "in play. Every step is a fix step of one of the three underlying "
            "rules: disable kde-material-you-colors (CL-config-kmycolors-001 / "
            "CL-folder-cleanup-legacy-001), remove the stale MaterialYou*.colors "
            "files (CL-folder-cleanup-legacy-001), enable 'Automatic Color "
            "scheme' (CL-config-schemereset-001), re-run the autostart step "
            "(CL-config-kmycolors-001)."),
        "citation": (
            "docs/TROUBLESHOOTING.md §3.7, §3.4; issues #14, #763 — carried by "
            "CL-config-kmycolors-001, CL-config-schemereset-001 and "
            "CL-folder-cleanup-legacy-001"),
    },
    {
        "id": "CR-shell-unit-convergence-001",
        "premises": [
            {"type": "match", "key": "rule", "value": "CL-runtime-shell-crashrecov-001"},
            {"type": "match", "key": "rule", "value": "CL-runtime-shell-nostart-001"},
        ],
        "conclusion": {"type": "derived", "key": "convergence",
                       "value": "caelestia-shell-unit-recovery-path-agreed"},
        "rationale": (
            "Both matched rules route recovery through the same "
            "caelestia-shell.service user unit. CL-runtime-shell-crashrecov-001 "
            "restarts the unit after a crash (force-stopping the shell processes "
            "first only if the restart hangs); CL-runtime-shell-nostart-001 "
            "starts the unit when the shell never appeared, and only if it still "
            "fails suggests running the shell in debug mode in a terminal. No "
            "fix step in either rule opposes the other."),
        "citation": (
            "docs/TROUBLESHOOTING.md §3.1 (issue #528) and issue #418 — the "
            "references carried by CL-runtime-shell-nostart-001 and "
            "CL-runtime-shell-crashrecov-001"),
    },
]


# ---------------------------------------------------------------------------
# Convenience + CLI enrichment (READ-ONLY consumers of engine.diagnose output)
# ---------------------------------------------------------------------------

def derive(facts: Sequence[Dict[str, str]],
           rules: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    """Assert `facts`, run to fixpoint, return the derived facts in
    derivation order (rules default to the shipped COMPOSITE_RULES)."""
    network = ReteNetwork(COMPOSITE_RULES if rules is None else rules)
    for fact in facts:
        network.assert_fact(fact)
    return network.run()


def facts_from_diagnosis(diagnosis: Dict[str, Any]) -> List[Dict[str, str]]:
    """READ-ONLY: one match fact per gate-passing candidate of a
    diagnose() result — the only fact vocabulary the shipped composite
    rules consume. The diagnosis is not modified."""
    facts: List[Dict[str, str]] = []
    for candidate in diagnosis.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        rule = candidate.get("rule")
        rule_id = rule.get("id") if isinstance(rule, dict) else None
        if rule_id is None:
            rule_id = candidate.get("id")
        if isinstance(rule_id, str) and rule_id:
            facts.append({"type": "match", "key": "rule", "value": rule_id})
    return facts


def derived_evidence(diagnosis: Dict[str, Any],
                     rules: Optional[Sequence[Dict[str, Any]]] = None
                     ) -> Optional[Dict[str, Any]]:
    """READ-ONLY enrichment payload for one diagnose() result; None when no
    composite rule fired (zero premises matched -> nothing derived -> no
    section rendered, per the honesty contract)."""
    network = ReteNetwork(COMPOSITE_RULES if rules is None else rules)
    for fact in facts_from_diagnosis(diagnosis):
        network.assert_fact(fact)
    derived = network.run()
    if not derived:
        return None
    return {
        "derived": [network.explain(fact) for fact in derived],
        "dormant": network.dormant_rules(),
    }


def render_derived_lines(payload: Dict[str, Any]) -> List[str]:
    """Render the derived-evidence section (deterministic plain text)."""
    lines = ["Derived evidence (Rete composite rules — Forgy 1982; derivations "
             "from the matched rules above, not new matches):"]
    index: Dict[Tuple[str, str, str], int] = {}
    for position, node in enumerate(payload["derived"], start=1):
        fact = node["fact"]
        index[(fact["type"], fact["key"], fact["value"])] = position
    for position, node in enumerate(payload["derived"], start=1):
        fact = node["fact"]
        lines.append(f"  [{position}] {fact['type']} / {fact['key']} = {fact['value']}")
        lines.append(f"      via {node['rule']} ({node['citation']})")
        lines.append(textwrap.fill(
            node["rationale"] or "", width=86,
            initial_indent="      because ", subsequent_indent="      "))
        rendered_premises = []
        for premise in node["premises"]:
            pf = premise["fact"]
            key = (pf["type"], pf["key"], pf["value"])
            suffix = f" (derived, see [{index[key]}])" if key in index else ""
            rendered_premises.append(
                f"{pf['type']} / {pf['key']} = {pf['value']}{suffix}")
        lines.append("      premises: " + "; ".join(rendered_premises))
    dormant = payload.get("dormant") or []
    if dormant:
        lines.append("  Dormant composite rules (premises not satisfied by this "
                     "report; listed, not hidden):")
        for entry in dormant:
            premise = entry.get("unsatisfied_premise")
            if premise is None:
                lines.append(f"    - {entry['rule']} ({entry.get('note', 'no matching facts')})")
            else:
                lines.append(
                    f"    - {entry['rule']} (missing premise: "
                    f"{premise['type']} / {premise['key']} = {premise['value']})")
    return lines

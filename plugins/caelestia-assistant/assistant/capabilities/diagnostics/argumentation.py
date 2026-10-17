"""diagnostics.argumentation — Dung abstract argumentation over
conflicting diagnostic rules.

Dung 1995, "On the Acceptability of Arguments and its Fundamental Role in
Nonmonotonic Reasoning, Logic Programming and n-Person Games",
Artificial Intelligence 77(2), pp. 321-357 — the semantics implemented
EXACTLY as defined there:

- admissible(S): S is conflict-free (no member attacks a member, itself
  included) and every attacker of a member is attacked by some member of S;
- grounded extension: the least fixed point of the characteristic function
  F(S) = {a | every attacker of a is attacked by some member of S},
  obtained by iterating F from the empty set until stable;
- preferred extensions: the subset-maximal admissible sets, found here by
  include-first DFS enumeration with conflict pruning and maximality
  filtering (a later-visited set can never be a strict superset of an
  earlier-visited one under include-first order, so one pass suffices);
  enumeration is REFUSED above 20 arguments with the stated reason —
  honesty over completeness: the search is exponential and the module says
  so instead of hanging; the grounded verdict stays available.

WHEN FIXES FIGHT, the framework answers "which fix wins and why" without
inventing a tie-break: symmetric two-rule conflicts (A says enable X, B
says disable X) have NO Dung winner — nothing defends either side, the
grounded extension is empty, and resolve() says "undecided between 2
admissible positions" and ABSTAINS. A winner needs a defender (an
unattacked rule that defeats the loser), and the verdict then carries the
winner's defense set — exactly which attackers it defeats and how.

Attack edges, two deterministic sources:

1. CONTRADICTIONS — an explicit table of rule-id pairs with reason +
   citation. The shipped table is INTENTIONALLY EMPTY, said plainly: a
   full read of the six rules.d files found NO pair of fixes that
   genuinely conflict — no same-unit opposing verbs, no opposing
   instructions about the same service, file, or binary (the corpus is
   deliberately consistent; pinned by test). Manufacturing an attack
   between non-conflicting rules would be a fake claim, so the table
   stays empty until a real conflicting pair ships in rules.d, at which
   point it gets an entry with reason + citation. The mechanism itself is
   pinned by tests with synthetic pairs.
2. Structural detection — a small deterministic parser over the fix
   command strings: commands that target the SAME unit (systemd unit,
   package, or file path) with OPPOSING verbs (enable/disable, start/
   stop, mask/unmask, install/remove; restart counts as start-side).
   Each attack carries the matched unit and verbs as its reason. Queries
   (systemctl status/is-enabled, pacman -Qs/-Ss) are not instructions and
   produce no ops; a rule's own reinstall (pacman -R x && pacman -S x)
   is cross-rule by construction and never attacks itself.

Pure module: no I/O, no execution, nothing written; deterministic by
construction (fixed iteration orders throughout).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["CONTRADICTIONS", "ArgumentationFramework",
           "PREFERRED_ENUMERATION_LIMIT", "arguments_from_report",
           "conflict_resolution", "parse_command_ops", "render_conflict_lines",
           "resolve", "structural_attacks", "table_attacks"]

# Honesty over completeness: preferred-extension enumeration is exponential
# (2^n candidate subsets); above this many arguments the module refuses and
# says why, leaving the (polynomial) grounded verdict standing.
PREFERRED_ENUMERATION_LIMIT = 20

# The explicit contradiction table (see module docstring: intentionally
# empty — no real rules.d pair genuinely conflicts). Entry shapes, for the
# day a real pair ships: {"rules": [id_a, id_b], "reason", "citation"} for
# a MUTUAL contradiction (attack registered both ways), or
# {"attacker": id_a, "defender": id_b, "reason", "citation"} for a
# DIRECTED defeat (one-way; this is the shape that can produce a grounded
# winner with a defense set, since structural opposition is always
# symmetric and by itself can only yield undecided verdicts).
CONTRADICTIONS: Tuple[Dict[str, Any], ...] = ()

_SYSTEMCTL_VERBS = frozenset({"enable", "disable", "start", "stop",
                              "restart", "mask", "unmask"})
_UP_VERBS = frozenset({"enable", "start", "restart", "unmask", "install"})
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=\S*")
_PACMAN_INSTALL_RE = re.compile(r"-S[yu]*")
_PACMAN_REMOVE_RE = re.compile(r"-R[snd]*")


# ---------------------------------------------------------------------------
# Dung's abstract framework + semantics
# ---------------------------------------------------------------------------

class ArgumentationFramework:
    """AF = (arguments, attacks); attacks are (attacker, defender) pairs.

    Semantics exactly per Dung 1995 (module docstring). All methods are
    deterministic; unknown argument ids are REJECTED with ValueError.
    """

    def __init__(self, arguments: Sequence[str],
                 attacks: Sequence[Tuple[str, str]]) -> None:
        self.arguments: List[str] = []
        seen: set = set()
        for argument in arguments:
            if not isinstance(argument, str) or not argument:
                raise ValueError(f"argument ids must be non-empty strings, got {argument!r}")
            if argument in seen:
                raise ValueError(f"duplicate argument id {argument!r}")
            seen.add(argument)
            self.arguments.append(argument)
        self._attackers: Dict[str, List[str]] = {a: [] for a in self.arguments}
        self._attackees: Dict[str, List[str]] = {a: [] for a in self.arguments}
        self._pairs: List[Tuple[str, str]] = []
        seen_pairs: set = set()
        for pair in attacks:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"attacks must be (attacker, defender) pairs, got {pair!r}")
            attacker, defender = pair
            if attacker not in self._attackers or defender not in self._attackees:
                raise ValueError(
                    f"attack {pair!r} references an unknown argument "
                    f"(known: {self.arguments})")
            if (attacker, defender) in seen_pairs:
                raise ValueError(
                    f"duplicate attack edge {(attacker, defender)!r} — the "
                    "attack relation is a set; repeated edges are rejected, "
                    "not silently absorbed (the fusion module's duplicate-"
                    "vote convention)")
            seen_pairs.add((attacker, defender))
            self._pairs.append((attacker, defender))
            self._attackers[defender].append(attacker)
            self._attackees[attacker].append(defender)

    def _known(self, name: str) -> str:
        if name not in self._attackers:
            raise ValueError(f"unknown argument {name!r} (known: {self.arguments})")
        return name


    def attackees_of(self, argument: str) -> List[str]:
        """The arguments that `argument` attacks, in declaration order."""
        return list(self._attackees[self._known(argument)])

    def is_conflict_free(self, subset: Sequence[str]) -> bool:
        """No member of `subset` attacks a member of `subset` (a self-
        attacking member violates conflict-freedom on its own)."""
        members = [self._known(a) for a in subset]
        member_set = set(members)
        for a in members:
            if any(b in member_set for b in self._attackees[a]):
                return False
        return True

    def admissible(self, subset: Sequence[str]) -> bool:
        """Dung's admissibility: conflict-free + every member defended by
        the subset (each attacker is attacked by some subset member)."""
        if not self.is_conflict_free(subset):
            return False
        members = [self._known(a) for a in subset]
        member_set = set(members)
        for a in members:
            for attacker in self._attackers[a]:
                if not any(attacker in self._attackees[m] for m in member_set):
                    return False
        return True

    def characteristic(self, subset: Sequence[str]) -> List[str]:
        """F(S) = the arguments defended by S (Dung's characteristic
        function), in framework order."""
        members = [self._known(a) for a in subset]
        member_set = set(members)
        defended: List[str] = []
        for a in self.arguments:
            attackers = self._attackers[a]
            if all(any(attacker in self._attackees[m] for m in member_set)
                   for attacker in attackers):
                defended.append(a)
        return defended

    def grounded_extension(self) -> List[str]:
        """The least fixed point of F, by iteration from the empty set
        (Dung 1995); returned in framework order."""
        current: List[str] = []
        for _ in range(len(self.arguments) + 2):
            nxt = self.characteristic(current)
            if nxt == current:
                return current
            current = nxt
        raise RuntimeError(
            "characteristic-function iteration failed to stabilize within "
            "n+2 rounds — internal invariant violation (F is monotone); "
            "please report")

    def preferred_extensions(self) -> List[List[str]]:
        """The maximal admissible sets, in include-first DFS discovery
        order (deterministic; each extension in framework order).

        REFUSED with ValueError above PREFERRED_ENUMERATION_LIMIT
        arguments — the enumeration is exponential; honesty over
        completeness (the grounded verdict remains available).
        """
        n = len(self.arguments)
        if n > PREFERRED_ENUMERATION_LIMIT:
            raise ValueError(
                f"refused: preferred-extension enumeration over {n} arguments "
                f"(> {PREFERRED_ENUMERATION_LIMIT}) is exponential (2^{n} "
                "candidate subsets); honesty over completeness — the grounded "
                "verdict remains available")
        index = {a: i for i, a in enumerate(self.arguments)}
        attackees_mask = [0] * n
        attackers_mask = [0] * n
        conflict_mask = [0] * n  # both directions + the self bit
        for attacker, defender in self._pairs:
            i, j = index[attacker], index[defender]
            attackees_mask[i] |= 1 << j
            attackers_mask[j] |= 1 << i
            conflict_mask[i] |= 1 << j
            conflict_mask[j] |= 1 << i

        def admissible_mask(mask: int) -> bool:
            # conflict-free holds by DFS construction; check defense only:
            # every attacker of a member must be covered by the members'
            # combined attackees.
            cover = 0
            rest = mask
            while rest:
                low = rest & -rest
                cover |= attackees_mask[low.bit_length() - 1]
                rest ^= low
            rest = mask
            while rest:
                low = rest & -rest
                if attackers_mask[low.bit_length() - 1] & ~cover:
                    return False
                rest ^= low
            return True

        kept: List[int] = []

        def record_if_maximal(mask: int) -> None:
            if any(mask | k == k for k in kept):
                return  # subset of (or equal to) an already-kept extension
            if admissible_mask(mask):
                kept.append(mask)

        def search(i: int, included: int, blocked: int) -> None:
            # Upper-bound prune: the best completion is included plus every
            # still-addable argument; if even that cannot beat a kept
            # extension, no completion of this branch is maximal.
            available = 0
            for j in range(i, n):
                if not (blocked >> j) & 1:
                    available |= 1 << j
            if any((included | available) | k == k for k in kept):
                return
            if i == n:
                record_if_maximal(included)
                return
            # Include-first (this ordering is what makes one-pass maximality
            # filtering sound: any strict superset is always visited first).
            if not (blocked >> i) & 1 and not (conflict_mask[i] >> i) & 1:
                search(i + 1, included | (1 << i), blocked | conflict_mask[i])
            search(i + 1, included, blocked)

        search(0, 0, 0)
        return [[self.arguments[j] for j in range(n) if (mask >> j) & 1]
                for mask in kept]


# ---------------------------------------------------------------------------
# The structural conflict parser (deterministic; commands are inert strings)
# ---------------------------------------------------------------------------

def _segments(command: str) -> List[str]:
    """Split a command line on command separators (&& and ;)."""
    return [part.strip() for part in re.split(r"&&|;", command) if part.strip()]


def _tokens(segment: str) -> List[str]:
    """Whitespace tokens without sudo prefixes or env assignments."""
    tokens = []
    for token in segment.split():
        if token == "sudo":
            continue
        if _ENV_ASSIGNMENT_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def _normalize_unit(unit: str) -> str:
    """Units are compared after stripping the .service suffix, so
    'bluetooth.service' and 'bluetooth' name the same unit."""
    if unit.endswith(".service"):
        return unit[: -len(".service")]
    return unit


def parse_command_ops(command: str) -> List[Dict[str, str]]:
    """The (verb, unit, direction) instructions carried by one command.

    Recognized shapes only, deliberately small: systemctl verb units;
    pacman -S/-R package lists (flags like -Ss search or -Q query are NOT
    instructions); dnf/apt install/remove lists; rm paths. Everything else
    (bash scripts, config writers, editors, read-only probes) yields no
    ops — an unrecognized command is no instruction, never a guess.
    """
    ops: List[Dict[str, str]] = []
    for segment in _segments(command):
        tokens = _tokens(segment)
        if not tokens:
            continue
        head = tokens[0]
        if head == "systemctl":
            verb: Optional[str] = None
            units: List[str] = []
            for token in tokens[1:]:
                if verb is None:
                    if token in _SYSTEMCTL_VERBS:
                        verb = token
                elif not token.startswith("-"):
                    units.append(token)
            if verb is not None:
                for unit in units:
                    ops.append({"verb": verb, "unit": _normalize_unit(unit),
                                "direction": "up" if verb in _UP_VERBS else "down"})
        elif head == "pacman":
            mode: Optional[str] = None
            packages: List[str] = []
            for token in tokens[1:]:
                if token.startswith("--"):
                    continue
                if token.startswith("-"):
                    if _PACMAN_INSTALL_RE.fullmatch(token):
                        mode = "install"
                    elif _PACMAN_REMOVE_RE.fullmatch(token):
                        mode = "remove"
                    else:
                        mode = None  # -Q/-Ss queries and friends: not instructions
                elif mode is not None:
                    packages.append(token)
            for package in packages:
                ops.append({"verb": mode, "unit": package,
                            "direction": "up" if mode == "install" else "down"})
        elif head in ("dnf", "apt", "apt-get"):
            if len(tokens) >= 2:
                sub = tokens[1]
                if sub == "install":
                    for package in tokens[2:]:
                        ops.append({"verb": "install", "unit": package,
                                    "direction": "up"})
                elif sub in ("remove", "erase", "purge"):
                    for package in tokens[2:]:
                        ops.append({"verb": "remove", "unit": package,
                                    "direction": "down"})
        elif head == "rm":
            for token in tokens[1:]:
                if not token.startswith("-"):
                    ops.append({"verb": "remove", "unit": token,
                                "direction": "down"})
    return ops


def _ops_by_unit(arguments: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """argument id -> unit -> ops (in command order), for pair scanning."""
    out: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for argument in arguments:
        per_unit: Dict[str, List[Dict[str, str]]] = {}
        for command in argument.get("fix_commands", []) or []:
            for op in parse_command_ops(command):
                per_unit.setdefault(op["unit"], []).append(op)
        out[argument["id"]] = per_unit
    return out


def structural_attacks(arguments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attacks between matched rules whose fix commands target the same
    unit with opposing verbs. Cross-rule only (a rule's own reinstall
    never attacks itself); each attack carries unit + verbs as its reason.
    """
    attacks: List[Dict[str, Any]] = []
    ops = _ops_by_unit(arguments)
    for i, a in enumerate(arguments):
        for b in arguments[i + 1:]:
            a_id, b_id = a["id"], b["id"]
            for unit in sorted(set(ops[a_id]) & set(ops[b_id])):
                # First opposing verb pair in command order (deterministic).
                opposing = [
                    (op_a, op_b)
                    for op_a in ops[a_id][unit]
                    for op_b in ops[b_id][unit]
                    if op_a["direction"] != op_b["direction"]
                ]
                if not opposing:
                    continue
                op_a, op_b = opposing[0]
                # Opposing instructions contradict each other mutually:
                # each rule's fix rebuts the other's, so the attack is
                # registered in BOTH directions with the same evidence.
                attacks.append({
                    "attacker": a_id, "defender": b_id, "source": "structural",
                    "unit": unit, "verbs": f"{op_a['verb']} vs {op_b['verb']}",
                    "reason": (f"fix commands oppose each other on unit {unit!r}: "
                               f"{a_id} runs {op_a['verb']}, {b_id} runs {op_b['verb']}"),
                })
                attacks.append({
                    "attacker": b_id, "defender": a_id, "source": "structural",
                    "unit": unit, "verbs": f"{op_b['verb']} vs {op_a['verb']}",
                    "reason": (f"fix commands oppose each other on unit {unit!r}: "
                               f"{b_id} runs {op_b['verb']}, {a_id} runs {op_a['verb']}"),
                })
    return attacks


def table_attacks(arguments: Sequence[Dict[str, Any]],
                  contradictions: Optional[Sequence[Dict[str, Any]]] = None
                  ) -> List[Dict[str, Any]]:
    # Late binding of the module-level table: the default is looked up at
    # CALL time, so a patched/test/future table constant is actually
    # consulted (a def-time default would freeze the empty shipped table
    # into the signature forever).
    if contradictions is None:
        contradictions = CONTRADICTIONS
    """Attacks from the explicit CONTRADICTIONS table. Entries referencing
    rules not in this report simply do not apply (a contradiction between
    two rules is irrelevant when only one of them matched).

    Two entry shapes, both requiring reason + citation:
    - {"rules": [a, b], ...} — a MUTUAL contradiction: each rule's fix
      rebuts the other's, so the attack is registered in both directions;
    - {"attacker": a, "defender": b, ...} — a DIRECTED defeat: a's fix
      defeats b's one-way (e.g. a documented supersession). Dung's attack
      relation is directed, and this shape is what lets a grounded winner
      carry a defense set (structural opposition is always symmetric, so
      by itself it can only produce undecided outcomes — see the module
      docstring's honesty note on symmetric fights).
    """
    known = {argument["id"] for argument in arguments}
    attacks: List[Dict[str, Any]] = []
    for entry in contradictions:
        keys = set(entry)
        if keys not in ({"rules", "reason", "citation"},
                        {"attacker", "defender", "reason", "citation"}):
            raise ValueError(
                "CONTRADICTIONS entry must have exactly the keys "
                "['reason', 'citation'] plus either ['rules'] (mutual pair) "
                f"or ['attacker', 'defender'] (directed), got {sorted(keys)}")
        if not isinstance(entry["reason"], str) or not entry["reason"]:
            raise ValueError("CONTRADICTIONS entry needs a non-empty reason")
        if not isinstance(entry["citation"], str) or not entry["citation"]:
            raise ValueError("CONTRADICTIONS entry needs a non-empty citation")
        if "rules" in entry:
            pair = entry["rules"]
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(x, str) and x for x in pair)
                    or pair[0] == pair[1]):
                raise ValueError(
                    f"CONTRADICTIONS entry 'rules' must be two distinct rule ids, got {pair!r}")
            a, b = pair
            edges = ((a, b), (b, a))
        else:
            a, b = entry["attacker"], entry["defender"]
            if (not isinstance(a, str) or not a or not isinstance(b, str)
                    or not b or a == b):
                raise ValueError(
                    "CONTRADICTIONS entry 'attacker'/'defender' must be two "
                    f"distinct rule ids, got {a!r} -> {b!r}")
            edges = ((a, b),)
        for attacker, defender in edges:
            if attacker in known and defender in known:
                attacks.append({"attacker": attacker, "defender": defender,
                                "source": "table", "unit": None, "verbs": None,
                                "reason": entry["reason"],
                                "citation": entry["citation"]})
    return attacks


# ---------------------------------------------------------------------------
# From a diagnose() report to a verdict
# ---------------------------------------------------------------------------

def arguments_from_report(report: Dict[str, Any],
                          rules_by_id: Optional[Dict[str, Dict[str, Any]]] = None
                          ) -> List[Dict[str, Any]]:
    """READ-ONLY: each matched rule id of a diagnose() result becomes one
    argument, carrying its title and fix summary for rendering plus the
    fix command strings for structural conflict detection. Candidates
    without a rule body (and without a rules_by_id fallback) carry no
    commands — nothing is guessed about their fixes."""
    rules_by_id = rules_by_id or {}
    arguments: List[Dict[str, Any]] = []
    for candidate in report.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        rule = candidate.get("rule")
        rule_id = rule.get("id") if isinstance(rule, dict) else None
        if rule_id is None:
            rule_id = candidate.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            continue
        source = rule if isinstance(rule, dict) else rules_by_id.get(rule_id)
        if source is None:
            arguments.append({"id": rule_id, "title": None,
                              "fix_summary": "", "fix_commands": []})
            continue
        title = source.get("title")
        steps = source.get("fix") or []
        texts = [str(step.get("text", "")) for step in steps if isinstance(step, dict)]
        commands = [str(step["command"]) for step in steps
                    if isinstance(step, dict) and isinstance(step.get("command"), str)]
        arguments.append({
            "id": rule_id,
            "title": title if isinstance(title, str) else None,
            "fix_summary": " | ".join(text for text in texts if text),
            "fix_commands": commands,
        })
    return arguments


def resolve(report: Dict[str, Any],
            rules_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
            contradictions: Optional[Sequence[Dict[str, Any]]] = None
            ) -> Dict[str, Any]:
    """The Dung verdict over one diagnose() report's matched rules.

    Returns a verdict dict: status ("no-conflict" | "resolved" |
    "undecided"), the grounded extension, per-argument status (in / out /
    undecided relative to grounded), attack edges with reasons, winners
    WITH their defense set, losers WITH their undefeated attackers — and
    when the grounded extension leaves a conflict undecided (odd attack
    cycles, symmetric fights), an explicit "undecided between N admissible
    positions" note with abstain=True: no winner is ever silently chosen.
    """
    if contradictions is None:
        contradictions = CONTRADICTIONS
    arguments = arguments_from_report(report, rules_by_id)
    attacks = table_attacks(arguments, contradictions) + structural_attacks(arguments)
    framework = ArgumentationFramework(
        [a["id"] for a in arguments],
        [(edge["attacker"], edge["defender"]) for edge in attacks])
    grounded = framework.grounded_extension()
    grounded_set = set(grounded)

    out_set: set = set()
    for member in grounded:
        out_set.update(framework.attackees_of(member))
    out_set -= grounded_set  # grounded is conflict-free; safety only

    by_id = {a["id"]: a for a in arguments}
    statuses: Dict[str, str] = {}
    for argument in framework.arguments:
        if argument in grounded_set:
            statuses[argument] = "in"
        elif argument in out_set:
            statuses[argument] = "out"
        else:
            statuses[argument] = "undecided"

    verdict: Dict[str, Any] = {
        "status": "no-conflict" if not attacks else "resolved",
        "grounded": grounded,
        "arguments": [{"id": a["id"], "title": a["title"],
                       "fix_summary": a["fix_summary"],
                       "status": statuses[a["id"]]} for a in arguments],
        "attack_edges": attacks,
        "winners": [],
        "losers": [],
        "undecided": None,
        "abstained": False,
    }
    if not attacks:
        return verdict

    # Winners: accepted rules that actually defeated an attacker, with the
    # defense set showing which attacker and how.
    for member in grounded:
        defense = [{"defeated": edge["defender"],
                    "how": edge["reason"]}
                   for edge in attacks
                   if edge["attacker"] == member]
        if defense:
            verdict["winners"].append({
                "id": member, "title": by_id[member]["title"],
                "fix_summary": by_id[member]["fix_summary"],
                "defense": defense})

    # Losers: rejected rules, with the accepted (hence undefeated) rules
    # that defeated them.
    for argument in framework.arguments:
        if statuses[argument] != "out":
            continue
        undefeated = [{"id": edge["attacker"], "how": edge["reason"]}
                      for edge in attacks
                      if edge["defender"] == argument
                      and edge["attacker"] in grounded_set]
        verdict["losers"].append({
            "id": argument, "title": by_id[argument]["title"],
            "fix_summary": by_id[argument]["fix_summary"],
            "undefeated_attackers": undefeated})

    # A conflict the grounded extension leaves undecided: an edge whose
    # BOTH endpoints are undecided (grounded accepted neither side and
    # defeated neither side). Then the module ABSTAINS — no winner.
    undecided_edges = [edge for edge in attacks
                       if statuses[edge["attacker"]] == "undecided"
                       and statuses[edge["defender"]] == "undecided"]
    if undecided_edges:
        verdict["status"] = "undecided"
        verdict["abstained"] = True
        undecided_arguments = [a for a in framework.arguments
                               if statuses[a] == "undecided"]
        positions: Optional[List[List[str]]] = None
        note: str
        try:
            positions = framework.preferred_extensions()
            listed = " | ".join("(the empty set)" if not p
                                else "(" + ", ".join(p) + ")"
                                for p in positions)
            note = (f"undecided between {len(positions)} admissible "
                    f"position{'s' if len(positions) != 1 else ''}: {listed} — "
                    "abstaining from a winner")
        except ValueError as exc:
            note = (f"undecided, and preferred-extension enumeration was refused "
                    f"({exc}); only the grounded verdict is reported — "
                    "abstaining from a winner")
        verdict["undecided"] = {
            "arguments": undecided_arguments,
            "positions": positions,
            "note": note,
        }
    return verdict


def conflict_resolution(report: Dict[str, Any],
                        rules_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
                        contradictions: Optional[Sequence[Dict[str, Any]]] = None
                        ) -> Optional[Dict[str, Any]]:
    """READ-ONLY CLI helper: resolve(report) when the matched rules
    actually conflict, else None (no conflicts -> no section rendered)."""
    verdict = resolve(report, rules_by_id=rules_by_id,
                      contradictions=contradictions)
    if verdict["status"] == "no-conflict":
        return None
    return verdict


def render_conflict_lines(verdict: Dict[str, Any]) -> List[str]:
    """Render the conflict-resolution section (deterministic plain text)."""
    by_id = {a["id"]: a for a in verdict["arguments"]}
    lines = ["Conflict resolution (Dung 1995, Artificial Intelligence 77 — "
             "grounded semantics; why one fix won, nothing auto-applied):"]
    lines.append("  Conflicting matched rules:")
    for edge in verdict["attack_edges"]:
        lines.append(f"    - {edge['attacker']} -> {edge['defender']}"
                     f" [{edge['source']}]: {edge['reason']}")
    for winner in verdict["winners"]:
        title = f": {winner['title']}" if winner.get("title") else ""
        lines.append(f"  Accepted (winner) {winner['id']}{title}")
        for defense in winner["defense"]:
            lines.append(f"    defeats {defense['defeated']} ({defense['how']})")
    for loser in verdict["losers"]:
        title = f": {loser['title']}" if loser.get("title") else ""
        lines.append(f"  Rejected (loser) {loser['id']}{title}")
        for attacker in loser["undefeated_attackers"]:
            lines.append(f"    defeated by {attacker['id']} ({attacker['how']})")
    if verdict["status"] == "undecided" and verdict["undecided"]:
        lines.append("  UNDECIDED — the grounded extension accepts no side of "
                     "the remaining conflict:")
        lines.append(f"    {verdict['undecided']['note']}")
        lines.append("    No winner is chosen; both fixes are printed above — "
                     "the human decides.")
    return lines

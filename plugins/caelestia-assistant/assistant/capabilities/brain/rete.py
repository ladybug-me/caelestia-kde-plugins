"""brain.rete — user-authored event-condition-action rules on a Rete
network (C14).

Some users KNOW their rule: "if the same app crashes three times in
ten minutes, propose muting its notifications". A statistical model
will never say that cleanly; an explicit rule will. This module runs
such rules — authored by the user in a plain JSON file — through a
small Rete network (Forgy 1982): an alpha network filters single
events per condition, a beta network joins the filtered sets across a
rule's conditions (shared-variable equality joins, window bounds),
and complete tokens fire the rule's ACTION.

The action is always a PROPOSAL: a dict labeled SUGGESTED_NOT_EXECUTED
carrying whatever the rule author asked for (typically a settings
change), which the caller routes through the ledger's approve/reject
flow like every other proposal. The engine never applies anything,
never writes, and does nothing at all until the user creates a rules
file — the opt-in IS the file.

Determinism: events are processed in the given order, condition sets
are sorted on evaluation, and the fired tokens are sorted before
returning, so the same events + rules always produce the same
proposals in the same order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Rete", "load_rules", "run_events", "render_report"]

DEFAULT_RULES_PATH = Path.home() / ".config" / "caelestia-assistant" \
    / "rules.json"

_MAX_EVENTS = 5000
_MAX_RULES = 100

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "contains": lambda a, b: str(b).lower() in str(a).lower(),
}


class _Condition:
    """One alpha test: event[field] OP value, optionally binding the
    matched value to a join variable."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        self.field = str(spec["field"])
        self.op = str(spec.get("op", "eq"))
        if self.op not in _OPS:
            raise ValueError(f"unknown op {self.op!r}")
        self.value = spec.get("value")
        self.var = spec.get("var")  # join variable name (optional)

    def test(self, event: Dict[str, Any]) -> bool:
        if self.field not in event:
            return False
        try:
            return _OPS[self.op](event[self.field], self.value)
        except TypeError:
            return False


class _Rule:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.id = str(spec.get("id") or "rule")
        self.conditions = [_Condition(c) for c in spec.get("when", [])]
        if not self.conditions:
            raise ValueError(f"rule {self.id!r} has no 'when' conditions")
        self.window_seconds = float(spec.get("within_seconds", 600))
        self.min_events = int(spec.get("min_events", len(self.conditions)))
        self.action = spec.get("then") or {}
        self.reason = str(spec.get("reason", ""))


class Rete:
    """The network: rules compiled once, then fed events."""

    def __init__(self, rules: List[Dict[str, Any]]) -> None:
        if len(rules) > _MAX_RULES:
            raise ValueError(f"bounded at {_MAX_RULES} rules")
        for i, r in enumerate(rules):
            if not isinstance(r, dict):
                raise ValueError(
                    f"every rule must be a JSON object; rule at index "
                    f"{i} is {type(r).__name__}")
        self.rules = [_Rule(r) for r in rules]

    def run(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Feed events (each a dict with at least a numeric 'time'),
        return fired proposals per rule. Non-dict records cannot carry
        a time or a field, so they are skipped, not crashed on."""
        events = [e for e in events if isinstance(e, dict)]
        if len(events) > _MAX_EVENTS:
            return {"verdict": "ABSTAIN",
                    "note": f"more than {_MAX_EVENTS} events: the "
                            f"network abstains"}
        fired: List[Dict[str, Any]] = []
        for rule in self.rules:
            tokens = self._match(rule, events)
            for token in tokens:
                fired.append({
                    "rule": rule.id,
                    "verdict": "SUGGESTED_NOT_EXECUTED",
                    "action": rule.action,
                    "reason": rule.reason,
                    "evidence": [{k: e.get(k) for k in sorted(e)}
                                 for e in token],
                })
        fired.sort(key=lambda f: (f["rule"], json.dumps(
            f["evidence"], sort_keys=True)))
        return {"verdict": "OK", "n_rules": len(self.rules),
                "n_events": len(events), "n_fired": len(fired),
                "fired": fired,
                "note": "every fired action is a proposal for the "
                        "ledger — the network applies nothing"}

    def _match(self, rule: _Rule, events: List[Dict[str, Any]]) \
            -> List[List[Dict[str, Any]]]:
        """Beta network: incrementally join per-condition event sets,
        enforcing variable consistency and the time window. A rule with
        ONE condition and min_events > 1 uses a sliding-window count
        instead (the 'N crashes in M minutes' shape needs no joins)."""
        # alpha: filter events per condition (deterministic order)
        per_condition: List[List[Dict[str, Any]]] = []
        for cond in rule.conditions:
            matched = [e for e in events if cond.test(e)]
            matched.sort(key=lambda e: (float(e.get("time", 0.0)),
                                        json.dumps(e, sort_keys=True)))
            per_condition.append(matched)

        if len(rule.conditions) == 1 and rule.min_events > 1:
            return self._sliding_window(rule, per_condition[0])

        # beta: build tokens of length k, joining condition k's events
        # against variable bindings and the window
        tokens: List[Tuple[Dict[str, Any], ...]] = \
            tuple((e,) for e in per_condition[0])
        for k in range(1, len(rule.conditions)):
            cond = rule.conditions[k]
            new_tokens: List[Tuple[Dict[str, Any], ...]] = []
            for token in tokens:
                for e in per_condition[k]:
                    if e in token:
                        continue
                    if not self._vars_consistent(rule.conditions[:k],
                                                 token, cond, e):
                        continue
                    if not self._within_window(rule, token + (e,)):
                        continue
                    new_tokens.append(token + (e,))
            tokens = new_tokens
            if not tokens:
                return []
        complete = [list(t) for t in tokens if len(t) >= rule.min_events]
        complete.sort(key=lambda t: json.dumps(t, sort_keys=True))
        return complete[:10]  # bounded evidence per rule

    def _sliding_window(self, rule: _Rule, matched: List[Dict[str, Any]]) \
            -> List[List[Dict[str, Any]]]:
        """The densest window of matched events, as one token (capped)
        — fires once when the count clears min_events."""
        if not matched:
            return []
        times = [float(e.get("time", 0.0)) for e in matched]
        best: List[int] = []
        for i in range(len(matched)):
            window = [j for j in range(i, len(matched))
                      if times[j] - times[i] <= rule.window_seconds]
            if len(window) > len(best):
                best = window
            if len(best) >= len(matched) - i:
                break  # no later window can beat this
        if len(best) < rule.min_events:
            return []
        token = [matched[j] for j in best[:10]]
        return [token]

    @staticmethod
    def _vars_consistent(conds: List["_Condition"],
                         token: Tuple[Dict[str, Any], ...],
                         cond: "_Condition", event: Dict[str, Any]) -> bool:
        """Join check: a variable bound by an earlier condition must
        match the new event's value (this is the beta network's join)."""
        if cond.var is None:
            return True
        for c, e in zip(conds, token):
            if c.var == cond.var and c.var is not None:
                if e.get(c.field) != event.get(cond.field):
                    return False
        return True

    @staticmethod
    def _within_window(rule: _Rule, token: Tuple[Dict[str, Any], ...]) \
            -> bool:
        times = [float(e.get("time", 0.0)) for e in token]
        return (max(times) - min(times)) <= rule.window_seconds


def load_rules(path: Path) -> List[Dict[str, Any]]:
    """The rules file -> the rule list. A non-object file is an error
    the caller reports honestly, not a crash."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"rules file unreadable or not JSON: {exc}") \
            from exc
    if not isinstance(data, dict):
        raise ValueError("rules file must be a JSON object with a "
                         "'rules' list")
    rules = data.get("rules")
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")
    return rules


def run_events(rules: List[Dict[str, Any]],
               events: List[Dict[str, Any]]) -> Dict[str, Any]:
    net = Rete(rules)
    return net.run(events)


def render_report(data: Dict[str, Any]) -> List[str]:
    if data.get("verdict") == "ABSTAIN":
        return [f"rules: ABSTAIN ({data['note']})"]
    lines = [f"rete: {data['n_rules']} rules over {data['n_events']} "
             f"events -> {data['n_fired']} proposals "
             f"(SUGGESTED_NOT_EXECUTED)"]
    for f in data["fired"]:
        action = json.dumps(f["action"], sort_keys=True)
        lines.append(f"  [{f['rule']}] {action}")
        if f.get("reason"):
            lines.append(f"    reason: {f['reason']}")
        lines.append(f"    evidence: {len(f['evidence'])} matched events")
    if not data["fired"]:
        lines.append("  (no rule fired)")
    return lines

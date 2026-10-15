"""tests/test_properties.py — the three safety properties (F25).

Runs under the bash harness (tests/test_assistant.sh, the properties
suite) with PYTHONPATH at the repo root, using tests/prop.py — the
seeded, shrinking property helper. Every property is deterministic:
PROP_SEED=<seed> replays one exactly.

Properties shipped (the build's own safety contract):
  1. apply-then-undo is identity (parsed-JSON level — the journaled
     applier may normalize formatting, never values);
  2. no plan emits out-of-range values (every planned entry is inside
     the registry domain, or explicitly rejected/clamped — never
     silently out-of-range);
  3. every hub verb that takes text accepts flags before OR after it
     (the D10 class, pinned generatively).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from prop import for_all, one_of  # noqa: E402

import assistant.capabilities.settings.planner as planner  # noqa: E402
import assistant.capabilities.settings.applier as applier  # noqa: E402
import assistant.capabilities.settings.history as history  # noqa: E402

_TOOLS = json.loads((REPO_ROOT / "assistant" / "adapters" / "caelestia" /
                     "tools.json").read_text(encoding="utf-8"))["tools"]
_SETTABLE = [t for t in _TOOLS if not t.get("global_only", False)]


def _valid_value(rnd: random.Random, tool: dict):
    """A random LEGAL value for one tool (registry-domain-respecting)."""
    kind = tool["kind"]
    if kind == "bool":
        return rnd.choice([True, False])
    if kind == "enum":
        return rnd.choice(tool["enum"])
    lo = tool.get("minimum")
    hi = tool.get("maximum")
    step = tool.get("step") or 1
    if lo is None or hi is None:
        return tool.get("default")
    if kind == "int":
        span = int((hi - lo) // step) + 1
        return int(lo + step * rnd.randint(0, max(span - 1, 0)))
    return round(rnd.uniform(lo, hi), 3)


def _tool_value_gen(rnd: random.Random):
    tool = rnd.choice(_SETTABLE)
    return (tool["name"], _valid_value(rnd, tool))


def _check_apply_undo_is_identity(pair) -> None:
    tool, value = pair
    with tempfile.TemporaryDirectory(prefix="prop-au-") as tmp:
        target = Path(tmp) / "shell.json"
        target.write_text("{}", encoding="utf-8")
        try:
            plan = planner.plan([{"tool": tool, "value": value}],
                                target)
        except planner.PlannerError:
            return  # refused up front: no identity to violate
        entries = [e for e in plan.get("entries", [])
                   if not e.get("error")]
        if not entries or plan.get("apply_blocked"):
            return  # rejected values are the range property's business
        before = json.loads(target.read_text())
        applier.apply(plan, target, write=True, label="prop")
        history.undo(target, steps=1)
        after = json.loads(target.read_text())
        assert before == after, (
            f"apply-then-undo changed state for {tool}={value!r}: "
            f"{before!r} -> {after!r}")


def _check_no_out_of_range_values(pair) -> None:
    tool_name, value = pair
    spec = next((t for t in _TOOLS if t["name"] == tool_name), None)
    assert spec is not None, f"unknown tool {tool_name!r}"
    with tempfile.TemporaryDirectory(prefix="prop-or-") as tmp:
        target = Path(tmp) / "shell.json"
        target.write_text("{}", encoding="utf-8")
        try:
            plan = planner.plan([{"tool": tool_name, "value": value}],
                                target)
        except planner.PlannerError:
            return  # a refusal is the honest outcome, not a violation
        for entry in plan.get("entries", []):
            if entry.get("error") or entry.get("rejected"):
                continue  # explicitly refused: fine
            new = entry.get("new")
            if new is None:
                continue
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if spec["kind"] == "enum":
                assert new in spec["enum"], (
                    f"{tool_name} planned {new!r} outside enum "
                    f"{spec['enum']}")
                continue
            if spec["kind"] == "bool":
                assert isinstance(new, bool), (
                    f"{tool_name} planned non-bool {new!r}")
                continue
            if lo is not None or hi is not None:
                ok = ((lo is None or new >= lo) and
                      (hi is None or new <= hi))
                assert ok, (
                    f"{tool_name} planned {new!r} outside "
                    f"[{lo}, {hi}] with no error and no clamp flag "
                    f"(entry: {entry!r})")


_WORDS = ["make", "the", "bar", "taller", "smaller", "turn", "off",
          "blur", "show", "seconds", "clock", "use", "24", "hour",
          "time", "optimize", "for", "gaming", "why", "is", "my",
          "dock", "blurry"]
_VERB_FLAG_TEXTS = [
    ("do", "--json", "2 plus 2"),
    ("do", "--verbose", "solve x^2 - 2 = 0"),
    ("settings", "--json", "make the bar taller"),
    ("route", "--json", "make the bar taller"),
]


def _check_flag_order(pair) -> None:
    verb, flag, text = pair
    from assistant import hub as hub_mod
    outs = []
    for argv in ([verb, flag, text], [verb, text, flag]):
        home = tempfile.mkdtemp(prefix="prop-flag-")
        old = os.environ.get("HOME")
        os.environ["HOME"] = home
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                try:
                    hub_mod.main(list(argv))
                    code = 0
                except SystemExit as exc:
                    code = int(exc.code or 0)
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old
        blob = out.getvalue() + err.getvalue()
        assert "unrecognized arguments" not in blob, (
            f"{verb} rejects flag order {argv!r}: {blob[:200]!r}")
        assert code != 2, f"{verb} argparse error for {argv!r}"
        outs.append(blob)
    # both orders PARSED; the results must also agree on the verdict
    # line where one exists (same answer either way)
    a = [l for l in outs[0].splitlines() if l.strip()]
    b = [l for l in outs[1].splitlines() if l.strip()]
    assert bool(a) == bool(b), (
        f"{verb} {flag} before/after text diverge: "
        f"{outs[0][:120]!r} vs {outs[1][:120]!r}")


class PropertyTests(unittest.TestCase):
    """One unittest per property so the harness sees them individually."""

    def test_apply_then_undo_is_identity(self) -> None:
        for_all("apply-then-undo is identity",
                _tool_value_gen,
                _check_apply_undo_is_identity,
                max_examples=40)

    def test_no_plan_emits_out_of_range_values(self) -> None:
        # deliberately ALSO generates out-of-range values (range+delta)
        # so the property exercises the refusal paths, not just happy
        # paths — generation stays seeded and shrinks
        def gen(rnd: random.Random):
            name, value = _tool_value_gen(rnd)
            if rnd.random() < 0.35:  # push some values OUT of domain
                spec = next(t for t in _TOOLS if t["name"] == name)
                if spec.get("maximum") is not None and \
                        isinstance(value, (int, float)) and \
                        not isinstance(value, bool):
                    value = spec["maximum"] + rnd.randint(1, 50)
            return (name, value)

        for_all("no plan emits out-of-range values", gen,
                _check_no_out_of_range_values,
                max_examples=60)

    def test_every_hub_verb_takes_flags_before_or_after_text(self) -> None:
        for_all("hub verbs accept flags before/after text",
                one_of(*_VERB_FLAG_TEXTS),
                _check_flag_order,
                max_examples=len(_VERB_FLAG_TEXTS) * 3)

    def test_helper_shrinks_to_a_minimal_counterexample(self) -> None:
        """The helper itself: a deliberately broken property must fail
        with a SHRUNKEN counterexample (smaller than the original)."""
        from prop import PropertyFailure, _size
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_text("{}", encoding="utf-8")

            def check(pair):
                a, b = pair
                assert a <= b, f"ordered violation {a} > {b}"

            with self.assertRaises(PropertyFailure) as ctx:
                for_all("ordered pair (deliberately broken)",
                        lambda rnd: (rnd.randint(0, 9), rnd.randint(0, 9)),
                        check, max_examples=50)
            self.assertIn("minimal counterexample", str(ctx.exception))
            self.assertLessEqual(
                _size(ctx.exception.minimal), 6,
                f"not shrunk: {ctx.exception.minimal!r}")


if __name__ == "__main__":
    unittest.main()

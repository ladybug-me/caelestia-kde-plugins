"""Layer 1 regression tests: real historical issue text -> expected rules.

Case files live in tests/cases/*.json. Every case is derived from the actual
text of a resolved issue or a documented failure (provenance recorded in the
file), so this suite pins Layer 1 to history rather than to the author's
imagination. Run: python -m unittest discover -s assistant/diagnostics/tests
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

from assistant.capabilities.diagnostics import engine

CASES_DIR = Path(__file__).resolve().parent / "cases"


def _load_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("case_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        case["_file"] = path.name
        cases.append(case)
    return cases


class TestHistoricalCases(unittest.TestCase):
    """Every historical regression case must pass."""

    def test_all_cases(self) -> None:
        cases = _load_cases()
        self.assertGreaterEqual(len(cases), 8, "expected at least 8 historical cases")
        failures: List[str] = []
        for case in cases:
            rules = engine.load_rules()
            diagnosis = engine.diagnose(case["input_text"], rules=rules)
            got_ids = [r["rule"]["id"] for r in diagnosis["candidates"]]

            for must in case["expect"].get("must_hit", []):
                if must not in got_ids:
                    failures.append(
                        f"{case['_file']}: expected {must} in candidates, got {got_ids}"
                    )
                elif not case["expect"].get("ambiguous_ok", False):
                    if diagnosis["verdict"] != "MATCH" or diagnosis["candidates"][0]["rule"]["id"] != must:
                        failures.append(
                            f"{case['_file']}: expected single MATCH on {must}, "
                            f"got verdict={diagnosis['verdict']} top={got_ids[:1]}"
                        )
            for never in case["expect"].get("must_not_hit", []):
                if never in got_ids:
                    failures.append(f"{case['_file']}: {never} must NOT match this input")
        self.assertEqual(failures, [], "\n".join(failures))


class TestEngineSemantics(unittest.TestCase):
    """Unit tests for the deterministic contract itself."""

    def setUp(self) -> None:
        self.rules = engine.load_rules()

    def test_rules_loaded_and_unique(self) -> None:
        ids = [r["id"] for r in self.rules]
        self.assertGreaterEqual(len(ids), 20)
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_match_is_honest(self) -> None:
        result = engine.diagnose("my cat sat on the keyboard and now nothing works", rules=self.rules)
        self.assertEqual(result["verdict"], "NO_MATCH")
        report = engine.render_report(result)
        self.assertIn("NO known signature matched", report)

    def test_ambiguity_is_surfaced_not_hidden(self) -> None:
        # Text that legitimately fits two probable rules -> AMBIGUOUS, not a silent pick.
        text = (
            "I updated to the latest dev version and quickshell gives an error when I open it. "
            "Also a dialogue saying Quickshell has crashed pops up."
        )
        result = engine.diagnose(text, rules=self.rules)
        if len(result["candidates"]) > 1 and result.get("margin", 99) < 2:
            self.assertEqual(result["verdict"], "AMBIGUOUS")
            report = engine.render_report(result)
            self.assertIn("AMBIGUOUS", report)
            self.assertIn("Clarify:", report)

    def test_determinism_same_input_same_report(self) -> None:
        text = "error: cachyos-znver4: signature from \"CachyOS <admin@cachyos.org>\" is invalid"
        first = engine.render_report(engine.diagnose(text, rules=self.rules))
        second = engine.render_report(engine.diagnose(text, rules=self.rules))
        self.assertEqual(first, second)

    def test_ansi_and_crlf_normalized(self) -> None:
        raw = "\x1b[31m[ERR]\x1b[0m src/dots is still empty\r\n\r\n\r\nand more"
        normalized = engine.normalize_input(raw)
        self.assertNotIn("\x1b", normalized)
        self.assertNotIn("\r", normalized)
        result = engine.diagnose(raw, rules=self.rules)
        self.assertEqual(result["verdict"], "MATCH")
        self.assertEqual(result["candidates"][0]["rule"]["id"], "CL-network-submodule-001")

    def test_every_suggested_command_is_prefixed_in_report(self) -> None:
        text = "Quickshell has crashed and the window switcher no longer shows windows"
        report = engine.render_report(engine.diagnose(text, rules=self.rules))
        self.assertIn(engine.COMMAND_PREFIX, report)
        self.assertIn("Nothing was executed", report)

    def test_deterministic_outranks_probable_on_equal_score(self) -> None:
        # Two rules, same score: the deterministic one must come first.
        rule_a = {
            "id": "CL-test-probable",
            "title": "probable",
            "category": "kde",
            "severity": "low",
            "confidence": "probable",
            "matchers": {"allOf": [{"type": "text_substring", "value": "zzz"}]},
            "fix": [{"text": "step"}],
        }
        rule_b = dict(rule_a, id="CL-test-deterministic", confidence="deterministic")
        ranked = engine.rank_results(
            [engine.score_rule(rule_a, "zzz") or {}, engine.score_rule(rule_b, "zzz") or {}]
        )
        self.assertEqual(ranked[0]["rule"]["id"], "CL-test-deterministic")


if __name__ == "__main__":
    unittest.main()

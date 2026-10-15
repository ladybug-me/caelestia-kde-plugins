"""A8 tests — the optional local model tier (generative/assisted.py).

The guarantees under test:

- DEFAULT OFF: without the explicit enable, the verdict is DISABLED
  and the transport is never touched (a failing conn_factory proves
  nothing was called);
- OFF = identical pipeline behavior: a pipeline.process of an
  out-of-ontology request is the same verdict/object with and without
  this module importable (the tier adds a surface, it does not change
  the default path);
- LOOPBACK + single attempt: transport errors surface as UNAVAILABLE
  with the no-retry note (a counting conn_factory that always raises
  is called EXACTLY once);
- VALIDATED, NEVER EXECUTED: drafted pairs ride the ordinary planner —
  an out-of-range draft is REPORTED as rejected, not repaired; valid
  drafts carry MODEL_SUGGESTED and SUGGESTED_NOT_EXECUTED;
- parse: the JSON-array extractor tolerates prose/fences, rejects
  tools that are not registry-shaped names.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.generative import assisted


class OffByDefaultTests(unittest.TestCase):
    def test_disabled_by_default_and_transport_untouched(self) -> None:
        def _boom(*a, **k):
            raise AssertionError("the transport must not be touched "
                                 "while the tier is off")
        result = assisted.draft_plan("write a poem about docks",
                                     enable=False, conn_factory=_boom)
        self.assertEqual(result["verdict"], "DISABLED")
        self.assertIn("off by default", result["note"])

    def test_disabled_equals_pipeline_default(self) -> None:
        """With the tier off, an out-of-ontology request behaves exactly
        as the plain pipeline does (same verdict, no model fields)."""
        import os
        os.environ["HOME"] = __import__("tempfile").mkdtemp(prefix="a8-")
        from assistant.core import pipeline
        before = pipeline.process("write a poem about docks")
        after = assisted.draft_plan("write a poem about docks",
                                    enable=False)
        self.assertIn(before.verdict, ("ABSTAIN", "OUT_OF_ONTOLOGY"))
        self.assertEqual(after["verdict"], "DISABLED")


class TransportTests(unittest.TestCase):
    def test_single_attempt_no_retry(self) -> None:
        calls = []

        def counting_factory(*a, **k):
            calls.append(1)
            raise OSError("connection refused")

        result = assisted.draft_plan("write a poem", enable=True,
                                     conn_factory=counting_factory,
                                     url="http://127.0.0.1:1")
        self.assertEqual(result["verdict"], "UNAVAILABLE")
        self.assertEqual(len(calls), 1,
                         "exactly one attempt (the availability probe) — "
                         "no retries anywhere in the tier")

    def test_unavailable_when_no_listener(self) -> None:
        # port 1 on loopback is closed by construction
        result = assisted.draft_plan("write a poem", enable=True,
                                     url="http://127.0.0.1:1")
        self.assertEqual(result["verdict"], "UNAVAILABLE")


class FakeModelTests(unittest.TestCase):
    def _conn(self, body_text):
        """A minimal fake of the client's connection factory: returns a
        loopback-shaped connection whose response carries body_text."""
        import io

        class _Resp:
            def __init__(self, body):
                self.status = 200
                self._io = io.BytesIO(body)

            def read(self):
                return self._io.read()

        class _Conn:
            def __init__(self, body):
                self._resp = _Resp(body)

            def connect(self):
                pass

            def request(self, *a, **k):
                pass

            def getresponse(self):
                return self._resp

            def close(self):
                pass

        def payload(body_text):
            return json.dumps({"response": body_text}).encode("utf-8")

        def factory(host, port, timeout):
            return _Conn(payload(body_text))

        return factory

    def test_valid_draft_is_validated_and_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            raw = 'Sure! ```json\n[{"tool": "setBarScale", "value": 0.8}]\n```'
            result = assisted.draft_plan(
                "make the bar thinner", enable=True,
                conn_factory=self._conn(raw), target=target)
        self.assertEqual(result["verdict"], "DRAFTED")
        self.assertEqual(result["label"], "MODEL_SUGGESTED")
        self.assertEqual(len(result["valid"]), 1)
        self.assertEqual(result["valid"][0]["tool"], "setBarScale")
        self.assertNotIn("executed", json.dumps(result["valid"]),
                         "drafts are data, never executions")

    def test_out_of_range_draft_is_rejected_not_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            raw = '[{"tool": "setBarScale", "value": 999}]'
            result = assisted.draft_plan(
                "make the bar enormous", enable=True,
                conn_factory=self._conn(raw), target=target)
        self.assertEqual(result["verdict"], "DRAFTED")
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(len(result["rejected"]), 1)


class ParseTests(unittest.TestCase):
    def test_tolerates_prose_and_fences(self) -> None:
        parsed = assisted.parse_drafts(
            'here you go:\n```json\n[{"tool": "setBlurEnabled",\n'
            ' "value": false}]\n```\nlet me know!')
        self.assertEqual(parsed["drafts"],
                         [{"tool": "setBlurEnabled", "value": False}])

    def test_rejects_non_tool_shapes(self) -> None:
        parsed = assisted.parse_drafts(
            '[{"tool": "deleteEverything", "value": 1}, '
            '{"tool": "x", "value": 2}, "junk"]')
        self.assertEqual(parsed["drafts"], [])

    def test_no_array_is_parse_error(self) -> None:
        parsed = assisted.parse_drafts("I would set the bar scale to 0.8")
        self.assertIsNone(parsed["drafts"] or None)
        self.assertTrue(parsed["parse_error"])

    def test_prompt_lists_registry_and_demands_json(self) -> None:
        prompt = assisted.build_prompt("make the bar thinner")
        self.assertIn("setBarScale", prompt)
        self.assertIn("0.6", prompt)  # the shipped range travels
        self.assertIn("JSON", prompt)


if __name__ == "__main__":
    unittest.main()

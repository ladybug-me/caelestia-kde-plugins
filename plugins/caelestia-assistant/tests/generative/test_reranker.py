"""generative.reranker tests — the model can reorder, nothing else.

All transport goes through a fake connection factory (the established
test_assisted.py pattern): no test touches a network, no test needs a
model. The contract under test: constrained parsing, registry-name
validation, MODEL_SUGGESTED marking, UNCHANGED degradation, and the
capability gate.
"""
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.generative import reranker  # noqa: E402
from assistant.capabilities.generative import client  # noqa: E402

CANDIDATES = [
    {"name": "setBarScale", "description": "scale of the bar"},
    {"name": "setDockIconSize", "description": "dock icon size"},
    {"name": "setBlur", "description": "blur strength"},
]


def _factory(answer: str):
    """Fake loopback connection whose /api/generate response carries
    the model text `answer` — the wire format client.generate parses."""
    body = json.dumps({"response": answer}).encode("utf-8")

    class _Resp:
        status = 200

        def __init__(self):
            self._io = io.BytesIO(body)

        def read(self):
            return self._io.read()

    class _Conn:
        def connect(self):
            pass

        def request(self, *a, **k):
            pass

        def getresponse(self):
            return _Resp()

        def close(self):
            pass

    return lambda host, port, timeout: _Conn()


def _boom(host, port, timeout):
    raise OSError("refused — and the module must degrade, not retry")


class TestConstrainedParsing(unittest.TestCase):
    def test_exact_name_line_wins(self):
        self.assertEqual(
            reranker._parse_choice("setBlur", ["setBarScale", "setBlur"]),
            "setBlur")

    def test_fence_and_commentary_are_dropped(self):
        raw = "```bash\nThe best candidate is:\nsetDockIconSize\n```\n"
        self.assertEqual(
            reranker._parse_choice(raw, [c["name"] for c in CANDIDATES]),
            "setDockIconSize")

    def test_hallucinated_name_is_never_accepted(self):
        self.assertIsNone(
            reranker._parse_choice("setBarWidth", [c["name"]
                                                   for c in CANDIDATES]))

    def test_empty_raw_is_none(self):
        self.assertIsNone(reranker._parse_choice("", ["setBarScale"]))


class TestRerankContract(unittest.TestCase):
    def test_model_suggested_reorders_and_marks(self):
        r = reranker.rerank("make the dock icons bigger", CANDIDATES,
                            conn_factory=_factory("setDockIconSize"),
                            capability_enabled=True)
        self.assertEqual(r["verdict"], "MODEL_SUGGESTED")
        self.assertEqual(r["suggested"], "setDockIconSize")
        self.assertEqual(r["order"][0], "setDockIconSize")
        # a STABLE reorder of the same set: nothing added, nothing lost
        self.assertEqual(sorted(r["order"]),
                         sorted(c["name"] for c in CANDIDATES))
        self.assertEqual([c["name"] for c in r["candidates"]], r["order"])

    def test_input_list_is_never_mutated(self):
        snapshot = json.dumps(CANDIDATES, sort_keys=True)
        reranker.rerank("make the dock icons bigger", CANDIDATES,
                        conn_factory=_factory("setBlur"),
                        capability_enabled=True)
        self.assertEqual(json.dumps(CANDIDATES, sort_keys=True), snapshot)

    def test_model_agreement_is_unchanged(self):
        r = reranker.rerank("scale the bar", CANDIDATES,
                            conn_factory=_factory("setBarScale"),
                            capability_enabled=True)
        self.assertEqual(r["verdict"], "UNCHANGED")
        self.assertIn("agreed", r["reason"])

    def test_server_down_degrades_to_unchanged(self):
        r = reranker.rerank("scale the bar", CANDIDATES,
                            conn_factory=_boom, capability_enabled=True)
        self.assertEqual(r["verdict"], "UNCHANGED")
        self.assertIn("unavailable", r["reason"])

    def test_no_usable_answer_degrades(self):
        r = reranker.rerank("scale the bar", CANDIDATES,
                            conn_factory=_factory("I cannot do that"),
                            capability_enabled=True)
        self.assertEqual(r["verdict"], "UNCHANGED")
        self.assertIn("no candidate", r["reason"])

    def test_empty_inputs(self):
        self.assertEqual(reranker.rerank("", CANDIDATES,
                                         capability_enabled=True)
                         ["verdict"], "UNCHANGED")
        self.assertEqual(reranker.rerank("query", [],
                                         capability_enabled=True)
                         ["verdict"], "UNCHANGED")

    def test_capability_gate_defaults_off(self):
        # the DEFAULTS table ships model_reranker=False; with no
        # capability file the answer is UNCHANGED before any transport
        # exists (conn_factory absent AND never called)
        r = reranker.rerank("scale the bar", CANDIDATES,
                            conn_factory=_factory("setBlur"))
        self.assertEqual(r["verdict"], "UNCHANGED")
        self.assertIn("capability", r["reason"])


class TestBridgeOp(unittest.TestCase):
    def test_op_is_gated_off_by_default(self):
        from assistant.capabilities.brain import bridge
        r = bridge.handle({"op": "model_rerank",
                           "query": "scale the bar",
                           "candidates": CANDIDATES}, None, None)
        self.assertTrue(r["ok"])  # op-level errors ride in the result
        self.assertIn("model_reranker", r["result"]["error"])

    def test_op_requires_candidates_when_enabled(self):
        from assistant.capabilities.brain import bridge
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            capfile = Path(tmp) / "caps.json"
            from assistant.core.features import DEFAULTS
            caps = dict(DEFAULTS)
            caps["model_reranker"] = True
            capfile.write_text(json.dumps(caps))
            old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
            os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(capfile)
            try:
                r = bridge.handle({"op": "model_rerank",
                                   "query": "scale the bar"}, None, None)
            finally:
                if old is None:
                    os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
                else:
                    os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old
        self.assertTrue(r["ok"])
        self.assertIn("candidates", r["result"]["error"])


class TestVerifiedCandidates(unittest.TestCase):
    def test_shape_and_sub1b_claim(self):
        for c in reranker.VERIFIED_CANDIDATES:
            self.assertIn(":", c["name"])  # an ollama tag, not a family
            self.assertTrue(c["download"].endswith("MB"))
            self.assertEqual(c["license"], "Apache 2.0")
        # every named candidate is sub-1B: the download number is the
        # library's own and stays under 1 GB
        for c in reranker.VERIFIED_CANDIDATES:
            mb = float(c["download"].split()[0])
            self.assertLess(mb, 1024.0)


class TestRender(unittest.TestCase):
    def test_model_suggested_is_loud(self):
        r = reranker.rerank("make the dock icons bigger", CANDIDATES,
                            conn_factory=_factory("setDockIconSize"),
                            capability_enabled=True)
        lines = reranker.render_lines(r)
        self.assertIn("[MODEL_SUGGESTED]", lines[0])
        self.assertTrue(any("plan/apply gates" in ln for ln in lines))

    def test_unchanged_is_explicit(self):
        r = reranker.rerank("scale the bar", CANDIDATES,
                            conn_factory=_boom, capability_enabled=True)
        lines = reranker.render_lines(r)
        self.assertIn("[UNCHANGED]", lines[0])
        self.assertTrue(any("deterministic order stands" in ln
                            for ln in lines))


if __name__ == "__main__":
    unittest.main()

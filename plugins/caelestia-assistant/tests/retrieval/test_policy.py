"""Policy tests: the retrieval layer stays inside the assistant safety box."""

from __future__ import annotations

import unittest

from assistant.capabilities.diagnostics import schema_lint
from assistant.core.ask_pipeline import assist
from assistant.capabilities.retrieval import cli, indexer, search  # noqa: F401  (imports must stay clean)

LAYER1_TEXT = (
    "ERROR: Unrecognized pragma \"DefaultEnv QS_NO_RELOAD_POPUP=1\"\n"
    "Error: org.freedesktop.DBus.Error.UnknownObject\n"
    "No such object path '/component/caelestia'"
)


class TestPipelineLayering(unittest.TestCase):
    def test_layer1_match_stays_on_rules(self) -> None:
        payload = assist(LAYER1_TEXT)
        self.assertEqual(payload["layer"], "rules")
        self.assertIn(payload["verdict"], ("MATCH", "AMBIGUOUS"))
        self.assertTrue(payload["candidates"])

    def test_no_match_falls_back_to_retrieval(self) -> None:
        # This phrasing deliberately misses every Layer 1 rule (verified) but
        # has a strong retrieval hit in the troubleshooting corpus.
        payload = assist("how do I free disk space used by the shell")
        self.assertEqual(payload["layer"], "retrieval")
        self.assertEqual(payload["verdict"], "NO_MATCH")
        self.assertIn("NOT verified diagnoses", payload["banner"])
        self.assertTrue(payload["results"])
        for hit in payload["results"]:
            self.assertIn("doc_id", hit)
            self.assertIn("snippet", hit)

    def test_unknown_text_is_honestly_empty(self) -> None:
        payload = assist("zzzqqq wugga blorptastic frobnicate the flimflam")
        self.assertEqual(payload["layer"], "retrieval")
        self.assertEqual(payload["results"], [])
        self.assertIn("honest", payload["note"].lower())


if __name__ == "__main__":
    unittest.main()

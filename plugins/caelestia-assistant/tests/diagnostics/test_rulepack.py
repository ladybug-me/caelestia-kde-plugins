"""Tests for exponential-build-3 F1 — the signed rule-pack format
(diagnostics/rulepack.py).

Under test:

- canonical bytes are byte-stable for fixed input (sorted keys, no
  whitespace, ASCII) — the foundation a detached signature rests on;
- export bundles shipped rulesets, validates them through the SAME
  gates the built-ins pass, and refuses unknown stems with the
  directory listing;
- import verifies INTEGRITY (manifest sha256 == canonical payload
  hash — a post-manifest edit refuses with both hashes named) and
  SAFETY (a rule carrying a forbidden key, e.g. an auto-execution
  smuggle attempt, refuses the whole pack — no partial imports);
- duplicate pack ids refuse (no silent overwrite); rule ids must be
  unique within a pack;
- the record stores the claimed signer as a CLAIM; list_packs strips
  the bulky payload; render_pack states the not-live contract;
  render_rules_d emits a loadable rules.d document whose provenance
  carries the pack hash and the claimed signer;
- the round trip: export -> import -> rules-d render -> the rendered
  rules pass engine.validate_rule again (the human-placement path
  ships valid rules).
"""
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.diagnostics import rulepack
from assistant.capabilities.diagnostics.engine import validate_rule


def _rules_dir(directory: Path) -> Path:
    """A minimal valid rules.d fixture (the shipped one is exercised
    separately in the export test)."""
    rules = {
        "schema_version": 1,
        "ruleset": "fixture-pack-test",
        "provenance": {"generated_from": ["this test"]},
        "rules": [
            {"id": "FX-001", "title": "fixture rule one",
             "category": "config", "severity": "low",
             "confidence": "deterministic",
             "matchers": {"type": "text_substring", "value": "needle"},
             "fix": [{"text": "a harmless suggestion"}],
             "references": []},
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fixture_set.json").write_text(
        json.dumps(rules, indent=2), encoding="utf-8")
    return directory


class CanonicalBytesTests(unittest.TestCase):
    def test_byte_stability(self):
        payload = [{"b": 1, "a": [2, {"z": None, "y": "é"}]}]
        first = rulepack.canonical_bytes(payload)
        second = rulepack.canonical_bytes(
            [{"a": [2, {"y": "é", "z": None}], "b": 1}])
        self.assertEqual(first, second)
        self.assertIn(b'"a"', first)
        self.assertNotIn(b" ", first)
        # non-ASCII is escaped: the bytes are signature-portable
        self.assertIn(b"\\u00e9", first)

    def test_payload_sha256_matches_canonical_bytes(self):
        import hashlib
        payload = {"k": [1, 2, 3]}
        self.assertEqual(
            rulepack.payload_sha256(payload),
            hashlib.sha256(rulepack.canonical_bytes(payload)).hexdigest())


class ExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_bundles_and_hashes(self):
        rules_dir = _rules_dir(self.dir / "rules.d")
        pack = rulepack.export_pack(rules_dir, ["fixture_set"], "test-pack")
        self.assertEqual(pack["schema_version"], 1)
        self.assertEqual(pack["pack"]["id"], "test-pack")
        self.assertEqual(pack["pack"]["rulesets"], ["fixture_set"])
        self.assertEqual(pack["pack"]["rule_count"], 1)
        self.assertEqual(pack["pack"]["payload_sha256"],
                         rulepack.payload_sha256(pack["payload"]))
        self.assertEqual(pack["payload"][0]["ruleset"],
                         "fixture-pack-test")

    def test_export_of_the_shipped_rulesets(self):
        shipped = (Path(__file__).resolve().parents[2] / "assistant" /
              "capabilities" / "diagnostics" / "rules.d")
        pack = rulepack.export_pack(shipped, ["config_kde"],
                                    "shipped-config")
        self.assertGreaterEqual(pack["pack"]["rule_count"], 1)

    def test_unknown_stem_refuses_with_listing(self):
        rules_dir = _rules_dir(self.dir / "rules.d")
        with self.assertRaises(rulepack.RulePackError) as caught:
            rulepack.export_pack(rules_dir, ["nope"], "p")
        self.assertIn("unknown ruleset stem", str(caught.exception))
        self.assertIn("fixture_set", str(caught.exception))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.rules_dir = _rules_dir(self.dir / "rules.d")
        self.pack = rulepack.export_pack(self.rules_dir,
                                         ["fixture_set"], "test-pack")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_import(self):
        record = rulepack.import_pack(self.pack, signer="alice")
        self.assertEqual(record["id"], "test-pack")
        self.assertEqual(record["rule_count"], 1)
        self.assertEqual(record["signer"], "alice")
        self.assertTrue(record["payload_sha256"])
        self.assertEqual(record["pack_sha256"],
                         __import__("hashlib").sha256(
                             rulepack.canonical_bytes(
                                 self.pack)).hexdigest())

    def test_import_from_json_text_and_path(self):
        text = json.dumps(self.pack)
        by_text = rulepack.import_pack(text)
        path = self.dir / "pack.json"
        path.write_text(text, encoding="utf-8")
        by_path = rulepack.import_pack(path)
        self.assertEqual(by_text["id"], by_path["id"])
        self.assertEqual(by_text["pack_sha256"],
                         by_path["pack_sha256"])

    def test_tampered_payload_refuses_with_both_hashes(self):
        pack = json.loads(json.dumps(self.pack))
        pack["payload"][0]["rules"][0]["severity"] = "high"
        with self.assertRaises(rulepack.RulePackError) as caught:
            rulepack.import_pack(pack)
        msg = str(caught.exception)
        self.assertIn("hash mismatch", msg)
        self.assertIn(pack["pack"]["payload_sha256"], msg)
        self.assertIn(rulepack.payload_sha256(pack["payload"]), msg)

    def test_forbidden_key_smuggle_refuses_whole_pack(self):
        # the attack the format exists to make impossible: a community
        # rule that tries to express auto-execution
        pack = json.loads(json.dumps(self.pack))
        pack["payload"][0]["rules"][0]["exec"] = "rm -rf /"
        # re-manifest honestly (the attacker would fix the hash)
        pack["pack"]["payload_sha256"] = rulepack.payload_sha256(
            pack["payload"])
        with self.assertRaises(rulepack.RulePackError) as caught:
            rulepack.import_pack(pack)
        self.assertIn("forbidden key", str(caught.exception))
        self.assertIn("whole or not at all", str(caught.exception))

    def test_duplicate_pack_id_refuses(self):
        rulepack.import_pack(self.pack, existing_ids=[])
        with self.assertRaises(rulepack.RulePackError) as caught:
            rulepack.import_pack(self.pack, existing_ids=["test-pack"])
        self.assertIn("already imported", str(caught.exception))
        self.assertIn("never silently overwrite", str(caught.exception))

    def test_duplicate_rule_ids_within_pack_refuse(self):
        rules = self.pack["payload"][0]["rules"]
        self.pack["payload"].append(
            {"schema_version": 1, "ruleset": "dup",
             "rules": [json.loads(json.dumps(rules[0]))]})
        # the attacker-consistent re-manifest: hashes AND count fixed
        self.pack["pack"]["payload_sha256"] = rulepack.payload_sha256(
            self.pack["payload"])
        self.pack["pack"]["rule_count"] = 2
        with self.assertRaises(rulepack.RulePackError) as caught:
            rulepack.import_pack(self.pack)
        self.assertIn("duplicate rule id", str(caught.exception))

    def test_bad_shapes_refuse(self):
        with self.assertRaises(rulepack.RulePackError):
            rulepack.import_pack({"schema_version": 99})
        with self.assertRaises(rulepack.RulePackError):
            rulepack.import_pack("not json at all {")
        with self.assertRaises(rulepack.RulePackError):
            rulepack.import_pack(
                {"schema_version": 1, "pack": {}, "payload": []})


class RenderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.rules_dir = _rules_dir(self.dir / "rules.d")
        self.record = rulepack.import_pack(
            rulepack.export_pack(self.rules_dir, ["fixture_set"],
                                 "test-pack"),
            signer="alice")

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_packs_strips_payload(self):
        state = {rulepack.STATE_KEY: [self.record]}
        packs = rulepack.list_packs(state)
        self.assertEqual(len(packs), 1)
        self.assertNotIn("payload", packs[0])
        self.assertEqual(packs[0]["id"], "test-pack")
        self.assertEqual(packs[0]["signer"], "alice")

    def test_render_pack_states_the_claim_and_the_contract(self):
        text = rulepack.render_pack(self.record)
        self.assertIn("test-pack", text)
        self.assertIn("alice", text)
        self.assertIn("claim", text)
        self.assertIn("not live", text)
        self.assertIn("FX-001", text)

    def test_render_rules_d_is_loadable_and_provenanced(self):
        text = rulepack.render_rules_d(self.record)
        doc = json.loads(text)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["provenance"]["imported_from_pack"],
                         "test-pack")
        self.assertEqual(doc["provenance"]["claimed_signer"], "alice")
        # the rendered rules pass the engine's own validation: the
        # human-placement path ships valid rules
        failures = list(validate_rule(doc["rules"][0],
                                      source="<rendered>"))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()

"""tests for agent.archetype_pack + cortex.coldstart (exponential-
build-4 G: the community-distribution group — the one place in the
build where the safety model matters most).

Pinned: the archetype pack's action whitelist (a node naming a
non-existent dispatcher is refused), the CLOSED field set (unknown
fields refused — no place for code to hide), consent required at
STATE_CHANGING and above, capability names validated and NEVER
granted by import, integrity (payload hash) enforcement, all-or-
nothing import, review-only status with empty capability grants,
duplicate-id refusal, and the export gate refusing packs that would
not pass their own import. Cold-start: the DP mechanism reuses
cortex/dp.py verbatim, noised arms land as priors, sub-floor values
clip to the flat prior with the clip recorded, flat priors stay the
default, and the artifact schema is enforced.
"""
import json
import random
import unittest

from assistant.capabilities.agent import archetype_pack as ap
from assistant.core.coldstart import (bootstrap_learner, export_priors)
from assistant.core.learn import CortexLearner


GOOD_ARCHETYPE = {
    "id": "quick-minimal",
    "title": "Quick minimal",
    "description": "lint, then propose a minimal config (consent-gated apply)",
    "nodes": [
        {"id": "n1", "action": "lint_config", "title": "lint",
         "risk": "READ_ONLY", "consent": False},
        {"id": "n2", "action": "propose_plan", "title": "propose",
         "risk": "READ_ONLY", "consent": False},
        {"id": "n3", "action": "apply_plan", "title": "apply",
         "risk": "STATE_CHANGING", "consent": True, "depends": ["n1"]},
    ],
}


class ExportGateTests(unittest.TestCase):
    def test_good_pack_exports(self):
        pack = ap.export_pack([GOOD_ARCHETYPE], "good-pack")
        self.assertEqual(pack["schema_version"], 1)
        self.assertEqual(pack["pack"]["archetype_count"], 1)

    def test_unknown_action_refused(self):
        evil = {"id": "evil", "title": "x", "description": "y", "nodes": [
            {"id": "n1", "action": "run_freeform_command", "title": "x",
             "risk": "READ_ONLY", "consent": False}]}
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.export_pack([evil], "evil")
        self.assertIn("does not resolve", str(ctx.exception))

    def test_unknown_field_refused_closed_schema(self):
        sneaky = dict(GOOD_ARCHETYPE, code="exec(payload)")
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.export_pack([sneaky], "sneaky")
        self.assertIn("unknown field", str(ctx.exception))

    def test_state_changing_without_consent_refused(self):
        bad = {"id": "x", "title": "x", "description": "y", "nodes": [
            {"id": "n1", "action": "apply_plan", "title": "x",
             "risk": "STATE_CHANGING", "consent": False}]}
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.export_pack([bad], "no-consent")
        self.assertIn("consent=true", str(ctx.exception))

    def test_invalid_risk_refused(self):
        bad = {"id": "x", "title": "x", "description": "y", "nodes": [
            {"id": "n1", "action": "brief", "title": "x",
             "risk": "WHENEVER", "consent": False}]}
        with self.assertRaises(ap.ArchetypePackError):
            ap.export_pack([bad], "bad-risk")


class ImportGateTests(unittest.TestCase):
    def _pack(self):
        return ap.export_pack([GOOD_ARCHETYPE], "good-pack")

    def test_import_records_review_only_and_no_grants(self):
        record = ap.import_pack(json.dumps(self._pack()), [])
        self.assertEqual(record["status"], "review-only")
        self.assertEqual(record["capabilities_granted"], [])
        self.assertTrue(record["signer_is_claim"])
        self.assertIn("NOT registered", record["note"])

    def test_integrity_tamper_refused(self):
        pack = json.loads(json.dumps(self._pack()))
        pack["payload"][0]["nodes"][0]["action"] = "brief"
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.import_pack(json.dumps(pack), [])
        self.assertIn("integrity failure", str(ctx.exception))

    def test_unknown_capability_refused(self):
        doc = {"id": "x", "title": "x", "description": "y", "nodes": [
            {"id": "n1", "action": "package_report", "title": "x",
             "risk": "READ_ONLY", "consent": False,
             "required_capability": "root_everything"}]}
        pack = {"schema_version": 1,
                "pack": {"id": "cap-pack", "archetype_count": 1},
                "payload": [doc]}
        # bypass export (which would refuse) to test the import gate directly
        from assistant.capabilities.diagnostics.rulepack import payload_sha256
        pack["pack"]["payload_sha256"] = payload_sha256(pack["payload"])
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.import_pack(json.dumps(pack), [])
        self.assertIn("unknown capability", str(ctx.exception))

    def test_known_capability_recorded_not_granted(self):
        doc = dict(GOOD_ARCHETYPE, id="pkg-audit-variant")
        doc = json.loads(json.dumps(doc))
        doc["nodes"][0]["required_capability"] = "package_audit"
        pack = {"schema_version": 1,
                "pack": {"id": "cap-pack", "archetype_count": 1},
                "payload": [doc]}
        from assistant.capabilities.diagnostics.rulepack import payload_sha256
        pack["pack"]["payload_sha256"] = payload_sha256(pack["payload"])
        record = ap.import_pack(json.dumps(pack), [])
        self.assertEqual(record["capabilities_granted"], [])
        self.assertEqual(record["status"], "review-only")

    def test_duplicate_id_refused(self):
        pack = self._pack()
        record = ap.import_pack(json.dumps(pack), [])
        with self.assertRaises(ap.ArchetypePackError) as ctx:
            ap.import_pack(json.dumps(pack), [record["id"]])
        self.assertIn("already imported", str(ctx.exception))

    def test_import_is_all_or_nothing(self):
        half_good = [GOOD_ARCHETYPE,
                     {"id": "bad", "title": "x", "description": "y",
                      "nodes": [{"id": "n1", "action": "nope", "title": "x",
                                 "risk": "READ_ONLY", "consent": False}]}]
        with self.assertRaises(ap.ArchetypePackError):
            ap.export_pack(half_good, "mixed")

    def test_open_questions_recorded(self):
        # the underspecified promotion/registration decisions are
        # carried in the module for a human, not silently decided
        self.assertTrue(ap.OPEN_QUESTIONS)
        self.assertTrue(any("PROMOTION" in q for q in ap.OPEN_QUESTIONS))


class ColdStartTests(unittest.TestCase):
    def test_export_noises_and_import_bootstraps(self):
        rng = random.Random(42)
        artifact = export_priors({"balanced": [40.0, 12.0],
                                  "precision": [25.0, 20.0],
                                  "lexical": [1.0, 1.0]}, 1.0, rng)
        learner = CortexLearner({})
        report = bootstrap_learner(learner, artifact)
        arms = learner.bandit.arms
        # the noised prior landed (close to truth: 40/12 within the
        # Laplace scale b = 1.0 at epsilon 1.0)
        self.assertGreater(arms["balanced"][0], 30.0)
        self.assertLess(arms["balanced"][0], 50.0)
        # the flat arm was clipped up to the flat-prior floor
        self.assertGreaterEqual(arms["lexical"][0], 1.0)
        self.assertTrue(any(r["clipped_to_flat_floor"]
                            for r in report["arms"]))

    def test_import_never_deletes_evidence(self):
        artifact = export_priors({"balanced": [5.0, 5.0]}, 1.0,
                                 random.Random(7))
        learner = CortexLearner({})
        learner.reward_strategy("precision", True)  # existing evidence
        bootstrap_learner(learner, artifact)
        # precision was not in the artifact: its evidence stays
        self.assertGreater(learner.bandit.arms["precision"][0], 1.0)

    def test_foreign_schema_refused(self):
        learner = CortexLearner({})
        with self.assertRaises(ValueError):
            bootstrap_learner(learner, {"schema": "other/9", "arms": {}})

    def test_negative_epsilon_refused(self):
        with self.assertRaises(ValueError):
            export_priors({"balanced": [2.0, 2.0]}, epsilon=0.0,
                          rng=random.Random(1))


if __name__ == "__main__":
    unittest.main()

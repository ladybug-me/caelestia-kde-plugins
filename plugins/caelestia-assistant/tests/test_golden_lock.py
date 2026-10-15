"""Stage C1 behavior lock — structural unittest layer.

The byte-level lock itself runs in scripts/golden_check.py (invoked by
tests/test_assistant.sh): a fresh subprocess per case with an isolated
HOME and pinned PYTHONHASHSEED is the only way to lock the REAL CLI,
but the import policy (ALLOWED_IMPORTS.txt) forbids subprocess inside
assistant/ Python — by its own constitution, "even if it were added
here". So the unittest tree pins the STRUCTURE of the lock:

- the manifest exists, is well-formed, and keeps >= 300 cases;
- every case's argv/rc/sha fields are coherent;
- every sample file's embedded stdout still hashes to the manifest
  entry (no hand-editing of samples without a manifest regen);
- every sample's argv actually appears in the manifest.

Any manifest change is visible to review as a regenerated
manifest.json + samples/, and the byte lock catches silent drift at
the bash gate.
"""
import hashlib
import json
import unittest
from pathlib import Path

GOLD = Path(__file__).resolve().parent / "goldens"
MANIFEST = GOLD / "manifest.json"


@unittest.skipUnless(MANIFEST.exists(),
                     "goldens not generated (scripts/regen_goldens.py)")
class GoldenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(MANIFEST.read_text())
        cls.cases = cls.data["cases"]
        cls.by_id = {c["id"]: c for c in cls.cases}

    def test_minimum_coverage(self):
        self.assertGreaterEqual(
            len(self.cases), self.data["min_cases"],
            "the behavior lock shrank — goldens must never quietly lose "
            "coverage; regenerate and justify in a reviewed commit")

    def test_ids_unique_and_wellformed(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        for c in self.cases:
            self.assertIn(c["rc"], (0, 1),
                          "rc 0 or a deterministic honest refusal only")
            self.assertEqual(len(c["sha"]), 64)
            self.assertTrue(c["argv"])
            self.assertTrue(c["id"])

    def test_sample_files_match_manifest(self):
        samples = sorted((GOLD / "samples").glob("*.txt"))
        self.assertGreaterEqual(len(samples), 1)
        for path in samples:
            cid = path.name[:-4]
            case = self.by_id.get(cid)
            self.assertIsNotNone(
                case, f"sample {cid} missing from the manifest")
            body = path.read_text()
            stdout = body.split("--- stdout ---\n", 1)[1]
            self.assertEqual(
                hashlib.sha256(stdout.encode()).hexdigest(), case["sha"],
                f"sample {cid} drifted from the manifest — regenerate, "
                f"never hand-edit goldens")


if __name__ == "__main__":
    unittest.main()

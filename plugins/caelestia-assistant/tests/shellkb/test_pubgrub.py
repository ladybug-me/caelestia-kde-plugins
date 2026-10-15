"""shellkb.pubgrub tests — specs, bounded search, because-chains."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.shellkb import pubgrub  # noqa: E402


def _uni(**packages):
    return {"packages": packages}


class TestSpecs(unittest.TestCase):
    def test_parse_and_match(self):
        s = pubgrub.parse_spec(">=1,<3")
        self.assertTrue(pubgrub.spec_ok("1.5", s))
        self.assertTrue(pubgrub.spec_ok("1", s))
        self.assertFalse(pubgrub.spec_ok("3", s))
        self.assertFalse(pubgrub.spec_ok("0.9", s))

    def test_bare_version_pins_exactly(self):
        s = pubgrub.parse_spec("1.2.0")
        self.assertTrue(pubgrub.spec_ok("1.2.0", s))
        self.assertFalse(pubgrub.spec_ok("1.2.1", s))

    def test_caret_and_tilde(self):
        caret = pubgrub.parse_spec("^1.2")
        self.assertTrue(pubgrub.spec_ok("1.9", caret))
        self.assertFalse(pubgrub.spec_ok("2.0", caret))
        tilde = pubgrub.parse_spec("~1.2")
        self.assertTrue(pubgrub.spec_ok("1.2.9", tilde))
        self.assertFalse(pubgrub.spec_ok("1.3", tilde))

    def test_numeric_not_lexicographic(self):
        s = pubgrub.parse_spec(">=1.9")
        self.assertTrue(pubgrub.spec_ok("1.10", s))
        self.assertFalse(pubgrub.spec_ok("1.8", s))

    def test_garbage_spec_raises(self):
        with self.assertRaises(ValueError):
            pubgrub.parse_spec("latest")


class TestSolver(unittest.TestCase):
    def test_satisfiable_picks_highest_versions(self):
        uni = _uni(**json.loads(pubgrub.UNIVERSE_PATH.read_text())
                   ["packages"])
        d = pubgrub.solve(uni, "caelestia-shell")
        self.assertEqual(d["verdict"], "SATISFIABLE")
        self.assertEqual(d["choices"]["quickshell"], "0.21.0")
        self.assertEqual(d["choices"]["qt6-base"], "6.9.0")

    def test_old_root_gets_old_dep(self):
        uni = _uni(**json.loads(pubgrub.UNIVERSE_PATH.read_text())
                   ["packages"])
        d = pubgrub.solve(uni, "caelestia-shell", root_spec="0.9.0")
        self.assertEqual(d["verdict"], "SATISFIABLE")
        self.assertEqual(d["choices"]["quickshell"], "0.18.0",
                         "0.9.0 pins quickshell >=0.18,<0.19")

    def test_unsat_produces_because_chain(self):
        uni = _uni(
            app={"1.0": {"depends": {"quickshell": ">=0.19",
                                     "qt6-declarative": "<6.7"}}},
            quickshell={"0.19.0": {"depends":
                                   {"qt6-declarative": ">=6.7"}},
                        "0.18.0": {"depends":
                                   {"qt6-declarative": "<6.7"}}},
            **{"qt6-declarative": {"6.9.0": {"depends": {}},
                                   "6.6.0": {"depends": {}}}})
        d = pubgrub.solve(uni, "app")
        self.assertEqual(d["verdict"], "UNSATISFIABLE")
        text = " ".join(d["because"])
        self.assertIn("requires qt6-declarative <6.7", text)
        self.assertIn("no version satisfying", text)
        self.assertIn(">=6.7", text)

    def test_conflict_edge_kills_later_assignment(self):
        uni = _uni(
            app={"1.0": {"depends": {"quickshell": ">=0.19",
                                     "legacy-shell": "*"}}},
            **{"legacy-shell": {"2.0": {"conflicts":
                                        {"quickshell": "*"}}}},
            quickshell={"0.19.0": {"depends": {}}})
        d = pubgrub.solve(uni, "app")
        self.assertEqual(d["verdict"], "UNSATISFIABLE")
        self.assertTrue(any("conflicts with" in line
                            for line in d["because"]))

    def test_unknown_package_is_honest(self):
        d = pubgrub.solve(_uni(), "nope")
        self.assertEqual(d["verdict"], "UNKNOWN_PACKAGE")

    def test_search_is_bounded(self):
        # a deep chain that would explode without the node budget
        packages = {}
        for i in range(60):
            packages[f"p{i:02d}"] = {
                f"{j}.0": {"depends": {f"p{i + 1:02d}": "*"}}
                for j in range(1, 5)}
        packages["p60"] = {"1.0": {"depends": {}}}
        d = pubgrub.solve(_uni(**packages), "p00")
        self.assertIn(d["verdict"],
                      ("SATISFIABLE", "ABSTAIN", "UNSATISFIABLE"))
        self.assertLessEqual(d.get("n_search_nodes", 0), 4000)


class TestInstalledCheck(unittest.TestCase):
    UNIVERSE = json.loads(pubgrub.UNIVERSE_PATH.read_text())

    def test_satisfied_install(self):
        d = pubgrub.check_installed(
            self.UNIVERSE,
            {"caelestia-shell": "1.0.0", "quickshell": "0.21.0",
             "qt6-base": "6.9.0", "qt6-declarative": "6.9.0",
             "libplasma": "6.2.0", "kwin": "6.2.0",
             "plasma-workspace": "6.2.0"},
            "caelestia-shell")
        self.assertEqual(d["verdict"], "INSTALLED_SATISFIES")

    def test_too_old_version_reported_with_requirer(self):
        d = pubgrub.check_installed(
            self.UNIVERSE,
            {"caelestia-shell": "1.0.0", "quickshell": "0.18.0",
             "qt6-base": "6.9.0", "qt6-declarative": "6.9.0",
             "libplasma": "6.2.0", "kwin": "6.2.0",
             "plasma-workspace": "6.2.0"},
            "caelestia-shell")
        self.assertEqual(d["verdict"], "INSTALLED_CONFLICTS")
        self.assertTrue(any("requires quickshell >=0.19" in p
                            for p in d["problems"]))

    def test_missing_package_reported(self):
        d = pubgrub.check_installed(
            self.UNIVERSE, {"caelestia-shell": "1.0.0"},
            "caelestia-shell")
        self.assertEqual(d["verdict"], "INSTALLED_CONFLICTS")
        self.assertTrue(any("not installed" in p for p in d["problems"]))


class TestFixture(unittest.TestCase):
    def test_fixture_declares_its_illustrative_nature(self):
        data = json.loads(pubgrub.UNIVERSE_PATH.read_text())
        self.assertIn("ILLUSTRATIVE", data["note"])
        self.assertIn("scripts", data["note"])
        self.assertIn("quickshell", data["packages"])


if __name__ == "__main__":
    unittest.main()

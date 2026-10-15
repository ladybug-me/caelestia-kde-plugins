"""Cleanup-brain falsifying tests: symlinks, unreadable files, hash
funnel soundness, survival decay, quarantine TTL restore."""
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assistant.capabilities.dedupe.scan import (
    quarantine, restore, scan_duplicates, survival_scores)


def _mk(root, rel, content):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)


class TestScanDuplicates(unittest.TestCase):
    def test_finds_true_duplicates(self):
        with TemporaryDirectory() as root:
            a = _mk(root, "a/f.bin", "same content here")
            b = _mk(root, "b/f.bin", "same content here")
            c = _mk(root, "c/f.bin", "different!")
            groups = scan_duplicates([root])
            self.assertEqual(len(groups), 1)
            self.assertEqual(sorted(groups[0]["paths"]), sorted([a, b]))
            self.assertEqual(groups[0]["wasted"], len("same content here"))

    def test_symlinks_are_ignored(self):
        with TemporaryDirectory() as root:
            a = _mk(root, "a/f.bin", "payload")
            os.symlink(a, Path(root) / "link.bin")
            self.assertEqual(scan_duplicates([root]), [])

    def test_same_partial_different_full_do_not_match(self):
        # same first 4KiB, different tails: the full-hash stage separates
        with TemporaryDirectory() as root:
            head = "x" * 5000
            _mk(root, "a/f", head + "AAA")
            _mk(root, "b/f", head + "BBB")
            self.assertEqual(scan_duplicates([root]), [])

    def test_deterministic_order(self):
        with TemporaryDirectory() as root:
            _mk(root, "z/1", "dup")
            _mk(root, "a/1", "dup")
            self.assertEqual(scan_duplicates([root]), scan_duplicates([root]))


class TestSurvival(unittest.TestCase):
    def test_recent_is_one(self):
        now = time.time()
        with TemporaryDirectory() as root:
            p = _mk(root, "f", "x")
            os.utime(p, (now, now))
            self.assertEqual(survival_scores([p], now)[p], 1.0)

    def test_decay_halves_monthly(self):
        now = time.time()
        with TemporaryDirectory() as root:
            p = _mk(root, "f", "x")
            old = now - 366 * 86400  # ~1 year untouched
            os.utime(p, (old, old))
            score = survival_scores([p], now)[p]
            self.assertLess(score, 0.1)
            self.assertGreaterEqual(score, 0.0)

    def test_unreadable_is_zero_not_crash(self):
        self.assertEqual(
            survival_scores(["/nonexistent/x"], time.time())
            ["/nonexistent/x"], 0.0)


class TestQuarantine(unittest.TestCase):
    def test_move_and_ttl_restore(self):
        now = time.time()
        with TemporaryDirectory() as root, TemporaryDirectory() as qdir:
            p = _mk(root, "sub/f.txt", "precious")
            out = quarantine([p], qdir, now=now, ttl_days=1)
            self.assertEqual(len(out["moved"]), 1)
            self.assertFalse(os.path.exists(p))
            # not expired yet: restore leaves it quarantined
            self.assertEqual(restore(qdir, now=now)["restored"], [])
            # expired: restore returns it
            out = restore(qdir, now=now + 3 * 86400)
            self.assertEqual(out["restored"], [p])
            self.assertTrue(os.path.exists(p))

    def test_missing_file_is_an_honest_error(self):
        now = time.time()
        with TemporaryDirectory() as qdir:
            out = quarantine(["/nonexistent/f"], qdir, now=now)
            self.assertEqual(out["moved"], [])
            self.assertEqual(len(out["errors"]), 1)


if __name__ == "__main__":
    unittest.main()

"""scan.simhash — SimHash near-duplicate fingerprints (§6.1).

Charikar 2002 rounding; Manku/Jain/Das Sarma 2007 near-duplicate
detection (64-bit fingerprints, Hamming <= 3). These tests pin the
operating properties, cross-check the similarity semantics against a
brute-force token-multiset Jaccard, and verify the bounded-memory
tracker's window eviction and serialization round-trip.
"""

from __future__ import annotations

import unittest

from assistant.capabilities.scan.simhash import (NearDupTracker, hamming, mask_digits,
                                   near_duplicate, simhash)


def _jaccard(a: str, b: str) -> float:
    """Brute-force token-multiset similarity for the cross-check."""
    ta, tb = a.lower().split(), b.lower().split()
    if not ta or not tb:
        return 0.0
    shared = sum(1 for w in set(ta) & set(tb))
    return shared / len(set(ta) | set(tb))


class SimHashTests(unittest.TestCase):

    def test_deterministic(self):
        a = "quickshell encountered an error at module load"
        self.assertEqual(simhash(a), simhash(a))
        self.assertEqual(simhash(a, bits=32), simhash(a, bits=32))

    def test_identical_texts_are_distance_zero(self):
        a = "failed to start service caelestia-shell.service"
        self.assertEqual(hamming(simhash(a), simhash(a)), 0)
        self.assertTrue(near_duplicate(simhash(a), simhash(a)))

    def test_small_edit_is_near_duplicate_with_digit_masking(self):
        # the realistic log case: same line, new PID / timestamp / count.
        # Raw short texts have too little feature mass for a small
        # Hamming threshold (measured ~15 bits) — the domain-correct fix
        # is journal-style volatile-field masking, which the tracker
        # applies by default.
        a = "quickshell error: dbus call failed (pid 1234, attempt 1)"
        b = "quickshell error: dbus call failed (pid 9183, attempt 2)"
        self.assertEqual(hamming(simhash(a, mask_digits=True),
                                  simhash(b, mask_digits=True)), 0)
        # and without masking the distance is honestly large (the
        # operating-point note's own measured number, pinned)
        self.assertGreater(hamming(simhash(a), simhash(b)), 8)

    def test_mask_digits_helper(self):
        self.assertEqual(mask_digits("pid 1234 attempt 1"), "pid # attempt #")
        self.assertEqual(mask_digits("2026-09-26T00:00:00"), "#-#-#T#:#:#")

    def test_unrelated_texts_are_far(self):
        pairs = [
            ("failed to start service caelestia-shell.service",
             "wallpaper changed to material you dark"),
            ("gpu widget averages all drm cards",
             "installing dependencies via pacman"),
            ("kwin effect reloaded successfully",
             "battery level dropped below twenty percent"),
        ]
        for a, b in pairs:
            d = hamming(simhash(a), simhash(b))
            self.assertGreater(
                d, 12, f"{a!r} vs {b!r} unexpectedly close ({d} bits)")

    def test_reworded_connective_text_stays_close(self):
        # same token mass, connective words swapped: still near
        a = "the bar widget crashed after reload"
        b = "bar widget crashed after the reload"
        self.assertLessEqual(hamming(simhash(a), simhash(b)), 6)

    def test_distance_orders_like_jaccard(self):
        # cross-check: Hamming distance must order pairs consistently with
        # token-multiset Jaccard (similar pair strictly closer than the
        # dissimilar pair)
        base = "caelestia shell restarted and reloaded the config"
        similar = "caelestia shell restarted and reloaded its config"
        dissimilar = "pacman upgraded the linux kernel headers today"
        d_sim = hamming(simhash(base), simhash(similar))
        d_dis = hamming(simhash(base), simhash(dissimilar))
        j_sim = _jaccard(base, similar)
        j_dis = _jaccard(base, dissimilar)
        self.assertGreater(j_sim, j_dis)
        self.assertLess(d_sim, d_dis)

    def test_empty_and_tiny_texts_fingerprint(self):
        self.assertEqual(simhash(""), 0)
        fp1 = simhash("a")
        fp2 = simhash("ab")
        self.assertIsInstance(fp1, int)
        self.assertNotEqual(fp1, fp2)

    def test_short_nonword_lines_fingerprint_stably_when_masked(self):
        # short ID lines: raw fingerprints stay deterministic; with digit
        # masking, same-skeleton IDs collapse to the identical fingerprint
        self.assertEqual(simhash("ERROR-1001"), simhash("ERROR-1001"))
        self.assertEqual(hamming(simhash("ERROR-1001", mask_digits=True),
                                 simhash("ERROR-1002", mask_digits=True)), 0)

    def test_bits_validation(self):
        with self.assertRaises(ValueError):
            simhash("text", bits=4)
        with self.assertRaises(ValueError):
            simhash("text", bits=65)


class NearDupTrackerTests(unittest.TestCase):

    def test_counts_near_duplicates_over_stream(self):
        tracker = NearDupTracker(window=10, threshold=6)
        d1 = tracker.observe("quickshell error: dbus call failed (pid 1234)")
        self.assertIsNone(d1)
        d2 = tracker.observe("quickshell error: dbus call failed (pid 9183)")
        self.assertIsNotNone(d2)
        self.assertLessEqual(d2, 6)
        self.assertEqual(tracker.seen, 2)
        self.assertEqual(tracker.near_dups, 1)

    def test_window_eviction_is_real(self):
        # with a window of 3, the first line's fingerprint must be evicted
        # after three more observations — bounded memory means actually
        # forgetting, not silently keeping everything.
        tracker = NearDupTracker(window=3, threshold=6)
        first = "quickshell error: dbus call failed (pid 1234)"
        tracker.observe(first)
        for i in range(4):  # four distinct unrelated lines evict it
            tracker.observe(f"unrelated log line number {i} about kernels")
        self.assertEqual(len(tracker._ring), 3)
        # the first line is now new again: no near-duplicate reported
        self.assertIsNone(tracker.observe(first + " (again)"))

    def test_serialization_round_trip(self):
        tracker = NearDupTracker(window=10, threshold=6)
        tracker.observe("caelestia shell restarted and reloaded the config")
        tracker.observe("caelestia shell restarted and reloaded its config")
        tracker.observe("caelestia shell restarted and reloaded the "
                        "config 42 times")
        saved = tracker.to_dict()
        restored = NearDupTracker.from_dict(saved)
        self.assertEqual(restored.seen, tracker.seen)
        self.assertEqual(restored.near_dups, tracker.near_dups)
        self.assertEqual(list(restored._ring), list(tracker._ring))
        self.assertTrue(restored.mask_digits)
        # and the restored tracker keeps answering correctly: the same
        # line with only its volatile count changed is a near-duplicate,
        # because digit masking is part of the restored state
        d = restored.observe("caelestia shell restarted and reloaded "
                             "the config 7 times")
        self.assertIsNotNone(d)
        self.assertLessEqual(d, 3)

    def test_window_validation(self):
        with self.assertRaises(ValueError):
            NearDupTracker(window=0)


if __name__ == "__main__":
    unittest.main()

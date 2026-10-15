"""F22 accessibility property tests: WCAG 2.x contrast math and the
Machado (2009) CVD simulation. The properties are the honesty guard —
identity at severity 0, exact published matrix at severity 1, neutral-gray
preservation at every severity (the published matrices' row sums are 1),
the WCAG corner cases (white/black = 21:1), and sRGB round-trips."""

from __future__ import annotations

import unittest

from assistant.capabilities.settings.accessibility import (
    CVD_TYPES, _MACHADO, contrast_ratio, hex_to_rgb, relative_luminance,
    rgb_to_hex, simulate_cvd, wcag_findings,
)

WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
GRAY = (0.5, 0.5, 0.5)


class HexTests(unittest.TestCase):
    def test_roundtrip(self):
        for hexes in ("#ff0000", "#00ff7f", "#1a2b3c", "#fff", "#000"):
            rgb = hex_to_rgb(hexes)
            back = rgb_to_hex(rgb)
            self.assertIn(back, (hexes, rgb_to_hex(hex_to_rgb(back))))
        with self.assertRaises(ValueError):
            hex_to_rgb("#12345")
        with self.assertRaises(ValueError):
            hex_to_rgb("zzzzzz")


class WcagTests(unittest.TestCase):
    def test_corner_cases(self):
        self.assertAlmostEqual(contrast_ratio(WHITE, BLACK), 21.0, places=10)
        self.assertAlmostEqual(contrast_ratio(BLACK, WHITE), 21.0, places=10)
        self.assertAlmostEqual(contrast_ratio(GRAY, GRAY), 1.0, places=10)
        self.assertAlmostEqual(relative_luminance(WHITE), 1.0, places=10)
        self.assertAlmostEqual(relative_luminance(BLACK), 0.0, places=10)

    def test_findings_shape(self):
        f = wcag_findings(WHITE, BLACK)
        self.assertTrue(all(x["pass"] for x in f))
        f2 = wcag_findings((0.75, 0.75, 0.75), WHITE)   # ~1.8:1 — fails all
        self.assertTrue(all(not x["pass"] for x in f2))
        for x in f2:
            self.assertIn("SC 1.4", x["cite"])


class CvdTests(unittest.TestCase):
    def test_identity_at_severity_zero(self):
        for kind in CVD_TYPES:
            for rgb in ((0.2, 0.5, 0.8), WHITE, (0.9, 0.1, 0.4)):
                sim = simulate_cvd(rgb, kind, severity=0.0)
                for a, b in zip(sim, rgb):
                    self.assertAlmostEqual(a, b, places=12)

    def test_full_severity_is_the_published_matrix(self):
        for kind in CVD_TYPES:
            mat = _MACHADO[kind]
            lin = tuple(c / 255.0 for c in (61, 127, 193))  # linearize-free probe:
            # severity 1 must equal applying the published matrix in linear
            # space; compare against an explicit multiply
            probe = tuple(1.0 / 255.0 * v for v in (255, 0, 0))
            from assistant.capabilities.settings.accessibility import _linearize
            l = [_linearize(c) for c in probe]
            expect = tuple(
                min(1.0, max(0.0, sum(mat[i][j] * l[j] for j in range(3))))
                for i in range(3))
            sim = simulate_cvd(probe, kind, severity=1.0)
            from assistant.capabilities.settings.accessibility import _delinearize
            for a, b in zip(sim, (_delinearize(c) for c in expect)):
                self.assertAlmostEqual(a, b, places=12)

    def test_neutral_gray_preserved_at_all_severities(self):
        # the published matrices' rows sum to 1, and a linear blend of two
        # row-stochastic matrices is row-stochastic: grays map to grays
        for kind in CVD_TYPES:
            for sev in (0.0, 0.25, 0.5, 0.75, 1.0):
                sim = simulate_cvd(GRAY, kind, severity=sev)
                for c in sim:
                    self.assertAlmostEqual(c, 0.5, delta=1e-3, msg=(kind, sev))

    def test_severity_endpoints_and_bounds(self):
        with self.assertRaises(ValueError):
            simulate_cvd(GRAY, "protanopia", severity=-0.1)
        with self.assertRaises(ValueError):
            simulate_cvd(GRAY, "protanopia", severity=1.5)
        with self.assertRaises(ValueError):
            simulate_cvd(GRAY, "not-a-kind")

    def test_published_matrix_rows_sum_to_one(self):
        # guards the transcribed constants themselves
        for kind in CVD_TYPES:
            for row in _MACHADO[kind]:
                self.assertAlmostEqual(sum(row), 1.0, delta=2e-6, msg=kind)


if __name__ == "__main__":
    unittest.main()

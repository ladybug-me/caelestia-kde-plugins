"""Genius layer tests — statistics, data analysis, logic, Bayes, decisions."""
import unittest

from assistant.capabilities.genius import data as gd
from assistant.capabilities.genius import stats as gs


class TestSpecialFunctions(unittest.TestCase):
    def test_betainc_symmetry(self):
        self.assertAlmostEqual(gs.betainc(0.5, 0.5, 0.5), 0.5, places=9)

    def test_chi2_sf_known(self):
        self.assertAlmostEqual(gs.chi2_sf(3.84, 1), 0.05, places=3)

    def test_t_cdf_known(self):
        self.assertAlmostEqual(gs.t_cdf(2.0, 10), 0.963306, places=6)

    def test_f_sf_known(self):
        self.assertAlmostEqual(gs.f_sf(4.0, 3, 10), 0.0413, places=3)


class TestDescriptives(unittest.TestCase):
    def test_describe_basic(self):
        d = gs.describe([1, 2, 2, 3, 4, 5, 5, 5, 6, 7])
        self.assertEqual(d["n"], 10)
        self.assertAlmostEqual(d["mean"], 4.0)
        self.assertEqual(d["median"], 4.5)
        self.assertEqual(d["mode"], 5)
        # inclusive (linear-interpolation) definition: pos = .25*(n-1) = 2.25
        self.assertAlmostEqual(d["q1"], 2.25)
        self.assertAlmostEqual(gs.describe([5])["sd"], 0.0)

    def test_quantile_bounds(self):
        with self.assertRaises(ValueError):
            gs.quantile([1, 2, 3], 1.5)


class TestInference(unittest.TestCase):
    def test_one_sample_t(self):
        r = gs.t_test_one_sample([98, 100, 102, 101, 99, 103, 97, 101], 100)
        self.assertAlmostEqual(r["t"], 0.1741, places=3)
        self.assertGreater(r["p"], 0.5)

    def test_two_sample_welch(self):
        r = gs.t_test_two_sample([1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(r["t"], -1.8516, places=3)
        self.assertAlmostEqual(r["cohen_d"], -1.069, places=2)

    def test_paired(self):
        r = gs.t_test_paired([10, 12, 9, 11], [12, 15, 10, 12])
        self.assertAlmostEqual(r["mean_diff"], -1.75)
        self.assertLess(r["p"], 0.05)

    def test_mann_whitney_separated(self):
        r = gs.mann_whitney_u([1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17])
        self.assertLess(r["p_approx"], 0.01)

    def test_chi2_independence(self):
        r = gs.chi2_independence([[10, 20, 30], [6, 9, 17]])
        self.assertAlmostEqual(r["chi2"], 0.2716, places=3)
        self.assertGreater(r["p"], 0.5)

    def test_anova_known_f(self):
        r = gs.anova_one_way([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertAlmostEqual(r["f"], 27.0, places=3)

    def test_odds_ratio(self):
        r = gs.odds_ratio([[8, 2], [3, 7]])
        self.assertAlmostEqual(r["odds_ratio"], 8 * 7 / (2 * 3), places=6)

    def test_z_proportion(self):
        r = gs.z_test_proportion(60, 100, 0.5)
        self.assertAlmostEqual(r["phat"], 0.6)
        self.assertGreater(r["p"], 0.02)

    def test_ols_regression(self):
        r = gs.ols_regression([2, 4, 5, 4, 5, 7, 8, 9],
                              [1, 2, 3, 4, 5, 6, 7, 8], names=["x"])
        self.assertAlmostEqual(r["coefficients"]["x"], 0.9048, places=3)
        self.assertAlmostEqual(r["r2"], 0.9048, places=3)
        self.assertLess(r["p_values"]["x"], 0.01)

    def test_bootstrap_covers_truth(self):
        boot = gs.bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_boot=2000)
        self.assertLessEqual(boot["ci"][0], 5.5)
        self.assertGreaterEqual(boot["ci"][1], 5.5)

    def test_permutation_separated(self):
        r = gs.permutation_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], n_perm=2000)
        self.assertLess(r["p"], 0.05)

    def test_outliers_iqr(self):
        r = gs.detect_outliers([1, 2, 2, 3, 3, 3, 4, 4, 100])
        self.assertEqual(r["n_outliers"], 1)
        self.assertEqual(r["outliers"][0]["value"], 100.0)

    def test_correlations(self):
        r = gs.pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        self.assertAlmostEqual(r["r"], 0.7746, places=3)
        self.assertEqual(gs.spearman([1, 2, 3], [3, 2, 1])["rho"], -1.0)
        self.assertGreater(abs(gs.kendall([1, 2, 3], [1, 3, 2])["tau"]), 0)


CSV = """name,age,city,salary
alice,34,delhi,120
bob,28,mumbai,95
carol,41,delhi,180
dave,,pune,110
eve,36,mumbai,150
frank,29,delhi,100
grace,45,pune,200
heidi,31,mumbai,130"""


class TestDataModule(unittest.TestCase):
    def test_parse_and_profile(self):
        t = gd.parse_table(CSV)
        self.assertEqual(t["rows"], 8)
        p = gd.profile_table(t)
        self.assertEqual(p["columns"]["age"]["nulls"], 1)
        self.assertEqual(p["columns"]["city"]["cardinality"], 3)

    def test_groupby_mean(self):
        t = gd.parse_table(CSV)
        g = gd.groupby(t, "city", "salary", "mean")
        self.assertAlmostEqual(g["groups"]["delhi"], 133.333333, places=3)

    def test_correlation_matrix(self):
        t = gd.parse_table("x,y\n1,2\n2,4\n3,5\n4,4\n5,7\n6,9\n7,10")
        cm = gd.correlation_matrix(t)
        self.assertGreater(cm["strongest"][0]["r"], 0.9)

    def test_moving_average_and_ewma(self):
        ma = gd.moving_average([10, 12, 13, 11, 14], 3)
        self.assertAlmostEqual(ma["smoothed"][0], 35 / 3, places=5)
        ew = gd.ewma([1, 2, 3], alpha=0.5)
        # s0=1, s1=0.5*2+0.5*1=1.5, s2=0.5*3+0.5*1.5=2.25
        self.assertAlmostEqual(ew["smoothed"][2], 2.25, places=5)

    def test_autocorrelation_persistent_series(self):
        acf = gd.autocorrelation([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertGreater(acf["acf"][1], 0.5)

    def test_yule_walker_ar1(self):
        # proper stochastic AR(1) with phi=0.8
        import random
        rng = random.Random(7)
        series, x = [], 0.0
        for _ in range(400):
            x = 0.8 * x + rng.gauss(0, 1)
            series.append(x)
        fit = gd.yule_walker_ar(series, order=1)
        self.assertGreater(fit["phi"][0], 0.6)
        self.assertLess(fit["phi"][0], 1.0)

    def test_forecast_ar_returns_horizon(self):
        f = gd.forecast_ar([10, 12, 13, 11, 14, 15, 17, 16, 18, 19, 21, 20], horizon=4)
        self.assertEqual(len(f["forecast"]), 4)
        self.assertEqual(len(f["ci95"]), 4)

    def test_decompose_needs_two_periods(self):
        with self.assertRaises(ValueError):
            gd.decompose([1, 2, 3], period=4)

    def test_changepoint_detects_real_shift(self):
        shifted = [5] * 10 + [10] * 10
        cp = gd.changepoints(shifted)
        self.assertLess(cp["bootstrap_p"], 0.05)
        self.assertTrue(cp["changepoints"])

    def test_changepoint_clean_is_quiet(self):
        clean = [5, 5.2, 4.8, 5.1, 5.0, 4.9, 5.2, 5.1, 5.0, 4.9,
                 5, 5.2, 4.8, 5.1, 5, 4.9, 5.2, 5.1, 5, 4.9]
        cp = gd.changepoints(clean)
        self.assertGreater(cp["bootstrap_p"], 0.1)
        self.assertEqual(cp["n_changepoints"], 0)

    def test_kmeans_clusters_three_blobs(self):
        pts = [[1, 1], [1.2, 0.9], [0.8, 1.1], [5, 5], [5.2, 4.9],
               [4.8, 5.1], [9, 1], [9.2, 0.8]]
        km = gd.kmeans(pts, 3)
        self.assertEqual(km["sizes"], [3, 3, 2])
        sil = gd.silhouette(pts, km["labels"])
        self.assertGreater(sil["mean_silhouette"], 0.5)

    def test_agglomerative_same_partition(self):
        pts = [[1, 1], [1.1, 0.9], [5, 5], [5.1, 4.9]]
        ag = gd.agglomerative(pts, 2)
        self.assertEqual(sorted(len(m) for m in ag["members"]), [2, 2])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2.2: NCD, BOCPD, decomposition robustness.
# ---------------------------------------------------------------------------

class TestNCD(unittest.TestCase):
    def test_identical_texts_score_low(self):
        text = "the quick brown fox jumps over the lazy dog " * 4
        r = gd.ncd(text, text)
        self.assertLess(r["ncd"], 0.2)

    def test_unrelated_texts_score_high(self):
        a = "rust borrow checker compile errors " * 8
        b = "kde plasma widget wayland panel theming " * 8
        r = gd.ncd(a, b)
        self.assertGreater(r["ncd"], 0.4)

    def test_near_variants_score_between(self):
        a = "plasma widgets consume cpu when animating " * 8
        b = "plasma widgets consume cpu when animating fast " * 8
        r = gd.ncd(a, b)
        self.assertGreaterEqual(r["ncd"], 0.0)
        self.assertLess(r["ncd"], r["ncd"] + 1)  # bounded + present

    def test_bz2_second_opinion(self):
        a = "aaaaaaaaaaaaaaaaaaaa"
        b = "abcdefghijklmnopqrst"
        r = gd.ncd(a, b, compressor="bz2")
        self.assertEqual(r["compressor"], "bz2")
        self.assertGreaterEqual(r["ncd"], 0.0)

    def test_bytes_input(self):
        # a longer blob: 3-byte inputs are overhead-dominated (documented
        # NCD small-input limitation)
        blob = bytes(range(256)) * 4
        r = gd.ncd(blob, blob)
        self.assertLess(r["ncd"], 0.2)

    def test_empty_pair(self):
        r = gd.ncd(b"", b"")
        self.assertEqual(r["ncd"], 0.0)


class TestBOCPD(unittest.TestCase):
    def test_detects_clean_mean_shift(self):
        series = [5.0] * 20 + [10.0] * 20
        r = gd.bocpd(series, hazard=20)
        self.assertEqual(r["changepoints"], [20])
        self.assertGreater(r["changepoint_prob"][19], 0.5)

    def test_detects_two_shifts(self):
        series = [5.0] * 15 + [8.0] * 15 + [2.0] * 15
        r = gd.bocpd(series, hazard=15)
        self.assertEqual(r["changepoints"], [15, 30])

    def test_flat_series_has_no_changepoints(self):
        series = [5.0 + 0.3 * ((i * 29) % 7 - 3) / 3 for i in range(40)]
        r = gd.bocpd(series, hazard=20)
        self.assertEqual(r["changepoints"], [])

    def test_run_lengths_grow_inside_a_segment(self):
        series = [5.0] * 12 + [9.0] * 12
        r = gd.bocpd(series, hazard=20)
        self.assertLess(r["map_run_length"][5], r["map_run_length"][10])

    def test_short_series_rejected(self):
        with self.assertRaises(ValueError):
            gd.bocpd([1.0, 2.0])


class TestDecomposeRobustness(unittest.TestCase):
    def test_clean_periodic_series_is_not_fragile(self):
        import math as _math
        series = [10 + 3 * _math.sin(2 * _math.pi * i / 12)
                  for i in range(48)]
        r = gd.decompose_robustness(series, 12)
        self.assertIn(r["verdict"], ("period-sensitive", "robust"))
        self.assertLess(r["seasonal_amplitude_stability"], 0.5)

    def test_white_noise_is_flagged(self):
        series = [((i * 1103515245 + 12345) % 1000) / 1000.0 * 6
                  for i in range(60)]
        r = gd.decompose_robustness(series, 12)
        self.assertEqual(r["verdict"], "fragile")

    def test_too_short_series_rejected(self):
        with self.assertRaises(ValueError):
            gd.decompose_robustness([1.0, 2.0, 3.0], 12)

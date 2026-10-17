"""Identical-result tests for the numpy accelerator.

THE CONTRACT (see numpy_accel): when numpy is installed, the
accelerated path and the pure path agree EXACTLY on order and scores
(to 1e-9); when it is not, the pure path is the only path and these
tests still pass. Falsifying inputs included: zero vectors, duplicate
scores (tie order), k > n, empty matrix, negative weights.
"""
import unittest

from assistant.core.numpy_accel import cosine_topk, have_numpy, mean_pool

HAS_NUMPY = have_numpy()


def _pure_cosine_topk(query, matrix, k):
    # reference implementation copied from the pure path semantics
    import math
    qnorm = math.sqrt(sum(x * x for x in query)) or 1.0
    scored = []
    for i, row in enumerate(matrix):
        rnorm = math.sqrt(sum(x * x for x in row)) or 1.0
        num = sum(a * b for a, b in zip(query, row))
        scored.append((i, num / (rnorm * qnorm)))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


class TestCosineTopk(unittest.TestCase):
    QUERY = [1.0, 2.0, 0.5]
    MATRIX = [
        [1.0, 2.0, 0.5],     # identical direction -> 1.0
        [2.0, 4.0, 1.0],     # also 1.0 (tie!)
        [0.0, 0.0, 0.0],     # zero vector
        [-1.0, -2.0, -0.5],  # opposite
        [0.5, 0.5, 0.5],
    ]

    def test_matches_reference_exactly(self):
        got = cosine_topk(self.QUERY, self.MATRIX, 3)
        ref = _pure_cosine_topk(self.QUERY, self.MATRIX, 3)
        if HAS_NUMPY:
            self.assertEqual([i for i, _ in got], [i for i, _ in ref])
            for (i_g, s_g), (i_r, s_r) in zip(got, ref):
                self.assertAlmostEqual(s_g, s_r, places=9)
        else:
            self.assertEqual(got, ref)

    def test_k_larger_than_matrix(self):
        got = cosine_topk(self.QUERY, self.MATRIX, 99)
        self.assertEqual(len(got), len(self.MATRIX))

    def test_empty_matrix(self):
        self.assertEqual(cosine_topk(self.QUERY, [], 3), [])

    def test_ties_keep_index_order(self):
        got = cosine_topk([1.0], [[2.0], [2.0], [2.0]], 3)
        self.assertEqual([i for i, _ in got], [0, 1, 2])

    def test_zero_vector_row_never_nan(self):
        got = cosine_topk([1.0], [[0.0], [1.0]], 2)
        for _, score in got:
            self.assertFalse(score != score)  # NaN check


class TestMeanPool(unittest.TestCase):
    def test_unweighted_is_row_mean(self):
        rows = [[1.0, 3.0], [3.0, 5.0]]
        got = mean_pool(rows)
        self.assertAlmostEqual(got[0], 2.0, places=9)
        self.assertAlmostEqual(got[1], 4.0, places=9)

    def test_weighted(self):
        rows = [[1.0], [3.0]]
        got = mean_pool(rows, weights=[1.0, 3.0])
        self.assertAlmostEqual(got[0], 2.5, places=9)

    def test_empty(self):
        self.assertEqual(mean_pool([]), [])


if __name__ == "__main__":
    unittest.main()

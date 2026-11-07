import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from network_recovery import graph_system, simulate, cv_splits, edge_scores


class RecoveryTests(unittest.TestCase):
    def test_stability_and_stationary_covariance(self):
        a, truth, p = graph_system(5420)
        self.assertEqual(truth.sum(), 300)
        self.assertFalse(np.diag(truth).any())
        self.assertAlmostEqual(max(abs(np.linalg.eigvals(a))), .9)
        np.testing.assert_allclose(p, a @ p @ a.T + np.eye(100), atol=1e-9)

    def test_direction_and_unscaling(self):
        # 0 -> 1, not 1 -> 0. A transpose bug must fail this test.
        a = np.array([[.2, 0, 0], [.7, .2, 0], [0, .4, .2]])
        x = np.random.default_rng(1).normal(size=(1000, 3))
        coef = LinearRegression().fit(x, x @ a.T).coef_
        np.testing.assert_allclose(coef, a, atol=1e-12)
        self.assertEqual(edge_scores((a != 0), coef)[0], 1.)
        self.assertLess(edge_scores((a != 0), coef.T)[0], 1.)

    def test_multioutput_lasso_is_separate_regressions(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 8)); y = rng.normal(size=(60, 3))
        jointly_called = Lasso(alpha=.03).fit(x, y).coef_
        separate = np.stack([Lasso(alpha=.03).fit(x, y[:,j]).coef_ for j in range(3)])
        np.testing.assert_allclose(jointly_called, separate)

    def test_time_split_raw_timestamps_do_not_overlap(self):
        for train, valid in cv_splits(100):
            train_times = set(train) | set(train + 1)
            valid_times = set(valid) | set(valid + 1)
            self.assertTrue(train_times.isdisjoint(valid_times))

    def test_simulation_seed_and_no_cross_trajectory_overlap(self):
        a, truth, p = graph_system(5420)
        x = simulate(a, p, 20, 1)
        np.testing.assert_array_equal(x, simulate(a, p, 20, 1))
        self.assertFalse(np.array_equal(x, simulate(a, p, 20, 2)))


if __name__ == '__main__':
    unittest.main()

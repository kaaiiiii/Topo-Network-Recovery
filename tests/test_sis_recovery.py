import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sis_recovery import sis_step, simulate_sis, transitions, episode_folds, fit_path, checked_fit
from network_recovery import graph_system


class SISTests(unittest.TestCase):
    def test_directed_infection_and_disease_free_state(self):
        b = np.array([[0., 0.], [1., 0.]])  # 0 -> 1
        np.testing.assert_array_equal(sis_step(np.zeros(2), b, 1), [0, 0])
        np.testing.assert_allclose(sis_step(np.array([.5, 0]), b, 1), [.475, .025])
        np.testing.assert_allclose(sis_step(np.array([0., .5]), b, 1), [0., .475])

    def test_no_transmission_is_pure_recovery(self):
        initial = np.array([[.2, .4]])
        states = simulate_sis(np.ones((2, 2)), initial, beta=0, steps=10)
        np.testing.assert_allclose(states[0], .95**np.arange(11)[:, None] * initial)

    def test_probability_bounds_and_reject_unsafe_step(self):
        for seed in range(1001, 1011):
            a, _, _ = graph_system(seed)
            b = (a - .3*np.eye(100))/.6
            initial = np.random.default_rng(seed).uniform(0, 1, size=(2, 100))
            states = simulate_sis(b, initial, 3., steps=50)
            self.assertGreaterEqual(states.min(), 0)
            self.assertLessEqual(states.max(), 1)
        with self.assertRaises(ValueError):
            simulate_sis(np.ones((2, 2)), np.ones((1, 2)), 3., dt=1.)

    def test_nonlinear_correction_scales_quadratically(self):
        b = np.array([[0., .8], [1., 0.]])
        p = np.array([.2, .3])
        linear = np.eye(2) + .05 * (1.5*b - np.eye(2))
        correction = sis_step(p, b, 1.5) - p @ linear.T
        half = sis_step(p/2, b, 1.5) - (p/2) @ linear.T
        np.testing.assert_allclose(half, correction/4, atol=1e-15)

    def test_episode_boundaries_and_cv(self):
        episodes = np.arange(20*4*2).reshape(20, 4, 2)
        x, y, groups = transitions(episodes)
        self.assertEqual(len(x), 60)  # not 79 transitions through concatenated episodes
        np.testing.assert_array_equal(y-x, np.full_like(x, 2))
        for train, valid in episode_folds(groups):
            self.assertTrue(set(groups[train]).isdisjoint(groups[valid]))
            self.assertEqual(len(set(groups[valid])), 4)

    def test_lars_path_matches_independent_scaled_regressions(self):
        rng = np.random.default_rng(73)
        x = rng.normal(size=(120, 4))*np.array([1., 2., 3., 4.])+2
        y = x@np.array([[.7, 0, .2, 0], [0, .3, 0, .8]]).T+rng.normal(size=(120, 2))*.02
        for method in ['lasso', 'ridge']:
            path = fit_path(x, y, method, [.001, .01, .1])
            for alpha, fitted in path.items():
                independent, count = checked_fit(x, y, method, alpha)
                self.assertEqual(count, 0)
                np.testing.assert_allclose(x@fitted['a_hat'].T+fitted['intercept'],
                                           independent.predict(x), atol=1e-5)


if __name__ == '__main__':
    unittest.main()

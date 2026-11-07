import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from sis_local_poly import regression_samples, checked_path
from sis_recovery import episode_folds, transitions


class LocalPolynomialTests(unittest.TestCase):
    def test_qp_refinement_matches_closed_form(self):
        design = 2*np.eye(4)
        target = np.array([1., -.4, 2., 0.])
        bad_path = SimpleNamespace(alphas_=np.array([1., 0.]), coef_path_=np.zeros((4,2)))
        with patch('sis_local_poly.LassoLars') as lars:
            lars.return_value.fit.return_value = bad_path
            path, info = checked_path(design, target, [0., .1, 1.])
        sx = np.sqrt(np.mean(design**2,axis=0))
        sy = np.sqrt(np.mean(target**2))
        correlations = (design/sx).T@(target/sy)/len(target)
        for fraction, coef in path.items():
            expected = np.maximum(correlations-fraction*correlations.max(),0)*sy/sx
            np.testing.assert_allclose(coef, expected, atol=1e-9)
        self.assertGreater(info['polished_candidates'], 0)
        self.assertLess(info['max_kkt_residual'], 1e-7)

    def test_exact_quadratic_states_and_derivatives(self):
        dt = .05
        t = np.arange(31)*dt
        values = np.stack([.2+.1*t+.3*t*t, .4-.2*t+.05*t*t], axis=-1)[None]
        for window in [5, 9, 15, 21]:
            r = window//2
            x, y = regression_samples(values, window, dt)
            np.testing.assert_allclose(x, values[0, r:-r], atol=1e-13)
            expected = np.stack([.1+.6*t[r:-r], -.2+.1*t[r:-r]], axis=-1)
            np.testing.assert_allclose((y-x)/dt, expected, atol=1e-12)

    def test_derivative_matches_paper_symmetric_difference_weights(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(3, 31, 4))
        for window in [5, 9, 15, 21]:
            radius = window//2
            expected = np.zeros((3, 31-2*radius, 4))
            for offset in range(1, radius+1):
                weight = 6*offset**2/(radius*(radius+1)*(2*radius+1))
                centers = np.arange(radius, 31-radius)
                expected += weight*(values[:, centers+offset]-values[:, centers-offset])/(2*offset*.05)
            x, y = regression_samples(values, window)
            np.testing.assert_allclose((y-x)/.05, expected.reshape(x.shape), atol=1e-12)

    def test_episodes_and_validation_stay_separate(self):
        values = np.random.default_rng(8).normal(size=(20, 51, 3))
        train, valid = episode_folds(np.arange(20))[0]
        x, y = regression_samples(values[train], 9)
        changed = values.copy()
        changed[valid] += 10000
        xx, yy = regression_samples(changed[train], 9)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(y, yy)
        separate = [regression_samples(values[i:i+1], 9)[0] for i in train]
        np.testing.assert_allclose(x, np.concatenate(separate))
        groups = transitions(values)[2]
        original_train, original_valid = episode_folds(groups)[0]
        np.testing.assert_array_equal(np.sort(train), np.unique(groups[original_train]))
        np.testing.assert_array_equal(np.sort(valid), np.unique(groups[original_valid]))

    def test_zero_window_preserves_original_transitions(self):
        values = np.random.default_rng(17).normal(size=(2, 11, 3))
        for actual, expected in zip(regression_samples(values, 0), transitions(values)[:2]):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

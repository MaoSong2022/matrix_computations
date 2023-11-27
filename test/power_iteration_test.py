import unittest
import numpy as np

from power_iteration import (
    power_iteration,
    rayleigh_quotient_iteration,
    orthogonal_iteration,
)


class TestPowerIterations(unittest.TestCase):
    def setUp(self):
        n = 5
        eig_vectors = np.random.randn(n, n)
        eig_vectors /= np.linalg.norm(eig_vectors, axis=0)
        eig_values = np.sort(np.random.rand(n))
        self.A = eig_vectors @ np.diag(eig_values) @ np.linalg.inv(eig_vectors)
        self.real_lamb = eig_values[-1]
        self.real_eigenvector = eig_vectors[:, -1]

    def test_power_iterations(self):
        test_lamb, test_eigenvector = power_iteration(self.A, max_iterations=1000)
        test_eigenvector /= np.linalg.norm(test_eigenvector)
        self.assertAlmostEqual(self.real_lamb, test_lamb, places=3)
        try:
            np.testing.assert_array_almost_equal(
                self.real_eigenvector, test_eigenvector
            )
        except AssertionError:
            np.testing.assert_array_almost_equal(
                self.real_eigenvector + test_eigenvector,
                np.zeros_like(test_eigenvector),
            )

    def test_rayleigh_quotient_iterations(self):
        # remark: the result depends on the initialization.
        v = self.real_eigenvector
        test_lamb, test_eigenvector = rayleigh_quotient_iteration(
            self.A, max_iterations=1000, v=v
        )
        self.assertAlmostEqual(self.real_lamb, test_lamb, places=3)
        try:
            np.testing.assert_array_almost_equal(
                self.real_eigenvector, test_eigenvector
            )
        except AssertionError:
            np.testing.assert_array_almost_equal(
                self.real_eigenvector + test_eigenvector,
                np.zeros_like(test_eigenvector),
            )

    def test_orthogonal_iterations(self):
        Q, sigma = orthogonal_iteration(self.A, k=5, max_iterations=1000)
        np.testing.assert_array_almost_equal(self.A, Q @ sigma @ Q.T, decimal=3)


if __name__ == "__main__":
    # print the numpy array in float format
    np.set_printoptions(suppress=True)
    unittest.main()

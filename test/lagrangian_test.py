import unittest
import numpy as np

from interpolation.lagrangian import lagrangian_interpolation


class TestLUDecomposition(unittest.TestCase):
    """
    Reference:
        Burden, Richard L., and J. Douglas Faires. Numerical Analysis. 9th ed.
        Section 3.1, Example 2
    """

    def setUp(self):
        self.data = np.array([[x, 1.0 / x] for x in [2.0, 2.75, 4.0]])
        self.x = np.array([3.0])

    def test_lagrangian(self):
        test_y = lagrangian_interpolation(self.data, self.x)
        np.testing.assert_array_almost_equal(test_y, 0.32955, decimal=3)


if __name__ == "__main__":
    unittest.main()

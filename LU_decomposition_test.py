import scipy.linalg as la
import unittest
import numpy as np

from .LU_decomposition import out_product_lu

class TestLUDecomposition(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randint(100, size=(5, 5))
        print(self.A)

    def test1(self):
        test_L, test_U = out_product_lu(self.A)
        print(self.A)
        print(test_L @ test_U)
        self.assertTrue(np.allclose(test_L @ test_U, self.A))

    def test2(self):
        P, real_L, real_U = la.lu(self.A)
        test_L, test_U = out_product_lu(np.dot(P.T, self.A))

        print(test_L)
        print(real_L)
        print(test_U)
        print(real_U)
        self.assertTrue(np.allclose(real_L, test_L))
        self.assertTrue(np.allclose(real_U, test_U))



if __name__ == '__main__':
    unittest.main()
import scipy.linalg as la
import unittest
import numpy as np

from LU_decomposition import out_product_lu, gaussian_lu

class TestLUDecomposition(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[ -50.81816222,   17.50242678,  -18.5187156,   104.48252865,  -35.6882816 ],
                           [  31.31130212,  -80.91854366, -34.54258943, -127.90181121,    1.88355938],
                           [ -15.0798181,   248.42511054,   77.33964127,   41.61896562,   10.75753459],
                           [-114.65834707,    0.84312809,   22.4641862,   -53.75328591, -152.18389341],
                           [ -71.75615361,  -80.83956885, -166.6450018,   142.40901914,   74.17852845]])

    def test_out_product1(self):
        B = self.A.copy()
        test_L, test_U = out_product_lu(B)
        np.testing.assert_array_almost_equal(test_L @ test_U, self.A, decimal=3)

    def test_out_product2(self):
        B = self.A.copy()
        P, real_L, real_U = la.lu(B)
        test_L, test_U = out_product_lu(np.dot(P.T, B))
        np.testing.assert_array_almost_equal(real_L, test_L, decimal=3)
        np.testing.assert_array_almost_equal(real_U, test_U, decimal=3)


    def test_gaussian1(self):
        B = self.A.copy()
        test_L, test_U = gaussian_lu(B)
        np.testing.assert_array_almost_equal(test_L @ test_U, self.A, decimal=3)

    def test_gaussian2(self):
        B = self.A.copy()
        P, real_L, real_U = la.lu(self.A)
        test_L, test_U = gaussian_lu(np.dot(P.T, B))
        np.testing.assert_array_almost_equal(real_L, test_L, decimal=3)
        np.testing.assert_array_almost_equal(real_U, test_U, decimal=3)

if __name__ == '__main__':
    unittest.main()

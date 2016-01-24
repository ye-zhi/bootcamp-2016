""" Test fit function
"""

import numpy as np

from ordlstsq.model import fit

psychopathy = np.array([11.416,   4.514,  12.204,  14.835,
                        8.416,   6.563,  17.343, 13.02,
                        15.19 ,  11.902,  22.721,  22.324])

clammy = np.array([0.389,  0.2  ,  0.241,  0.463,
                   4.585,  1.097,  1.642,  4.972,
                   7.957,  5.585,  5.527,  6.964])

known_b = np.array([ 10.071286,   0.999257])

def test_fit():
    N = len(psychopathy)
    X = np.ones((N, 2))
    X[:, 1] = clammy
    B = fit(psychopathy, X)
    np.testing.assert_allclose(B, known_b, 5)

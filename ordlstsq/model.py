""" Modeling ordinary least squares
"""

import numpy as np
import numpy.linalg as npl


def fit(Y, X):
    """ Do ordinary least squares fit of data Y to design matrix X

    Parameters
    ----------
    Y : array
        Data vector shape (N,):
    X : array
        Design matrix 2D shape (N, P) where P is the number of parameters.

    Returns
    -------
    B : array
        Coefficient vector shape (P,)
    """
    return npl.pinv(X).dot(Y)

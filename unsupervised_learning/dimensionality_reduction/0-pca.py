#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def pca(X, var=0.95):
    """
    funcion documentada
    """
    U, S, Vt = np.linalg.svd(X)
    lambdas = S**2
    total = np.sum(lambdas)
    variance_ratio = np.cumsum(lambdas) / total

    nd = np.searchsorted(variance_ratio, var) + 1

    W = Vt[:nd].T

    return W

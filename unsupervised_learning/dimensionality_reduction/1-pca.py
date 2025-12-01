#!/usr/bin/env python3
"""
Modulo documentado
"""

import numpy as np


def pca(X, ndim):
    """
    funcion documentada
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered)

    W = Vt[:ndim].T

    T = np.matmul(X_centered, W)

    return T

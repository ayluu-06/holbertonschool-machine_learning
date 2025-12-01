#!/usr/bin/env python3
"""
Modulo documentado
"""

import numpy as np


def pca(X, var=0.95):
    """
    funcion documentada
    """
    cov = np.matmul(X.T, X) / (X.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    total = np.sum(eig_vals)

    cum_var = np.cumsum(eig_vals) / total
    nd = np.searchsorted(cum_var, var) + 1

    W = eig_vecs[:, :nd]
    return W

#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def shuffle_data(X, Y):
    """
    funcion documentada
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]

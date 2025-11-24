#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


def correlation(C):
    """
    funcion documentada
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    denom = np.outer(std, std)
    return C / denom

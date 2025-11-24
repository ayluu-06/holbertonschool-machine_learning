#!/usr/bin/env python3
"""
Module documented
"""

import numpy as np


def definiteness(matrix):
    """
    funcion documentada
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigen = np.linalg.eigvals(matrix)
    except Exception:
        return None

    pos = np.all(eigen > 0)
    semi_pos = np.all(eigen >= 0)
    neg = np.all(eigen < 0)
    semi_neg = np.all(eigen <= 0)

    if pos:
        return "Positive definite"
    if semi_pos:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if semi_neg:
        return "Negative semi-definite"

    return "Indefinite"

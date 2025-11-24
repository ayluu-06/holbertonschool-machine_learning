#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np


class MultiNormal:
    """
    clase documentada
    """

    def __init__(self, data):
        """
        funcion documentada
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean

        X = data - mean
        cov = (X @ X.T) / (n - 1)
        self.cov = cov

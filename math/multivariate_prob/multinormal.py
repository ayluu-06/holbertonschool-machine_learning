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

    def pdf(self, x):
        """
        funcion documentada
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        norm_const = np.sqrt(((2 * np.pi) ** d) * det)
        exponent = -0.5 * (diff.T @ inv @ diff)

        return np.exp(exponent) / norm_const

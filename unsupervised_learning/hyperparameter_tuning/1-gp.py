#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


class GaussianProcess:
    """
    clase documentada
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        funcion documentada
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        funcion documentada
        """
        sqdist = (X1 - X2.T) ** 2
        return (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))

    def predict(self, X_s):
        """
        funcion documentada
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mu = (K_s.T @ K_inv @ self.Y).reshape(-1)

        cov = K_ss - (K_s.T @ K_inv @ K_s)

        sigma = np.diag(cov)

        return mu, sigma

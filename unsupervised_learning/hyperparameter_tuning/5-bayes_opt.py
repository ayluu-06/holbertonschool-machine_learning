#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    clase documentada
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        funcion documentada
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        funcion documentada
        """
        mu, sigma = self.gp.predict(self.X_s)
        std = np.sqrt(sigma)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        EI = np.zeros_like(mu)

        mask = std > 0
        Z = np.zeros_like(mu)
        Z[mask] = imp[mask] / std[mask]

        EI[mask] = imp[mask] * norm.cdf(Z[mask]) + std[
            mask] * norm.pdf(Z[mask])
        EI = np.maximum(EI, 0)

        X_next = self.X_s[np.argmax(EI)].reshape(1,)

        return X_next, EI

    def optimize(self, iterations=100):
        """
        funcion documentada
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X.reshape(-1), X_next[0])):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)

        return X_opt, Y_opt

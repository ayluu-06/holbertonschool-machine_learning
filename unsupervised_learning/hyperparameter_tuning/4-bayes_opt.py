#!/usr/bin/env python3
"""modulo documentado"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """clase documentada"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """funcion documentada"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """funcion documentada"""
        mu, var = self.gp.predict(self.X_s)
        std = np.sqrt(var)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        EI = np.zeros(mu.shape)

        mask = std > 0
        Z = np.zeros(mu.shape)
        Z[mask] = imp[mask] / std[mask]
        EI[mask] = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)].reshape(1,)
        return X_next, EI

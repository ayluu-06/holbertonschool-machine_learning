#!/usr/bin/env python3
"""modulo documentado"""

import numpy as np


class RNNCell:
    """clase documentada"""

    def __init__(self, i, h, o):
        """funcion documentada"""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """funcion documentada"""
        hx = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(hx @ self.Wh + self.bh)

        z = h_next @ self.Wy + self.by
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y

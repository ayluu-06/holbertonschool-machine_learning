#!/usr/bin/env python3
"""modulo documentado"""

import numpy as np


class GRUCell:
    """clase documentada"""

    def __init__(self, i, h, o):
        """
        funcion documentada
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x):
        """funcion documentada"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(x):
        """funcion documentada"""
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        funcion documentada
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(concat @ self.Wz + self.bz)
        r = self._sigmoid(concat @ self.Wr + self.br)

        concat_hat = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(concat_hat @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_hat

        y = self._softmax(h_next @ self.Wy + self.by)
        return h_next, y

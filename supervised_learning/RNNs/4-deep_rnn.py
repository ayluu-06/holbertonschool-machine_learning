#!/usr/bin/env python3
"""
funcion documentada
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    funcion documentada
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for step in range(t):
        x = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x)

            H[step + 1, layer] = h_next
            x = h_next

            if layer == l - 1:
                Y[step] = y

    return H, Y

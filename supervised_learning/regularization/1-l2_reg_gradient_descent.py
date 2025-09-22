#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    funcion documentada
    """
    m = Y.shape[1]
    AL = cache['A' + str(L)]
    dZ = AL - Y
    for lam in range(L, 0, -1):
        A_prev = cache['A' + str(lam - 1)]
        W = weights['W' + str(lam)]
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(lam)] = W - alpha * dW
        weights['b' + str(lam)] = weights['b' + str(lam)] - alpha * db
        if lam > 1:
            dA_prev = np.matmul(W.T, dZ)
            A_prev = cache['A' + str(lam - 1)]
            dZ = dA_prev * (1 - np.square(A_prev))

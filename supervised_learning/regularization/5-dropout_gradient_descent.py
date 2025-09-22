#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    funcion documentada
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for lam in range(L, 0, -1):
        A_prev = cache['A' + str(lam - 1)]
        W = weights['W' + str(lam)]
        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(lam)] = W - alpha * dW
        weights['b' + str(lam)] = weights['b' + str(lam)] - alpha * db
        if lam > 1:
            dA_prev = W.T @ dZ
            D_prev = cache['D' + str(lam - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob
            A_prev = cache['A' + str(lam - 1)]
            dZ = dA_prev * (1 - np.square(A_prev))

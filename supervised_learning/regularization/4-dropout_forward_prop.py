#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    funcion docuemntada
    """
    cache = {'A0': X}
    for ele in range(1, L):
        Z = np.matmul(weights['W' + str(ele)], cache[
            'A' + str(ele - 1)]) + weights['b' + str(ele)]
        A = np.tanh(Z)
        D = (np.random.rand(*A.shape) < keep_prob).astype(int)
        A = (A * D) / keep_prob
        cache['A' + str(ele)] = A
        cache['D' + str(ele)] = D
    ZL = np.matmul(weights['W' + str(L)], cache[
        'A' + str(L - 1)]) + weights['b' + str(L)]
    expZ = np.exp(ZL - np.max(ZL, axis=0, keepdims=True))
    cache['A' + str(L)] = expZ / np.sum(expZ, axis=0, keepdims=True)
    return cache

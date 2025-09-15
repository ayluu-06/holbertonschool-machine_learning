#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    funcion documentada
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    return gamma * Z_norm + beta

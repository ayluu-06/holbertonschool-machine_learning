#!/usr/bin/env python3
"""
Module documented
"""
import numpy as np


def normalization_constants(X):
    """
    funcion documentada
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

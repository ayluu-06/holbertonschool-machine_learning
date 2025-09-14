#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    funcion documentada
    """
    X_shuf, Y_shuf = shuffle_data(X, Y)

    m = X_shuf.shape[0]
    batches = []
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuf[start:end]
        Y_batch = Y_shuf[start:end]
        batches.append((X_batch, Y_batch))

    return batches

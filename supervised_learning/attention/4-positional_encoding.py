#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    funcion documentada
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)
    angles = position * angle_rates

    pe = np.zeros((max_seq_len, dm))

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe

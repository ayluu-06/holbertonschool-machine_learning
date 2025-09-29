#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    funcion documentada
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            window = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

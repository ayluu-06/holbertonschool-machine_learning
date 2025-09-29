#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    funcion documentada
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h_total = kh - 1
    pad_w_total = kw - 1

    pad_top = (pad_h_total + 1) // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = (pad_w_total + 1) // 2
    pad_right = pad_w_total - pad_left

    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    out = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            window = padded[:, i:i + kh, j:j + kw]
            out[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return out

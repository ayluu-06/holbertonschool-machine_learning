#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    funcion documentada
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            vs = i * sh
            ve = vs + kh
            hs = j * sw
            he = hs + kw
            window = images[:, vs:ve, hs:he, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(window, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return pooled

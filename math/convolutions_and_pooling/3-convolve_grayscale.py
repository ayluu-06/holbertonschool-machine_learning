#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    funcion documentada
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))
        pad_h_total = max((out_h - 1) * sh + kh - h, 0)
        pad_w_total = max((out_w - 1) * sw + kw - w, 0)
        ph = pad_h_total // 2
        pw = pad_w_total // 2
    else:
        raise ValueError(
            "padding must be 'same', 'valid', or a (ph, pw) tuple")

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    H = h + 2 * ph
    W = w + 2 * pw
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            vs = i * sh
            ve = vs + kh
            hs = j * sw
            he = hs + kw
            window = padded[:, vs:ve, hs:he]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

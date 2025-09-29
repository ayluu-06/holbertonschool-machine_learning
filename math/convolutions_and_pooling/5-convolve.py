#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    funcion documentada
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    if kc != c:
        raise ValueError(
            "kernels' channel dimension must match image channels")

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

    padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    H = h + 2 * ph
    W = w + 2 * pw
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    out = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        vs = i * sh
        ve = vs + kh
        for j in range(out_w):
            hs = j * sw
            he = hs + kw
            window = padded[:, vs:ve, hs:he, :]
            for n in range(nc):
                k = kernels[:, :, :, n]
                out[:, i, j, n] = np.sum(window * k, axis=(1, 2, 3))

    return out

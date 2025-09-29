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
        pad_top = ph
        pad_bottom = 0
        pad_left = pw
        pad_right = 0
    elif padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = 0
    elif padding == 'same':
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))
        pad_h_total = max((out_h - 1) * sh + kh - h, 0)
        pad_w_total = max((out_w - 1) * sw + kw - w, 0)
        pad_top = (pad_h_total + 1) // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = (pad_w_total + 1) // 2
        pad_right = pad_w_total - pad_left
    else:
        raise ValueError(
            "padding must be 'same', 'valid', or a (ph, pw) tuple")

    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    H = h + pad_top + pad_bottom
    W = w + pad_left + pad_right
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

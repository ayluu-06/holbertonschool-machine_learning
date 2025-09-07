#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np


def one_hot(labels, classes=None):
    """
    funcion documentada
    """
    if classes is None:
        classes = np.max(labels) + 1
    return np.eye(classes)[labels]

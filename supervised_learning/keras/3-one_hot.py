#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    funcion documentada
    """
    return K.utils.to_categorical(labels, num_classes=classes)

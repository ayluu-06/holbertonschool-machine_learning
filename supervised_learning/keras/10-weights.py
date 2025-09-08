#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    funcion documentada
    """
    fmt = None if save_format == 'keras' else save_format
    network.save_weights(filename, save_format=fmt)
    return None


def load_weights(network, filename):
    """
    funcion documentada
    """
    network.load_weights(filename)
    return None

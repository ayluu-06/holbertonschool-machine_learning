#!/usr/bin/env python3
"""
modulo docuemtando
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    funcion documentada
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    funcion documentada
    """
    return K.models.load_model(filename)

#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    funcion documentada
    """
    config_json = network.to_json()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config_json)
    return None


def load_config(filename):
    """
    funcion documentada
    """
    with open(filename, "r", encoding="utf-8") as f:
        config_json = f.read()
    model = K.models.model_from_json(config_json)
    return model

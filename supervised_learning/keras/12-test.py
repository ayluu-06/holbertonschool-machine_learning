#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    funcion documentada
    """
    return network.evaluate(data, labels, verbose=verbose)

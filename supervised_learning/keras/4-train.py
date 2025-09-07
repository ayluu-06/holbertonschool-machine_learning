#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, verbose=True, shuffle=False):
    """
    funcion documentada
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history

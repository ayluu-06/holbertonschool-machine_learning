#!/usr/bin/env python3
"""
Module documented
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    funcion documentada
    """
    model = K.models.Sequential()
    reg = K.regularizers.l2(lambtha)

    for i, (units, act) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(K.layers.Dense(
                units,
                activation=act,
                kernel_regularizer=reg,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units,
                activation=act,
                kernel_regularizer=reg
            ))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model

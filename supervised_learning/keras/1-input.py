#!/usr/bin/env python3
"""
modulo documentado
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    funcion documemtada
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model

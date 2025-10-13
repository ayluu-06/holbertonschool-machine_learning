#!/usr/bin/env python3
"""
modulo documemtado
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    funcion documentada
    """
    he = K.initializers.he_normal(seed=0)
    concat = X

    for _ in range(layers):
        x = K.layers.BatchNormalization(axis=3)(concat)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                            kernel_initializer=he)(x)

        x = K.layers.BatchNormalization(axis=3)(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                            kernel_initializer=he)(x)

        concat = K.layers.Concatenate(axis=3)([concat, x])
        nb_filters += growth_rate

    return concat, nb_filters

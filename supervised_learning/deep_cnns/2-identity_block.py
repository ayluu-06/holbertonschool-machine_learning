#!/usr/bin/env python3
"""
modulo documentado
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    funcion documentada
    """
    F11, F3, F12 = filters
    he = K.initializers.he_normal(seed=0)

    x = K.layers.Conv2D(
        F11, (1, 1), padding='same', kernel_initializer=he)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F3, (3, 3), padding='same', kernel_initializer=he)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F12, (1, 1), padding='same', kernel_initializer=he)(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x

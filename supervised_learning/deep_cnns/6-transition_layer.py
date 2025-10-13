#!/usr/bin/env python3
"""
moduoo documentado
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    funcion documemtada
    """
    he = K.initializers.he_normal(seed=0)
    comp_filters = int(nb_filters * compression)

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(comp_filters, (1, 1), padding='same',
                        kernel_initializer=he)(x)
    x = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x, comp_filters

#!/usr/bin/env python3

"""
modulo documentado
"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    funcion documentada
    """
    he = K.initializers.he_normal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    x = K.layers.BatchNormalization(axis=3)(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                        kernel_initializer=he)(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x, nb_filters = dense_block(x, 64, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(pool_size=(7, 7))(x)
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=he)(x)

    return K.models.Model(inputs=inputs, outputs=outputs)

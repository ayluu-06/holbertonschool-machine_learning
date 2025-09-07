#!/usr/bin/env python3
"""
Module documented
"""

import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    funcion documentada
    """
    model = keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            ))
        if i != len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))
    return model

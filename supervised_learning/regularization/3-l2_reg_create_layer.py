#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    funcion documentada
    """
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")
    reg = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=kernel_init,
                                  kernel_regularizer=reg)
    return layer(prev)

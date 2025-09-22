#!/usr/bin/env python3
"""
Modulo documentado
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    funcion documentada
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")
    dense = tf.keras.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    x = dense(prev)
    return dropout(x, training=training)
